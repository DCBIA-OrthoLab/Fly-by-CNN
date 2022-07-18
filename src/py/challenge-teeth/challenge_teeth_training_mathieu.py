##
## IMPORTS
##

import numpy   as np
import pandas as pd
import json
from icecream import ic
import sys
import os
parent_dir = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
sys.path.append(parent_dir)
import utils
import post_process
import vtk
import random
from datetime import datetime
NOW = datetime.now().strftime("%d_%m_%Hh%M")
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# rendering components
from pytorchtools import EarlyStopping
from pytorch3d.renderer import (
		FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
		RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
		SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex, TexturesAtlas
)
# datastructures
from pytorch3d.structures import Meshes

import monai
from monai.inferers import (sliding_window_inference,SimpleInferer)
from monai.transforms import ToTensor
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import (
		Activations,
		AddChannel,
		AsDiscrete,
		Compose,
		LoadImage,
		RandRotate90,
		RandSpatialCrop,
		ScaleIntensity,
		EnsureType,
)
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from teeth_dataset import TeethDataset, RandomRemoveTeethTransform, UnitSurfTransform

from datetime import datetime
NOW = datetime.now().strftime("%d_%m_%Hh%M")


SEED = 5861
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True


##
## DATASET
##
class MathieuTeethDataset(Dataset):
	def __init__(self,csv_split,split='train'):
		
		if split not in ["train", "val", "test"]:
			raise ValueError("Split must be 'train', 'val', or 'test'")

		df_split = pd.read_csv(csv_split, dtype = str)
		df_split = df_split.query("split == @split")
		self.df = df_split.reset_index(drop=True)

		"""
		self.surf_paths = []
		self.label_paths = []

			for idx, row in df_split.iterrows():            
					self.surf_paths.append(row["surf"])
					self.label_paths.append(row["label"])
		"""

		self.LUT = np.array([33,0,0,0,0,0,0,0,0,0,0,8,7,6,5,4,3,2,1,0,0,9,10,11,12,13,14,15,
													16,0,0,24,23,22,21,20,19,18,17,0,0,25,26,27,28,29,30,31,32])

		

	def __len__(self):
		return len(self.df.index)

	def __getitem__(self,idx):
			model = self.df.iloc[idx] 
			surf_path = model['surf']
			label_path = model['label']

			reader = vtk.vtkOBJReader()
			reader.SetFileName(surf_path)
			reader.Update()
			surf = reader.GetOutput()
			
			#surf = utils.ReadSurf('/NIRAL/work/leclercq/data/lower_rescaled/scan2_rescaled.vtk')
			surf = utils.GetUnitSurf(surf, copy=False)
			surf, _a, _v = utils.RandomRotation(surf)
			surf = utils.ComputeNormals(surf) 
			

			color_normals = vtk_to_numpy(utils.GetColorArray(surf, "Normals"))/128.0 - 1.0

			verts = vtk_to_numpy(surf.GetPoints().GetData())
			faces = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]    

			
			with open(label_path) as f:
				json_data = json.load(f)

			# one label per vertex 
			vertex_labels_FDI = np.array(json_data['labels'])  # FDI World Dental Federation notation

			vertex_labels = self.LUT[vertex_labels_FDI] # UNIVERSAL NUMBERING SYSTEM

			# vertex_labels_vtk = vtk.vtkIntArray()
			# vertex_labels_vtk.SetName("UniversalID")
			# vertex_labels_vtk.SetNumberOfComponents(1)
			# vertex_labels_vtk.SetData(numpy_to_vtk(vertex_labels))

			# surf.GetPointData().SetScalars(vertex_labels_vtk)



			faces_pid0 = faces[:,0:1]
			face_labels = np.take(vertex_labels, faces_pid0) # one label per face


			return verts, faces, face_labels, color_normals




class FlyByDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def set_env_params(self, params):
        self.params = params

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):        
        surf = utils.ReadSurf(self.df.iloc[idx]["surf"])
        surf = utils.GetUnitSurf(surf)
        surf, _a, _v = utils.RandomRotation(surf)
        
        # surf_point_data = surf.GetPointData().GetScalars("UniversalID") 
        surf_point_data = surf.GetPointData().GetScalars("PredictedID") 

        ## Remove crown
        unique, counts  = np.unique(surf_point_data, return_counts = True)

        nb_teeth_to_remove = random.randint(0,4)
        l_no_removal = [1,16,17,32]
        for removal in range(nb_teeth_to_remove):
            id_to_remove = 1
            while id_to_remove in l_no_removal: # don't remove wisdom teeth
                id_to_remove = random.choice(unique[:-1])
            surf = post_process.Threshold(surf, "PredictedID" ,id_to_remove-0.5,id_to_remove+0.5, invert=True)
            l_no_removal.append(id_to_remove)

        surf_point_data = surf.GetPointData().GetScalars("PredictedID") # update data after threshold

        surf = utils.ComputeNormals(surf)
        color_normals = vtk_to_numpy(utils.GetColorArray(surf, "Normals"))/255.0
        verts = vtk_to_numpy(surf.GetPoints().GetData())
        faces = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]
        
        region_id = vtk_to_numpy(surf_point_data)
        region_id = np.clip(region_id,0,None)
        faces_pid0 = faces[:,0:1]
        region_id_faces = np.take(region_id, faces_pid0)


        """
        ic(verts.shape)
        ic(faces.shape)
        ic(region_id.shape)
        ic(region_id_faces.shape)
        ic(faces_pid0.shape)
        ic(color_normals.shape)
        """

        return verts, faces, region_id, region_id_faces, faces_pid0, color_normals



def main(rank, world_size): 

		
		batch_size = 60
		num_classes = 34 #34 # background + gum + 32 teeth
		nb_epoch = 2000
		dist_cam = 1.35
		model_name= f"checkpoints/{NOW}_seed{SEED}weighted_dataset_class_weight.pth"
		# model_name = 'checkpoints/trash.pt'y
		patience = 100	
		nb_loop = 4
		load_model = True
		mount_point = "/work/leclercq/data/challenge_teeth_vtk"
		model_to_load = f"checkpoints/15_07_13h43_seed5861weighted_dataset_noclass_weight.pth"
		
		class_weights = np.load(os.path.join(mount_point, 'train_weights.npy'))
		
		
		df_train = pd.read_csv(os.path.join(mount_point, "mathieu_challenge_teeth_weighted_TRAIN.csv"))
		df_val = pd.read_csv(os.path.join(mount_point, "mathieu_challenge_teeth_weighted_VAL.csv"))

		train_data = TeethDataset(df_train,
								  mount_point=mount_point,
								  transform=RandomRemoveTeethTransform(surf_property="UniversalID", random_rotation=True),
								  surf_column='surf',
								  surf_property="UniversalID")

		val_data = TeethDataset(df_val,
								  mount_point=mount_point,
								  transform=RandomRemoveTeethTransform(surf_property="UniversalID", random_rotation=True),
								  surf_column='surf',
								  surf_property="UniversalID")


		# Set the cuda device 
		global device
		dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)
		print(
		    f"Rank {rank + 1}/{world_size} process initialized.\n"
		)
		device = torch.device(f"cuda:{rank}")
		torch.cuda.set_device(device)

		# Initialize a perspective camera.
		cameras = FoVPerspectiveCameras(device=device)
		image_size = 320
		# We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
		raster_settings = RasterizationSettings(
				image_size=image_size, 
				blur_radius=0, 
				faces_per_pixel=1,
		)
		# We can add a point light in front of the object. 

		lights = AmbientLights(device=device)
		rasterizer = MeshRasterizer(
						cameras=cameras, 
						raster_settings=raster_settings
				)
		phong_renderer = MeshRenderer(
				rasterizer=rasterizer,
				shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
		)


		# Datasets 
		# train_data = TeethDataset(csv_path,split='train')
		# val_data = TeethDataset(csv_path,split='val')



		train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
		val_sampler = DistributedSampler(val_data, shuffle=False, num_replicas=world_size, rank=rank)

	
		train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_verts_faces, num_workers=4,pin_memory=True)
		val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=pad_verts_faces, num_workers=4,pin_memory=True)


		post_trans = AsDiscrete(argmax=True, to_onehot=num_classes, num_classes=num_classes)
		post_label = AsDiscrete(to_onehot=num_classes, num_classes=num_classes)
		post_pred = AsDiscrete(argmax=True, to_onehot=num_classes, num_classes=num_classes)


		# create UNet, DiceLoss and Adam optimizer
		model = monai.networks.nets.UNet(
				spatial_dims=2,
				in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
				out_channels=num_classes, 
				channels=(16, 32, 64, 128, 256),
				strides=(2, 2, 2, 2),
				num_res_units=2,
		)
		
		if load_model:
			model.load_state_dict(torch.load(model_to_load))
		model.to(device)
		model = DDP(model, device_ids=[device])

		class_weights = torch.tensor(class_weights).to(torch.float32).to(device)
		loss_function =  monai.losses.DiceCELoss(to_onehot_y=True,softmax=True, ce_weight=class_weights)
		# loss_function =  monai.losses.DiceCELoss(to_onehot_y=True,softmax=True)
		optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
		best_metric = -1
		best_metric_epoch = -1
		epoch_loss_values = list()
		metric_values = list()
		writer = SummaryWriter()


		camera_position = ToTensor(dtype=torch.float32, device=device)([[0, 0, dist_cam]])
		R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
		T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

		# Start training
		val_interval = 1
		write_image_interval = 1
		writer = SummaryWriter()
		nb_val = 0
		

		# initialize the early_stopping object
		
		
		early_stopping = EarlyStopping(patience=patience, verbose=True,path=model_name)


		for epoch in range (nb_epoch):
			train_sampler.set_epoch(epoch)
			model.train()
			if rank == 0:
				print("-" * 20)
				print(f"epoch {epoch + 1}/{nb_epoch}")
				if epoch % 20 == 0:
					print(f'model name: {model_name}')				
			epoch_loss = 0
			step = 0
			for batch, (V, F, YF, CN, FL) in enumerate(train_dataloader):
				V = ToTensor(dtype=torch.float32, device=device)(V)
				F = ToTensor(dtype=torch.int64, device=device)(F)
				YF = ToTensor(dtype=torch.int64, device=device)(YF)
				CN = ToTensor(dtype=torch.float32, device=device)(CN)

				blockPrint()
				for s in range(nb_loop):							
					batch_views, y_p = GetView(V,F,CN,YF,dist_cam,phong_renderer,device)
					inputs, labels = batch_views.to(device), y_p.to(device)								
					optimizer.zero_grad()
					outputs = model(inputs)
					loss = loss_function(outputs,labels)
					loss.backward()
					optimizer.step()
					epoch_loss += loss
					step += 1
				
				if rank == 0:
					epoch_len = len(train_dataloader)
					print(f"{batch+1}/{epoch_len}, train_loss: {loss.item():.4f}")

			enablePrint()
			epoch_loss /= (step*world_size)
			dist.all_reduce(epoch_loss)
			if rank == 0:
				#print(f'step *nb_loop = {step*nb_loop}')    
				print(f"epoch {epoch + 1} average loss: {epoch_loss.item():.4f}")				
				writer.add_scalar("training_loss", epoch_loss.item(), epoch + 1)

			# Validation
			if (epoch) % val_interval == 0: # every two epochs : validation
					nb_val += 1 
					model.eval()
					with torch.no_grad():
						val_loss = 0
						step = 0
						for batch, (V, F, YF,CN, FL) in enumerate(val_dataloader):   

							V = ToTensor(dtype=torch.float32, device=device)(V)
							F = ToTensor(dtype=torch.int64, device=device)(F)
							YF = ToTensor(dtype=torch.int64, device=device)(YF)
							CN = ToTensor(dtype=torch.float32, device=device)(CN)
							blockPrint()

							val_images, val_y_p = GetView(V,F,CN,YF,dist_cam,phong_renderer,device)          
							val_images, val_labels = val_images.to(device), val_y_p.to(device)

							val_outputs = model(val_images)
							loss = loss_function(val_outputs,val_labels)
							val_loss += loss
							step += 1

						val_loss /= (step * world_size)
						dist.all_reduce(val_loss)
						enablePrint()
						if rank == 0:
							if nb_val % 10 == 0: # save every 4 validations
								auto_save_name = f"checkpoints/autocheckpoints_{NOW}.pth"
								torch.save(model.state_dict(),auto_save_name )
								print(f'auto-saving model: {auto_save_name}')
							print(f"Epoch: {epoch + 1}: val loss: {val_loss.item():.4f}")
							early_stopping(val_loss, model.module)
							if early_stopping.early_stop:
								early_stop_indicator = torch.tensor([1.0]).to(device)
							else:
								early_stop_indicator = torch.tensor([0.0]).cuda()

						else:
							early_stop_indicator = torch.tensor([0.0]).to(device)

						dist.all_reduce(early_stop_indicator)

						if early_stop_indicator.cpu().item() == 1.0:
							print("Early stopping")            
							break

						writer.add_scalar("val_loss", val_loss, epoch + 1)
						imgs_output = torch.argmax(val_outputs, dim=1).detach().cpu()
						imgs_output = imgs_output.unsqueeze(1)  # insert dim of size 1 at pos. 1
						imgs_normals = val_images[:,0:3,:,:]
						val_rgb = torch.cat((255*(1-2*val_labels/33),255*(2*val_labels/33-1),val_labels),dim=1) 
						out_rgb = torch.cat((255*(1-2*imgs_output/33),255*(2*imgs_output/33-1),imgs_output),dim=1) 								
						val_rgb[:,2,...] = 255 - val_rgb[:,1,...] - val_rgb[:,0,...]
						out_rgb[:,2,...] = 255 - out_rgb[:,1,...] - out_rgb[:,0,...]
						norm_rgb = imgs_normals
						val_rgb = val_rgb[0:8,...]
						out_rgb = out_rgb[0:8,...]
						norm_rgb = norm_rgb[0:8,...]

						if nb_val %  write_image_interval == 0:       
								writer.add_images("labels",val_rgb,epoch)
								writer.add_images("output", out_rgb,epoch)
								writer.add_images("normals",norm_rgb,epoch)
						
		if rank == 0:    						
			print(f"Early-Stopping")
			writer.close()



def pad_verts_faces(batch):

		verts = [v for v, f, ridf, cn in batch]
		faces = [f for v, f, ridf,  cn in batch]
		region_ids_faces = [ridf for v, f, ridf,  cn in batch]
		color_normals = [cn for v, f, ridf, cn in batch]

		max_length_verts = max(arr.shape[0] for arr in verts)
		max_length_faces = max(arr.shape[0] for arr in faces)
		max_length_normals = max(arr.shape[0] for arr in color_normals)


		pad_verts = [np.pad(v,[(0,max_length_verts-v.shape[0]),(0,0)],constant_values=0.0) for v in verts]  # pad every array so that they have the same shape
		pad_seq_verts = np.stack(pad_verts)  # stack on a new dimension (batch first)
		pad_faces = [np.pad(f,[(0,max_length_faces-f.shape[0]),(0,0)],constant_values=-1) for f in faces] 
		pad_seq_faces = np.stack(pad_faces)
		pad_cn = [np.pad(cn,[(0,max_length_normals-cn.shape[0]),(0,0)],constant_values=0.) for cn in color_normals]  
		pad_seq_cn = np.stack(pad_cn)

		l = [f.shape[0] for f in faces]
		
		return pad_seq_verts, pad_seq_faces, np.concatenate(region_ids_faces), pad_seq_cn, l


def GetView(V,F,CN,YF,dist_cam,phong_renderer,device):

	array_coord = np.random.normal(0, 1, 3)
	array_coord *= dist_cam/(np.linalg.norm(array_coord))
	camera_position = ToTensor(dtype=torch.float32, device=device)([array_coord.tolist()])
	R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
	T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

	textures = TexturesVertex(verts_features=CN)
	meshes = Meshes(verts=V, faces=F, textures=textures)
	batch_views = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
	pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone())
	depth_map = zbuf
	batch_views = torch.cat([batch_views[:,:,:,0:3], depth_map], dim=-1)
	y_p = torch.take(YF, pix_to_face)*(pix_to_face >= 0) # YF=input, pix_to_face=index. shape of y_p=shape of pix_to_face
	batch_views = batch_views.permute(0,3,1,2)
	y_p = y_p.permute(0,3,1,2)
	return batch_views, y_p




# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



WORLD_SIZE = torch.cuda.device_count()
if __name__ == '__main__':

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9999'

    mp.spawn(
        main, args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE, join=True
    )
