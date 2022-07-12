##
## IMPORTS
##

import numpy   as np
import pandas as pd
import json

from icecream import ic
import sys
import os
sys.path.insert(0,'..')
import utils
import vtk
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

##
## DATASET
##
class TeethDataset(Dataset):
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
		self.LUT = {'0':33,
								'18':1,
								'17':2,
								'16':3,
								'15':4,
								'14':5,
								'13':6,
								'12':7,
								'11':8,
								'21':9,
								'22':10,
								'23':11,
								'24':12,
								'25':13,
								'26':14,
								'27':15,
								'28':16,
								'38':17,
								'37':18,
								'36':19,
								'35':20,
								'34':21,
								'33':22,
								'32':23,
								'31':24,
								'41':25,
								'42':26,
								'43':27,
								'44':28,
								'45':29,
								'46':30,
								'47':31,
								'48':32,
								}

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

			surf = utils.GetUnitSurf(surf, copy=False)
			surf = utils.ComputeNormals(surf) 

			color_normals = vtk_to_numpy(utils.GetColorArray(surf, "Normals"))/128.0 - 1.0

			verts = vtk_to_numpy(surf.GetPoints().GetData())
			faces = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]    

			with open(label_path) as f:
				json_data = json.load(f)

			# one label per vertex 
			vertex_labels_FDI = np.array(json_data['labels'])  # FDI World Dental Federation notation
			vertex_labels = self.LUT[vertex_labels_FDI] # UNIVERSAL NUMBERING SYSTEM

			faces_pid0 = faces[:,0:1]
			face_labels = np.take(vertex_labels, faces_pid0) # one label per face

			"""
			ic(surf_path)
			unique_labels, counts_labels  = np.unique(face_labels, return_counts = True)
			ic(unique_labels)
			ic(counts_labels)

			ic(verts.shape)
			ic(faces.shape)
			ic(vertex_labels.shape)
			ic(face_labels.shape)
			ic(faces_pid0.shape)
			ic(color_normals.shape)
			"""
			ic(vertex_labels_FDI)
			ic(vertex_labels)
			ic(vertex_labels_FDI.shape)
			ic(vertex_labels.shape)
			return verts, faces, vertex_labels, face_labels, faces_pid0, color_normals





def pad_verts_faces(batch):

		verts = [v for v, f, rid, ridf, fpid0, cn in batch]
		faces = [f for v, f, rid, ridf, fpid0, cn in batch]
		region_ids = [rid for v, f, rid, ridf, fpid0, cn in batch]
		region_ids_faces = [ridf for v, f, rid, ridf, fpid0, cn in batch]
		faces_pid0s = [fpid0 for v, f, rid, ridf, fpid0, cn in batch]
		color_normals = [cn for v, f, rid, ridf, fpid0, cn in batch]

		max_length_verts = max(arr.shape[0] for arr in verts)
		max_length_faces = max(arr.shape[0] for arr in faces)
		max_length_region_ids = max(arr.shape[0] for arr in region_ids)
		max_length_faces_pid0s = max(arr.shape[0] for arr in faces_pid0s)
		max_length_normals = max(arr.shape[0] for arr in color_normals)

		#print(f'max length region_ids : {max_length_region_ids}')
		#print(f'max width region_ids : {max(arr.shape[1] for arr in region_ids)}')
		#print(region_ids[0].shape, region_ids[1].shape)

		pad_verts = [np.pad(v,[(0,max_length_verts-v.shape[0]),(0,0)],constant_values=0.0) for v in verts]  # pad every array so that they have the same shape
		pad_seq_verts = np.stack(pad_verts)  # stack on a new dimension (batch first)
		pad_faces = [np.pad(f,[(0,max_length_faces-f.shape[0]),(0,0)],constant_values=-1) for f in faces] 
		pad_seq_faces = np.stack(pad_faces)
		pad_region_ids = [np.pad(rid,(0,max_length_region_ids-rid.shape[0]),constant_values=0) for rid in region_ids]  
		pad_seq_rid = np.stack(pad_region_ids)
		pad_faces_pid0s = [np.pad(fpid0,[(0,max_length_faces_pid0s-fpid0.shape[0]),(0,0)],constant_values=-1) for fpid0 in faces_pid0s] 
		pad_seq_faces_pid0s = np.stack(pad_faces_pid0s)
		pad_cn = [np.pad(cn,[(0,max_length_normals-cn.shape[0]),(0,0)],constant_values=0.) for cn in color_normals]  
		pad_seq_cn = np.stack(pad_cn)

		l = [f.shape[0] for f in faces]


		
		return pad_seq_verts, pad_seq_faces, pad_seq_rid, np.concatenate(region_ids_faces), pad_seq_faces_pid0s, pad_seq_cn, l

def main(): 
		# Set the cuda device 
		global device
		if torch.cuda.is_available():
				device = torch.device("cuda:0")
				torch.cuda.set_device(device)
		else:
				device = torch.device("cpu")


		# Initialize a perspective camera.
		cameras = FoVPerspectiveCameras(device=device)
		image_size = 320
		# We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
		raster_settings = RasterizationSettings(
				image_size=image_size, 
				blur_radius=0, 
				faces_per_pixel=1, 
				bin_size=0,
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


		#df = pd.read_csv("/NIRAL/work/leclercq/source/flybyCNN/fly-by-cnn/src/py/FiboSeg/train_sets_csv/1.csv")

		# Split data between training and validation 
		#df_train, df_val = train_test_split(df, test_size=0.1)  

		
		csv_path = '/NIRAL/work/leclercq/data/challenge_teeth_all.csv'

		"""    
		df_train.to_csv('train_sets_csv/train_data_3.csv',index=False)
		df_val.to_csv('train_sets_csv/val_data_3.csv',index = False)
		"""

		# Datasets 
		train_data = TeethDataset(csv_path,split='train')
		val_data = TeethDataset(csv_path,split='val')

		# Dataloaders
		batch_size = 30
		num_classes = 50 #34 # background + gum + 32 teeth


		"""
		train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_verts_faces, num_workers=4,pin_memory=True)
		val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=pad_verts_faces, num_workers=4,pin_memory=True)
		"""

		
		train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_verts_faces)
		val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=pad_verts_faces)
		

		dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
		post_trans = AsDiscrete(argmax=True, to_onehot=num_classes, num_classes=num_classes)
		post_label = AsDiscrete(to_onehot=num_classes, num_classes=num_classes)
		post_pred = AsDiscrete(argmax=True, to_onehot=num_classes, num_classes=num_classes)

		"""
		# create UNETR, DiceLoss and Adam optimizer
		model = monai.networks.nets.UNETR(
				spatial_dims=2,
				in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
				img_size=image_size,
				out_channels=num_classes, 
		).to(device)
		"""


		# create UNet, DiceLoss and Adam optimizer
		model = monai.networks.nets.UNet(
				spatial_dims=2,
				in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
				out_channels=num_classes, 
				channels=(16, 32, 64, 128, 256),
				strides=(2, 2, 2, 2),
				num_res_units=2,
		).to(device)

		#model.load_state_dict(torch.load("early_stopping/checkpoint_1.pt"))

		loss_function = monai.losses.DiceCELoss(to_onehot_y=True,softmax=True)
		optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
		best_metric = -1
		best_metric_epoch = -1
		epoch_loss_values = list()
		metric_values = list()
		writer = SummaryWriter()

		nb_epoch = 2000
		dist_cam = 1.35

		camera_position = ToTensor(dtype=torch.float32, device=device)([[0, 0, dist_cam]])
		R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
		T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

		# Start training
		val_interval = 2
		write_image_interval = 1
		best_metric = -1
		best_metric_epoch  = -1
		epoch_loss_values = list()
		metric_values = list()
		writer = SummaryWriter()
		nb_val = 0
		nb_loop = 12

		# initialize the early_stopping object
		model_name= "/NIRAL/work/leclercq/source/flybyCNN/fly-by-cnn/src/py/challenge-teeth/checkpoints/1.pt"
		patience = 500
		early_stopping = EarlyStopping(patience=patience, verbose=True,path=model_name)


		for epoch in range (nb_epoch):
				print("-" * 20)
				print(f"epoch {epoch + 1}/{nb_epoch}")
				model.train() # Switch to training mode
				epoch_loss = 0
				step = 0
				for batch, (V, F, Y, YF, F0, CN, FL) in enumerate(train_dataloader):
						V = ToTensor(dtype=torch.float32, device=device)(V)
						F = ToTensor(dtype=torch.int64, device=device)(F)
						Y = ToTensor(dtype=torch.int64, device=device)(Y)
						YF = ToTensor(dtype=torch.int64, device=device)(YF)
						F0 = ToTensor(dtype=torch.int64, device=device)(F0)
						CN = ToTensor(dtype=torch.float32, device=device)(CN)

						step += 1
						for s in range(nb_loop):
								array_coord = np.random.normal(0, 1, 3)
								array_coord *= dist_cam/(np.linalg.norm(array_coord))
								camera_position = ToTensor(dtype=torch.float32, device=device)([array_coord.tolist()])
								R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
								T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

								textures = TexturesVertex(verts_features=CN)
								meshes = Meshes(verts=V, faces=F, textures=textures)
								images = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
								pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone())
								y_p = torch.take(YF, pix_to_face)*(pix_to_face >= 0) # YF=input, pix_to_face=index. shape of y_p=shape of pix_to_face
								images = images.permute(0,3,1,2)
								y_p = y_p.permute(0,3,1,2)
								inputs, labels = images.to(device), y_p.to(device)
								
								optimizer.zero_grad()
								outputs = model(inputs)
								loss = loss_function(outputs,labels)
								loss.backward()
								optimizer.step()
								epoch_loss += loss.item()
								epoch_len = int(np.ceil(len(train_data) / train_dataloader.batch_size))
						print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
				#print(f'step *nb_loop = {step*nb_loop}')    
				epoch_loss /= (step*nb_loop)
				epoch_loss_values.append(epoch_loss)
				print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
				
				writer.add_scalar("training_loss", epoch_loss, epoch + 1)

				# Validation
				if (epoch) % val_interval == 0: # every two epochs : validation
						nb_val += 1 
						model.eval()
						with torch.no_grad():
								val_images = None
								val_yp = None
								val_outputs = None
								for batch, (V, F, Y, YF, F0, CN, FL) in enumerate(val_dataloader):   

										V = ToTensor(dtype=torch.float32, device=device)(V)
										F = ToTensor(dtype=torch.int64, device=device)(F)
										Y = ToTensor(dtype=torch.int64, device=device)(Y)
										YF = ToTensor(dtype=torch.int64, device=device)(YF)
										F0 = ToTensor(dtype=torch.int64, device=device)(F0)
										CN = ToTensor(dtype=torch.float32, device=device)(CN)

										array_coord = np.random.normal(0, 1, 3)
										array_coord *= dist_cam/(np.linalg.norm(array_coord))
										camera_position = ToTensor(dtype=torch.float32, device=device)([array_coord.tolist()])
										R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
										T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)        

										textures = TexturesVertex(verts_features=CN)
										meshes = Meshes(verts=V, faces=F, textures=textures)
										val_images = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)    
										pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone()) 
										val_y_p = torch.take(YF, pix_to_face)*(pix_to_face >= 0)
										val_images, val_y_p = val_images.permute(0,3,1,2), val_y_p.permute(0,3,1,2)            
										val_images, val_labels = val_images.to(device), val_y_p.to(device)
										
										roi_size = (image_size, image_size)
										sw_batch_size = batch_size
										val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)               


										val_labels_list = decollate_batch(val_labels)                
										val_labels_convert = [
												post_label(val_label_tensor) for val_label_tensor in val_labels_list
										]
										
										val_outputs_list = decollate_batch(val_outputs)
										val_outputs_convert = [
												post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
										]
										
										dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)
										
								# aggregate the final mean dice result
								metric = dice_metric.aggregate().item()
								# reset the status for next validation round
								dice_metric.reset()
								metric_values.append(metric)

								if nb_val % 4 == 0: # save every 4 validations
										torch.save(model.state_dict(), model_name)
										print(f'saving model: {model_name}')

								if metric > best_metric:
										best_metric = metric
										best_metric_epoch = epoch + 1
										torch.save(model.state_dict(),"/NIRAL/work/leclercq/source/flybyCNN/fly-by-cnn/src/py/challenge-teeth/checkpoints/best_1.pt")
										print("saved new best metric model")
										print(model_name)
								print(
										"current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
												epoch + 1, metric, best_metric, best_metric_epoch
										)
								)

								# early_stopping needs the validation loss to check if it has decresed, 
								# and if it has, it will make a checkpoint of the current model
								early_stopping(1-metric, model)

								if early_stopping.early_stop:
										print("Early stopping")
										break 

								writer.add_scalar("validation_mean_dice", metric, epoch + 1)
								imgs_output = torch.argmax(val_outputs, dim=1).detach().cpu()
								imgs_output = imgs_output.unsqueeze(1)  # insert dim of size 1 at pos. 1
								imgs_normals = val_images[:,0:3,:,:]
								val_rgb = torch.cat((255*(1-2*val_labels/33),255*(2*val_labels/33-1),val_labels),dim=1) 
								out_rgb = torch.cat((255*(1-2*imgs_output/33),255*(2*imgs_output/33-1),imgs_output),dim=1) 
								
								val_rgb[:,2,...] = 255 - val_rgb[:,1,...] - val_rgb[:,0,...]
								out_rgb[:,2,...] = 255 - out_rgb[:,1,...] - out_rgb[:,0,...]

								norm_rgb = imgs_normals

								if nb_val %  write_image_interval == 0:       
										writer.add_images("labels",val_rgb,epoch)
										writer.add_images("output", out_rgb,epoch)
										writer.add_images("normals",norm_rgb,epoch)
								
								
		print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
		writer.close()

if __name__ == '__main__':

		main()
