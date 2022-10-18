##
## IMPORTS
##

import numpy   as np
import pandas as pd
import json
from icecream import ic
import sys
import os
import platform
system = platform.system()
if system == 'Windows':
  code_path = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])
else:
  code_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
sys.path.append(code_path)
import utils
import post_process
import vtk
import random
from timeit import default_timer as timer
from datetime import datetime
NOW = datetime.now().strftime("%d_%m_%Hh%M")
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# rendering components
from pytorch3d.renderer import (
        FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
        RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
        SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex, TexturesAtlas
)
# datastructures
from pytorch3d.structures import Meshes

import monai
from monai.transforms import ToTensor
from monai.transforms import (
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
from torch.nn.utils.rnn import pad_sequence

from teeth_dataset import TeethDataset, RandomRemoveTeethTransform, UnitSurfTransform

import argparse

from datetime import datetime
NOW = datetime.now().strftime("%d_%m_%Hh%M")


SEED = 994
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def main(rank, args, world_size): 
        
        batch_size = args.batch_size
        num_classes = 34 #34 # background + gum + 32 teeth
        nb_epoch = args.epochs
        dist_cam = 1.35
        model_name= os.path.join(args.out, f"checkpoints/{NOW}_seed{SEED}weighted_dataset_noclass_weight.pth")

        if rank == 0:
            output_dir = os.path.dirname(model_name)
            if(not os.path.exists(output_dir)):
                os.makedirs(output_dir)

        patience = args.patience
        train_sphere_samples = args.train_sphere_samples
        
        mount_point = args.mount_point      
        
        # class_weights = np.load(os.path.join(mount_point, 'train_weights.npy'))       
        
        df_train = pd.read_csv(os.path.join(mount_point, args.csv_train))
        df_val = pd.read_csv(os.path.join(mount_point, args.csv_valid))

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
        # global device
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


        train_sampler = DistributedSampler(train_data, shuffle=True, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_data, shuffle=False, num_replicas=world_size, rank=rank)

    
        train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=pad_verts_faces, num_workers=args.num_workers, persistent_workers=True, pin_memory=True, sampler=train_sampler)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, collate_fn=pad_verts_faces, num_workers=args.num_workers, persistent_workers=True, pin_memory=True, sampler=val_sampler)

        # create UNet, DiceLoss and Adam optimizer
        model = monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
                out_channels=num_classes, 
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
        )
        
        if args.model is not None:
            model.load_state_dict(torch.load(args.model))
        model.to(device)
        model = DDP(model, device_ids=[device])

        # class_weights = torch.tensor(class_weights).to(torch.float32).to(device)
        # loss_function =  monai.losses.DiceCELoss(to_onehot_y=True,softmax=True, ce_weight=class_weights)
        loss_function =  monai.losses.DiceCELoss(to_onehot_y=True,softmax=True)
        optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = list()
        metric_values = list()
        writer = SummaryWriter()

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
            for batch, (V, F, YF, CN) in enumerate(train_dataloader):
                # start = timer()
                V = V.to(device, non_blocking=True)
                F = F.to(device, non_blocking=True)
                YF = YF.to(device, non_blocking=True)
                CN = CN.to(torch.float32).to(device, non_blocking=True)

                # blockPrint()
                for s in range(train_sphere_samples):                           
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
                
                # end = timer()
                # print(end - start)

            # enablePrint()
            epoch_loss /= (step*world_size)
            dist.all_reduce(epoch_loss)
            if rank == 0:
                print(f"epoch {epoch + 1} average loss: {epoch_loss.item():.4f}")               
                writer.add_scalar("training_loss", epoch_loss.item(), epoch + 1)

            # Validation
            if (epoch) % val_interval == 0: # every two epochs : validation
                    nb_val += 1 
                    model.eval()
                    with torch.no_grad():
                        val_loss = 0
                        step = 0
                        for batch, (V, F, YF, CN) in enumerate(val_dataloader):   

                            V = V.to(device, non_blocking=True)
                            F = F.to(device, non_blocking=True)
                            YF = YF.to(device, non_blocking=True)
                            CN = CN.to(torch.float32).to(device, non_blocking=True)
                            # blockPrint()

                            val_images, val_y_p = GetView(V,F,CN,YF,dist_cam,phong_renderer,device)          
                            val_images, val_labels = val_images.to(device), val_y_p.to(device)

                            val_outputs = model(val_images)
                            loss = loss_function(val_outputs,val_labels)
                            val_loss += loss
                            step += 1

                        val_loss /= (step * world_size)
                        dist.all_reduce(val_loss)
                        # enablePrint()
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

    verts = [v for v, f, vdf, cn in batch]
    faces = [f for v, f, vdf, cn in batch]        
    verts_data_faces = [vdf for v, f, vdf, cn in batch]        
    color_normals = [cn for v, f, vdf, cn in batch]        
    
    verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
    faces = pad_sequence(faces, batch_first=True, padding_value=-1)
    verts_data_faces = torch.cat(verts_data_faces)
    color_normals = pad_sequence(color_normals, batch_first=True, padding_value=0.0)

    return verts, faces, verts_data_faces, color_normals


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

    parser = argparse.ArgumentParser(description='Teeth challenge Training')
    parser.add_argument('--csv_train', help='CSV with column surf', type=str, required=True)    
    parser.add_argument('--csv_valid', help='CSV with column surf', type=str, required=True)    
    parser.add_argument('--csv_test', help='CSV with column surf', type=str, required=True)    
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')    
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=2000)
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=60)    
    parser.add_argument('--train_sphere_samples', help='Number of training sphere samples or views used during training and validation', type=int, default=4)    
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=200)

    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9999'

    mp.spawn(
        main, args=(args, WORLD_SIZE,),
        nprocs=WORLD_SIZE, join=True
    )
