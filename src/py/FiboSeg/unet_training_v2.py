#!/usr/bin/env python
# coding: utf-8


####
####
"""
V2: Ambient lights to have faster prediction (rotate camera instead of surface)
"""
####
####


import os
import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.models as models
# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)

import vtk
import sys
sys.path.insert(0,'..')
import fly_by_features as fbf
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage import img_as_ubyte

import monai
from monai.data import ITKReader, PILReader
from monai.transforms import (
    ToTensor, LoadImage, Lambda, AddChannel, RepeatChannel, ScaleIntensityRange, RandSpatialCrop,
    Resized, Compose
)
from monai.config import print_config
from monai.metrics import DiceMetric
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Imports for monai model
import logging
import tempfile
from glob import glob

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from monai.data import ArrayDataset, create_test_image_2d, decollate_batch
from monai.inferers import sliding_window_inference
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
from monai.visualize import plot_2d_or_3d_image
print("imports done")


# Set the cuda device 
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


class FlyByDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def set_env_params(self, params):
        self.params = params

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):        
        surf = fbf.ReadSurf(df.iloc[idx]["surf"])
        surf = fbf.GetUnitSurf(surf)
        surf, _a, _v = fbf.RandomRotation(surf)

        surf = fbf.ComputeNormals(surf)

        color_normals = ToTensor(dtype=torch.float32, device=device)(vtk_to_numpy(fbf.GetColorArray(surf, "Normals"))/255.0)
        verts = ToTensor(dtype=torch.float32, device=device)(vtk_to_numpy(surf.GetPoints().GetData()))
        faces = ToTensor(dtype=torch.int64, device=device)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
        region_id = ToTensor(dtype=torch.int64, device=device)(vtk_to_numpy(surf.GetPointData().GetScalars("UniversalID")))
        region_id = torch.clamp(region_id, min=0)
        faces_pid0 = faces[:,0:1]
        region_id_faces = torch.take(region_id, faces_pid0)
        
        return verts, faces, region_id, region_id_faces, faces_pid0, color_normals,df.iloc[idx]["surf"]
        
def pad_verts_faces(batch):
    #names = [n for v,f,rid,ridf,fpid0,cn,n in batch]
    verts = [v for v, f, rid, ridf, fpid0, cn,n in batch]
    faces = [f for v, f, rid, ridf, fpid0, cn,n in batch]
    region_ids = [rid for v, f, rid, ridf, fpid0, cn,n in batch]
    region_ids_faces = [ridf for v, f, rid, ridf, fpid0, cn,n in batch]
    faces_pid0s = [fpid0 for v, f, rid, ridf, fpid0, cn,n in batch]
    color_normals = [cn for v, f, rid, ridf, fpid0, cn,n in batch]
    
    pad_seq_verts = pad_sequence(verts, batch_first=True, padding_value=0.0)
    pad_seq_faces = pad_sequence(faces, batch_first=True, padding_value=-1)
    pad_seq_rid = pad_sequence(region_ids, batch_first=True, padding_value=0)
    pad_seq_faces_pid0s = pad_sequence(faces_pid0s, batch_first=True, padding_value=-1)
    pad_seq_cn = pad_sequence(color_normals, batch_first=True, padding_value=0.)
    l = [f.shape[0] for f in faces]
    
    return pad_seq_verts, pad_seq_faces, pad_seq_rid, torch.cat(region_ids_faces), pad_seq_faces_pid0s, pad_seq_cn, l
        
df = pd.read_csv("/NIRAL/work/leclercq/docs/training_UID.csv")

# Split data between training and validation 
#df_train, df_val = train_test_split(df, test_size=0.1)  
df_train = pd.read_csv("/NIRAL/work/leclercq/source/flybyCNN/fly-by-cnn/src/py/notebooks/train_data.csv")
df_val = pd.read_csv("/NIRAL/work/leclercq/source/flybyCNN/fly-by-cnn/src/py/notebooks/val_data.csv")


df_train.to_csv('train_data.csv',index=False)
df_val.to_csv('val_data.csv',index = False)

# Datasets 
train_data = FlyByDataset(df_train)
val_data = FlyByDataset(df_val)

# Dataloaders
batch_size = 10
num_classes = 34 # background + gum + 32 teeth

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_verts_faces)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=pad_verts_faces)

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = AsDiscrete(argmax=True, to_onehot=True, num_classes=num_classes)
post_label = AsDiscrete(to_onehot=True, num_classes=num_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=True, num_classes=num_classes)


# create UNet, DiceLoss and Adam optimizer
model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
    out_channels=num_classes, 
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(torch.load("best_metric_model_segmentation2d_array_v2.pth"))

loss_function = monai.losses.DiceCELoss(to_onehot_y=True,softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
writer = SummaryWriter()

nb_epoch = 1_000_000
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

for epoch in range (nb_epoch):
    print("-" * 20)
    print(f"epoch {epoch + 1}/{nb_epoch}")
    model.train() # Switch to training mode
    epoch_loss = 0
    step = 0
    for batch, (V, F, Y, YF, F0, CN, FL) in enumerate(train_dataloader):
        step += 1

        array_coord = np.random.normal(0, 1, 3)
        array_coord *= dist_cam/(np.linalg.norm(array_coord))
        camera_position = ToTensor(dtype=torch.float32, device=device)([array_coord.tolist()])
        R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(verts=V, faces=F, textures=textures)
        images = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
        pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone())
        y_p = torch.take(YF, pix_to_face)*(pix_to_face >= 0)
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
        #writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        
    epoch_loss /= step
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
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_segmentation2d_array_v2_5.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )
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