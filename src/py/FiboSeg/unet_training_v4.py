#!/usr/bin/env python
# coding: utf-8


####
####
"""
V4: UNETR  & Early Stopping, multiple loops with camera rotations in training, no torch functions in getitem to use multiprocessing
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
import time
import random


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
from pytorchtools import EarlyStopping
import fly_by_features as fbf
import post_process
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


class FlyByDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def set_env_params(self, params):
        self.params = params

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):        
        surf = fbf.ReadSurf(self.df.iloc[idx]["surf"])
        surf = fbf.GetUnitSurf(surf)
        surf, _a, _v = fbf.RandomRotation(surf)
        
        surf_point_data = surf.GetPointData().GetScalars("UniversalID") 
        ## Remove crown
        unique, counts  = np.unique(surf_point_data, return_counts = True)
        id_to_remove = 1
        while id_to_remove in [1,16,17,32]: # don't remove wisdom teeth
            id_to_remove = random.choice(unique[:-1])
        surf = post_process.Threshold(surf, "UniversalID" ,id_to_remove-0.5,id_to_remove+0.5, invert=True)

        surf_point_data = surf.GetPointData().GetScalars("UniversalID") # update data after threshold

        surf = fbf.ComputeNormals(surf)
        color_normals = vtk_to_numpy(fbf.GetColorArray(surf, "Normals"))/255.0
        verts = vtk_to_numpy(surf.GetPoints().GetData())
        faces = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]
        
        region_id = vtk_to_numpy(surf_point_data)
        region_id = np.clip(region_id,0,None)
        faces_pid0 = faces[:,0:1]
        region_id_faces = np.take(region_id, faces_pid0)
        
        return verts, faces, region_id, region_id_faces, faces_pid0, color_normals
        
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

    
    df_train = pd.read_csv("train_sets_csv/train_data_1.csv")
    df_val = pd.read_csv("train_sets_csv/val_data_1.csv")
    
    """    
    df_train.to_csv('train_sets_csv/train_data_3.csv',index=False)
    df_val.to_csv('train_sets_csv/val_data_3.csv',index = False)
    """

    # Datasets 
    train_data = FlyByDataset(df_train)
    val_data = FlyByDataset(df_val)

    # Dataloaders
    batch_size = 20
    num_classes = 34 # background + gum + 32 teeth

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_verts_faces, num_workers=4,pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=pad_verts_faces, num_workers=4,pin_memory=True)


    """
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_verts_faces)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=pad_verts_faces)
    """

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

    model.load_state_dict(torch.load("early_stopping/checkpoint_1.pt"))

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
    nb_loop = 64

    # initialize the early_stopping object
    model_name= "early_stopping/checkpoint_1156.pt"
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
                    torch.save(model.state_dict(), "best_metric_model_segmentation2d_array_02_07_unet_csv1_restarted_batch20.pth")
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
