from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

import os
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence

import pytorch_lightning as pl
from torchvision import transforms

from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization
)

import sys
from icecream import ic

import platform
system = platform.system()
if system == 'Windows':
  code_path = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])
else:
  code_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
sys.path.append(code_path)

import utils
import post_process

from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk


class TeethDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, surf_column="surf", surf_property=None):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.surf_column = surf_column
        self.surf_property = surf_property        

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        

        surf_path = f'{self.mount_point}/{self.df.iloc[idx][self.surf_column]}'
        surf = utils.ReadSurf(surf_path)     


        if self.transform:
            surf = self.transform(surf)

        surf = utils.ComputeNormals(surf)
        color_normals = torch.tensor(vtk_to_numpy(utils.GetColorArray(surf, "Normals"))).to(torch.float32)/255.0
        verts = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
        faces = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)        

        if self.surf_property:            

            faces_pid0 = faces[:,0:1]
            surf_point_data = surf.GetPointData().GetScalars(self.surf_property)
            
            surf_point_data = torch.tensor(vtk_to_numpy(surf_point_data)).to(torch.float32)            
            surf_point_data_faces = torch.take(surf_point_data, faces_pid0)            

            surf_point_data_faces[surf_point_data_faces==-1] = 33            

            return verts, faces, surf_point_data_faces, color_normals

        return verts, faces, color_normals


class TeethDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, surf_column="surf", surf_property=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.surf_column = surf_column
        self.surf_property = surf_property        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = TeethDataset(self.df_train, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, transform=self.train_transform)
        self.val_ds = TeethDataset(self.df_val, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, transform=self.valid_transform)
        self.test_ds = TeethDataset(self.df_test, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, transform=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)

    def pad_verts_faces(self, batch):

        verts = [v for v, f, vdf, cn in batch]
        faces = [f for v, f, vdf, cn in batch]        
        verts_data_faces = [vdf for v, f, vdf, cn in batch]        
        color_normals = [cn for v, f, vdf, cn in batch]        
        
        verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
        faces = pad_sequence(faces, batch_first=True, padding_value=-1)
        verts_data_faces = torch.cat(verts_data_faces)
        color_normals = pad_sequence(color_normals, batch_first=True, padding_value=0.0)

        return verts, faces, verts_data_faces, color_normals


class UnitSurfTransform:

    def __init__(self, random_rotation=False):
        
        self.random_rotation = random_rotation

    def __call__(self, surf):

        surf = utils.GetUnitSurf(surf)
        if self.random_rotation:
            surf, _a, _v = utils.RandomRotation(surf)
        return surf

class RandomRemoveTeethTransform:

    def __init__(self, surf_property = None, random_rotation=False, max_remove=4):

        self.surf_property = surf_property
        self.random_rotation = random_rotation
        self.max_remove = max_remove

    def __call__(self, surf):

        surf = utils.GetUnitSurf(surf)
        if self.random_rotation:
            surf, _a, _v = utils.RandomRotation(surf)

        if self.surf_property:
            surf_point_data = surf.GetPointData().GetScalars(self.surf_property) 
            # ## Remove crown
            unique, counts  = np.unique(surf_point_data, return_counts = True)

            for i in range(self.max_remove):        
                id_to_remove = np.random.choice(unique[:-1])            
                if id_to_remove not in [1,16,17,32] and np.random.rand() > 0.5:
                    surf = post_process.Threshold(surf, self.surf_property ,id_to_remove-0.5,id_to_remove+0.5, invert=True)        
            return surf

