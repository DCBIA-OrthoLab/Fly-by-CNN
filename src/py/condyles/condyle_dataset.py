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

class CondyleDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, surf_column="surf", surf_property=None, class_column=None, suffix="", **kwargs):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.surf_column = surf_column
        self.surf_property = surf_property
        self.class_column = class_column
        self.suffix = suffix

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        surf = self.getSurf(idx)

        if self.transform:
            surf = self.transform(surf)

        surf = utils.ComputeNormals(surf)
        color_normals = torch.tensor(vtk_to_numpy(surf.GetPointData().GetArray("Normals"))).to(torch.float32)*0.5 + 0.5
        verts = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
        faces = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int32)

        if self.surf_property:            

            faces_pid0 = faces[:,0:1]
            surf_point_data = surf.GetPointData().GetScalars(self.surf_property)
            
            surf_point_data = torch.tensor(vtk_to_numpy(surf_point_data)).to(torch.float32)            
            surf_point_data_faces = torch.take(surf_point_data, faces_pid0)

            return verts, faces, surf_point_data_faces, color_normals

        if self.class_column:
            cl = torch.tensor(self.df.iloc[idx][self.class_column], dtype=torch.int64)
            return verts, faces, color_normals, cl

        return verts, faces, color_normals

    def getSurf(self, idx):
        surf_path = f'{self.mount_point}/{self.df.iloc[idx][self.surf_column]}' + self.suffix
        return utils.ReadSurf(surf_path)


class CondyleDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, surf_column="surf", class_column="class", surf_property=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val   
        self.df_test = df_test     
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.surf_column = surf_column
        self.class_column = class_column
        self.surf_property = surf_property        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = CondyleDataset(self.df_train, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, class_column=self.class_column, transform=self.train_transform)
        self.val_ds = CondyleDataset(self.df_val, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, class_column=self.class_column, transform=self.valid_transform)
        self.test_ds = CondyleDataset(self.df_test, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, class_column=self.class_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, collate_fn=self.pad_verts_faces)

    def pad_verts_faces(self, batch):

        verts = [v for v, f, cn, l in batch]
        faces = [f for v, f, cn, l in batch]        
        color_normals = [cn for v, f, cn, l in batch]
        labels = [l for v, f, cn, l in batch]
        
        verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
        faces = pad_sequence(faces, batch_first=True, padding_value=-1)        
        color_normals = pad_sequence(color_normals, batch_first=True, padding_value=0.0)
        labels = torch.tensor(labels)

        return verts, faces, color_normals, labels


class UnitSurfTransform:
    def __init__(self, scale_factor=0.13043011372797356):
        self.scale_factor = scale_factor
    def __call__(self, surf):
        return utils.GetUnitSurf(surf, scale_factor=self.scale_factor)

class RandomRotation:
    def __call__(self, surf):
        surf, _a, _v = utils.RandomRotation(surf)
        return surf

class TrainTransform:
    def __init__(self):
        self.train_transform = transforms.Compose(
            [
                UnitSurfTransform(),
                RandomRotation()
            ]
        )

    def __call__(self, surf):
        return self.train_transform(surf)

class EvalTransform:
    def __init__(self):
        self.eval_transform = transforms.Compose(
            [
                UnitSurfTransform()
            ]
        )

    def __call__(self, surf):
        return self.eval_transform(surf)