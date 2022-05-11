import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pytorch3d.datasets import shapenet

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from sklearn.utils import class_weight

import utils

from monai.transforms import ToTensor

class FiberDataset(Dataset):
    def __init__(self, df, dataset_dir = "", mean_arr=None, scale_factor=None, training=False, load_bundle=False, class_column="class"):

        
        self.dataset_dir = dataset_dir
        self.df = df
        
        ico_sphere = utils.CreateIcosahedron(1.0, 1)
        
        ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_faces = ico_sphere_faces
        self.ico_sphere_edges = ico_sphere_edges.type(torch.int64)

        self.mean_arr = mean_arr
        self.scale_factor = scale_factor
        self.training = training
        self.load_bundle = load_bundle
        self.class_column = class_column
        

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        fiber_class = row[self.class_column]        

        surf = utils.ReadSurf(os.path.join(self.dataset_dir, row["surf"]))

        if not self.load_bundle:
            if(surf.GetNumberOfCells() == 0):            
                color_normals = np.array([])
                verts = np.array([])
                faces = np.array([])
            else:
        
                i_cell = np.random.randint(0, high=surf.GetNumberOfCells())

                fiber_surf = utils.ExtractFiber(surf, i_cell)

                if self.mean_arr is not None and self.scale_factor is not None:
                    fiber_surf = utils.GetUnitSurf(fiber_surf, mean_arr=self.mean_arr, scale_factor=self.scale_factor, copy=False)
                else:
                    fiber_surf = utils.GetUnitSurf(fiber_surf, copy=False)

                if self.training:
                    fiber_surf, _a, _v = utils.RandomRotation(fiber_surf)

                fiber_surf = utils.ComputeNormals(fiber_surf) 
                
                color_normals = vtk_to_numpy(utils.GetColorArray(fiber_surf, "Normals"))
                verts = vtk_to_numpy(fiber_surf.GetPoints().GetData())
                faces = vtk_to_numpy(fiber_surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]

            color_normals = ToTensor(dtype=torch.float32)(color_normals)
            verts = ToTensor(dtype=torch.float32)(verts)
            faces = ToTensor(dtype=torch.int32)(faces)

            return verts, faces, color_normals, fiber_class
        else:

            verts_array = []
            faces_array = []
            color_normals_array = []
            fiber_class_array = []

            for i_cell in range(surf.GetNumberOfCells()):

                fiber_surf = utils.ExtractFiber(surf, i_cell)

                if self.mean_arr is not None and self.scale_factor is not None:
                    fiber_surf = utils.GetUnitSurf(fiber_surf, mean_arr=self.mean_arr, scale_factor=self.scale_factor, copy=False)
                else:
                    fiber_surf = utils.GetUnitSurf(fiber_surf, copy=False)

                if self.training:
                    fiber_surf, _a, _v = utils.RandomRotation(fiber_surf)

                fiber_surf = utils.ComputeNormals(fiber_surf) 
                
                color_normals = vtk_to_numpy(utils.GetColorArray(fiber_surf, "Normals"))
                verts = vtk_to_numpy(fiber_surf.GetPoints().GetData())
                faces = vtk_to_numpy(fiber_surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]

                verts_array.append(verts)
                faces_array.append(faces)
                color_normals_array.append(color_normals)
                fiber_class_array.append(fiber_class)

                # if i_cell == 10:
                #     break

            return verts_array, faces_array, color_normals_array, fiber_class_array
        