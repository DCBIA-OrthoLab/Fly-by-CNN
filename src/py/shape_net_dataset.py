import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import json
import warnings
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pytorch3d.datasets import shapenet

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from sklearn.utils import class_weight

import utils
from icecream import ic
import random

class ShapeNetDataset(Dataset):
    def __init__(self, data_dir, csv_split, version=1, split="train", concat=False, use_vtk=False):        

        self.shapenet_dir = data_dir
        self.concat = concat

        if version not in [1, 2]:
            raise ValueError("Version number must be either 1 or 2.")
        self.model_dir = "model.obj" if version == 1 else "models/model_normalized.obj"

        if(use_vtk):
            self.model_dir = self.model_dir.replace(".obj", ".vtk")

        # Synset dictionary mapping synset offsets to corresponding labels.
        SYNSET_DICT_DIR = Path(shapenet.__file__).resolve().parent
        dict_file = "shapenet_synset_dict_v%d.json" % version
        with open(os.path.join(SYNSET_DICT_DIR, dict_file), "r") as read_dict:
            self.synset_dict = json.load(read_dict)
        # Inverse dictionary mapping synset labels to corresponding offsets.
        self.synset_inv = {label: offset for offset, label in self.synset_dict.items()}
        
        synset_set = {
            synset
            for synset in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, synset))
            and synset in self.synset_dict
        }

        # Check if there are any categories in the official mapping that are not loaded.
        # Update self.synset_inv so that it only includes the loaded categories.
        synset_not_present = set(self.synset_dict.keys()).difference(synset_set)
        [self.synset_inv.pop(self.synset_dict[synset]) for synset in synset_not_present]

        if len(synset_not_present) > 0:
            msg = (
                "The following categories are included in ShapeNetCore ver.%d's "
                "official mapping but not found in the dataset location %s: %s"
                ""
            ) % (version, data_dir, ", ".join(synset_not_present))
            warnings.warn(msg)
            
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be 'train', 'val', or 'test'")
            
        df_split = pd.read_csv(csv_split, dtype = str)
        df_split = df_split.query("split == @split")
        
        self.synset_ids = []
        self.model_ids = []
        
        for idx, row in df_split.iterrows():
            
            self.synset_ids.append(row["synsetId"])
            self.model_ids.append(row["modelId"])
        
        synset_unique_ids, synset_unique_counts = np.unique(self.synset_ids, return_counts=True)
        self.synset_num_models = dict(zip(list(synset_unique_ids), list(synset_unique_counts)))
        self.synset_class_num = dict(zip(list(synset_unique_ids), range(len(list(synset_unique_ids)))))
        
        self.unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=synset_unique_ids, y=df_split['synsetId']))
        
        self.df = df_split.reset_index(drop=True)
        
        ico_sphere = utils.CreateIcosahedron(3.0, 1)
        
        ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_faces = ico_sphere_faces
        self.ico_sphere_edges = ico_sphere_edges.type(torch.int64)

        renderer = vtk.vtkRenderer()
        
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindow.SetSize(224, 224)
        renderWindow.OffScreenRenderingOn()   
        renderWindow.SetMultiSamples(0)         
        renderWindow.UseOffScreenBuffersOn()

        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInputBufferTypeToRGB()
        windowToImageFilter.SetInput(renderWindow)

        windowToImageFilterZ = vtk.vtkWindowToImageFilter()
        windowToImageFilterZ.SetInputBufferTypeToZBuffer()
        windowToImageFilterZ.SetInput(renderWindow)
        windowToImageFilterZ.SetScale(1)
        
        self.renderer = renderer
        self.renderWindow = renderWindow

        self.windowToImageFilter = windowToImageFilter
        self.windowToImageFilterZ = windowToImageFilterZ        
        
        self.cameras = []

        for idx, sphere_point in enumerate(ico_sphere_verts):
            camera = vtk.vtkCamera()
            camera.SetPosition(sphere_point[0], sphere_point[1], sphere_point[2])
            self.cameras.append(camera)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        model = self.df.iloc[idx]        
        model_path = os.path.join(
            self.shapenet_dir, model["synsetId"], model["modelId"], self.model_dir
        )
        
        surf = utils.ReadSurf(model_path)        
        surf = utils.GetUnitSurf(surf, copy=False)
        surf, _a, _v = utils.RandomRotation(surf)        
        surf_actor = utils.GetNormalsActor(surf)        
        self.renderer.AddActor(surf_actor)

        img_o_views_np = []
        img_z_views_np = []
        for idx, camera in enumerate(self.cameras):
            self.renderer.SetActiveCamera(camera)
            self.renderer.ResetCameraClippingRange()

            self.windowToImageFilter.Modified()
            self.windowToImageFilter.Update()
            
            img_o = self.windowToImageFilter.GetOutput()
            img_o_np = vtk_to_numpy(img_o.GetPointData().GetScalars())            
            
            num_components = img_o.GetNumberOfScalarComponents()
            img_o_np = img_o_np.reshape([d for d in img_o.GetDimensions() if d != 1] + [num_components])
            img_o_views_np.append(img_o_np)

            self.windowToImageFilterZ.Modified()
            self.windowToImageFilterZ.Update()

            img_z = self.windowToImageFilterZ.GetOutput()
            img_z_np = vtk_to_numpy(img_z.GetPointData().GetScalars())
            img_z_np = img_z_np.reshape([d for d in img_z.GetDimensions() if d != 1] + [1])
            
            z_near, z_far = camera.GetClippingRange()

            img_z_np = 2.0*z_far*z_near / (z_far + z_near - (z_far - z_near)*(2.0*img_z_np - 1.0))
            img_z_np[img_z_np > (z_far - 0.01)] = 0

            img_z_views_np.append(img_z_np)
            
        img_o_views_np = np.array(img_o_views_np)
        img_z_views_np = np.array(img_z_views_np)
        
        self.renderer.RemoveActor(surf_actor)

        if self.concat:
            return np.concatenate([img_o_views_np, img_z_views_np], axis=-1), self.synset_class_num[model["synsetId"]]
        else:        
            return img_o_views_np, img_z_views_np, self.synset_class_num[model["synsetId"]]



class ShapeNetDataset_Torch(Dataset):
    def __init__(self, data_dir, csv_split, version=1, split="train", concat=False, use_vtk=False):        

        self.shapenet_dir = data_dir
        self.concat = concat

        if version not in [1, 2]:
            raise ValueError("Version number must be either 1 or 2.")
        self.model_dir = "model.obj" if version == 1 else "models/model_normalized.obj"

        if(use_vtk):
            self.model_dir = self.model_dir.replace(".obj", ".vtk")

        # Synset dictionary mapping synset offsets to corresponding labels.
        SYNSET_DICT_DIR = Path(shapenet.__file__).resolve().parent
        dict_file = "shapenet_synset_dict_v%d.json" % version
        with open(os.path.join(SYNSET_DICT_DIR, dict_file), "r") as read_dict:
            self.synset_dict = json.load(read_dict)
        # Inverse dictionary mapping synset labels to corresponding offsets.
        self.synset_inv = {label: offset for offset, label in self.synset_dict.items()}
        
        synset_set = {
            synset
            for synset in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, synset))
            and synset in self.synset_dict
        }

        # Check if there are any categories in the official mapping that are not loaded.
        # Update self.synset_inv so that it only includes the loaded categories.
        synset_not_present = set(self.synset_dict.keys()).difference(synset_set)
        [self.synset_inv.pop(self.synset_dict[synset]) for synset in synset_not_present]

        if len(synset_not_present) > 0:
            msg = (
                "The following categories are included in ShapeNetCore ver.%d's "
                "official mapping but not found in the dataset location %s: %s"
                ""
            ) % (version, data_dir, ", ".join(synset_not_present))
            warnings.warn(msg)
            
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be 'train', 'val', or 'test'")
            
        df_split = pd.read_csv(csv_split, dtype = str)
        df_split = df_split.query("split == @split")
        
        self.synset_ids = []
        self.model_ids = []
        
        for idx, row in df_split.iterrows():
            
            self.synset_ids.append(row["synsetId"])
            self.model_ids.append(row["modelId"])
        
        synset_unique_ids, synset_unique_counts = np.unique(self.synset_ids, return_counts=True)
        self.synset_num_models = dict(zip(list(synset_unique_ids), list(synset_unique_counts)))
        self.synset_class_num = dict(zip(list(synset_unique_ids), range(len(list(synset_unique_ids)))))

        #ic(synset_unique_ids)

        #ic(df_split['synsetId'])
        
        self.unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=synset_unique_ids, y=df_split['synsetId']))  
        #ic(self.unique_class_weights)      
        self.df = df_split.reset_index(drop=True) 
             
        ico_sphere = utils.CreateIcosahedron(3.0, 1)
        
        ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_faces = ico_sphere_faces
        self.ico_sphere_edges = ico_sphere_edges.type(torch.int64)
        

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        model = self.df.iloc[idx]        
        model_path = os.path.join(
           self.shapenet_dir, model["synsetId"], model["modelId"], self.model_dir
        )
        # if random.random() > 0.5:
        #     model_path = '/work/jprieto/data/ShapeNet/ShapeNetCore.v1/03691459/137c1194efbdcc1bfa0892265dbce8fd/model.vtk' # armoire
        # else:
        #     model_path = '/work/jprieto/data/ShapeNet/ShapeNetCore.v1/03001627/c97f8f3d828173607301b41e73e277b0/model.vtk' # chaise
        #ic(model_path)
        surf = utils.ReadSurf(model_path) 
   
        surf = utils.GetUnitSurf(surf, copy=False)
        surf = utils.ComputeNormals(surf) 
        
        #color_normals = vtk_to_numpy(utils.GetColorArray(surf, "Normals"))/128.0 - 1.0
        color_normals = vtk_to_numpy(surf.GetPointData().GetArray("Normals"))  # [-1;1] 
        #color_normals = vtk_to_numpy(utils.GetColorArray(surf, "Normals"))/255.0
        verts = vtk_to_numpy(surf.GetPoints().GetData())
        faces = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]

        return verts, faces, color_normals, self.synset_class_num[model["synsetId"]]
        
class ShapeNetDatasetNrrd(Dataset):
    def __init__(self, data_dir, csv_split, version=1, split="train", concat=True):

        self.shapenet_dir = data_dir
        self.concat = concat

        if version not in [1, 2]:
            raise ValueError("Version number must be either 1 or 2.")        

        # Synset dictionary mapping synset offsets to corresponding labels.
        SYNSET_DICT_DIR = Path(shapenet.__file__).resolve().parent
        dict_file = "shapenet_synset_dict_v%d.json" % version
        with open(os.path.join(SYNSET_DICT_DIR, dict_file), "r") as read_dict:
            self.synset_dict = json.load(read_dict)
        # Inverse dictionary mapping synset labels to corresponding offsets.
        self.synset_inv = {label: offset for offset, label in self.synset_dict.items()}
        
        synset_set = {
            synset
            for synset in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, synset))
            and synset in self.synset_dict
        }

        # Check if there are any categories in the official mapping that are not loaded.
        # Update self.synset_inv so that it only includes the loaded categories.
        synset_not_present = set(self.synset_dict.keys()).difference(synset_set)
        [self.synset_inv.pop(self.synset_dict[synset]) for synset in synset_not_present]

        if len(synset_not_present) > 0:
            msg = (
                "The following categories are included in ShapeNetCore ver.%d's "
                "official mapping but not found in the dataset location %s: %s"
                ""
            ) % (version, data_dir, ", ".join(synset_not_present))
            warnings.warn(msg)
            
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be 'train', 'val', or 'test'")
            
        df_split = pd.read_csv(csv_split, dtype = str)
        df_split = df_split.query("split == @split")
        
        self.synset_ids = []
        self.model_ids = []
        
        for idx, row in df_split.iterrows():
            
            self.synset_ids.append(row["synsetId"])
            self.model_ids.append(row["modelId"])
        
        synset_unique_ids, synset_unique_counts = np.unique(self.synset_ids, return_counts=True)
        self.synset_num_models = dict(zip(list(synset_unique_ids), list(synset_unique_counts)))
        self.synset_class_num = dict(zip(list(synset_unique_ids), range(len(list(synset_unique_ids)))))
        
        self.unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=synset_unique_ids, y=df_split['synsetId']))
        
        self.df = df_split.reset_index(drop=True)
        
        ico_sphere = utils.CreateIcosahedron(3.0, 1)
        
        ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_faces = ico_sphere_faces
        self.ico_sphere_edges = ico_sphere_edges.type(torch.int64)
        self.epoch = 0
        

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        model = self.df.iloc[idx]        
        model_path = os.path.join(
            self.shapenet_dir, model["synsetId"], model["modelId"], str(self.epoch)
        )

        normals_path = model_path + "_norm.nrrd"
        normals_np, _ = nrrd.read(normals_path)

        z_path = model_path + "_z.nrrd"
        z_np, _ = nrrd.read(z_path)        

        img_np = np.concatenate([normals_np, z_np], axis=-1).astype(np.float32)        
        return img_np, self.synset_class_num[model["synsetId"]]

class ModelNet40(Dataset):  # from https://github.com/meder411/PointNet-PyTorch/blob/master/dataloader.py

    def __init__(self, csv_split,  split="train", concat=False):

        """
        self.test = test
        
        # Build path list
        self.input_pairs, self.gt_key = self.create_input_list(
            dataset_root_path, test)


        ico_sphere = utils.CreateIcosahedron(3.0, 1)        
        ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_faces = ico_sphere_faces
        self.ico_sphere_edges = ico_sphere_edges.type(torch.int64)


        self.unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=self.gt_key, y=df_split['synsetId'])) 
        """

        self.concat = concat
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be 'train', 'val', or 'test'")
            
        df_split = pd.read_csv(csv_split, dtype = str)
        df_split = df_split.query("split == @split")

        self.class_ids = []
        self.model_ids = []

        for idx, row in df_split.iterrows():            
            self.class_ids.append(row["class"])
            self.model_ids.append(row["surf"])

        class_unique_ids, class_unique_counts = np.unique(self.class_ids, return_counts=True)
        self.class_num_models = dict(zip(list(class_unique_ids), list(class_unique_counts)))
        self.class_num = dict(zip(list(class_unique_ids), range(len(list(class_unique_ids)))))



        self.unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=class_unique_ids, y=df_split['class']))  
      
        self.df = df_split.reset_index(drop=True)    


        # Create ico-sphere
        ico_sphere = utils.CreateIcosahedron(3.0, 1)        
        ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_faces = ico_sphere_faces
        self.ico_sphere_edges = ico_sphere_edges.type(torch.int64)


    def __len__(self):
        return len(self.df.index)

   
    def __getitem__(self, idx):
        # Select the path
        model = self.df.iloc[idx] 
        #model_path = model['surf']
        #model_path = '/NIRAL/work/leclercq/data/ModelNet40/chair/train/chair_0276.off'
        label = self.class_num[model['class']]
        ic(model_path)
        ic(model['class'])
        ic(label)
        """
        # Parse the vertices from the file
        vertices = self.off_vertex_parser(path)
        
        if not self.test:
            vertices = self.augment_data(vertices)

        # Convert numpy format to torch variable
        return [torch.from_numpy(vertices), label, path]
        """

        surf = utils.ReadSurf(model_path)        
        surf = utils.GetUnitSurf(surf, copy=False)
        surf = utils.ComputeNormals(surf) 
        
        color_normals = vtk_to_numpy(utils.GetColorArray(surf, "Normals"))/128.0 - 1.0

        verts = vtk_to_numpy(surf.GetPoints().GetData())
        faces = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]        

        return verts, faces, color_normals, label



    """    
    def augment_data(self, vertices):
        # Random rotation about the Y-axis
        theta = 2 * np.pi * np.random.rand(1)
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]])
        vertices = np.matmul(Ry, vertices)

        # Add Gaussian noise with standard deviation of 0.2
        vertices += np.random.normal(scale=0.02, size=vertices.shape)

        return vertices
    

    def off_vertex_parser(self, path_to_off_file):
        # Read the OFF file
        with open(path_to_off_file, 'r') as f:
            contents = f.readlines()

        # Find the number of vertices contained
        # (Handle mangled header lines in .off files)
        if contents[0].strip().lower() != 'off':
            num_vertices = int(contents[0].strip()[4:].split(' ')[0])
            start_line = 1
        else:
            num_vertices = int(contents[1].strip().split(' ')[0])
            start_line = 2

        # Convert all the vertex lines to a list of lists
        vertex_list = [map(float, contents[i].strip().split(' ')) 
                    for i in range(start_line, start_line+num_vertices)]
        
        # Return the vertices as a 3 x N numpy array
        return np.array(vertex_list).transpose(1,0)
    """