# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2022-04-20 10:10:19
# @Last Modified by:   Your name
# @Last Modified time: 2022-05-04 10:38:36

from typing import Dict

import SimpleITK

from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

#### Import librairies requiered for your model and predictions
import os
import sys
import pandas as pd
import numpy as np
import nibabel as nib
sys.path.insert(0,'../..')
import utils

import math
from tqdm import tqdm
from icecream import ic
from random import randint

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
import torchvision.models as models
from torch import from_numpy
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence
from torch.utils.data import DataLoader
from fsl.data import gifti
from pytorch3d.ops.graph_conv import GraphConv
# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)

# datastructures
from pytorch3d.structures import Meshes

from pathlib import Path
import json
from glob import glob


execute_in_docker = False

class Slcn_algorithm(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path = Path("/input/images/cortical-surface-mesh/") if execute_in_docker else Path("/NIRAL/work/leclercq/source/SLCN_challenge_Mathieu/test/"),
            output_file= Path("/output/birth-age.json") if execute_in_docker else Path("/NIRAL/work/leclercq/source/SLCN_challenge_Mathieu/output/birth-age.json")
        )
        
        ###                                                                                                     ###
        ###  TODO: adapt the following part for YOUR submission: should create your model and load the weights
        ###                                                                                                     ###

        # use GPU if available otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("===> Using ", self.device)

        #This path should lead to your model weights
        if execute_in_docker:
            self.path_model = "/opt/algorithm/checkpoints/ckpt.pth"
        else:
            self.path_model = "/NIRAL/work/leclercq/source/SLCN_challenge_Mathieu/weights/ckpt.pth"

        #Model hyperparameters
        self.image_size = 224

        self.ico_sphere = utils.CreateIcosahedron(2.2, 1)
        self.ico_sphere_verts, self.ico_sphere_faces, self.ico_sphere_edges = utils.PolyDataToTensors(self.ico_sphere)

        self.ico_sphere_edges = self.ico_sphere_edges.type(torch.int64)

        if execute_in_docker:
            path_ico = '/opt/algorithm/ico-6.surf.gii'
        else:
            path_ico = '/NIRAL/work/leclercq/data/geometric-deep-learning-benchmarking/Icospheres/ico-6.surf.gii'

        # load icosahedron
        self.ico_surf = nib.load(path_ico)

        # extract points and faces
        self.coords = self.ico_surf.agg_data('pointset')
        self.coords = np.multiply(self.coords,0.01)
        self.triangles = self.ico_surf.agg_data('triangle')
        self.nb_triangles = len(self.triangles)

        #You may adapt this to your model/algorithm here.
        self.model = ShapeNet_GraphClass(self.ico_sphere_edges.to(self.device),0.2)
        #loading model weights
        self.model.load_state_dict(torch.load(self.path_model,map_location=self.device),strict=False)
        self.model.to(self.device)
    
    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, _ = self._load_input_image(case=case)
        # Detect and score candidates
        prediction = self.predict(input_image=input_image)
        # Return a float for prediction
        return float(prediction)


    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        test_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Regression/birth_age_confounded/validation.npy'
        data_dir = '/CMF/data/geometric-deep-learning-benchmarking/Data/Regression/Template_Space'
        test_array = np.load(test_split_path,allow_pickle=True)
        ic(test_array)

        l_truth = test_array[:,2]

        # Initialize a perspective camera.
        cameras = FoVPerspectiveCameras(device=self.device)
        
        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0, 
            faces_per_pixel=1, 
        )
        # We can add a point light in front of the object. 

        lights = AmbientLights(device=self.device)
        rasterizer = MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            )
        phong_renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=HardPhongShader(device=self.device, cameras=cameras, lights=lights)
        )
        list_sphere_points = self.ico_sphere_verts.tolist()

        ##
        ## PREDICTION
        ##
        l_outputs = []
        self.model.eval()
        with torch.no_grad():
            for item in test_array:


                path_features = f"{data_dir}/regression_template_space_features/{item[0]}_L.shape.gii"
                vertex_features = gifti.loadGiftiVertexData(path_features)[1] # vertex features

                # Extract a numpy array with image data from the SimpleITK Image
                offset = np.zeros((self.nb_triangles,4), dtype=int) + np.array([0,1,2,3])
                faces_pid0 = self.triangles[:,0:1]
                faces_pid0_offset = offset + np.multiply(faces_pid0,4)
                face_features = np.take(vertex_features,faces_pid0_offset) 

                # convert ico verts / faces to tensor
                ico_verts = torch.from_numpy(self.coords).unsqueeze(0).to(self.device)
                ico_faces = torch.from_numpy(self.triangles).unsqueeze(0).to(self.device)

                vertex_features = torch.tensor(np.expand_dims(vertex_features,axis=0)).to(self.device) # Simulate batch
                vertex_features = vertex_features[:,:,0:3]
                face_features = torch.tensor(np.expand_dims(face_features,axis=0)).to(self.device)

                l_inputs = []
                for cam_coords in list_sphere_points:  # multiple views of the object

                    inputs = self.GetView(vertex_features,face_features,
                                    ico_verts,ico_faces,phong_renderer,cam_coords)
                    inputs = torch.unsqueeze(inputs, 1)
                    l_inputs.append(inputs)    

                X = torch.cat(l_inputs,dim=1).to(self.device)
                X = X.type(torch.float32)                    

                outputs = self.model(X)
                ic(item[2])
                ic(outputs)
                l_outputs.append(outputs.item())

        np_truth_predict = np.zeros((len(l_outputs),2))
        for index, item in enumerate(l_outputs):
            np_truth_predict[index,0] = l_truth[index]
            np_truth_predict[index,1] = l_outputs[index]
        ic(np_truth_predict)
        return outputs


    def GetView(self,vertex_features,face_features,
                ico_verts,ico_faces,phong_renderer,cam_coords):

        textures = TexturesVertex(verts_features=vertex_features)

        meshes = Meshes(
            verts=ico_verts,   
            faces=ico_faces, 
            textures=textures
        )

        camera_position = torch.FloatTensor([cam_coords]).to(self.device)
        R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

        batch_views = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
        pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone())

        l_features = []

        for index in range(4):
            l_features.append(torch.take(face_features[:,:,index],pix_to_face)*(pix_to_face >= 0)) # take each feature     
        inputs = torch.cat(l_features,dim=3)
        inputs = inputs.permute(0,3,1,2)
        return inputs



class ShapeNet_GraphClass(nn.Module):
    def __init__(self, edges,dropout_lvl):
        super(ShapeNet_GraphClass, self).__init__()



        efficient_net = models.efficientnet_b0(pretrained=False)
        efficient_net.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        efficient_net.classifier = Identity()

        self.drop = nn.Dropout(p=dropout_lvl)
        self.TimeDistributed = TimeDistributed(efficient_net)

        self.WV = nn.Linear(1280, 512)
        self.Attention = SelfAttention(1280, 128)
        self.Prediction = nn.Linear(512, 1)

        
 
    def forward(self, x):
        
        x = self.drop(x)
        x = self.TimeDistributed(x)
        x_v = self.WV(x)
        x_a, w_a = self.Attention(x, x_v)
        x = self.Prediction(x_a)

        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
 
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)
 
        output = self.module(reshaped_input)
        
        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        #alloc_timedistrib = torch.cuda.memory_allocated(0)/(10**9)
        #ic(alloc_timedistrib)
        return output


class SelfAttention(nn.Module):
    def __init__(self, in_units, out_units):
        #super(SelfAttention, self).__init__()
        super().__init__()
        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def forward(self, query, values):        

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)

        score = nn.Sigmoid()(self.V(nn.Tanh()(self.W1(query))))
        
        attention_weights = score/torch.sum(score, dim=1,keepdim=True)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score

if __name__ == "__main__":

    Slcn_algorithm().process()