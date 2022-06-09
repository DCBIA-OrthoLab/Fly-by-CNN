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

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
import torchvision.models as models
from torch import from_numpy
#from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence

import matplotlib.pyplot as plt
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

from effnetv2 import effnetv2_s

print('Imports done')

class BrainDataset(Dataset):
    def __init__(self,np_split,triangles):
        self.np_split = np_split
        self.triangles = triangles  
        self.nb_triangles = len(triangles)


        ico_sphere = utils.CreateIcosahedron(3.0, 1)
        ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_faces = ico_sphere_faces
        self.ico_sphere_edges = ico_sphere_edges.type(torch.int64)

    
    def __len__(self):
        return(len(self.np_split))

    def __getitem__(self,idx):

        data_dir = '/CMF/data/geometric-deep-learning-benchmarking/Data/Regression/Native_Space'
        item = self.np_split[idx][0]
        
        path_features = f"{data_dir}/regression_native_space_features/sub-{item.split('_')[0]}_ses-{item.split('_')[1]}_L.shape.gii"


        vertex_features = gifti.loadGiftiVertexData(path_features)[1] # vertex features
                
        age = self.np_split[idx][1]
        faces_pid0 = self.triangles[:,0:1]         
    
        #offset = np.arange(self.nb_triangles*4).reshape((self.nb_triangles,4))
        offset = np.zeros((self.nb_triangles,4), dtype=int) + np.array([0,1,2,3])
        faces_pid0_offset = offset + np.multiply(faces_pid0,4)        
        
        face_features = np.take(vertex_features,faces_pid0_offset)    

        
        return vertex_features,face_features, age


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

class GraphAttention(nn.Module):
    def __init__(self, in_units, out_units, edges):
        super(GraphAttention, self).__init__()

        self.gconv1 = GraphConv(in_units, out_units)
        self.gconv2 = GraphConv(out_units, 1)
        self.edges = edges

    def forward(self, query, values):        

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)

        score = torch.cat([
            torch.unsqueeze(self.gconv2(nn.Tanh()(self.gconv1(q, self.edges)),self.edges), 0) for q in query], axis=0)
        
        attention_weights = nn.Softmax(dim=1)(score)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score

class SelfAttentionSoftmax(nn.Module):
    def __init__(self, in_units, out_units):
        #super(SelfAttentionSoftmax, self).__init__()
        super().__init__()

        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def forward(self, query, values):        

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)

        score = self.V(nn.Tanh()(self.W1(query)))
        
        attention_weights = nn.Softmax(dim=1)(score)

        ic(score.shape)
        ic(attention_weights.shape)
        ic(score)
        ic(attention_weights)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score


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


class ShapeNet_GraphClass(nn.Module):
    def __init__(self, edges):
        super(ShapeNet_GraphClass, self).__init__()

        # resnet50 = models.resnet50()
        # resnet50.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #resnet50.fc = Identity()

        # efficient_net = effnetv2_s()
        # efficient_net.classifier = Identity()

        efficient_net = models.efficientnet_b0(pretrained=True)
        efficient_net.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        efficient_net.classifier = Identity()


        #self.TimeDistributed = TimeDistributed(resnet50)
        self.TimeDistributed = TimeDistributed(efficient_net)


        #self.WV = nn.Linear(2048, 512)
        self.WV = nn.Linear(1280, 512)

        #self.Attention = SelfAttention(2048, 128)
        self.Attention = SelfAttention(1280, 128)
        self.Prediction = nn.Linear(512, 1)

        
 
    def forward(self, x):
 
        x = self.TimeDistributed(x)
        x_v = self.WV(x)

        x_a, w_a = self.Attention(x, x_v)

        x = self.Prediction(x_a)

        return x





def main():
    batch_size = 6
    image_size = 320
    num_epochs = 600

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    model_fn = "checkpoints/regression_left3_res320"
    early_stop = EarlyStopping(patience=100, verbose=True, path=model_fn)

    path_ico = '/NIRAL/work/leclercq/data/geometric-deep-learning-benchmarking/Icospheres/ico-6.surf.gii'

    # load icosahedron
    ico_surf = nib.load(path_ico)

    # extract points and faces
    coords = ico_surf.agg_data('pointset')
    triangles = ico_surf.agg_data('triangle')
    nb_faces = len(triangles)
    connectivity = triangles.reshape(nb_faces*3,1) # 3 points per triangle
    connectivity = np.int64(connectivity)   
    offsets = [3*i for i in range (nb_faces)]
    offsets.append(nb_faces*3) #  The last value is always the length of the Connectivity array.
    offsets = np.array(offsets)

    # rescale icosphere [0,1]
    coords = np.multiply(coords,0.01)


    # convert ico verts / faces to tensor
    ico_verts = torch.from_numpy(coords).unsqueeze(0).to(device)
    ico_faces = torch.from_numpy(triangles).unsqueeze(0).to(device)

    # Match icosphere vertices and faces tensor with batch size
    l_ico_verts = []
    l_ico_faces = []
    for i in range(batch_size):
        l_ico_verts.append(ico_verts)
        l_ico_faces.append(ico_faces)    
    batched_ico_verts = torch.cat(l_ico_verts,dim=0)
    batched_ico_faces = torch.cat(l_ico_faces,dim=0)

    train_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Regression/scan_age/train.npy'
    val_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Regression/scan_age/train.npy'
    train_split = np.load(train_split_path,allow_pickle=True)
    val_split = np.load(val_split_path,allow_pickle=True)
    train_dataset = BrainDataset(train_split,triangles)
    val_dataset = BrainDataset(val_split,triangles)

    

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True)


    # Initialize a perspective camera.
    cameras = FoVPerspectiveCameras(device=device)
    
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


    model = ShapeNet_GraphClass(train_dataset.ico_sphere_edges.to(device))

    #model.load_state_dict(torch.load(model_fn))
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    list_sphere_points = train_dataset.ico_sphere_verts.tolist()


    #ic(list_sphere_points)

    ##
    ## STARTING TRAINING
    ##



    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        print("-" * 20)
        print(f'epoch {epoch+1}/{num_epochs}')

        for batch, (vertex_features, face_features, age) in tqdm(enumerate(train_dataloader),desc='training:'):  # TRAIN LOOP

            vertex_features = vertex_features.to(device)
            vertex_features = vertex_features[:,:,0:3]
            Y = age.double().to(device)
            face_features = face_features.to(device)
            l_inputs = []

            for coords in list_sphere_points:  # multiple views of the object

                textures = TexturesVertex(verts_features=vertex_features)
                try:
                    meshes = Meshes(
                        verts=batched_ico_verts,   
                        faces=batched_ico_faces, 
                        textures=textures
                    )
                except ValueError:
                    reduced_batch_size = vertex_features.shape[0]
                    l_ico_verts = []
                    l_ico_faces = []
                    for i in range(reduced_batch_size):
                        l_ico_verts.append(ico_verts)
                        l_ico_faces.append(ico_faces)  
                    batched_ico_verts,batched_ico_faces  = torch.cat(l_ico_verts,dim=0), torch.cat(l_ico_faces,dim=0)
                    meshes = Meshes(
                        verts=batched_ico_verts,   
                        faces=batched_ico_faces, 
                        textures=textures
                    )

                camera_position = torch.FloatTensor([coords]).to(device)
                R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
                T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

                batch_views = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
                pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone())

                l_features = []

                for index in range(4):
                    l_features.append(torch.take(face_features[:,:,index],pix_to_face)*(pix_to_face >= 0)) # take each feature     
                inputs = torch.cat(l_features,dim=3)
                inputs = inputs.permute(0,3,1,2)
                inputs = torch.unsqueeze(inputs, 1)
                l_inputs.append(inputs)            

            X = torch.cat(l_inputs,dim=1).to(device)
            X = X.type(torch.float32)
            X = abs(X)            
            optimizer.zero_grad()
            x = model(X) 
            x = x.double() 
            x = torch.squeeze(x)

            loss = loss_fn(x, Y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)
        print(f"average epoch loss: {train_loss:>7f}, [{epoch:>5d}/{num_epochs:>5d}]")


        model.eval()  
        with torch.no_grad():  # VALIDATION LOOP
            running_loss = 0.0
            for batch, (vertex_features, face_features, age) in enumerate(train_dataloader):
                vertex_features = vertex_features.to(device)
                vertex_features = vertex_features[:,:,0:3]
                Y = age.double().to(device)
                face_features = face_features.to(device)
                l_inputs = []

                for coords in list_sphere_points:  # multiple views of the object

                    textures = TexturesVertex(verts_features=vertex_features)
                    try:
                        meshes = Meshes(
                            verts=batched_ico_verts,   
                            faces=batched_ico_faces, 
                            textures=textures
                        )
                    except ValueError:
                        reduced_batch_size = vertex_features.shape[0]
                        l_ico_verts = []
                        l_ico_faces = []
                        for i in range(reduced_batch_size):
                            l_ico_verts.append(ico_verts)
                            l_ico_faces.append(ico_faces)  
                        batched_ico_verts,batched_ico_faces  = torch.cat(l_ico_verts,dim=0), torch.cat(l_ico_faces,dim=0)
                        meshes = Meshes(
                            verts=batched_ico_verts,   
                            faces=batched_ico_faces, 
                            textures=textures
                        )
                    camera_position = torch.FloatTensor([coords]).to(device)
                    R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
                    T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

                    batch_views = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
                    
                    pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone())                    

                    l_features = []
                    for index in range(4):
                        l_features.append(torch.take(face_features[:,:,index],pix_to_face)*(pix_to_face >= 0)) # take each feature     
                    inputs = torch.cat(l_features,dim=3) 

                    inputs = inputs.permute(0,3,1,2)
                    inputs = torch.unsqueeze(inputs, 1)
                    l_inputs.append(inputs)

                X = torch.cat(l_inputs,dim=1).to(device)
                X = X.type(torch.float32)
                X = abs(X)
                x = model(X) 
                x = x.double() 
                x = torch.squeeze(x)
                loss = loss_fn(x, Y)

                running_loss += loss.item()


        val_loss = running_loss / len(val_dataloader)
        print(f'val loss: {val_loss}')
        early_stop(val_loss, model)

        if early_stop.early_stop:
            print("Early stopping")
            break


def GetView():
    

def fibonacci_sphere(samples, dist_cam):

    points = []
    phi = math.pi * (3. -math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y*y)  # radius at y
        theta = phi*i 
        x = math.cos(theta)*radius
        z = math.sin(theta)*radius
        points.append((x*dist_cam, y*dist_cam, z*dist_cam))
    return points


if __name__ == '__main__':
    main()