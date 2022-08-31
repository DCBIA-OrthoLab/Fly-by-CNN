import os

import pandas as pd
import numpy as np
import fly_by_features as fbf
import math
from tqdm import tqdm
from icecream import ic


import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
import torchvision.models as models
from torch import from_numpy
#from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence

import shape_net_dataset as snd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import utils

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
        super(SelfAttention, self).__init__()

        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def forward(self, query, values):        

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)

        score = self.V(nn.Tanh()(self.W1(query)))
        
        attention_weights = nn.Softmax(dim=1)(score)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score

class SelfAttention(nn.Module):
    def __init__(self, in_units, out_units):
        super(SelfAttention, self).__init__()

        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def forward(self, query, values):        

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)

        score = nn.Sigmoid()(self.V(nn.Tanh()(self.W1(query))))

        
        attention_weights = score/torch.sum(score, dim=1)


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

        resnet50 = models.resnet50()
        resnet50.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #resnet50.fc = Identity()

        efficient_net = effnetv2_s()
        efficient_net.classifier = Identity()


        #self.TimeDistributed = TimeDistributed(resnet50)
        self.TimeDistributed = TimeDistributed(efficient_net)


        #self.WV = nn.Linear(2048, 512)
        self.WV = nn.Linear(1792, 512)

        #self.Attention = SelfAttention(2048, 128)
        self.Attention = SelfAttention(1792, 128)
        self.Prediction = nn.Linear(512, 55)

        
 
    def forward(self, x):
 
        x = self.TimeDistributed(x)
        x_v = self.WV(x)

        x_a, w_a = self.Attention(x, x_v)

        x = self.Prediction(x_a)

        return x


def main():
    batch_size = 4
    image_size = 256
    num_epochs = 200

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")


    data_dir = "/work/jprieto/data/ShapeNet/ShapeNetCore.v1_vtk"
    #csv_split = '/work/jprieto/data/ShapeNet/ShapeNetCore.v1/all.csv'
    csv_split = "/NIRAL/work/leclercq/data/shapenet_all.csv"
    # model_fn = "/work/jprieto/data/ShapeNet/ShapeNetCore.v1/train/03012021/checkpoint.pt"
    model_fn = "/NIRAL/work/leclercq/data/shapenet_save.pt"

    early_stop = EarlyStopping(patience=50, verbose=True, path=model_fn)

    snd_train = snd.ShapeNetDataset_Torch(data_dir, csv_split=csv_split, split="train", concat=True, use_vtk=True)
    snd_val = snd.ShapeNetDataset_Torch(data_dir, csv_split=csv_split, split="val", concat=True, use_vtk=True)


    
    train_dataloader = DataLoader(snd_train, batch_size=batch_size, shuffle=True, collate_fn=pad_verts_faces,num_workers=4,pin_memory=True)
    val_dataloader = DataLoader(snd_val, batch_size=batch_size,shuffle=True, collate_fn=pad_verts_faces, num_workers=4,pin_memory=True)

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


    model = ShapeNet_GraphClass(snd_train.ico_sphere_edges.to(device))

    #model = ShapeNet_GraphClass(edges.to(device))
    #model.load_state_dict(torch.load(model_fn))
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(snd_train.unique_class_weights, dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    """
    dist_cam = 1.35
    nb_loop = 12
    list_sphere_points = fibonacci_sphere(samples=nb_loop, dist_cam=dist_cam)
    list_sphere_points[0],list_sphere_points[-1] = (0.0001, 1.35, 0.0001),(0.0001, -1.35, 0.0001) # To avoid "invalid rotation matrix" error
    """

    list_sphere_points = snd_train.ico_sphere_verts.tolist()


    #ic(list_sphere_points)

    ##
    ## STARTING TRAINING
    ##

    #torch.autograd.set_detect_anomaly(True)

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        print("-" * 20)
        print(f'epoch {epoch+1}/{num_epochs}')
        sum_correct = 0

        for batch, (V,F,CN,Y) in tqdm(enumerate(train_dataloader),desc='training:'):  # TRAIN LOOP

            #ic(Y)
            V = from_numpy(V).float().to(device)
            F = from_numpy(F).long().to(device)
            Y = torch.LongTensor(Y).to(device)
            CN = from_numpy(CN).float().to(device)

            l_inputs = []

            for coords in list_sphere_points:  # multiple views of the object

                camera_position = torch.FloatTensor([coords]).to(device)
                R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
                T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

                textures = TexturesVertex(verts_features=CN)

                meshes = Meshes(verts=V, faces=F, textures=textures)
                batch_views = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
                #pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone())

                batch_views = batch_views.permute(0,3,1,2)
                batch_views = torch.unsqueeze(batch_views, 1)
                l_inputs.append(batch_views)              

            #ic(batch_views.shape)
            
            X = torch.cat(l_inputs,dim=1).to(device)
            X = X.type(torch.float32)
            #X = X/128.0 - 1.0 
            #ic(X[0,0,0:3,127,127])
            ic(X.shape)
            X = abs(X)            
            optimizer.zero_grad()    
            x = model(X)   
            loss = loss_fn(x, Y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            free_memory = (r-a) / 10**9  # free inside reserved
            #ic(free_memory)
            #ic(loss.item())

            #ic(torch.cuda.memory_summary(device=device, abbreviated=True))

            _, predicted = torch.max(x, 1)
            sum_correct += (predicted == Y).sum()
            acc = sum_correct/(batch_size*batch+1)


            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f" acc: {acc:>7f}, [{current:>5d}/{len(snd_train):>5d}]")

        train_loss = running_loss / len(train_dataloader)
        print(f"average epoch loss: {train_loss:>7f},  acc: {acc:>7f}, [{epoch:>5d}/{num_epochs:>5d}]")


        model.eval()  
        sum_correct = 0
        with torch.no_grad():  # VALIDATION LOOP
            running_loss = 0.0
            for batch, (V,F,CN,Y) in enumerate(val_dataloader):
                #ic(batch)

                V = from_numpy(V).float().to(device)
                F = from_numpy(F).long().to(device)
                Y = torch.LongTensor(Y).to(device)
                CN = from_numpy(CN).float().to(device)


                l_inputs = []

                for coords in list_sphere_points:   # multiple views of the object

                    camera_position = torch.FloatTensor([list(coords)]).to(device)
                    R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
                    T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

                    textures = TexturesVertex(verts_features=CN)
                    meshes = Meshes(verts=V, faces=F, textures=textures)
                    batch_views = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
                    pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone())

                    batch_views = batch_views.permute(0,3,1,2)
                    batch_views = torch.unsqueeze(batch_views, 1)
                    l_inputs.append(batch_views)    

                X = torch.cat(l_inputs,dim=1).to(device)

                X = X.type(torch.float32)
                X = abs(X)

                x = model(X)   
                loss = loss_fn(x, Y)

                running_loss += loss.item()

                _, predicted = torch.max(x, 1)
                sum_correct += (predicted == Y).sum()
                acc = sum_correct/(batch_size*batch+1)

        val_loss = running_loss / len(val_dataloader)
        print(f'acc: {acc:>7f}')

        early_stop(val_loss, model)

        if early_stop.early_stop:
            print("Early stopping")
            break

def pad_verts_faces(batch):
    verts = [v for v, f, cn, sc  in batch]
    faces = [f for v, f, cn, sc  in batch]
    color_normals = [cn for v, f, cn, sc, in batch]
    synset_class = [sc for v, f, cn, sc, in batch]
    

    max_length_verts = max(arr.shape[0] for arr in verts)
    max_length_faces = max(arr.shape[0] for arr in faces)
    max_length_normals = max(arr.shape[0] for arr in color_normals)


    pad_verts = [np.pad(v,[(0,max_length_verts-v.shape[0]),(0,0)],constant_values=0.0) for v in verts]  # pad every array so that they have the same shape
    pad_seq_verts = np.stack(pad_verts)  # stack on a new dimension (batch first)
    pad_faces = [np.pad(f,[(0,max_length_faces-f.shape[0]),(0,0)],constant_values=-1) for f in faces] 
    pad_seq_faces = np.stack(pad_faces)
    pad_cn = [np.pad(cn,[(0,max_length_normals-cn.shape[0]),(0,0)],constant_values=0.) for cn in color_normals]  
    pad_seq_cn = np.stack(pad_cn)

    #return pad_sequence(verts, batch_first=True, padding_value=0.0), pad_sequence(faces, batch_first=True, padding_value=-1), pad_sequence(color_normals, batch_first=True, padding_value=0.), synset_class
    return pad_seq_verts, pad_seq_faces,  pad_seq_cn, synset_class



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