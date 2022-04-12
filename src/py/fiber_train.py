import os

import pandas as pd
import numpy as np
import fly_by_features as fbf

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
import torchvision.models as models
# from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence

import FiberDataset as fbd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import utils

from pytorch3d.ops.graph_conv import GraphConv
from pytorch3d.renderer import look_at_rotation
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PointLights, TexturesVertex
)

import SimpleITK as sitk
from sklearn.utils import class_weight

from monai.transforms import ToTensor
from monai.metrics import ConfusionMatrixMetric

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

class SelfAttention(nn.Module):
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

        return output


class FiberNetGraph(nn.Module):
    def __init__(self, edges):
        super(FiberNetGraph, self).__init__()

        resnet50 = models.resnet50()
        resnet50.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet50.fc = Identity()

        self.TimeDistributed = TimeDistributed(resnet50)
        
        self.WV = nn.Linear(2048, 512)
        self.Attention = GraphAttention(2048, 128, edges)
        self.Prediction = nn.Linear(512, 57)
        
 
    def forward(self, x):
 
        x = self.TimeDistributed(x)
        x_v = self.WV(x)

        x_a, w_a = self.Attention(x, x_v)

        x = self.Prediction(x_a)

        return x

class FiberNet(nn.Module):
    def __init__(self):
        super(FiberNet, self).__init__()

        resnet50 = models.resnet50()
        resnet50.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet50.fc = Identity()

        self.TimeDistributed = TimeDistributed(resnet50)
        
        self.WV = nn.Linear(2048, 512)
        self.Attention = SelfAttention(2048, 128)
        self.Prediction = nn.Linear(512, 57)
        
 
    def forward(self, x):
 
        x = self.TimeDistributed(x)
        x_v = self.WV(x)

        x_a, w_a = self.Attention(x, x_v)

        x = self.Prediction(x_a)

        return x


num_epochs = 200

data_dir = "/SIRIUS_STOR/lumargot/data/tracts"

model_fn = "/SIRIUS_STOR/lumargot/train/fiber_train/checkpoint.pt"

early_stop = EarlyStopping(patience=50, verbose=True, path=model_fn)


df_train = pd.read_csv("/SIRIUS_STOR/lumargot/data/tracts/tracts_filtered_train_train.csv")

classes = df_train["class"].unique()
classes.sort()
classes_enum = dict(zip(classes, range(len(classes))))

df_train["class"] = df_train["class"].replace(classes_enum)

fbd_train = fbd.FiberDataset(df_train, dataset_dir="/SIRIUS_STOR/lumargot/data/tracts", mean_arr=np.array([0.625, 18.5, 18.625]), scale_factor=0.005949394383637981, training=True)

df_valid = pd.read_csv("/SIRIUS_STOR/lumargot/data/tracts/tracts_filtered_train_valid.csv")
df_valid["class"] = df_valid["class"].replace(classes_enum)

fbd_val = fbd.FiberDataset(df_valid, dataset_dir="/SIRIUS_STOR/lumargot/data/tracts", mean_arr=np.array([0.625, 18.5, 18.625]), scale_factor=0.005949394383637981)

def pad_verts_faces(batch):
    verts = [v for v, f, cn, sc  in batch]
    faces = [f for v, f, cn, sc  in batch]
    color_normals = [cn for v, f, cn, sc, in batch]
    synset_class = [sc for v, f, cn, sc, in batch]

    return pad_sequence(verts, batch_first=True, padding_value=0.0), pad_sequence(faces, batch_first=True, padding_value=-1), pad_sequence(color_normals, batch_first=True, padding_value=0.), torch.tensor(synset_class)

batch_size = 8
train_dataloader = DataLoader(fbd_train, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=8,
    collate_fn=pad_verts_faces,
    pin_memory=True,
    persistent_workers=True)

val_dataloader = DataLoader(fbd_val, 
    batch_size=batch_size, 
    num_workers=8,
    collate_fn=pad_verts_faces,
    pin_memory=True,
    persistent_workers=True)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


model = FiberNetGraph(fbd_train.ico_sphere_edges.to(device))
# model = FiberNet()
# model.load_state_dict(torch.load(model_fn))
model.to(device)

unique_classes = np.unique(df_train['class'])
unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train['class']))

print("Unique classes:", unique_classes, unique_class_weights)
print("Train size:", len(df_train), "Valid size:", len(df_valid))


loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(unique_class_weights, dtype=torch.float32).to(device))

val_metric = ConfusionMatrixMetric(include_background=True, metric_name="accuracy", reduction="mean", get_not_nans=False)


optimizer = optim.Adam(model.parameters(), lr=1e-4)


lights = PointLights(device=device) # light in front of the object. 

cameras = FoVPerspectiveCameras(device=device) # Initialize a perspective camera.

raster_settings = RasterizationSettings(        
        image_size=224, 
        blur_radius=0, 
        faces_per_pixel=1, 
    )

rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )

renderer = MeshRenderer(
    rasterizer=rasterizer,
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
)

sphere_points = fbd_train.ico_sphere_verts.to(device)
radius = 1.1
sphere_centers = torch.zeros([batch_size, 3]).type(torch.float32).to(device)

for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0

    for batch, (V, F, N, y) in enumerate(train_dataloader):

        y = y.to(device)
        textures = TexturesVertex(verts_features=N)
        
        meshes =  Meshes(
            verts=V,   
            faces=F, 
            textures=textures
        ).to(device)

        img_seq = []
        cam_coords = []
        for sp in sphere_points:
            sp_i = sp*radius
            # sp = sp.unsqueeze(0).repeat(self.batch_size,1)
            spc = sphere_centers[0:V.shape[0]]
            current_cam_pos = spc + sp_i
            
            R = look_at_rotation(current_cam_pos, at=spc, device=device)  # (1, 3, 3)
            # print( 'R shape :',R.shape)
            # print(R)
            T = -torch.bmm(R.transpose(1, 2), current_cam_pos[:, :, None])[:, :, 0]  # (1, 3)

            images = renderer(meshes_world=meshes, R=R, T=T.to(device))
            images = images.permute(0,3,1,2)
            images = images[:,:-1,:,:]
            
            # print(images.shape)
            pix_to_face, zbuf, bary_coords, dists = renderer.rasterizer(meshes)
            zbuf = zbuf.permute(0, 3, 1, 2)
            # print(dists.shape)
            images = torch.cat([images, zbuf], dim=1)
            img_seq.append(torch.unsqueeze(images, dim=1))
            # print(y)

        img_batch = torch.cat(img_seq, dim=1)

        optimizer.zero_grad()        

        x = model(img_batch)        

        loss = loss_fn(x, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(V)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(fbd_train):>5d}]")

    train_loss = running_loss / len(train_dataloader)
    print(f"average epoch loss: {train_loss:>7f}  [{epoch:>5d}/{num_epochs:>5d}]")

    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for batch, (V, F, N, y) in enumerate(val_dataloader):
            y = y.to(device)
            textures = TexturesVertex(verts_features=N)
            
            meshes =  Meshes(
                verts=V,   
                faces=F, 
                textures=textures
            ).to(device)

            img_seq = []
            cam_coords = []
            for sp in sphere_points:
                sp_i = sp*radius
                # sp = sp.unsqueeze(0).repeat(self.batch_size,1)
                spc = sphere_centers[0:V.shape[0]]
                current_cam_pos = spc + sp_i
                
                R = look_at_rotation(current_cam_pos, at=spc, device=device)  # (1, 3, 3)
                # print( 'R shape :',R.shape)
                # print(R)
                T = -torch.bmm(R.transpose(1, 2), current_cam_pos[:, :, None])[:, :, 0]  # (1, 3)

                images = renderer(meshes_world=meshes, R=R, T=T.to(device))
                images = images.permute(0,3,1,2)
                images = images[:,:-1,:,:]
                
                # print(images.shape)
                pix_to_face, zbuf, bary_coords, dists = renderer.rasterizer(meshes)
                zbuf = zbuf.permute(0, 3, 1, 2)
                # print(dists.shape)
                images = torch.cat([images, zbuf], dim=1)
                img_seq.append(torch.unsqueeze(images, dim=1))
                # print(y)

            img_batch = torch.cat(img_seq, dim=1)

            x = model(img_batch)        

            loss = loss_fn(x, y)

            running_loss += loss.item()

            val_labels = nn.functional.one_hot(y, num_classes=x.shape[-1])
            val_outputs = nn.functional.one_hot(torch.argmax(x, dim=1), num_classes=x.shape[-1])
            val_metric(y_pred=val_outputs, y=val_labels)

    metric = val_metric.aggregate()
    # reset the status for next validation round
    val_metric.reset()

    print("Val confusion matrix", metric)
    val_loss = running_loss / len(val_dataloader)

    early_stop(val_loss, model)

    if early_stop.early_stop:
        print("Early stopping")
        break
