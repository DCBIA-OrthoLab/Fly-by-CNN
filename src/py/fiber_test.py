import os

import pandas as pd
import numpy as np

from tqdm import tqdm

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

import pickle

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

        x_a, x_s = self.Attention(x, x_v)

        x = self.Prediction(x_a)
        x_v_p = self.Prediction(x_v)

        return x, x_a, x_s, x_v_p, x_v

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

data_dir = "/work/jprieto/data/fly_by_fibers"

model_fn = "/work/jprieto/data/fly_by_fibers/train/fiber_train/checkpoint.pt"

df_test = pd.read_csv("/work/jprieto/data/fly_by_fibers/tracts_filtered_test.csv")

output_dir = "/work/jprieto/data/fly_by_fibers/fiber_test"

classes = df_test["class"].unique()
classes.sort()
classes_enum = dict(zip(classes, range(len(classes))))
df_test["class_enum"] = df_test["class"].replace(classes_enum)
df_test = df_test.sample(frac=1).reset_index(drop=True)

fbd_test = fbd.FiberDataset(df_test, dataset_dir=data_dir, mean_arr=np.array([0.625, 18.5, 18.625]), scale_factor=0.005949394383637981, load_bundle=True, class_column="class_enum")

def pad_verts_faces_test(batch):
    
    verts = [v for v, f, cn, sc  in batch]
    faces = [f for v, f, cn, sc  in batch]
    color_normals = [cn for v, f, cn, sc, in batch]
    synset_class = [sc for v, f, cn, sc, in batch]

    return verts, faces, color_normals, synset_class

    

batch_size = 1
test_dataloader = DataLoader(fbd_test, 
    batch_size=batch_size, 
    num_workers=2,
    collate_fn=pad_verts_faces_test,
    pin_memory=True,
    persistent_workers=True)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

model = FiberNetGraph(fbd_test.ico_sphere_edges.to(device))
model.load_state_dict(torch.load(model_fn))
model.to(device)
model.eval()

print("Test size:", len(df_test))

loss_fn = nn.CrossEntropyLoss()
test_metric = ConfusionMatrixMetric(include_background=True, metric_name="accuracy", reduction="mean", get_not_nans=False)

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

sphere_points = fbd_test.ico_sphere_verts.to(device)
radius = 1.1
sphere_centers = torch.zeros([batch_size, 3]).type(torch.float32).to(device)


with torch.no_grad():
    for batch, (V_b, F_b, N_b, y_b)  in enumerate(test_dataloader):

        row = df_test.iloc[batch]

        out_bundle_name = os.path.join(output_dir, str(row["id"]) + "_" + str(row["class"]) + "_fiber_test.pickle")

        if not os.path.exists(out_bundle_name):

            print("Start:", batch, row)

            bundle_fibers_class_truth = []
            bundle_fibers_class_pred = []
            bundle_fiber_outputs = []

            for V, F, N, y in tqdm(zip(V_b[0], F_b[0], N_b[0], y_b[0])):

                V = ToTensor(dtype=torch.float32)(V)        
                F = ToTensor(dtype=torch.int32)(F)
                N = ToTensor(dtype=torch.float32)(N)

                V = V.unsqueeze(dim=0).to(device)
                F = F.unsqueeze(dim=0).to(device)
                N = N.unsqueeze(dim=0).to(device)

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
                    
                    pix_to_face, zbuf, bary_coords, dists = renderer.rasterizer(meshes)
                    zbuf = zbuf.permute(0, 3, 1, 2)
                    images = torch.cat([images, zbuf], dim=1)
                    img_seq.append(torch.unsqueeze(images, dim=1))

                img_batch = torch.cat(img_seq, dim=1)
                
                x, x_a, x_s, x_v_p, x_v = model(img_batch)
                x = x.cpu().numpy()
                x_a = x_a.cpu().numpy()
                x_s = x_s.cpu().numpy()
                x_v_p = x_v_p.cpu().numpy()
                x_v = x_v.cpu().numpy()

                fiber_output = (x, x_a, x_s, x_v_p, x_v, y)
                print(np.argmax(x, axis=1), y)

                bundle_fiber_outputs.append(fiber_output)

            with open(out_bundle_name, 'wb') as f:
                pickle.dump(bundle_fiber_outputs, f)