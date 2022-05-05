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
from sklearn.metrics import classification_report

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

output_dir = "/work/jprieto/data/fly_by_fibers/fiber_test"

df_test = pd.read_csv("/work/jprieto/data/fly_by_fibers/tracts_filtered_test.csv")
classes = df_test["class"].unique()
classes.sort()
classes_enum = dict(zip(classes, range(len(classes))))
df_test["class_enum"] = df_test["class"].replace(classes_enum)
print(classes_enum)

fiber_class_truth = []
fiber_class_pred = []

for idx, row in df_test.iterrows():

    out_bundle_name = os.path.join(output_dir, str(row["id"]) + "_" + str(row["class"]) + "_fiber_test.pickle")

    if os.path.exists(out_bundle_name):
        bundle_fiber_outputs = pickle.load(open(out_bundle_name, 'rb'))

        for idf, (x, x_a, x_s, x_v_p, x_v) in enumerate(bundle_fiber_outputs):
            fiber_class_truth.append(row["class_enum"])
            fiber_class_pred.append(np.argmax(x, axis=1))

print(classification_report(fiber_class_truth, fiber_class_pred))