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

import shape_net_dataset as snd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import utils

from pytorch3d.ops.graph_conv import GraphConv


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

class SelfAttention(nn.Module):
    def __init__(self, in_units, out_units, edges):
        super(SelfAttention, self).__init__()

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


class ShapeNet_GraphClass(nn.Module):
    def __init__(self, edges):
        super(ShapeNet_GraphClass, self).__init__()

        resnet50 = models.resnet50()
        resnet50.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet50.fc = Identity()

        self.TimeDistributed = TimeDistributed(resnet50)
        
        self.WV = nn.Linear(2048, 512)
        self.Attention = SelfAttention(2048, 128, edges)
        self.Prediction = nn.Linear(512, 55)
        
 
    def forward(self, x):
 
        x = self.TimeDistributed(x)
        x_v = self.WV(x)

        x_a, w_a = self.Attention(x, x_v)

        x = self.Prediction(x_a)

        return x



num_epochs = 200

data_dir = "/work/jprieto/data/ShapeNet/ShapeNetCore.v1"
csv_split = "/work/jprieto/data/ShapeNet/ShapeNetCore.v1/all.csv"
model_fn = "/work/jprieto/data/ShapeNet/ShapeNetCore.v1/train/03012021/checkpoint.pt"

early_stop = EarlyStopping(patience=50, verbose=True, path=model_fn)

snd_train = snd.ShapeNetDataset(data_dir, csv_split=csv_split, split="train", concat=True, use_vtk=True)
snd_val = snd.ShapeNetDataset(data_dir, csv_split=csv_split, split="val", concat=True, use_vtk=True)


def pad_verts_faces(batch):
    verts = [v for v, f, cn, sc  in batch]
    faces = [f for v, f, cn, sc  in batch]
    color_normals = [cn for v, f, cn, sc, in batch]
    synset_class = [sc for v, f, cn, sc, in batch]

    return pad_sequence(verts, batch_first=True, padding_value=0.0), pad_sequence(faces, batch_first=True, padding_value=-1), pad_sequence(color_normals, batch_first=True, padding_value=0.), synset_class


train_dataloader = DataLoader(snd_train, batch_size=4, shuffle=True, collate_fn=pad_verts_faces)
val_dataloader = DataLoader(snd_val, batch_size=1)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


model = ShapeNet_GraphClass(snd_train.ico_sphere_edges.to(device))
# model.load_state_dict(torch.load(model_fn))
model.to(device)


loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(snd_train.unique_class_weights, dtype=torch.float32).to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0

    for batch, (X, y) in enumerate(train_dataloader):

        optimizer.zero_grad()        

        X = X.permute(0, 1, 4, 2, 3)
        X = X.type(torch.float32)
        X = X/128.0 - 1.0 

        X = X.to(device)
        y = y.to(device)

        x = model(X)
        
        loss = loss_fn(x, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(snd_train):>5d}]")

    train_loss = running_loss / len(train_dataloader)
    print(f"average epoch loss: {train_loss:>7f}  [{epoch:>5d}/{num_epochs:>5d}]")

    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for batch, (X, y) in enumerate(val_dataloader):
            
            X = X.to(device)
            y = y.to(device)

            x = model(X)
            
            loss = loss_fn(x, y)

            running_loss += loss.item()

    val_loss = running_loss / len(val_dataloader)

    early_stop(val_loss, model)

    if early_stop.early_stop:
        print("Early stopping")
        break