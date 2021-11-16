from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np 
import torchvision.models as models
from pytorch3d.renderer import look_at_rotation
import torch
from monai.transforms import ToTensor
from vtk.util.numpy_support import vtk_to_numpy
import fly_by_features as fbf
from utils import *
import json
from collections import deque
import statistics


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class MoveNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet = models.resnet34(pretrained=True)
        resnet.fc = Identity()
        # resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = resnet
        self.MovePrediction = nn.Linear(512, 3)
        # self.activation = nn.Tanh()


    def forward(self, x):
        
        x = self.resnet(x)
        x = self.MovePrediction(x)
        # x = self.activation(x)
        
        return x
    
class CameraNet:
    def __init__(self, meshes, renderer, radius=2.0, lenque=5):
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        self.radius = radius
        # print(len(meshes))
        self.set_random_position()
        self.max_que = lenque
        self.position_memory = deque(maxlen=self.max_que)

    def move(self,x):
        self.camera_position = x

        # Render the image using the updated camera position. Based on the new position of the 
        # camera we calculate the rotation and translation matrices
    
    def shot(self):
        R = look_at_rotation(self.camera_position, device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[:, :, None])[:, :, 0]   # (1, 3)

        images = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        images = images.permute(0,3,1,2)
        images = images[:,:-1,:,:]
        # print(images.size())
        
        return images
    
    def set_random_position(self):
        list_random_pos_cam = [] # list of random coord for each meshes 
        for i in range(len(self.meshes)):
            rand_coord = np.random.rand(3)
            rand_coord = (rand_coord/np.linalg.norm(rand_coord))*self.radius
            list_random_pos_cam.append(rand_coord)

        self.camera_position = torch.from_numpy(np.array(list_random_pos_cam,dtype=np.float32)).to(self.device)
        # print(self.camera_position.size())

    def found(self,min_variance):
        found = False
        # print(len(self.position_memory))
        if len(self.position_memory) == self.max_que:
            variance = np.mean(np.var(np.array(list(self.position_memory)),axis=0),axis=1) #list variance
            print('variance :', variance)
            if np.max(variance)<min_variance:
                found = True     
        return found   

    def search(self,move_net,min_variance):
        self.set_random_position()
        while not self.found(min_variance):
            images = self.shot().to(self.device)  #[batchsize,3,224,224]
            # print(images.shape)
            x = move_net(images)  # [batchsize,3]  return the deplacment 
            x += self.camera_position
            self.move(x.detach().clone())
            self.position_memory.append(self.camera_position.numpy())
        
        

class FlyByDataset(Dataset):
    def __init__(self, df, radius,device):
        self.df = df
        self.radius = radius
        self.device = device
    def set_env_params(self, params):
        self.params = params

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        surf = ReadSurf(self.df[idx]["model"]) # list of dico like [{"model":... ,"landmarks":...},...]
        surf, self.mean_arr, self.scale_factor= ScaleSurf(surf) # resize my surface to center it to [0,0,0], return the distance between the [0,0,0] and the camera center and the rescale_factor
        # surf, _a, _v = fbf.RandomRotation(surf)
        surf = ComputeNormals(surf) 

        landmarks_position = self.get_landmarks_position(idx)
        ideal_position = np.array([landmarks_position[0],landmarks_position[1],2]) # to put ideal position just above the landmark
        # ideal_position = (landmarks_position/np.linalg.norm(landmarks_position))*self.radius
        ideal_position = ToTensor(dtype=torch.float32, device=self.device)(ideal_position)
        color_normals = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
        verts = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(surf.GetPoints().GetData()))
        faces = ToTensor(dtype=torch.int32, device=self.device)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
        region_id = ToTensor(dtype=torch.int64, device=self.device)(vtk_to_numpy(surf.GetPointData().GetScalars("UniversalID")))
        region_id = torch.clamp(region_id, min=0)
        faces_pid0 = faces[:,0:1]
        
  
        return verts, faces, region_id, faces_pid0, color_normals,ideal_position
    
    def get_landmarks_position(self,idx):
       
        print(self.df[idx]["landmarks"])
        data = json.load(open(self.df[idx]["landmarks"]))
        markups = data['markups']
        control_point = markups[0]['controlPoints']
        number_landmarks = len(control_point)
        
        for i in range(number_landmarks):
            print(control_point[i]["label"])
            if control_point[i]["label"] == 'Lower_O-1':
                print(control_point[i]["label"])
                position = control_point[i]["position"]
                landmark_position = (position - self.mean_arr)*self.scale_factor

        return landmark_position

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