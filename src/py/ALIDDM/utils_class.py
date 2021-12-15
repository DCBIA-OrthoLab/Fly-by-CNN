from torch._C import device
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np 
import torchvision.models as models
from pytorch3d.renderer import look_at_rotation
import torch
from monai.transforms import ToTensor
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import fly_by_features as fbf
from utils import *
import json
from collections import deque
import statistics
import matplotlib.pyplot as plt
import math
import os
from torch.utils.tensorboard import SummaryWriter

class Agent(nn.Module):
    def __init__(self, renderer, features_net, aid, device,run_folder = "",min_radius=0.5,max_radius=2.5,sl=1,lenque = 10):
        super(Agent, self).__init__()
        self.renderer = renderer
        self.device = device
        self.writer = SummaryWriter(os.path.join(run_folder,f"runs_{aid}"))
        self.min_radius=min_radius
        self.max_radius=max_radius
        # self.list_cam_pos = LIST_POINT
        self.max_que = lenque
        self.position_center_memory = deque(maxlen=self.max_que)
        self.best_loss = 9999
        self.best_epoch_loss = 9999
        self.radius = torch.tensor(2.0)
        icosahedron = CreateIcosahedron(1, sl)
        sphere_points = []
        for pid in range(icosahedron.GetNumberOfPoints()):
            spoint = icosahedron.GetPoint(pid)
            sphere_points.append([point for point in spoint])
            # sphere_points.append([(point+0.00001)*0.5 for point in spoint])


        sphere_points = np.array(sphere_points)
        self.sphere_points = torch.tensor(sphere_points).type(torch.float32).to(self.device)

        self.features_net = features_net
        self.attention = TimeAttention(12, 128).to(self.device)
        # self.delta_move = nn.Linear(512, 4).to(self.device)
        self.delta_move = nn.Linear(512, 3).to(self.device)

        self.agent_id = aid
        self.tanh = nn.Tanh() 
        # self.trainable(False)
    
    def reset_sphere_center(self,radius, center_agent, batch_size=1, random=False):
        self.batch_size = batch_size
        if(random):
            self.sphere_centers = (torch.rand(self.batch_size, 3) * radius + center_agent).to(self.device)
        else:
            self.sphere_centers = torch.zeros([self.batch_size, 3]).type(torch.float32).to(self.device)
        

    def get_parameters(self):
        att_param = self.attention.parameters()
        move_param = self.delta_move.parameters()
        return list(att_param) + list(move_param)
    
    def set_radius(self,delta_rad):
        self.radius = self.tanh(delta_rad) * self.max_radius + self.min_radius #[batchsize,1]
        # print(self.radius)
   
    def set_rad(self,radius):
        self.radius = radius
        # print(self.radius)

    def forward(self,x):

        spc = self.sphere_centers
        img_lst = torch.empty((0)).to(self.device)

        for sp in self.sphere_points:
            sp_i = sp*self.radius
            # sp = sp.unsqueeze(0).repeat(self.batch_size,1)
            current_cam_pos = spc + sp_i
            R = look_at_rotation(current_cam_pos, at=spc, device=self.device)  # (1, 3, 3)
            # print( 'R shape :',R.shape)
            # print(R)
            T = -torch.bmm(R.transpose(1, 2), current_cam_pos[:, :, None])[:, :, 0]  # (1, 3)

            images = self.renderer(meshes_world=x.clone(), R=R, T=T.to(self.device))
            images = images.permute(0,3,1,2)
            images = images[:,:-1,:,:]
            
            # print(images.shape)
            pix_to_face, zbuf, bary_coords, dists = self.renderer.rasterizer(x)
            zbuf = zbuf.permute(0, 3, 1, 2)
            # print(dists.shape)
            y = torch.cat([images, zbuf], dim=1)
            # print(y)

            img_lst = torch.cat((img_lst,y.unsqueeze(0)),dim=0)
        img_batch =  img_lst.permute(1,0,2,3,4)

        x = img_batch
        x = self.features_net(x)
        x, s = self.attention(x)
        x = self.delta_move(x)
        x = self.tanh(x)

        return x        


    def trainable(self, train = False):
        for param in self.attention.parameters():
            param.requires_grad = train
            
        for param in self.delta_move.parameters():
            param.requires_grad = train

    
    def found(self,min_variance):
        found = False
        # print(len(self.position_memory))
        if len(self.position_center_memory) == self.max_que:
            # print(self.position_center_memory)
            # print(np.var(np.array(list(self.position_center_memory)),axis=0))
            variance_center_sphere = np.mean(np.var(np.array(list(self.position_center_memory)),axis=0),axis=1) #list variance
            print('variance :', variance_center_sphere)
            if np.max(variance_center_sphere)<min_variance:
                found = True     
        return found   

    def search(self,meshes,min_variance):
        while not self.found(min_variance):
            x = self(meshes)  #[batchsize,time_steps,3,224,224]
            delta_pos =  x[...,0:3]
            delta_pos += self.sphere_centers
            # self.set_radius(x[...,3:4].clone().detach()) 
            new_coord = delta_pos.detach().clone()
            self.position_center_memory.append(new_coord.cpu().numpy())
            self.sphere_centers = new_coord

        
        return self.sphere_centers

    # def affichage(self,list_images):
    #     print('affichage')
    #     new_list = []
    #     for images in list_images:
    #         images = images.permute(0,2,3,1).squeeze(0)
    #         new_list.append(images)
    #         print(images.shape)
    #     # print(list_images)
    #     n_col = 5
    #     n_row = int(math.ceil(len(LIST_POINT)/n_col))
    #     fig,axes = plt.subplots(n_row, n_col)
    #     k = 0
        
    #     for r in range(n_row):
    #         for c in range(n_col):
    #             if k<len(LIST_POINT):
    #                 # print(new_list[k])
    #                 axes[r,c].imshow(new_list[k])
    #             k += 1
    #     plt.show()

class SelfAttention(nn.Module):
    def __init__(self, in_units, out_units):
        super(SelfAttention, self).__init__()
        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def forward(self, x):        

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(nn.Tanh()(self.W1(x)))
        attention_weights = nn.Softmax(dim=1)(score)

        context_vector = attention_weights * x
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score

class TimeAttention(nn.Module):
    def __init__(self, in_units, out_units):
        super(TimeAttention, self).__init__()
        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def forward(self, x):        

        x_t = x.transpose(dim0=1, dim1=2)
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(nn.Tanh()(self.W1(x_t)))
        attention_weights = nn.Softmax(dim=1)(score)

        context_vector = attention_weights.transpose(dim0=1, dim1=2) * x
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        size = x.size()
        batch_size = size[0]
        time_steps = size[1]
        
        size_reshape = [batch_size*time_steps] + list(size[2:])
        x_reshape = x.contiguous().view(size_reshape)  # (batch * timesteps, channels, H, W)
        y = self.module(x_reshape)

        # We have to reshape Y
        output_size = y.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        y = y.contiguous().view(output_size)

        return y # (batch, timesteps, features)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class FeaturesNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet = models.resnet34(pretrained=True)
        resnet.fc = Identity()
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = resnet
        # self.activation = nn.Tanh()
        self.timedist = TimeDistributed(self.resnet)

    def forward(self, x):

        x = self.timedist(x)
        
        return x #Output is [batch, timesteps, 512]

class CameraNet:
    def __init__(self, meshes, renderer, radius=2.0, lenque=5):
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        self.radius = radius
        # print(len(meshes))
        self.set_random_position()
        self.max_que = lenque
        self.position_cam_memory = deque(maxlen=self.max_que)
        self.position_land_memory = deque(maxlen=self.max_que)
        self.set_landmark_position()

    def move(self,x):
        self.camera_position = x[:,:3]
        # print(x)
        # print(self.camera_position)
    
    def move_focal(self,x):
        self.focal_pos = x[:,3:]
        # Render the image using the updated camera position. Based on the new position of the 
        # camera we calculate the rotation and translation matrices
    
    def shot(self):
        print(self.camera_position.shape)
        print(self.focal_pos.shape)
        R = look_at_rotation(self.camera_position, at=self.focal_pos, device=self.device)  # (1, 3, 3)
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
            # print(list_random_pos_cam)

        self.camera_position = torch.from_numpy(np.array(list_random_pos_cam,dtype=np.float32)).to(self.device)
        # print(self.camera_position.size())

    def set_landmark_position(self):
        list_set_land_pos = [] # list of random coord for each meshes 
        for i in range(len(self.meshes)):
            focal_pos= np.array([0,0,0])
            list_set_land_pos.append(focal_pos)
            # print(list_set_land_pos)

        self.focal_pos = torch.from_numpy(np.array(list_set_land_pos,dtype=np.float32)).to(self.device)

    def found(self,min_variance):
        found = False
        # print(len(self.position_memory))
        if len(self.position_cam_memory) == self.max_que:
            variance_cam = np.mean(np.var(np.array(list(self.position_cam_memory)),axis=0),axis=1) #list variance
            variance_land = np.mean(np.var(np.array(list(self.position_land_memory)),axis=0),axis=1) #list variance
            global_variance = (variance_cam+variance_land)/2
            print('variance :', global_variance)
            if np.max(global_variance)<min_variance:
                found = True     
        return found   

    def search(self,move_net,min_variance,writer,device):
        self.set_random_position()
        img_batch = torch.empty((0)).to(device)
        while not self.found(min_variance):
            images = self.shot().to(self.device)  #[batchsize,3,224,224]
            img_batch = torch.cat((img_batch,images),dim=0)
            # print(images.shape)
            x = move_net(images)  # [batchsize,6]  return the deplacment 
            x += torch.cat((self.camera_position,self.focal_pos),dim=1)
            self.move(x.detach().clone())
            self.move_focal(x.detach().clone())
            self.position_cam_memory.append(self.camera_position.cpu().numpy())
            self.position_land_memory.append(self.focal_pos.cpu().numpy())
        
        writer.add_images('image',img_batch)
        
        return self.focal_pos.cpu().numpy()

class FlyByDataset(Dataset):
    def __init__(self, df, device, dataset_dir='', rotate=False):
        self.df = df
        self.device = device
        self.max_landmarks = np.max(self.df["number_of_landmarks"])
        self.dataset_dir = dataset_dir
        self.rotate = rotate
    def set_env_params(self, params):
        self.params = params

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        surf = ReadSurf(os.path.join(self.dataset_dir, self.df.iloc[idx]["surf"])) # list of dico like [{"model":... ,"landmarks":...},...]
        surf, mean_arr, scale_factor= ScaleSurf(surf) # resize my surface to center it to [0,0,0], return the distance between the [0,0,0] and the camera center and the rescale_factor
        if self.rotate:
            surf, angle, vector = RandomRotation(surf)
        else:
            angle = 0 
            vector = np.array([0, 0, 1])
        
        surf = ComputeNormals(surf) 
        landmark_pos = self.get_landmarks_position(idx, mean_arr, scale_factor, self.max_landmarks, angle, vector)
        color_normals = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
        verts = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(surf.GetPoints().GetData()))
        faces = ToTensor(dtype=torch.int32, device=self.device)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
        # region_id = ToTensor(dtype=torch.int64, device=self.device)(vtk_to_numpy(surf.GetPointData().GetScalars("UniversalID")))
        # region_id = torch.clamp(region_id, min=0)
        # faces_pid0 = faces[:,0:1]
        landmark_pos = ToTensor(dtype=torch.float32, device=self.device)(landmark_pos)
        # print(landmark_pos)
        # print('m',mean_arr)
        # print('s',scale_factor)
        # mean_arr = ToTensor( dtype=torch.float64,device=self.device)(mean_arr)
        # scale_factor = ToTensor(dtype=torch.float64, device=self.device)(scale_factor)
        mean_arr = torch.tensor(mean_arr,dtype=torch.float64).to(self.device)
        scale_factor = torch.tensor(scale_factor,dtype=torch.float64).to(self.device)
        # print('m',mean_arr)
        # print('s',scale_factor)

        return verts, faces, color_normals,landmark_pos,mean_arr,scale_factor
   
    def get_landmarks_position(self,idx, mean_arr, scale_factor, number_of_landmarks, angle, vector):
       
        print(self.df.iloc[idx]["landmarks"])
        data = json.load(open(os.path.join(self.dataset_dir,self.df.iloc[idx]["landmarks"])))
        markups = data['markups']
        landmarks_dict = markups[0]['controlPoints']

        landmarks_position = np.zeros([number_of_landmarks, 3])
        # resc_landmarks_position = np.zeros([number_of_landmarks, 3])        
        for idx, landmark in enumerate(landmarks_dict):
            lid = int((landmark["label"]).split("-")[-1]) - 1
            # print('position du landmark avant rescale :',landmark["position"])
            # landmarks_position[lid] = (landmark["position"] - mean_arr) * scale_factor
            # print('m',mean_arr)
            # print('s',scale_factor)
            landmarks_position[lid] = Downscale(landmark["position"],mean_arr,scale_factor)
            # print('position apres dowacaling :', landmarks_position[lid])
            # resc_landmarks_position[lid] = Upscale(landmarks_position[lid],scale_factor,mean_arr)
            # print('position apres upscaling', resc_landmarks_position[lid])

        landmarks_pos = np.array([np.transpose(np.append(pos,1)) for pos in landmarks_position])

        if angle:
            transform = GetTransform(angle, vector)

            transform_matrix = arrayFromVTKMatrix(transform.GetMatrix())
            
            landmarks_pos = np.matmul(landmarks_pos, transform_matrix)

        return landmarks_pos[:, 0:3]



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
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
    def __call__(self, val_loss, agents):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, agents)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, agents)
            self.counter = 0

    def save_checkpoint(self, val_loss, agents):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        for a in agents:
            torch.save(a.state_dict(), os.path.join(self.path, "_aid_" + str(a.agent_id) + ".pt"))
        self.val_loss_min = val_loss


class FlyByDatasetPrediction(Dataset):
    def __init__(self, df, device, dataset_dir=''):
        self.df = df
        self.device = device
        self.dataset_dir = dataset_dir
    
    def set_env_params(self, params):
        self.params = params

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        surf = ReadSurf(os.path.join(self.dataset_dir, self.df.iloc[idx]["surf"])) # list of dico like [{"model":... ,"landmarks":...},...]
        surf, mean_arr, scale_factor= ScaleSurf(surf) # resize my surface to center it to [0,0,0], return the distance between the [0,0,0] and the camera center and the rescale_factor
        surf = ComputeNormals(surf) 

        color_normals = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
        verts = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(surf.GetPoints().GetData()))
        faces = ToTensor(dtype=torch.int32, device=self.device)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
        mean_arr = ToTensor(dtype=torch.float32, device=self.device)(mean_arr)
        scale_factor = ToTensor(dtype=torch.float32, device=self.device)(scale_factor)

        return verts, faces, color_normals,mean_arr,scale_factor,self.df.iloc[idx]["surf"]

def arrayFromVTKMatrix(vmatrix):
  """Return vtkMatrix4x4 or vtkMatrix3x3 elements as numpy array.
  The returned array is just a copy and so any modification in the array will not affect the input matrix.
  To set VTK matrix from a numpy array, use :py:meth:`vtkMatrixFromArray` or
  :py:meth:`updateVTKMatrixFromArray`.
  """
  from vtk import vtkMatrix4x4
  from vtk import vtkMatrix3x3
  import numpy as np
  if isinstance(vmatrix, vtkMatrix4x4):
    matrixSize = 4
  elif isinstance(vmatrix, vtkMatrix3x3):
    matrixSize = 3
  else:
    raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
  narray = np.eye(matrixSize)
  vmatrix.DeepCopy(narray.ravel(), vmatrix)
  return narray

def Upscale(landmark_pos,mean_arr,scale_factor):
    new_pos_center = (landmark_pos/scale_factor) + mean_arr
    return new_pos_center

def Downscale(pos_center,mean_arr,scale_factor):
    landmarks_position = (pos_center - mean_arr) * scale_factor
    return landmarks_position