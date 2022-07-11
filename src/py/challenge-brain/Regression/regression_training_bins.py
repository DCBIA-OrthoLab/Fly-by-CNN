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
import random
from datetime import datetime
NOW = datetime.now().strftime("%d_%m_%Hh%M")

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
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils import class_weight

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

# from effnetv2 import effnetv2_s


SEED = 184143
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

print('Imports done')

class BrainDataset(Dataset):
    def __init__(self,np_split,y_class,class_weights,triangles):
        self.np_split = np_split
        self.y_class = y_class
        self.class_weights = class_weights
        self.triangles = triangles  
        self.nb_triangles = len(triangles)

    def __len__(self):
        return(len(self.np_split)) # *2 (Left & Right) *2 (Native & Feature space)

    def __getitem__(self,idx):




        item = self.np_split[idx]

        idx_space = random.randint(0,1)
        idx_feature = random.randint(0,1)
        if idx_space == 0:
            data_dir = '/CMF/data/geometric-deep-learning-benchmarking/Data/Regression/Template_Space'
        else:
            data_dir = '/CMF/data/geometric-deep-learning-benchmarking/Data/Regression/Native_Space'        
        l_space = ['template','native']
        l_hemishpere =['L','R']        
        # path_features = f"{data_dir}/regression_{l_space[idx_space]}_space_features/sub-{item.split('_')[0]}_ses-{item.split('_')[1]}_{l_hemishpere[idx_feature]}.shape.gii"
        path_features = f"{data_dir}/regression_{l_space[idx_space]}_space_features/{item[0]}_{l_hemishpere[idx_feature]}.shape.gii"

        vertex_features = gifti.loadGiftiVertexData(path_features)[1] # vertex features

        age_at_birth = item[2]
        scan_age = item[1]

        faces_pid0 = self.triangles[:,0:1]         
    
        #offset = np.arange(self.nb_triangles*4).reshape((self.nb_triangles,4))
        offset = np.zeros((self.nb_triangles,4), dtype=int) + np.array([0,1,2,3])
        faces_pid0_offset = offset + np.multiply(faces_pid0,4)        
        
        face_features = np.take(vertex_features,faces_pid0_offset)    
        age_bin = self.y_class[idx]
        weight = self.class_weights[idx]
        # ic(age_at_birth)
        # ic(age_bin)
        # ic(weight)
        # print('\n')


        
        return vertex_features,face_features, age_at_birth, age_bin, weight


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
    def __init__(self,dropout_lvl):
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
        self.drop = nn.Dropout(p=dropout_lvl)
        self.TimeDistributed = TimeDistributed(efficient_net)


        #self.WV = nn.Linear(2048, 512)
        self.WV = nn.Linear(1280, 512)

        #self.Attention = SelfAttention(2048, 128)
        self.Attention = SelfAttention(1280, 128)
        self.Prediction = nn.Linear(512, 1)
        self.Classification = nn.Linear(512,5)

        
 
    def forward(self, x):
        
        x = self.drop(x)
        x = self.TimeDistributed(x)
        x_v = self.WV(x)
        x_a, w_a = self.Attention(x, x_v)
        x = self.Prediction(x_a)
        x_c = self.Classification(x_a)

        return x,x_c




def main():
    batch_size = 18
    image_size = 224
    num_epochs = 6_000
    ico_lvl = 1
    noise_lvl = 0.01
    dropout_lvl = 0.1
    model_fn = f"checkpoints/{NOW}_BINS_early_stop_MSE_with_test_split_res224_train_shuffle_icolvl{ico_lvl}_noise{noise_lvl}_dropout{dropout_lvl}_seed{SEED}.pt"
    #model_fn = "checkpoints/trash.pt"
    path_ico = '/NIRAL/work/leclercq/data/geometric-deep-learning-benchmarking/Icospheres/ico-6.surf.gii'
    # train_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Regression/birth_age_confounded/new_train.npy'
    # val_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Regression/birth_age_confounded/new_val.npy'
    train_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Regression/birth_age_confounded/train.npy'
    val_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Regression/birth_age_confounded/validation.npy'
    load_model = False
    model_to_load = '/NIRAL/work/leclercq/source/flybyCNN/fly-by-cnn/src/py/challenge-brain/Regression/checkpoints/best_metric_07_05_MSE-0.18_BINS.pt'

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    
    early_stop = EarlyStopping(patience=500, verbose=True, path=model_fn)

    
    ico_sphere = utils.CreateIcosahedron(2.2, ico_lvl)


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


    train_split = np.load(train_split_path,allow_pickle=True)
    val_split = np.load(val_split_path,allow_pickle=True)


    y_train = train_split[:,2]
    y_train = np.array(list(y_train[:]), dtype=float)

    y_val = val_split[:,2]
    y_val = np.array(list(y_val[:]), dtype=float)


    hist, bin_edges = np.histogram(y_train, bins=5, range=(min(y_train-0.1),max(y_train+0.1)))


    y_train = np.digitize(y_train, bin_edges) - 1
    y_val = np.digitize(y_val, bin_edges) - 1

    unique_classes = np.sort(np.unique(y_train))
    unique_class_weights = np.array(class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)) 

    unique_classes_obj = {}
    unique_classes_obj_str = {}
    for uc, cw in zip(unique_classes, unique_class_weights):
        unique_classes_obj[uc] = cw
        unique_classes_obj_str[str(uc)] = cw

    class_weights_train = []
    for y in y_train:
        class_weights_train.append(unique_classes_obj[y])

    class_weights_val = []
    for y in y_val:
        class_weights_val.append(unique_classes_obj[y])     


    # ic(y_train[0:5])
    # ic(y_val[0:5])
    # ic(class_weights_train[0:5])
    # ic(class_weights_val[0:5])   
    # ic(unique_classes)
    # ic(unique_class_weights)
    # ic(bin_edges)

    train_dataset = BrainDataset(train_split,y_train,class_weights_train,triangles)
    val_dataset = BrainDataset(val_split,y_val,class_weights_val,triangles)
    

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
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

    model = ShapeNet_GraphClass(dropout_lvl)
    if load_model:
        model.load_state_dict(torch.load(model_to_load))
    model.to(device)


    # CE_loss = nn.CrossEntropyLoss(weight = torch.tensor(unique_class_weights).to(device))
    ms_val_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter()
    # list_sphere_points = train_dataset.ico_sphere_verts.tolist()


    ##
    ## STARTING TRAINING
    ##


    ic(model_fn)
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        print("-" * 20)
        print(f'epoch {epoch+1}/{num_epochs}')
        if epoch % 20 == 0:
            print(f'Model name: {model_fn}')
            print(f'seed: {SEED}')
        step = 0
        pbar = tqdm(enumerate(train_dataloader),desc='training:', total=len(train_dataloader))

        for batch, (vertex_features, face_features, age_at_birth, age_bin, weights) in pbar:  # TRAIN LOOP

            vertex_features = vertex_features.to(device)
            vertex_features = vertex_features[:,:,0:3]
            age_bin = age_bin.to(device)
            weights = weights.to(device)
            Y = age_at_birth.to(device)
            #scan_age = torch.unsqueeze(scan_age.double().to(device),1)
            face_features = face_features.to(device)


            ico_sphere, _a, _v = utils.RandomRotation(ico_sphere)
            ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
            list_sphere_points = ico_sphere_verts.tolist()
            
            l_inputs = []
            for coords in list_sphere_points:  # multiple views of the object

                inputs = GetView(vertex_features,face_features,
                                batched_ico_verts,batched_ico_faces,ico_verts,
                                ico_faces,phong_renderer,device,coords)
                inputs = torch.unsqueeze(inputs, 1)
                l_inputs.append(inputs)    


            X = torch.cat(l_inputs,dim=1).to(device)
            X = X.type(torch.float32)
            noise = torch.normal(0.0, noise_lvl,size=X.shape).to(device)*(X!=0) # add noise on sphere (not on background)
            X = (X + noise)
            optimizer.zero_grad()

            x,x_c = model(X) # x=age ; x_c = class 
            x = torch.squeeze(x)
            x_c = x_c.double()            

            mse_loss = weighted_mse_loss(x, Y,weights) 
            # ce_loss = CE_loss(x_c,age_bin)
            #loss = mse_loss + ce_loss
            loss = mse_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            step += 1



        train_loss /= step
        print(f"average epoch loss: {train_loss:>7f}, [{epoch+1:>5d}/{num_epochs:>5d}]")
        writer.add_scalar("training_loss", train_loss, epoch + 1)



        model.eval()  
        with torch.no_grad():  # VALIDATION LOOP
            val_loss = 0.0
            step = 0
            for batch, (vertex_features, face_features, age_at_birth, age_bin, weights) in enumerate(val_dataloader):
                vertex_features = vertex_features.to(device)
                vertex_features = vertex_features[:,:,0:3]
                age_bin = age_bin.to(device)
                weights = weights.to(device)
                Y = age_at_birth.to(device)
                #scan_age = torch.unsqueeze(scan_age.double().to(device),1)
                face_features = face_features.to(device)
                

                ico_sphere, _a, _v = utils.RandomRotation(ico_sphere)
                ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
                list_sphere_points = ico_sphere_verts.tolist()

                l_inputs = []
                for coords in list_sphere_points:  # multiple views of the object

                    val_inputs= GetView(vertex_features,face_features,
                                batched_ico_verts,batched_ico_faces,ico_verts,
                                ico_faces,phong_renderer,device,coords)

                    val_inputs = torch.unsqueeze(val_inputs, 1)
                    l_inputs.append(val_inputs) 

                X = torch.cat(l_inputs,dim=1).to(device)
                X = X.type(torch.float32)              

                x, x_c = model(X)
                x = torch.squeeze(x)
                x_c = x_c.double()
                    
                mse_loss = weighted_mse_loss(x, Y,weights) 
                #ce_loss = CE_loss(x_c,age_bin)
                
                #loss = mse_loss + ce_loss
                loss = mse_loss
                val_loss += loss.item()


                step += 1

                # ic(x)
                # ic(Y)

            val_loss  /= step
            val_MSE /= step
            print(f'val loss (CE + MSE): {val_loss}')
            print(f'val mean squared error: {val_MSE}')
            writer.add_scalar("val loss", val_loss, epoch + 1)
            # EARLY-STOPPING ON MEAN SQUARED ERROR!
            early_stop(val_MSE, model)

            if early_stop.early_stop:
                print("Early stopping")
                break


def weighted_mse_loss(inputs, target, weight):
    return torch.sum(weight * (inputs - target) ** 2)


def GetView(vertex_features,face_features,
            batched_ico_verts,batched_ico_faces,ico_verts,
            ico_faces,phong_renderer,device,coords):

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
    R = look_at_rotation(camera_position, device=device)
    # check if camera coords vector and up vector for R are collinear
    if torch.equal(torch.cross(camera_position,torch.tensor([[0.,1.,0.]]).to(device)),torch.tensor([[0., 0., 0.]]).to(device)): 
        R = look_at_rotation(camera_position, up = torch.tensor([[0.0, 0.0, 1.0]]).to(device),device=device)    
    T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

    batch_views = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
    pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone())

    l_features = []

    for index in range(4):
        l_features.append(torch.take(face_features[:,:,index],pix_to_face)*(pix_to_face >= 0)) # take each feature     
    inputs = torch.cat(l_features,dim=3)
    inputs = inputs.permute(0,3,1,2)
    return inputs


if __name__ == '__main__':
    main()

