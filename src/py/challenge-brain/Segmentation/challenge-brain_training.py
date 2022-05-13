
import numpy   as np
import nibabel as nib
from fsl.data import gifti
from icecream import ic
import sys
sys.path.insert(0,'..')
import utils
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex, TexturesAtlas
)
# datastructures
from pytorch3d.structures import Meshes

import monai
from monai.inferers import (sliding_window_inference,SimpleInferer)
from monai.transforms import ToTensor


from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    EnsureType,
)
from torch.utils.tensorboard import SummaryWriter


# In[2]:


class BrainDataset(Dataset):
    def __init__(self,np_split,triangles):
        self.np_split = np_split
        self.triangles = triangles  
        self.nb_triangles = len(triangles)

    
    def __len__(self):
        return(len(self.np_split))

    def __getitem__(self,idx):
        data_dir = '/CMF/data/geometric-deep-learning-benchmarking/Data/Segmentation/Native_Space'
        #data_dir = '/CMF/data/geometric-deep-learning-benchmarking/Data/Segmentation/Template_Space'
        item = self.np_split[idx][0]

        # for now just try with Left
        path_features = f'{data_dir}/segmentation_template_space_features/{item}_L.shape.gii'
        path_labels = f'{data_dir}/segmentation_template_space_labels/{item}_L.label.gii'
        
        path_features = f'{data_dir}/segmentation_native_space_features/{item}_L.shape.gii'
        path_labels = f'{data_dir}/segmentation_native_space_labels/{item}_L.label.gii'
        
        vertex_features = gifti.loadGiftiVertexData(path_features)[1] # vertex features
                
        vertex_labels = gifti.loadGiftiVertexData(path_labels)[1] # vertex labels
        faces_pid0 = self.triangles[:,0:1]

        # vertex_features_0 = vertex_features[:,0]
        # vertex_features_1 = vertex_features[:,1]
        # vertex_features_2 = vertex_features[:,2]
        # vertex_features_3 = vertex_features[:,3]  
        
        # face_features_0 = np.take(vertex_features_0,faces_pid0)
        # face_features_1 = np.take(vertex_features_1,faces_pid0)
        # face_features_2 = np.take(vertex_features_2,faces_pid0)
        # face_features_3 = np.take(vertex_features_3,faces_pid0)
        # l_face_features = [face_features_0,face_features_1,face_features_2,face_features_3]
        
        # face_features = np.stack(l_face_features,axis=-1)
        # face_features = np.squeeze(face_features)        
        
        
        face_labels = np.take(vertex_labels,faces_pid0) # face labels (taking first vertex for each face)        
        
        #offset = np.arange(self.nb_triangles*4).reshape((self.nb_triangles,4))
        offset = np.zeros((self.nb_triangles,4), dtype=int) + np.array([0,1,2,3])
        faces_pid0_offset = offset + np.multiply(faces_pid0,4)        
        
        face_features = np.take(vertex_features,faces_pid0_offset)
        
        
        """
        ic(vertex_features.shape)
        ic(face_features.shape)
        ic(vertex_labels.shape)
        ic(face_labels.shape)
        ic(faces_pid0.shape)
        ic(faces_pid0_offset.shape)
        ic(faces_pid0)
        ic(faces_pid0_offset)
        ic(vertex_features)
        ic(face_features)
        ic(face_features_test)
        """
        ic(min(face_labels))
        ic(max(face_labels))
        
        return vertex_features,face_features, face_labels


# In[3]:


# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
# Initialize a perspective camera.
cameras = FoVPerspectiveCameras(device=device)


# We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0, 
    faces_per_pixel=1, 
)
# We can add a point light in front of the object. 

# lights = AmbientLights(device=device)
lights = PointLights(device=device)
rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
phong_renderer = MeshRenderer(
    rasterizer=rasterizer,
    shader=HardPhongShader(device=device, cameras=cameras)
)


# In[4]:


# Setup Neural Network model



num_classes = 40 #37
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = AsDiscrete(argmax=True, to_onehot=num_classes, num_classes=num_classes)
post_label = AsDiscrete(to_onehot=num_classes, num_classes=num_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=num_classes, num_classes=num_classes)

"""
# create UNETR, DiceLoss and Adam optimizer
model = monai.networks.nets.UNETR(
    spatial_dims=2,
    in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
    img_size=image_size,
    out_channels=num_classes, 
).to(device)
"""


# create UNet, DiceLoss and Adam optimizer
model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
    out_channels=num_classes, 
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

# model.load_state_dict(torch.load("early_stopping/checkpoint_1.pt"))

loss_function = monai.losses.DiceCELoss(to_onehot_y=True,softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
writer = SummaryWriter()


# In[5]:


path_ico = '/CMF/data/geometric-deep-learning-benchmarking/Icospheres/ico-6.surf.gii'

#mesh = gifti.loadGiftiMesh(path_ico)

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

# convert to vtk
vtk_coords = vtk.vtkPoints()
vtk_coords.SetData(numpy_to_vtk(coords))
vtk_triangles = vtk.vtkCellArray()
vtk_offsets = numpy_to_vtkIdTypeArray(offsets)
vtk_connectivity = numpy_to_vtkIdTypeArray(connectivity)
vtk_triangles.SetData(vtk_offsets,vtk_connectivity)

"""
# Create icosahedron as a VTK polydata 
ico_polydata = vtk.vtkPolyData() # initialize polydata
ico_polydata.SetPoints(vtk_coords) # add points
ico_polydata.SetPolys(vtk_triangles) # add polys
"""

# convert ico verts / faces to tensor
ico_verts = torch.from_numpy(coords).unsqueeze(0).to(device)
ico_faces = torch.from_numpy(triangles).unsqueeze(0).to(device)


# load train / test splits
train_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Segmentation/M-CRIB-S_train_TEA.npy'
val_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Segmentation/M-CRIB-S_val_TEA.npy'
train_split = np.load(train_split_path)
val_split = np.load(val_split_path)
train_dataset = BrainDataset(train_split,triangles)
val_dataset = BrainDataset(val_split,triangles)

# Setup training parameters
dist_cam = 2
batch_size =4
nb_epochs = 1_000
nb_loops = 12

# Match icosphere vertices and faces tensor with batch size
l_ico_verts = []
l_ico_faces = []
for i in range(batch_size):
    l_ico_verts.append(ico_verts)
    l_ico_faces.append(ico_faces)    
ico_verts = torch.cat(l_ico_verts,dim=0)
ico_faces = torch.cat(l_ico_faces,dim=0)

    
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size, shuffle=True)


# In[8]:


for epoch in range(nb_epochs):
    print("-" * 20)
    print(f"epoch {epoch + 1}/{nb_epochs}")
    epoch_loss = 0
    step = 0
    for batch, (vertex_features, face_features, face_labels) in enumerate(train_dataloader):
        
        vertex_features = vertex_features.to(device)
        vertex_features = vertex_features[:,:,0:3]
        face_labels = torch.squeeze(face_labels,0)
        face_labels = face_labels.to(device)
        face_features = face_features.to(device)

        step += 1
        for s in range(nb_loops):
            textures = TexturesVertex(verts_features=vertex_features)
            ic(vertex_features.shape)
            meshes = Meshes(
                verts=ico_verts,   
                faces=ico_faces, 
                textures=textures
            )
            array_coord = np.random.normal(0, 1, 3)
            array_coord *= dist_cam/(np.linalg.norm(array_coord))
            camera_position = ToTensor(dtype=torch.float32, device=device)([array_coord.tolist()])
            R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

            images = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)    
            pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone())  


            #inputs = torch.take(face_features,pix_to_face)*(pix_to_face >= 0) 
            labels = torch.take(face_labels, pix_to_face)*(pix_to_face >= 0) 

            l_inputs = []
            for index in range(4):
                l_inputs.append(torch.take(face_features[:,:,index],pix_to_face)*(pix_to_face >= 0)) # take each feature        

            inputs = torch.cat(l_inputs,dim=3)
            
            inputs = inputs.permute(0,3,1,2)
            labels = labels.permute(0,3,1,2)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = int(np.ceil(len(train_dataloader) / train_dataloader.batch_size))
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            break
            
        
        """
        ic(ico_verts.shape)
        ic(ico_faces.shape)
        ic(vertex_features.shape)
        ic(face_features.shape)
        ic(face_labels.shape)
        ic(type(ico_verts))
        ic(type(ico_faces))
        ic(type(vertex_features))   
        ic(inputs.shape)
        ic(face_labels)
        """
        break
    break

    

