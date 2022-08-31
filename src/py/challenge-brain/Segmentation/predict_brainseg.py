print("Importing libraries...")


####
####
"""
V4: Run on whole directory
    Random crown removal
"""
####
####

import os
import glob
import argparse
import torch
import time
from tqdm import tqdm
import numpy as np
import random
import math
import nibabel as nib
from fsl.data import gifti
from icecream import ic

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader, PointLights,AmbientLights,TexturesVertex
)
from vtk import vtkPolyData, vtkPoints, vtkCellArray
from vtk import vtkPolyDataWriter
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
import sys

code_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])
sys.path.append(code_path)  

import utils
import post_process

import monai
from monai.inferers import (sliding_window_inference,SimpleInferer)
from monai.transforms import ToTensor

print("Initializing model...")
# Set the cuda device 
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)
else:
  device = torch.device("cpu") 

def main(args):
  # Initialize a perspective camera.
  cameras = FoVPerspectiveCameras(device=device)
  image_size = args.res
  # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
  raster_settings = RasterizationSettings(
      image_size=image_size, 
      blur_radius=0, 
      faces_per_pixel=1, 
  )
  # We can add a point light in front of the object. 

  lights = AmbientLights(device=device)
  rasterizer = MeshRasterizer(cameras=cameras,raster_settings=raster_settings)
  phong_renderer = MeshRenderer(rasterizer=rasterizer,shader=HardPhongShader(device=device, cameras=cameras, lights=lights))
  num_classes = 37 


  model = monai.networks.nets.UNet(
      spatial_dims=2,
      in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
      out_channels=num_classes, 
      channels=(16, 32, 64, 128, 256),
      strides=(2, 2, 2, 2),
      num_res_units=2,
  )



  try:
    model.load_state_dict(torch.load(args.model,map_location=device))
  except RuntimeError: # if training was done with DDP
    from collections import OrderedDict
    dict_load2 = OrderedDict()
    dict_load = torch.load(args.model,map_location=device)
    for k in dict_load:
      newkey = k[7:]
      dict_load2[newkey] = dict_load[k]
    model.load_state_dict(dict_load2)

  model.to(device)
  softmax = torch.nn.Softmax(dim=1)

  nb_rotations = args.rot
  l_outputs = []

  ## Camera position
  dist_cam = 2


  path = args.surf
  if os.path.isdir(path):
    l_inputs = glob.glob(f"{path}/*.gii")

    if not (os.path.isdir(args.out)):
      raise Exception ('The input is a folder, but the output is not.')
  elif os.path.isfile(path):
    l_inputs = [path]
  else:
    raise Exception ('Incorrect input.')



  ico_verts,ico_faces,ico_verts_tensor,ico_faces_tensor,faces_pid0 = LoadIcosahedron(args.ico)
  nb_faces = len(ico_faces)

  for index,path in enumerate(l_inputs):
    print(f'\nFile {index+1}/{len(l_inputs)}:')

    if os.path.isdir(args.out):
      output = f'{args.out}/{os.path.splitext(os.path.basename(path))[0]}_out.gii'
    else:
      output = args.out


    vertex_features,face_features =  GetFeatures(path,nb_faces,faces_pid0) 


    array_faces = np.zeros((num_classes,nb_faces))
    tensor_faces = torch.zeros(num_classes,nb_faces).to(device)
    model.eval() # Switch to eval mode
    simple_inferer = SimpleInferer()


    
    list_sphere_points = fibonacci_sphere(samples=nb_rotations, dist_cam=dist_cam)

    # list_sphere_points[0] = (0.0001, 1.35, 0.0001) # To avoid "invalid rotation matrix" error
    # list_sphere_points[-1] = (0.0001, -1.35, 0.0001)



    ###
    ###
    # PREDICTION
    ###
    ###
    # for verts in tqdm(list_sphere_points, desc = 'Prediction      '):
    for idx in tqdm(range(nb_rotations), desc = 'Prediction      '):

      inputs,pix_to_face = GetView(vertex_features,face_features, ico_verts_tensor,ico_faces_tensor, phong_renderer,dist_cam,device)
      import pandas as pd
      df =  pd.DataFrame(pix_to_face.detach().cpu().numpy())
      df.to_csv('out.csv')
      outputs = simple_inferer(inputs,model)  
      outputs_softmax = softmax(outputs).squeeze().detach().cpu().numpy() # t: negligeable  

      for x in range(image_size):
          for y in range (image_size): # Browse pixel by pixel
              array_faces[:,pix_to_face[x,y]] += outputs_softmax[...,x,y]

    array_faces[:,-1][0] = 0 # pixels that are background (id: 0) =-1
    faces_argmax = np.argmax(array_faces,axis=0)
    final_faces_array = faces_argmax
    # NO MASK WHEN FOR BLANK FACE
    unique, counts  = np.unique(final_faces_array, return_counts = True)

    # CREATE VTK POLYDATA
    vtk_ico,np_connectivity = CreateVTKIco(ico_verts,ico_faces)

    nb_points = vtk_ico.GetNumberOfPoints()
    polys = vtk_ico.GetPolys()
    #np_connectivity = vtk_to_numpy(polys.GetConnectivityArray())


    # id_points = np.random.randint(37, size=nb_points) # fill with random ID
    id_points = np.full((nb_points,),num_classes+1) # fill with ID that's going to be removed afterwards

    for index,uid in enumerate(final_faces_array.tolist()):
        id_points[np_connectivity[3*index]] = uid

    #ic(id_points.shape)
    vtk_id = numpy_to_vtk(id_points)
    vtk_id.SetName(args.scal)
    vtk_ico.GetPointData().AddArray(vtk_id)
    print(id_points.shape)

    ###
    ###
    # POST-PROCESS
    ###
    ###

    # Remove Islands
    for label in tqdm(range(num_classes),desc = 'Removing islands'):
      post_process.RemoveIslands(vtk_ico, vtk_id, label, 2,ignore_neg1 = True) 

    post_process.RemoveIslands(vtk_ico, vtk_id, num_classes+1, 20,ignore_neg1 = True) 



    path_labels = "/NIRAL/work/leclercq/data/geometric-deep-learning-benchmarking/Data/Segmentation/Native_Space/segmentation_native_space_labels/sub-CC00060XX03_ses-12501_R.label.gii"
    vertex_labels = gifti.loadGiftiVertexData(path_labels)[1] # vertex labels
    vertex_labels = np.squeeze(vertex_labels)
    vtk_truth_id = numpy_to_vtk(vertex_labels)
    vtk_truth_id.SetName("TRUTH")
    vtk_ico.GetPointData().AddArray(vtk_truth_id)


    # Output: all teeth + gum
    utils.Write(vtk_ico,output)





  print("Done.")


def fibonacci_sphere(samples, dist_cam):

  # points = []
  # phi = math.pi * (3. -math.sqrt(5.))  # golden angle in radians
  # for i in range(samples):
  #     y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
  #     radius = math.sqrt(1 - y*y)  # radius at y
  #     theta = phi*i 
  #     x = math.cos(theta)*radius
  #     z = math.sin(theta)*radius
  #     points.append((x*dist_cam, y*dist_cam, z*dist_cam))
  # return points

  n = samples
  i = np.arange(0, n, dtype=float) + 0.5
  phi = np.arccos(1 - 2*i/n)
  goldenRatio = (1 + 5**0.5)/2
  theta = 2 * np.pi * i / goldenRatio
  x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
  list_sphere_points = []
  for idx in range (samples):
    list_sphere_points.append([dist_cam * x[idx],dist_cam * y[idx],dist_cam * z[idx]])
  return list_sphere_points


def LoadIcosahedron(path_ico):
  # load icosahedron
  ico_surf = nib.load(path_ico)

  # extract points and faces
  verts = ico_surf.agg_data('pointset')
  faces = ico_surf.agg_data('triangle')
  nb_faces = len(faces)

  """
  connectivity = faces.reshape(nb_faces*3,1) # 3 points per triangle
  connectivity = np.int64(connectivity) 
  offsets = [3*i for i in range (nb_faces)]
  offsets.append(nb_faces*3) #  The last value is always the length of the Connectivity array.
  offsets = np.array(offsets)
  """

  # rescale icosphere [0,1]
  verts = np.multiply(verts,0.01)

  # convert ico verts / faces to tensor
  ico_verts = torch.from_numpy(verts).unsqueeze(0).to(device)
  ico_faces = torch.from_numpy(faces).unsqueeze(0).to(device)
  faces_pid0 = faces[:,0:1]
  return verts,faces,ico_verts,ico_faces,faces_pid0


def CreateVTKIco(verts,faces):
    nb_faces = len(faces)
    connectivity = faces.reshape(nb_faces*3,1) # 3 points per triangle
    connectivity = np.int64(connectivity) 
    a = np.array([1])

    ic('')
    ic('')

    offsets = [3*i for i in range (nb_faces)]
    offsets.append(nb_faces*3) #  The last value is always the length of the Connectivity array.
    offsets = np.array(offsets)

    # convert to vtk
    vtk_verts = vtkPoints()
    vtk_verts.SetData(numpy_to_vtk(verts))
    vtk_faces = vtkCellArray()
    vtk_offsets = numpy_to_vtkIdTypeArray(offsets)
    vtk_connectivity = numpy_to_vtkIdTypeArray(connectivity)
    vtk_faces.SetData(vtk_offsets,vtk_connectivity)

    # Create icosahedron as a VTK polydata 
    ico_polydata = vtkPolyData() # initialize polydata
    ico_polydata.SetPoints(vtk_verts) # add points
    ico_polydata.SetPolys(vtk_faces) # add polys


    return ico_polydata,connectivity

def GetFeatures(path,nb_faces,faces_pid0):
  vertex_features = gifti.loadGiftiVertexData(path)[1] # vertex features
  #offset = np.arange(self.nb_faces*4).reshape((self.nb_faces,4))
  offset = np.zeros((nb_faces,4), dtype=int) + np.array([0,1,2,3])
  faces_pid0_offset = offset + np.multiply(faces_pid0,4)        

  face_features = np.take(vertex_features,faces_pid0_offset)
  vertex_features = torch.tensor(vertex_features).unsqueeze(0).to(device)
  vertex_features = vertex_features[:,:,0:3]
  face_features = torch.tensor(face_features).unsqueeze(0).to(device)


  return vertex_features,face_features

def GetView(vertex_features,face_features,
             ico_verts,ico_faces,
             phong_renderer,dist_cam,device):
  textures = TexturesVertex(verts_features=vertex_features)

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

  l_features = []
  for index in range(4):
      l_features.append(torch.take(face_features[:,:,index],pix_to_face)*(pix_to_face >= 0)) # take each feature     
  inputs = torch.cat(l_features,dim=3)        
  inputs  = inputs.permute(0,3,1,2)
  pix_to_face = pix_to_face.squeeze()
  return inputs,pix_to_face




if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Choose a .vtk file.')
  parser.add_argument('--surf',type=str, help='Input. Either .vtk file or folder containing vtk files.', required=True)
  parser.add_argument('--ico',type=str, help='Path for icosahedron file : ico-6.', required=True)
  parser.add_argument('--out',type=str,help ='Name of output file is input is a single file, or name of output folder if input is a folder.',required=True)
  parser.add_argument('--rot',type=int, help = 'Number of rotations (default: 40)', default=40)
  parser.add_argument('--model',type=str, help = 'Model for segmentation', default="best_metric_model_segmentation2d_array.pth")
  parser.add_argument('--res',type=int, help = 'Image resolution for the fly-by (default: 512)', default=512)
  parser.add_argument('--scal',type=str, help = 'Predicted ID name', default="PredictedID")
  parser.add_argument('--sep', help = 'Create one file per label', action='store_true')
  args = parser.parse_args()
  main(args)
