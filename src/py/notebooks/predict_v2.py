print("Importing libraries...")
import os
import argparse
import torch
from tqdm import tqdm
import numpy as np

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader, PointLights,TexturesVertex
)

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import sys
sys.path.insert(0,'..')
import fly_by_features as fbf
import post_process

import monai
from monai.inferers import (sliding_window_inference,SimpleInferer)
from monai.transforms import ToTensor

# Imports for monai model


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

  lights = PointLights(device=device)
  rasterizer = MeshRasterizer(
          cameras=cameras, 
          raster_settings=raster_settings
      )
  phong_renderer = MeshRenderer(
      rasterizer=rasterizer,
      shader=HardPhongShader(device=device, cameras=cameras)
  )
  num_classes = 34
  # create UNet, DiceLoss and Adam optimizer
  model = monai.networks.nets.UNet(
      spatial_dims=2,
      in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
      out_channels=num_classes, 
      channels=(16, 32, 64, 128, 256),
      strides=(2, 2, 2, 2),
      num_res_units=2,
  ).to(device)

  model.load_state_dict(torch.load(args.model))
  path = args.surf
  dist_cam = 1.35
  softmax = torch.nn.Softmax(dim=1)

  nb_rotations = args.rot
  l_outputs = []

  camera_position = ToTensor(dtype=torch.float32, device=device)([[0, 0, dist_cam]])
  R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
  T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)


  (V, F, Y, YF, F0, CN,surf) = GetSurfProp(path)
  num_faces = F.size(1)
  l_faces = []
  for i in range(num_faces):
      l_faces.append([])
 
  array_faces = np.zeros((num_classes,num_faces))
  print('array faces: ',array_faces.shape)
  model.eval() # Switch to eval mode
  simple_inferer = SimpleInferer()

  ## PREDICTION
  for i in tqdm(range(nb_rotations),total = nb_rotations, desc = 'Prediction      '):
      (V, F, Y, YF, F0, CN,surf) = GetSurfProp(path)
      
      textures = TexturesVertex(verts_features=CN)
      meshes = Meshes(verts=V, faces=F, textures=textures)
      images = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
      pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone())
      images = images.permute(0,3,1,2)
      inputs = images.to(device)
      outputs = simple_inferer(inputs,model)
      print(outputs.shape)
      outputs_softmax = softmax(outputs)
      print(outputs[:,0,...])
      print('outputs: \n',outputs[...,128,128])
      print('outputs_softmax: \n',outputs_softmax[...,128,128])
      outputs_softmax = outputs_softmax.squeeze().detach().cpu()
      outputs_softmax = outputs_softmax.numpy()
      outputs_argmax = torch.argmax(outputs, dim=1).detach().cpu()
      print(outputs_argmax.shape)
      for x in range(image_size):
          for y in range (image_size): # Browse pixel by pixel
              l_faces[pix_to_face[0,x,y,0]].append(outputs_argmax[0,x,y].item()) # .item(): returns number instead of tensor
              """
              print("shape array_faces[:,pix_to_face[0,x,y,0]] : ",array_faces[:,pix_to_face[0,x,y,0]].shape )
              print("shape outputs_softmax[...,x,y] : ", outputs_softmax[...,x,y].shape)

              print("type array_faces[:,pix_to_face[0,x,y,0]] : ",type(array_faces[:,pix_to_face[0,x,y,0]]))
              print("type outputs_softmax[...,x,y] : ", type(outputs_softmax[...,x,y]))
              """
              array_faces[:,pix_to_face[0,x,y,0]] += outputs_softmax[...,x,y]           

  l_faces[-1] = [x for x in l_faces[-1] if x!= 0] # pixels that are to assigned to any face get value -1: last position in the list
  l_uniq = l_faces
  for index, value in enumerate(l_faces):
      if value:
          l_uniq[index] = max(set(value),key=value.count)                
      else:
          l_uniq[index] = 33 #ID of the gum 


  surf = fbf.ReadSurf(path)
  nb_points = surf.GetNumberOfPoints()
  polys = surf.GetPolys()
  np_connectivity = vtk_to_numpy(polys.GetConnectivityArray())

  id_points = np.full((nb_points,),33) # fill with ID 33 (gum)

  for index,uid in enumerate (l_uniq):
      id_points[np_connectivity[3*index]] = uid    


  vtk_id = numpy_to_vtk(id_points)
  vtk_id.SetName(args.scal)
  surf.GetPointData().AddArray(vtk_id)

  # Remove Islands
  for label in tqdm(range(num_classes),desc = 'Removing islands'):
    post_process.RemoveIslands(surf, vtk_id, label, 200)

  out_filename = args.out
  polydatawriter = vtk.vtkPolyDataWriter()
  polydatawriter.SetFileName(out_filename)
  polydatawriter.SetInputData(surf)
  polydatawriter.Write()
  print("Done.")


def GetSurfProp(path):        
    surf = fbf.ReadSurf(path)
    surf = fbf.GetUnitSurf(surf)
    surf, _a, _v = fbf.RandomRotation(surf)
    surf = fbf.ComputeNormals(surf)
    color_normals = ToTensor(dtype=torch.float32, device=device)(vtk_to_numpy(fbf.GetColorArray(surf, "Normals"))/255.0)
    verts = ToTensor(dtype=torch.float32, device=device)(vtk_to_numpy(surf.GetPoints().GetData()))
    faces = ToTensor(dtype=torch.int64, device=device)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
    region_id = ToTensor(dtype=torch.int64, device=device)(vtk_to_numpy(surf.GetPointData().GetScalars("UniversalID")))
    region_id = torch.clamp(region_id, min=0)
    faces_pid0 = faces[:,0:1]
    region_id_faces = torch.take(region_id, faces_pid0)
    return verts.unsqueeze(0), faces.unsqueeze(0), region_id.unsqueeze(0), region_id_faces.unsqueeze(0), faces_pid0.unsqueeze(0), color_normals.unsqueeze(0),surf

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Choose a .vtk file of a jaw.')
  parser.add_argument('--surf',type=str, help='Input surface (.vtk file)', required=True)
  parser.add_argument('--out',type=str, help = 'Output', required=True)
  parser.add_argument('--rot',type=int, help = 'Number of rotations (default: 70)', default=70)
  parser.add_argument('--model',type=str, help = 'Model for segmentation', default="best_metric_model_segmentation2d_array.pth")
  parser.add_argument('--res',type=int, help = 'Image resolution for the fly-by (default: 256)', default=256)
  parser.add_argument('--scal',type=str, help = 'Predicted ID name', default="PredictedID")
  args = parser.parse_args()
  main(args)