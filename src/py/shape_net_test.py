import shape_net_dataset as snd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import utils
import numpy as np

data_dir = "/work/jprieto/data/ShapeNet/ShapeNetCore.v1"
csv_split = "/work/jprieto/data/ShapeNet/ShapeNetCore.v1/all.csv"

snd_train = snd.ShapeNetDataset(data_dir, csv_split=csv_split, split="train")


# train_dataloader = DataLoader(snd_train, batch_size=4)
# for batch, (img_np, img_np_z, synset_class) in enumerate(train_dataloader):

# 	idx = 1
# 	plt.figure(figsize=(20, 20))
# 	for binp in img_np:
# 		for inp in binp:			
# 			plt.subplot(4, 12, idx)
# 			plt.imshow(inp)
# 			plt.grid(False)
# 			idx += 1
# 	plt.show()

# 	idx = 1
# 	plt.figure(figsize=(20, 20))
# 	for binp in img_np_z:
# 		for inp in binp:			
# 			plt.subplot(4, 12, idx)
# 			plt.imshow(inp)
# 			plt.grid(False)
# 			idx += 1
# 	plt.show()





for batch, (img_np, img_np_z, synset_class) in enumerate(snd_train):
	print(img_np.shape)
	idx = 1
	plt.figure(figsize=(20, 20))	
	for inp in img_np:
		print(idx)
		plt.subplot(4, 3, idx)
		plt.imshow(inp)
		plt.grid(False)
		idx += 1
	plt.show()

	idx = 1
	plt.figure(figsize=(20, 20))
	for inp in img_np_z:
		print(np.max(inp))
		plt.subplot(4, 3, idx)
		plt.imshow(inp)
		plt.grid(False)
		idx += 1
	plt.show()











# import torch
# from pytorch3d.ops import mesh_face_areas_normals

# # io utils
# from pytorch3d.io import load_obj

# # datastructures
# from pytorch3d.structures import Meshes
# from pytorch3d.structures import utils as struct_utils
# # 3D transformations functions
# from pytorch3d.transforms import Rotate, Translate

# # rendering components
# from pytorch3d.renderer import (
#     FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
#     RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
#     SoftSilhouetteShader, HardFlatShader, HardPhongShader, SoftPhongShader, HardGouraudShader, SoftGouraudShader, AmbientLights, PointLights, TexturesVertex,
# )

# from pytorch3d.ops import mesh_face_areas_normals



# def UnitVerts(verts):
#     min_verts, _ = torch.min(verts, axis=0)
#     max_verts, _ = torch.max(verts, axis=0)
#     mean_v = (min_verts + max_verts)/2.0
    
#     verts = verts - mean_v
#     scale_factor = 1/torch.linalg.vector_norm(max_verts - mean_v)
#     verts = verts*scale_factor
    
#     return verts, mean_v, scale_factor

# def ComputeVertexNormals(verts, faces):
#     face_area, face_normals = mesh_face_areas_normals(verts, faces)

#     vert_normals = []

#     for idx in range(len(v)):
#         normals = face_normals[(faces == idx).nonzero(as_tuple=True)[0]] #Get all adjacent normal faces for the given point id
#         areas = face_area[(faces == idx).nonzero(as_tuple=True)[0]] # Get all adjacent normal areas for the given point id

#         normals = torch.mul(normals, areas.reshape(-1, 1)) # scale each normal by the area
#         normals = torch.sum(normals, axis=0) # sum everything
#         normals = torch.nn.functional.normalize(normals, dim=0) #normalize

#         vert_normals.append(normals.numpy())
    
#     return torch.as_tensor(vert_normals)



# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)
# else:
#     device = torch.device("cpu")

# # Initialize a perspective camera.
# cameras = FoVPerspectiveCameras(device=device)

# # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
# raster_settings = RasterizationSettings(
#     image_size=224, 
#     blur_radius=0.0, 
#     faces_per_pixel=1, 
# )
# # We can add a point light in front of the object. 
# # lights = AmbientLights(device=device)
# # lights = PointLights(device=device, location=[[0.0, 0.0, 1.0]])

# phong_renderer = MeshRenderer(
#     rasterizer=MeshRasterizer(
#         cameras=cameras, 
#         raster_settings=raster_settings
#     ),
#     shader=HardFlatShader(
#         device=device, 
#         cameras=cameras        
#         # lights=lights
#     )
# )


# SNC_train = snd.ShapeNetCoreSplit(data_dir, csv_split=csv_split, split="train")

# m = SNC_train[20]
# v, mean, scale = UnitVerts(m["verts"])
# normals = ComputeVertexNormals(m["verts"], m["faces"])

# verts = m["verts"]
# faces = m["faces"]

# # Initialize each vertex to be white in color.
# verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
# textures = TexturesVertex(verts_features=verts_rgb.to(device))

# # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
# mesh = Meshes(
#     verts=[verts.to(device)],   
#     faces=[faces.to(device)], 
#     textures=textures
# )

# # Select the viewpoint using spherical angles  
# distance = 0.5  # distance from camera to the object
# elevation = 50.0   # angle of elevation in degrees
# azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 

# # Get the position of the camera based on the spherical angles
# R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

# # Render the teapot providing the values of R and T. 
# image_ref = phong_renderer(meshes_world=mesh, R=R, T=T)

# image_ref = image_ref.cpu().numpy()

# plt.figure(figsize=(20, 20))
# plt.imshow(image_ref.squeeze())
# plt.grid(False)
# plt.show()