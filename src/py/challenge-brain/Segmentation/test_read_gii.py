import numpy   as np
import nibabel as nib
from fsl.data import gifti
from icecream import ic
import sys
sys.path.insert(0,'../..')
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
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)

class BrainDataset(Dataset):
	def __init__(self,np_split):
		self.np_split = np_split
	
	def __len__(self):
		return(len(self.np_split))

	def __getitem__(self,idx):
		data_dir = '/CMF/data/geometric-deep-learning-benchmarking/Data/Segmentation/Native_Space'
		item = self.np_split[idx][0]

		# for now just try with Left
		path_features = f'{data_dir}/segmentation_native_space_features/{item}_L.shape.gii'
		path_labels = f'{data_dir}/segmentation_native_space_labels/{item}_L.label.gii'
		data_features = gifti.loadGiftiVertexData(path_features)
		data_labels = gifti.loadGiftiVertexData(path_labels)

		return data_features,data_labels

def main():

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
	    image_size=224, 
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


	path_features = '/CMF/data/geometric-deep-learning-benchmarking/Data/Segmentation/Native_Space/segmentation_native_space_features/sub-CC00062XX05_ses-13801_R.shape.gii'
	path_ico = '/CMF/data/geometric-deep-learning-benchmarking/Icospheres/ico-1.surf.gii'

	#mesh = gifti.loadGiftiMesh(path_ico)

	# load icosahedron
	surf = nib.load(path_ico)

	# extract points and faces
	coords = surf.agg_data('pointset')
	triangles = surf.agg_data('triangle')
	nb_triangles = len(triangles)
	connectivity = triangles.reshape(nb_triangles*3,1) # 3 points per triangle
	connectivity = np.int64(connectivity)	
	offsets = [3*i for i in range (nb_triangles)]
	offsets.append(nb_triangles*3) #  The last value is always the length of the Connectivity array.
	offsets = np.array(offsets)

	# convert to vtk
	vtk_coords = vtk.vtkPoints()
	vtk_coords.SetData(numpy_to_vtk(coords))
	vtk_triangles = vtk.vtkCellArray()
	vtk_offsets = numpy_to_vtkIdTypeArray(offsets)
	vtk_connectivity = numpy_to_vtkIdTypeArray(connectivity)
	vtk_triangles.SetData(vtk_offsets,vtk_connectivity)

	# Create icosahedron as a VTK polydata 
	ico_polydata = vtk.vtkPolyData() # initialize polydata
	ico_polydata.SetPoints(vtk_coords) # add points
	ico_polydata.SetPolys(vtk_triangles) # add polys

	# test: write surface
	# utils.Write(ico_polydata,"/home/leclercq/Documents/TOEIJT.vtk")

	# convert ico verts / faces to tensor
	ico_verts = torch.from_numpy(coords)
	ico_faces = torch.from_numpy(triangles)


	# load train / test splits
	train_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Segmentation/M-CRIB-S_train_TEA.npy'
	val_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Segmentation/M-CRIB-S_val_TEA.npy'
	train_split = np.load(train_split_path)
	val_split = np.load(val_split_path)
	train_data = BrainDataset(train_split)
	val_dataset = BrainDataset(val_split)
	train_item, val_item = train_data[5]





if __name__ == '__main__':
	main()
