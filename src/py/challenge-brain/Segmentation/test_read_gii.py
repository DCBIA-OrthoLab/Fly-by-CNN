import numpy   as np
import nibabel as nib
from fsl.data import gifti
from icecream import ic
import sys
sys.path.insert(0,'../..')
import utils
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
from torch.utils.data import Dataset
from torch.utils.data import DataLoader




class BrainDataset(Dataset):
	def __init__(self,np_split):
		self.np_split = np_split
	
	def __len__(self):
		return(len(self.np_split))

	def __getitem__(self,idx):
		data_dir = '/CMF/data/geometric-deep-learning-benchmarking/Data/Segmentation/Native_Space'
		item = self.np_split[idx]

		# for now just try with Left
		path_features = f'{data_dir}/segmentation_native_space_features/{item}_L.shape.gii'
		path_labels = f'{data_dir}/segmentation_native_space_labels/{item}_L.label.gii'
		data_features = gifti.loadGiftiVertexData(path_features)
		data_labels = gifti.loadGiftiVertexData(path_labels)

def main():
	path_features = '/CMF/data/geometric-deep-learning-benchmarking/Data/Segmentation/Native_Space/segmentation_native_space_features/sub-CC00062XX05_ses-13801_R.shape.gii'
	path_ico = '/CMF/data/geometric-deep-learning-benchmarking/Icospheres/ico-1.surf.gii'
	# vertex_data = gifti.loadGiftiVertexData(path_ico)  # tuple[0] : gifti image, [1] : list of len dim0
	#mesh = gifti.loadGiftiMesh(path_ico)

	"""
	# Load icosphere mesh
	Mesh = gifti.GiftiMesh(path_ico)
	# Add features data
	Mesh.loadVertexData(path_features)
	"""
	# load icosahedron
	surf = nib.load(path_ico)

	# extract points and faces
	coords = surf.agg_data('pointset')
	triangles = surf.agg_data('triangle')
	nb_triangles = len(triangles)
	ic(nb_triangles)
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

	# load train / test splits
	train_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Segmentation/M-CRIB-S_train_TEA.npy'
	val_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Segmentation/M-CRIB-S_val_TEA.npy'
	train_split = np.load(train_split_path)
	val_split = np.load(val_split_path)
	
	train_data = BrainDataset(train_split)
	val_dataset = BrainDataset(val_split)



if __name__ == '__main__':
	main()
