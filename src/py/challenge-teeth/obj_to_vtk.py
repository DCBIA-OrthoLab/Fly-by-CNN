import pandas as pd
from icecream import ic
import vtk
import json
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.insert(0,'..')
import utils

csv_path = '/NIRAL/work/leclercq/data/challenge_teeth_all.csv'
output_dir = '/NIRAL/work/leclercq/source/3DTeethSeg22_challenge_07-19/test/test_local/'
df_split = pd.read_csv(csv_path, dtype = str)
df = df_split.reset_index(drop=True)





LUT = np.array([33,0,0,0,0,0,0,0,0,0,0,8,7,6,5,4,3,2,1,0,0,9,10,11,12,13,14,15,
											16,0,0,24,23,22,21,20,19,18,17,0,0,25,26,27,28,29,30,31,32])

pbar = tqdm(range(len(df)),desc='Converting...', total=len(df))
for idx in pbar:

	# load obj
	model = df.iloc[idx]
	#surf_path = model['surf']
	surf_path = "/NIRAL/work/leclercq/source/3DTeethSeg22_challenge_07-19/test/0JN50XQR_lower.obj"
	pbar.set_description(f'{surf_path}')
	#label_path = model['label']
	label_path = '/NIRAL/work/leclercq/source/3DTeethSeg22_challenge_07-19/test/test_local/expected_output.json'
	split = model['split']
	reader = vtk.vtkOBJReader()
	reader.SetFileName(surf_path)
	reader.Update()
	surf = reader.GetOutput()

	# extract verts and faces
	verts = vtk_to_numpy(surf.GetPoints().GetData())
	faces = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]  

	# get labels
	with open(label_path) as f:
		json_data = json.load(f)
	vertex_labels_FDI = np.array(json_data['labels'])  # FDI World Dental Federation notation	

	# convert to universal label
	vertex_labels = LUT[vertex_labels_FDI] # UNIVERSAL NUMBERING SYSTEM

	vertex_instances = np.array(json_data['instances'])

	# convert to vtk
	vertex_labels_vtk = numpy_to_vtk(vertex_labels)
	vertex_labels_vtk.SetName("UniversalID")

	vertex_instances_vtk = numpy_to_vtk(vertex_instances)
	vertex_instances_vtk.SetName("instances")

	surf.GetPointData().AddArray(vertex_labels_vtk)
	surf.GetPointData().AddArray(vertex_instances_vtk)

	# write file 
	file_basename = Path(surf_path).stem
	#out_path = f'{output_dir}/{split}/{file_basename}.vtk'
	out_path = f'{output_dir}/{file_basename}.vtk'
	utils.Write(surf,out_path,print_out=False)
	break

