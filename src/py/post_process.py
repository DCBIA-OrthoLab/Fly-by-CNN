import vtk
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mesh', help='Insert mesh path')
parser.add_argument('--out', help='Insert output path+name')
arg = parser.parse_args()

def ChangeLabel(vtkdata, label_array, label2change, change):
	# Set all the label 'label2change' in 'change'
	for pid in range (vtkdata.GetNumberOfPoints()):
		if int(label_array.GetTuple(pid)[0]) == label2change:
			label_array.SetTuple(pid, (change, ))
	return vtkdata, label_array

def Clip(vtkdata, value, scalars, InsideOut):
	# CLip the vtkdata following the scalar
	# scalars = 'RegionID' | 'Minimum_Curvature'
	# InsideOut = 'on' | 'off'
	vtkdata.GetPointData().SetActiveScalars(scalars)
	Clipper = vtk.vtkClipPolyData()
	Clipper.SetInputData(vtkdata)
	Clipper.SetValue(value)
	Clipper.GenerateClipScalarsOff()
	if InsideOut == 'off':
		Clipper.InsideOutOff()
	else:
		Clipper.InsideOutOn()
	Clipper.GetOutput().GetPointData().CopyScalarsOff()
	Clipper.Update()
	clipped_name = Clipper.GetOutput()
	return clipped_name

def Connectivity(vtkdata):
	# Labelize all the objects
	connectivityFilter = vtk.vtkConnectivityFilter()
	connectivityFilter.SetInputData(vtkdata)
	connectivityFilter.SetExtractionModeToAllRegions()
	connectivityFilter.ColorRegionsOn()
	connectivityFilter.Update()
	vtkdata = connectivityFilter.GetOutput()
	label_label = vtkdata.GetPointData().GetArray('RegionId')
	return vtkdata, label_label

def CountIDPoint(vtkdata, label_array):
	# Count the number of point of each IDs
	number_of_points = []
	for label in range(np.max(np.array(label_array)) + 1):
		current_nb = 0
		for pid in range(vtkdata.GetNumberOfPoints()):
			if int(label_array.GetTuple(pid)[0]) == label:
				current_nb += 1
		number_of_points.append(current_nb)
	return number_of_points

def ChangeSmallCompID(vtkdata, label_array, number_of_points, threshold, label):
	#Set the labek of all the object smaller than the threshold into 'label'
	for i in range(len(number_of_points)):
		number = number_of_points[i]
		if number < threshold :
			for pid in range (vtkdata.GetNumberOfPoints()):
				if int(label_array.GetTuple(pid)[0]) == i:
					label_array.SetTuple(pid, (label, ))
	return vtkdata


def LocateLabels(vtkdata_vtkdata, label_array_or, vtkdata_clipped, label_array_clip, label2change, label):
	# Assign the label 'label' to the vtkdata vtkdata points which are labeled 'label2change' in the clipped vtkdata 
	locator = vtk.vtkPointLocator()
	locator.SetDataSet(vtkdata_vtkdata) 
	locator.BuildLocator()
	number_of_changes = 0
	for pid in range(vtkdata_clipped.GetNumberOfPoints()):
		if int(label_array_clip.GetTuple(pid)[0]) == label2change:
			coordinates = vtkdata_clipped.GetPoint(pid)
			vtkdata_Equivalent_ID = locator.FindClosestPoint(coordinates)
			label_array_or.SetTuple(vtkdata_Equivalent_ID, (label, ))
			number_of_changes += 1
	# print('LocateLabels ===> ', 'Number of changes :', number_of_changes)
	return vtkdata_vtkdata

def Write(vtkdata, output_name):
	outfilename = output_name
	print("Writting:", outfilename)
	polydatawriter = vtk.vtkPolyDataWriter()
	polydatawriter.SetFileName(outfilename)
	polydatawriter.SetInputData(vtkdata)
	polydatawriter.Write()


def GetCurvature(vtkdata):
	curve=vtk.vtkCurvatures()
	curve.SetCurvatureTypeToMinimum()
	curve.SetInputData(vtkdata)
	curve.Update()
	vtkdata=curve.GetOutput()
	return vtkdata

def RegionGrowing(vtkdata, label_array, label2change, exception):
	#Take a look of the neghbor's id and take it if it s different set it to label2change. Exception is done to avoid unwanted labels.
	ids = []
	for pid in range(vtkdata.GetNumberOfPoints()):
		if int(label_array.GetTuple(pid)[0]) == label2change:
			ids.append(pid)
	first_len = len(ids)
	# print('RegionGrowing ===> ', 'Number of ids that will change :', first_len)
	count = 0
	while len(ids) > 0:
		count += 1
		for pid in ids[:]:
			neighbor_ids = NeighborPoints(vtkdata, pid)
			for nid in neighbor_ids:
				neighbor_label = int(label_array.GetTuple(nid)[0])
				if(neighbor_label != label2change) and (neighbor_label != exception):
					label_array.SetTuple(pid, (neighbor_label,))
					ids.remove(pid)
					count = 0
					break
		if count == 2:
			#Sometimes the while loop can t find any label != -1 
			print('RegionGrowing ===> WARNING :', len(ids), '/', first_len, 'label(s) has been undertermined. Setting to 0.')
			break		

	for pid in ids:
		#Then we set theses label to 0
		label_array.SetTuple(pid, (0,))
	return vtkdata

def NeighborPoints(vtkdata,CurrentID):
	cells_id = vtk.vtkIdList()
	vtkdata.GetPointCells(CurrentID, cells_id)
	all_neighbor_pid = []
	for ci in range(cells_id.GetNumberOfIds()):
		cells_id_inner = vtk.vtkIdList()
		vtkdata.GetCellPoints(cells_id.GetId(ci), cells_id_inner)
		for pi in range(cells_id_inner.GetNumberOfIds()):
			all_neighbor_pid.append(cells_id_inner.GetId(pi))
	
	all_neighbor_pid = np.unique(all_neighbor_pid)
	return all_neighbor_pid

def GetBoundaries(vtkdata, label_array, label1, label2, Set_label):
	# Set a label 'Set_label' each time label1 and label2 are connected
	for pid in range(vtkdata.GetNumberOfPoints()):
		if int(label_array.GetTuple(pid)[0]) == label1:
			neighbor_ids = NeighborPoints(vtkdata, pid)
			for nid in neighbor_ids:
				neighbor_label = int(label_array.GetTuple(nid)[0])
				if(neighbor_label == label2):
					label_array.SetTuple(nid, (Set_label, ))
					label_array.SetTuple(pid, (Set_label, ))
	return vtkdata


def RealLabels(vtkdata, label_array):
	# Change the label to the label used for the model training
	vtkdata, label_array= ChangeLabel(vtkdata, label_array, 1, 11)
	vtkdata, label_array= ChangeLabel(vtkdata, label_array, 2, 22)
	vtkdata, label_array= ChangeLabel(vtkdata, label_array, 0, 2)
	vtkdata, label_array= ChangeLabel(vtkdata, label_array, 11, 0)
	vtkdata, label_array= ChangeLabel(vtkdata, label_array, 22, 1)
	return vtkdata


def Post_processing(vtkdata):
	#Remove all the smalls comoponents by setting their label to 
	label_array = vtkdata.GetPointData().GetArray('RegionId')
	vtkdata = GetBoundaries(vtkdata, label_array, 1,2,0)
	teeth = Clip(vtkdata, 1.5, 'RegionId', 'off')

	teeth = Clip(vtkdata, 1.5, 'RegionId', 'off')
	teeth, teeth_label = Connectivity(teeth)
	nb_teeth = CountIDPoint(teeth, teeth_label)
	teeth = GetCurvature(teeth)
	teeth = ChangeSmallCompID(teeth, teeth_label, nb_teeth, 1000, -1)

	vtkdata, label_array= ChangeLabel(vtkdata, label_array, 1, 3)
	gum = Clip(vtkdata, 2.5, 'RegionId', 'off')
	vtkdata, label_array = ChangeLabel(vtkdata, label_array, 3, 1)
	gum, gum_label = Connectivity(gum)
	nb_gum = CountIDPoint(gum, gum_label)
	gum = ChangeSmallCompID(gum, gum_label, nb_gum, 1000, -1)

	bound = Clip(vtkdata, 0.5, 'RegionId', 'on')
	bound, bound_label = Connectivity(bound)
	nb_bound = CountIDPoint(bound, bound_label)
	bound = ChangeSmallCompID(bound, bound_label, nb_bound, 1000, -1)
	vtkdata = LocateLabels(vtkdata, label_array, teeth, teeth_label , -1, -2)
	vtkdata = LocateLabels(vtkdata, label_array, bound, bound_label , -1, -1)
	vtkdata = LocateLabels(vtkdata, label_array, gum, gum_label , -1, -3)

	vtkdata = GetBoundaries(vtkdata, label_array, 1,2,0)

	vtkdata = RegionGrowing(vtkdata, label_array, -1, -1)
	vtkdata = RegionGrowing(vtkdata, label_array, -2, 2)
	vtkdata = RegionGrowing(vtkdata, label_array, -3, 1)
	vtkdata = RealLabels(vtkdata, label_array)
	# Write(vtkdata, 'test.vtk')
	return vtkdata,label_array

def ReadFile(filename):
	inputSurface = filename
	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(inputSurface)
	reader.Update()
	vtkdata = reader.GetOutput()
	label_array = vtkdata.GetPointData().GetArray('RegionId')
	return vtkdata, label_array

def Write(vtkdata, output_name):
	outfilename = output_name
	print("Writting:", outfilename)
	polydatawriter = vtk.vtkPolyDataWriter()
	polydatawriter.SetFileName(outfilename)
	polydatawriter.SetInputData(vtkdata)
	polydatawriter.Write()


def Label_Teeth(vtkdata, label_array):
	vtkdata, label_array= ChangeLabel(vtkdata, label_array, 1, 3)
	predict_teeth = Clip(vtkdata, 2.5, 'RegionId', 'off')
	predict_mesh, predict_labels = ChangeLabel(predict_mesh, predict_labels, 3, 1)

	predict_teeth, predict_teeth_label = Connectivity(predict_teeth)
	nb_predteeth = CountIDPoint(predict_teeth, predict_teeth_label)
	predict_teeth = ChangeSmallCompID(predict_teeth, predict_teeth_label, nb_predteeth, 1000, -1)

	predict_teeth = Clip(predict_teeth, -0.5, 'RegionId', 'off')
	predict_teeth, predict_teeth_label = Connectivity(predict_teeth) 
	for label in range(np.max(np.array(predict_teeth_label))+1):
		predict_mesh = LocateLabels(vtkdata, label_array, predict_teeth, predict_teeth_label, label, label + 3) #Labels have to start at 3
		
	return vtkdata


mesh, mesh_label = ReadFile(arg.mesh)
mesh, mesh_label = Post_processing(mesh)
mesh = Label_Teeth(mesh, mesh_label)
Write(mesh, arg.out)

# python3 post_process.py --mesh /Users/mdumont/Downloads/scan2_test.vtk --out /Users/mdumont/Desktop/DCBIA-projects/Output/scan2_PP.vtk


