import vtk
import numpy as np
import argparse
import sys
import os

# parser = argparse.ArgumentParser()
# parser.add_argument('--mesh', help='Insert mesh path')
# parser.add_argument('--out', help='Insert output path+name')
# arg = parser.parse_args()

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
	connectivityFilter.ScalarConnectivityOn()
	connectivityFilter.SetScalarRange([2,2])
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
	vtkdata, label_array = ChangeLabel(vtkdata, label_array, 3, 1)

	predict_teeth, predict_teeth_label = Connectivity(predict_teeth)
	nb_predteeth = CountIDPoint(predict_teeth, predict_teeth_label)
	predict_teeth = ChangeSmallCompID(predict_teeth, predict_teeth_label, nb_predteeth, 1000, -1)

	predict_teeth = Clip(predict_teeth, -0.5, 'RegionId', 'off')
	predict_teeth, predict_teeth_label = Connectivity(predict_teeth) 
	for label in range(np.max(np.array(predict_teeth_label))+1):
		vtkdata = LocateLabels(vtkdata, label_array, predict_teeth, predict_teeth_label, label, label + 3) #Labels have to start at 3
		
	return vtkdata

def GetAllNeighbors(vtkdata, pids):
	all_neighbors = pids
	for pid in pids:
		neighbors = GetNeighbors(vtkdata, pid)
		all_neighbors = np.concatenate((all_neighbors, neighbors))
	return np.unique(all_neighbors)

def GetNeighbors(vtkdata, pid):
	cells_id = vtk.vtkIdList()
	vtkdata.GetPointCells(pid, cells_id)
	neighbor_pids = []

	for ci in range(cells_id.GetNumberOfIds()):
		points_id_inner = vtk.vtkIdList()
		vtkdata.GetCellPoints(cells_id.GetId(ci), points_id_inner)
		for pi in range(points_id_inner.GetNumberOfIds()):
			pid_inner = points_id_inner.GetId(pi)
			if pid_inner != pid:
				neighbor_pids.append(pid_inner)

	return np.unique(neighbor_pids).tolist()

def GetNeighborIds(vtkdata, pid, labels, label, pid_visited):
	cells_id = vtk.vtkIdList()
	vtkdata.GetPointCells(pid, cells_id)
	neighbor_pids = []

	for ci in range(cells_id.GetNumberOfIds()):
		points_id_inner = vtk.vtkIdList()
		vtkdata.GetCellPoints(cells_id.GetId(ci), points_id_inner)
		for pi in range(points_id_inner.GetNumberOfIds()):
			pid_inner = points_id_inner.GetId(pi)
			if labels.GetTuple(pid_inner)[0] == label and pid_inner != pid and pid_visited[pid_inner] == 0:
				pid_visited[pid_inner] = 1
				neighbor_pids.append(pid_inner)

	return np.unique(neighbor_pids).tolist()

def ConnectedRegion(vtkdata, pid, labels, label, pid_visited):

	neighbor_pids = GetNeighborIds(vtkdata, pid, labels, label, pid_visited)
	all_connected_pids = [pid]
	all_connected_pids.extend(neighbor_pids)

	while len(neighbor_pids):
		npid = neighbor_pids.pop()
		next_neighbor_pids = GetNeighborIds(vtkdata, npid, labels, label, pid_visited)
		neighbor_pids.extend(next_neighbor_pids)
		all_connected_pids = np.append(all_connected_pids, next_neighbor_pids)

	return np.unique(all_connected_pids)

def NeighborLabel(vtkdata, labels, label, connected_pids):
	neighbor_ids = []
	
	for pid in connected_pids:
		cells_id = vtk.vtkIdList()
		vtkdata.GetPointCells(int(pid), cells_id)
		for ci in range(cells_id.GetNumberOfIds()):
			points_id_inner = vtk.vtkIdList()
			vtkdata.GetCellPoints(cells_id.GetId(ci), points_id_inner)
			for pi in range(points_id_inner.GetNumberOfIds()):
				pid_inner = points_id_inner.GetId(pi)
				if labels.GetTuple(pid_inner)[0] != label:
					neighbor_ids.append(pid_inner)

	neighbor_ids = np.unique(neighbor_ids)
	neighbor_labels = []

	for nid in neighbor_ids:
		neighbor_labels.append(labels.GetTuple(nid)[0])
	
	if len(neighbor_labels) > 0:
		return max(neighbor_labels, key=neighbor_labels.count)
	return -1



def RemoveIslands(vtkdata, labels, label, min_count):

	pid_visited = np.zeros(labels.GetNumberOfTuples())
	for pid in range(labels.GetNumberOfTuples()):
		if labels.GetTuple(pid)[0] == label and pid_visited[pid] == 0:
			connected_pids = ConnectedRegion(vtkdata, pid, labels, label, pid_visited)
			if connected_pids.shape[0] < min_count:
				neighbor_label = NeighborLabel(vtkdata, labels, label, connected_pids)
				for cpid in connected_pids:
					labels.SetTuple(int(cpid), (neighbor_label,))

def ConnectivityLabeling(vtkdata, labels, label, start_label):
	pid_visited = np.zeros(labels.GetNumberOfTuples())
	for pid in range(labels.GetNumberOfTuples()):
		if labels.GetTuple(pid)[0] == label and pid_visited[pid] == 0:
			connected_pids = ConnectedRegion(vtkdata, pid, labels, label, pid_visited)
			for cpid in connected_pids:
				labels.SetTuple(int(cpid), (start_label,))
			start_label += 1


def ErodeLabel(vtkdata, labels, label):
	
	pid_labels = []
	for pid in range(labels.GetNumberOfTuples()):
		if labels.GetTuple(pid)[0] == label:
			pid_labels.append(pid)

	while pid_labels:
		pid_labels_remain = pid_labels
		pid_labels = []

		all_neighbor_pids = []
		all_neighbor_labels = []

		while pid_labels_remain:

			pid = pid_labels_remain.pop()

			neighbor_pids = GetNeighbors(vtkdata, pid)
			is_neighbor = False

			for npid in neighbor_pids:
				neighbor_label = labels.GetTuple(npid)[0]
				if neighbor_label != label:
					all_neighbor_pids.append(pid)
					all_neighbor_labels.append(neighbor_label)
					is_neighbor = True
					break

			if not is_neighbor:
				pid_labels.append(pid)

		if(all_neighbor_pids):
			for npid, nlabel in zip(all_neighbor_pids, all_neighbor_labels):
				labels.SetTuple(int(npid), (nlabel,))
		else:
			break

def MeanCoordinatesTeeth(surf,labels):
	nlabels, pid_labels = [], []

	for pid in range(labels.GetNumberOfTuples()):
		nlabels.append(int(labels.GetTuple(pid)[0]))
		pid_labels.append(pid)

	currentlabel = 2
	L = []

	while currentlabel != np.max(nlabels)+1:
		Lcoordinates = []
		for i in range(len(nlabels)):
			if nlabels[i]==currentlabel:
				xyzCoordinates = surf.GetPoint(pid_labels[i])
				Lcoordinates.append(xyzCoordinates)

		meantuple = np.mean(Lcoordinates,axis=0)
		L.append(meantuple)		
		currentlabel+=1			

	return L


def Alignement(surf,surf_GT):
	print('Alignement ')
	# CenterSurf = surf.GetCenter()
	# print("center CenterSurf: ", CenterSurf)
	# print(' ')
	# CenterSurf_GT = surf_GT.GetCenter()
	# print("center CenterSurf_GT: ", CenterSurf_GT)
	# print(' ')
	# direction = np.array(list(surf_GT.GetCenter())) - np.array(list(surf.GetCenter()))
	# print('direction = ', direction)
	# print(' ')

	# trnf = vtk.vtkTransform()
	# trnf.Translate(direction)

	# tpd = vtk.vtkTransformPolyDataFilter()
	# tpd.SetTransform(trnf)
	# tpd.SetInputData(surf)
	# tpd.Update()

	# return tpd.GetOutput()


	icp = vtk.vtkIterativeClosestPointTransform()
	icp.StartByMatchingCentroidsOn()
	icp.SetSource(surf)
	icp.SetTarget(surf_GT)
	icp.GetLandmarkTransform().SetModeToRigidBody()
	icp.SetMaximumNumberOfLandmarks(100)
	icp.SetMaximumMeanDistance(.00001)
	icp.SetMaximumNumberOfIterations(500)
	icp.CheckMeanDistanceOn()
	icp.StartByMatchingCentroidsOn()
	icp.Update()

	lmTransform = icp.GetLandmarkTransform()
	transform = vtk.vtkTransformPolyDataFilter()
	transform.SetInputData(surf)
	transform.SetTransform(lmTransform)
	transform.SetTransform(icp)
	transform.Update()

	return transform.GetOutput()


def UniversalID(surf, labels, LowerOrUpper):	
	real_labels = vtk.vtkIntArray()
	real_labels.SetNumberOfComponents(1)
	real_labels.SetNumberOfTuples(surf.GetNumberOfPoints())
	real_labels.SetName("UniversalID")
	real_labels.Fill(-1)

	for pid in range(labels.GetNumberOfTuples()):
		if not LowerOrUpper: # Lower
			real_labels.SetTuple(pid, (int(labels.GetTuple(pid)[0])+15,))
			
		if LowerOrUpper: # Upper
			real_labels.SetTuple(pid, (int(labels.GetTuple(pid)[0])-1,))
			
	surf.GetPointData().AddArray(real_labels)


def Labelize(surf,labels, Lsurf, Lsurf_GT):
	L_label, L_label_GT = [], []

	# print(' ')
	# print("Lsurf :  || Nbre Elem: ", len(Lsurf))
	# for i in range(len(Lsurf)):
	# 	print(i+2,"  ",Lsurf[i])
	# print(' ')
	# print(' ')

	# print("Lsurf_GT :  || Nbre Elem: ", len(Lsurf_GT))
	# for i in range(len(Lsurf_GT)):
	# 	print(i+2,"  ",Lsurf_GT[i])
	# print(' ')
	# print(' ')


	for j in range(len(Lsurf)):
		Ldist = []
		for i in range (len(Lsurf_GT)):
			Xdist = Lsurf_GT[i][0]-Lsurf[j][0]
			Ydist = Lsurf_GT[i][1]-Lsurf[j][1]
			Zdist = Lsurf_GT[i][2]-Lsurf[j][2]

			dist = np.sqrt(pow(Xdist,2)+pow(Ydist,2)+pow(Zdist,2))		

			if dist<10:
				Ldist.append([dist,i+2,j+2])

		if Ldist:
			minDist = min(Ldist)
			L_label.append(minDist[2])
			L_label_GT.append(minDist[1])	

		# print(Ldist)
		# print(' ')

	L_label_bias = [x+20 for x in L_label]

	# print(L_label)
	# print(L_label_bias)
	# print(L_label_GT)
	# print(' ')


	bias = 0
	for i in range(len(Lsurf)):
		if i+2 not in L_label:
			print(i+2)
			ChangeLabel(surf, labels, i+2, -2)
			bias = 1

	for i in range(len(L_label_GT)):
		ChangeLabel(surf, labels, L_label[i], L_label_bias[i])

	for i in range(len(L_label_GT)):
		if bias:
			ChangeLabel(surf, labels, L_label_bias[i], L_label_GT[i])
		else:
			ChangeLabel(surf, labels, L_label_bias[i], L_label_GT[i])


	# for i in range(len(L_label_GT)):
	# 	if L_label_GT[i]>= 9:
	# 		if bias:
	# 			ChangeLabel(surf, labels, L_label_bias[i], abs(L_label_GT[i]-max(L_label_GT))+2+bias)
	# 		else:
	# 			ChangeLabel(surf, labels, L_label_bias[i], abs(L_label_GT[i]-max(L_label_GT))+2)
	# 	else:
	# 		if bias:
	# 			ChangeLabel(surf, labels, L_label_bias[i], L_label_GT[i]-(min(L_label_GT)-2)+bias)
	# 		else:
	# 			ChangeLabel(surf, labels, L_label_bias[i], L_label_GT[i]-(min(L_label_GT)-2))

	ChangeLabel(surf, labels, -2, 1)


def ReLabel(surf, labels, label, relabel):
	for pid in range(labels.GetNumberOfTuples()):
		if labels.GetTuple(pid)[0] == label:
			labels.SetTuple(pid, (relabel,))

def Threshold(vtkdata, labels, threshold_min, threshold_max):
	
	threshold = vtk.vtkThreshold()
	threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "RegionId")
	threshold.SetInputData(vtkdata)
	threshold.ThresholdBetween(threshold_min,threshold_max)
	threshold.Update()

	geometry = vtk.vtkGeometryFilter()
	geometry.SetInputData(threshold.GetOutput())
	geometry.Update()
	return geometry.GetOutput()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Predict an input with a trained neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--surf', type=str, help='Input surface mesh to label', required=True)
	parser.add_argument('--remove_islands', type=bool, help='Remove islands from mesh by labeling with the closes one', default=False)
	parser.add_argument('--connectivity', type=bool, help='Label all elements with unique labels', default=False)
	parser.add_argument('--connectivity_label', type=int, help='Connectivity label', default=2)
	parser.add_argument('--erode', type=bool, help='Erode label until it dissapears changing it with the neighboring label', default=False)
	parser.add_argument('--erode_label', type=int, help='Eroding label', default=0)
	parser.add_argument('--threshold', type=bool, help='Threshold between two values', default=False)
	parser.add_argument('--threshold_min', type=int, help='Threshold min value', default=2)
	parser.add_argument('--threshold_max', type=int, help='Threshold max value', default=100)
	parser.add_argument('--min_count', type=int, help='Minimum count to remove', default=500)
	
	parser.add_argument('--labelize', type=bool, help='label the teeth', default=False)
	parser.add_argument('--label_groundtruth', type=str, help='groundtruth of the label', default="groundtruth.vtk")
	parser.add_argument('--universalID', type=bool, help='label the teeth with Universal ID', default=False)
	parser.add_argument('--LowerOrUpper', type=int, help='0 == Lower | 1 == Upper', default=0)
	
	parser.add_argument('--out', type=str, help='Output model with labels', default="out.vtk")

	args = parser.parse_args()
	surf, labels = ReadFile(args.surf)

	if(args.remove_islands):
		labels_range = np.zeros(2)
		labels.GetRange(labels_range)
		for label in range(int(labels_range[0]), int(labels_range[1]) + 1):
			print("Removing islands:", label)
			RemoveIslands(surf, labels, label, args.min_count)
	
	if(args.connectivity):
		print("Connectivity...")
		ConnectivityLabeling(surf, labels, args.connectivity_label, 2)

	if(args.erode):
		print("Eroding...")
		ErodeLabel(surf, labels, args.erode_label)

	if(args.threshold):
		print("Thresholding...")
		surf = Threshold(surf, labels, args.threshold_min, args.threshold_max)

	if(args.labelize):
		print("Labelizing...")
		surf_groundtruth, labels_groundtruth = ReadFile(args.label_groundtruth)
		# For now it doesnt work with all cases may be because of the GT
		surf = Alignement(surf,surf_groundtruth)
		Lsurf = MeanCoordinatesTeeth(surf,labels)
		Lsurf_GT = MeanCoordinatesTeeth(surf_groundtruth,labels_groundtruth)
		Labelize(surf,labels,Lsurf,Lsurf_GT)

	if(args.universalID):
		print("UniversalID...")
		UniversalID(surf, labels, args.LowerOrUpper)


	Write(surf, args.out)

# mesh, mesh_label = ReadFile(arg.mesh)
# mesh, mesh_label = Post_processing(mesh)
# mesh = Label_Teeth(mesh, mesh_label)
# Write(mesh, arg.out)

# python3 post_process.py --mesh /Users/mdumont/Downloads/scan2_test.vtk --out /Users/mdumont/Desktop/DCBIA-projects/Output/scan2_PP.vtk


