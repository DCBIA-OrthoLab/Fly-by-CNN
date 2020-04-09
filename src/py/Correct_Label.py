import vtk 
import numpy as np

def NeighborPoints(vtkdata,CurrentID):
	#Find all the CurrentID's neighbors
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

inputSurface = '../Github/fly-by-cnn/src/py/newlabeled.vtk'
reader = vtk.vtkPolyDataReader()
reader.SetFileName(inputSurface)
reader.Update()
original_surf = reader.GetOutput()

real_labels = original_surf.GetPointData().GetArray('unknown')
real_labels.SetName("real_labels")

#Create the list of all the -1 labels
minus1_ids = []
for pid in range(original_surf.GetNumberOfPoints()):
	if int(real_labels.GetTuple(pid)[0]) == -1:
		minus1_ids.append(pid)

#Region growing for unknowed ID (-1)
#Take a look of the neghbor's id and take it if it s different -1
count = 0
while len(minus1_ids) > 0:
	count += 1
	for pid in minus1_ids[:]:
		neighbor_ids = NeighborPoints(original_surf, pid)
		for nid in neighbor_ids:
			neighbor_label = int(real_labels.GetTuple(nid)[0])
			if(neighbor_label != -1):
				real_labels.SetTuple(pid, (neighbor_label,))
				minus1_ids.remove(pid)
				count = 0
				break
	if count == 2:
		#Sometimes the while loop can t find any label != -1 
		print('WARNING :', len(minus1_ids), 'label(s) has been undertermined. Setting to 0.')
		break		

for pid in minus1_ids:
	#Then we set theses label to 0
	real_labels.SetTuple(pid, (0,))

original_surf.GetPointData().AddArray(real_labels)
original_surf.GetPointData().SetActiveScalars('real_labels')

#Clip the teeth 
Clipper = vtk.vtkClipPolyData()
Clipper.SetInputData(original_surf)
Clipper.SetValue(0.5)
Clipper.GenerateClipScalarsOff()
Clipper.InsideOutOff()
Clipper.GetOutput().GetPointData().CopyScalarsOff()
Clipper.Update()
Teeth = Clipper.GetOutput()

#Labelise the teeth and element that aren't connected to the teeth in RegionId
connectivityFilter = vtk.vtkConnectivityFilter()
connectivityFilter.SetInputData(Teeth)
connectivityFilter.SetExtractionModeToAllRegions()
connectivityFilter.ColorRegionsOn()
connectivityFilter.Update()
Teeth = connectivityFilter.GetOutput()


RegionId = Teeth.GetPointData().GetArray('RegionId')
IDRange = RegionId.GetRange()
ID_max = int(IDRange[1])

#Get the number of points of each label
Nb_Pts=[]
number_list = []
for label in range(ID_max+1):
	NumberOfPoints=0
	SameLabeledID=[]
	for i in range(Teeth.GetNumberOfPoints()):
		if int(RegionId.GetTuple(i)[0]) == label:
			NumberOfPoints += 1
	number_list.append(NumberOfPoints)

teeth_label = number_list.index(np.max(number_list))

#Put all the label that aren't the teeth to -1
for pid in range(Teeth.GetNumberOfPoints()):
	current_label = int(RegionId.GetTuple(pid)[0])
	if current_label != teeth_label:
		RegionId.SetTuple(pid, (-1,))

#Set 1 for the teeth 0 for the others
for pid in range(Teeth.GetNumberOfPoints()):
	current_label = int(RegionId.GetTuple(pid)[0])
	if teeth_label != 1:
		if current_label == teeth_label:
			RegionId.SetTuple(pid, (1,))
		else:
			RegionId.SetTuple(pid, (0,))
	else:
		if current_label == -1:
			RegionId.SetTuple(pid, (0,))

Teeth.GetPointData().AddArray(RegionId)

#Clip the gum 
Clipper = vtk.vtkClipPolyData()
Clipper.SetInputData(original_surf)
Clipper.SetValue(0.5)
Clipper.GenerateClipScalarsOff()
Clipper.InsideOutOn()
Clipper.GetOutput().GetPointData().CopyScalarsOff()
Clipper.Update()
gum = Clipper.GetOutput()

#Create the RegionId array
connectivityFilter = vtk.vtkConnectivityFilter()
connectivityFilter.SetInputData(gum)
connectivityFilter.SetExtractionModeToAllRegions()
connectivityFilter.ColorRegionsOn()
connectivityFilter.Update()
gum = connectivityFilter.GetOutput()

RegionId = gum.GetPointData().GetArray('RegionId').Fill(0)
gum.GetPointData().AddArray(RegionId)

#Merge the two polydatas
appendFilter = vtk.vtkAppendPolyData()
appendFilter.AddInputData(Teeth)
appendFilter.AddInputData(gum)
appendFilter.Update()
original_surf = appendFilter.GetOutput()

outfilename = '../Output/PostProcess.vtk'
print("Writting:", outfilename)
polydatawriter = vtk.vtkPolyDataWriter()
polydatawriter.SetFileName(outfilename)
polydatawriter.SetInputData(original_surf)
polydatawriter.Write()


# ren1 = vtk.vtkRenderer()

# colors = vtk.vtkNamedColors()

# renWin = vtk.vtkRenderWindow()
# renWin.AddRenderer(ren1)

# iren = vtk.vtkRenderWindowInteractor()
# iren.SetRenderWindow(renWin)

# mapper = vtk.vtkPolyDataMapper()
# mapper.SetInputData(Teeth)
# mapper.ScalarVisibilityOn()

# letter = vtk.vtkActor()
# letter.SetMapper(mapper)

# ren1.AddActor(letter)
# ren1.SetBackground(colors.GetColor3d("WhiteSmoke"))
# ren1.ResetCamera()
# ren1.GetActiveCamera().Dolly(1.2)
# ren1.ResetCameraClippingRange()
# renWin.SetSize(1080, 960)
# renWin.Render()
# iren.Start()





