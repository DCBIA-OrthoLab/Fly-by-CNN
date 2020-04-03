import subdivision
import vtk
import numpy as np
import nrrd
import time
import LinearSubdivisionFilter as lsf
import itk
import tensorflow as tf

import Run_Model
import Set_Label

# [background gum teeth]
# gum = 0 tooth = 1


start_time = time.time()



def ReadFile(filename):
	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(filename)
	reader.Update()
	vtkdata = reader.GetOutput()
	return vtkdata

def Write(vtkdata,name):
	Writter=vtk.vtkPolyDataWriter()
	Writter.SetFileName(name)
	Writter.SetInputData(vtkdata)
	Writter.Write()


def CreateRenderer():
	ren1 = vtk.vtkRenderer()
	renWin = vtk.vtkRenderWindow()
	renWin.AddRenderer(ren1)
	iren = vtk.vtkRenderWindowInteractor()
	iren.SetRenderWindow(renWin)
	return ren1,renWin,iren

def AddActor(ren1,vtkdata):
	mapper = vtk.vtkPolyDataMapper()
	mapper.SetInputData(vtkdata)
	vtkdataactor = vtk.vtkActor()
	vtkdataactor.SetMapper(mapper)
	ren1.AddActor(vtkdataactor)
	return ren1

def DisplayRenderer(ren1,renWin,iren):
	ren1.SetBackground(1,.5,1)
	ren1.ResetCamera()
	ren1.GetActiveCamera().Dolly(1.2)
	ren1.ResetCameraClippingRange()
	renWin.SetSize(1080, 960)
	renWin.Render()
	iren.Start()
	return 0

def CreateLocator(vtkdata):
	my_locator=vtk.vtkCellLocator()
	my_locator.SetDataSet(vtkdata) 
	my_locator.BuildLocator()
	return my_locator

def Normalisation(vtkdata):
	polypoints = vtkdata.GetPoints()
	nppoints = []
	for pid in range(polypoints.GetNumberOfPoints()):
		spoint = polypoints.GetPoint(pid)
		spoint = np.array(spoint)
		nppoints.append(spoint)
	nppoints=np.array(nppoints)
	Mean=np.mean(nppoints)
	nppoints=nppoints-Mean
	MaxValue=np.max(np.abs(np.reshape(nppoints,-1)))
	nppoints=nppoints/MaxValue
	for pid in range(polypoints.GetNumberOfPoints()):
		vtkdata.GetPoints().SetPoint(pid,nppoints[pid])
	return vtkdata

def normalize_points(poly, radius):
	polypoints = poly.GetPoints()
	for pid in range(polypoints.GetNumberOfPoints()):
		spoint = polypoints.GetPoint(pid)
		spoint = np.array(spoint)
		norm = np.linalg.norm(spoint)
		spoint = spoint/norm * radius
		polypoints.SetPoint(pid, spoint)
	poly.SetPoints(polypoints)
	return poly

def normalize_vector(x):
	return x/np.linalg.norm(x)

def GetNormal(vtkdata):
	normals=vtk.vtkPolyDataNormals()
	normals.SetInputData(vtkdata)
	normals.Update()
	vtkdata=normals.GetOutput()
	normals_array=vtkdata.GetPointData().GetArray('Normals')
	return vtkdata,normals_array

def GetCurvature(vtkdata):
	curve=vtk.vtkCurvatures()
	curve.SetCurvatureTypeToMinimum()
	curve.SetInputData(vtkdata)
	curve.Update()
	vtkdata=curve.GetOutput()
	MinCurv_Array=vtkdata.GetPointData().GetArray('Minimum_Curvature')

	curve_max=vtk.vtkCurvatures()
	curve_max.SetCurvatureTypeToMaximum()
	curve_max.SetInputData(vtkdata)
	curve_max.Update()
	vtkdata=curve_max.GetOutput()

	curve_mean=vtk.vtkCurvatures()
	curve_mean.SetCurvatureTypeToMean()
	curve_mean.SetInputData(vtkdata)
	curve_mean.Update()
	vtkdata=curve_mean.GetOutput()
	
	curve_Gauss=vtk.vtkCurvatures()
	curve_Gauss.SetCurvatureTypeToGaussian()
	curve_Gauss.SetInputData(vtkdata)
	curve_Gauss.Update()
	vtkdata=curve_Gauss.GetOutput()

	return vtkdata,MinCurv_Array

def CreateIcosahedron(radius):
	icosahedronsource = vtk.vtkPlatonicSolidSource()
	icosahedronsource.SetSolidTypeToIcosahedron()
	icosahedronsource.Update()
	icosahedron = icosahedronsource.GetOutput()

	sl = 8
	# subdivfilter = vtk.vtkLinearSubdivisionFilter()
	subdivfilter = lsf.LinearSubdivisionFilter()
	subdivfilter.SetInputData(icosahedron)
	subdivfilter.SetNumberOfSubdivisions(sl)
	subdivfilter.Update()

	icosahedron = subdivfilter.GetOutput()
	icosahedron = normalize_points(icosahedron, radius)

	return icosahedron


# def CreatePlan(point,normal,Resolution, sphere_radius, PixelSpacing):
def CreatePlan(Origin,Point1,Point2,Resolution,Sphere_Point,PixelSpacing,sphere_point_delta_v):
	Plan = vtk.vtkPlaneSource()
	# Plan.SetCenter(point)
	
	# Plan.SetNormal(normal[0:3])
	Plan.SetOrigin(Origin)
	Plan.SetPoint1(Point1)
	Plan.SetPoint2(Point2)
	Plan.SetXResolution(Resolution)
	Plan.SetYResolution(Resolution)
	Plan.Update()
	poly = Plan.GetOutput()

	for i in range(poly.GetNumberOfPoints()):
		p = np.array(poly.GetPoint(i))*PixelSpacing + sphere_point_delta_v
		poly.GetPoints().SetPoint(i, p)

	return poly,Plan

# PicturePointId = 10
filename='../data/Intraoral_scanner_meshes/P1_STL.vtk'

# filename='ManuSeg_Scan1.vtk'

# Load NN model
saved_model_path = 'Model_Folder'

vtkdata=ReadFile(filename)

vtkdata=Normalisation(vtkdata)

vtkdata,Normals_vtkdata=GetNormal(vtkdata)
vtkdata,MinCurv_vtkdata=GetCurvature(vtkdata)

LabelArray=vtkdata.GetPointData().GetArray('RegionId')
label_array = vtk.vtkDoubleArray()
label_array.SetNumberOfComponents(3)
label_array.SetNumberOfTuples(vtkdata.GetNumberOfPoints())
# print(vtkdata.GetNumberOfPoints())
label_array.Fill(-1)
sphere_radius = 1.1
icosahedron=CreateIcosahedron(sphere_radius)

	#Create the tangent plan to a point of the icosahedron

ren1,renWin,iren=CreateRenderer()
ren1=AddActor(ren1,vtkdata)
ren1=AddActor(ren1,icosahedron)

tree=CreateLocator(vtkdata)

with tf.Session() as sess:
	loaded = tf.saved_model.load(sess=sess, tags=[tf.saved_model.SERVING], export_dir=saved_model_path)

	for PicturePointId in range(400,401):
		outfilename = '/Users/mdumont/Desktop/DCBIA-projects/data/snap_shots/Model_Output_Features'+str(PicturePointId)+'.nrrd'

		CurrentNormal = icosahedron.GetPoint(PicturePointId)
		# print(CurrentNormal)
		CurrentNormal = -1*np.array(CurrentNormal)
		# print(CurrentNormal)

		Resolution=512 #512
		PixelSpacing = 1


		Sphere_Point = icosahedron.GetPoint(PicturePointId)
		Sphere_Point_v = np.array(Sphere_Point) ######################
		sphere_point_delta_v = Sphere_Point_v - Sphere_Point_v*PixelSpacing;

		Sphere_Point_Normal_v = normalize_vector(Sphere_Point_v)

		Sphere_north_v = np.array([0,0,1])
		Sphere_south_v = np.array([0,0,-1])

		plane_orient_x_v = np.array([0,0,0])
		plane_orient_y_v = np.array([0,0,0])

		if (np.array_equal(Sphere_Point_Normal_v,Sphere_north_v) or np.array_equal(Sphere_Point_Normal_v,Sphere_south_v)) :
			plane_orient_x_v[0] = 1
			plane_orient_y_v[1] = 1
		else :
			plane_orient_x_v = normalize_vector(np.cross(Sphere_Point_Normal_v,Sphere_north_v))
			plane_orient_y_v = normalize_vector(np.cross(Sphere_Point_Normal_v,plane_orient_x_v))

		Plane_point_origin_v = np.subtract(Sphere_Point_v,plane_orient_x_v*0.5)
		Plane_point_origin_v = np.subtract(Plane_point_origin_v,plane_orient_y_v*0.5)
		Plane_point_1_v = np.add(Plane_point_origin_v,plane_orient_x_v)
		Plane_point_2_v = np.add(Plane_point_origin_v,plane_orient_y_v)


		planPOLY,plan = CreatePlan(Plane_point_origin_v, Plane_point_1_v, Plane_point_2_v, Resolution-1, Sphere_Point, PixelSpacing, sphere_point_delta_v)
		ren1 = AddActor(ren1, planPOLY)


		######### Cell Locator


		planPoints = planPOLY.GetPoints()

		NumFeatures = 4
		features = np.ones([Resolution*Resolution,NumFeatures])*-1
		label = np.zeros([Resolution*Resolution])
		tol = 1.e-8
		t = vtk.mutable(0)
		x = [0,0,0]
		pcoords = [0,0,0]
		subId = vtk.mutable(0)

		# 1. Create an image
		pointid_array = np.ones([Resolution*Resolution])*-1

		for i in range(planPoints.GetNumberOfPoints()):
			point_plane = np.array(planPoints.GetPoint(i))
			point_target = point_plane + 2*(sphere_radius)*CurrentNormal
			cellId=vtk.mutable(-1)
			code = tree.IntersectWithLine(point_plane, point_target, tol, t, x, pcoords, subId, cellId)
			if(code):
				cellPointsId = vtk.vtkIdList()
				vtkdata.GetCellPoints(cellId,cellPointsId)
				
				pointId = cellPointsId.GetId(0)
				pointid_array[i] = pointId
				point_surface = vtkdata.GetPoint(pointId)
				point_normal = np.array(Normals_vtkdata.GetTuple(pointId))
				
				features[i][0] = np.linalg.norm(point_plane - point_surface)
				features[i][1:4] = point_normal[0:3]

		# 2. Run through model
		# print('max id list : ', max(pointid_array), '\n mesh max id : ', vtkdata.GetNumberOfPoints(), '\n label array size : ', label_array.GetNumberOfTuples())
		features = np.reshape(features, [Resolution, Resolution, NumFeatures])
		PixelDimension = NumFeatures
		Dimension = 2

		ComponentType = itk.ctype('float')
		OutputImageType = itk.VectorImage[ComponentType, Dimension]

		out_img = OutputImageType.New()
		out_img.SetNumberOfComponentsPerPixel(PixelDimension)
		  
		size = itk.Size[Dimension]()
		size.Fill(1)
		features_shape = list(features.shape[0:-1])
		features_shape.reverse()

		for i, s in enumerate(features_shape):
		  size[i] = s

		index = itk.Index[Dimension]()
		index.Fill(0)

		RegionType = itk.ImageRegion[Dimension]
		region = RegionType()
		region.SetIndex(index)
		region.SetSize(size)

		out_img.SetRegions(region)
		out_img.SetOrigin([0, 0])
		out_img.SetSpacing([PixelSpacing, PixelSpacing])
		out_img.Allocate()

		out_img_np = itk.GetArrayViewFromImage(out_img)
		out_img_np.setfield(np.reshape(features, out_img_np.shape), out_img_np.dtype)

		model_output = Run_Model.RUN(out_img, loaded, sess)


		print("File Number :", outfilename)
		writer = itk.ImageFileWriter.New(FileName=outfilename, Input=model_output)
		writer.UseCompressionOn()
		writer.Update()



		# 3. Set labels
		# print(model_output[1], '\n')
		model_output_array = itk.GetArrayFromImage(model_output)
		model_output_array = np.reshape(model_output_array, [Resolution*Resolution, model_output_array.shape[-1]])



		label_array = Set_Label.Label_Set(vtkdata, model_output_array, pointid_array, label_array)
	real_label = Set_Label.Set_Real_Label(label_array)
	vtkdata.GetPointData().AddArray(real_label)

	print("Writing:", 'newlabeled.vtk')
	Write(vtkdata,'newlabeled.vtk')

	














# print(features.reshape([Resolution, Resolution, NumFeatures]))

# filename = '/Users/mdumont/Desktop/DCBIA-projects/data/snap_shots/Features'+str(PicturePointId)+'.nrrd'
# nrrd.write(filename, features.reshape([Resolution, Resolution, NumFeatures]))
# filename = '/Users/mdumont/Desktop/DCBIA-projects/data/snap_shots/Label'+str(PicturePointId)+'.nrrd'
# nrrd.write(filename, label.reshape([Resolution, Resolution, 1]))





# Picker=vtk.vtkCellPicker()	
# PlanFeatures=[]	
# for PlanPoint in range(100):
# 	Picker.Pick3DRay(planPOLY.GetPoint(PlanPoint),CurrentNormal,ren1)
# 	CellPickedID=Picker.GetCellId()
# 	PointsOnCell=vtk.vtkIdList()
# 	vtkdata.GetCellPoints(CellPickedID,PointsOnCell)
# 	CellFeatures=[]
# 	for i in range(3):
# 		CurrentID=PointsOnCell.GetId(i)
# 		CurrentTuple=(MinCurv_vtkdata.GetTuple(CurrentID))
# 		CurrentTuple+=(Normals_vtkdata.GetTuple(CurrentID)[0],Normals_vtkdata.GetTuple(CurrentID)[1],Normals_vtkdata.GetTuple(CurrentID)[2])
# 		CellFeatures.append(CurrentTuple)
# 	PlanFeatures.append(CellFeatures)
# print(PlanFeatures)


# print(PointsOnCell)
# DisplayRenderer(ren1,renWin,iren)
# vtkdata.GetPointData().SetActiveScalars('NormalizedPoint')

# colors = vtk.vtkNamedColors()

# Now create the RenderWindow, Renderer and Interactor.
#




# print("Maximum : ",maxX,maxY,maxZ,"\nMinimum : ", minX,minY,minZ,"\nMean : ",meanX,meanY,meanZ)