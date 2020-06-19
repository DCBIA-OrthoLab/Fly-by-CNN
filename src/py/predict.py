
import vtk
import numpy as np
import time
import LinearSubdivisionFilter as lsf
import itk
import tensorflow as tf
import argparse
import os
import post_process

def Normalisation(vtkdata):
	polypoints = vtkdata.GetPoints()
	
	nppoints = []
	for pid in range(polypoints.GetNumberOfPoints()):
		spoint = polypoints.GetPoint(pid)
		nppoints.append(spoint)
	nppoints=np.array(nppoints)

	nppoints -= np.mean(nppoints)
	nppoints /= np.max(np.abs(np.reshape(nppoints,-1)))

	for pid in range(polypoints.GetNumberOfPoints()):
		vtkdata.GetPoints().SetPoint(pid, nppoints[pid])

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

def CreateIcosahedron(radius, sl):
	icosahedronsource = vtk.vtkPlatonicSolidSource()
	icosahedronsource.SetSolidTypeToIcosahedron()
	icosahedronsource.Update()
	icosahedron = icosahedronsource.GetOutput()
	
	subdivfilter = lsf.LinearSubdivisionFilter()
	subdivfilter.SetInputData(icosahedron)
	subdivfilter.SetNumberOfSubdivisions(sl)
	subdivfilter.Update()

	icosahedron = subdivfilter.GetOutput()
	icosahedron = normalize_points(icosahedron, radius)

	return icosahedron

def CreatePlane(Origin,Point1,Point2,Resolution):
	plane = vtk.vtkPlaneSource()
	
	plane.SetOrigin(Origin)
	plane.SetPoint1(Point1)
	plane.SetPoint2(Point2)
	plane.SetXResolution(Resolution)
	plane.SetYResolution(Resolution)
	plane.Update()
	return plane.GetOutput()

start_time = time.time()

parser = argparse.ArgumentParser(description='Predict an input with a trained neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--surf', type=str, help='Input surface mesh to label', required=True)
parser.add_argument('--model', type=str, help='Directory with saved model', required=True)
parser.add_argument('--numberOfSubdivisions', type=int, help='Number of subdivisions for the icosahedron', default=10)
parser.add_argument('--sphereRadius', type=float, help='Radius of the surrounding sphere', default=1.1)
parser.add_argument('--planeResolution', type=int, help='Radius of the surrounding sphere', default=512)
parser.add_argument('--planeSpacing', type=float, help='Spacing of the plane', default=1.0)
parser.add_argument('--out', type=str, help='Output model with labels', default="out.vtk")

args = parser.parse_args()

inputSurface = args.surf
planeSpacing = args.planeSpacing
planeResolution = args.planeResolution
savedModelPath = args.model
sphereRadius = args.sphereRadius
numberOfSubdivisions = args.numberOfSubdivisions
outfilename = args.out

print("planeSpacing", planeSpacing)
print("planeResolution", planeResolution)
print("savedModelPath", savedModelPath)
print("sphereRadius", sphereRadius)
print("numberOfSubdivisions", numberOfSubdivisions)
print("outfilename", outfilename)

print("Reading:", inputSurface)
reader = vtk.vtkPolyDataReader()
reader.SetFileName(inputSurface)
reader.Update()
original_surf = reader.GetOutput()

print('Surf points : ', original_surf.GetNumberOfPoints())
surf = Normalisation(original_surf)

print('Surf points : ', surf.GetNumberOfPoints())
normals = vtk.vtkPolyDataNormals()
normals.SetInputData(surf)
normals.Update()
surf = normals.GetOutput()

tree = vtk.vtkCellLocator()
tree.SetDataSet(surf) 
tree.BuildLocator()

label_array = np.zeros([surf.GetNumberOfPoints(), 3])
print('Surf points : ', surf.GetNumberOfPoints())
icosahedron = CreateIcosahedron(sphereRadius, numberOfSubdivisions)

with tf.Session() as sess:
	loaded = tf.saved_model.load(sess=sess, tags=[tf.saved_model.SERVING], export_dir=savedModelPath)

	print("Total number of points:", icosahedron.GetNumberOfPoints())
	for icoid in range(icosahedron.GetNumberOfPoints()):

		print("id:", icoid)
		sphere_point = icosahedron.GetPoint(icoid)
		sphere_point_v = np.array(sphere_point) ######################
		sphere_point_delta_v = sphere_point_v - sphere_point_v*planeSpacing;
		sphere_point_normal_v = normalize_vector(sphere_point_v)

		sphere_north_v = np.array([0,0,1])
		sphere_south_v = np.array([0,0,-1])

		plane_orient_x_v = np.array([0,0,0])
		plane_orient_y_v = np.array([0,0,0])

		if (np.array_equal(sphere_point_normal_v, sphere_north_v) or np.array_equal(sphere_point_normal_v, sphere_south_v)) :
			plane_orient_x_v[0] = 1
			plane_orient_y_v[1] = 1
		else :
			plane_orient_x_v = normalize_vector(np.cross(sphere_point_normal_v, sphere_north_v))
			plane_orient_y_v = normalize_vector(np.cross(sphere_point_normal_v, plane_orient_x_v))

		plane_point_origin_v = sphere_point_v - plane_orient_x_v*0.5 - plane_orient_y_v*0.5
		plane_point_1_v = plane_point_origin_v + plane_orient_x_v
		plane_point_2_v = plane_point_origin_v + plane_orient_y_v

		plane = CreatePlane(plane_point_origin_v, plane_point_1_v, plane_point_2_v, planeResolution-1)
		planePoints = plane.GetPoints()

		numFeatures = 4
		features = np.ones([planeResolution*planeResolution, numFeatures])*-1
		label = np.zeros([planeResolution*planeResolution])
		tol = 1.e-8
		t = vtk.mutable(0)
		x = [0,0,0]
		pcoords = [0,0,0]
		subId = vtk.mutable(0)

		# 1. Create an image
		pointid_array = np.ones([planeResolution*planeResolution], dtype=int)*-1

		for pid in range(planePoints.GetNumberOfPoints()):
			point_plane = np.array(planePoints.GetPoint(pid))*planeSpacing + sphere_point_delta_v
			point_target = point_plane - 2*(sphereRadius)*sphere_point_normal_v
			cellId=vtk.mutable(-1)
			
			code = tree.IntersectWithLine(point_plane, point_target, tol, t, x, pcoords, subId, cellId)
			if(code):
				cellPointsId = vtk.vtkIdList()
				surf.GetCellPoints(cellId, cellPointsId)
				
				pointId = cellPointsId.GetId(0)
				pointid_array[pid] = pointId
				point_surface = surf.GetPoint(pointId)
				point_normal = np.array(surf.GetPointData().GetArray("Normals").GetTuple(pointId))
				
				features[pid][0:3] = point_normal[0:3]
				features[pid][3] = np.linalg.norm(point_plane - point_surface)

		# 2. Run through model
		# print('max id list : ', max(pointid_array), '\n mesh max id : ', vtkdata.GetNumberOfPoints(), '\n label array size : ', label_array.GetNumberOfTuples())
		features = np.reshape(features, [planeResolution, planeResolution, numFeatures])

		prediction = sess.run(
			'output_y:0',
			feed_dict={
				'input_x:0': np.reshape(features, (1,) + features.shape)
			}
		)
		print(np.array(label_array).shape)
		prediction = np.reshape(np.array(prediction[0]), [planeResolution*planeResolution])
		print(np.array(prediction[0]))
		for index in range(planeResolution*planeResolution):
			pointId = pointid_array[index]
			if(pointId != -1):
				label_array[pointId][int(prediction[index])] += 1


	real_labels = vtk.vtkIntArray()
	real_labels.SetNumberOfComponents(1)
	real_labels.SetNumberOfTuples(surf.GetNumberOfPoints())
	real_labels.SetName("RegionId")
	real_labels.Fill(0)

	for pointId,labels in enumerate(label_array):
		label = np.argmax(labels)
		real_labels.SetTuple(pointId, (label,))

	original_surf.GetPointData().AddArray(real_labels)

	original_surf, surf_label = post_process.Post_processing(original_surf)
	original_surf = post_process.Label_Teeth(original_surf, surf_label)

	print("Writting:", outfilename)
	polydatawriter = vtk.vtkPolyDataWriter()
	polydatawriter.SetFileName(outfilename)
	polydatawriter.SetInputData(original_surf)
	polydatawriter.Write()

		# Dimension = 2

		# ComponentType = itk.ctype('float')
		# OutputImageType = itk.VectorImage[ComponentType, Dimension]

		# out_img = OutputImageType.New()
		# out_img.SetNumberOfComponentsPerPixel(numFeatures)
		  
		# size = itk.Size[Dimension]()
		# size.Fill(1)
		# features_shape = list(features.shape[0:-1])
		# features_shape.reverse()

		# for i, s in enumerate(features_shape):
		#   size[i] = s

		# index = itk.Index[Dimension]()
		# index.Fill(0)

		# RegionType = itk.ImageRegion[Dimension]
		# region = RegionType()
		# region.SetIndex(index)
		# region.SetSize(size)

		# out_img.SetRegions(region)
		# out_img.SetOrigin([0, 0])
		# out_img.SetSpacing([planeSpacing, planeSpacing])
		# out_img.Allocate()

		# out_img_np = itk.GetArrayViewFromImage(out_img)
		# out_img_np.setfield(np.reshape(features, out_img_np.shape), out_img_np.dtype)

		# # model_output = Run_Model.RUN(out_img, loaded, sess)


		# outfilename = os.path.join(args.out, str(pi) + ".nrrd")
		# print("File Number :", outfilename)
		# writer = itk.ImageFileWriter.New(FileName=outfilename, Input=out_img)
		# writer.UseCompressionOn()
		# writer.Update()

		# ren1 = vtk.vtkRenderer()
		# renWin = vtk.vtkRenderWindow()
		# renWin.AddRenderer(ren1)
		# iren = vtk.vtkRenderWindowInteractor()
		# iren.SetRenderWindow(renWin)

		# surfmapper = vtk.vtkPolyDataMapper()
		# surfmapper.SetInputData(surf)
		# surfactor = vtk.vtkActor()
		# surfactor.SetMapper(surfmapper)
		# ren1.AddActor(surfactor)

		# icosahedronmapper = vtk.vtkPolyDataMapper()
		# icosahedronmapper.SetInputData(icosahedron)
		# icosahedronactor = vtk.vtkActor()
		# icosahedronactor.SetMapper(icosahedronmapper)
		# ren1.AddActor(icosahedronactor)

		# planemapper = vtk.vtkPolyDataMapper()
		# planemapper.SetInputData(plane)
		# planeactor = vtk.vtkActor()
		# planeactor.SetMapper(planemapper)
		# ren1.AddActor(planeactor)
		
		# ren1.SetBackground(0,.5,1)
		# ren1.ResetCamera()
		# ren1.GetActiveCamera().Dolly(1.2)
		# ren1.ResetCameraClippingRange()
		# renWin.SetSize(1080, 960)
		# renWin.Render()
		# iren.Start()

		# 3. Set labels
		# print(model_output[1], '\n')
		# model_output_array = itk.GetArrayFromImage(model_output)
		# model_output_array = np.reshape(model_output_array, [Resolution*Resolution, model_output_array.shape[-1]])



		# label_array = Set_Label.Label_Set(vtkdata, model_output_array, pointid_array, label_array)
	# real_label = Set_Label.Set_Real_Label(label_array)
	# vtkdata.GetPointData().AddArray(real_label)

	# print("Writing:", 'newlabeled.vtk')
	# Write(vtkdata,'newlabeled.vtk')

