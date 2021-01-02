import vtk
import LinearSubdivisionFilter as lsf
import numpy as np
import math 
import os
import sys
import itk
from readers import OFFReader

from multiprocessing import Pool, cpu_count
from vtk.util.numpy_support import vtk_to_numpy

def Normalization(vtkdata):
	polypoints = vtkdata.GetPoints()
	
	nppoints = []
	for pid in range(polypoints.GetNumberOfPoints()):
		spoint = polypoints.GetPoint(pid)
		nppoints.append(spoint)

	npmean = np.mean(np.array(nppoints), axis=0)
	nppoints -= npmean
	npscale = np.max([np.linalg.norm(p) for p in nppoints])
	nppoints /= npscale

	for pid in range(polypoints.GetNumberOfPoints()):
		vtkdata.GetPoints().SetPoint(pid, nppoints[pid])

	return vtkdata, npmean, npscale

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

def CreateSpiral(sphereRadius=4, numberOfSpiralSamples=64, numberOfSpiralTurns=4):
	
    sphere = vtk.vtkPolyData()
    sphere_points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    vertices = vtk.vtkCellArray()

    c = 2.0*float(numberOfSpiralTurns)
    prevPid = -1

    for i in range(numberOfSpiralSamples):
      p = [0, 0, 0]
      #angle = i * 180.0/numberOfSpiralSamples * math.pi/180.0
      angle = (i*math.pi)/numberOfSpiralSamples
      p[0] = sphereRadius * math.sin(angle)*math.cos(c*angle)
      p[1] = sphereRadius * math.sin(angle)*math.sin(c*angle)
      p[2] = sphereRadius * math.cos(angle)

      pid = sphere_points.InsertNextPoint(p)
      
      if(prevPid != -1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, prevPid)
        line.GetPointIds().SetId(1, pid)
        lines.InsertNextCell(line)

      prevPid = pid

      vertex = vtk.vtkVertex()
      vertex.GetPointIds().SetId(0, pid)

      vertices.InsertNextCell(vertex)
    
    sphere.SetVerts(vertices)
    sphere.SetLines(lines)
    sphere.SetPoints(sphere_points)

    return sphere

def CreatePlane(Origin,Point1,Point2,Resolution):
	plane = vtk.vtkPlaneSource()
	
	plane.SetOrigin(Origin)
	plane.SetPoint1(Point1)
	plane.SetPoint2(Point2)
	plane.SetXResolution(Resolution)
	plane.SetYResolution(Resolution)
	plane.Update()
	return plane.GetOutput()

def ReadSurf(fileName):

	print("Reading:", fileName)

	fname, extension = os.path.splitext(fileName)
	extension = extension.lower()
	if extension == ".vtk":
		reader = vtk.vtkPolyDataReader()
		reader.SetFileName(fileName)
		reader.Update()
		surf = reader.GetOutput()
	elif extension == ".stl":
		reader = vtk.vtkSTLReader()
		reader.SetFileName(fileName)
		reader.Update()
		surf = reader.GetOutput()
	elif extension == ".off":
		reader = OFFReader()
		reader.SetFileName(fileName)
		reader.Update()
		surf = reader.GetOutput()
	elif extension == ".obj":
		if os.path.exists(fname + ".mtl"):
			obj_import = vtk.vtkOBJImporter()
			obj_import.SetFileName(fileName)
			obj_import.SetFileNameMTL(fname + ".mtl")
			textures_path = os.path.normpath(os.path.dirname(fname) + "/../images")
			if os.path.exists(textures_path):
				obj_import.SetTexturePath(textures_path)
			obj_import.Read()

			actors = obj_import.GetRenderer().GetActors()
			actors.InitTraversal()
			append = vtk.vtkAppendPolyData()

			for i in range(actors.GetNumberOfItems()):
				surfActor = actors.GetNextActor()
				append.AddInputData(surfActor.GetMapper().GetInputAsDataSet())
			
			append.Update()
			surf = append.GetOutput()
			
		else:
			reader = vtk.vtkOBJReader()
			reader.SetFileName(fileName)
			reader.Update()
			surf = reader.GetOutput()

	return surf

def GetActor(surf):
	surfMapper = vtk.vtkPolyDataMapper()
	surfMapper.SetInputData(surf)

	surfActor = vtk.vtkActor()
	surfActor.SetMapper(surfMapper)

	return surfActor

def RotateSurf(surf, rotationAngle, rotationVector):
	print("angle:", rotationAngle, "vector:", rotationVector)
	transform = vtk.vtkTransform()
	transform.RotateWXYZ(rotationAngle, rotationVector[0], rotationVector[1], rotationVector[2]);

	transformFilter = vtk.vtkTransformPolyDataFilter()
	transformFilter.SetTransform(transform)
	transformFilter.SetInputData(surf)
	transformFilter.Update()
	return transformFilter.GetOutput()

def RandomRotation(surf):
	rotationVector = np.random.random(3)*2.0 - 1.0
	rotationVector = rotationVector/np.linalg.norm(rotationVector)
	rotationAngle = np.random.random()*360.0
	return RotateSurf(surf, rotationAngle, rotationVector)

def GetUnitSurf(surf):
	surf, surf_mean, surf_scale = Normalization(surf)
	return surf

def GetNormalsActor(surf):

	try:

		normals = vtk.vtkPolyDataNormals()
		normals.SetInputData(surf);
		normals.ComputeCellNormalsOff();
		normals.ComputePointNormalsOn();
		normals.SplittingOff();
		normals.Update()
		surf = normals.GetOutput()

		# mapper
		surf_actor = GetActor(surf)

		sp = surf_actor.GetShaderProperty();
		sp.AddVertexShaderReplacement(
			"//VTK::Normal::Dec",
			True,
			"//VTK::Normal::Dec\n" + 
			"  varying vec3 myNormalMCVSOutput;\n",
			False
		)

		sp.AddVertexShaderReplacement(
			"//VTK::Normal::Impl",
			True,
			"//VTK::Normal::Impl\n" +
			"  myNormalMCVSOutput = normalMC;\n",
			False
		)

		sp.AddVertexShaderReplacement(
			"//VTK::Color::Impl",
			True, "VTK::Color::Impl\n", False)

		sp.ClearVertexShaderReplacement("//VTK::Color::Impl", True)

		sp.AddFragmentShaderReplacement(
			"//VTK::Normal::Dec",
			True,
			"//VTK::Normal::Dec\n" + 
			"  varying vec3 myNormalMCVSOutput;\n",
			False
		)

		sp.AddFragmentShaderReplacement(
			"//VTK::Light::Impl",
			True,
			"//VTK::Light::Impl\n" +
			"  gl_FragData[0] = vec4(myNormalMCVSOutput*0.5f + 0.5, 1.0);\n",
			False
		)

		return surf_actor
	except Exception as e:
		print(e, file=sys.stderr)
		return None

def GetCellIdMapActor(surf):

	colored_points = vtk.vtkUnsignedCharArray()
	colored_points.SetName('cell_ids')
	colored_points.SetNumberOfComponents(3)

	for cell_id in range(0, surf.GetNumberOfCells()):
		r = cell_id % 255.0 + 1
		g = int(cell_id / 255.0) % 255.0
		b = int(int(cell_id / 255.0) / 255.0) % 255.0
		colored_points.InsertNextTuple3(r, g, b)

		# cell_id_color = int(b*255*255 + g*255 + r - 1)

	surf.GetCellData().SetScalars(colored_points)

	surf_actor = GetActor(surf)
	surf_actor.GetMapper().SetScalarModeToUseCellData()
	surf_actor.GetProperty().LightingOff()
	surf_actor.GetProperty().ShadingOff()
	surf_actor.GetProperty().SetInterpolationToFlat()

	return surf_actor

def GetPointIdMapActor(surf):

	colored_points = vtk.vtkUnsignedCharArray()
	colored_points.SetName('point_ids')
	colored_points.SetNumberOfComponents(3)

	for cell_id in range(0, surf.GetNumberOfCells()):

		point_ids = vtk.vtkIdList()
		surf.GetCellPoints(cell_id, point_ids)

		point_id = point_ids.GetId(0)

		r = point_id % 255.0 + 1
		g = int(point_id / 255.0) % 255.0
		b = int(int(point_id / 255.0) / 255.0) % 255.0
		colored_points.InsertNextTuple3(r, g, b)

		# cell_id_color = int(b*255*255 + g*255 + r - 1)

	surf.GetCellData().SetScalars(colored_points)

	surf_actor = GetActor(surf)
	surf_actor.GetMapper().SetScalarModeToUseCellData()
	surf_actor.GetProperty().LightingOff()
	surf_actor.GetProperty().ShadingOff()
	surf_actor.GetProperty().SetInterpolationToFlat()

	return surf_actor

# def ExtractPointFeatures(surf, cellids_rgb, point_features_name):

# 	point_features_np = []
	
# 	cellids_rgb_shape = cellids_rgb.shape
# 	cellids_rgb = cellids_rgb.reshape(-1, 3)

# 	point_features = surf.GetPointData().GetScalars(point_features_name)
# 	zero = np.zeros(point_features.GetNumberOfComponents())

# 	for cell_id in cellids_rgb:
# 		r = cell_id[0]
# 		g = cell_id[1]
# 		b = cell_id[2]

# 		cell_id_color = int(b*255*255 + g*255 + r - 1)

# 		if cell_id_color >= 0 and cell_id_color < surf.GetNumberOfCells():
# 			point_ids = vtk.vtkIdList()
# 			surf.GetCellPoints(cell_id_color, point_ids)
# 			point_id = point_ids.GetId(0)
# 			point_features_np.append(point_features.GetTuple(point_id))
# 		else:
# 			point_features_np.append(zero)

# 	return np.array(point_features_np).reshape(cellids_rgb_shape[0:-1] + (point_features.GetNumberOfComponents(),))

class ExtractPointFeaturesClass():
	def __init__(self, point_features_np, zero):
		self.point_features_np = point_features_np
		self.zero = zero

	def __call__(self, point_ids_rgb):

		point_ids_rgb = point_ids_rgb.reshape(-1, 3)
		point_features = []

		for point_id_rgb in point_ids_rgb:
			r = point_id_rgb[0]
			g = point_id_rgb[1]
			b = point_id_rgb[2]

			point_id = int(b*255*255 + g*255 + r - 1)

			point_features_np_shape = np.shape(self.point_features_np)
			if point_id >= 0 and point_id < point_features_np_shape[0]:
				point_features.append(self.point_features_np[point_id])
			else:
				point_features.append(self.zero)

		return point_features

def ExtractPointFeatures(surf, point_ids_rgb, point_features_name):
	
	point_ids_rgb_shape = point_ids_rgb.shape

	point_features = surf.GetPointData().GetScalars(point_features_name)
	point_features_np = vtk_to_numpy(point_features)
	zero = np.zeros(point_features.GetNumberOfComponents())

	with Pool(cpu_count()) as p:
		feat = p.map(ExtractPointFeaturesClass(point_features_np, zero), point_ids_rgb)
	return np.array(feat).reshape(point_ids_rgb_shape[0:-1] + (point_features.GetNumberOfComponents(),))

def ReadImage(fName, image_dimension=2, pixel_dimension=-1):
	if(image_dimension == 1):
		if(pixel_dimension != -1):
			ImageType = itk.Image[itk.Vector[itk.F, pixel_dimension], 2]
		else:
			ImageType = itk.VectorImage[itk.F, 2]
	else:
		if(pixel_dimension != -1):
			ImageType = itk.Image[itk.Vector[itk.F, pixel_dimension], image_dimension]
		else:
			ImageType = itk.VectorImage[itk.F, image_dimension]

	img_read = itk.ImageFileReader[ImageType].New(FileName=fName)
	img_read.Update()
	img = img_read.GetOutput()

	return img

def GetImage(img_np):
	img_np_shape = np.shape(img_np)
	ComponentType = itk.ctype('float')

	Dimension = img_np.ndim - 1
	PixelDimension = img_np.shape[-1]
	print("Dimension:", Dimension, "PixelDimension:", PixelDimension)

	if Dimension == 1:
		OutputImageType = itk.VectorImage[ComponentType, 2]
	else:
		OutputImageType = itk.VectorImage[ComponentType, Dimension]
	
	out_img = OutputImageType.New()
	out_img.SetNumberOfComponentsPerPixel(PixelDimension)

	size = itk.Size[OutputImageType.GetImageDimension()]()
	size.Fill(1)
	
	prediction_shape = list(img_np.shape[0:-1])
	prediction_shape.reverse()

	if Dimension == 1:
		size[1] = prediction_shape[0]
	else:
		for i, s in enumerate(prediction_shape):
			size[i] = s

	index = itk.Index[OutputImageType.GetImageDimension()]()
	index.Fill(0)

	RegionType = itk.ImageRegion[OutputImageType.GetImageDimension()]
	region = RegionType()
	region.SetIndex(index)
	region.SetSize(size)

	out_img.SetRegions(region)
	out_img.Allocate()

	out_img_np = itk.GetArrayViewFromImage(out_img)
	out_img_np.setfield(img_np.reshape(out_img_np.shape), out_img_np.dtype)

	return out_img