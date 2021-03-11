import vtk
import LinearSubdivisionFilter as lsf
import numpy as np
import math 
import os
import sys
import itk
import nibabel as ni
from readers import OFFReader

from multiprocessing import Pool, cpu_count
from vtk.util.numpy_support import vtk_to_numpy

from scaling import *


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

def WriteSurf(surf, fileName):
    fname, extension = os.path.splitext(fileName)
    extension = extension.lower()
    print("Writing:", fileName)
    if extension == ".vtk":
        writer = vtk.vtkPolyDataWriter()
    elif extension == ".stl":
        writer = vtk.vtkSTLWriter()

    writer.SetFileName(fileName)
    writer.SetInputData(surf)
    writer.Update()


def ScaleSurf(surf, scale_factor):
    shapedatapoints = surf.GetPoints()
    
    #calculate bounding box
    bounds = [0.0] * 6
    mean_v = [0.0] * 3
    bounds_max_v = [0.0] * 3
    bounds = shapedatapoints.GetBounds()
    mean_v[0] = (bounds[0] + bounds[1])/2.0
    mean_v[1] = (bounds[2] + bounds[3])/2.0
    mean_v[2] = (bounds[4] + bounds[5])/2.0
    bounds_max_v[0] = max(bounds[0], bounds[1])
    bounds_max_v[1] = max(bounds[2], bounds[3])
    bounds_max_v[2] = max(bounds[4], bounds[5])

    shape_points = []
    for i in range(shapedatapoints.GetNumberOfPoints()):
        p = shapedatapoints.GetPoint(i)
        shape_points.append(p)


    #centering points of the shape
    shape_points = np.array(shape_points)
    mean_arr = np.array(mean_v)
    shape_points = shape_points - mean_arr

    #Computing scale factor if it is not provided
    if(scale_factor == -1):
        bounds_max_arr = np.array(bounds_max_v)
        scale_factor = 1/np.linalg.norm(bounds_max_arr - mean_arr)
        # print(scale_factor)

    #scale points of the shape by scale factor
    shape_points = np.array(shape_points)
    shape_points_scaled = np.multiply(shape_points, scale_factor)

    #assigning scaled points back to shape
    for i in range(shapedatapoints.GetNumberOfPoints()):
       shapedatapoints.SetPoint(i, shape_points_scaled[i])    

    surf.SetPoints(shapedatapoints)

    return surf, mean_arr, scale_factor

def GetActor(surf):
	surfMapper = vtk.vtkPolyDataMapper()
	surfMapper.SetInputData(surf)

	surfActor = vtk.vtkActor()
	surfActor.SetMapper(surfMapper)

	return surfActor
def GetTransform(rotationAngle, rotationVector):
    transform = vtk.vtkTransform()
    transform.RotateWXYZ(rotationAngle, rotationVector[0], rotationVector[1], rotationVector[2])
    return transform

def RotateSurf(surf, rotationAngle, rotationVector):
	print("angle:", rotationAngle, "vector:", rotationVector)
	transform = GetTransform(rotationAngle, rotationVector)
	return RotateTransform(surf, transform)

def RotateInverse(surf, rotationAngle, rotationVector):
    print("angle:", rotationAngle, "vector:", rotationVector)
    transform = vtk.vtkTransform()
    transform.RotateWXYZ(rotationAngle, rotationVector[0], rotationVector[1], rotationVector[2])
   
    transform_i = vtk.vtkTransform()
    m_inverse = vtk.vtkMatrix4x4()
    transform.GetInverse(m_inverse)
    transform_i.SetMatrix(m_inverse)

    return RotateTransform(surf, transform_i)

def RotateTransform(surf, transform):
	transformFilter = vtk.vtkTransformPolyDataFilter()
	transformFilter.SetTransform(transform)
	transformFilter.SetInputData(surf)
	transformFilter.Update()
	return transformFilter.GetOutput()

def RotateNpTransform(surf, angle, np_transform):
	np_tran = np.load(np_transform)

	rotationAngle = -angle
	rotationVector = np_tran
	return RotateInverse(surf, rotationAngle, rotationVector)

def RandomRotation(surf):
	rotationVector = np.random.random(3)*2.0 - 1.0
	rotationVector = rotationVector/np.linalg.norm(rotationVector)
	rotationAngle = np.random.random()*360.0
	return RotateSurf(surf, rotationAngle, rotationVector)

def GetUnitSurf(surf, scale=None, translate=None, shape=None):

	if scale and translate and shape :
		surf = scalingSurf(surf, scale, translate, shape)
	else :
		surf, surf_mean, surf_scale = Normalization(surf)

# def orientationTraining(surf, pathLabel):
# 	rotationVector = np.random.random(3)*2.0 - 1.0
# 	rotationVector = rotationVector/np.linalg.norm(rotationVector)
# 	rotationAngle = np.random.random()*360.0

# 	print("angle:", rotationAngle, "vector:", rotationVector)
# 	transform = vtk.vtkTransform()
# 	transform.RotateWXYZ(rotationAngle, rotationVector[0], rotationVector[1], rotationVector[2])

# 	transformFilter = vtk.vtkTransformPolyDataFilter()
# 	transformFilter.SetTransform(transform)
# 	transformFilter.SetInputData(surf)
# 	transformFilter.Update()

# 	theta = np.radians(rotationAngle)
# 	c, s = np.cos(theta), np.sin(theta)
# 	x, y, z = rotationVector[0], rotationVector[1], rotationVector[2]
# 	R = np.array([[c+pow(x,2)*(1-c), x*y*(1-c)-z*s,    x*z*(1-c)+y*s], 
# 				  [y*x*(1-c)+z*s,    c+pow(y,2)*(1-c), y*z*(1-c)-x*s], 
# 				  [z*x*(1-c)-y*s,    z*y*(1-c)+x*s,    c+pow(z,2)*(1-c)]])

# 	# print("matrice rotation")
# 	# print(R)
# 	# print("matrice inverse/transpose:")
# 	Rt = np.transpose(R)
# 	# print(Rt)
# 	np.save(pathLabel, Rt)
# 	return transformFilter.GetOutput()

def GetColoredActor(surf, property_name):
    hueLut = vtk.vtkLookupTable()
    hueLut.SetTableRange(-1, 4)
    hueLut.SetHueRange(0, 1)
    hueLut.SetSaturationRange(1, 1)
    hueLut.SetValueRange(1, 1)
    hueLut.Build()

    surf.GetPointData().SetActiveScalars(property_name)

    actor = GetActor(surf)
    actor.GetMapper().ScalarVisibilityOn()
    actor.GetMapper().SetScalarModeToUsePointData()
    actor.GetMapper().SetColorModeToMapScalars()
    actor.GetMapper().SetUseLookupTableScalarRange(True)

    actor.GetMapper().SetLookupTable(hueLut)

    return actor


def GetPropertyActor(surf, property_name):

    #display property on surface
    point_data = vtk.vtkDoubleArray()
    point_data.SetNumberOfComponents(1)

    with open(property_name) as property_file:
        for line in property_file:
            point_val = float(line[:-1])
            point_data.InsertNextTuple([point_val])
                
        surf.GetPointData().SetScalars(point_data)

    surf_actor = GetActor(surf)
    surf_actor.GetProperty().LightingOff()
    surf_actor.GetProperty().ShadingOff()
    surf_actor.GetProperty().SetInterpolationToFlat()

    surfMapper = surf_actor.GetMapper()
    surfMapper.SetUseLookupTableScalarRange(True)

    
    #build lookup table
    number_of_colors = 512
    low_range = 0
    high_range = 1  
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(low_range, high_range)
    lut.SetNumberOfColors(number_of_colors)

    #Color transfer function  
    ctransfer = vtk.vtkColorTransferFunction()
    ctransfer.AddRGBPoint(0.0, 1.0, 1.0, 0.0) # Yellow
    ctransfer.AddRGBPoint(0.5, 1.0, 0.0, 0.0) # Red

    #Calculated new colors for LUT via color transfer function
    for i in range(number_of_colors):
        new_colour = ctransfer.GetColor( (i * ((high_range-low_range)/number_of_colors) ) )
        lut.SetTableValue(i, *new_colour)

    lut.Build()

    surfMapper.SetLookupTable(lut)


    return surfActor

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

		if vtk.VTK_MAJOR_VERSION > 8:

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

		else:
			colored_points = vtk.vtkUnsignedCharArray()
			colored_points.SetName('colors')
			colored_points.SetNumberOfComponents(3)

			normals = surf.GetPointData().GetArray('Normals')
			for pid in range(surf.GetNumberOfPoints()):
				normal = np.array(normals.GetTuple(pid))
				rgb = (normal*0.5 + 0.5)*255.0
				colored_points.InsertNextTuple3(rgb[0], rgb[1], rgb[2])

			surf.GetPointData().SetScalars(colored_points)

			surf_actor = GetActor(surf)
			surf_actor.GetProperty().LightingOff()
			surf_actor.GetProperty().ShadingOff()
			surf_actor.GetProperty().SetInterpolationToFlat()


		return surf_actor
	except Exception as e:
		print(e, file=sys.stderr)
		return None

# def GetUnitActor(fileName, property, scale_factor, random_rotation=False, normal_shaders=True):

#     try:

#         surf = ReadSurf(fileName)

#         surf, surf_mean, surf_scale = Normalization(surf)

#         if(random_rotation):
#             rotationVector = np.random.random(3)*2.0 - 1.0
#             rotationVector = rotationVector/np.linalg.norm(rotationVector)
#             rotationAngle = np.random.random()*360.0
#             surf = RotateSurf(surf, rotationAngle, rotationVector)

#         #Color with normals
#         if(normal_shaders):
#             normals = vtk.vtkPolyDataNormals()
#             normals.SetInputData(surf);
#             normals.ComputeCellNormalsOff();
#             normals.ComputePointNormalsOn();
#             # normals.AutoOrientNormalsOn();
#             normals.SplittingOff();
#             normals.Update()
#             surf = normals.GetOutput()

#         # mapper
#         surfActor = GetActor(surf, property, scale_factor)


#         if(normal_shaders):

#             sp = surfActor.GetShaderProperty();
#             sp.AddVertexShaderReplacement(
#                 "//VTK::Normal::Dec",
#                 True,
#                 "//VTK::Normal::Dec\n" + 
#                 "  varying vec3 myNormalMCVSOutput;\n",
#                 False
#             )

#             sp.AddVertexShaderReplacement(
#                 "//VTK::Normal::Impl",
#                 True,
#                 "//VTK::Normal::Impl\n" +
#                 "  myNormalMCVSOutput = normalMC;\n",
#                 False
#             )

#             sp.AddVertexShaderReplacement(
#                 "//VTK::Color::Impl",
#                 True, "VTK::Color::Impl\n", False)

#             sp.ClearVertexShaderReplacement("//VTK::Color::Impl", True)

#             sp.AddFragmentShaderReplacement(
#                 "//VTK::Normal::Dec",
#                 True,
#                 "//VTK::Normal::Dec\n" + 
#                 "  varying vec3 myNormalMCVSOutput;\n",
#                 False
#             )

#             sp.AddFragmentShaderReplacement(
#                 "//VTK::Normal::Impl",
#                 True,
#                 "//VTK::Normal::Impl\n" +
#                 "  diffuseColor = myNormalMCVSOutput*0.5f + 0.5f;\n",
#                 False
#             )

#         return surfActor
#     except Exception as e:
#         print(e, file=sys.stderr)
#         return None

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

def ExtractPointFeatures(surf, point_ids_rgb, point_features_name, zero=0):
	
	point_ids_rgb_shape = point_ids_rgb.shape

	point_features = surf.GetPointData().GetScalars(point_features_name)
	point_features_np = vtk_to_numpy(point_features)
	zero = np.zeros(point_features.GetNumberOfComponents()) + zero

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

def GetTubeFilter(surf, list_random_id):

	ids = vtk.vtkIdTypeArray()
	ids.SetNumberOfComponents(1)
	ids.InsertNextValue(list_random_id) 

	# extract a subset from a dataset
	selectionNode = vtk.vtkSelectionNode() 
	selectionNode.SetFieldType(0) 		# CELL_DATA = 0
	selectionNode.SetContentType(4) 	# INDICES = 4
	selectionNode.SetSelectionList(ids) # add the cell we want to extract
	
	# set containing cell to 1 = extract cell
	selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1) 

	selection = vtk.vtkSelection()
	selection.AddNode(selectionNode)

	# extract the cell from the cluster
	extractSelection = vtk.vtkExtractSelection()
	extractSelection.SetInputData(0, surf)
	extractSelection.SetInputData(1, selection)
	extractSelection.Update()

	# convert the extract cell to a polygonal type (a line here)
	geometryFilter = vtk.vtkGeometryFilter()
	geometryFilter.SetInputData(extractSelection.GetOutput())
	geometryFilter.Update()

	# create a tube fiber 
	tubeFilter = vtk.vtkTubeFilter()
	tubeFilter.SetNumberOfSides(50)
	tubeFilter.SetRadius(0.008)
	tubeFilter.SetInputData(geometryFilter.GetOutput())
	tubeFilter.Update()

	return tubeFilter.GetOutput()
