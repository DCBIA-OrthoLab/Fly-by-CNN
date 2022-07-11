import vtk
import LinearSubdivisionFilter as lsf
import numpy as np
import math 
import os
import sys
import itk
from readers import OFFReader
import pandas as pd
from multiprocessing import Pool, cpu_count
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk

import torch
import monai
from monai.transforms import (
    ToTensor
)

random_table = {0: [0.20609576603761093, 0.049483664108669334, 0.12125618974900243], 1: [0.8226083537896619, 0.4351012583759224, 0.9832181474162848], 2: [0.7816089005179315, 0.10346579724552551, 0.23487367424552041], 3: [0.025519074513923767, 0.3171905857331965, 0.15567817079277158], 4: [0.6663883241545112, 0.6000648624913231, 0.37658036275159334], 5: [0.8776899967271671, 0.7616932316027937, 0.7829308539989045], 6: [0.898071921630757, 0.551591502770238, 0.821443825353693], 7: [0.2151654905680822, 0.6862729312525537, 0.6138162023396329], 8: [0.3045080015195074, 0.18899392802023163, 0.40911076426399806], 9: [0.11765999434984209, 0.6975068194921006, 0.7879352132497024], 10: [0.7008483630188418, 0.29605323524580973, 0.8198879200426741], 11: [0.37520055941188546, 0.8341404734565594, 0.26177124408355934], 12: [0.3934631970864203, 0.623997372518547, 0.6436244132804986], 13: [0.4448759237168636, 0.20539400268295127, 0.13780334209661682], 14: [0.1512125043260758, 0.8497980664920212, 0.9668755390459113], 15: [0.4226479922022087, 0.6474852944643369, 0.8416756096111622], 16: [0.3433546138380138, 0.4753996230412757, 0.9250164530083743], 17: [0.3568056105452764, 0.18334522038240408, 0.5530289655651638], 18: [0.20720530535494464, 0.30149437038255333, 0.8620008177802112], 19: [0.5064194091445487, 0.371008925644291, 0.7615128573103098], 20: [0.8113590149155854, 0.991276799639109, 0.4678565663944516], 21: [0.3918520636479679, 0.4516887244764669, 0.10207228081624953], 22: [0.08985831157936286, 0.7697455049664752, 0.1074750433948709], 23: [0.14858286085833028, 0.4702633232662803, 0.8486448165595275], 24: [0.8841279840560545, 0.4172026503952089, 0.8533483665140427], 25: [0.7068243612992762, 0.6953362178559506, 0.8568301873334966], 26: [0.49107423909005066, 0.29355390600500664, 0.7047031635033804], 27: [0.8420869188007641, 0.09722208762577988, 0.5679188126651894], 28: [0.7048997343179865, 0.5842467928067391, 0.3599437108217626], 29: [0.031132795477680664, 0.47856137471759597, 0.7714699856897599], 30: [0.7534987166987384, 0.10452057614858501, 0.5444979823604356], 31: [0.7474303156543861, 0.7925880784420662, 0.3604102361881435], 32: [0.8628363999159102, 0.21101681904692737, 0.9013222342779957], 33: [0.629879190497666, 0.984613775038266, 0.19913181707246874], 34: [0.5658644598064076, 0.2802941241818787, 0.7731639034323629], 35: [0.7161432827471007, 0.07320876761101225, 0.8036390200148513], 36: [0.7652556867768864, 0.29238753434799825, 0.9531094559385218], 37: [0.15696680412837793, 0.3931011713942679, 0.4243212240743779], 38: [0.42187683148471333, 0.18912842290221588, 0.6819018502705237], 39: [0.4442228218708715, 0.5778909523966065, 0.9642724982954931], 40: [0.9758068203300386, 0.05027079482550001, 0.2650237477395715], 41: [0.08943656541981004, 0.13753428811570412, 0.5603078234439788], 42: [0.010201621306042186, 0.327628786871763, 0.3890688609399522], 43: [0.8210302259754378, 0.6115200256108921, 0.15031238781810718], 44: [0.7001777727283601, 0.3645027551754205, 0.07376451995446232], 45: [0.9698531436998832, 0.751003571749085, 0.5743430137745764], 46: [0.468449857802292, 0.7595685633637632, 0.053443953365929664], 47: [0.7004671870884197, 0.1500730383346197, 0.3643145018601389], 48: [0.5450051853727357, 0.20296281917381587, 0.6905551030638406], 49: [0.25030733321324905, 0.5641932299258015, 0.8330475173454799], 50: [0.09985199678325007, 0.05718411388275524, 0.04637388264853759], 51: [0.46781693913408406, 0.4601588232568161, 0.5980582562940925], 52: [0.6530110537697219, 0.024486312086407724, 0.1202412863786424], 53: [0.3118463907417399, 0.2189245452784624, 0.0023464268741028027], 54: [0.8241142867493606, 0.0926370028968565, 0.1972737972668107], 55: [0.20008926867870458, 0.2724840513608894, 0.24981531935472734], 56: [0.8502674870599242, 0.3921126629717022, 0.6819512401878478], 57: [0.5302783999655287, 0.8505255934951383, 0.49144132387275363], 58: [0.5115899748202699, 0.9350860104306851, 0.32636803398265724], 59: [0.21445563074258867, 0.5014534480264361, 0.6877952670334156]}

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

def CreateIcosahedron(radius, sl=0):
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

def CleanPoly(surf, merge_points = False):
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(surf)
    clean.PointMergingOff()
    if merge_points:
        clean.PointMergingOn()
    clean.Update()

    return clean.GetOutput()

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

    fname, extension = os.path.splitext(fileName)
    extension = extension.lower()
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
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
                textures_path = os.path.normpath(fname.replace(os.path.basename(fname), ''))
                obj_import.SetTexturePath(textures_path)
            else:
                textures_path = os.path.normpath(fname.replace(os.path.basename(fname), ''))                
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
    elif extension == '.gii':
        import nibabel as nib
        from fsl.data import gifti

        surf = nib.load(fileName)
        coords = surf.agg_data('pointset')
        triangles = surf.agg_data('triangle')

        points = vtk.vtkPoints()

        for c in coords:
            points.InsertNextPoint(c[0], c[1], c[2])

        cells = vtk.vtkCellArray()

        for t in triangles:
            t_vtk = vtk.vtkTriangle()
            t_vtk.GetPointIds().SetId(0, t[0])
            t_vtk.GetPointIds().SetId(1, t[1])
            t_vtk.GetPointIds().SetId(2, t[2])
            cells.InsertNextCell(t_vtk)

        surf = vtk.vtkPolyData()
        surf.SetPoints(points)
        surf.SetPolys(cells)

    return surf

def WriteSurf(surf, fileName, use_binary=False):
    fname, extension = os.path.splitext(fileName)
    extension = extension.lower()
    print("Writing:", fileName)
    if extension == ".vtk":
        writer = vtk.vtkPolyDataWriter()
    elif extension == ".stl":
        writer = vtk.vtkSTLWriter()

    writer.SetFileName(fileName)
    writer.SetInputData(surf)
    if(use_binary):
        writer.SetFileTypeToBinary()
    writer.Update()

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

def ScaleSurf(surf, mean_arr = None, scale_factor = None, copy=True):
    if(copy):
        surf_copy = vtk.vtkPolyData()
        surf_copy.DeepCopy(surf)
        surf = surf_copy

    shapedatapoints = surf.GetPoints()

    #calculate bounding box
    mean_v = [0.0] * 3
    bounds_max_v = [0.0] * 3

    bounds = shapedatapoints.GetBounds()

    mean_v[0] = (bounds[0] + bounds[1])/2.0
    mean_v[1] = (bounds[2] + bounds[3])/2.0
    mean_v[2] = (bounds[4] + bounds[5])/2.0
    bounds_max_v[0] = max(bounds[0], bounds[1])
    bounds_max_v[1] = max(bounds[2], bounds[3])
    bounds_max_v[2] = max(bounds[4], bounds[5])

    shape_points = vtk_to_numpy(shapedatapoints.GetData())
    
    #centering points of the shape
    if mean_arr is None:
        mean_arr = np.array(mean_v)
    # print("Mean:", mean_arr)
    shape_points = shape_points - mean_arr

    #Computing scale factor if it is not provided
    if(scale_factor is None):
        bounds_max_arr = np.array(bounds_max_v)
        scale_factor = 1/np.linalg.norm(bounds_max_arr - mean_arr)

    #scale points of the shape by scale factor
    # print("Scale:", scale_factor)
    shape_points = np.multiply(shape_points, scale_factor)

    #assigning scaled points back to shape
    shapedatapoints.SetData(numpy_to_vtk(shape_points))

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
    transform = GetTransform(rotationAngle, rotationVector)
    return RotateTransform(surf, transform)

def RotateInverse(surf, rotationAngle, rotationVector):
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
    rotationAngle = np.random.random()*360.0
    rotationVector = np.random.random(3)*2.0 - 1.0
    rotationVector = rotationVector/np.linalg.norm(rotationVector)
    return RotateSurf(surf, rotationAngle, rotationVector), rotationAngle, rotationVector

def GetUnitSurf(surf, mean_arr = None, scale_factor = None, copy=True):
  unit_surf, surf_mean, surf_scale = ScaleSurf(surf, mean_arr, scale_factor, copy)
  return unit_surf

def GetColoredActor(surf, property_name, range_scalars = None):

    if range_scalars == None:
        range_scalars = surf.GetPointData().GetScalars(property_name).GetRange()

    hueLut = vtk.vtkLookupTable()
    hueLut.SetTableRange(0, range_scalars[1])
    hueLut.SetHueRange(0.0, 0.9)
    hueLut.SetSaturationRange(1.0, 1.0)
    hueLut.SetValueRange(1.0, 1.0)
    hueLut.Build()

    surf.GetPointData().SetActiveScalars(property_name)

    actor = GetActor(surf)
    actor.GetMapper().ScalarVisibilityOn()
    actor.GetMapper().SetScalarModeToUsePointData()
    actor.GetMapper().SetColorModeToMapScalars()
    actor.GetMapper().SetUseLookupTableScalarRange(True)

    actor.GetMapper().SetLookupTable(hueLut)

    return actor

def GetRandomColoredActor(surf, property_name, range_scalars = [0, 1000]):

    if range_scalars == None:
        range_scalars = surf.GetPointData().GetScalars(property_name).GetRange()

    
    ctf = vtk.vtkColorTransferFunction()        
    ctf.SetColorSpaceToRGB()

    for i in range(range_scalars[0], range_scalars[1]):
        ctf.AddRGBPoint(i, np.random.rand(), np.random.rand(), np.random.rand())

    surf.GetPointData().SetActiveScalars(property_name)

    actor = GetActor(surf)
    actor.GetMapper().ScalarVisibilityOn()
    actor.GetMapper().SetScalarModeToUsePointData()
    actor.GetMapper().SetColorModeToMapScalars()
    actor.GetMapper().SetUseLookupTableScalarRange(True)

    actor.GetMapper().SetLookupTable(ctf)

    return actor

def GetSeparateColoredActor(surf, property_name, range_scalars = [0, 60]):
    # dico = {}
    # for i in range(60):
    #     dico[i] = [np.random.rand(), np.random.rand(), np.random.rand()]
    # print(dico)

    if range_scalars == None:
        range_scalars = surf.GetPointData().GetScalars(property_name).GetRange()

    
    ctf = vtk.vtkColorTransferFunction()        
    ctf.SetColorSpaceToRGB()

    for i in range(range_scalars[0], range_scalars[1]):
        ctf.AddRGBPoint(i, random_table[i][0],random_table[i][1],random_table[i][2])

    surf.GetPointData().SetActiveScalars(property_name)

    actor = GetActor(surf)
    actor.GetMapper().ScalarVisibilityOn()
    actor.GetMapper().SetScalarModeToUsePointData()
    actor.GetMapper().SetColorModeToMapScalars()
    actor.GetMapper().SetUseLookupTableScalarRange(True)

    actor.GetMapper().SetLookupTable(ctf)

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

    # return surfActor
    return surfMapper

def ComputeNormals(surf):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surf);
    normals.ComputeCellNormalsOn();
    normals.ComputePointNormalsOn();
    normals.SplittingOff();
    normals.Update()
    
    return normals.GetOutput()

def GetColorArray(surf, array_name):
    colored_points = vtk.vtkUnsignedCharArray()
    colored_points.SetName('colors')
    colored_points.SetNumberOfComponents(3)

    normals = surf.GetPointData().GetArray(array_name)

    for pid in range(surf.GetNumberOfPoints()):
        normal = np.array(normals.GetTuple(pid))
        rgb = (normal*0.5 + 0.5)*255.0
        colored_points.InsertNextTuple3(rgb[0], rgb[1], rgb[2])
    return colored_points

def GetNormalsActor(surf):
    try:
        
        surf = ComputeNormals(surf)
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
            
            colored_points = GetColorArray(surf, "Normals")
            surf.GetPointData().SetScalars(colored_points)

            surf_actor = GetActor(surf)
            surf_actor.GetProperty().LightingOff()
            surf_actor.GetProperty().ShadingOff()
            surf_actor.GetProperty().SetInterpolationToFlat()


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

def GetPointIdColors(surf):
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
    
    return colored_points

def GetPointIdMapActor(surf):

    colored_points = GetPointIdColors(surf)

    # cell_id_color = int(b*255*255 + g*255 + r - 1)

    surf.GetCellData().SetScalars(colored_points)

    surf_actor = GetActor(surf)
    surf_actor.GetMapper().SetScalarModeToUseCellData()
    surf_actor.GetProperty().LightingOff()
    surf_actor.GetProperty().ShadingOff()
    surf_actor.GetProperty().SetInterpolationToFlat()

    return surf_actor
  
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
                point_features.append([self.point_features_np[point_id]])
            else:
                point_features.append(self.zero)

        return point_features

def ExtractPointFeatures(surf, point_ids_rgb, point_features_name, zero=0, use_multi=True):

    point_ids_rgb_shape = point_ids_rgb.shape

    if point_features_name == "coords" or point_features_name == "points":
        points = surf.GetPoints()
        point_features_np = vtk_to_numpy(points.GetData())
        number_of_components = 3
    else:    
        point_features = surf.GetPointData().GetScalars(point_features_name)
        point_features_np = vtk_to_numpy(point_features)
        number_of_components = point_features.GetNumberOfComponents()
    
    zero = np.zeros(number_of_components) + zero

    if use_multi:
        with Pool(cpu_count()) as p:
            feat = p.map(ExtractPointFeaturesClass(point_features_np, zero), point_ids_rgb)
    else:
        feat = ExtractPointFeaturesClass(point_features_np, zero)(point_ids_rgb)
    return np.array(feat).reshape(point_ids_rgb_shape[0:-1] + (number_of_components,))

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

def GetTubeFilter(vtkpolydata):

    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetNumberOfSides(50)
    tubeFilter.SetRadius(0.01)
    tubeFilter.SetInputData(vtkpolydata)
    tubeFilter.Update()

    return tubeFilter.GetOutput()

def ExtractFiber(surf, list_random_id) :
    ids = vtk.vtkIdTypeArray()
    ids.SetNumberOfComponents(1)
    ids.InsertNextValue(list_random_id) 

    # extract a subset from a dataset
    selectionNode = vtk.vtkSelectionNode() 
    selectionNode.SetFieldType(0)
    selectionNode.SetContentType(4)
    selectionNode.SetSelectionList(ids) 

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


    tubefilter = GetTubeFilter(geometryFilter.GetOutput())

    return tubefilter

def Write(vtkdata, output_name, print_out = True):
    outfilename = output_name
    if print_out == True:
        print("Writing:", outfilename)
    polydatawriter = vtk.vtkPolyDataWriter()
    polydatawriter.SetFileName(outfilename)
    polydatawriter.SetInputData(vtkdata)
    polydatawriter.Write()

def json2vtk(jsonfile,number_landmarks,radius_sphere,outdir):
    
    json_file = pd.read_json(jsonfile)
    json_file.head()
    markups = json_file.loc[0,'markups']
    controlPoints = markups['controlPoints']
    number_landmarks = len(controlPoints)
    L_landmark_position = []
    
    for i in range(number_landmarks):
        L_landmark_position.append(controlPoints[i]["position"])
        # Create a sphere
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(L_landmark_position[i][0],L_landmark_position[i][1],L_landmark_position[i][2])
        sphereSource.SetRadius(radius_sphere)

        # Make the surface smooth.
        sphereSource.SetPhiResolution(100)
        sphereSource.SetThetaResolution(100)
        sphereSource.Update()
        vtk_landmarks = vtk.vtkAppendPolyData()
        vtk_landmarks.AddInputData(sphereSource.GetOutput())
        vtk_landmarks.Update()

        basename = os.path.basename(jsonfile).split(".")[0]
        filename = basename + "_landmarks.vtk"
        output = os.path.join(outdir, filename)
        Write(vtk_landmarks.GetOutput(), output)
    return output
    
def ArrayToTensor(vtkarray, device='cpu', dtype=torch.int64):
    return ToTensor(dtype=dtype, device=device)(vtk_to_numpy(vtkarray))

def PolyDataToTensors(surf, device='cpu'):

    verts, faces, edges = PolyDataToNumpy(surf)
    
    verts = ToTensor(dtype=torch.float32, device=device)(verts)
    faces = ToTensor(dtype=torch.int32, device=device)(faces)
    edges = ToTensor(dtype=torch.int32, device=device)(edges)
    
    return verts, faces, edges

def PolyDataToNumpy(surf):

    edges_filter = vtk.vtkExtractEdges()
    edges_filter.SetInputData(surf)
    edges_filter.Update()

    verts = vtk_to_numpy(surf.GetPoints().GetData())
    faces = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]
    edges = vtk_to_numpy(edges_filter.GetOutput().GetLines().GetData()).reshape(-1, 3)[:,1:]
    
    return verts, faces, edges

def UnitVerts(verts):
    min_verts, _ = torch.min(verts, axis=0)
    max_verts, _ = torch.max(verts, axis=0)
    mean_v = (min_verts + max_verts)/2.0
    
    verts = verts - mean_v
    scale_factor = 1/torch.linalg.vector_norm(max_verts - mean_v)
    verts = verts*scale_factor
    
    return verts, mean_v, scale_factor

def ComputeVertexNormals(verts, faces):
    face_area, face_normals = mesh_face_areas_normals(verts, faces)

    vert_normals = []

    for idx in range(len(v)):
        normals = face_normals[(faces == idx).nonzero(as_tuple=True)[0]] #Get all adjacent normal faces for the given point id
        areas = face_area[(faces == idx).nonzero(as_tuple=True)[0]] # Get all adjacent normal areas for the given point id

        normals = torch.mul(normals, areas.reshape(-1, 1)) # scale each normal by the area
        normals = torch.sum(normals, axis=0) # sum everything
        normals = torch.nn.functional.normalize(normals, dim=0) #normalize

    verts, faces, edges = PolyDataToNumpy(surf)
    
    verts = ToTensor(dtype=torch.float32, device=device)(verts)
    faces = ToTensor(dtype=torch.int32, device=device)(faces)
    edges = ToTensor(dtype=torch.int32, device=device)(edges)
    
    return verts, faces, edges

def PolyDataToNumpy(surf):

    edges_filter = vtk.vtkExtractEdges()
    edges_filter.SetInputData(surf)
    edges_filter.Update()

    verts = vtk_to_numpy(surf.GetPoints().GetData())
    faces = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]
    edges = vtk_to_numpy(edges_filter.GetOutput().GetLines().GetData()).reshape(-1, 3)[:,1:]
    
    return verts, faces, edges

def UnitVerts(verts):
    min_verts, _ = torch.min(verts, axis=0)
    max_verts, _ = torch.max(verts, axis=0)
    mean_v = (min_verts + max_verts)/2.0
    
    verts = verts - mean_v
    scale_factor = 1/torch.linalg.vector_norm(max_verts - mean_v)
    verts = verts*scale_factor
    
    return verts, mean_v, scale_factor

def ComputeVertexNormals(verts, faces):
    face_area, face_normals = mesh_face_areas_normals(verts, faces)

    vert_normals = []

    for idx in range(len(v)):
        normals = face_normals[(faces == idx).nonzero(as_tuple=True)[0]] #Get all adjacent normal faces for the given point id
        areas = face_area[(faces == idx).nonzero(as_tuple=True)[0]] # Get all adjacent normal areas for the given point id

        normals = torch.mul(normals, areas.reshape(-1, 1)) # scale each normal by the area
        normals = torch.sum(normals, axis=0) # sum everything
        normals = torch.nn.functional.normalize(normals, dim=0) #normalize

        vert_normals.append(normals.numpy())
    
    return torch.as_tensor(vert_normals)

def ReadJSONMarkups(fname, idx=0):
    fiducials = json.load(open(fname))
    markups = fiducials['markups']
    controlPoints = markups[idx]['controlPoints']

    controlPoints_np = []

    for cp in controlPoints:
        controlPoints_np.append(cp["position"])

    return np.array(controlPoints_np)
