import vtk
import numpy as np
import argparse
import os
from utils import *
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk

def main(args):
    
    colors = vtk.vtkNamedColors()
    # Read all the data from the file
    if(args.normals):
        surf = ReadSurf(args.surf)
        surfActor = GetNormalsActor(surf)
    elif(args.property):
        property_array = numpy_to_vtk(np.loadtxt(args.property))
        property_array.SetName(args.property)
        surf = ReadSurf(args.surf)
        surf.GetPointData().AddArray(property_array)
        surfActor = GetColoredActor(surf, args.property)
    else:
        surf = ReadSurf(args.surf)
        surfActor = GetActor(surf)

    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderer.AddActor(surfActor)
    renderer.SetBackground(1, 1, 1)
    renderer.ResetCamera()
    # renderer.GetActiveCamera().SetViewUp(0, 0, 1)


    renderWindow.SetSize(1900, 1200)
    renderWindow.Render()
    renderWindowInteractor.Start()

    if args.out:
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(args.out)
        writer.SetInputData(surf)
        writer.Write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick view a mesh', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--surf', type=str, help='Mesh surface in .vtk, .obj, .stl format', required=True)
    parser.add_argument('--normals', type=int, help='Color using the normals', default=0)
    parser.add_argument('--property', type=str, help='Color using the property in the file (text file, one value per line)', default=None)
    parser.add_argument('--property_name', type=str, help='Property name to be added', default="property")
    parser.add_argument('--out', type=str, help='Save output model', default=None)
    args = parser.parse_args()

    main(args)