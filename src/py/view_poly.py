import vtk
import numpy as np
import argparse
import os
from utils import *

def main(args):
    
    colors = vtk.vtkNamedColors()
    # Read all the data from the file
    if(args.normals):
        surfActor = GetNormalsActor(ReadSurf(args.surf))
    else:
        surfActor = GetActor(ReadSurf(args.surf))

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick view a mesh', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--surf', type=str, help='Mesh surface in .vtk, .obj, .stl format', required=True)
    parser.add_argument('--normals', type=int, help='Color using the normals', default=0)
    args = parser.parse_args()

    main(args)