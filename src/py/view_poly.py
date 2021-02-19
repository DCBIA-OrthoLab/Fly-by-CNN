import vtk
import numpy as np
import argparse
import os
from utils import *

def main(args):
    
    colors = vtk.vtkNamedColors()
    # Read all the data from the file
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
    parser = argparse.ArgumentParser(description='Perform a random rotation of a polydata', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--surf', type=str, help='Input surface mesh to label', required=True)
    args = parser.parse_args()

    main(args)