import vtk
import numpy as np
import argparse
import os
from utils import *


def main(args):

    inputSurface = args.surf
    if args.n_rot > 1 and not os.path.exists(args.out):
        os.mkdir(args.out)

    path, extension = os.path.splitext(inputSurface)
    extension = extension.lower()
    surf = ReadSurf(args.surf)

    if args.np_transform:
        rotated_surf = RotateNpTransform(surf, args.angle, args.np_transform)
        outfilename = args.out
        print("Writting:", outfilename)
        polydatawriter = vtk.vtkPolyDataWriter()
        polydatawriter.SetFileName(outfilename)
        polydatawriter.SetInputData(rotated_surf)
        polydatawriter.Write()

    elif args.n_rot == 1:
        
        if args.angle and args.rot_v:
            rotated_surf = RotateSurf(surf, args.angle, args.rot_v)
        else:
            rotated_surf = RandomRotation(surf)
        outfilename = args.out

        print("Writting:", outfilename)
        polydatawriter = vtk.vtkPolyDataWriter()
        polydatawriter.SetFileName(outfilename)
        polydatawriter.SetInputData(rotated_surf)
        polydatawriter.Write()
    else:
        for i in range(args.n_rot):
            if args.angle and args.rot_v:
                rotated_surf = RotateSurf(surf, args.angle, args.rot_v)
            else:
                rotate_surf = RandomRotation(surf)

            outfilename, ext = os.path.splitext(inputSurface)
            outfilename = os.path.join(args.out, os.path.basename(outfilename)) + "_" + str(i) + ".vtk"

            print("Writting:", outfilename)
            polydatawriter = vtk.vtkPolyDataWriter()
            polydatawriter.SetFileName(outfilename)
            polydatawriter.SetInputData(rotated_surf)
            polydatawriter.Write()


    if(args.visualize):
        mapper_orig = vtk.vtkPolyDataMapper()
        mapper_orig.SetInputData(surf)

        mapper_transform = vtk.vtkPolyDataMapper()
        mapper_transform.SetInputData(rotated_surf)

        actor1 = vtk.vtkActor()
        actor1.SetMapper(mapper_orig)

        actor2 = vtk.vtkActor()
        actor2.SetMapper(mapper_transform)

        colors = vtk.vtkNamedColors()
        actor1.GetProperty().SetColor(colors.GetColor3d('Red'))
        actor2.GetProperty().SetColor(colors.GetColor3d('Blue'))

        
        ren = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        WIDTH = 640
        HEIGHT = 480
        renWin.SetSize(WIDTH, HEIGHT)

        # create a renderwindowinteractor
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        # assign actor to the renderer
        ren.AddActor(actor1)
        ren.AddActor(actor2)

        # enable user interface interactor
        iren.Initialize()
        renWin.Render()
        iren.Start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform a random rotation of a polydata', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--surf', type=str, help='Input surface mesh to label', required=True)
    parser.add_argument('--np_transform', type=str, help='')
    parser.add_argument('--rot_v', type=float, nargs='+', help='rotation axis', default=None)
    parser.add_argument('--angle', type=float, help='rotation angle', default=None)
    parser.add_argument('--n_rot', type=int, help='Number of rotations to perform', default=1)
    parser.add_argument('--visualize', type=bool, help='Visualize the outputs', default=False)
    parser.add_argument('--out', type=str, help='Output directory or filename for polydatas', default="./out")

    args = parser.parse_args()

    main(args)