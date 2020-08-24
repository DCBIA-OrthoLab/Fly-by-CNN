import vtk
import numpy as np
import argparse
import os

def transform(surf, rot_v = None, angle = None):
    # create a transform that rotates the cone
    transform = vtk.vtkTransform()
    if rot_v is not None:
        rot_vector = args.rot_v
    else:
        rot_vector = np.random.normal(loc=0.0, scale=1.0, size=(3,))
        rot_vector = rot_vector/np.linalg.norm(rot_vector)

    if angle is None:
        angle = np.random.uniform(low=0.0, high=1.0)*360
    transform.RotateWXYZ(angle, rot_vector[0], rot_vector[1], rot_vector[2])
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(surf)
    transformFilter.Update()
    return transformFilter.GetOutput()



def main(args):

    inputSurface = args.surf
    if args.n_rot > 1 and not os.path.exists(args.out):
        os.mkdir(args.out)

    path, extension = os.path.splitext(inputSurface)
    extension = extension.lower()
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(inputSurface)
        reader.Update()
        original_surf = reader.GetOutput()
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(inputSurface)
        reader.Update()
        original_surf = reader.GetOutput()


    if args.n_rot == 1:
        
        rotated_surf = transform(original_surf, args.rot_v, args.angle)
        outfilename = args.out

        print("Writting:", outfilename)
        polydatawriter = vtk.vtkPolyDataWriter()
        polydatawriter.SetFileName(outfilename)
        polydatawriter.SetInputData(rotated_surf)
        polydatawriter.Write()
    else:
        for i in range(args.n_rot):
            rotated_surf = transform(original_surf)

            outfilename, ext = os.path.splitext(inputSurface)
            outfilename = os.path.join(args.out, os.path.basename(outfilename)) + "_" + str(i) + ".vtk"

            print("Writting:", outfilename)
            polydatawriter = vtk.vtkPolyDataWriter()
            polydatawriter.SetFileName(outfilename)
            polydatawriter.SetInputData(rotated_surf)
            polydatawriter.Write()


    if(args.visualize):
        mapper_orig = vtk.vtkPolyDataMapper()
        mapper_orig.SetInputData(original_surf)

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
    parser.add_argument('--rot_v', type=float, nargs='+', help='rotation axis', default=None)
    parser.add_argument('--angle', type=float, help='rotation angle', default=None)
    parser.add_argument('--n_rot', type=int, help='Number of rotations to perform', default=1)
    parser.add_argument('--visualize', type=bool, help='Visualize the outputs', default=False)
    parser.add_argument('--out', type=str, help='Output directory or filename for polydatas', default="./out")

    args = parser.parse_args()

    main(args)