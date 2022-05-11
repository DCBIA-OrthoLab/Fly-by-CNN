import sys
import numpy as np
import time
import itk
import argparse
import os
import vtk
import pandas as pd
import glob


def Write(vtkdata, output_name):
    outfilename = output_name
    print("Writting:", outfilename)
    polydatawriter = vtk.vtkPolyDataWriter()
    polydatawriter.SetFileName(outfilename)
    polydatawriter.SetInputData(vtkdata)
    polydatawriter.Write()



def main(args):

    list_file = []
    normpath = os.path.normpath("/".join([args.dir, '**', '*']))

    if args.dir:
        for jsonfile in sorted(glob.iglob(normpath, recursive=True)):
            if os.path.isfile(jsonfile) and True in [ext in jsonfile for ext in [".json"]]:
                list_file.append(jsonfile)

    
    for file in sorted(list_file):
        vtk_landmarks = vtk.vtkAppendPolyData()
        json_file = pd.read_json(file)
        json_file.head()
        markups = json_file.loc[0,'markups']
        controlPoints = markups['controlPoints']
        number_landmarks = len(controlPoints)

        L_landmark = []
        for i in range(number_landmarks):
            L_landmark.append(controlPoints[i])
        
        for cp in L_landmark:
            # Create a sphere
            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetCenter(cp["position"][0],cp["position"][1],cp["position"][2])
            sphereSource.SetRadius(args.radius_sphere)

            # Make the surface smooth.
            sphereSource.SetPhiResolution(100)
            sphereSource.SetThetaResolution(100)
            sphereSource.Update()

            vtk_landmarks.AddInputData(sphereSource.GetOutput())
            vtk_landmarks.Update()

        basename = os.path.basename(file).split(".")[0]
        filename = basename + "_landmarks.vtk"
        output = os.path.join(args.out, filename)
        Write(vtk_landmarks.GetOutput(), output)


    if args.visualize:
        colors = vtk.vtkNamedColors()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vtk_landmarks.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d("Cornsilk"))

        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetWindowName("Sphere")
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        renderer.AddActor(actor)
        renderer.SetBackground(colors.GetColor3d("DarkGreen"))

        renderWindow.Render()
        renderWindowInteractor.Start()



if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Convert a json file with landmarks position in it into a vtk file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_parser = parser.add_argument_group('Input parameters')
    input_parser.add_argument('--dir', type=str, help='input dir', required=True)

    param_parser = parser.add_argument_group('Shere parameters')
    param_parser.add_argument('--radius_sphere', type=float, help='sphere radius', default=0.5)

    visualization_parser = parser.add_argument_group('Visualization tool')
    visualization_parser.add_argument('--visualize', type=bool, help='to visualize set 1', default=0)

    out_parser = parser.add_argument_group('Output parameters')
    out_parser.add_argument('--out', type=str, help='output dir', default='')

    args = parser.parse_args()

    main(args)










