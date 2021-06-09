import vtk
import itk
import argparse
import glob
import os

import numpy as np


def Write(vtkdata, output_name):
	outfilename = output_name
	print("Writting:", outfilename)
	polydatawriter = vtk.vtkPolyDataWriter()
	polydatawriter.SetFileName(outfilename)
	polydatawriter.SetInputData(vtkdata)
	polydatawriter.Write()

def ReadFile(filename):
    print("Reading:", filename)
    inputSurface = filename
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(inputSurface)
    reader.Update()
    vtkdata = reader.GetOutput()
    return vtkdata

def AssignLabelToRoots(root_canal, surf):
    label_array = surf.GetPointData().GetArray('RegionId')
      
    surfID = vtk.vtkOctreePointLocator()
    surfID.SetDataSet(surf)
    surfID.BuildLocator()

    labelID = vtk.vtkIntArray()
    labelID.SetNumberOfComponents(1)
    labelID.SetNumberOfTuples(root_canal.GetNumberOfPoints())
    labelID.SetName("RegionId")
    labelID.Fill(-1)

    for pid in range(labelID.GetNumberOfTuples()):
        xyzCoord = root_canal.GetPoint(pid)
        ID = surfID.FindClosestPoint(xyzCoord[0], xyzCoord[1], xyzCoord[2], vtk.reference(20))
        labelID.SetTuple(pid, (int(label_array.GetTuple(ID)[0]),))

    root_canal.GetPointData().AddArray(labelID)

    return root_canal


def main(args):
    img_fn_array, L_root = [], []

    normpath = os.path.normpath("/".join([args.dir_root, '**', '*']))
    for img_fn in glob.iglob(normpath, recursive=True):
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".vtk"]]:
            img_obj = {}
            L_root.append(img_fn)
            img_obj["root"] = L_root
            img_obj["surf"] = args.surf
            img_obj["out"] = args.out
    img_fn_array.append(img_obj)


    for img_obj in img_fn_array:
        surf = img_obj["surf"]
        root = img_obj["root"]
        out = img_obj["out"]

        surf = ReadFile(surf)
 
        merge = vtk.vtkAppendPolyData()
        merge.AddInputData(surf)

        for i in range(len(root)):
            root_canal = ReadFile(root[i])
            root_canal = AssignLabelToRoots(root_canal, surf)
            merge.AddInputData(root_canal)
            merge.Update()

        Write(merge.GetOutput(), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create RootCanal object from a segmented file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--surf', type=str, help='input teeth', required=True)
    parser.add_argument('--dir_root', type=str, help='input root canals', required=True)

    parser.add_argument('--out', type=str, help='output', default='')

    args = parser.parse_args()

    main(args)






