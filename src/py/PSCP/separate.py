import vtk
import itk
import argparse
import glob
import os
import shutil

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


def main(args):
    img_fn_array = []

    if args.surf:
        img_obj = {}
        img_obj["surf"] = args.surf
        img_obj["out_dir"] = args.out_dir
        img_fn_array.append(img_obj)

    if args.dir:
        normpath = os.path.normpath("/".join([args.dir, '**', '*']))
        for img_fn in glob.iglob(normpath, recursive=True):
            if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".vtk", ".stl"]]:
                img_obj = {}
                img_obj["surf"] = img_fn
                img_obj["out_dir"] = os.path.normpath("/".join([args.out_dir, os.path.splitext(os.path.splitext(os.path.basename(img_fn))[0])[0]]))
                img_fn_array.append(img_obj)

    for img_obj in img_fn_array:

        filename, _ = os.path.splitext(os.path.basename(img_obj["surf"]))
        output_dir = img_obj["out_dir"]

        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        # else:
        #     shutil.rmtree(output_dir)
        #     os.makedirs(output_dir)

        if args.regionID:
            ScalarName = "RegionId"

        if args.universalID:
            ScalarName = "UniversalID"



        surf = ReadFile(img_obj["surf"])
        Write(surf, output_dir+"/"+filename+".vtk")

        label_array = surf.GetPointData().GetArray(ScalarName)
        labels = []
        for pid in range(label_array.GetNumberOfTuples()):
            labels.append(int(label_array.GetTuple(pid)[0]))
        
        labels = list(set(labels))

        for i in range(len(labels)):
            threshold = vtk.vtkThreshold()
            threshold.SetInputData(surf)
            threshold.ThresholdBetween(labels[i], labels[i])
            threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, ScalarName)
            threshold.Update()

            geometry = vtk.vtkGeometryFilter()
            geometry.SetInputData(threshold.GetOutput())
            geometry.Update()

            output = img_obj["out_dir"]+"/"+filename+"_"+str(labels[i])+".vtk"
            Write(geometry.GetOutput(), output)


        




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create RootCanal object from a segmented file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    in_group_features = parser.add_mutually_exclusive_group(required=True)
    in_group_features.add_argument('--surf', type=str, help='input teeth and roots')
    in_group_features.add_argument('--dir', type=str, help='input teeth and roots')

    scalarID = parser.add_mutually_exclusive_group(required=True)
    scalarID.add_argument('--regionID', type=bool, help='seperate based on the region ID label')
    scalarID.add_argument('--universalID', type=bool, help='seperate based on the universal ID label')

    parser.add_argument('--out_dir', type=str, help='output dir', required=True)

    args = parser.parse_args()

    main(args)
