import argparse
import glob
import os
import sys
from collections import namedtuple

import itk
import numpy as np
from numpy.core.records import array
import tensorflow as tf
import vtk

import predict_LU
from utils import *



def ReadFile(filename):
    inputSurface = filename
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(inputSurface)
    reader.Update()
    vtkdata = reader.GetOutput()
    label_array = vtkdata.GetPointData().GetArray('RegionId')
    return vtkdata, label_array

def ChangeLabel(vtkdata, label_array, label2change, change):
    # Set all the label 'label2change' in 'change'
    for pid in range (vtkdata.GetNumberOfPoints()):
        if int(label_array.GetTuple(pid)[0]) == label2change:
            label_array.SetTuple(pid, (change, ))
    return vtkdata, label_array

def MeanCoordinatesTeeth(surf,labels):
    nlabels, pid_labels = [], []

    for pid in range(labels.GetNumberOfTuples()):
        nlabels.append(int(labels.GetTuple(pid)[0]))
        pid_labels.append(pid)

    currentlabel = 2
    L = []

    while currentlabel != np.max(nlabels)+1:
        Lcoordinates = []
        for i in range(len(nlabels)):
            if nlabels[i]==currentlabel:
                xyzCoordinates = surf.GetPoint(pid_labels[i])
                Lcoordinates.append(xyzCoordinates)

        meantuple = np.mean(Lcoordinates,axis=0)
        L.append(meantuple)		
        currentlabel+=1			

    return L

def Alignement(surf,surf_GT):
    copy_surf = vtk.vtkPolyData()
    copy_surf.DeepCopy(surf)

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.StartByMatchingCentroidsOn()
    icp.SetSource(copy_surf)
    icp.SetTarget(surf_GT)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfLandmarks(100)
    icp.SetMaximumMeanDistance(.00001)
    icp.SetMaximumNumberOfIterations(500)
    icp.CheckMeanDistanceOn()
    icp.StartByMatchingCentroidsOn()
    icp.Update()

    lmTransform = icp.GetLandmarkTransform()
    transform = vtk.vtkTransformPolyDataFilter()
    transform.SetInputData(copy_surf)
    transform.SetTransform(lmTransform)
    transform.SetTransform(icp)
    transform.Update()

    return surf, transform.GetOutput()

def Labelize(surf,labels, Lsurf, Lsurf_GT):
    L_label, L_label_GT = [], []

    for j in range(len(Lsurf)):
        Ldist = []
        for i in range (len(Lsurf_GT)):
            Xdist = Lsurf_GT[i][0]-Lsurf[j][0]
            Ydist = Lsurf_GT[i][1]-Lsurf[j][1]
            Zdist = Lsurf_GT[i][2]-Lsurf[j][2]

            dist = np.sqrt(pow(Xdist,2)+pow(Ydist,2)+pow(Zdist,2))		

            if dist<10:
                Ldist.append([dist,i+2,j+2])

        if Ldist:
            minDist = min(Ldist)
            L_label.append(minDist[2])
            L_label_GT.append(minDist[1])	


    L_label_bias = [x+20 for x in L_label]

    bias = 0
    for i in range(len(Lsurf)):
        if i+2 not in L_label:
            print("label considered as artifact:", i+2)
            ChangeLabel(surf, labels, i+2, -2)
            # bias = 1

    for i in range(len(L_label_GT)):
        ChangeLabel(surf, labels, L_label[i], L_label_bias[i])

    for i in range(len(L_label_GT)):
        # if bias:
        # 	ChangeLabel(surf, labels, L_label_bias[i], L_label_GT[i])
        # else:
        ChangeLabel(surf, labels, L_label_bias[i], L_label_GT[i])

    ChangeLabel(surf, labels, -2, 0)

def UniversalID(surf, labels, path_surf, model_feature, model_LU, out_feature):	
    real_labels = vtk.vtkIntArray()
    real_labels.SetNumberOfComponents(1)
    real_labels.SetNumberOfTuples(surf.GetNumberOfPoints())
    real_labels.SetName("UniversalID")
    real_labels.Fill(-1)

    # Load the code & prediction model to know if it is a lower or upper scan
    split_obj = {}
    split_obj["surf"] = path_surf
    split_obj["spiral"] = 64
    split_obj["model_feature"] = model_feature
    split_obj["model_LU"] = model_LU
    split_obj["out_feature"] = out_feature

    split_args = namedtuple("Split", split_obj.keys())(*split_obj.values())
    LowerOrUpper = predict_LU.main(split_args)
    LowerOrUpper = LowerOrUpper[0][0]

    for pid in range(labels.GetNumberOfTuples()):
        if LowerOrUpper<=0.5: # Lower
            real_labels.SetTuple(pid, (int(labels.GetTuple(pid)[0])+15,))
            
        if LowerOrUpper>=0.5: # Upper
            real_labels.SetTuple(pid, (int(labels.GetTuple(pid)[0])-1,))
            
    surf.GetPointData().AddArray(real_labels)



def main(args):
    surf, labels = ReadFile(args.surf)

    print("Labelizing...")
    surf_groundtruth, labels_groundtruth = ReadFile(args.label_groundtruth)
    surf, copy_surf = Alignement(surf,surf_groundtruth)
    Lsurf = MeanCoordinatesTeeth(copy_surf,labels)
    Lsurf_GT = MeanCoordinatesTeeth(surf_groundtruth,labels_groundtruth)
    Labelize(surf,labels,Lsurf,Lsurf_GT)

    if(args.universalID):
        print("UniversalID...")
        UniversalID(surf, labels, args.surf, args.model_feature, args.model_LU, args.out_feature)

    WriteSurf(surf, args.out)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label the teeth from 2 to 16 or with the universal IDs used by clinicians', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--surf', type=str, help='Input surface mesh to label', required=True)

    labelize_parser = parser.add_argument_group('Label parameters')
    labelize_parser.add_argument('--label_groundtruth', type=str, help='groundtruth of the label', required=True)

    universalID_parser = parser.add_argument_group('Universal ID parameters')
    universalID_parser.add_argument('--universalID', type=bool, help='label the teeth with Universal ID', default=False)
    universalID_parser.add_argument('--model_feature', type=str, help='path of the VGG19 model', default='')
    universalID_parser.add_argument('--model_LU', type=str, help='path of the LowerUpper model', default='')
    universalID_parser.add_argument('--out_feature', type=str, help='out of the feature', default='')

    parser.add_argument('--out', type=str, help='Output model with labels', default="out.vtk")

    args = parser.parse_args()

    main(args)








