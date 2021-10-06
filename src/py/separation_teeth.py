import argparse
import os
import glob
from posixpath import basename
import vtk
import post_process
import fly_by_features as fbf
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import json
import pandas as pd
from utils import json2vtk
from math import *
import fly_by_features as fbf
from utils import *

def ComputeFeatures(unit_surf):
    sphere = CreateSpiral(sphereRadius=4, numberOfSpiralSamples=16)
    flyby = fbf.FlyByGenerator(sphere, resolution=256, visualize=False, use_z=True, split_z=True, rescale_features=args.rescale_features)

    surf_actor = GetNormalsActor(unit_surf)
    flyby.addActor(surf_actor)

    print("FlyBy features ...")
    img_np = flyby.getFlyBy()

    flyby_features = fbf.FlyByGenerator(sphere, 256, visualize=False)
    surf_actor = GetPointIdMapActor(unit_surf)
    flyby_features.addActor(surf_actor)
    out_point_ids_rgb_np = flyby_features.getFlyBy()

    print("Extracting:", "coords")
    out_features_np = ExtractPointFeatures(unit_surf, out_point_ids_rgb_np, "coords", 0)
    out_np = np.concatenate([img_np, out_features_np], axis=-1)
    out_img = GetImage(out_np)
        
    flyby.removeActors()
    flyby_features.removeActors()
    
    return out_img


def main(args):  
    
    model_normpath = os.path.normpath("/".join([args.model_dir,'**','']))
    landmarks_normpath = os.path.normpath("/".join([args.landmarks_dir,'**','']))
    
    list_jonfile_L = []
    list_jonfile_U= []
    list_model_L = []
    list_model_U = []
    list_patients = []
    lenght = 41
    radius = 0.5

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    if args.model_dir:
        for jsonfile in sorted(glob.iglob(landmarks_normpath, recursive=True)):
            if os.path.isfile(jsonfile) and True in [ext in jsonfile for ext in [".json"]]:
                if True in ['P'+ str(ocl) + '_' in jsonfile for ocl in list(range(1,lenght))]:
                    list_jonfile_L.append(jsonfile)
                else :
                    list_jonfile_U.append(jsonfile)   

    if args.landmarks_dir:    
         for model in sorted(glob.iglob(model_normpath, recursive=True)):
            if os.path.isfile(model) and True in [ext in model for ext in [".vtk"]]:
                if True in ['_P'+ str(ocl) + "_" in model for ocl in list(range(1,lenght))]:
                    list_model_L.append(model)                    
                else :
                    list_model_U.append(model)

    for object in range(0,len(list_jonfile_L)):
        list_patients.append({'path_model_U':list_model_U[object],
                                'path_model_L':list_model_L[object],
                                'path_landmarks_U':list_jonfile_U[object],
                                'path_landmarks_L':list_jonfile_L[object] 
                                })

    for obj in list_patients:

        surf_u = fbf.ReadSurf(obj["path_model_U"])
        surf_l = fbf.ReadSurf(obj["path_model_L"])

        real_labels_u = surf_u.GetPointData().GetArray("UniversalID")
        surf_u.GetPointData().SetActiveScalars("UniversalID")
        real_labels_l = surf_l.GetPointData().GetArray("UniversalID")
        surf_l.GetPointData().SetActiveScalars("UniversalID")

        real_labels_np_u = vtk_to_numpy(real_labels_u)
        real_labels_np_l = vtk_to_numpy(real_labels_l)
        
        outdir_patient_l = os.path.join(args.out,os.path.basename(obj["path_model_L"]).split("_")[1])
        outdir_u = os.path.join(outdir_patient_l,"Upper")
        outdir_l = os.path.join(outdir_patient_l,"Lower")

        if not os.path.exists(outdir_patient_l):
            os.makedirs(outdir_patient_l)
        if not os.path.exists(outdir_u):
            os.makedirs(outdir_u)
        if not os.path.exists(outdir_l):
            os.makedirs(outdir_l)


###################################################################################################################################
#                                                   FOR UPPER JAW                                                                 #
###################################################################################################################################



        for teeth in range(np.min(real_labels_np_u),np.max(real_labels_np_u)+1):
            
            outdir_teet_u = os.path.join(outdir_u,f"teeth_{teeth}")
            if not os.path.exists(outdir_teet_u):
                os.makedirs(outdir_teet_u)
            
            outfilename_teeth_u = os.path.join(outdir_teet_u,os.path.basename(obj["path_model_U"]))
            teeth_surf = post_process.Threshold(surf_u, "UniversalID", teeth, teeth )
            outfilename = os.path.splitext(outfilename_teeth_u)[0] + f"_teeth_{teeth}.vtk"
            # print("Writting:", outfilename)
            polydatawriter = vtk.vtkPolyDataWriter()
            polydatawriter.SetFileName(outfilename)
            polydatawriter.SetInputData(teeth_surf)
            polydatawriter.Write()
        
            # print(obj["path_landmarks_U"])
            data_u = json.load(open(obj["path_landmarks_U"]))
            json_file = pd.read_json(obj["path_landmarks_U"])
            json_file.head()
            markups = json_file.loc[0,'markups']
            controlPoints = markups['controlPoints']
            number_landmarks = len(controlPoints)
   
            locator = vtk.vtkIncrementalOctreePointLocator()
            locator.SetDataSet(teeth_surf) 
            locator.BuildLocator()
            
            new_lst= []
            
            for i in range(number_landmarks):
                position = controlPoints[i]["position"]
                pid = locator.FindClosestPoint(position)
                point = teeth_surf.GetPoint(pid)
                distance = sqrt((list(point)[0]-position[0])**2+(list(point)[1]-position[1])**2+(list(point)[2]-position[2])**2)
                # print(distance)
                if distance < 0.5:
                    new_lst.append(controlPoints[i])

            data_u['markups'][0]['controlPoints'] = new_lst

            # print(len(new_lst))
            vtk_landmarks = vtk.vtkAppendPolyData()
            for cp in new_lst:
                # Create a sphere
                sphereSource = vtk.vtkSphereSource()
                sphereSource.SetCenter(cp["position"][0],cp["position"][1],cp["position"][2])
                sphereSource.SetRadius(radius)

                # Make the surface smooth.
                sphereSource.SetPhiResolution(100)
                sphereSource.SetThetaResolution(100)
                sphereSource.Update()
                
                vtk_landmarks.AddInputData(sphereSource.GetOutput())
                vtk_landmarks.Update()
            
            basename = os.path.basename(obj["path_landmarks_U"]).split(".")[0]
            filename = basename + "_landmarks.vtk"
            output_LM_U_path = os.path.join(outdir_teet_u, filename)
            Write(vtk_landmarks.GetOutput(), output_LM_U_path)


            ####################  apply same rotation to each landmarks ########################

    
            for i in range (args.n_rotations):
                surf_LM = ReadSurf(output_LM_U_path)
                surf_merged = ReadSurf(outfilename)

                unit_surf_LM, _, scale_factor = ScaleSurf(surf_LM, mean_arr=np.array([0,0,0]))
                unit_surf_merged, _, scale_factor = ScaleSurf(surf_merged, mean_arr=np.array([0,0,0]), scale_factor=scale_factor)

                if args.random_rotation:
                    unit_surf_LM, rotationAngle, rotationVector = RandomRotation(unit_surf_LM)
                    unit_surf_merged = RotateSurf(unit_surf_merged, rotationAngle, rotationVector)

                out_img_LM = ComputeFeatures(unit_surf_LM)
                out_img_merged = ComputeFeatures(unit_surf_merged)

                if args.n_rotations == 1:
                    n_rot = ""
                else:
                    n_rot = "_rot"+str(i)

                Feature_filename = os.path.splitext(os.path.basename(outfilename))[0]
                output_filename_feature = os.path.join(outdir_teet_u,Feature_filename+n_rot+".nrrd")
                print("Writing:", output_filename_feature)
                writer = itk.ImageFileWriter.New(FileName=output_filename_feature, Input=out_img_merged)
                writer.UseCompressionOn()
                writer.Update()

                Label_filename = os.path.splitext(os.path.basename(output_LM_U_path))[0]
                output_filename_label = os.path.join(outdir_teet_u,Label_filename+n_rot+".nrrd")
                print("Writing:",output_filename_label)
                writer = itk.ImageFileWriter.New(FileName=output_filename_label, Input=out_img_LM)
                writer.UseCompressionOn()
                writer.Update()
        
###################################################################################################################################
#                                                   FOR LOWER JAW                                                                 #
###################################################################################################################################

        for teeth in range(np.min(real_labels_np_l),np.max(real_labels_np_l)+1):
            outdir_teet_l = os.path.join(outdir_l,f"teeth_{teeth}")
            if not os.path.exists(outdir_teet_l):
                os.makedirs(outdir_teet_l)
            outfilename_teeth_l = os.path.join(outdir_teet_l,os.path.basename(obj["path_model_L"]))

            teeth_surf_l = post_process.Threshold(surf_l, "UniversalID", teeth, teeth )
            outfilename_l = os.path.splitext(outfilename_teeth_l)[0] + f"_teeth_{teeth}.vtk"
            # print("Writting:", outfilename)
            polydatawriter = vtk.vtkPolyDataWriter()
            polydatawriter.SetFileName(outfilename_l)
            polydatawriter.SetInputData(teeth_surf_l)
            polydatawriter.Write()

            data_l = json.load(open(obj["path_landmarks_L"]))
            json_file = pd.read_json(obj["path_landmarks_L"])
            json_file.head()
            markups = json_file.loc[0,'markups']
            controlPoints = markups['controlPoints']
            number_landmarks = len(controlPoints)

            locator_l = vtk.vtkIncrementalOctreePointLocator()
            locator_l.SetDataSet(teeth_surf_l) 
            locator_l.BuildLocator()
            
            new_lst= []
            
            for i in range(number_landmarks):
                position = controlPoints[i]["position"]
                pid = locator_l.FindClosestPoint(position)
                point = teeth_surf_l.GetPoint(pid)
                distance = sqrt((list(point)[0]-position[0])**2+(list(point)[1]-position[1])**2+(list(point)[2]-position[2])**2)
                # print(distance)
                if distance < 0.5:
                    new_lst.append(controlPoints[i])

            data_l['markups'][0]['controlPoints'] = new_lst

            # print(len(new_lst))
            vtk_landmarks = vtk.vtkAppendPolyData()
            for cp in new_lst:
                
                # Create a sphere
                sphereSource = vtk.vtkSphereSource()
                sphereSource.SetCenter(cp["position"][0],cp["position"][1],cp["position"][2])
                sphereSource.SetRadius(radius)

                # Make the surface smooth.
                sphereSource.SetPhiResolution(100)
                sphereSource.SetThetaResolution(100)
                sphereSource.Update()
                
                vtk_landmarks.AddInputData(sphereSource.GetOutput())
                vtk_landmarks.Update()
            
            basename = os.path.basename(obj["path_landmarks_L"]).split(".")[0]
            filename = basename + "_landmarks.vtk"
            output_LM_L_path = os.path.join(outdir_teet_l, filename)
            Write(vtk_landmarks.GetOutput(), output_LM_L_path)
        
        #####################  apply same rotation to each landmarks ########################

    
            for i in range (args.n_rotations):
                surf_LM = ReadSurf(output_LM_L_path)
                surf_merged = ReadSurf(outfilename_l)

                unit_surf_LM, _, scale_factor = ScaleSurf(surf_LM, mean_arr=np.array([0,0,0]))
                unit_surf_merged, _, scale_factor = ScaleSurf(surf_merged, mean_arr=np.array([0,0,0]), scale_factor=scale_factor)

                if args.random_rotation:
                    unit_surf_LM, rotationAngle, rotationVector = RandomRotation(unit_surf_LM)
                    unit_surf_merged = RotateSurf(unit_surf_merged, rotationAngle, rotationVector)

                out_img_LM = ComputeFeatures(unit_surf_LM)
                out_img_merged = ComputeFeatures(unit_surf_merged)

                if args.n_rotations == 1:
                    n_rot = ""
                else:
                    n_rot = "_rot"+str(i)

                Feature_filename = os.path.splitext(os.path.basename(outfilename_l))[0]
                output_filename_feature = os.path.join(outdir_teet_l,Feature_filename+n_rot+".nrrd")
                print("Writing:", output_filename_feature)
                writer = itk.ImageFileWriter.New(FileName=output_filename_feature, Input=out_img_merged)
                writer.UseCompressionOn()
                writer.Update()

                Label_filename = os.path.splitext(os.path.basename(output_LM_L_path))[0]
                output_filename_label = os.path.join(outdir_teet_l,Label_filename+n_rot+".nrrd")
                print("Writing:",output_filename_label)
                writer = itk.ImageFileWriter.New(FileName=output_filename_label, Input=out_img_LM)
                writer.UseCompressionOn()
                writer.Update()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='separAte all the teeth from a vtk file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--landmarks_dir', type=str, help='landmarks directory', required=True)
    input_param.add_argument('--model_dir', type=str, help='model file directory', required=True)

    data_augment_parser = parser.add_argument_group('Data augment parameters')
    data_augment_parser.add_argument('--n_rotations', type=int, help='Number of random rotations', default=1)
    data_augment_parser.add_argument('--random_rotation', type=bool, help='activate or not a random rotation', default=True)

    param_parser = parser.add_argument_group('Parameters')
    param_parser.add_argument('--rescale_features', type=int, help='1 to rescale features (Normals, Depth map) between 0 and 1', default = 1)
   
    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('--out', type=str, help='Output directory', required=True)
    
    parser.add_argument('--out_features', type=str, help='output features', default="out_f.vtk")
    parser.add_argument('--out_labels', type=str, help='output labels', default="out_l.vtk")
    args = parser.parse_args()
    main(args)