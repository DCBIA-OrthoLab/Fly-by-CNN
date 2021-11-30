import argparse
import os
import glob
from posixpath import basename
import shutil
import vtk
import post_process
import fly_by_features as fbf
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import json
import pandas as pd
from math import *
from utils import *



def main(args):  
    
    model_normpath = os.path.normpath("/".join([args.model_dir,'**','']))
    landmarks_normpath = os.path.normpath("/".join([args.landmarks_dir,'**','']))

    lenght = 41
    # radius = 0.5

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    dic_patient = {}


    if args.landmarks_dir:
        for jsonfile in sorted(glob.iglob(landmarks_normpath, recursive=True)):
            if os.path.isfile(jsonfile) and True in [ext in jsonfile for ext in [".json"]]:
                num = os.path.basename(jsonfile).split('_')[0][1:]
                if True in ['P'+ str(ocl) + '_' in jsonfile for ocl in list(range(1,lenght))]:
                    # list_jonfile_L.append(jsonfile)
                    if num in dic_patient.keys():
                        dic_patient[num]['path_landmarks_L'] = jsonfile
                    else:
                        dic_patient[num] = {'path_landmarks_L' : jsonfile}
                else :
                    if str(int(num)-lenght+1) in dic_patient.keys():
                        dic_patient[str(int(num)-lenght+1)]['path_landmarks_U'] = jsonfile                    
                    else:
                        dic_patient[str(int(num)-lenght+1)] = {'path_landmarks_U' : jsonfile}
                   

    if args.model_dir:    
        for model in sorted(glob.iglob(model_normpath, recursive=True)):
            if os.path.isfile(model) and True in [ext in model for ext in [".vtk"]]:
                num = os.path.basename(model).split('.')[0][1:]
                if True in ['P'+ str(ocl) +'.' in model for ocl in list(range(1,lenght))]:
                    if num in dic_patient.keys():
                        dic_patient[num]['path_model_L'] = model
                    else:
                        dic_patient[num] = {'path_model_L' : model}
                else :
                    if str(int(num)-lenght+1) in dic_patient.keys():
                        dic_patient[str(int(num)-lenght+1)]['path_model_U'] = model                    
                    else:
                        dic_patient[str(int(num)-lenght+1)] = {'path_model_U' : model}

    # print(dic_patient['path_model_L'])
    for obj in dic_patient.items():
        obj=obj[1]
        
        # surf_u = ReadSurf(obj["path_model_U"])
        # surf_l = ReadSurf(obj["path_model_L"])

        # real_labels_u = surf_u.GetPointData().GetArray("UniversalID")
        # surf_u.GetPointData().SetActiveScalars("UniversalID")
        # real_labels_l = surf_l.GetPointData().GetArray("UniversalID")
        # surf_l.GetPointData().SetActiveScalars("UniversalID")
        # # print(real_labels_u)
        # real_labels_np_u = vtk_to_numpy(real_labels_u)
        # real_labels_np_l = vtk_to_numpy(real_labels_l)
        patient_id = os.path.basename(obj["path_model_L"]).split(".")[0]
        outdir_patient_l = os.path.join(args.out,patient_id)
        outdir_u = os.path.join(outdir_patient_l,"Upper")
        outdir_l = os.path.join(outdir_patient_l,"Lower")

        if not os.path.exists(outdir_patient_l):
            os.makedirs(outdir_patient_l)
        if not os.path.exists(outdir_u):
            os.makedirs(outdir_u)
        if not os.path.exists(outdir_l):
            os.makedirs(outdir_l)

        data_u = json.load(open(obj["path_landmarks_U"]))
        json_file = pd.read_json(obj["path_landmarks_U"])
        json_file.head()
        markups = json_file.loc[0,'markups']
        controlPoints = markups['controlPoints']
        number_landmarks = len(controlPoints)
        new_lst = []

        for i in range(number_landmarks):
            label = controlPoints[i]["label"]
            # controlPoints[i]["label"] = "Upper_"+'_'.join(label.split("_")[1:])
            controlPoints[i]["label"] = "Upper_"+label.split("_")[-1]
            new_lst.append(controlPoints[i])
        
        data_u['markups'][0]['controlPoints'] = new_lst
        
        with open(os.path.join(outdir_u,f'Upper_{patient_id}.json'),'w') as json_file:
            json.dump(data_u,json_file,indent=4)  

        data_l = json.load(open(obj["path_landmarks_L"]))
        json_file = pd.read_json(obj["path_landmarks_L"])
        json_file.head()
        markups = json_file.loc[0,'markups']
        controlPoints = markups['controlPoints']
        number_landmarks = len(controlPoints)
        new_lst = []

        for i in range(number_landmarks):
            label = controlPoints[i]["label"]
            # controlPoints[i]["label"] = "Upper_"+'_'.join(label.split("_")[1:])
            controlPoints[i]["label"] = "Lower_"+label.split("_")[-1]
            new_lst.append(controlPoints[i])

        data_l['markups'][0]['controlPoints'] = new_lst
       
        with open(os.path.join(outdir_l,f'Lower_{patient_id}.json'),'w') as json_file:
            json.dump(data_l,json_file,indent=4)
        

        shutil.copy(obj['path_model_L'],outdir_l)
        dst_file_L = os.path.join(outdir_l,os.path.basename(obj['path_model_L']))
        new_dst_file_name_L = os.path.join(outdir_l, f'Lower_{patient_id}.vtk')
        os.rename(dst_file_L, new_dst_file_name_L)#rename
        shutil.copy(obj['path_model_U'],outdir_u)
        dst_file_U = os.path.join(outdir_u,os.path.basename(obj['path_model_U']))
        new_dst_file_name_U = os.path.join(outdir_u, f'Upper_{patient_id}.vtk')
        os.rename(dst_file_U, new_dst_file_name_U)#rename

###################################################################################################################################
#                                                   FOR UPPER JAW                                                                 #
###################################################################################################################################



    #     for teeth in range(np.min(real_labels_np_u),np.max(real_labels_np_u)+1):
    #         if teeth>1 and teeth<16:
    #             outdir_teet_u = os.path.join(outdir_u,f"teeth_{teeth}")
    #             new_num = str(int(os.path.basename(obj["path_model_U"]).split('_')[0][1:])-40)
    #             outfilename_teeth_u = os.path.join(outdir_teet_u,os.path.basename(obj["path_model_U"]).split('_')[0][0]+ new_num + "_Upper_" + os.path.basename(obj["path_model_U"]).split('_')[1] + f"_{teeth}_" + os.path.basename(obj["path_model_U"]).split('_')[2])
    #             teeth_surf = post_process.Threshold(surf_u, "UniversalID", teeth, teeth )
    #             if teeth_surf.GetNumberOfPoints() > 0:
    #                 if not os.path.exists(outdir_teet_u):
    #                     os.makedirs(outdir_teet_u)
                    
    #                 print("Writting:", outfilename_teeth_u)
    #                 polydatawriter = vtk.vtkPolyDataWriter()
    #                 polydatawriter.SetFileName(outfilename_teeth_u)
    #                 polydatawriter.SetInputData(teeth_surf)
    #                 polydatawriter.Write()
                
    #                 # print(obj["path_landmarks_U"])
    #                 data_u = json.load(open(obj["path_landmarks_U"]))
    #                 json_file = pd.read_json(obj["path_landmarks_U"])
    #                 json_file.head()
    #                 markups = json_file.loc[0,'markups']
    #                 controlPoints = markups['controlPoints']
    #                 number_landmarks = len(controlPoints)
        
    #                 locator = vtk.vtkIncrementalOctreePointLocator()
    #                 locator.SetDataSet(teeth_surf) 
    #                 locator.BuildLocator()
                    
    #                 new_lst= []
                    
    #                 for i in range(number_landmarks):
    #                     label = controlPoints[i]["label"]
    #                     # controlPoints[i]["label"] = "Upper_"+'_'.join(label.split("_")[1:])
    #                     controlPoints[i]["label"] = "Upper_"+label.split("_")[-1]

    #                     position = controlPoints[i]["position"]
    #                     pid = locator.FindClosestPoint(position)
    #                     point = teeth_surf.GetPoint(pid)
    #                     distance = sqrt((list(point)[0]-position[0])**2+(list(point)[1]-position[1])**2+(list(point)[2]-position[2])**2)
    #                     # print(distance)
    #                     if distance < 0.5:
    #                         new_lst.append(controlPoints[i])
                    
    #                 data_u['markups'][0]['controlPoints'] = new_lst
                    
    #                 filename = os.path.join(outdir_teet_u,os.path.basename(obj["path_landmarks_U"]).split('_')[0][0]+ new_num + f"_Upper_teeth_{teeth}_landmarks_" + os.path.basename(obj["path_landmarks_U"]).split('_')[1])
    #                 # filename = basename + "_landmarks.vtk"
    #                 output_LM_U_path = os.path.join(outdir_teet_u, filename)
    #                 with open(output_LM_U_path,'w') as json_file:
    #                     json.dump(data_u,json_file,indent=4)  

    #                 # print(len(new_lst))
    #                 # vtk_landmarks = vtk.vtkAppendPolyData()
    #                 # for cp in new_lst:
    #                 #     # Create a sphere
    #                 #     sphereSource = vtk.vtkSphereSource()
    #                 #     sphereSource.SetCenter(cp["position"][0],cp["position"][1],cp["position"][2])
    #                 #     sphereSource.SetRadius(radius)

    #                 #     # Make the surface smooth.
    #                 #     sphereSource.SetPhiResolution(100)
    #                 #     sphereSource.SetThetaResolution(100)
    #                 #     sphereSource.Update()
                        
    #                 #     vtk_landmarks.AddInputData(sphereSource.GetOutput())
    #                 #     vtk_landmarks.Update()
                    
    #                 # basename = os.path.basename(obj["path_landmarks_U"]).split(".")[0]
    #                 # filename = basename + "_landmarks.vtk"
    #                 # output_LM_U_path = os.path.join(outdir_teet_u, filename)
    #                 # Write(vtk_landmarks.GetOutput(), output_LM_U_path)

            
    # ##################################################################################################################################
    #                                                 #   FOR LOWER JAW                                                                 #
    # ##################################################################################################################################

    #     for teeth in range(np.min(real_labels_np_l),np.max(real_labels_np_l)+1):
    #         if teeth>17 and teeth<32:
    #             # if teeth
    #             #     teeth+1
    #             outdir_teet_l = os.path.join(outdir_l,f"teeth_{teeth}")
    #             outfilename_teeth_l = os.path.join(outdir_teet_l,os.path.basename(obj["path_model_L"]).split('_')[0] + "_Lower_" + os.path.basename(obj["path_model_L"]).split('_')[1] + f"_{teeth}_" + os.path.basename(obj["path_model_L"]).split('_')[2])

    #             teeth_surf_l = post_process.Threshold(surf_l, "UniversalID", teeth, teeth )
    #             if teeth_surf_l.GetNumberOfPoints() > 0:
    #                 if not os.path.exists(outdir_teet_l):
    #                     os.makedirs(outdir_teet_l)
                    
    #                 print("Writting:", outfilename_teeth_l)
    #                 polydatawriter = vtk.vtkPolyDataWriter()
    #                 polydatawriter.SetFileName(outfilename_teeth_l)
    #                 polydatawriter.SetInputData(teeth_surf_l)
    #                 polydatawriter.Write()

    #                 data_l = json.load(open(obj["path_landmarks_L"]))
    #                 json_file = pd.read_json(obj["path_landmarks_L"])
    #                 json_file.head()
    #                 markups = json_file.loc[0,'markups']
    #                 controlPoints = markups['controlPoints']
    #                 number_landmarks = len(controlPoints)

    #                 locator_l = vtk.vtkIncrementalOctreePointLocator()
    #                 locator_l.SetDataSet(teeth_surf_l) 
    #                 locator_l.BuildLocator()
                    
    #                 new_lst= []
                    
    #                 for i in range(number_landmarks):
    #                     label = controlPoints[i]["label"]
    #                     # controlPoints[i]["label"] = "Lower_"+'_'.join(label.split("_")[1:])
    #                     controlPoints[i]["label"] = "Lower_"+label.split("_")[-1]


    #                     position = controlPoints[i]["position"]
    #                     pid = locator_l.FindClosestPoint(position)
    #                     point = teeth_surf_l.GetPoint(pid)
    #                     distance = sqrt((list(point)[0]-position[0])**2+(list(point)[1]-position[1])**2+(list(point)[2]-position[2])**2)
    #                     # print(distance)
    #                     if distance < 0.5:
    #                         new_lst.append(controlPoints[i])

    #                 data_l['markups'][0]['controlPoints'] = new_lst
                    
    #                 filename = os.path.join(outdir_teet_l,os.path.basename(obj["path_landmarks_L"]).split('_')[0] + f"_Lower_teeth_{teeth}_landmarks_" + os.path.basename(obj["path_landmarks_L"]).split('_')[1])
    #                 # filename = basename + "_landmarks.vtk"
    #                 output_LM_L_path = os.path.join(outdir_teet_l, filename)
    #                 with open(output_LM_L_path,'w') as json_file:
    #                     json.dump(data_l,json_file,indent=4) 

    #                 # # print(len(new_lst))
    #                 # vtk_landmarks = vtk.vtkAppendPolyData()
    #                 # for cp in new_lst:

    #                 #     # Create a sphere
    #                 #     sphereSource = vtk.vtkSphereSource()
    #                 #     sphereSource.SetCenter(cp["position"][0],cp["position"][1],cp["position"][2])
    #                 #     sphereSource.SetRadius(radius)

    #                 #     # Make the surface smooth.
    #                 #     sphereSource.SetPhiResolution(100)
    #                 #     sphereSource.SetThetaResolution(100)
    #                 #     sphereSource.Update()
                        
    #                 #     vtk_landmarks.AddInputData(sphereSource.GetOutput())
    #                 #     vtk_landmarks.Update()
                    
    #                 # basename = os.path.basename(obj["path_landmarks_L"]).split(".")[0]
    #                 # filename = basename + "_landmarks.vtk"
    #                 # output_LM_L_path = os.path.join(outdir_teet_l, filename)
    #                 # Write(vtk_landmarks.GetOutput(), output_LM_L_path)
                
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='separAte all the teeth from a vtk file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    # input_param.add_argument('--dir_project', type=str, help='Directory with all the project', default='/Users/luciacev-admin/Documents/AutomatedLandmarks')
    # input_param.add_argument('--dir_data', type=str, help='Input directory with 3D images', default=parser.parse_args().dir_project+'/fly-by-cnn/data')
    input_param.add_argument('--dir_data', type=str, help='Input directory with 3D images', default='/Users/luciacev-admin/Desktop/data_O')
    input_param.add_argument('--landmarks_dir', type=str, help='landmarks directory', default=parser.parse_args().dir_data+'/landmarks')
    input_param.add_argument('--model_dir', type=str, help='model file directory', default=parser.parse_args().dir_data+'/teeth_gum')

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('--out', type=str, help='Output directory', default=parser.parse_args().dir_data+'/dataset')
    
    args = parser.parse_args()
    main(args)