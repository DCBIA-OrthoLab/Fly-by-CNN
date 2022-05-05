import argparse
import os
import glob
from posixpath import basename
import vtk
import post_process
import fly_by_features as fbf
import numpy as np
import pandas as pd
import json

def main(args):
    # model_normpath = os.path.normpath("/".join([args.model_dir,'**','']))
    landmarks_normpath = os.path.normpath("/".join([args.landmarks_dir,'**','']))
    outdir = args.out

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    list_jonfile_L = []
    list_jonfile_U= []
    # list_model_L = []
    # list_model_U = []
    # list_patients = []
    
    if args.landmarks_dir:
        for jsonfile in sorted(glob.iglob(landmarks_normpath, recursive=True)):
            if os.path.isfile(jsonfile) and True in [ext in jsonfile for ext in [".mrk.json"]]:
                if True in ['P'+ str(ocl) + '_' in jsonfile for ocl in list(range(1,41))]:
                    list_jonfile_L.append(jsonfile)
                else :
                    list_jonfile_U.append(jsonfile)
    
    # if args.model_dir:    
    #     for model in sorted(glob.iglob(model_normpath, recursive=True)):
    #         if os.path.isfile(model) and True in [ext in model for ext in [".vtk"]]:
    #             if True in ['lower_P'+ str(ocl) + '_scan_lower_RCSeg_merged' in model for ocl in list(range(1,41))]:
    #                 list_model_L.append(model)                    
    #             else :
    #                 list_model_U.append(model)

    

    if len(list_jonfile_L) != len(list_jonfile_U) :
        # or len(list_jonfile_U)!= len(list_model_L) or len(list_model_L)!= len(list_model_U):
        raise Exception("files have not the same length")

    # for obj in range(0,len(list_jonfile_L)):
    #     list_patients.append({'path_model_U':list_model_U[obj],
    #                           'path_model_L':list_model_L[obj],
    #                           'path_landmarks_U':list_jonfile_U[obj],
    #                           'path_landmarks_L':list_jonfile_L[obj] 
    #                          })
    

    # print(dico_teeth)

    
    teeth18=['O-1','O-2','O-3','O_1-1','O_1-2']
    teeth19=['O-4','O-5','O-6','O_1-3','O_1-4']
    teeth20=['O-7','O-8','O-9']
    teeth21=['O-10','O-11','O-12']
    teeth22=['O-13','O-14','O-15']
    teeth23=['O-16','O-17','O-18']
    teeth24=['O-19','O-20','O-21']
    teeth25=['O-22','O-23','O-24']
    teeth26=['O-25','O-26','O-27']
    teeth27=['O-28','O-29','O-30']
    teeth28=['O-31','O-32','O-33']
    teeth29=['O-34','O-35','O-36']
    teeth30=['O-37','O-38','O-39','0_1-7','O_1-8']
    teeth31=['O-40','O-41','O-42','0_1-5','O_1-6']
    low_teeth = [teeth18,teeth19,teeth20,teeth21,teeth22,teeth23,teeth24,teeth25,teeth26,teeth27,teeth28,teeth29,teeth30,teeth31]

    teeth1=['O-40','O-41','O-42','O_1-5','O_1-6']
    teeth2=['O-37','O-38','O-39','O-1-7','O_1-8']
    teeth3=['O-34','O-35','O-36']
    teeth4=['O-31','O-32','O-33']
    teeth5=['O-28','O-29','O-30']
    teeth6=['O-25','O-26','O-27']
    teeth7=['O-22','O-23','O-24']
    teeth8=['O-19','O-20','O-21']
    teeth9=['O-16','O-17','O-18']
    teeth10=['O-13','O-14','O-15']
    teeth11=['O-10','O-11','O-12']
    teeth12=['O-7','O-8','O-9']
    teeth13=['O-4','O-5','O-6','O_1-3','O_1-4']
    teeth14=['O-1','O-2','O-3','O_1-1','O_1-2']
    up_teeth = [teeth1,teeth2,teeth3,teeth4,teeth5,teeth6,teeth7,teeth8,teeth9,teeth10,teeth11,teeth12,teeth13,teeth14]
    
    for file in list_jonfile_L:
       
        dico_teeth = {}
        for i in range(len(low_teeth)):
            dico_teeth[str(i)] = []

        data = json.load(open(file))
        markups = data['markups']
        control_point = markups[0]['controlPoints']
        # print(file)
        for key,element in enumerate(control_point):
            for index,teeth in enumerate(low_teeth):
                if element['label'].split('_')[1] in teeth:
                    dico_teeth[str(index)].append(element)

    # print(dico_teeth['0'])
        
            for num_teeth in range(len(low_teeth)):
                data['markups'][0]['controlPoints'] = dico_teeth[str(num_teeth)]
                outfile_lower = os.path.join(outdir,os.path.basename(file).split('.')[0]) + f"_teeth_{num_teeth}.json"
                with open(outfile_lower,'w') as json_file:
                    json.dump(data,json_file,indent=4)  

    for file in list_jonfile_U:
       
        dico_teeth = {}
        for i in range(len(up_teeth)):
            dico_teeth[str(i)] = []

        data = json.load(open(file))
        markups = data['markups']
        control_point = markups[0]['controlPoints']
        # print(file)
        for key,element in enumerate(control_point):
            for index,teeth in enumerate(low_teeth):
                if element['label'].split('_')[1] in teeth:
                    dico_teeth[str(index)].append(element)

    # print(dico_teeth['0'])
        
            for num_teeth in range(len(up_teeth)):
                data['markups'][0]['controlPoints'] = dico_teeth[str(num_teeth)]
                outfile_upper = os.path.join(outdir,os.path.basename(file).split('.')[0]) + f"_teeth_{num_teeth}.json"
                with open(outfile_upper,'w') as json_file:
                    json.dump(data,json_file,indent=4)       

    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='separete all the teeth from a vtk file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    parser.add_argument('--landmarks_dir', type=str, help='landmarks directory', required=True)
    parser.add_argument('--model_dir', type=str, help='model file directory', required=True)
   
    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('--out', type=str, help='Output directory', required=True)
   
    args = parser.parse_args()
    
    main(args)