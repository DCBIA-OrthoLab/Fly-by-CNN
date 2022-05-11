
import argparse
import os
import glob
import json
from posixpath import basename
import shutil
import numpy as np
import copy

def main(args):

    list_label_O_L = ["O-5","O-6","O-19","O-22","O-38","O-39"]
    list_label_O_U = ["O-5","O-6","O-13","O-28","O-38","O-39"]

    list_jonfile_L = []
    list_jonfile_U= []

    normpath = os.path.normpath("/".join([args.dir,'**','']))
    lower_path = os.path.join(args.out,'Lower')
    upper_path = os.path.join(args.out,'Upper')

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    if not os.path.exists(lower_path):
        os.makedirs(lower_path)
    if not os.path.exists(upper_path):
        os.makedirs(upper_path)

    if args.dir:

        for jsonfile in sorted(glob.iglob(normpath, recursive=True)):

            if os.path.isfile(jsonfile) and True in [ext in jsonfile for ext in ["_O.mrk.json"]]:
               
                dic_jsonfile_L = {}
                dic_jsonfile_U = {}

                if True in ['P'+ str(ocl) + '_' in jsonfile for ocl in list(range(1,41))]:
                    dic_jsonfile_L['path'] = jsonfile
                    dic_jsonfile_L['out'] = os.path.join(lower_path,os.path.basename(jsonfile))
                    # print(dic_jsonfile_L['path'])
                    list_jonfile_L.append(dic_jsonfile_L)
                else :
                    dic_jsonfile_U['path'] = jsonfile
                    # Pnum = os.path.basename(jsonfile).split('_')[0][1:]
                    # Int_pnum = int(Pnum)-40
                    # dic_jsonfile_U['out'] = os.path.join(upper_path,os.path.basename(jsonfile).split('_')[0][0]+ str(Int_pnum) + '_' + os.path.basename(jsonfile).split('_')[1])
                    
                    dic_jsonfile_U['out'] = os.path.join(upper_path,os.path.basename(jsonfile))
                    # print(dic_jsonfile_U['out'])
                    list_jonfile_U.append(dic_jsonfile_U)
            
        
        for file in list_jonfile_L:

            path = file['path']
            out = file['out']

            data = json.load(open(path))
            markups = data['markups']
            control_point = markups[0]['controlPoints']


            new_lst= []
            for key,element in enumerate(control_point):

                if element['label'].split('_')[1] in list_label_O_L:
                    new_lst.append(control_point[key])
        
            
            data['markups'][0]['controlPoints'] = new_lst
            with open(out,'w') as json_file:
                json.dump(data,json_file,indent=4)  
		
  
        for file in list_jonfile_U:  
            # print(list_jonfile_input_U)
            path2 = file['path']
            out2 = file['out']

            data = json.load(open(path2))
            markups = data['markups']
            control_point = markups[0]['controlPoints']

            new_lst= []
            for key,element in enumerate(control_point):

                if element['label'].split('_')[1] in list_label_O_U:
                    new_lst.append(control_point[key])
            
            data['markups'][0]['controlPoints'] = new_lst
     
            with open(out2,'w') as json_file:
                json.dump(data,json_file,indent=4)
              
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='delete some landmarks in a model 3D', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--dir', type=str, help='path input directory', required=True)
   
    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('--out', type=str, help='Output directory', required=True)
   
    args = parser.parse_args()
    
    main(args)