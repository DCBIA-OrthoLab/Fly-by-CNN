import pandas as pd
from icecream import ic
import os
import vtk
import json
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from tqdm import tqdm
from pathlib import Path
import sys
import utils
import argparse


def main(args):

    csv_path = args.csv
    output_dir = args.out
    df = pd.read_csv(csv_path, dtype = str)

    LUT = np.array([33,0,0,0,0,0,0,0,0,0,0,8,7,6,5,4,3,2,1,0,0,9,10,11,12,13,14,15,
                                                16,0,0,24,23,22,21,20,19,18,17,0,0,25,26,27,28,29,30,31,32])

    pbar = tqdm(range(len(df)),desc='Converting...', total=len(df))
    for idx in pbar:

        # load obj
        model = df.iloc[idx]
        surf_path = model['surf']   
        pbar.set_description(f'{surf_path}')
        label_path = model['label'] 
        split = model['split']
        reader = vtk.vtkOBJReader()
        reader.SetFileName(surf_path)
        reader.Update()
        surf = reader.GetOutput()

        # extract verts and faces
        verts = vtk_to_numpy(surf.GetPoints().GetData())
        faces = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]  

        # get labels
        with open(label_path) as f:
            json_data = json.load(f)
        vertex_labels_FDI = np.array(json_data['labels'])  # FDI World Dental Federation notation   

        # convert to universal label
        vertex_labels = LUT[vertex_labels_FDI] # UNIVERSAL NUMBERING SYSTEM

        vertex_instances = np.array(json_data['instances'])

        # convert to vtk
        vertex_labels_vtk = numpy_to_vtk(vertex_labels)
        vertex_labels_vtk.SetName("UniversalID")

        vertex_instances_vtk = numpy_to_vtk(vertex_instances)
        vertex_instances_vtk.SetName("instances")

        surf.GetPointData().AddArray(vertex_labels_vtk)
        surf.GetPointData().AddArray(vertex_instances_vtk)

        # write file 
        file_basename = Path(surf_path).stem
        out_path = f'{output_dir}/{split}/{file_basename}.vtk'

        if(not os.path.exists(os.path.dirname(out_path))):
            os.makedirs(os.path.dirname(out_path))
        
        utils.Write(surf,out_path,print_out=False)

    for split in df['split'].unique():        
        surf = []
        for root, dirs, files in os.walk(f'{output_dir}/{split}'):
            for f in files:
                if os.path.splitext(f)[1] == '.vtk':
                    surf.append(os.path.join(root, f))

        df_split = {'surf': surf}

        df_split = pd.DataFrame(df_split)

        df_split.to_csv(os.path.join(output_dir, f'{split}.csv'), index=False)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge convert to VTK')
    parser.add_argument('--csv', help='CSV with columns surf,label,split', type=str, required=True)        
    parser.add_argument('--out', help='Output directory', type=str, default="./")

    args = parser.parse_args()

    main(args)