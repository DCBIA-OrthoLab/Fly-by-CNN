import vtk
import numpy as np 
import os
import glob
import argparse
import utils


def main(args):
    shapes_arr = []
    if(args.surf):
        shapes_arr.append(args.surf)    

    if(args.dir):
        shapes_dir = os.path.join(os.path.dirname(args.dir), '**/*.vtk')
        for svtk in glob.iglob(shapes_dir, recursive=True):
            shapes_arr.append(svtk)
    
    scale_factors = []
    max_value=-1
    for vtkfilename in shapes_arr:
        scale_factor = -1

        surf = utils.ReadSurf(vtkfilename)
        surf, mean_arr, scale_factor = utils.ScaleSurf(surf)

        if(scale_factor>max_value):
            max_value=scale_factor

        print("Scale factor: "+str(scale_factor))
        
        scale_factors.append(scale_factor)


    with open(args.out, "w") as f:
        for scale_factor in scale_factors:
            f.write(str(scale_factor)+"\n")

    print("Max scale factor: "+str(max_value))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes maximum magnitude/scaling factor using bounding box and appends to file')
    #TODO: MAKE AT LEAST ONE OF THE DIRECTRY OR SURF REQUIRED
    parser.add_argument('--surf', type=str, default=None, help='Target surface or mesh')
    parser.add_argument('--out', type=str, default="scale_factor.txt", help='Output filename')
    parser.add_argument('--dir', type=str, default=None, help='Directory with vtk files')

    args = parser.parse_args()


    main(args)
