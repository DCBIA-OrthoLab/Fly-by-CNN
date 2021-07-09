import vtk
import itk
import argparse
import glob
import os
import shutil

import numpy as np




def main(args):
    img_fn_array = []

    if args.nii:
        img_obj = {}
        img_obj["img"] = args.nii
        img_obj["out"] = args.out
        img_fn_array.append(img_obj)

    elif args.dir:
        normpath = os.path.normpath("/".join([args.dir, '**', '*']))
        for img_fn in glob.iglob(normpath, recursive=True):
            if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM"]]:
                img_obj = {}
                img_obj["img"] = img_fn
                img_obj["out"] = os.path.normpath("/".join([args.out]))
                img_fn_array.append(img_obj)

    
    for img_obj in img_fn_array:
        image = img_obj["img"]
        out = img_obj["out"]
        print("Reading:", image)

        ImageType = itk.Image[itk.US, 3]
        
        reader = itk.ImageFileReader[ImageType].New(FileName=image)
        reader.Update()
        img = reader.GetOutput()

        new_filepath = image
        if ".gz" in image:
            new_filepath = new_filepath.replace('.gz','')
        if ".nii" in image:
            new_filepath = new_filepath.replace('.nii','.nrrd')

        writer = itk.ImageFileWriter[ImageType].New(FileName=new_filepath, Input=img)
        writer.Update()
        os.remove(image)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create RootCanal object from a segmented file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    in_group_features = parser.add_mutually_exclusive_group(required=True)
    in_group_features.add_argument('--nii', type=str, help='input file')
    in_group_features.add_argument('--dir', type=str, help='input dir')

    parser.add_argument('--out', type=str, help='output dir', default='')

    args = parser.parse_args()

    main(args)