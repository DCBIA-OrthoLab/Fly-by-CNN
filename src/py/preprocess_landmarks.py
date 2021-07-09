import sys
import numpy as np
import time
import itk
import argparse
import glob
import os
import fly_by_features as fbf
import tensorflow as tf
import vtk

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
    L_landmarks_paths = []
    L_merged_paths = []

    if args.landmarks_dir:
        normpath = os.path.normpath("/".join([args.landmarks_dir, '**', '*']))
        for img_fn in glob.iglob(normpath, recursive=True):
            if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".vtk"]]:
                L_landmarks_paths.append(img_fn)

    if args.merged_dir:
        normpath = os.path.normpath("/".join([args.merged_dir, '**', '*']))
        for img_fn in glob.iglob(normpath, recursive=True):
            if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".vtk"]]:
                L_merged_paths.append(img_fn)


    for LM_path, merged_path in zip(sorted(L_landmarks_paths), sorted(L_merged_paths)):
        for i in range (args.n_rotations):
            surf_LM = ReadSurf(LM_path)
            surf_merged = ReadSurf(merged_path)

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

            Feature_filename, _ = os.path.splitext(os.path.basename(merged_path))
            output_filename_feature = os.path.join(args.out_features, Feature_filename+n_rot+".nrrd")
            print("Writing:", output_filename_feature)
            writer = itk.ImageFileWriter.New(FileName=output_filename_feature, Input=out_img_merged)
            writer.UseCompressionOn()
            writer.Update()

            Label_filename, _ = os.path.splitext(os.path.basename(LM_path))
            output_filename_label = os.path.join(args.out_labels, Label_filename+n_rot+".nrrd")
            print("Writing:", output_filename_label)
            writer = itk.ImageFileWriter.New(FileName=output_filename_label, Input=out_img_LM)
            writer.UseCompressionOn()
            writer.Update()
        


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Pre-processing the landmarks and the mesh + apply same rotations', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--landmarks_dir', type=str, help='landmarks directory', required=True)
    parser.add_argument('--merged_dir', type=str, help='merged file directory', required=True)

    data_augment_parser = parser.add_argument_group('Data augment parameters')
    data_augment_parser.add_argument('--n_rotations', type=int, help='Number of random rotations', default=1)
    data_augment_parser.add_argument('--random_rotation', type=bool, help='activate or not a random rotation')

    param_parser = parser.add_argument_group('Parameters')
    param_parser.add_argument('--rescale_features', type=int, help='1 to rescale features (Normals, Depth map) between 0 and 1', default = 1)
    
    parser.add_argument('--out_features', type=str, help='output features', default="out_f.vtk")
    parser.add_argument('--out_labels', type=str, help='output labels', default="out_l.vtk")

    args = parser.parse_args()

    main(args)