#!/bin/sh


input_dir_json=$1
output_dir_vtk_landmarks=$2



python3 src/py/Landmarks/json2vtk.py --dir $input_dir_json --out $output_dir_vtk_landmarks


python3 src/py/preprocess_landmarks.py --landmarks_dir $--teeth_dir --n_rotations --random_rotation --out_features --out_labels



























