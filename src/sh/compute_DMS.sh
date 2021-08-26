#!/bin/bash


input_dir=$1
model=$2
output_dir=$3
dilate=$4



files=($input_dir/*)


for file in "${files[@]}"; do
    filename=$(basename $file)
    filename="${filename%.*}"
    mkdir $output_dir/$filename/
    python3 src/py/predict_v3.py --surf $file --dilate $dilate --model $model --out $output_dir/$filename/$filename.vtk
done





























