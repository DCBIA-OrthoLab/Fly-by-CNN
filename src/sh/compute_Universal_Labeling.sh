#!/bin/sh


input_dir=$1
label_GT=$2
model_ft=$3
model_LU=$4
out_ft=$5
output_dir=$6



files=($input_dir/*)


for file in "${files[@]}"; do
    filename=$(basename $file)
    filename="${filename%.*}"
    python3 src/py/labeling.py --surf $file --label_groundtruth $label_GT --model_feature $model_ft --model_LU $model_LU --out_feature $out_ft --out $output_dir/$filename"_uid.vtk"
done