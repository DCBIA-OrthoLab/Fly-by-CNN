#!/bin/bash

Help()
{
# Display Help
echo "Program to run the Universal Labeling Merging and Separated algorithm"
echo
echo "Syntax: compute_ULMS.sh [--options]"
echo "options:"
echo "--src_code                    Path of the source code "
echo "--input_file_surf             Input file surface with only the teeth."
echo "--label_GT_dir                Folder containing the template for the Upper/Lower classification."
echo "--model_ft                    Path to the feature model ."
echo "--model_LU                    Path to the LowerUpper classification model."
echo "--out_feature                 Output of the feature."
echo "--output_dir_uid              Output directory of the teeth with the universal labels."
echo "--input_file_root             Root canal segmentation file."
echo "--out_tmp                     Temporary output folder."
echo "--out_merge                   Output directory of the merged surfaces."
echo "--out_separate                Output directory of the separated surfaces."
}

while [ "$1" != "" ]; do
    case $1 in
        --src_code )  shift
            src_code=$1;;
        --input_file_surf )  shift
            input_file_surf=$1;;
        --label_GT_dir )  shift
            label_GT_dir=$1;;
        --model_ft )  shift
            model_ft=$1;;
        --model_LU )  shift
            model_LU=$1;;
        --out_feature )  shift
            out_feature=$1;;
        --output_dir_uid )  shift
            output_dir_uid=$1;;
        --input_file_root )  shift
            input_file_root=$1;;
        --out_tmp )  shift
            out_tmp=$1;;
        --out_merge )  shift
            out_merge=$1;;
        --out_separate )  shift
            out_separate=$1;;
        -h | --help )
            Help
            exit;;
        * ) 
            echo ' - Error: Unsupported flag'
            Help
            exit 1
    esac
    shift
done


label_GT_dir="${label_GT_dir:-/app/groundtruth}"
model_ft="${model_ft:-/app/models/model_features}"
model_LU="${model_LU:-/app/models/nnLU_model_5.hdf5 }"
out_feature="${out_feature:-/app/data/feature.nrrd}"
output_dir_uid="${output_dir_uid:-/app/data/uid}"
out_tmp="${out_tmp:-/app/data/out_tmp}"
out_merge="${out_merge:-/app/data/out}"


echo "==================================="
echo 
echo "Universal Labeling"
echo
echo "==================================="

filename=$(basename $input_file_surf)
filename="${filename%.*}"
filename="${filename%.*}"

python3 $src_code/py/universal_labeling.py --surf $input_file_surf --label_groundtruth $label_GT_dir --model_feature $model_ft --model_LU $model_LU --out_feature $out_feature --out $output_dir_uid/$filename"_uid.vtk"


echo "==================================="
echo 
echo "Transforming itk files to vtk files..."
echo
echo "==================================="

output=$out_tmp/$(basename $input_file_root)
output="${output%.*}"
output="${output%.*}"

python3 $src_code/py/PSCP/create_RC_object.py --image $input_file_root --out $output
python3 $src_code/py/PSCP/nii2nrrd.py --dir $out_tmp --out $out_tmp


echo "==================================="
echo 
echo "Merging files..."
echo
echo "==================================="

universalID=1
name_property="UniversalID"
output_filename=$out_merge/$(basename $output)_merged.vtk
dir_surf_uid=($output_dir_uid/*)


python3 $src_code/py/PSCP/create_3D_RC.py --dir $out_tmp/$(basename $output)  --out $out_tmp/$(basename $output)

for surf_uid in "${dir_surf_uid[@]}"; do

    python3 $src_code/py/PSCP/merge.py --surf $surf_uid --dir_root $out_tmp/$(basename $output) --label_name $name_property --out $output_filename
done


echo "==================================="
echo 
echo "Separating files..."
echo
echo "==================================="

merged_files=($out_merge/*)

for file in "${merged_files[@]}"; do
    filename=$(basename $file)
    filename="${filename%.*}"
    mkdir $out_separate/$filename/

    python3 $src_code/py/PSCP/separate.py --surf $file --universalID $universalID --out $out_separate/$filename
done


