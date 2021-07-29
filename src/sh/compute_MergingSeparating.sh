#!/bin/sh

Help()
{
# Display Help
echo "Program to run the Merging and Separating algorithm"
echo
echo "Syntax: compute_MergingSeparating.sh [--options]"
echo "options:"
echo "--src_code                    Path of the source code "
echo "--input_file_uid              Output directory of the teeth with the universal labels."
echo "--input_file_root             Input directory with the root canal segmentation files."
echo "--out_tmp                     Temporary output folder."
echo "--out_merge                   Output directory of the merged surfaces."
echo "--out_separate                Output directory of the separated surfaces."
}

while [ "$1" != "" ]; do
    case $1 in
        --src_code )  shift
            src_code=$1;;
        --input_file_uid )  shift
            input_file_uid=$1;;
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


out_tmp="${out_tmp:-/app/data/out_tmp}"
out_merge="${out_merge:-/app/data/out}"


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

python3 $src_code/py/PSCP/create_3D_RC.py --dir $out_tmp/$(basename $output)  --out $out_tmp/$(basename $output)
python3 $src_code/py/PSCP/merge.py --surf $input_file_uid --dir_root $out_tmp/$filename --label_name $name_property --out $output_filename


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


