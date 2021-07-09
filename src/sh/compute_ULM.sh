#!/bin/sh

Help()
{
# Display Help
echo "Program to run the Universal Labeling and Merging algorithm"
echo
echo "Syntax: compute_ULM.sh [--options]"
echo "options:"
echo "--input_dir_surf              Input directory where there are the surfaces with only the teeth."
echo "--label_GT_dir                Folder containing the template for the Upper/Lower classification."
echo "--model_ft                    Path to the feature model ."
echo "--model_LU                    Path to the LowerUpper classification model."
echo "--out_ft                      Output of the fearure."
echo "--output_dir_uid              Output directory of the teeth with the universal labels."
echo "--input_dir_root              Input directory with the root canal segmentation files."
echo "--out_tmp                     Temporary output folder."
echo "--out_merge                   Output directory of the merged surfaces."
echo "--out_separate                Output directory of the separated surfaces."
}

while [ "$1" != "" ]; do
    case $1 in
        --input_dir_surf )  shift
            input_dir_surf=$1;;
        --label_GT_dir )  shift
            label_GT_dir=$1;;
        --model_ft )  shift
            model_ft=$1;;
        --model_LU )  shift
            model_LU=$1;;
        --out_ft )  shift
            out_ft=$1;;
        --output_dir_uid )  shift
            output_dir_uid=$1;;
        --input_dir_root )  shift
            input_dir_root=$1;;
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


echo "==================================="
echo 
echo "Universal Labeling"
echo
echo "==================================="

files=($input_dir_surf/*)

for file in "${files[@]}"; do
    filename=$(basename $file)
    filename="${filename%.*}"

    python3 src/py/universal_labeling.py --surf $file --label_groundtruth $label_GT_dir --model_feature $model_ft --model_LU $model_LU --out_feature $out_ft --out $output_dir_uid/$filename"_uid.vtk"
done


echo "==================================="
echo 
echo "Transforming itk files to vtk files..."
echo
echo "==================================="

python3 src/py/PSCP/create_RC_object.py --dir $input_dir_root --out $out_tmp
python3 src/py/PSCP/nii2nrrd.py --dir $out_tmp --out $out_tmp


echo "==================================="
echo 
echo "Merging files..."
echo
echo "==================================="

dir_roots=($input_dir_root/*)
dir_surf=($output_dir_uid/*)
universalID=1

for root in "${dir_roots[@]}"; do
    for surf in "${dir_surf[@]}"; do
        surf_name=$(basename $surf)
        split=(${surf_name//_/ })
        surf_patient=${split[0]}  

        root_name=$(basename $root)
        split=(${root_name//_/ })
        root_patient=${split[1]}  

        if [ "$surf_patient" = "$root_patient" ]; then
            surf_path=$surf
        fi
    done

    root_filename=$(basename $root)
    root_filename="${root_filename%.*}"
    root_filename="${root_filename%.*}"
    dirname_root=$out_tmp/$root_filename
    output_filename=$out_merge/$(basename $dirname_root)_merged.vtk

    python3 src/py/PSCP/create_3D_RC.py --dir $out_tmp/$(basename $dirname_root) --out $out_tmp/$(basename $dirname_root)
    python3 src/py/PSCP/merge.py --surf $surf_path --dir_root $dirname_root --label_name $universalID --out $output_filename
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

    python3 src/py/PSCP/separate.py --surf $file --universalID $universalID --out $out_separate/$filename
done


