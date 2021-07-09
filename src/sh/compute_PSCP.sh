#!/bin/sh

input_dir_root=$1
input_dir_surf=$2
out_tmp=$3
out_merge=$4

dir_roots=($input_dir_root/*)
dir_surf=($input_dir_surf/*)

# 0 = RegionId // 1 = UniversalID
label_name=1


echo "==================================="
echo 
echo "Transform itk files to vtk files..."
echo
echo "==================================="

python3 src/py/PSCP/itkRT.py --dir $input_dir_root --out $out_tmp
python3 src/py/PSCP/nii2nrrd.py --dir $out_tmp --out $out_tmp

echo echo "==================================="
echo 
echo "Merging files..."
echo
echo echo "==================================="

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

    python3 src/py/PSCP/vtkRT.py --dir $out_tmp/$(basename $dirname_root) --out $out_tmp/$(basename $dirname_root)
    python3 src/py/PSCP/merge.py --surf $surf_path --dir_root $dirname_root --label_name $label_name --out $output_filename
done






