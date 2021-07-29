# Running the code

The easiest way is to use the docker container.

```
docker pull dcbia/fly-by-cnn:latest
```

```
docker run --gpus all -t -i -u $(id -u):$(id -g) --name fly_by_cnn dcbia/fly-by-cnn:latest /bin/bash
```
Once inside the container you can run to get all the available options. 
```
python /app/fly-by-cnn/src/py/fly_by_features.py --help
```

* Example:
This example will extract the normals and the depth map as a separate component and generate a single 3D image with all the views. 
```
python /app/fly-by-cnn/src/py/fly_by_features.py --surf my_surf.(vtk,.stl,.obj) --subdivision 2 --resolution 512 --out out.nrrd --radius 2 --use_z 1 --split_z 1"
```


## Running Universal Labeling, Merging and Separating

To get the docker:

```
docker pull dcbia/ulms:latest
```

To run the docker:

**For the Universal Labeling:**

Input: 
- Dental crown model that contains the teeth of a patient (vtk file)

Output: 
- Dental crown model with the universal labels on each tooth (vtk file)

```
docker run --rm -v */my/input/file*:/app/data/input/P1_teeth.vtk -v */my/output/folder*:/app/data/out ulms:latest python3 fly-by-cnn-2.1.7/src/py/universal_labeling.py --surf /app/data/input/P1_teeth.vtk --label_groundtruth /app/groundtruth --model_feature /app/models/model_features --model_LU /app/models/nnLU_model_5.hdf5 --out_feature /app/data/feature.nrrd --out /app/data/out
```

**For the Merging and Separating:**

Inputs: 
- Dental crown model that contains the teeth of a patient with the universal labels on each tooth (vtk file)
- The root canal segmentation file (.nii)

Output: 
- A merged model with the teeth and roots labeled based on the universal labels and individual teeth and roots (vtk files)

```
docker run --rm -v */my/input/file*:/app/data/input/P1_teeth_uid.vtk -v */my/input/file*:/app/data/input/lower_P1_scan_lower_RCSeg.nii.gz -v */my/output/folder*:/app/data/out ulms:latest bash fly-by-cnn-2.1.7/src/sh/compute_MergingSeparating.sh --src_code fly-by-cnn-2.1.7/src --input_file_uid /app/data/input/P1_teeth_uid.vtk --input_file_root /app/data/input/lower_P1_scan_lower_RCSeg.nii.gz --out_tmp /app/data/out_tmp --out_merge /app/data/merged --out_separate /app/data/out
```

**For the ULMS:**

Inputs: 
- Dental crown model that contains the teeth of a patient (vtk file)
- The root canal segmentation file (.nii)

Output: 
- A merged model with the teeth and roots labeled based on the universal labels and individual teeth and roots (vtk files)

```
docker run --rm -v */my/input/file*:/app/data/input/P1_teeth.vtk -v */my/input/file*:/app/data/input/lower_P1_scan_lower_RCSeg.nii.gz -v */my/output/folder*:/app/data/out ulms:latest bash fly-by-cnn-2.1.7/src/sh/compute_ULMS.sh --src_code fly-by-cnn-2.1.7/src --input_file_surf /app/data/input/P1_teeth.vtk --label_GT_dir /app/groundtruth --model_ft /app/models/model_features --model_LU /app/models/nnLU_model_5.hdf5 --out_feature /app/data/feature.nrrd --output_dir_uid /app/data/uid --input_file_root /app/data/input/lower_P1_scan_lower_RCSeg.nii.gz --out_tmp /app/data/out_tmp --out_merge /app/data/merged --out_separate /app/data/out
```
