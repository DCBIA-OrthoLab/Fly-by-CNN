# Fly by CNN

Contributors: Juan Prieto, Maxime Dumont, Louis Bumbolo, Mathieu Leclerq, Baptiste Baquero


## What is it?
Fly by CNN is an approach to capture 2D views from 3D objects and use the generated images to train deep learning algorithms or make inference once you have a trained model. 

## How it works?
It scales the mesh to the unit sphere, then, depending on the type of sampling chosen, it captures views from these view points. There are currently 2 types of sampling, i.e., icosahedron subdivision or following the path of a spiral. 
* Icosahedron subdivision: regular subdivision of the sphere.
* Spiral: Smooth path in the sphere that generates a sequence of images. 

### Icosahedron subidivision
<!-- ![Sphere_and_plane](https://github.com/MaximeDum/fly-by-cnn/tree/master/docs/Sphere_and_plane.png?raw=true) -->

![Sphere_and_plane](./docs/Sphere_and_plane.png?raw=true)

### Spherical spiral subivision
<!-- ![Spherical_spiral](https://github.com/lbumbolo/fly-by-cnn/tree/master/docs/Spherical_spiral.gif?raw=true) -->

![Spherical_spiral](./docs/Spherical_spiral.gif?raw=true)

## Running the code

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

## Running Universal Labeling, Merging and Separating algorithm

The easiest way to use the docker container.
[Docker/README.md](https://github.com/RomainUSA/fly-by-cnn/tree/master/Docker)

Here are the command lines to run the scripts:

**Universal Labeling:**

Input: 
- Dental crown model that contains the teeth of a patient (vtk file)

Output: 
- Dental crown model with the universal labels on each tooth (vtk file)

```
python3 fly-by-cnn/src/py/universal_labeling.py --help
```

```
usage: universal_labeling.py [-h] --surf SURF --label_groundtruth
                             LABEL_GROUNDTRUTH --model_feature MODEL_FEATURE
                             --model_LU MODEL_LU --out_feature OUT_FEATURE
                             [--out OUT]

Label the teeth from 2 to 16 or with the universal IDs used by clinicians

optional arguments:
  -h, --help            show this help message and exit
  --surf SURF           Input surface mesh to label (default: None)
  --out OUT             Output model with labels (default: out.vtk)

Label parameters:
  --label_groundtruth LABEL_GROUNDTRUTH
                        directory of the template labels (default: None)

Prediction parameters:
  --model_feature MODEL_FEATURE
                        path of the VGG19 model (default: None)
  --model_LU MODEL_LU   path of the LowerUpper model (default: None)
  --out_feature OUT_FEATURE
                        out of the feature (default: None)
```

**Merging and Separating:**

Inputs: 
- Dental crown model that contains the teeth of a patient with the universal labels on each tooth (vtk file)
- The root canal segmentation file (.nii)

Output: 
- A merged model with the teeth and roots labeled based on the universal labels and individual teeth and roots (vtk files)

```
bash fly-by-cnn/src/sh/compute_MergingSeparating.sh --help
```

```
Program to run the Merging and Separating algorithm

Syntax: compute_MergingSeparating.sh [--options]
options:
--src_code                    Path of the source code 
--input_file_uid              Output directory of the teeth with the universal labels.
--input_file_root             Input directory with the root canal segmentation files.
--out_tmp                     Temporary output folder.
--out_merge                   Output directory of the merged surfaces.
--out_separate                Output directory of the separated surfaces.
```

**ULMS algortihm:**

Inputs: 
- Dental crown model that contains the teeth of a patient (vtk file)
- The root canal segmentation file (.nii)

Output: 
- A merged model with the teeth and roots labeled based on the universal labels and individual teeth and roots (vtk files)

```
bash fly-by-cnn/src/sh/compute_ULMS.sh --help
```

```
Program to run the Universal Labeling Merging and Separated algorithm

Syntax: compute_ULMS.sh [--options]
options:
--src_code                    Path of the source code 
--input_file_surf             Input file surface with only the teeth.
--label_GT_dir                Folder containing the template for the Upper/Lower classification.
--model_ft                    Path to the feature model .
--model_LU                    Path to the LowerUpper classification model.
--out_feature                 Output of the feature.
--output_dir_uid              Output directory of the teeth with the universal labels.
--input_file_root             Root canal segmentation file.
--out_tmp                     Temporary output folder.
--out_merge                   Output directory of the merged surfaces.
--out_separate                Output directory of the separated surfaces.
```

