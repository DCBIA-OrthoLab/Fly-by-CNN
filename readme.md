# Fly by CNN

Contributors: Juan Prieto, Maxime Dumont, Louis Bumbolo

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

