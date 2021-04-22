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
