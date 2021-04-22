# Building VTK-9.0

When building in linux use to enable offscreen rendering so you can run it remotely without X Server

```
VTK_USE_X=OFF

VTK_OPENGL_HAS_EGL=ON
```


### Build the code - You can use the already compiled versions when bulding in the NIRAL system
Here are the paths useful for the cmake configuration
```
 ITK_DIR 
 /tools/ITK/ITKv5.1.1/lib/cmake/ITK-5.1

 SlicerExecutionModel_DIR     
 /tools/devel/linux/SlicerExecutionModel/SlicerExecutionModel-build_ITKv5.1.1

 VTK_DIR                          
 /tools/VTK/VTK-9.0.1-gcc4.8.5/lib64/cmake/vtk-9.0
```


### Run the fly-by-features on one shape

#### Using an icosahedron subdivision
```
/build_path/fly_by_features 
	--surf 			PATH_IN/C13LM_aligned.vtk 
	--out 			PATH_OUT/C13LM_aligned.nrrd 
	--subdivision  	[num_subvisions] 
	--resolution   	[num_resolution] 
	--planeSpacing 	[num_planeSpacing] 
	--radius 		[radius_of_the_sphere]
	--flyByCompose 

	# To normalize the meshes use src/py/compute_max.py to get the maxMagnitude
	--maxMagnitude 12.898168980522572

	# if you want to visualize the subdivisions add
	--visualize
```

#### Using an spherical spiral subdivision
```
/build_path/fly_by_features 
	--surf 			PATH_IN/C13LM_aligned.vtk 
	--out 			PATH_OUT/C13LM_aligned.nrrd   
	--resolution 	[num_resolution] 
	--spiral 		[number_of_samples] 
	--turns 		[number_of_turns]
	--planeSpacing 	[num_planeSpacing]
	--radius 		[radius_of_the_sphere] 
	--flyByCompose

	# To normalize the meshes use src/py/compute_max.py to get the maxMagnitude
	--maxMagnitude 12.898168980522572

	# if you want to visualize the subdivisions add
	--visualize
```
