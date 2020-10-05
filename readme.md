# Fly by CNN

Contributors: Juan Prieto, Maxime Dumont, Louis Bumbolo

## What is it?
Fly by CNN is a C++ code that takes a 3D mesh and create 2D images of this one following the unit sphere and the number of subdivisions associated. The 2D images created contained the mesh features and labels.

## How it works?
It normalizes the mesh and create the unit sphere around this one. Then it can use different ways of running through the unit sphere:
* It subdivides the sphere in a certain number of regular points following the number of subdivisions, it is the icosahedron approach.
* It creates a spherical spiral around the unit sphere  with a certain number of points following the spiral. 

A tangent oriented plan is then created with a certain number of points. It then projects the mesh in this plan getting the associated features and label. The images are saved and then it creates another tangent plane centered on the following sphere point.

### Icosahedron subidivision
<!-- ![Sphere_and_plane](https://github.com/MaximeDum/fly-by-cnn/tree/master/docs/Sphere_and_plane.png?raw=true) -->

![Sphere_and_plane](./docs/Sphere_and_plane.png?raw=true)

### Spherical spiral subivision
<!-- ![Spherical_spiral](https://github.com/lbumbolo/fly-by-cnn/tree/master/docs/Spherical_spiral.gif?raw=true) -->

![Spherical_spiral](./docs/Spherical_spiral.gif?raw=true)

## Running the code
To run the Fly-by-CNN, you mostly need to follow the following explanations but we will also explain here how to train and evaluate a model with the fly-by-cnn datas created.

### Building VTK-9.0

When building in linux use to enable offscreen rendering!

```
VTK_USE_X=OFF

VTK_OPENGL_HAS_EGL=ON
```


### Build the code - You can use the already compiled versions when bulding in the system
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


### Run on a big dataset
To run this code on a big dataset, you need to create a bash script to run the ./fly_by_features 
Here is an example of the bash_script to use

```
#!/bin/zsh

#Create a .txt with all of the shape paths with a find command
cat all_shapes.txt | while read vtk;
do
	output_folder=/output_for_the_data_created/
	input_folder=/input_data_folder/

	out=${vtk/$input_folder/$output_folder}
	out=${out/.vtk/.nrrd}
	out_dir=${out/$(basename $out)/}

	if [ ! -d $out_dir ];
	then
		mkdir -p $out_dir
	fi

	max=[max_computed_by_compute_max.py]

	command="./fly_by_features --surf $vtk --radius 4 --out $out --spiral 64 --resolution 256 --planeSpacing 1 --flyByCompose --maxMagnitude $max"
	echo $command
	eval $command

done
```


### Deep Learning with the new datas
To learn with the new datasets, we recommend to use the code from the [US-famli repository](https://github.com/juanprietob/US-famli)

#### Create the TfRecord dataset
```
python3 tfRecords.py 
	#the input is a csv file containing nrrd Files,class
	--csv 		/path_to_csv/file.csv
	--enumerate "class" 
	#if you want to split the data to have an evaluation data
	--split 	0.2 
	--out 		/output_path
	--imageDimension 3
```

#### Train the model
The best way to make your model have good results with this data is to use a LSTM Network, and the US-famli repository have a pre-made lstm network but you can also do your own and add it to the src/py/dl/nn_v2 folder
```
python3 train_v2.py 
	--json 				/path/fly_by_features_split_train.json 
	--out 				/output_dir
	--batch_size 		[num_batch_size]
	--learning_rate 	[num_learning_rate]
	--num_epochs 		[num_epochs]
	--buffer_size 		[num_buffer_size] 
	--drop_prob 		[num_drop_prob] 
	--nn 				name_of_the_model(example:lstm_class_nn)
	# to change the summary write rate (summary made every __ steps)
	--summary_writer 	[num_summary_writer]

	# if you want to continue to train a model
	--in_model /path_to_model/model 

	#if you need a saved_model format
	--save_model /path/saved_model
```

#### Evaluate the model
```
python3 eval_v2.py 
	--json 		/path/fly_by_features_split_train.json
	--model 	/path/saved_model
```
