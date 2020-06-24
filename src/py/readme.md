# Python source

## predict.py
Does the prediction for the mesh label using the fly-by-cnn idea. It first creates the images with features, does the prediction on this images and projects it in the mesh. This code uses the *LinearSubdivisionFilter.py* and *post_process.py* .

You can select the model with the flag 

__--surf__ : Select your mesh

__--model__ : Select the model

__--numberOfSubdivisions__ : Select the number of point you'll divide the sphere 

__--out__ : Output name


To run the program, run the following command : 
python3 predict.py __--surf__ path/to/surf __--model__ /path/to/model/folder __--numberOfSubdivisions__ *integer* __--out__ /path/to/output.vtk

Other flags are available by running python3 predict.py --help

## LinearSubdivisionFilter.py
This code does the sphere subdivion. For example : numberOfSubdivisions = 10 -> 1026 ids / numberOfSubdivisions = 5 -> 268 ids 

## post_process.py
Post preocess is done after the prediction in order to correct the wrong labels. It takes the smallest components in the mesh (< 1000 ids) set their labels to -1 and apply a region growing on all the -1 label.