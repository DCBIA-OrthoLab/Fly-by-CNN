﻿# README: Automated jaw segmentation extension

This extension aims to provide a GUI for the deep learning method for surface segmentation that we developed. 

# Python environment
For the extension to work, you need to setup a python 3.9 environment with the dependencies listed below:

-   vtk 			9.0.3
-   monai 			0.7.0
-   torch                   1.9.0+cu111
-   torchaudio              0.10.0+cu113
-   torchvision             0.11.1+cu113
-   pytorch3d 		0.6.0
-   tqdm
-   numpy
-   argparse
-   pandas
-   itk
-   SimpleITK
-   scikit-image  / scikit-learn
-	scipy

 *NB: The extension was tested with these specific versions, however it may work with more recent ones.
Additional packages may be required. you can install them with "pip install ... ".  No specific version should be required for these packages.*

# How to run the extension

## Opening the extension
The python environment must be active when opening Slicer (the extension looks for the startup environment when running the segmentation algorithm). 
Open the module named "Prediction - FiboSeg".
 

## Running the module
The input file must be a .vtk file of a lower or upper jaw. The model works better with models of jaws with no wisdom teeth. You can find examples in the "Examples" folder.
To visualize the results, open the output file and set scalars to "visible" and select the correct scalar in  Slicer's "Models" module.  

 ![example](https://github.com/MathieuLeclercq/fly-by-cnn/blob/master/src/slicer_jaw_segmentation/examples/segmentation_example.png?raw=true) 

<!-- ![Example of a jaw model](examples/segmentation_example.png?raw=true) -->

![Example of a jaw model](examples/segmentation_example.png?raw=true)


