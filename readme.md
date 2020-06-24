# Fly by CNN

## What is it?
Fly by CNN is a C++ code that takes a 3D mesh and create 2D images of this one following the unit sphere and the number of subdivisions associated. The 2D images created contained the mesh features and labels.

## How it works?
It normalizes the mesh and create the unit sphere around this one. Then it subdivides the sphere in a certain number of regular points following the number of subdivisions. A tangent oriented plan is then created with a certain number of points. It then projects the mesh in this plan getting the associated features and label. The images are saved and then it creeates another tengent plane centered on the following sphere point.

![coucou]https://github.com/DCBIA-OrthoLab/fly-by-cnn/docs/Sphere_and_plane.png?raw=true



