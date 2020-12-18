import vtk
import numpy as np 
import os
import glob
import argparse


def main(args):
	vtkfilename = args.surf
	scale_factor = -1

	print("Reading", vtkfilename)
	vtkfilename = vtkfilename.rstrip()
	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(vtkfilename)
	reader.Update()
	shapedata = reader.GetOutput()
	shapedatapoints = shapedata.GetPoints()
	

	shape_points = []
	for i in range(shapedatapoints.GetNumberOfPoints()):
		p = shapedatapoints.GetPoint(i)
		shape_points.append(p)

	#centering points of the shape
	shape_points = np.array(shape_points)
	shape_mean = np.mean(shape_points, axis=0)
	shape_points = shape_points - shape_mean

	#assigning centered points back to shape
	for i in range(shapedatapoints.GetNumberOfPoints()):
		shapedatapoints.SetPoint(i, shape_points[i])

	#computing bounds and mean
	bounds = [0.0] * 6
	mean_v = [0.0] * 3
	bounds_max_v = [0.0] * 3
	bounds = shapedatapoints.GetBounds()
	mean_v[0] = (bounds[0] + bounds[1])/2.0
	mean_v[1] = (bounds[2] + bounds[3])/2.0
	mean_v[2] = (bounds[4] + bounds[5])/2.0
	bounds_max_v[0] = max(bounds[0], bounds[1])
	bounds_max_v[1] = max(bounds[2], bounds[3])
	bounds_max_v[2] = max(bounds[4], bounds[5])

	#Computing scale factor
	bounds_max_arr = np.array(bounds_max_v)
	mean_arr = np.array(mean_v)
	scale_factor = np.linalg.norm(bounds_max_arr - mean_arr)

	print(scale_factor)
	with open(args.out, "a+") as f:
		f.write(str(scale_factor))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Computes maximum magnitude/scaling factor using bounding box and appends to file')
	parser.add_argument('--surf', type=str, default=None, help='Target surface or mesh', required=True)
	parser.add_argument('--out', type=str, default="scale_factor.txt", help='Output filename')
	args = parser.parse_args()

	main(args)
