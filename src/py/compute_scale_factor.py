import vtk
import numpy as np 
import os
import glob
import argparse


def main(args):
	vtkfilename = args.surf
	max_value = -1

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

	shape_points = np.array(shape_points)
	shape_mean = np.mean(shape_points, axis=0)
	shape_points = shape_points - shape_mean
	max_value = np.maximum(max_value, np.amax(np.linalg.norm(shape_points, axis=1)))

	print(max_value)
	with open(args.out, "w") as f:
		f.write(str(max_value))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Computes maximum magnitude/scaling factor')
	parser.add_argument('--surf', type=str, default=None, help='Target surface or mesh', required=True)
	parser.add_argument('--out', type=str, default="max.txt", help='Output filename')
	args = parser.parse_args()

	main(args)
