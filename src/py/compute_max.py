import vtk
import numpy as np 
import os
import glob
import argparse


def main(args):
	shapes_arr = []
	shapes_dir = os.path.join(os.path.dirname(args.dir), '**/*.vtk')
	for svtk in glob.iglob(shapes_dir, recursive=True):
		shapes_arr.append(svtk)

	max_value = -1
	for vtkfilename in shapes_arr:
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
		max_value = np.maximum(max_value, np.amax(np.abs(np.reshape(shape_points, -1))))

	with open(args.out, "w") as f:
		f.write(str(max_value))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Computes maximum coordinate of all shapes after centering each at 0.')
	parser.add_argument('--dir', type=str, default=None, help='Directory with vtk files', required=True)
	parser.add_argument('--out', type=str, default="max.txt", help='Output filename')
	args = parser.parse_args()

	main(args)
