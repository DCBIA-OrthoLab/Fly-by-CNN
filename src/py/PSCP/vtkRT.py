import vtk
import itk
import argparse
import glob
import os
import shutil

import numpy as np


def Write(vtkdata, output_name):
	outfilename = output_name
	print("Writting:", outfilename)
	polydatawriter = vtk.vtkPolyDataWriter()
	polydatawriter.SetFileName(outfilename)
	polydatawriter.SetInputData(vtkdata)
	polydatawriter.Write()


def main(args):
	img_fn_array = []

	if args.image:
		img_obj = {}
		img_obj["img"] = args.image
		img_obj["out"] = args.out
		img_fn_array.append(img_obj)

	elif args.dir:
		normpath = os.path.normpath("/".join([args.dir, '**', '*']))
		for img_fn in glob.iglob(normpath, recursive=True):
			if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd"]]:
				img_obj = {}
				img_obj["img"] = img_fn
				img_obj["out"] = os.path.normpath("/".join([args.out]))
				img_fn_array.append(img_obj)

	
	for img_obj in img_fn_array:
		image = img_obj["img"]
		out = img_obj["out"]
		print("Reading:", image)

		surf = vtk.vtkNrrdReader()
		surf.SetFileName(image)
		surf.Update()


		dmc = vtk.vtkDiscreteMarchingCubes()
		dmc.SetInputConnection(surf.GetOutputPort())
		dmc.GenerateValues(100, 1, 100)

		# LAPLACIAN smooth
		SmoothPolyDataFilter = vtk.vtkSmoothPolyDataFilter()
		SmoothPolyDataFilter.SetInputConnection(dmc.GetOutputPort())
		SmoothPolyDataFilter.SetNumberOfIterations(10)
		SmoothPolyDataFilter.SetFeatureAngle(120.0)
		SmoothPolyDataFilter.SetRelaxationFactor(0.6)
		SmoothPolyDataFilter.Update()
		
		# SINC smooth
		# smoother = vtk.vtkWindowedSincPolyDataFilter()
		# smoother.SetInputConnection(dmc.GetOutputPort())
		# smoother.SetNumberOfIterations(30)
		# smoother.BoundarySmoothingOff()
		# smoother.FeatureEdgeSmoothingOff()
		# smoother.SetFeatureAngle(120.0)
		# smoother.SetPassBand(0.001)
		# smoother.NonManifoldSmoothingOn()
		# smoother.NormalizeCoordinatesOn()
		# smoother.Update()

		outputFilename = out+"/"+os.path.splitext(os.path.basename(image))[0]+".vtk"
		Write(SmoothPolyDataFilter.GetOutput(), outputFilename)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='create RootCanal object from a segmented file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	in_group_features = parser.add_mutually_exclusive_group(required=True)
	in_group_features.add_argument('--image', type=str, help='input file')
	in_group_features.add_argument('--dir', type=str, help='input dir')

	parser.add_argument('--out', type=str, help='output dir', default='')

	args = parser.parse_args()

	main(args)
