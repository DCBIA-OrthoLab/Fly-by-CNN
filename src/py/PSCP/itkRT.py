import vtk
import itk
import argparse
import glob
import os
import shutil

import numpy as np


def main(args):
	img_fn_array = []

	if args.image:
		img_obj = {}
		img_obj["img"] = args.image
		img_obj["out"] = args.out
		img_fn_array.append(img_obj)

	if args.dir:
		normpath = os.path.normpath("/".join([args.dir, '**', '*']))
		for img_fn in glob.iglob(normpath, recursive=True):
			if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM"]]:
				img_obj = {}
				img_obj["img"] = img_fn
				img_obj["out"] = os.path.normpath("/".join([args.out, os.path.splitext(os.path.splitext(os.path.basename(img_fn))[0])[0]]))
				img_fn_array.append(img_obj)

	
	for img_obj in img_fn_array:
		image = img_obj["img"]
		out = img_obj["out"]
		print("Reading:", image)

		if not os.path.exists(out):
			os.makedirs(out)
		else:
			shutil.rmtree(out)
			os.makedirs(out)

		
		ImageType = itk.Image[itk.US, 3]

		img_read = itk.ImageFileReader[ImageType].New(FileName=image)
		img_read.Update()
		img = img_read.GetOutput()


		label = itk.ConnectedComponentImageFilter[ImageType, ImageType].New()
		label.SetInput(img)
		label.Update()

		labelStatisticsImageFilter = itk.LabelStatisticsImageFilter[ImageType, ImageType].New()
		labelStatisticsImageFilter.SetLabelInput(label.GetOutput())
		labelStatisticsImageFilter.SetInput(img)
		labelStatisticsImageFilter.Update()
		NbreOfLabel = len(labelStatisticsImageFilter.GetValidLabelValues())

		for i in range(1,NbreOfLabel):
			extractImageFilter = itk.ExtractImageFilter[ImageType, ImageType].New()
			extractImageFilter.SetExtractionRegion(labelStatisticsImageFilter.GetRegion(i))
			extractImageFilter.SetInput(img)
			extractImageFilter.Update()

			extractImageFilter = extractImageFilter.GetOutput()


			vector = itk.Size[3]
			Size = vector()
			Size[0] = extractImageFilter.GetLargestPossibleRegion().GetSize()[0]*2
			Size[1] = extractImageFilter.GetLargestPossibleRegion().GetSize()[1]*2
			Size[2] = extractImageFilter.GetLargestPossibleRegion().GetSize()[2]*2

			vector = itk.Vector[itk.D,3]
			Spacing = vector()
			Spacing[0] = extractImageFilter.GetSpacing()[0]/2.0
			Spacing[1] = extractImageFilter.GetSpacing()[1]/2.0
			Spacing[2] = extractImageFilter.GetSpacing()[2]/2.0

			vector = itk.Index[3]
			Index = vector()
			Index[0] = extractImageFilter.GetLargestPossibleRegion().GetIndex()[0]*2
			Index[1] = extractImageFilter.GetLargestPossibleRegion().GetIndex()[1]*2
			Index[2] = extractImageFilter.GetLargestPossibleRegion().GetIndex()[2]*2


			TransformType = itk.IdentityTransform[itk.D, 3].New()
			resample = itk.ResampleImageFilter[ImageType, ImageType].New()
			resample.SetInput(extractImageFilter)
			resample.SetSize(Size)
			resample.SetOutputSpacing(Spacing)

			resample.SetOutputDirection(extractImageFilter.GetDirection())
			resample.SetOutputOrigin(extractImageFilter.GetOrigin())
			resample.SetOutputStartIndex(Index)

			resample.SetTransform(TransformType)
			resample.UpdateOutputInformation()
			resample.Update()


			writer = itk.ImageFileWriter[ImageType].New()
			outputFilename = out+"/"+os.path.basename(out)+"_"+str(i)+".nii.gz"
			print("Writing:", outputFilename)
			writer.SetFileName(outputFilename)
			writer.SetInput(resample.GetOutput())
			writer.Update()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='create RootCanal object from a segmented file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	in_group_features = parser.add_mutually_exclusive_group(required=True)
	in_group_features.add_argument('--image', type=str, help='input file')
	in_group_features.add_argument('--dir', type=str, help='input dir')
	
	parser.add_argument('--out', type=str, help='output dir', default='')

	args = parser.parse_args()

	main(args)







