import argparse
import itk
import numpy as np

def main(args):

	file_name = args.nii

	ImageType = itk.VectorImage[itk.F,3]
	reader = itk.ImageFileReader[ImageType].New()
	reader.SetFileName(args.nii)
	reader.Update()
	img = reader.GetOutput()

	ImageMaskType = itk.Image[itk.UC, 3]
	img_mask = ImageMaskType.New()
	img_mask.SetLargestPossibleRegion(img.GetLargestPossibleRegion())
	img_mask.SetDirection(img.GetDirection())
	img_mask.SetSpacing(img.GetSpacing())
	img_mask.SetOrigin(img.GetOrigin())
	img_mask.Allocate()
	img_mask.FillBuffer(1)

	so = itk.ImageSpatialObject[3, itk.UC].New()
	so.SetImage(img_mask)
	so.Update()
	so.GetMyBoundingBoxInWorldSpace().GetBounds()

	print("scale factor: ", img.GetSpacing())
	print("centering vector: ", img.GetOrigin())
	print("Bounding box:", so.GetMyBoundingBoxInWorldSpace().GetBounds())
	mean_v = [0.0] * 3
	bounds_max_v = [0.0] * 3

	bounds = so.GetMyBoundingBoxInWorldSpace().GetBounds() 

	mean_v[0] = (bounds[0] + bounds[1])/2.0
	mean_v[1] = (bounds[2] + bounds[3])/2.0
	mean_v[2] = (bounds[4] + bounds[5])/2.0
	bounds_max_v[0] = max(bounds[0], bounds[1])
	bounds_max_v[1] = max(bounds[2], bounds[3])
	bounds_max_v[2] = max(bounds[4], bounds[5])

	bounds_max_arr = np.array(bounds_max_v)
	scale_factor = 1/np.linalg.norm(bounds_max_arr - np.array(mean_v))

	print("mean:", mean_v)
	print("scale factor:", scale_factor)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Compute the average of the scale factor, centering vector of a nifti file')
	parser.add_argument('--nii', type=str, help='NIfTI file', required=True)

	args = parser.parse_args()	
	main(args)
