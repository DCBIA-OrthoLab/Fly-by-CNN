# rescale all fibers with the factors given in parameters 
# should display the real location of the fiber

import numpy as np
import argparse
import nibabel as ni
import vtk


def scalingSurf(vtkdata, scale, centering, shape):
	polypoints = vtkdata.GetPoints()
	
	born_sup =scale*np.array(shape) + np.array(centering)

	for pid in range(polypoints.GetNumberOfPoints()):
		spoint = np.array(polypoints.GetPoint(pid))
		new_point = normalize(spoint, centering, born_sup)

		vtkdata.GetPoints().SetPoint(pid,new_point)

	return vtkdata


def normalize(x, xmin, xmax) :

	return 2*(x - xmin)/(xmax-xmin) - np.ones(x.size, dtype=float)

def readNiiFiles(niiFile, scaleFactor, translateVector, shape):

	image = ni.load(niiFile)

	scaleFactor += image.affine[0][0]
	translateVector += np.array([image.affine[0][3], image.affine[1][3], image.affine[2][3]])
	shape += np.array(image.shape)

	return scaleFactor, translateVector, shape


def main(args):

	meanScale = 0.
	meanCentering = np.zeros(3)
	meanShape = np.zeros(3)

	for niiFile in args.nii :
		meanScale, meanCentering, meanShape = readNiiFiles(niiFile, meanScale, meanCentering, meanShape)

	meanScale/=len(args.nii)
	meanCentering/=len(args.nii)
	meanShape = np.array( meanShape/len(args.nii), dtype=int )

	print("scale factor: ", meanScale)
	print("centering vector: ", meanCentering)
	print("matrix shape: ", meanShape)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Compute the average of the scale factor, centering vector and shape for any given file')
	parser.add_argument('--nii', type=str, nargs="+", help='NIfTI files', required=True)

	args = parser.parse_args()	
	main(args)
