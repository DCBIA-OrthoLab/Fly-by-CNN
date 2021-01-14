import vtk
import argparse
import os 
import glob

def main(args) :

	output_list = []
	idx = 0
	normpath = os.path.normpath("/".join([args.dir, '*.vtp']))
	totalCell = 0

	for surf in glob.iglob(normpath, recursive=True):
		clusterName = os.path.split( os.path.splitext(surf)[0])[1]	

		reader = vtk.vtkXMLPolyDataReader()
		reader.SetFileName(surf)
		reader.Update()
		surf = reader.GetOutput()

		# print("Reading:", clusterName, idx)

		nbCell = surf.GetNumberOfCells()
		totalCell += nbCell
		line = clusterName + ' ' + str(nbCell) + '\n'
		output_list.append(line)
		output_list.sort()

	output_file = open(args.out, 'w')
	output_file.writelines(output_list)
	output_file.close()
	print(totalCell)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Create a text file containing the number of fiber per cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dir', type=str, help='Directory containing clusters')
	parser.add_argument('--out', type=str, help='Output filename')

	args = parser.parse_args()

	main(args)
