import vtk
from utils import *
import argparse

def main(gum,teeth,nb):


  #surf_gum = ReadSurf(gum)
  #surf_teeth = ReadSurf(teeth)


  outFilename = "/NIRAL/work/leclercq/data/merge/merge_scan" + str(nb) + '.vtk'

  gumList = GetFiles(gum)
  surf_gum = ''
  for item in gumList:
    if 'scan'+str(nb)+'_' in item:
      surf_gum = gum + '/' + item

  print(surf_gum)


  teethList = GetFiles(teeth)
  surf_teeth = ''
  for item in teethList:
    if 'scan'+str(nb)+'_' in item:
      surf_teeth = teeth + '/' + item

  print(surf_teeth)


  surf_gum = ReadSurf(surf_gum)
  surf_teeth = ReadSurf(surf_teeth)

  # adding universal ID to gum surface
  real_labels = vtk.vtkIntArray()
  real_labels.SetNumberOfComponents(1)
  real_labels.SetNumberOfTuples(surf_gum.GetNumberOfPoints())
  real_labels.SetName("UniversalID")
  real_labels.Fill(0)
  surf_gum.GetPointData().AddArray(real_labels)


  # Merge
  merge = vtk.vtkAppendPolyData()
  merge.AddInputData(surf_gum)
  merge.AddInputData(surf_teeth)
  merge.Update()
  out = merge.GetOutput()

  # Write file 

  print('Writing file: ', outFilename)
  polydatawriter = vtk.vtkPolyDataWriter()  
  polydatawriter.SetFileName(outFilename)
  polydatawriter.SetInputData(out)
  polydatawriter.Write()
  print("Done")

def GetFiles(path):
  filesList = os.listdir(path)
  filesList.sort()
  return filesList

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Merge teeth and gum')
  parser.add_argument('--nb',type=int,help='file index')
  parser.add_argument('--gum',type=str, help='gum file')
  parser.add_argument('--teeth',type=str, help='teeth file')
  parser.add_argument('--out',type=str, help = 'name of output')
  args = parser.parse_args()
  main(args.gum,args.teeth,args.nb)

