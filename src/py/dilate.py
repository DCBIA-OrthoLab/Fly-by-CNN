import vtk
from utils import *
import post_process
import argparse


def main(path,out):
  filesList = GetFiles (path)
  finalList = filesList[:]

  for item in filesList:
    if 'RegionID.vtk' not in item:
      finalList.remove(item)
  
  for item in finalList:
    fileName = path + "/" + item
    outfilename= out + "/" + item[:-4] + '_dil_gum_.vtk'

    surf = ReadSurf(fileName)

    real_labels = surf.GetPointData().GetAbstractArray('RegionId')
    post_process.DilateLabel(surf, real_labels, 1, iterations=1)

    gum_surf = post_process.Threshold(surf, real_labels, 0, 1)

    print("Writing:", outfilename)
    polydatawriter = vtk.vtkPolyDataWriter()
    polydatawriter.SetFileName(outfilename)
    polydatawriter.SetInputData(gum_surf)
    polydatawriter.Write()



def GetFiles(path):
  filesList = os.listdir(path)
  filesList.sort()
  return filesList


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Choose a folder')
  parser.add_argument('--dir',type=str, help='Input directory')
  parser.add_argument('--out',type=str, help= 'output directory')
  args = parser.parse_args()
  main(args.dir,args.out)