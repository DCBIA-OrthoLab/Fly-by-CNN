#!/usr/bin/env python-real

import fly_by_features as fbf
from utils import *
import os
import sys
import argparse

def main():

  args = GetArguments()
  surfaceFolder = args.surfaceFolder
  outputFolder = args.outputFolder
  prop = args.prop
  nbRotation = int(args.nbRotation)
  filesList = GetFiles(surfaceFolder)
  nbOperation = len(filesList) * nbRotation
  flyby = GenerateFB()
  a = 0
  for index in range(len(filesList)): 
    surf = GetSurf(surfaceFolder,filesList,index)
    for indexRotation in range(nbRotation):
      a += 1
      surfRot = fbf.RandomRotation(surf)
      ApplyFlyby(flyby,surfRot[0],index,indexRotation, outputFolder)
      progress = math.floor(100 * (a/nbOperation)) # percentage
      print(f'Progress: {progress}%')
      print('\n\n')

  print("Done")
  return

def GetArguments():  
  parser = argparse.ArgumentParser(description="Apply fly-by features")
  parser.add_argument("surfaceFolder")
  parser.add_argument("outputFolder")
  parser.add_argument("prop")
  parser.add_argument("nbRotation")
  args = parser.parse_args()
  return args

def GetFiles(path):
  filesList = os.listdir(path)
  filesList.sort()
  return filesList

def GenerateFB():
  sphere = fbf.CreateIcosahedron(radius=2.75, sl=3)
  flyby = fbf.FlyByGenerator(sphere, resolution=512, visualize=False, use_z=True, split_z=True)
  return flyby

def GetSurf(folder, filesList, index):    
  path = folder + "/" + filesList[index]
  print(f"path for the surface file:{path}")
  surf = fbf.ReadSurf(path)
  return surf

def ApplyFlyby(flyby,surf,index,indexRotation, outputFolder): 
  unit_surf = fbf.GetUnitSurf(surf)

  surf_actor = fbf.GetNormalsActor(unit_surf)
  flyby.addActor(surf_actor)

  print("FlyBy features ...")
  img_np = flyby.getFlyBy()
  
  flyby.removeActors()

  BlockPrint()
  img = fbf.GetImage(img_np)
  EnablePrint()

  fileName =  outputFolder+"/out_"+str(index+1)+"_rot_"+str(indexRotation+1)
  if os.path.isfile(fileName+".nrrd"):
    fileName += '(1)'
  fileName += ".nrrd"
  print(f'Output file Name : {fileName}')
  fbf.WriteNrrd(img, fileName)

def BlockPrint():
  sys.stdout = open(os.devnull, 'w')

def EnablePrint():
  sys.stdout = sys.__stdout__

if __name__ == "__main__":
    if len (sys.argv) < 5:
        print("Usage: flybyscripte <surfaceFolder> <outputFolder> <prop> <nbRotation>")
        sys.exit (1)
    main()
