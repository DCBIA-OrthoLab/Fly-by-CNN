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
  sphere = fbf.CreateIcosahedron(radius=2.75, sl=3)
  flyby = fbf.FlyByGenerator(sphere, resolution=512, visualize=False, use_z=True, split_z=True)
  flyby_features = fbf.FlyByGenerator(sphere, 512, visualize=False)
  a = 0
  for index in range(len(filesList)): 
    surf = GetSurf(surfaceFolder,filesList,index)
    for indexRotation in range(nbRotation):
      a += 1
      surfRot = fbf.RandomRotation(surf)
      ApplyFlyby(flyby,surfRot[0],index,indexRotation, outputFolder)
      ApplyFlyby2(flyby_features,surfRot[0],index,indexRotation, outputFolder, prop)
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

  img = fbf.GetImage(img_np)

  fileName = outputFolder+"/out"+ str(index+1)+"_rot"+str(indexRotation+1) + "_"
  if os.path.isfile(fileName+".nrrd"):
    fileName += '(1)'
  fileName += ".nrrd"
  print(f'Output file Name : {fileName}')
  fbf.WriteNrrd(img, fileName)



def ApplyFlyby2(flyby_features,surf,index,indexRotation, outputFolder, prop): 
  unit_surf = fbf.GetUnitSurf(surf)
  surf_actor = fbf.GetPointIdMapActor(unit_surf)
  flyby_features.addActor(surf_actor)

  out_point_ids_rgb_np = flyby_features.getFlyBy()

  flyby_features.removeActors()
  
  """
  out_point_id_img = fbf.GetImage(out_point_ids_rgb_np)
  
  out_filename = outputFolder+"/out_"+ "_point_id_map" + str(index+1)+"_rot_"+str(indexRotation+1)
  if os.path.isfile(out_filename + ".nrrd"):
    out_filename += '(1)'
  out_filename += ".nrrd"

  print("Writing:", out_filename)

  fbf.WriteNrrd(out_point_id_img)

  """

  out_features_np = ExtractPointFeatures(surf, out_point_ids_rgb_np, prop,0)

  img = GetImage(out_features_np)

  fileName = outputFolder+"/out"+ str(index+1)+"_rot"+str(indexRotation+1) + "_" + prop 

  if os.path.isfile(fileName+".nrrd"):
    fileName += '(1)'
  fileName += ".nrrd"
  
  
  fbf.WriteNrrd(img,fileName)

  


def BlockPrint():
  sys.stdout = open(os.devnull, 'w')

def EnablePrint():
  sys.stdout = sys.__stdout__

if __name__ == "__main__":
    if len (sys.argv) < 5:
        print("Usage: flyby_cli <surfaceFolder> <outputFolder> <prop> <nbRotation>")
        sys.exit (1)
    main()
