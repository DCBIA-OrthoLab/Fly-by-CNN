import vtk
import sys
sys.path.insert(0,'..')
import fly_by_features as fbf
import post_process
import numpy as np
import pandas as pd
from math import dist
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import argparse


from vtkmodules.vtkCommonDataModel import vtkPolyData

from vtkmodules.vtkFiltersCore import (
    vtkClipPolyData,
    vtkFeatureEdges,
    vtkStripper
)
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkFiltersSources import vtkDiskSource



def main(args):
  
  
  truth = fbf.ReadSurf(args.truth)
  surf = fbf.ReadSurf(args.surf)

  truth_point_data = vtk_to_numpy(truth.GetPointData().GetScalars("UniversalID"))
  surf_point_data = vtk_to_numpy(surf.GetPointData().GetScalars(args.scal))

  unique_truth, counts_truth = np.unique(truth_point_data, return_counts = True)
  unique, counts  = np.unique(surf_point_data, return_counts = True)

  l_ids_truth = [unique_truth[index] for index,nb in enumerate(counts_truth) if unique_truth[index] != 33]
  l_ids = [unique[index] for index,nb in enumerate(counts) if counts[index] > 200 and unique[index] != 33]
  
  if (l_ids_truth != l_ids):
    print("Warning!! \n Ids on the predicted surface and on groundtruth are different.")
    print(f' ids in groundtruth: {l_ids_truth}') 
    print(f' ids in prediction: {l_ids}')

  l_crowns_truth = [post_process.Threshold(truth,"UniversalID",uid,uid+0.5) for uid in l_ids_truth]
  l_crowns = [post_process.Threshold(surf, args.scal,uid,uid+0.5) for uid in l_ids]

  crowns_dist_dict = {}
  count_match = 0
  for index,crown in enumerate(l_crowns):    
    crown_id = l_ids[index]
    if crown_id in l_ids_truth:
      index_truth_id = l_ids_truth.index(crown_id)
      print (f"ID for truth: {l_ids_truth[index_truth_id]}\nID for pred:  {crown_id}")
      print(f'index_truth_id == index : {index_truth_id == index}')
      
      # if l_ids_truth[index] != l_ids[index]:
      #   raise ValueError (f"Index doesn't match: {l_ids_truth[index]},{l_ids[index]} ")

      featureEdges = vtkFeatureEdges()
      featureEdges.SetInputData(l_crowns_truth[index_truth_id])
      featureEdges.BoundaryEdgesOn()
      featureEdges.FeatureEdgesOff()
      featureEdges.ManifoldEdgesOff()
      featureEdges.NonManifoldEdgesOff()
      featureEdges.ColoringOn()
      featureEdges.Update()
      edges_truth = vtkPolyData()
      edges_truth.SetPoints(featureEdges.GetOutput().GetPoints())
      print(f'edges truth nb points: {edges_truth.GetNumberOfPoints()}')

      featureEdges = vtkFeatureEdges()
      featureEdges.SetInputData(l_crowns[index])
      featureEdges.BoundaryEdgesOn()
      featureEdges.FeatureEdgesOff()
      featureEdges.ManifoldEdgesOff()
      featureEdges.NonManifoldEdgesOff()
      featureEdges.ColoringOn()
      featureEdges.Update()
      edges = vtkPolyData()
      edges.SetPoints(featureEdges.GetOutput().GetPoints())
      print(f'edges nb points: {edges.GetNumberOfPoints()}')

      locator_truth = vtk.vtkPointLocator()
      locator_truth.SetDataSet(edges_truth)
      locator_truth.BuildLocator()  

      total_dist = 0
      max_dist = 0
      for pid in range(edges.GetNumberOfPoints()):
        point = edges.GetPoint(pid)
        id_truth = locator_truth.FindClosestPoint(point)
        point_truth = edges_truth.GetPoint(id_truth)
        distance = dist(point,point_truth)
        if distance > max_dist:
          max_dist = distance
        total_dist += distance



      mean_dist = total_dist / edges.GetNumberOfPoints() 
      print(f'total dist: {total_dist}, mean dist: {mean_dist}')
      print(f'max dist: {max_dist}')
      l_dist = [mean_dist,max_dist]

      l_for_np = [int(l_ids[index]),mean_dist,max_dist,args.surf]
      if count_match == 0:
        np_array = np.array(l_for_np)
      else:
        np_array = np.vstack([np_array,l_for_np])

      crowns_dist_dict[l_ids[index]] = l_dist
      count_match += 1
    else:
      print(f"no match for id {crown_id} in groundtruth.")
    print("-" * 20)


  df = pd.DataFrame(np_array, columns=['crown', 'avr_dist','max_dist','file'])
  filename = args.csv
  if filename is None:
    filename = 'data.csv'

  else:
    df_csv = pd.read_csv(filename,index_col=[0])
    df = df_csv.append(df, ignore_index=True)

  df.to_csv(filename)
  print(df)



if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Choose a surface to compare with groundtruth')
  parser.add_argument('--surf',type=str, help='Input surface (.vtk file)', required=True)
  parser.add_argument('--truth',type=str, help='groundtruth surface (.vtk file)', required=True)
  parser.add_argument('--scal',type=str, help='Name of the scalar', required=True)
  parser.add_argument('--csv',type=str, help='Append to existing .csv. If none: will create data.csv', default=None)

  args = parser.parse_args()
  main(args)
