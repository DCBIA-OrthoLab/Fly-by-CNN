import vtk
import argparse
import utils
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np 

def main(args):
    
    surf = utils.ReadSurf(args.surf)

    landmarks = utils.ReadJSONMarkups(args.landmarks)

    landmarks_points = vtk.vtkPoints()
    landmarks_points.SetData(numpy_to_vtk(landmarks))

    landmarks_poly = vtk.vtkPolyData()
    landmarks_poly.SetPoints(landmarks_points)

    locator = vtk.vtkOctreePointLocator()
    locator.SetDataSet(landmarks_poly)
    locator.BuildLocator()

    distance_array_np = []

    for idx in range(surf.GetNumberOfPoints()):
        p = surf.GetPoint(idx)
        idx_landmark = locator.FindClosestPoint(p)

        d = np.linalg.norm(np.array(p) - np.array(landmarks_points.GetPoint(idx_landmark)))

        distance_array_np.append(d)


    distance_array_np = np.array(distance_array_np)

    landmarks_distance = numpy_to_vtk(distance_array_np)
    landmarks_distance.SetName("landmarks_distance")


    landmarks_distance_norm_vtk = numpy_to_vtk(distance_array_np/np.max(distance_array_np))
    landmarks_distance_norm_vtk.SetName("landmarks_distance_norm")

    expm1 = np.expm1(distance_array_np/np.max(distance_array_np))
    expm1 = np.clip(expm1, 0, args.max)/args.max
    landmarks_distance_exp_vtk = numpy_to_vtk(expm1)
    landmarks_distance_exp_vtk.SetName("landmarks_distance_expm1")

    surf.GetPointData().AddArray(landmarks_distance)
    surf.GetPointData().AddArray(landmarks_distance_norm_vtk)
    surf.GetPointData().AddArray(landmarks_distance_exp_vtk)


    utils.Write(surf, args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean poly', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--surf', type=str, help='Mesh surface in .vtk, .obj, .stl format', required=True)
    parser.add_argument('--landmarks', type=str, help='Landmarks in json format', required=True)
    parser.add_argument('--max', type=float, help='Max distance, threshold with this max value', default=0.1)
    parser.add_argument('--out', type=str, help='Output surface with distance map', default="out.vtk")

    args = parser.parse_args()

    main(args)