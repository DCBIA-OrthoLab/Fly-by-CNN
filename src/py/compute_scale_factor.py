import vtk
import numpy as np 
import os
import glob
import argparse

def Normalization(vtkdata):
    polypoints = vtkdata.GetPoints()
    
    nppoints = []
    for pid in range(polypoints.GetNumberOfPoints()):
        spoint = polypoints.GetPoint(pid)
        nppoints.append(spoint)

    npmean = np.mean(np.array(nppoints), axis=0)
    nppoints -= npmean
    npscale = np.max([np.linalg.norm(p) for p in nppoints])
    nppoints /= npscale

    for pid in range(polypoints.GetNumberOfPoints()):
        vtkdata.GetPoints().SetPoint(pid, nppoints[pid])

    return vtkdata, npmean, npscale

def main(args):
    shapes_arr = []
    if(args.surf):
        shapes_arr.append(args.surf)    

    if(args.dir):
        shapes_dir = os.path.join(os.path.dirname(args.dir), '**/*.vtk')
        for svtk in glob.iglob(shapes_dir, recursive=True):
            shapes_arr.append(svtk)
    
    max_value=-1
    for vtkfilename in shapes_arr:
        scale_factor = -1

        print("Reading: ", vtkfilename)
        vtkfilename = vtkfilename.rstrip()
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(vtkfilename)
        reader.Update()
        shapedata = reader.GetOutput()
        #QUESTION: Do i normalize the shape before finding scale factor - bc the surface is being normalized in the utils.py file before using the scale factor
        shapedata, shape_mean, shape_scale = Normalization(shapedata)
        shapedatapoints = shapedata.GetPoints()
        
        bounds = [0.0] * 6
        mean_v = [0.0] * 3
        bounds_max_v = [0.0] * 3
        bounds = shapedatapoints.GetBounds()
        mean_v[0] = (bounds[0] + bounds[1])/2.0
        mean_v[1] = (bounds[2] + bounds[3])/2.0
        mean_v[2] = (bounds[4] + bounds[5])/2.0
        bounds_max_v[0] = max(bounds[0], bounds[1])
        bounds_max_v[1] = max(bounds[2], bounds[3])
        bounds_max_v[2] = max(bounds[4], bounds[5])

        #Getting points from shape
        shape_points = []
        for i in range(shapedatapoints.GetNumberOfPoints()):
            p = shapedatapoints.GetPoint(i)
            shape_points.append(p)

        #centering points of the shape
        shape_points = np.array(shape_points)
        mean_arr = np.array(mean_v)
        shape_points = shape_points - mean_arr 

        #Computing scale factor
        bounds_max_arr = np.array(bounds_max_v)
        scale_factor = 1/np.linalg.norm(bounds_max_arr - mean_arr)
        
        if(scale_factor>max_value):
            max_value=scale_factor

        print("Scale factor: "+str(scale_factor))
        with open(args.out, "a+") as f:
            f.write(str(scale_factor)+"\n")

    print("Max scale factor: "+str(max_value))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes maximum magnitude/scaling factor using bounding box and appends to file')
    #TODO: MAKE AT LEAST ONE OF THE DIRECTRY OR SURF REQUIRED
    parser.add_argument('--surf', type=str, default=None, help='Target surface or mesh')
    parser.add_argument('--out', type=str, default="scale_factor.txt", help='Output filename')
    parser.add_argument('--dir', type=str, default=None, help='Directory with vtk files')

    args = parser.parse_args()


    main(args)
