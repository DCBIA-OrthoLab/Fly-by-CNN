import vtk
import numpy as np
import argparse
import os
from utils import *
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk

from sklearn.decomposition import PCA


def main(args):
    
    # Read all the data from the file
    if(args.normals):
        surf = ReadSurf(args.surf)
        surfActor = GetNormalsActor(surf)
    elif(args.property_fn):

        fname, extension = os.path.splitext(args.property_fn)
        extension = extension.lower()

        if extension == ".gii":
            from fsl.data import gifti
            vertex_features = gifti.loadGiftiVertexData(args.property_fn)[1]
            vertex_features = (vertex_features - np.min(vertex_features, axis=0))/(np.max(vertex_features, axis=0) - np.min(vertex_features, axis=0))

            pca = PCA(n_components=3)
            vertex_features = pca.fit_transform(vertex_features)

            vertex_features = (vertex_features - np.min(vertex_features))/(np.max(vertex_features) - np.min(vertex_features))*255

            colored_points = vtk.vtkUnsignedCharArray()
            colored_points.SetName('colors')
            colored_points.SetNumberOfComponents(3)

            for pid, feat in enumerate(vertex_features):
                colored_points.InsertNextTuple3(feat[0], feat[1], feat[2])

            surf = ReadSurf(args.surf)
            surf = GetUnitSurf(surf)
            surf.GetPointData().SetScalars(colored_points)
            surfActor = GetActor(surf)
            surfActor.GetMapper().SetScalarModeToUsePointData()
            surfActor.GetProperty().LightingOff()
            surfActor.GetProperty().ShadingOff()
            surfActor.GetProperty().SetInterpolationToFlat()
        else:
            property_array = numpy_to_vtk(np.loadtxt(args.property_fn))
            property_name = "property"
            if args.property is not None:
                property_name = args.property
            property_array.SetName(property_name)
            surf = ReadSurf(args.surf)
            surf.GetPointData().AddArray(property_array)
            surfActor = GetColoredActor(surf, property_name)
    else:
        surf = ReadSurf(args.surf)
        if args.property:
            surfActor = GetRandomColoredActor(surf, args.property)    
        else:
            surfActor = GetActor(surf)    


    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderer.AddActor(surfActor)


    if args.ico > 0:
        ico = CreateIcosahedron(args.ico_radius, args.ico)
        ico_actor = GetActor(ico)
        ico_actor.GetProperty().SetRepresentationToWireframe()
        ico_actor.GetProperty().SetColor(1.0, 69.0/255.0, 0.0);
        ico_actor.GetProperty().SetLineWidth(20.0);
        renderer.AddActor(ico_actor)

    renderer.SetBackground(1, 1, 1)
    renderer.ResetCamera()
    # renderer.GetActiveCamera().SetViewUp(0, 0, 1)

    renderWindow.SetSize(1900, 1200)
    renderWindow.Render()
    renderWindowInteractor.Start()

    if args.out:
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(args.out)
        writer.SetInputData(surf)
        writer.Write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick view a mesh', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--surf', type=str, help='Mesh surface in .vtk, .obj, .stl format', required=True)
    parser.add_argument('--normals', type=int, help='Color using the normals', default=0)
    parser.add_argument('--ico', type=int, help='View the icosahedron subdivision', default=0)
    parser.add_argument('--ico_radius', type=float, help='View the icosahedron subdivision', default=1.25)
    parser.add_argument('--property_fn', type=str, help='Color using the property in the file', default=None)
    parser.add_argument('--property', type=str, help='Property name to be added', default=None)
    parser.add_argument('--out', type=str, help='Save the 3D surface model', default=None)
    args = parser.parse_args()

    main(args)