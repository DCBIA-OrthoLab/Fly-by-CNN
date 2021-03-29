
import numpy as np
import time
import itk
import argparse
import os
import post_process
import fly_by_features as fbf
import tensorflow as tf
import vtk


parser = argparse.ArgumentParser(description='Predict an input with a trained neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--surf', type=str, help='Input surface mesh to label', required=True)
parser.add_argument('--model', type=str, help='Model to do segmentation', required=True)
parser.add_argument('--out', type=str, help='Output model with labels', default="out.vtk")

args = parser.parse_args()

start_time = time.time()

surf = fbf.ReadSurf(args.surf)
unit_surf = fbf.GetUnitSurf(surf)

sphere = fbf.CreateIcosahedron(radius=2, sl=2)
flyby = fbf.FlyByGenerator(sphere, resolution=512, visualize=False, use_z=True, split_z=True)

surf_actor = fbf.GetNormalsActor(unit_surf)
flyby.addActor(surf_actor)

print("FlyBy features ...")
img_np = flyby.getFlyBy()
flyby.removeActors()

point_id_actor = fbf.GetPointIdMapActor(unit_surf)
flyby_features = fbf.FlyByGenerator(sphere, 512, visualize=False)
flyby_features.addActor(point_id_actor)

print("FlyBy features point id map ...")
img_point_id_map_np = flyby_features.getFlyBy()
img_point_id_map_np = img_point_id_map_np.reshape((-1, img_point_id_map_np.shape[-1]))
flyby_features.removeActors()


model = tf.keras.models.load_model(args.model, custom_objects={'tf': tf})
model.summary()

print("Predict ...")
img_predict_np = model.predict(img_np)
img_predict_np = img_predict_np.reshape((-1, img_predict_np.shape[-1]))

prediction_array_count = np.zeros([surf.GetNumberOfPoints(), int(np.max(img_predict_np) + 1)])

for point_id_rgb, prediction in  zip(img_point_id_map_np, img_predict_np):

	r = point_id_rgb[0]
	g = point_id_rgb[1]
	b = point_id_rgb[2]

	point_id = int(b*255*255 + g*255 + r - 1)

	prediction_array_count[point_id][int(prediction)] += 1

real_labels = vtk.vtkIntArray()
real_labels.SetNumberOfComponents(1)
real_labels.SetNumberOfTuples(surf.GetNumberOfPoints())
real_labels.SetName("RegionId")
real_labels.Fill(-1)

for pointId,prediction in enumerate(prediction_array_count):
	if np.max(prediction) > 0:
		label = np.argmax(prediction)
		real_labels.SetTuple(pointId, (label,))

surf.GetPointData().AddArray(real_labels)

outfilename = args.out
outfilename_pre = outfilename
outfilename_pre = os.path.splitext(outfilename_pre)[0] + "_pre.vtk"
print("Writting:", outfilename_pre)
polydatawriter = vtk.vtkPolyDataWriter()
polydatawriter.SetFileName(outfilename_pre)
polydatawriter.SetInputData(surf)
polydatawriter.Write()

labels_range = np.zeros(2)
real_labels.GetRange(labels_range)
for label in range(int(labels_range[0]), int(labels_range[1]) + 1):
	print("Removing islands:", label)
	post_process.RemoveIslands(surf, real_labels, label, 500)


outfilename = args.out
outfilename_islands = outfilename
outfilename_islands = os.path.splitext(outfilename_islands)[0] + "_islands.vtk"
print("Writting:", outfilename_pre)
polydatawriter = vtk.vtkPolyDataWriter()
polydatawriter.SetFileName(outfilename_islands)
polydatawriter.SetInputData(surf)
polydatawriter.Write()

print("Relabel...")
post_process.ReLabel(surf, real_labels, 3, -2)

print("Connectivity...")
post_process.ConnectivityLabeling(surf, real_labels, 2, 2)

print("Eroding...")
post_process.ErodeLabel(surf, real_labels, -2)

print("Writting:", outfilename)
polydatawriter = vtk.vtkPolyDataWriter()
polydatawriter.SetFileName(outfilename)
polydatawriter.SetInputData(surf)
polydatawriter.Write()

gum_surf = post_process.Threshold(surf, real_labels, 0, 1)
outfilename_gum = outfilename
outfilename_gum = os.path.splitext(outfilename_gum)[0] + "_gum.vtk"
print("Writting:", outfilename_gum)
polydatawriter = vtk.vtkPolyDataWriter()
polydatawriter.SetFileName(outfilename_gum)
polydatawriter.SetInputData(gum_surf)
polydatawriter.Write()

teeth_surf = post_process.Threshold(surf, real_labels, 2, 999999)
outfilename_teeth = outfilename
outfilename_teeth = os.path.splitext(outfilename_teeth)[0] + "_teeth.vtk"
print("Writting:", outfilename_teeth)
polydatawriter = vtk.vtkPolyDataWriter()
polydatawriter.SetFileName(outfilename_teeth)
polydatawriter.SetInputData(teeth_surf)
polydatawriter.Write()