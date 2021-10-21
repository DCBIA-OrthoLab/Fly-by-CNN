import sys
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
parser.add_argument('--model', type=str, help='Model to do segmentation', default="/app/u_seg_nn_v3.1")
parser.add_argument('--dilate', type=int, help='Number of iterations to dilate the boundary', default=0)
parser.add_argument('--out', type=str, help='Output model with labels', default="out.vtk")

args = parser.parse_args()

start_time = time.time()

surf = fbf.ReadSurf(args.surf)
unit_surf = fbf.GetUnitSurf(surf)

sphere = fbf.CreateIcosahedron(radius=2.75, sl=3)
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


if os.path.exists(args.model):
	model = tf.keras.models.load_model(args.model, custom_objects={'tf': tf})
	model.summary()
else:
	print("Please set the model directory to a valid path", file=sys.stderr)

print("Predict ...")
img_predict_np = model.predict(tf.expand_dims(img_np, axis=0))
img_predict_np = np.argmax(np.squeeze(img_predict_np, axis=0), axis=-1)
img_predict_np = img_predict_np.reshape(-1)

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
real_labels.Fill(0)

for pointId,prediction in enumerate(prediction_array_count):
	if np.max(prediction) > 0:
		label = np.argmax(prediction)
		real_labels.SetTuple(pointId, (label,))

surf.GetPointData().AddArray(real_labels)

outfilename = args.out
outfilename_pre = outfilename
outfilename_pre = os.path.splitext(outfilename_pre)[0] + "_pre.vtk"
print("Writing:", outfilename_pre)
polydatawriter = vtk.vtkPolyDataWriter()
polydatawriter.SetFileName(outfilename_pre)
polydatawriter.SetInputData(surf)
polydatawriter.Write()

if args.dilate:
	print("Dilate...")
	#Dilate GUM label
	post_process.DilateLabel(surf, real_labels, 3, iterations=args.dilate)

labels_range = np.zeros(2)
real_labels.GetRange(labels_range)
for label in range(int(labels_range[0]), int(labels_range[1]) + 1):
	print("Removing islands:", label)
	post_process.RemoveIslands(surf, real_labels, label, 200)


out_filename = args.out
outfilename_islands = out_filename
outfilename_islands = os.path.splitext(outfilename_islands)[0] + "_islands.vtk"
print("Writting:", outfilename_islands)
polydatawriter = vtk.vtkPolyDataWriter()
polydatawriter.SetFileName(outfilename_islands)
polydatawriter.SetInputData(surf)
polydatawriter.Write()

print("Relabel...")
#Re label the gum which is label 3 to label -1
post_process.ReLabel(surf, real_labels, 3, -1)

print("Connectivity...")
#Do the connected component analysis and assign labels starting at label 2
post_process.ConnectivityLabeling(surf, real_labels, 2, 2)

out_filename = args.out
outfilename_connectivity = out_filename
outfilename_connectivity = os.path.splitext(outfilename_connectivity)[0] + "_connectivity.vtk"
print("Writing:", outfilename_connectivity)
polydatawriter = vtk.vtkPolyDataWriter()
polydatawriter.SetFileName(outfilename_connectivity)
polydatawriter.SetInputData(surf)
polydatawriter.Write()

print("Eroding...")
#Erode the gum label 
post_process.ErodeLabel(surf, real_labels, -1, ignore_label=0)

print("Writing:", outfilename)
polydatawriter = vtk.vtkPolyDataWriter()
polydatawriter.SetFileName(outfilename)
polydatawriter.SetInputData(surf)
polydatawriter.Write()

teeth_surf = post_process.Threshold(surf, real_labels, 2, 999999)
outfilename_teeth = outfilename
outfilename_teeth = os.path.splitext(outfilename_teeth)[0] + "_teeth.vtk"
print("Writing:", outfilename_teeth)
polydatawriter = vtk.vtkPolyDataWriter()
polydatawriter.SetFileName(outfilename_teeth)
polydatawriter.SetInputData(teeth_surf)
polydatawriter.Write()


gum_surf = post_process.Threshold(gum_surf, real_labels, 0, 1)
outfilename_gum = outfilename
outfilename_gum = os.path.splitext(outfilename_gum)[0] + "_gum.vtk"
print("Writing:", outfilename_gum)
polydatawriter = vtk.vtkPolyDataWriter()
polydatawriter.SetFileName(outfilename_gum)
polydatawriter.SetInputData(gum_surf)
polydatawriter.Write()

end_time = time.time()

print("Prediction time took:", (end_time - start_time), "seconds")
