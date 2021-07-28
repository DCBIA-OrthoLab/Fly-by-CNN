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
import pandas as pd
import json

parser = argparse.ArgumentParser(description='Predict an input with a trained neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--surf', type=str, help='Input fiber bundle', required=True)
parser.add_argument('--model', type=str, help='Model to do segmentation', default="/app/u_seg_nn_v3.1")
parser.add_argument('--out', type=str, help='Output csv with classification labels and confidence', default="out.csv")
parser.add_argument('--scale_factor', type=float, help='Scale the surface by this vale', default= -1)
parser.add_argument('--translate', nargs="+", type=float, help='Center the surface at this point', default=None)
parser.add_argument('--turns', type=int, default=4, help='Number of spiral turns')
parser.add_argument('--radius', type=float, help='Radius of the sphere for the view points', default=4)
parser.add_argument('--spiral', type=int, help='Number of samples along the spherical spiral')

parser.add_argument('--json', type=str, help='class description')



args = parser.parse_args()

start_time = time.time()

model = tf.keras.models.load_model(args.model, custom_objects={'tf': tf})
model.summary()

surf = fbf.ReadSurf(args.surf)
surf = fbf.GetUnitSurf(surf, args.translate, args.scale_factor)

sphere = fbf.CreateSpiral(args.radius, args.spiral, args.turns)
# CreateIcosahedron(radius=2.75, sl=3)
flyby = fbf.FlyByGenerator(sphere, resolution=256, visualize=False, use_z=True, split_z=False)



with open(args.json, "r") as f:
	data_description = json.load(f)

class_obj = {}
for key in data_description:
	class_obj[data_description[key]] = key

print(class_obj)

prediction_arr = []
confidence_arr = []
fiber_idx = []

filenames_df = pd.DataFrame()
nbFiber = surf.GetNumberOfCells()
print(nbFiber)
for i_cell in range(nbFiber):
	fiber_surf = fbf.ExtractFiber(surf, i_cell)


	surf_actor = fbf.GetNormalsActor(fiber_surf)
	flyby.addActor(surf_actor)

	img_np = flyby.getFlyBy()

	predict_np, confidence = model.predict(tf.expand_dims(img_np, axis=0))
	argmax = np.argmax(predict_np)

	prediction_arr.append(class_obj[argmax])
	confidence_arr.append(np.max(confidence))
	fiber_idx.append(i_cell)

	print("confidence", confidence, "class", class_obj[argmax])

	flyby.removeActors()

filenames_df["idx"] = fiber_idx
filenames_df["prediction"] = prediction_arr
filenames_df["confidence"] = confidence_arr

print("Writing:", args.out)
filenames_df.to_csv(args.out, index=False)


end_time = time.time()

print("Prediction time took:", (end_time - start_time), "seconds")
