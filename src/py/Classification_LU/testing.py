from model import *
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

import argparse
import os
import itk

import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser(description='Predict an input with a trained neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

in_group = parser.add_mutually_exclusive_group(required=True)
in_group.add_argument('--features', type=str, help='Input features to be predict as an Upper or Lower teeth')
in_group.add_argument('--dir_features', type=str, help='Input dir features to be predict as an Upper or Lower teeth')

parser.add_argument('--load_model', type=str, help='Saved model', required=True)
parser.add_argument('--display', type=bool, help='display the prediction', default=False)

args = parser.parse_args()

inputFeatures = args.features
inputdirFeatures = args.dir_features
loadModelPath = args.load_model



def ReadImage(fName, image_dimension=2, pixel_dimension=-1):
	if(image_dimension == 1):
		if(pixel_dimension != -1):
			ImageType = itk.Image[itk.Vector[itk.F, pixel_dimension], 2]
		else:
			ImageType = itk.VectorImage[itk.F, 2]
	else:
		if(pixel_dimension != -1):
			ImageType = itk.Image[itk.Vector[itk.F, pixel_dimension], image_dimension]
		else:
			ImageType = itk.VectorImage[itk.F, image_dimension]

	img_read = itk.ImageFileReader[ImageType].New(FileName=fName)
	img_read.Update()
	img = img_read.GetOutput()
	return img

def process_scan(path):
    img = ReadImage(path)
    img_np = itk.GetArrayViewFromImage(img)
    img_np = np.reshape(img_np, [s for s in img_np.shape if s != 1])
    return img_np



if (args.dir_features):
    features_paths = [os.path.join(inputdirFeatures, x) for x in os.listdir(inputdirFeatures) if not x.startswith(".")]
    features_name = [y for y in os.listdir(inputdirFeatures) if not y.startswith(".")]
    features = np.array([process_scan(path) for path in features_paths])

if (args.features):
    features_name = inputFeatures.split("/")[-1]
    features = np.array([process_scan(inputFeatures)])


model = load_model(loadModelPath)
predictions = model.predict(features)


y_true = []
for i in range(len(features_name)):
    if features_name[i][:1]=="L":
        y_true.append(0)

    if features_name[i][:1]=="U":
        y_true.append(1)

predictions[predictions>0.5]=1
predictions[predictions<=0.5]=0



if (args.display):
    target_names = ['Lower', 'Upper']
    print(classification_report(y_true, predictions, target_names=target_names))




