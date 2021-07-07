from model import *
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import load_model

import os
import argparse
import datetime
import itk

import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser(description='Training a neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir_lower', type=str, help='Input dir for lower teeth for the training', required=True)
parser.add_argument('--dir_upper', type=str, help='Input dir for upper teeth for the training', required=True)
parser.add_argument('--save_model', type=str, help='Directory with saved model', required=True)
parser.add_argument('--log_dir', type=str, help='Directory for the log of the model', default="logModel")

args = parser.parse_args()

InputdirLower = args.dir_lower
InputdirUpper = args.dir_upper
savedModelPath = args.save_model
savedModel = os.path.join(savedModelPath,"nnLU_model_{epoch}")
logPath = args.log_dir


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



Lower_paths = [os.path.join(InputdirLower, x) for x in os.listdir(InputdirLower)]
Upper_paths = [os.path.join(InputdirUpper, x) for x in os.listdir(InputdirUpper)]

LowerTeeth = np.array([process_scan(path) for path in Lower_paths])
UpperTeeth = np.array([process_scan(path) for path in Upper_paths])

# Create the labels for the training
Lower_labels = np.array([0 for _ in range(len(LowerTeeth))])
Upper_labels = np.array([1 for _ in range(len(UpperTeeth))])

valSplit = round(len(UpperTeeth)*0.9)
x_train = np.concatenate((UpperTeeth[:valSplit], LowerTeeth[:valSplit]), axis=0)
x_val   = np.concatenate((UpperTeeth[valSplit:], LowerTeeth[valSplit:]), axis=0)
y_train = np.concatenate((Upper_labels[:valSplit], Lower_labels[:valSplit]), axis=0)
y_val   = np.concatenate((Upper_labels[valSplit:], Lower_labels[valSplit:]), axis=0)


print("Lower features: || Lower labels:")
print(np.shape(LowerTeeth), " || ", np.shape(Lower_labels))
print()
print("Upper features: || Upper labels:")
print(np.shape(UpperTeeth), " || ", np.shape(Upper_labels))
print()
print("ValTrainFeatures: || ValLabelFeatures: ")
print(np.shape(x_val), " || ", np.shape(y_val))
print()

model = LSTM_model()

epochs = 16
batch_size = 32
model_checkpoint = ModelCheckpoint(savedModel, monitor='loss',verbose=1, period=2)
log_dir = logPath+datetime.datetime.now().strftime("%Y_%d_%m-%H:%M:%S")
tensorboard_callback = TensorBoard(log_dir=log_dir,histogram_freq=1)
callbacks_list = [model_checkpoint, tensorboard_callback]

model.fit(
    x_train, y_train,
    batch_size=batch_size,
    validation_data=(x_val,y_val),
    epochs=epochs,
    shuffle=True,
    verbose=1,
    callbacks=callbacks_list,
)



