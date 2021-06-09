import argparse
from collections import namedtuple

import itk
import numpy as np
import tensorflow as tf

import fly_by_features


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
    img_np = np.reshape(img_np,(1,)+img_np.shape)
    return img_np



def main(args):
    # We generate an object with the corresponding parameters
    split_obj = {}
    split_obj["surf"] = args.surf
    split_obj["subdivision"] = 0
    split_obj["spiral"] = 64
    split_obj["turns"] = 4
    split_obj["radius"] = 4
    split_obj["resolution"] = 256
    split_obj["visualize"] = 0
    split_obj["use_z"] = 1
    split_obj["split_z"] = 0
    split_obj["point_features"] = None
    split_obj["random_rotation"] = False
    split_obj["save_label"] = False
    split_obj["property"] = None
    split_obj["concatenate"] = 1
    split_obj["model"] = args.model_feature
    split_obj["out"] = args.out_feature

    # We convert the dictionary to a namedtuple, a.k.a, python object, i.e., argparse object
    split_args = namedtuple("Split", split_obj.keys())(*split_obj.values())
    # Call the main of the script
    fly_by_features.main(split_args)


    model = tf.keras.models.load_model(args.model_LU)
    return model.predict(process_scan(args.out_feature))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict an input feature as an upper or lower jaw with a trained neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--surf', type=str, help='Pre-teeth surface to have the feature', required=True)
    parser.add_argument('--spiral', type=str, help='methods to acquire the feature (sub or spiral)', required=True)
    parser.add_argument('--model_feature', type=str, help='VGG19 model', required=True)
    parser.add_argument('--model_LU', type=str, help='LU model', required=True)
    parser.add_argument('--out_feature', type=str, help='save the feature', required=True)

    args = parser.parse_args()

    main(args)











