from posixpath import dirname
import numpy as np
import tensorflow as tf
import argparse
import os
import glob
import sys
import itk

import fly_by_features as fbf
from utils import *
from collections import namedtuple
import nrrd



def main(args):
    img_fn_array = []

    if args.file:
        img_obj = {}
        img_obj["merged"] = args.file
        img_obj["out"] = args.out_dir
        img_fn_array.append(img_obj)

    if args.dir:
        normpath = os.path.normpath("/".join([args.dir, '**', '*']))
        for img_fn in glob.iglob(normpath, recursive=True):
            if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".vtk"]]:
                img_obj = {}
                img_obj["merged"] = img_fn
                img_obj["out"] = args.out_dir
                img_fn_array.append(img_obj)

    for img_obj in img_fn_array:
        merged_file = img_obj["merged"]
        out = img_obj["out"]
        print("Reading:", merged_file)

        filename = os.path.splitext(os.path.basename(merged_file))[0]
        ouput_filename = os.path.join(args.dir_ft, filename+".nrrd")

        # Run fly_by_feature code to get the feature
        split_obj = {}
        split_obj["surf"] = merged_file
        split_obj["split_z"] = 1
        split_obj["use_z"] = 1
        split_obj["rescale_features"] = 1
        split_obj["point_features"] = ["coords"]
        split_obj["point_features_concat"] = 1
        split_obj["spiral"] = 16
        split_obj["resolution"] = 256
        split_obj["out"] = ouput_filename

        split_obj["n_rotations"] = 0
        split_obj["subdivision"] = 0
        split_obj["radius"] = 4
        split_obj["turns"] = 4
        split_obj["visualize"] = 0
        split_obj["verbose"] = 0
        split_obj["random_rotation"] = False
        split_obj["save_label"] = False
        split_obj["property"] = None
        split_obj["concatenate"] = 1
        split_obj["model"] = None
        split_obj["extract_components"] = None
        split_obj["out_point_id"] = 0
        split_obj["uuid"] = False
        split_obj["ow"] = 1
        split_obj["zero"] = -100

        split_args = namedtuple("Split", split_obj.keys())(*split_obj.values())
        
        fbf.main(split_args)


        ImageType = itk.VectorImage[itk.F,3]

        img_read = itk.ImageFileReader[ImageType].New(FileName=ouput_filename)
        img_read.Update()
        feature_np = img_read.GetOutput()
        feature_np = itk.GetArrayViewFromImage(feature_np)
        feature_np = feature_np.reshape([1] +list(feature_np.shape))

        model = tf.keras.models.load_model(args.landmarks_model)
        prediction = model.predict(feature_np)

        prediction = np.reshape(prediction, [s for s in prediction.shape if s != 1])

        prediction[prediction[:,:,:,3] < 0, 3] = 0

        prediction_ft = GetImage(prediction)
        
        ouput = os.path.join(out, filename+"_pred.nrrd")
        print("Writing:", ouput)
        writer = itk.ImageFileWriter[ImageType].New(FileName=ouput)
        writer.SetInput(prediction_ft)
        writer.UseCompressionOn()
        writer.Update()




if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Prediction of the landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_parser = parser.add_argument_group("Input parameters")
    input_parser = input_parser.add_mutually_exclusive_group(required=True)
    input_parser.add_argument('--file', type=str, help='merged file')
    input_parser.add_argument('--dir', type=str, help='directory of the merged files')

    parser.add_argument('--dir_ft', type=str, help='directory to save the features', required=True)

    param_parser = parser.add_argument_group("Parameters")
    param_parser.add_argument('--landmarks_model', type=str, help='load the model', default='')

    output_parser = parser.add_argument_group("Output parameters")
    output_parser.add_argument('--out_dir', type=str, help='output_dir')

    args = parser.parse_args()

    main(args)
