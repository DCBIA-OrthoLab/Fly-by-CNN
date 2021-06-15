from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import X

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import argparse
import json
import os
import glob
import sys
import pandas as pd
import itk
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Accuracy, MeanAbsoluteError, MeanSquaredError



parser = argparse.ArgumentParser(description='create RootCanal object from a segmented file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

in_group_features = parser.add_argument_group("gh")
in_group_features.add_argument('--train_dir', type=str, help='input file')
in_group_features.add_argument('--val_dir', type=str, help='input dir')

in_group_features.add_argument('--checkpoint', type=str, help='output dir', default='')

args = parser.parse_args()




class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def make_seg_model(drop_prob=0, argmax=False):

        x0 = tf.keras.Input(shape=[256, 256, 7])

        x = layers.Conv2D(32, (7, 7), strides=(2, 2), activation='tanh', padding='same')(x0)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU()(x)
        x = layers.AveragePooling2D()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='tanh', padding='same')(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d0 = x

        x = layers.Conv2D(128, (3, 3), strides=(2, 2), activation='tanh', padding='same')(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d1 = x

        x = layers.Conv2D(256, (3, 3), strides=(2, 2), activation='tanh', padding='same')(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d2 = x

        x = layers.Conv2D(512, (3, 3), strides=(2, 2), activation='tanh', padding='same')(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d3 = x

        x = layers.Conv2D(1024, (3, 3), strides=(2, 2), activation='tanh', padding='same')(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), activation='tanh', padding='same')(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Concatenate(axis=-1)([x, d3])
        x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='tanh', padding='same')(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Concatenate(axis=-1)([x, d2])
        x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='tanh', padding='same')(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Concatenate(axis=-1)([x, d1])
        x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='tanh', padding='same')(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Concatenate(axis=-1)([x, d0])
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='tanh', padding='same')(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), activation='tanh', padding='same')(x)

        x = layers.Conv2DTranspose(7, (3, 3), strides=(2, 2), padding='same')(x)
        

        seg_model = tf.keras.Model(inputs=x0, outputs=x)

        x0 = tf.keras.Input(shape=[None, 256, 256, 7])

        x = layers.TimeDistributed(seg_model)(x0)

        return tf.keras.Model(inputs=x0, outputs=x)

class DatasetGenerator:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.float32), 
            output_shapes=((16, 256, 256, 7), [16, 256, 256, 7])
            )

        self.dataset = self.dataset.batch(16)
        self.dataset = self.dataset.prefetch(48)

    def get(self):
        return self.dataset
    
    def generator(self):
        for ft, lb in zip(self.x, self.y):
            ImageType = itk.VectorImage[itk.F, 3]
            img_read = itk.ImageFileReader[ImageType].New(FileName=ft)
            img_read.Update()
            img_np = itk.GetArrayViewFromImage(img_read.GetOutput())
            img_np = img_np.reshape([16, 256, 256, 7])
            # print("Inputs shape:", np.shape(img_np), "min:", np.amin(img_np), "max:", np.amax(img_np), "unique:", len(np.unique(img_np)))

            ImageType = itk.VectorImage[itk.F, 3]
            img_read_seg = itk.ImageFileReader[ImageType].New(FileName=lb)
            img_read_seg.Update()
            seg_np = itk.GetArrayViewFromImage(img_read_seg.GetOutput())
            seg_np = seg_np.reshape([16, 256, 256, 7])
            # print("Labels shape:", np.shape(seg_np), "min:", np.amin(seg_np), "max:", np.amax(seg_np), "unique:", len(np.unique(seg_np)))

            yield img_np, seg_np


gpus_index = [0]
print("Using gpus:", 0)
gpus = tf.config.list_physical_devices('GPU')


if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        gpu_visible_devices = []
        for i in gpus_index:
            gpu_visible_devices.append(gpus[i])
        
        print(bcolors.OKGREEN, "Using gpus:", gpu_visible_devices, bcolors.ENDC)

        tf.config.set_visible_devices(gpu_visible_devices, 'GPU')
        # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(bcolors.FAIL, e, bcolors.ENDC)
# else:
#     # strategy = tf.distribute.get_strategy() 

# df = pd.read_csv("/ASD/juan_flyby/DCBIA/Upper_Lower_tooth_crown_surface_samples.csv")

# train_df, valid_df = train_test_split(df, test_size=0.1)

# train_df = pd.read_csv("/ASD/juan_flyby/DCBIA/Upper_Lower_tooth_crown_surface_samples_train_train.csv")
# valid_df = pd.read_csv("/ASD/juan_flyby/DCBIA/Upper_Lower_tooth_crown_surface_samples_train_val.csv")



TrainDir = args.train_dir
ValDir = args.val_dir

InputdirTrain = [os.path.join(TrainDir,'features')] 
InputdirLabel = [os.path.join(TrainDir,'labels')]
InputdirValTrain = [os.path.join(TrainDir,'features')]
InputdirValLabel = [os.path.join(TrainDir,'labels')]

print("Loading paths...")
# Input files and labels
input_paths = sorted([file for file in [os.path.join(dir, fname) for dir in InputdirTrain for fname in os.listdir(dir)] if not os.path.basename(file).startswith(".")])
label_paths = sorted([file for file in [os.path.join(dir, fname) for dir in InputdirLabel for fname in os.listdir(dir)] if not os.path.basename(file).startswith(".")])
# Folder with the validations scans and labels
ValInput_paths = sorted([file for file in [os.path.join(dir, fname) for dir in InputdirValTrain for fname in os.listdir(dir)] if not os.path.basename(file).startswith(".")])
ValLabel_paths = sorted([file for file in [os.path.join(dir, fname) for dir in InputdirValLabel for fname in os.listdir(dir)] if not os.path.basename(file).startswith(".")])



dataset = DatasetGenerator(input_paths, label_paths).get()
dataset_validation = DatasetGenerator(ValInput_paths, ValLabel_paths).get()

# with strategy.scope():

batch_size = 8

model = make_seg_model()
model.summary()

optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsoluteError(), metrics=[[MeanSquaredError()]]) ###, Precision(), Recall(), MeanAbsoluteError(), MeanSquaredError()]])

# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
# checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3, checkpoint_name=modelname)

checkpoint_path = args.checkpoint

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# model_loss_error_callback = LossAndErrorPrintingCallback()

model.fit(dataset, validation_data=dataset_validation, epochs=200, verbose=2, callbacks=[model_early_stopping_callback, model_checkpoint_callback])
