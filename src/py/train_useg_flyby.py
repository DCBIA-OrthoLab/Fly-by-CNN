from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import json
import os
import glob
import sys
import pandas as pd
import itk
from sklearn.model_selection import train_test_split

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

def IoU(y_true, y_pred):

    # weights = np.array([0.5, 1, 1, 1, 2])
    y_true = tf.cast(y_true, tf.float32)
    num_classes = tf.shape(y_true)[-1]

    y_true = tf.reshape(y_true, [-1, num_classes])
    y_pred = tf.reshape(y_pred, [-1, num_classes])
    intersection = 2.0*tf.reduce_sum(y_true * y_pred, axis=0) + 1
    union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0) + 1.

    # y_true = tf.reshape(y_true, [-1])
    # y_pred = tf.reshape(y_pred, [-1])

    # intersection = 2.0*tf.reduce_sum(y_true * y_pred) + 1
    # union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1.

    iou = 1.0 - intersection / union
    # iou *= weights

    return tf.reduce_sum(iou)

def make_seg_model():

        drop_prob=0.1

        x0 = tf.keras.Input(shape=[512, 512, 4])

        x = tf.keras.layers.GaussianNoise(1.0)(x0)

        x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d0 = x

        x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d1 = x

        x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d2 = x

        x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d3 = x

        x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d4 = x

        x = layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d5 = x

        x = layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Concatenate(axis=-1)([x, d5])
        x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Concatenate(axis=-1)([x, d4])
        x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Concatenate(axis=-1)([x, d3])
        x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Concatenate(axis=-1)([x, d2])
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Concatenate(axis=-1)([x, d1])
        x = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Concatenate(axis=-1)([x, d0])
        x = layers.Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='same', activation='softmax')(x)

        # if argmax:
        #     x = tf.expand_dims(tf.math.argmax(x, axis=-1), axis=-1)
        #     x = tf.cast(x, dtype=tf.uint8)

        seg_model = tf.keras.Model(inputs=x0, outputs=x)

        x0 = tf.keras.Input(shape=[None, 512, 512, 4])

        x = layers.TimeDistributed(seg_model)(x0)

        return tf.keras.Model(inputs=x0, outputs=x)

class DatasetGenerator:
    def __init__(self, df):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32), 
            output_shapes=((12, 512, 512, 4), [12, 512, 512, 4])
            )

        self.dataset = self.dataset.batch(16)
        self.dataset = self.dataset.prefetch(48)


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = row["img"]
            seg = row["seg"]

            ImageType = itk.VectorImage[itk.F, 3]
            img_read = itk.ImageFileReader[ImageType].New(FileName=img)
            img_read.Update()
            img_np = itk.GetArrayViewFromImage(img_read.GetOutput())
            img_np = img_np.reshape([12, 512, 512, 4])

            ImageType = itk.Image[itk.UC, 3]
            img_read_seg = itk.ImageFileReader[ImageType].New(FileName=seg)
            img_read_seg.Update()
            seg_np = itk.GetArrayViewFromImage(img_read_seg.GetOutput())
            seg_np = seg_np.reshape([12, 512, 512])

            yield img_np, tf.one_hot(seg_np, 4)


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

train_df = pd.read_csv("/ASD/juan_flyby/DCBIA/Upper_Lower_tooth_crown_surface_samples_512_train_train.csv")
valid_df = pd.read_csv("/ASD/juan_flyby/DCBIA/Upper_Lower_tooth_crown_surface_samples_512_train_val.csv")

dataset = DatasetGenerator(train_df).get()
dataset_validation = DatasetGenerator(valid_df).get()

# with strategy.scope():

batch_size = 8

model = make_seg_model()
model.summary()

optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, loss=IoU, metrics=["acc"])

# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
# checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3, checkpoint_name=modelname)

checkpoint_path = "/ASD/juan_flyby/DCBIA/train/Oriented_Upper_tooth_crown_surface_clean_512"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# model_loss_error_callback = LossAndErrorPrintingCallback()

model.fit(dataset, validation_data=dataset_validation, epochs=200, callbacks=[model_early_stopping_callback, model_checkpoint_callback])
