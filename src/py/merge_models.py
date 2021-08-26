import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
import os
import json


x0 = layers.Input(shape=[None, 256, 256, 3])


model_feat = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_tensor=layers.Input(shape=[256, 256, 3]), pooling='max')
model_feat.summary()


model = tf.keras.models.load_model('/SIRIUS_STOR/lumargot/data/data_spiral/spiral_16/model/saved/vgg19', custom_objects={'tf': tf})
model.summary()


x = layers.TimeDistributed(model_feat)(x0)
x = model(x)

model_full = tf.keras.Model(inputs=x0, outputs=x)

model_full.summary()
model_full.save("/SIRIUS_STOR/lumargot/data/data_spiral/spiral_16/model/saved/vgg19_full_model")
