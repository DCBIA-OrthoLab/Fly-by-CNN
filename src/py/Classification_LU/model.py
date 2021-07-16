import os

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Bidirectional
from tensorflow.keras.optimizers import *


def LSTM_model():
    model = Sequential()

    # Two bidirectional LSTM layers 
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(64,512)))
    model.add(Bidirectional(LSTM(64)))

    # Sequence voting layer
    model.add(Dense(64))

    # Fully connected layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model





