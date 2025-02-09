import tensorflow as tf
from setuptools.extern import names
from tensorflow.keras.layers import Dense
from  tensorflow.keras import Sequential


def build_model():
    model_1 = Sequential([
        Dense(units=25, activation='relu'),
        Dense(units=15, activation='relu'),
        Dense(units=1),
    ], name='model_1')

    model_2 = Sequential([
        Dense(units=20, activation='relu'),
        Dense(units=12, activation='relu'),
        Dense(units=12, activation='relu'),
        Dense(units=20, activation='relu'),
        Dense(units=1),
    ],name='model_2')

    model_3 = Sequential([
        Dense(units=32, activation='relu'),
        Dense(units=16, activation='relu'),
        Dense(units=8, activation='relu'),
        Dense(units=4, activation='relu'),
        Dense(units=12, activation='relu'),
        Dense(units=1),
    ], name='model_3')

    return [model_1, model_2, model_3]