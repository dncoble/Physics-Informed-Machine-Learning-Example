"""
Troubleshooting making tf.gradients work
"""
# import numpy as np
# import tensorflow as tf
# import tensorflow.keras as keras
# import matplotlib.pyplot as plt
# import keras.backend as K

# class GradientLayer(keras.layers.Layer):
    
#     def __init__(self,
#                trainable=True,
#                name=None,
#                dtype=None,
#                dynamic=False,
#                **kwargs):
#         super().__init__(trainable=True,
#                        name=None,
#                        dtype=None,
#                        dynamic=False,
#                        **kwargs)
    
#     @tf.function
#     def call(self, inputs):
#         print(tf.executing_eagerly())
#         y = inputs[0]
#         x = inputs[1]
#         return K.gradients([y], x)

# model = keras.models.Sequential([
#         keras.layers.Dense(10, activation='sigmoid'),
#         keras.layers.Dense(10, activation='sigmoid'),
#         keras.layers.Dense(1)
# ])

# x_inp = keras.layers.Input(batch_input_shape=[1, 1])

# m = model(x_inp)

# g = GradientLayer()([m, x_inp])

# model1 = keras.models.Model(
#     inputs=[x_inp],
#     outputs=[g]
# )
# #%%
# @tf.function
# def execute(a, b):
#     print(a.graph)
#     return b@a

# @tf.function
# def example(c, a):
#     print(a.graph)
#     return tf.gradients(c, [a], unconnected_gradients='zero')

# g = tf.Graph()
# with g.as_default():
#     a = tf.ones([1, 2])
#     b = tf.ones([2, 1])
#     c = execute(a, b)
#     r = example(c, a)
#%%
import tensorflow as tf


import tensorflow.keras as keras
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

# Creating a model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Concatenate,
    Input,
    Lambda,
)

# Custom activation function
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

import tensorboard

layer_width = 200
dense_layer_number = 3

tf.compat.v1.disable_eager_execution()

@tf.function
def lambda_gradient(args):
    layer = args[0]
    inputs = args[1]
    return tf.gradients(layer, inputs)[0]

class GradientLayer(keras.layers.Layer):
    
    def __init__(self,
               trainable=True,
               name=None,
               dtype=None,
               dynamic=False,
               **kwargs):
        super().__init__(trainable=True,
                       name=None,
                       dtype=None,
                       dynamic=False,
                       **kwargs)
    
    def call(self, inputs):
        y = inputs[0]
        x = inputs[1]
        return tf.gradients(y, x)[0]

# Input is a 2 dimensional vector
inputs = tf.keras.Input(shape=(2,), name="coordinate_input")

# Build `dense_layer_number` times a dense layers of width `layer_width`
stream = inputs
for i in range(dense_layer_number):
    stream = Dense(
        layer_width, activation="relu", name=f"dense_layer_{i}"
    )(stream)

# Build one dense layer that reduces the 200 nodes to a scalar output
scalar = Dense(1, name="network_to_scalar")(stream)

# # Take the gradient of the scalar w.r.t. the model input
# gradient = Lambda(lambda_gradient, name="gradient_layer")([scalar, inputs])

gradient = GradientLayer()([scalar, inputs])

# Combine them to form the model output
concat = Concatenate(name="concat_scalar_gradient")([scalar, gradient])

# Wrap everything in a model
model = tf.keras.Model(inputs=inputs, outputs=concat)

loss = "MSE"
optimizer = "Adam"

# And compile
model.compile(loss=loss, optimizer=optimizer)

# fake data
x = np.random.rand(64,2)
y = model.predict(x)