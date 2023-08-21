import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import keras.backend as K


from piml_classes import GradientLayer

tf.compat.v1.disable_eager_execution()
"""
minimal working error to troubleshoot
"""
#%%
# x_input = keras.Input(shape=[1])
# y_input = keras.Input(shape=[1])
model = keras.Sequential([
    keras.layers.Dense(30, input_shape=[1]),
    keras.layers.Dense(30),
    keras.layers.Dense(1),
])
model.build()

# grad1 = tf.gradients(modelcall, x_input)[0]
# err = keras.losses.MeanSquaredError()(modelcall, y_input) + tf.reduce_mean(tf.square(grad1))
# err = keras.losses.MeanSquaredError()(modelcall, y_input)
# err = tf.reduce_mean(tf.square(modelcall - y_input))

@tf.function
def train_function(x, y):
    # x = tf.constant(x)
    # y = tf.constant(y)
    modelcall = model(x)
    err = tf.reduce_mean(tf.square(modelcall - y))
    return err
#%%
# training_model = keras.Model(
#     inputs=[x_input],
#     outputs=[modelcall],
# )
#%%
# @tf.function
def calculate_grads(x, y):
    modelcall = model(x)
    err = tf.reduce_mean(tf.square(modelcall -y))
    grads = tf.gradients(err, model.weights)
    return err, grads

# random input data
x = np.random.rand(64, 1)
y = np.random.rand(64, 1)

x = tf.constant(x, dtype=tf.float32)
y = tf.constant(y, dtype=tf.float32)

# err = train_function(x, y)
err, grads = calculate_grads(x, y)

# [grad, err] = calculate_grad(training_model, model, x, y)