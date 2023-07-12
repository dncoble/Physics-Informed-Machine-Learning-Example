import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import keras.backend as K

from piml_classes import ModelInterior, ModelInitial, ModelBoundary, WaveLoss,\
    InitialLoss, BoundaryLoss, WeightSum

from piml_classes import GradientLayer

tf.compat.v1.disable_eager_execution()

"""
Free vibration of a string as described on pg. 49 of the paper.

Solving a wave equation using PIML
"""
#%% describe system
# physical constants
T = 1
L = 1
lam = 1
# g1 - initial position
@tf.function
def g1(x):
    M = 1/(2*np.pi)
    return .1*tf.math.sin(M*x)
# g2 - initial velocity
zeros = tf.zeros(shape=[64, 1])
@tf.function
def g2(x):
    return zeros
# training constants
rhof = 1
rho0 = .2
rhob = .2
#%%
# define model
# dense approximating NN
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='sigmoid', use_bias=True, trainable=True),
    keras.layers.Dense(30, activation='sigmoid', use_bias=True, trainable=True),
    keras.layers.Dense(1, activation='sigmoid', use_bias=True, trainable=True),
])

# build computation graph
x_interior = keras.layers.Input(batch_input_shape=[64, 1], name='x_interior')
t_interior = keras.layers.Input(batch_input_shape=[64, 1], name='t_interior')
# initial and boundary inputs
x_initial = keras.layers.Input(batch_input_shape=[64, 1], name='x_inital')
t_boundary = keras.layers.Input(batch_input_shape=[64, 1], name='t_boundary')

model_interior = ModelInterior(model)([x_interior, t_interior])
model_initial = ModelInitial(model)(x_initial)
model_boundary = ModelBoundary(model)(t_boundary)

# wave layer enforcing differential equation
wave_loss = WaveLoss(x_interior, t_interior, lam)(model_interior)
# enforcing initial conditions
initial_loss = InitialLoss(g1, g2, x_initial)(model_initial)
# enforcing boundary conditions
boundary_loss = BoundaryLoss()(model_boundary)
#total loss
loss = WeightSum(rhof, rho0, rhob)([wave_loss, initial_loss, boundary_loss])
# loss_grads = GradientLayer()([loss, model.trainable_variables])

training_model = keras.Model(
    inputs=[x_interior, t_interior, x_initial, t_boundary],
    outputs=[loss]
)

# loss and optimizer don't matter since doing custom training
training_model.compile(loss="MSE", optimizer="Adam")
#%% custom training
def grad(training_model, inputs):
    with tf.GradientTape() as tape:
        loss_value = training_model(inputs)
    return loss_value, tape.gradient(loss_value, training_model.trainable_variables)

optimizer = keras.optimizers.Adam(learning_rate=.01)

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 200
n_batch = 64 # use batches of 64
Nf = 4096 # number of interior points
N0 = 4096 # number of initial points
Nb = 4096 # number of boundary points
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    # create stochastic training points
    x_int = np.random.rand(Nf)
    t_int = np.random.rand(Nf)
    x_init = np.random.rand(N0)
    t_bound = np.random.rand(Nb)
    
    x_int = x_int.reshape(-1, n_batch, 1)
    t_int = t_int.reshape(-1, n_batch, 1)
    x_init = x_init.reshape(-1, n_batch, 1)
    t_bound = t_bound.reshape(-1, n_batch, 1)
    
    for x_int1, t_int1, x_init1, t_bound1 in zip(x_int, t_int, x_init, t_bound):
        # Optimize the model
        loss_value, grads = grad(training_model, [x_int1, t_int1, x_init1, t_bound1])
        optimizer.apply_gradients(zip(grads, training_model.trainable_variables))
        
        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        
        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
    
    if(epoch % 1 == 0):
        print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
#%% extract the approximating model