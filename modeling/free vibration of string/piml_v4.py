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
Trying to remove errors and have everything run in graph mode.
"""
#%%
class WeightSum(keras.layers.Layer):
    '''
    pass weights
    '''
    def __init__(self, *args,
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        self.args = args
        super().__init__(trainable=True,
                       name=None,
                       dtype=None,
                       dynamic=False,
                       **kwargs)
    
    def call(self, inputs):
        rtrn = inputs[0]*self.args[0]
        for arg, inp in zip(self.args[1:], inputs[1:]):
            rtrn += tf.reduce_mean(inp)*arg
        return rtrn
'''
If you're using this function you're doing something wrong
'''
class IdentityLoss(keras.losses.Loss):
    
    
    
    def call(self, y_true, y_pred):
        return tf.reduce_mean(y_pred, axis=-1)
#%% describe system
# physical constants
T = 1
L = 1
lam = 1
# g1 - initial position
M = 1/(2*np.pi)
@tf.function
def g1(x):
    return tf.math.sin(M*x)
# g2 - initial velocity
zeros = tf.zeros(shape=[64, 1])
@tf.function
def g2(x):
    return zeros
# training constants
rhof = 1
rho0 = 10
rhob = 1
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

# enforcing wave equation on interior
concat_interior = keras.layers.Concatenate()([x_interior, t_interior])

w_int = model(concat_interior)
# grads_concat = tf.gradients(w_int, concat_interior)
# grads2_concat = tf.gradients(grads_concat, concat_interior)[0]
# d2wdx2 = grads2_concat[:,0:1]
# d2wdt2 = grads2_concat[:,1:2]
dwdx = GradientLayer()([w_int, x_interior])
d2wdx2 = GradientLayer()([dwdx, x_interior])
dwdt = GradientLayer()([w_int, t_interior])
d2wdt2 = GradientLayer()([dwdt, t_interior])

e_int = tf.square(lam*d2wdx2 + d2wdt2)

# enforce initial condition loss
t_initial = tf.zeros(x_initial.shape)
concat = keras.layers.Concatenate()([x_initial, t_initial])

w_init = model(concat)
dwdt = GradientLayer()([w_init, t_initial])

e_init = tf.square(w_init - g1(x_initial)) + tf.square(dwdt - g2(x_initial))

# enforce boundary condition loss
x_boundary0 = tf.zeros(t_boundary.shape)
x_boundary1 = tf.ones(t_boundary.shape)
concat0 = keras.layers.Concatenate()([x_boundary0, t_boundary])
concat1 = keras.layers.Concatenate()([x_boundary1, t_boundary])

w_boundary0 = model(concat0)
w_boundary1 = model(concat1)
e_bound = tf.square(w_boundary0) + tf.square(w_boundary1)

loss = WeightSum(rhof, rho0, rhob)([e_int, e_init, e_bound])

training_model = keras.Model(
    inputs=[x_interior, t_interior, x_initial, t_boundary],
    outputs=[loss]
)
#%%
optimizer = keras.optimizers.legacy.Adam(learning_rate=.01)
identityloss = IdentityLoss()
training_model.compile(optimizer=optimizer, loss=identityloss)
#%%
num_epochs = 200
n_batch = 64 # use batches of 64
Nf = 640000 # number of interior points
N0 = 640000 # number of initial points
Nb = 640000 # number of boundary points
# create stochastic training points
x_int = np.random.rand(Nf, 1)
t_int = np.random.rand(Nf, 1)
x_init = np.random.rand(N0, 1)
t_bound = np.random.rand(Nb, 1)
y_always = np.zeros([N0, 1]) # part of workaround

# very lenient early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0,
    patience=15,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

training_model.fit(
    x=[x_int, t_int, x_init, t_bound],
    y=y_always,
    batch_size=64,
    epochs=num_epochs,
    callbacks=[early_stopping],
    shuffle=True,
)
#%% plot the approximating model

# make a new model copying the weights from the original, may not be necessary
model1 = keras.models.Sequential([
    keras.layers.Dense(30, activation='sigmoid', use_bias=True, trainable=True, input_shape=[2]),
    keras.layers.Dense(30, activation='sigmoid', use_bias=True, trainable=True),
    keras.layers.Dense(1, activation='sigmoid', use_bias=True, trainable=True),
])
for layer1, layer2 in zip(model1.layers, model.layers):
    layer1.set_weights(layer2.get_weights())

model = model1

x_axis = np.linspace(0, 1, num=500)
t_axis = np.linspace(0, 1, num=500)

x_mesh, t_mesh = np.meshgrid(x_axis, t_axis)

coords = np.vstack((x_mesh.reshape(-1), t_mesh.reshape(-1))).T

w_pred = model.predict(coords)
w_pred = w_pred.reshape(500, 500)

fig = plt.figure(figsize=(7, 2))
pc = plt.pcolormesh(x_mesh, t_mesh, w_pred)
fig.colorbar(pc)
plt.xlabel(r'$t$ (ul)')
plt.ylabel(r'$x$ (ul)')
plt.savefig("physics-informed results v2.png", dpi=500)