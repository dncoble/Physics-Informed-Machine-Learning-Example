import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

tf.compat.v1.disable_eager_execution()
"""
tensorflow/keras classes for physics-informed machine learning
"""
#%% compute at interior, initial, and boundary
# needed for dispatching
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

class ModelInterior(keras.layers.Layer):
    
    '''
    wraps model and computes for interior
    '''
    def __init__(self, model,
               trainable=True,
               name=None,
               dtype=None,
               dynamic=False,
               **kwargs):
        self.model = model
        super().__init__(trainable=True,
                       name=None,
                       dtype=None,
                       dynamic=False,
                       **kwargs)
    
    def call(self, inputs):
        x = inputs[0]
        t = inputs[1]
        xt = tf.concat([x, t], -1)
        return self.model(xt)
        
class ModelInitial(keras.layers.Layer):
    '''
    wraps model and computes for only t=0
    '''
    def __init__(self, model, 
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        self.model = model
        super().__init__(trainable=True,
                       name=None,
                       dtype=None,
                       dynamic=False,
                       **kwargs)
    
    '''
    returns also t (needed for gradient calculation downstream)
    '''
    def call(self, inputs):
        x = inputs
        t = tf.zeros(x.shape)
        xt = tf.concat([x, t], -1)
        
        return [self.model(xt), t]

class ModelBoundary(keras.layers.Layer):
    '''
    wraps model and computes for x=0, x=1
    '''
    def __init__(self, model,
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        self.model = model
        super().__init__(trainable=True,
                       name=None,
                       dtype=None,
                       dynamic=False,
                       **kwargs)
    
    '''
    returns at x=0 and x=1
    '''
    def call(self, inputs):
        t = inputs
        x0 = tf.zeros(t.shape)
        x1 = tf.ones(t.shape)
        x0t = tf.concat([x0, t], -1)
        x1t = tf.concat([x1, t], -1)
        
        w0 = self.model(x0t)
        w1 = self.model(x1t)
        
        return [w0, w1]

#%% losses
class WaveLoss(keras.layers.Layer):
    '''
    x: tensor reference to displacement input
    t: tensor reference to time input
    rhof: weight of interior 
    '''
    def __init__(self, x, t, lam,
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        self.lam = np.array([lam], dtype=np.float32)
        self.x = x
        self.t = t
        super().__init__(trainable=True,
                       name=None,
                       dtype=None,
                       dynamic=False,
                       **kwargs)
    
    def build(self, input_shape):
        # self.input_shape = input_shape # should be [None, 2]
        # make lam a tf variable
        self.lam = tf.Variable(initial_value=self.lam, trainable=False)
    
    def call(self, inputs):
        w = inputs
        dwdx = GradientLayer()([w, self.x])
        d2wdx2 = GradientLayer()([dwdx, self.x])
        dwdt = GradientLayer()([w, self.t])
        d2wdt2 = GradientLayer()([dwdt, self.t])
        
        return self.lam*d2wdx2 - d2wdt2

class InitialLoss(keras.layers.Layer):
    '''
    g1: tf function for initial displacement
    g2: tf function for initial velocity
    t: tf tensor for time input
    '''
    def __init__(self, g1, g2,
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        self.g1 = g1
        self.g2 = g2
        super().__init__(trainable=True,
                       name=None,
                       dtype=None,
                       dynamic=False,
                       **kwargs)
    
    def call(self, inputs):
        w, x, t = inputs
        dwdt = tf.gradients(w, t)
        g1x = self.g1(x)
        g2x = self.g2(x)
        
        return tf.square(w - g1x) + tf.square(dwdt - g2x)
    
class BoundaryLoss(keras.layers.Layer):
    
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
    '''
    boundary values should always be zero, no constraint on derivatives
    '''
    def call(self, inputs):
        w0 = inputs[0]
        w1 = inputs[1]
        
        return tf.square(w0) + tf.square(w1)

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
        return y_pred

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
@tf.function
def g2(x):
    return tf.zeros(x.shape)

rhof = 1
rho0 = .2
rhob = .2
batch_size = 64
# number of points at initial, boundary, and interior during each epoch. Each must be a multiple of the batch size. 
Nf = 4096 # number of interior points
N0 = 4096 # number of initial points
Nb = 4096 # number of boundary points
num_epochs = 200
learning_rate=.01
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
x_initial = keras.layers.Input(batch_input_shape=[64, 1], name='x_initial')
t_boundary = keras.layers.Input(batch_input_shape=[64, 1], name='t_boundary')

model_interior = ModelInterior(model)([x_interior, t_interior])
model_initial, t_initial = ModelInitial(model)(x_initial)
model_boundary = ModelBoundary(model)(t_boundary)

# wave layer enforcing differential equation
wave_loss = WaveLoss(x_interior, t_interior, lam)(model_interior)
# enforcing initial conditions
initial_loss = InitialLoss(g1, g2)([model_initial, x_initial, t_initial])
# enforcing boundary conditions
boundary_loss = BoundaryLoss()(model_boundary)
#total loss
loss = WeightSum(rhof, rho0, rhob)([wave_loss, initial_loss, boundary_loss])

# gradient of loss w.r.t. approximating function weights
grads = GradientLayer()([loss, model.weights])
training_model = keras.Model(
    inputs=[x_interior, t_interior, x_initial, t_boundary],
    outputs=[loss]
)

# optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# # keep results for plotting
# train_loss_results = []
# train_accuracy_results = []

# n_batch = 64 # use batches of 64
# for epoch in range(num_epochs):
#     epoch_loss_avg = tf.keras.metrics.Mean()
#     # create stochastic training points
#     x_int = np.random.rand(Nf)
#     t_int = np.random.rand(Nf)
#     x_init = np.random.rand(N0)
#     t_bound = np.random.rand(Nb)
    
#     x_int = x_int.reshape(-1, n_batch, 1)
#     t_int = t_int.reshape(-1, n_batch, 1)
#     x_init = x_init.reshape(-1, n_batch, 1)
#     t_bound = t_bound.reshape(-1, n_batch, 1)
    
#     for x_int1, t_int1, x_init1, t_bound1 in zip(x_int, t_int, x_init, t_bound):
#         # Optimize the model
#         loss_value, grads = training_model(x_int, t_int, x_init, t_bound)
#         optimizer.apply_gradients(zip(grads, training_model.trainable_variables))
        
#         # Track progress
#         epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        
#         # End epoch
#         train_loss_results.append(epoch_loss_avg.result())
    
#     if(epoch % 1 == 0):
#         print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))