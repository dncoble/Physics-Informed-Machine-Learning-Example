import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
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
    def __init__(self, g1, g2, t,
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        self.g1 = g1
        self.g2 = g2
        self.t = t
        super().__init__(trainable=True,
                       name=None,
                       dtype=None,
                       dynamic=False,
                       **kwargs)
    
    def call(self, inputs):
        w = inputs[0]
        x = inputs[1]
        dwdt = tf.gradients(w, self.t)
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