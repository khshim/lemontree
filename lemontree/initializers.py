# Kyuhong Shim 2016
"""
Initializers set initial values for shared variables.
No returns.
"""

import numpy as np
import theano


class BaseInitializer(object):

    def _generate(self, shape):
        """Generate random (or not) values for given shape."""
        raise NotImplementedError('Abstract class method')

    def initialize(self, params):
        """Initialize parameters."""
        if not isinstance(params, list):
            params = [params]
        for var in params:
            shape = var.get_value(borrow=True).shape
            var.set_value(self._generate(shape))


class Constant(BaseInitializer):

    def __init__(self, constant):
        self.constant = np.asarray(constant)

    def _generate(self, shape):
        return np.asarray(np.ones(shape) * self.constant, dtype=theano.config.floatX)


class Normal(BaseInitializer):

    def __init__(self, mean=0, std=0.001):
        self.mean = mean
        self.std = std

    def _generate(self, shape):
        return np.random.normal(self.mean, self.std, shape).astype(theano.config.floatX)


class Uniform(BaseInitializer):

    def __init__(self, low=-0.001, high=0.001):
        self.low = low
        self.high = high

    def _generate(self, shape):
        return np.random.uniform(self.low, self.high, shape).astype(theano.config.floatX)


class Identity(BaseInitializer):

    def __init__(self, gain=1):
        self.gain = gain

    def _generate(self, shape):
        if len(shape) != 2:
            raise ValueError('Dimension mismatch')
        rows, cols = shape
        return np.eye(rows, cols, dtype=theano.config.floatX)


class GlorotNormal(BaseInitializer):

    def __init__(self, mean=0):
        self.mean = mean

    def _generate(self, shape):
        n_in = shape[0]
        n_out = shape[1]
        n_res = np.prod(shape[2:])
        std = np.sqrt(2.0 / ((n_in + n_out) * n_res))
        return np.random.normal(self.mean, std, shape).astype(theano.config.floatX)


class HeNormal(BaseInitializer):

    def __init__(self, mean=0):
        self.mean = mean

    def _generate(self, shape):
        n_in = shape[0]
        n_out = shape[1]
        n_res = np.prod(shape[2:])
        std = np.sqrt(2.0 / (n_in * n_res))
        return np.random.normal(self.mean, std, shape).astype(theano.config.floatX)


class GlorotUniform(BaseInitializer):

    def __init__(self, mean=0):
        self.mean = mean

    def _generate(self, shape):
        n_in = shape[0]
        n_out = shape[1]
        n_res = np.prod(shape[2:])
        std = np.sqrt(6.0 / ((n_in + n_out) * n_res))
        return np.random.uniform(mean - std, mean + std, shape).astype(theano.config.floatX)


class Orthogonal(BaseInitializer):

    def __init__(self, scale=1.1):
        self.scale = scale

    def _generate(self, shape):
        n_in = shape[0]
        n_flat = np.prod(shape[1:])
        a = np.random.normal(0.0, 1.0, (n_in, n_flat))
        u, _, v = np.linalg.svd(a, full_matrices=False)
        if u.shape == (n_in, n_flat):
            q = u
        else:
            q = v
        q = np.reshape(q, shape)
        return np.asarray(self.scale * q[:shape[0], :shape[1]], dtype=theano.config.floatX)
