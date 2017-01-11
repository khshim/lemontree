"""
This code includes initialization classes for shared variables.
At this stage, theano shared variable is not related.
Only numpy value is created and returned.
"""

import numpy as np
import theano


class BaseInitializer(object):
    """
    This class defines abstract base class for initializers.
    """
    def generate(self, shape):
        """
        This function generates random (or not) values for given shape.

        Parameters
        ----------
        shape: tuple
            a tuple of required shape information of tensor (ndarray).

        Returns
        -------
        ndarray
            an array with given shape and desired values.
        """
        raise NotImplementedError('Abstract class method')

    def initialize_params(self, params):
        """
        This function initializes parameters.

        Parameters
        ----------
        params: list
            a list of (shared variable) parameters.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(params, list), '"params" should be a list type.'
        # initialize
        for pp in params:
            shape = pp.get_value(borrow=True).shape
            pp.set_value(self.generate(shape))


class Constant(BaseInitializer):
    """
    This class generates a tensor filled with given constant.
    """
    def __init__(self, constant=0):
        """
        This function initializes the class.

        Parameters
        ----------
        constant: float, default: 0
            a float value to fill the tensor.

        Returns
        -------
        None.
        """
        # set members
        self.constant = np.asarray(constant)

    def generate(self, shape):
        """
        This function overrides the parents' one.
        Generate random (or not) values for given shape.

        Parameters
        ----------
        shape: tuple
            a tuple of required shape information of tensor (ndarray).

        Returns
        -------
        ndarray
            an array with given shape and desired values.
        """
        # generate
        return np.asarray(np.ones(shape) * self.constant, dtype=theano.config.floatX)


class Normal(BaseInitializer):
    """
    This class generates a tensor with Gaussian normal sampling.
    """
    def __init__(self, mean=0, std=0.001):
        """
        This function initializes the class.

        Parameters
        ----------
        mean: float, default: 0
            a float value for Gaussian sampling mean.
        std: float, default: 0.001
            a float value for Gaussain sampling standard deviation.

        Returns
        -------
        None.
        """
        # set members
        self.mean = mean
        self.std = std

    def generate(self, shape):
        """
        This function overrides the parents' one.
        Generate random (or not) values for given shape.

        Parameters
        ----------
        shape: tuple
            a tuple of required shape information of tensor (ndarray).

        Returns
        -------
        ndarray
            an array with given shape and desired values.
        """
        # generate
        return np.random.normal(self.mean, self.std, shape).astype(theano.config.floatX)


class Uniform(BaseInitializer):
    """
    This class generates a tensor with uniform sampling.
    """
    def __init__(self, low=-0.001, high=0.001):
        """
        This function initializes the class.

        Parameters
        ----------
        low: float, default: -0.001
            a float value for low boundary of the range.
        high: float, default: 0.001
            a float value for high boundary of the range.

        Returns
        -------
        None.
        """
        # set members
        self.low = low
        self.high = high

    def generate(self, shape):
        """
        This function overrides the parents' one.
        Generate random (or not) values for given shape.

        Parameters
        ----------
        shape: tuple
            a tuple of required shape information of tensor (ndarray).

        Returns
        -------
        ndarray
            an array with given shape and desired values.
        """
        # generate
        return np.random.uniform(self.low, self.high, shape).astype(theano.config.floatX)


class Identity(BaseInitializer):
    """
    This class generates a tensor of identity matrix.
    """
    def __init__(self, gain=1):
        """
        This function initializes the class.

        Parameters
        ----------
        gain: float, default: 1
            a float value to multiply to identity matrix.
            returned matrix will have value "gain" for diagonals.

        Returns
        -------
        None.
        """
        # set members
        self.gain = gain

    def generate(self, shape):
        """
        This function overrides the parents' one.
        Generate random (or not) values for given shape.

        Parameters
        ----------
        shape: tuple
            a tuple of required shape information of tensor (ndarray).
            tuple should indicate a square matrix.

        Returns
        -------
        ndarray
            an array with given shape and desired values.
        """
        # check asserts
        assert len(shape) is 2 and shape[0] == shape[1], '"shape" should be square matrix for identity.'
        # generate
        rows, cols = shape
        return np.eye(rows, cols, dtype=theano.config.floatX)


class GlorotNormal(BaseInitializer):
    """
    This class generates a tensor with Glorot Normal distribution.
    See "Understanding the difficulty of training deep feedforward neural networks".
    (Xavier Glorot, Yoshua Bengio, 2010.)
    """
    def __init__(self, mean=0):
        """
        This function initializes the class.

        Parameters
        ----------
        mean: float, default: 0
            a float value for Gaussian distribution mean.
            almost always this value should be 0.

        Returns
        -------
        None.
        """
        # set members
        self.mean = mean

    def generate(self, shape):
        """
        This function overrides the parents' one.
        Generate random (or not) values for given shape.

        Parameters
        ----------
        shape: tuple
            a tuple of required shape information of tensor (ndarray).

        Returns
        -------
        ndarray
            an array with given shape and desired values.
        """
        # generate
        n_in = shape[0]
        n_out = shape[1]
        n_res = np.prod(shape[2:])  # effected area size for each weight
        std = np.sqrt(2.0 / ((n_in + n_out) * n_res))
        return np.random.normal(self.mean, std, shape).astype(theano.config.floatX)


class HeNormal(BaseInitializer):
    """
    This class generates a tensor filled with He normal distribution.
    See "Delving deep into rectifiers: surpassing human-level performance on imagenet classification".
    (Kaiming He, Xiangyu Zhang, Shaoquing Ren, Jian Sun, 2015.)
    """
    def __init__(self, mean=0):
        """
        This function initializes the class.

        Parameters
        ----------
        mean: float, default: 0
            a float value for Gaussian distribution mean.
            almost always this value should be 0.

        Returns
        -------
        None.
        """
        # set members
        self.mean = mean

    def generate(self, shape):
        """
        This function overrides the parents' one.
        Generate random (or not) values for given shape.

        Parameters
        ----------
        shape: tuple
            a tuple of required shape information of tensor (ndarray).

        Returns
        -------
        ndarray
            an array with given shape and desired values.
        """
        # generate
        n_in = shape[0]
        n_out = shape[1]
        n_res = np.prod(shape[2:])  # effected area size for each weight
        std = np.sqrt(2.0 / (n_in * n_res))
        return np.random.normal(self.mean, std, shape).astype(theano.config.floatX)


class GlorotUniform(BaseInitializer):
    """
    This class generates a tensor filled with Glorot uniform distribution.
    See "Understanding the difficulty of training deep feedforward neural networks".
    (Xavier Glorot, Yoshua Bengio, 2010.)
    """
    def __init__(self, mean=0):
        """
        This function initializes the class.

        Parameters
        ----------
        constant: float, default: 0
            a float value to fill the tensor.

        Returns
        -------
        None.
        """
        self.mean = mean

    def generate(self, shape):
        """
        This function overrides the parents' one.
        Generate random (or not) values for given shape.

        Parameters
        ----------
        shape: tuple
            a tuple of required shape information of tensor (ndarray).

        Returns
        -------
        ndarray
            an array with given shape and desired values.
        """
        # generate
        n_in = shape[0]
        n_out = shape[1]
        n_res = np.prod(shape[2:])  # effected area size for each weight
        std = np.sqrt(6.0 / ((n_in + n_out) * n_res))
        return np.random.uniform(mean - std, mean + std, shape).astype(theano.config.floatX)


class Orthogonal(BaseInitializer):
    """
    This class generates a tensor filled with orthogonal rows and columns.
    This initialization techinques is often used with LSUV pre-training.
    See "All you need is a good init".
    (Dmytro Mishkin, Jiri Matas, 2015.)
    See "https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py" for original code.
    """
    def __init__(self, scale=1):
        """
        This function initializes the class.

        Parameters
        ----------
        scale: float, default: 0
            a float value to multiply after sampling.
            usually keep as 1.

        Returns
        -------
        None.
        """
        # set members
        self.scale = scale

    def generate(self, shape):
        """
        This function overrides the parents' one.
        Generate random (or not) values for given shape.
        First generate matrix using SVD, then reshape to original shape.

        Parameters
        ----------
        shape: tuple
            a tuple of required shape information of tensor (ndarray).

        Returns
        -------
        ndarray
            an array with given shape and desired values.
        """
        # generate
        n_in = shape[0]
        n_flat = np.prod(shape[1:])  # flatten to do SVD
        a = np.random.normal(0.0, 1.0, (n_in, n_flat))
        u, _, v = np.linalg.svd(a, full_matrices=False)  # each column of u / each row of v is orthogonal
        if u.shape == (n_in, n_flat):
            q = u  # only one of u, v is same shape
        else:
            q = v  # the other one is reversed
        q = np.reshape(q, shape)  # return to original shape
        return np.asarray(self.scale * q[:shape[0], :shape[1]], dtype=theano.config.floatX)
