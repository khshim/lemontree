"""
This code incudes variational latent variable generator.
"""

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG
from collections import OrderedDict
from lemontree.layers.layer import BaseLayer


class Latent1DLayer(BaseLayer):
    """
    This class implements vae latent variable generation from mu and (log) variance.
    """
    def __init__(self, input_shape, output_shape, seed=25623):
        """
        This function initializes the class.
        The shape of two tensor should be double.

        Parameters
        ----------
        input_shape: tuple
            a tuple of single value, i.e., (input dim,).
        output_shape: tuple
            a tuple of single value, i.e., (input channel,) or (input dim,).
        seed: int
            an integer for random seed.
        """
        super(Latent1DLayer, self).__init__()
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 1, '"input_shape" should be a tuple.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 1, '"output_shape" should be a tuple.'
        assert output_shape[0] * 2 == input_shape[0], '"output_shape" is half of "input_shape", since it contains mu and logvar.'
        # set members
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.rng = MRG(seed)

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        mu = input_[:, :self.output_shape[0]]
        logvar = input_[:, self.output_shape[0]:]
        return mu + T.sqrt(T.exp(logvar)) * self.rng.normal((self.batch_size, self.output_shape[0]), 0, 1)


class GaussianNoise1DLayer(BaseLayer):
    """
    This class implements adding Gaussian noise to input.
    """
    def __init__(self, input_shape,  sigma=1.0, seed=256567):
        """
        This function initializes the class.
        The shape of two tensor should be double.

        Parameters
        ----------
        input_shape: tuple
            a tuple of single value, i.e., (input dim,).
        seed: int
            an integer for random seed.
        sigma: float, default: 1.0
            a float value that will be sigma (std) of Gaussian noise.
        """
        super(GaussianNoise1DLayer, self).__init__()
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 1, '"input_shape" should be a tuple.'
        # set members
        self.input_shape = input_shape
        self.rng = MRG(seed)
        self.sigma = sigma

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return input_ + self.rng.normal((self.batch_size, self.input_shape[0]), 0, self.sigma)


class GaussianNoise3DLayer(BaseLayer):
    """
    This class implements adding Gaussian noise to input.
    """
    def __init__(self, input_shape,  sigma=1.0, seed=256567):
        """
        This function initializes the class.
        The shape of two tensor should be double.

        Parameters
        ----------
        input_shape: tuple
            a tuple of three value, i.e., (channel, width, height).
        seed: int
            an integer for random seed.
        sigma: float, default: 1.0
            a float value that will be sigma (std) of Gaussian noise.
        """
        super(GaussianNoise3DLayer, self).__init__()
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 3, '"input_shape" should be a tuple of three values.'
        # set members
        self.input_shape = input_shape
        self.rng = MRG(seed)
        self.sigma = sigma

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return input_ + self.rng.normal((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), 0, self.sigma)

