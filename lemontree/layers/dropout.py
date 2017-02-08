"""
This code includes dropout layer.
Some classifies dropout in noise layer.
"""

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG
from lemontree.layers.layer import BaseLayer


class DropoutLayer(BaseLayer):
    """
    This class implements dropout for layer output energy (activations).
    """
    def __init__(self, drop_probability=0.5, rescale=False, seed=4455):
        """
        This function initializes the class.
        Input and output tensor shape is equal.

        Parameters
        ----------
        drop_probability: float, default: 0.5
            a float value ratio of how many activations will be zero.
        rescale: bool, default: True
            a bool value whether we rescale the output or not.
            multiply ratio and preserve the variance.
        seed: int
            an integer for random seed.
        """
        super(DropoutLayer, self).__init__()
        # check asserts
        assert drop_probability >= 0 and drop_probability < 1, '"drop_probability" should be in range [0, 1).'
        assert isinstance(rescale, bool), '"rescale" should be a bool value whether we use dropout rescaling or not.'

        # set members        
        self.drop_probability = drop_probability
        self.rescale = rescale
        self.rng = MRG(seed)  # random number generator

    def set_shared(self):
        """
        This function overrides the parents' one.
        Set shared Variables.

        Shared Variables
        ----------------
        flag: scalar
            a scalar value to distinguish training mode and inference mode.
        """        
        self.flag = theano.shared(1, self.name + '_flag')  # 1: train / -1: inference
        self.flag.tags = ['flag', self.name]

    def change_flag(self, new_flag):
        """
        This function change flag to change training and inference mode.
        If flag > 0, training mode, else, inference mode.

        Parameters
        ---------
        new_flag: int (or float)
            a single scalar value to be a new flag.
        """
        self.flag.set_value(float(new_flag)) # 1: train / -1: inference

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.
        The symbolic function use theano switch function conditioned by flag.

        Math Expression
        ---------------
        For inference:
            y = x
        For training:
            mask ~ U[0, 1] and sampled to binomial.
            y = 1 / ( 1 - drop_probability) * x * mask

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        Tensorvariable         
        """
        if self.rescale is True:
            coeff = 1 / (1 - self.drop_probability)
        else:
            coeff = 1
        mask = self.rng.binomial(input_.shape, p=1 - self.drop_probability, dtype=input_.dtype)
        return T.switch(T.gt(self.flag, 0), input_ * mask * coeff, input_)
