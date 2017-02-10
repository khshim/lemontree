"""
This code includes various non-linear activation layer classes.
"""

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG
from collections import OrderedDict
from lemontree.layers.layer import BaseLayer


class ReLU(BaseLayer):
    """
    This class implements ReLU activation function.
    """
    def __init__(self, alpha=0):
        """
        This function initializes the class.

        Parameters
        ----------
        alpha: float, default: 0
            a positive float value which indicates the tangent of x < 0 range.
            if alpha is not 0, this function become a leaky ReLU.
        """
        super(ReLU, self).__init__()
        # check asserts
        assert alpha >= 0, '"alpha" should be a non-negative float value.'

        # set members
        self.alpha = alpha        

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.
        ReLU is element-wise operation.

        Math Expression
        -------------------
        y = maximum(x, 0)
        y = ifelse(x > 0, x, \alpha * x)

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return T.nnet.relu(input_, self.alpha)


class Linear(BaseLayer):
    """
    This class implements Linear activation function.
    Not non-linear, just return itself.
    """
    def __init__(self):
        """
        This function initializes the class.
        """
        super(Linear, self).__init__()

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Return input itself.

        Math Expression
        -------------------
        y = maximum(x, 0)
        y = ifelse(x > 0, x, \alpha * x)

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return input_



class Tanh(BaseLayer):
    """
    This class implements tanh activation function.
    """
    def __init__(self):
        """
        This function initializes the class.
        """
        super(Tanh, self).__init__()

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Tanh is element-wise operation.

        Math Expression
        -------------------
        y = tanh(x)

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return T.tanh(input_)



class ELU(BaseLayer):
    """
    This class implements ELU activation function.
    """
    def __init__(self, alpha=1):
        """
        This function initializes the class.

        Parameters
        ----------
        alpha: float, default: 1
            a positive float value which indicates the tangent of x < 0 range.
            if alpha is not 0, this function become a leaky ReLU.
        """
        # check asserts
        super(ELU, self).__init__()
        assert alpha >= 0, '"alpha" should be a non-negative float value.'
        self.alpha = alpha        

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        ELU is element-wise operation.
        If alpha = 0, same as ReLU.

        Math Expression
        -------------------
        y = ifelse(x > 0, x, \alpha * (exp(x) - 1))

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return T.switch(T.gt(input_, 0), input_, self.alpha * (T.exp(input_) - 1))



class Sigmoid(BaseLayer):
    """
    This class implements Sigmoid activation function.
    """
    def __init__(self):
        """
        This function initializes the class.
        """
        super(Sigmoid, self).__init__()

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Sigmoid is element-wise operation.

        Math Expression
        -------------------
        y = 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return T.nnet.sigmoid(input_)


class HardSigmoid(BaseLayer):
    """
    This class implements Hard sigmoid activation function.
    """
    def __init__(self):
        """
        This function initializes the class.
        """
        super(HardSigmoid, self).__init__()

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Sigmoid is element-wise operation.

        Math Expression
        -------------------
        y = 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return T.nnet.hard_sigmoid(input_)


class Softmax(BaseLayer):
    """
    This class implements softmax activation function.
    """
    def __init__(self):
        """
        This function initializes the class.
        """
        super(Softmax, self).__init__()

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Softmax converts output energy to probability distributuion.

        Math Expression
        -------------------
        y_k = exp(x_k) / \sum(exp(x_i))

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return T.nnet.softmax(input_)


class DistilationSoftmax(BaseLayer):
    """
    This class implements distillation softmax activation function.
    Softmax distillation is widely used for sharpen / soften distribution.
    Also used for knowledge transfer.
    """
    def __init__(self, temperature_init=1.0):
        """
        This function initializes the class.

        Parameters
        ----------
        temperature_init: float, default: 1
            a positive float value.
            if T > 1, become more soften, and T < 1, become more sharpen.
            if temperature is 1, same as normal softmax.
        """
        super(DistilationSoftmax, self).__init__(name)
        # check asserts
        assert temperature_init > 0, '"temperature_init" should be a positive float.'
        self.temperature_init = temperature_init

    def set_shared(self):
        """
        This function overrides the parents' one.
        Set shared variables.

        Shared Variables
        ----------------
        temperature: scalar
        """
        # set members
        temperature = np.array(self.temperature_init).astype('float32')
        self.temperature = theano.shared(temperature, self.name + '_temperature')
        self.temperature.tags = ['temperature', self.name]

    def change_temperature(self, new_temperature):
        """
        This function changes the temperature for softmax.

        Parameters
        ----------
        new_temperature: float
            a positive float value which will be a new temperature.
        """
        # check asserts
        assert new_temperature > 0, '"new_temperature" should be a positive float.'
        self.temperature.set_value(float(new_temperature))

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Softmax converts output energy to probability distributuion.

        Math Expression
        -------------------
        y_k = exp(x_k / T) / \sum(exp(x_i / T))

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        return T.nnet.softmax(input_ / self.temperature)  # divide by temperature


class GumbelSoftmax(BaseLayer):
    """
    This class implements gumbel softmax activation.
    See "Categorical Reparameterization with Gumbel-softmax".
    (Eric Jang, Shixiang Gu, Ben Poole, 2016.)
    """
    def __init__(self, input_shape, temperature_init=0.1, seed=9683):
        """
        This function initializes the class.

        Parameters
        ----------
        shape: tuple
            a tuple of shape of gumbel random distribution.
            this is required for scan to not affected.
        temperature_init: float, default: 1
            a positive float value.
            if T > 1, become more soften, and T < 1, become more sharpen.
            if temperature is 1, same as normal softmax.
        """
        super(GumbelSoftmax, self).__init__()
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 1, '"input_shape" should be a tuple of shape.'
        assert temperature_init > 0, '"temperature_init" should be a positive float.'
        
        # set members
        self.input_shape = input_shape
        self.temperature_init = temperature_init
        self.rng = MRG(seed)

    def set_shared(self):
        """
        This function overrides the parents' one.
        Set shared variables.

        Shared Variables
        ----------------
        temperature: scalar
        """
        temperature = np.array(self.temperature_init).astype('float32')
        self.temperature = theano.shared(temperature, self.name + '_temperature')
        self.temperature.tags = ['temperature', self.name]        

    def change_temperature(self, new_temperature):
        """
        This function changes the temperature for softmax.

        Parameters
        ----------
        new_temperature: float
            a positive float value which will be a new temperature.
        """
        # check asserts
        assert new_temperature > 0, '"new_temperature" should be a positive float.'
        self.temperature.set_value(float(new_temperature))

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Softmax converts output energy to probability distributuion.

        Math Expression
        -------------------
        g_k ~ Gumbel(0, 1)
        y_k = exp((x_k + g_k) / T) / \sum(exp((x_i + g_k) / T))

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        # generate random gumbel distribution
        uniform_random = self.rng.uniform((self.batch_size, self.input_shape[0]), 0, 1)
        gumbel_random = -T.log(-T.log(uniform_random + 1e-7) + 1e-7)
        return T.nnet.softmax((input_ + gumbel_random) / self.temperature)  # divide by temperature