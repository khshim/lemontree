"""
This code includes various non-linear activation layer classes.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from lemontree.layers.layer import BaseLayer


class ReLU(BaseLayer):
    """
    This class implements ReLU activation function.
    """
    def __init__(self, alpha=0, name=None):
        """
        This function initializes the class.

        Parameters
        ----------
        alpha: float
            a positive float value which indicates the tangent of x < 0 range.
        name: string
            a string name of this class.

        Returns
        -------
        None.
        """
        # check asserts
        assert alpha > 0, '"alpha" should be a positive float value.'

        # set members
        self.alpha = alpha
        super(ReLU, self).__init__(name)

    def get_output(self, input_,):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.

        Parameters
        ----------
        input_: TensorVariable
            a TensorVariable 
        """
        return T.nnet.relu(input_, self.alpha)

    def get_params(self):
        return []

    def get_updates(self):
        return OrderedDict()


class Linear(BaseLayer):

    def __init__(self, name=None):
        super(ReLU, self).__init__(name)

    def get_output(self, input):
        return input

    def get_params(self):
        return []

    def get_updates(self):
        return OrderedDict()


class Tanh(BaseLayer):

    def __init__(self, name = None):
        super(BaseLayer, self).__init__(name)

    def get_output(self, input):
        return T.tanh(input)

    def get_params(self):
        return []

    def get_updates(self):
        return OrderedDict()


class ELU(BaseLayer):

    def __init__(self, alpha=1.0, name=None):
        self.alpha = alpha
        super(ELU, self).__init__(name)

    def get_output(self, input):
        return T.nnet.elu(input, self.alpha)

    def get_params(self):
        return []

    def get_updates(self):
        return OrderedDict()


class Sigmoid(BaseLayer):

    def __init__(self, name=None):
        super(Sigmoid, self).__init__(name)

    def get_output(self, input):
        return T.nnet.sigmoid(input)

    def get_params(self):
        return []

    def get_updates(self):
        return OrderedDict()


class Softmax(BaseLayer):

    def __init__(self, name=None):
        super(Softmax, self).__init__(name)

    def get_output(self, input):
        return T.nnet.softmax(input)

    def get_params(self):
        return []

    def get_updates(self):
        return OrderedDict()


class DistilledSoftmax(BaseLayer):

    def __init__(self, temperature_init=1.0, name=None):
        temperature = np.array(temperature_init).astype('float32')
        self.temperature = theano.shared(temperature, 'temperature')
        self.temperature.tags = [' temperature']
        super(DistilledSoftmax, self).__init__(name)

    def change_temperature(self, new_temperature):
        self.temperature.set_value(float(new_temperature))

    def get_output(self, input):
        return T.nnet.softmax(input / self.temperature)

    def get_params(self):
        return []

    def get_updates(self):
        return OrderedDict()
