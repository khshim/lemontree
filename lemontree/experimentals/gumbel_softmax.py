"""
This code includes Gumbel sofmax classes.
"""

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG
from lemontree.layers.layer import BaseLayer


class GumbelSoftmax(BaseLayer):
    """
    This class implements gumbel softmax activation.
    See "Categorical Reparameterization with Gumbel-softmax".
    (Eric Jang, Shixiang Gu, Ben Poole, 2016.)
    """
    def __init__(self, temperature_init=1.0, name=None):
        """
        This function initializes the class.

        Parameters
        ----------
        temperature_init: float, default: 1
            a positive float value.
            if T > 1, become more soften, and T < 1, become more sharpen.
            if temperature is 1, same as normal softmax.
        name: string
            a string name of this class.

        Returns
        -------
        None.
        """
        super(GumbelSoftmax, self).__init__(name)
        # check asserts
        assert temperature_init > 0, '"temperature_init" should be a positive float.'
        
        # set members
        temperature = np.array(temperature_init).astype('float32')
        self.temperature = theano.shared(temperature, 'temperature')
        self.temperature.tags = ['temperature']
        self.rng = MRG(np.random.randint(1, 2147462569))

    def change_temperature(self, new_temperature):
        """
        This function changes the temperature for softmax.

        Parameters
        ----------
        new_temperature: float
            a positive float value which will be a new temperature.

        Returns
        -------
        None.
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
        gumbel_random = self.rng.uniform(input_.shape, 0, 1)
        gumbel_noise= -T.log(-T.log(gumbel_random + 1e-8) + 1e-8)
        return T.nnet.softmax((input_ + gumbel_noise) / self.temperature)  # divide by temperature
