"""
This code includes Gumbel sofmax classes.
"""

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG
from lemontree.layers.layer import BaseLayer
from lemontree.objectives import BaseObjective
from lemontree.utils.data_utils import int_to_onehot


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
        gumbel_noise= -T.log(-T.log(gumbel_random + 1e-7) + 1e-7)
        return T.nnet.softmax((input_ + gumbel_noise) / self.temperature)  # divide by temperature


class GumbelCategoricalCrossentropy(BaseObjective):

    def __init__(self, temperature_init=1.0, stabilize=False, mode='mean'):
        """
        This function initializes the class.

        Parameters
        ----------
        temperature_init: float, default: 1
            a positive float value.
            if T > 1, become more soften, and T < 1, become more sharpen.
            if temperature is 1, same as normal softmax.
        stabilize: bool, default: False
            a bool value to use stabilization or not.
            if yes, predictions are clipped to small, nonnegative values to prevent NaNs.
            the prediction slightly ignores the probability distribution assumtion of sum = 1.
            for most cases, it is OK to use 'False'.
            however, if you are using many-class such as imagenet, this option may matter.
        mode: string {'mean', 'sum'}, default: 'mean'
            a string to choose how to compute loss as a scalar.
            'mean' computes loss as an average loss through (mini) batch.
            'sum' computes loss as a sum loss through (mini) batch.

        Returns
        -------
        None.
        """
        # check assert
        assert isinstance(stabilize, bool), '"stabilize" should be a bool value.'
        assert mode in ['mean', 'sum'], '"mode" should be either "mean" or "sum".'
        assert temperature_init > 0, '"temperature_init" should be a positive float.'
        
        # set members
        temperature = np.array(temperature_init).astype('float32')
        self.temperature = theano.shared(temperature, 'temperature')
        self.temperature.tags = ['temperature']
        self.rng = MRG(np.random.randint(1, 2147462569))

        # set members
        self.tags = ['loss', 'categorical_crossentropy']
        self.stabilize = stabilize
        self.mode = mode

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

    def get_loss(self, predict, label):
        """
        This function overrides the parents' one.
        Computes the loss by model prediction and real label.
        use theano implemented categorical_crossentropy directly.

        Parameters
        ----------
        predict: ndarray
            an array of (batch size, prediction).
            for cross entropy task, "predict" is 2D matrix.
        label: ndarray
            an array of (batch size, answer) or (batchsize,) if label is a list of class labels.
            for classification, highly recommend second one.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        gumbel_random = self.rng.uniform(label.shape, 0, 1)
        gumbel_noise= -T.log(-T.log(gumbel_random + 1e-7) + 1e-7)
        gumbel_label = T.nnet.softmax((label + gumbel_noise) / self.temperature)
        # do
        if self.mode == 'mean':
            if self.stabilize:
                return T.mean(T.nnet.categorical_crossentropy(T.clip(predict, 1e-7, 1.0 - 1e-7), gumbel_label))
            else:
                return T.mean(T.nnet.categorical_crossentropy(predict, gumbel_label))
        elif self.mode == 'sum':
            if self.stabilize:
                return T.sum(T.nnet.categorical_crossentropy(T.clip(predict, 1e-7, 1.0 - 1e-7), gumbel_label))
            else:
                return T.sum(T.nnet.categorical_crossentropy(predict, gumbel_label))
        else:
            raise ValueError('Not implemented mode entered. Mode should be in {mean, sum}.')