"""
This code includes batch normalization layer.
Use theano included (cudnn) batch normalization.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from lemontree.layers.layer import BaseLayer


class BatchNormalization1DLayer(BaseLayer):
    """
    This class implements batch normalization for 1D representation.
    """
    def __init__(self, input_shape, momentum=0.99, mean_only=False):
        """
        This function initializes the class.
        Input is 2D tenor, output is 2D tensor.
        For long term experiment, use momentum = 0.99, else, 0.9.
        If we use mean_only = True, not using Theano / cuDNN batch normalization.

        Parameters
        ----------
        input_shape: tuple
            a tuple of single value, i.e., (input dim,)
            since input shape is same to output shape, there is no output shape argument.
        momentum: float, default: 0.99
            a float value which will used to average inference mean and variance.
            using exponential moving average.
        mean_only: bool, default: False
            a bool value whether use substract mean only, not divide with std.
        """
        super(BatchNormalization1DLayer, self).__init__()
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 1, '"input_shape" should be a tuple with single value.'
        assert momentum > 0 and momentum < 1, '"momentum" should be a float value in range (0, 1).'
        assert isinstance(mean_only, bool), '"mean_only" should be a bool value.'

        # set members
        self.input_shape = input_shape
        self.momentum = momentum
        self.mean_only = mean_only
        self.updates = OrderedDict()

    def set_shared(self):
        """
        This function overrides the parents' one.
        Set shared variables.

        Shared Variables
        ----------------
        flag: scalar
            a scalar value to distinguish training mode and inference mode.
        bn_mean: 1D vector
            shape is (input dim,).
        bn_std: 1D vector
            shape is (input dim,).
        gamma: 1D vector
            shape is (input dim,).
        beta: 1D vector
            shape is (input dim,).
        """
        self.flag = theano.shared(1, self.name + '_flag')  # 1: train / -1: inference
        self.flag.tags = ['flag', self.name]
        bn_mean = np.zeros((self.input_shape[0],)).astype(theano.config.floatX)  # initialize with zero
        self.bn_mean = theano.shared(bn_mean, self.name + '_bn_mean')
        self.bn_mean.tags = ['bn_mean', self.name]
        bn_std = np.ones((self.input_shape[0],)).astype(theano.config.floatX)  # initialize with one
        self.bn_std = theano.shared(bn_std, self.name + '_bn_std')
        self.bn_std.tags = ['bn_std', self.name]
        gamma = np.ones((self.input_shape[0],)).astype(theano.config.floatX)  # initialize with one
        self.gamma = theano.shared(gamma, self.name + '_gamma')
        self.gamma.tags = ['gamma', self.name]
        beta = np.zeros((self.input_shape[0],)).astype(theano.config.floatX)  # initialize with zero
        self.beta = theano.shared(beta, self.name + '_beta')
        self.beta.tags = ['beta', self.name]

    def set_shared_by(self, params):
        self.gamma = params[0]
        self.beta = params[1]
        self.bn_mean = params[2]
        self.bn_std = params[3]

    def change_flag(self, new_flag):
        """
        This function change flag to change training and inference mode.
        If flag > 0, training mode, else, inference mode.

        Parameters
        ---------
        new_flag: int (or float)
            a single scalar value to be a new flag.
        """
        self.flag.set_value(float(new_flag))  # 1: train, -1: inference

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.
        The symbolic function use theano switch function conditioned by flag.

        Math Expression
        ---------------
        For inference:
            y = (x - mean(x)) / std(x)
            mean and std through mini-batch.
        For training:
            y = (x - batch_mean) / batch_std
            mean and std for inference.
        batch_mean = momentum * batch_mean + (1 - momentum) * mean(x)
        batch_std = momentum * batch_std + (1 - momentum) * std(x)

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        Tensorvariable         
        """
        # mean and std for current mini-batch
        if self.mean_only:
            batch_mean = T.mean(input_, axis=0)
            self.updates[self.bn_mean] = self.bn_mean * self.momentum + batch_mean * (1 - self.momentum)
        else:
            batch_mean = T.mean(input_, axis=0)
            batch_std = T.std(input_, axis=0)
            self.updates[self.bn_mean] = self.bn_mean * self.momentum + batch_mean * (1 - self.momentum)
            self.updates[self.bn_std] = self.bn_std * self.momentum + batch_std * (1 - self.momentum)

        # conditional compute
        if self.mean_only:
            return T.switch(T.gt(self.flag, 0),
                            (input_ - batch_mean) * self.gamma + self.beta,
                            (input_ - self.bn_mean) * self.gamma + self.beta)
        else:
            return T.switch(T.gt(self.flag, 0),
                            T.nnet.batch_normalization(input_, self.gamma, self.beta, batch_mean, batch_std),
                            T.nnet.batch_normalization(input_, self.gamma, self.beta, self.bn_mean, self.bn_std))

    def get_params(self):
        """
        This function overrides the parents' one.
        Returns interal layer parameters.

        Returns
        -------
        list
            a list of shared variables used.
        """
        return [self.gamma, self.beta, self.bn_mean, self.bn_std]

    def get_updates(self):
        """
        This function overrides the parents' one.
        Returns internal updates.

        Returns
        -------
        OrderedDict
            a dictionary of internal updates.
        """
        return self.updates


class BatchNormalization3DLayer(BaseLayer):
    """
    This class implements batch normalization for 3D representation.
    """
    def __init__(self, input_shape, momentum=0.99, mean_only=False):
        """
        This function initializes the class.
        Input is 4D tenor, output is 4D tensor.
        For long term experiment, use momentum = 0.99, else, 0.9.
        If we use mean_only = True, not using Theano / cuDNN batch normalization.

        Parameters
        ----------
        input_shape: tuple
            a tuple of three values, i.e., (input channel, input width, input height)
            since input shape is same to output shape, there is no output shape argument.
        momentum: float, default: 0.99
            a float value which will used to average inference mean and variance.
            using exponential moving average.
        mean_only: bool, default: False
            a bool value whether use substract mean only, not divide with std.
        """
        super(BatchNormalization3DLayer, self).__init__()
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 3, '"input_shape" should be a tuple with three value.'
        assert momentum > 0 and momentum < 1, '"momentum" should be a float value in range (0, 1).'
        assert isinstance(mean_only, bool), '"mean_only" should be a bool value.'

        # set members
        self.input_shape = input_shape
        self.momentum = momentum
        self.mean_only = mean_only
        self.updates = OrderedDict()

    def set_shared(self):
        """
        This function overrides the parents' one.
        Set shared variables.

        Shared Variables
        ----------------
        flag: scalar
            a scalar value to distinguish training mode and inference mode.
        bn_mean: 1D vector
            shape is (input channel,).
        bn_std: 1D vector
            shape is (input channle,).
        gamma: 1D vector
            shape is (input channle,).
        beta: 1D vector
            shape is (input channel,).
        """
        self.flag = theano.shared(1, self.name + '_flag')  # 1: train / -1: inference
        self.flag.tags = ['flag', self.name]
        bn_mean = np.zeros((self.input_shape[0],)).astype(theano.config.floatX)
        self.bn_mean = theano.shared(bn_mean, self.name + '_bn_mean')
        self.bn_mean.tags = ['bn_mean', self.name]
        bn_std = np.ones((self.input_shape[0],)).astype(theano.config.floatX)
        self.bn_std = theano.shared(bn_std, self.name + '_bn_std')
        self.bn_std.tags = ['bn_std', self.name]
        gamma = np.ones((self.input_shape[0],)).astype(theano.config.floatX)
        self.gamma = theano.shared(gamma, self.name + '_gamma')
        self.gamma.tags = ['gamma', self.name]
        beta = np.zeros((self.input_shape[0],)).astype(theano.config.floatX)
        self.beta = theano.shared(beta, self.name + '_beta')
        self.beta.tags = ['beta', self.name]

    def set_shared_by(self, params):
        self.gamma = params[0]
        self.beta = params[1]
        self.bn_mean = params[2]
        self.bn_std = params[3]

    def change_flag(self, new_flag):
        """
        This function change flag to change training and inference mode.
        If flag > 0, training mode, else, inference mode.

        Parameters
        ---------
        new_flag: int (or float)
            a single scalar value to be a new flag.
        """
        self.flag.set_value(float(new_flag))  # 1: train, -1: inference

    def get_output(self, input_):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.
        The symbolic function use theano switch function conditioned by flag.

        Math Expression
        ---------------
        For inference:
            y = (x - mean(x)) / std(x)
            mean and std through mini-batch.
        For training:
            y = (x - batch_mean) / batch_std
            mean and std for inference.
        batch_mean = momentum * batch_mean + (1 - momentum) * mean(x)
        batch_std = momentum * batch_std + (1 - momentum) * std(x)

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        Tensorvariable         
        """
        # mean and std for current mini-batch
        if self.mean_only:
            batch_mean = T.mean(input_, axis=[0, 2, 3])  # mean through batch, width, height
            self.updates[self.bn_mean] = self.bn_mean * self.momentum + batch_mean * (1 - self.momentum)
        else:
            batch_mean = T.mean(input_, axis=[0, 2, 3])  # mean through batch, width, height
            batch_std = T.std(input_, axis=[0, 2, 3])  # std through batch, width, height
            self.updates[self.bn_mean] = self.bn_mean * self.momentum + batch_mean * (1 - self.momentum)
            self.updates[self.bn_std] = self.bn_std * self.momentum + batch_std * (1 - self.momentum)

        # conditional compute
        if self.mean_only:
            return T.switch(T.gt(self.flag, 0),
                            (input_ - batch_mean.dimshuffle('x',0,'x','x')) * self.gamma.dimshuffle('x',0,'x','x') + self.beta.dimshuffle('x',0,'x','x'),
                            (input_ - self.bn_mean.dimshuffle('x',0,'x','x')) * self.gamma.dimshuffle('x',0,'x','x') + self.beta.dimshuffle('x',0,'x','x'))
        else:
            return T.switch(T.gt(self.flag, 0),
                            T.nnet.batch_normalization(input_, self.gamma.dimshuffle('x', 0, 'x', 'x'), self.beta.dimshuffle('x', 0, 'x', 'x'),
                            batch_mean.dimshuffle('x', 0, 'x', 'x'), batch_std.dimshuffle('x', 0, 'x', 'x'), 'high_mem'),
                            T.nnet.batch_normalization(input_, self.gamma.dimshuffle('x', 0, 'x', 'x'), self.beta.dimshuffle('x', 0, 'x', 'x'),
                            self.bn_mean.dimshuffle('x', 0, 'x', 'x'), self.bn_std.dimshuffle('x', 0, 'x', 'x'), 'high_mem'))

    def get_params(self):
        """
        This function overrides the parents' one.
        Returns interal layer parameters.

        Returns
        -------
        list
            a list of shared variables used.
        """
        return [self.gamma, self.beta, self.bn_mean, self.bn_std]

    def get_updates(self):
        """
        This function overrides the parents' one.
        Returns internal updates.

        Returns
        -------
        OrderedDict
            a dictionary of internal updates.
        """
        return self.updates


class LayerNormalization1DLayer(BaseLayer):

    def __init__(self, input_shape, momentum=0.99):
        super(LayerNormalization1DLayer, self).__init__()
        self.input_shape = input_shape
        self.momentum = momentum

    def set_shared(self):
        gamma = np.ones((self.input_shape[0],)).astype(theano.config.floatX)
        self.gamma = theano.shared(gamma, self.name + '_gamma')
        self.gamma.tags = ['gamma', self.name]
        beta = np.zeros((self.input_shape[0],)).astype(theano.config.floatX)
        self.beta = theano.shared(beta, self.name + '_beta')
        self.beta.tags = ['beta', self.name]

    def set_shared_by(self, params):
        self.gamma = params[0]
        self.beta = params[1]

    def get_output(self, input_):
        dim_mean = T.mean(input_, axis=1)
        dim_std = T.std(input_, axis=1)
        return self.gamma * (input_ - dim_mean.dimshuffle(0, 'x')) / (dim_std.dimshuffle(0, 'x') + 1e-7) + self.beta

    def get_params(self):
        return [self.gamma, self.beta]


class LayerNormalization3DLayer(BaseLayer):

    def __init__(self, input_shape, momentum=0.99):
        super(LayerNormalization3DLayer, self).__init__()
        self.input_shape = input_shape  # (number of maps, width of map, height of map)
        assert len(self.input_shape) == 3
        self.momentum = momentum

    def set_shared(self):       
        gamma = np.ones((self.input_shape[0],)).astype(theano.config.floatX)
        self.gamma = theano.shared(gamma, self.name + '_gamma')
        self.gamma.tags = ['gamma', self.name]
        beta = np.zeros((self.input_shape[0],)).astype(theano.config.floatX)
        self.beta = theano.shared(beta, self.name + '_beta')
        self.beta.tags = ['beta', self.name]

    def set_shared_by(self, params):
        self.gamma = params[0]
        self.beta = params[1]

    def get_output(self, input_):
        dim_mean = T.mean(input_, axis=[1, 2, 3])
        dim_std = T.std(input_, axis=[1, 2, 3])
        return self.gamma.dimshuffle('x', 0, 'x', 'x') * (input_ - dim_mean.dimshuffle(0, 'x', 'x', 'x')) / (dim_std.dimshuffle(0, 'x', 'x', 'x') + 1e-7) + self.beta.dimshuffle('x', 0, 'x', 'x')

    def get_params(self):
        return [self.gamma, self.beta]

