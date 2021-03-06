"""
This code includes objectives for deep learning loss function.
Every computations are tensor operations.
"""

import numpy as np
import theano
import theano.tensor as T
from lemontree.layers.layer import BaseLayer


class CategoricalCrossentropy(BaseLayer):

    def __init__(self, stabilize=False, mode='mean'):
        """
        This function initializes the class.

        Parameters
        ----------
        stabilize: bool, default: False
            a bool value to use stabilization or not.
            if yes, input_ions are clipped to small, nonnegative values to prevent NaNs.
            the input_ion slightly ignores the probability distribution assumtion of sum = 1.
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

        # set members
        self.stabilize = stabilize
        self.mode = mode

    def get_output(self, input_, label, mask=None):
        """
        This function overrides the parents' one.
        Computes the loss by model input_ion and real label.
        use theano implemented categorical_crossentropy directly.

        Parameters
        ----------
        input_: TensorVariable
            an array of (batch size, input_ion).
            for cross entropy task, "input_" is 2D matrix.
        label: TensorVariable
            an array of (batch size, answer) or (batchsize,) if label is a list of class labels.
            for classification, highly recommend second one.
        mask: TensorVariable
            an array of (batchsize,) only contains 0 and 1.
            loss are summed or averaged only through 1.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # do
        if mask is None:
            if self.mode == 'mean':
                if self.stabilize:
                    return T.mean(T.nnet.categorical_crossentropy(T.clip(input_, 1e-7, 1.0 - 1e-7), label))
                else:
                    return T.mean(T.nnet.categorical_crossentropy(input_, label))
            elif self.mode == 'sum':
                if self.stabilize:
                    return T.sum(T.nnet.categorical_crossentropy(T.clip(input_, 1e-7, 1.0 - 1e-7), label))
                else:
                    return T.sum(T.nnet.categorical_crossentropy(input_, label))
            else:
                raise ValueError('Not implemented mode entered. Mode should be in {mean, sum}.')
        else:
            if self.mode == 'mean':
                if self.stabilize:
                    return T.sum(T.nnet.categorical_crossentropy(T.clip(input_, 1e-7, 1.0 - 1e-7), label) * mask) / T.sum(mask)
                else:
                    return T.sum(T.nnet.categorical_crossentropy(input_, label) * mask) / T.sum(mask)
            elif self.mode == 'sum':
                if self.stabilize:
                    return T.sum(T.nnet.categorical_crossentropy(T.clip(input_, 1e-7, 1.0 - 1e-7), label) * mask)
                else:
                    return T.sum(T.nnet.categorical_crossentropy(input_, label) * mask)
            else:
                raise ValueError('Not implemented mode entered. Mode should be in {mean, sum}.')


class CategoricalAccuracy(BaseLayer):

    def __init__(self, top_k=1):
        """
        This function initializes the class.

        Parameters
        ----------
        top_k: int, default: 1
            an integer that determines what will be correct.
            for k > 1, if an answer is in top-k probable labels, assigned as correct one.

        Returns
        -------
        None.
        """
        # check assert
        assert isinstance(top_k, int) and top_k > 0, '"top_k" should be a positive integer.'

        # set members
        self.top_k = top_k

    def get_output(self, input_, label, mask=None):
        """
        This function overrides the parents' one.
        Computes the loss by model input_ion and real label.

        Parameters
        ----------
        input_: TensorVariable
            an array of (batch size, input_ion).
            for accuracy task, "input_" is 2D matrix.
        label: TensorVariable
            an array of (batch size, answer) or (batchsize,) if label is a list of class labels.
            for classification, highly recommend second one.
            should make label as integer.
        mask: TensorVariable
            an array of (batchsize,) only contains 0 and 1.
            loss are summed or averaged only through 1.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # do
        if mask is None:
            if self.top_k == 1:
                if label.ndim == 1:
                    return T.mean(T.eq(T.argmax(input_, axis=-1), label))
                elif label.ndim == 2:
                    return T.mean(T.eq(T.argmax(input_, axis=-1), T.argmax(label, axis=-1)))
                else:
                    raise ValueError()
            else:
                # TODO: not yet tested
                top_k_input_ = T.argsort(input_)[:, -self.top_k:]  # sort by values and keep top k indices
                if label.ndim == 1:
                    return T.mean(T.any(T.eq(top_k_input_, label), axis=-1))
                elif label.ndim == 2:
                    return T.mean(T.any(T.eq(top_k_input_, T.argmax(label,axis=-1)), axis=-1))
                raise ValueError()
        else:
            if self.top_k == 1:
                if label.ndim == 1:
                    return T.sum(T.eq(T.argmax(input_, axis=-1), label) * mask) / T.sum(mask)
                elif label.ndim == 2:
                    return T.sum(T.eq(T.argmax(input_, axis=-1), T.argmax(label, axis=-1)) * mask) / T.sum(mask)
                else:
                    raise ValueError()
            else:
                # TODO: not yet tested
                top_k_input_ = T.argsort(input_)[:, -self.top_k:]  # sort by values and keep top k indices
                if label.ndim == 1:
                    return T.sum(T.any(T.eq(top_k_input_, label), axis=-1) * mask) / T.sum(mask)
                elif label.ndim == 2:
                    return T.sum(T.any(T.eq(top_k_input_, T.argmax(label,axis=-1)), axis=-1) * mask) / T.sum(mask)
                raise ValueError()


class BinaryCrossentropy(BaseLayer):

    def __init__(self, stabilize=False, mode='mean'):
        """
        This function initializes the class.

        Parameters
        ----------
        stabilize: bool, default: False
            a bool value to use stabilization or not.
            if yes, input_ions are clipped to small, nonnegative values to prevent NaNs.
            the input_ion slightly ignores the probability distribution assumtion of sum = 1.
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

        # set members
        self.stabilize = stabilize
        self.mode = mode

    def get_output(self, input_, label, mask=None):
        """
        This function overrides the parents' one.
        Computes the loss by model input_ion and real label.
        use theano implemented binary_crossentropy directly.

        Parameters
        ----------
        input_: TensorVariable
            an array of (batch size, input_ion).
            for cross entropy task, "input_" is 2D matrix.
        label: TensorVariable
            an array of or (batchsize,) whose value is 0 or 1.
        mask: TensorVariable
            an array of (batchsize,) only contains 0 and 1.
            loss are summed or averaged only through 1.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # do
        if mask is None:
            if self.mode == 'mean':
                if self.stabilize:
                    return T.mean(T.nnet.binary_crossentropy(T.clip(input_, 1e-7, 1.0 - 1e-7), label))
                else:
                    return T.mean(T.nnet.binary_crossentropy(input_, label))
            elif self.mode == 'sum':
                if self.stabilize:
                    return T.sum(T.nnet.binary_crossentropy(T.clip(input_, 1e-7, 1.0 - 1e-7), label))
                else:
                    return T.sum(T.nnet.binary_crossentropy(input_, label))
            else:
                raise ValueError('Not implemented mode entered. Mode should be in {mean, sum}.')
        else:
            if self.mode == 'mean':
                if self.stabilize:
                    return T.sum(T.nnet.binary_crossentropy(T.clip(input_, 1e-7, 1.0 - 1e-7), label) * mask) / T.sum(mask)
                else:
                    return T.sum(T.nnet.binary_crossentropy(input_, label) * mask) / T.sum(mask)
            elif self.mode == 'sum':
                if self.stabilize:
                    return T.sum(T.nnet.binary_crossentropy(T.clip(input_, 1e-7, 1.0 - 1e-7), label) * mask)
                else:
                    return T.sum(T.nnet.binary_crossentropy(input_, label) * mask)
            else:
                raise ValueError('Not implemented mode entered. Mode should be in {mean, sum}.')


class BinaryAccuracy(BaseLayer):

    def get_output(self, input_, label):
        """
        This function overrides the parents' one.
        Computes the loss by model input_ion and real label.

        Parameters
        ----------
        input_: TensorVariable
            an array of (batch size, input_ion).
            for accuracy task, "input_" is 2D matrix.
        label: TensorVariable
            an array of (batch size, answer) or (batchsize,) if label is a list of class labels.
            for classification, highly recommend second one.
            should make label as integer.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # do
        # TODO: Not tested
        return T.mean(T.eq(T.gt(input_, 0.5), label)) 


class SquareLoss(BaseLayer):

    def __init__(self, mode='mean'):
        """
        This function initializes the class.

        Parameters
        ----------
        mode: string {'mean', 'sum'}, default: 'mean'
            a string to choose how to compute loss as a scalar.
            'mean' computes loss as an average loss through (mini) batch.
            'sum' computes loss as a sum loss through (mini) batch.

        Returns
        -------
        None.
        """
        # check assert
        assert mode in ['mean', 'sum'], '"mode" should be either "mean" or "sum".'

        # set members
        self.mode = mode

    def get_output(self, input_, label, mask=None):
        """
        This function overrides the parents' one.
        Computes the loss by model input_ion and real label.

        Parameters
        ----------
        input_: TensorVariable
            an array of (batch size, input_ion).
            for accuracy task, "input_" is 2D matrix.
        label: TensorVariable
            an array of (batch size, answer) or (batchsize,) if label is a list of class labels.
            for classification, highly recommend first one.
            should make label as one-hot encoding.
        mask: TensorVariable
            an array of (batchsize,) only contains 0 and 1.
            loss are summed or averaged only through 1.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # do
        if mask is None:
            if self.mode == 'mean':
                return 0.5 * T.mean(T.square(input_ - label))
            elif self.mode == 'sum':
                return 0.5 * T.sum(T.square(input_ - label))
            else:
                raise ValueError('Not implemented mode entered. Mode should be in {mean, sum}.')
        else:
            if self.mode == 'mean':
                return 0.5 * T.sum(T.sum(T.square(input_ - label), axis=-1) * mask) / T.sum(mask)
            elif self.mode == 'sum':
                return 0.5 * T.sum(T.sum(T.square(input_ - label), axis=-1) * mask)
            else:
                raise ValueError('Not implemented mode entered. Mode should be in {mean, sum}.')


class WordPerplexity(BaseLayer):

    def get_output(self, input_, label, mask):
        """
        This function overrides the parents' one.
        Computes the loss by mode input_ion and real label.
        
        Parameters
        ----------
        input_: TensorVariable
            an array of (batch size, input_ion).
            for accuracy task, "input_" is 2D matrix.
        label: TensorVariable
            an array of (batch size, answer) or (batchsize,) if label is a list of class labels.
            for word perplexity case, currently only second one is supported.
            should make label as integer.
        mask: TensorVariable
            an array of (batchsize,) only contains 0 and 1.
            loss are summed or averaged only through 1.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # do
        if mask is None:
            return T.pow(2, -T.mean(T.log2(input_[T.arange(label.shape[0]), label])))
        else:
            return T.pow(2, -T.sum(T.log2(input_[T.arange(label.shape[0]), label]) * mask) / T.sum(mask))


class KLGaussianNormal(BaseLayer):

    def __init__(self, input_shape, output_shape):
        """
        This function initializes the class.
        The shape of two tensor should be double.

        Parameters
        ----------
        input_shape: tuple
            a tuple of three values, i.e., (input channel, input width, input height).
        output_shape: tuple
            a tuple of single value, i.e., (input channel,) or (input dim,).
        """
        super(KLGaussianNormal, self).__init__()
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 1, '"input_shape" should be a tuple.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 1, '"output_shape" should be a tuple.'
        assert output_shape[0] * 2 == input_shape[0], '"output_shape" is half of "input_shape", since it contains mu and logvar.'
        # set members
        self.input_shape = input_shape
        self.output_shape = output_shape

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
        return 0.5 * T.mean(T.square(mu) + T.exp(logvar) - logvar - 1)


class JSTwoGaussian(BaseLayer):

    def __init__(self, input_shape, output_shape):
        """
        This function initializes the class.
        The shape of two tensor should be double.

        Parameters
        ----------
        input_shape: tuple
            a tuple of three values, i.e., (input channel, input width, input height).
        output_shape: tuple
            a tuple of single value, i.e., (input channel,) or (input dim,).
        """
        super(JSTwoGaussian, self).__init__()
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 1, '"input_shape" should be a tuple.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 1, '"output_shape" should be a tuple.'
        assert output_shape[0] * 2 == input_shape[0], '"output_shape" is half of "input_shape", since it contains mu and logvar.'
        # set members
        self.input_shape = input_shape
        self.output_shape = output_shape

    def get_output(self, input1_, input2_):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input.
        http://stats.stackexchange.com/questions/66271/kullback-leibler-divergence-of-two-normal-distributions

        Parameters
        ----------
        input_: TensorVariable

        Returns
        -------
        TensorVariable
        """
        mu1 = input1_[:, :self.output_shape[0]]
        logvar1 = input1_[:, self.output_shape[0]:]
        mu2 = input2_[:, :self.output_shape[0]]
        logvar2 = input2_[:, self.output_shape[0]:]
        return 0.5 * T.mean((T.square(mu1 - mu2) + T.exp(logvar1) + T.exp(logvar2)) * (1 / T.exp(logvar1) + 1 / T.exp(logvar2))) - 2

# TODO: Fix L1, L2 to work!
'''
class L1norm(BaseLayer):

    def get_output(self, params):
        """
        This function overrides the parents' one.
        Computes the loss by summing absolute parameter values.

        Parameters
        ----------
        params: list
            a list of (shared variable) parameters.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # check asserts
        assert isinstance(params, list), '"params" should be a list type.'
        # do
        loss_sum = 0
        for pp in params:
            loss_sum += T.sum(T.abs_(pp))
        return loss_sum


class L2norm(BaseLayer):

    def get_output(self, params):
        """
        This function overrides the parents' one.
        Computes the loss by summing squared parameter values.

        Parameters
        ----------
        params: list
            a list of (shared variable) parameters.

        Returns
        -------
        TensorVariable
            a symbolic tensor variable which is scalar.
        """
        # check asserts
        assert isinstance(params, list), '"params" should be a list type.'
        # do
        loss_sum = 0
        for pp in params:
            loss_sum += T.sum(T.square(pp))
        return loss_sum

'''