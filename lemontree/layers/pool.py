"""
This code includes various pooling layers.
"""

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from collections import OrderedDict
from lemontree.layers.layer import BaseLayer


class Pooling3DLayer(BaseLayer):
    """
    This class implements pooling for 3D representation.
    """
    def __init__(self, input_shape, output_shape,
                 kernel_shape=(2, 2), pool_mode='max', stride=(2, 2), padding = (0,0), name=None):
        """
        This function initializes the class.
        Input is 4D tensor, output is 4D tensor.

        Parameters
        ----------
        input_shape: tuple
            a tuple of three values, i.e., (input channel, input width, input height).
        output_shape: tuple
            a tuple of three values, i.e., (output channel, output width, output height).
            output width and height should be match to real convolution output.
        kernel_shape: tuple, default: (2, 2)
            a tuple of two values, i.e., (kernel width, kernel height).
        pool_mode: string {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}, default: 'max'
            a string to determine which mode theano pooling will use.
            'max': max pooling
            'sum': sum pooling
            'average_inc_pad': average pooling contains padding
            'average_exc_pad': average pooling does not contain padding
            'half': pad input with (kernel width //2, kernel height //2) symmetrically and do 'valid'.
                if kernel width and height is odd number, output = input
            int: pad input with (int, int) symmetrically.
            (int1, int2): pad input with (int1, int2) symmetrically.
        stride: tuple, default: (1,1)
            a tuple of two value, i.e., (stride width, stride height).
            also known as subsample.
        padding: tuple, default: (0, 0)
            a tuple of two value, i.e., (padding updown, padding leftright).
            a symmetric padding. padding first, pooling second.
        name: string
            a string name of this layer.

        Returns
        -------
        None.
        """
        super(Pooling3DLayer, self).__init__(name)
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 3, '"input_shape" should be a tuple with three values.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 3, '"output_shape" should be a tuple with three values.'
        assert isinstance(kernel_shape, tuple) and len(kernel_shape) == 2, '"kernel_shape" should be a tuple with two values.'
        assert isinstance(stride, tuple) and len(stride) == 2, '"stride" should be a tuple with two values.'
        assert isinstance(padding, tuple) and len(padding) == 2, '"padding" should be a tuple with two values.'
        assert pool_mode in ['max', 'sum', 'average_inc_pad', 'average_exc_pad'], '"poolmode should be a string mode. see theano.tensor.signal.pool.pool_2d for details.'

        # set members
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_shape = kernel_shape
        self.pool_mode = pool_mode
        self.stride = stride
        self.padding = padding

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
        return pool_2d(input_,
                       ds=self.kernel_shape,
                       ignore_border=True,  # if you don't want to ignore border, use padding
                       st=self.stride,
                       padding=self.padding,
                       mode=self.pool_mode)


class GlobalAveragePooling3DLayer(BaseLayer):
    """
    This class implements global average pooling for 3D representation.
    Global average pooling averages energy through each map (channel).
    """
    def __init__(self, input_shape, output_shape, padding=(0,0), name=None):
        """
        This function initializes the class.
        Input is 4D tensor, output is 2D tensor.

        Parameters
        ----------
        input_shape: tuple
            a tuple of three values, i.e., (input channel, input width, input height).
        output_shape: tuple
            a tuple of single value, i.e., (output channel,) or (output dim,).
        padding: tuple, default: (0, 0)
            a tuple of two value, i.e., (padding updown, padding leftright).
            a symmetric padding. padding first, pooling second.
        name: string
            a string name of this layer.

        Returns
        -------
        None.
        """
        super(GlobalAveragePooling3DLayer, self).__init__(name)
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 3, '"input_shape" should be a tuple with three values.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 1, '"output_shape" should be a tuple with single value.'
        assert isinstance(padding, tuple) and len(padding) == 2, '"padding" should be a tuple with two values.'
        assert input_shape[0] == output_shape[0], 'Global average pooling result is 2D tensor of (batch size, output channel).'

        # set members
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.padding = padding

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
        result = pool_2d(input_,
                         ds=self.input_shape[1:],
                         ignore_border=True,
                         st=self.input_shape[1:],
                         padding=self.padding,
                         mode='average_exc_pad')  # result is 4D tensor yet, (batch size, output channel, 1, 1)
        return T.reshape(result, (input_.shape[0], input_.shape[1]))  # flatten to 2D matrix

