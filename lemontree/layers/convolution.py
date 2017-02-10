"""
This code includes various convolution layers.
Use theano implemented convolutions.
Recent convolutions, such as dilated, will be supported soon.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from lemontree.layers.layer import BaseLayer


class Convolution3DLayer(BaseLayer):
    """
    This class implements convolution for 3D representation.
    """
    def __init__(self, input_shape, output_shape, kernel_shape,
                 border_mode='valid', stride=(1, 1), use_bias=True):
        """
        This function initializes the class.
        Input is 4D tensor, output is 4D tensor.
        For efficient following batch normalization, use_bias = False.
        
        Parameters
        ----------
        input_shape: tuple
            a tuple of three values, i.e., (input channel, input width, input height).
        output_shape: tuple
            a tuple of three values, i.e., (output channel, output width, output height).
            output width and height should be match to real convolution output.
        kernel_shape: tuple
            a tuple of two values, i.e., (kernel width, kernel height).
        border_mode: string {'valid', 'full', 'half', int, (int1, int2)}, default: 'valid'
            a string to determine which mode theano convolution will use.
            'valid': output = input - kernel + 1
            'full': output = input + kernel - 1
            'half': pad input with (kernel width //2, kernel height //2) symmetrically and do 'valid'.
                if kernel width and height is odd number, output = input
            int: pad input with (int, int) symmetrically.
            (int1, int2): pad input with (int1, int2) symmetrically.
        stride: tuple, default: (1,1)
            a tuple of two value, i.e., (stride width, stride height).
            also known as subsample.
        use_bias: bool, default: True
            a bool value whether we use bias or not.
        """
        super(Convolution3DLayer, self).__init__()
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 3, '"input_shape" should be a tuple with three values.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 3, '"output_shape" should be a tuple with three values.'
        assert isinstance(kernel_shape, tuple) and len(kernel_shape) == 2, '"kernel_shape" should be a tuple with two values.'
        assert border_mode in ['valid', 'full', 'half', 'int'] or isinstance(border_mode, int) or isinstance(border_mode, tuple), '"border_mode should be a string mode. see theano.tensor.nnet.conv2d for details.'
        assert isinstance(stride, tuple) and len(stride) == 2, '"stride" should be a tuple with two values.'
        assert isinstance(use_bias, bool), '"use_bias" should be a bool value.'
        # TODO: assert given output shape is same as computed shape.

        # set members
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_shape = kernel_shape
        self.border_mode = border_mode
        self.stride = stride
        self.use_bias = use_bias

    def set_shared(self):
        """
        This function overrides the parents' one.
        Set shared variables.

        Shared Variables
        ----------------
        W: 4D matrix
            shape is (output channel, input channel, kernel width, kernel height).
        b: 1D vector
            shape is (output channel).
        """
        W = np.zeros((self.output_shape[0], self.input_shape[0], self.kernel_shape[0], self.kernel_shape[1])).astype(theano.config.floatX)
        self.W = theano.shared(W, self.name + '_weight')
        self.W.tags = ['weight', self.name]
        b = np.zeros((self.output_shape[0])).astype(theano.config.floatX)
        self.b = theano.shared(b, self.name + '_bias')
        self.b.tags = ['bias', self.name]

    def set_shared_by(self, params):
        if self.use_bias:
            self.W = params[0]
            self.b = params[1]
        else:
            self.W = params[0]

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
        result = T.nnet.conv2d(input_,
                               filters=self.W,
                               input_shape=(None,) + self.input_shape,  # unknown dimension for batch size at compile time
                               filter_shape=(self.output_shape[0], self.input_shape[0], self.kernel_shape[0], self.kernel_shape[1]),
                               border_mode=self.border_mode,
                               subsample=self.stride)
        if self.use_bias:
            return result + self.b.dimshuffle('x', 0, 'x', 'x')  # dimshuffle to channel dimension
        else:
            return result

    def get_params(self):
        """
        This function overrides the parents' one.
        Returns interal layer parameters.

        Returns
        -------
        list
            a list of shared variables used.
        """
        if self.use_bias:
            return [self.W, self.b]
        else:
            return [self.W]


class Padding3DLayer(BaseLayer):
    """
    This class implements padding for 3D representation.
    If you use this class for padding before convolution, better use 'half' or 'int' or (int1, int2) for border_mode in Convolution3D.
    Convolution3D layer only supports symmetric padding (which is major), but this class also supports non symmetric padding.
    """
    def __init__(self, input_shape, output_shape, padding=(1, 1, 1, 1)):
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
        padding: tuple
            a tuple of four values, i.e., (padding up, padding down, padding left, padding right).
        """
        super(Padding3DLayer, self).__init__()
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 3, '"input_shape" should be a tuple with three values.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 3, '"output_shape" should be a tuple with three values.'
        assert isinstance(padding, tuple) and len(padding) == 4, '"padding" should be a tuple with four values.'
        assert output_shape[1] == input_shape[1] + padding[0] + padding[1]
        assert output_shape[2] == input_shape[2] + padding[2] + padding[3]

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
        shape = (input_.shape[0],) + (self.input_shape[0], self.input_shape[1] + self.padding[0] + self.padding[1], self.input_shape[2] + self.padding[2] + self.padding[3])
        result = T.zeros(shape, dtype=theano.config.floatX)  # make zero output
        indices = (slice(None),
                   slice(None),
                   slice(self.padding[0], self.input_shape[1] + self.padding[0]),
                   slice(self.padding[2], self.input_shape[2] + self.padding[2])
                   )
        return T.set_subtensor(result[indices], input)

