"""
This code includes simple recurrent layer, which is developed by Elman.
Also known as (and should be) Elman Recurrent Layer.
"""

import numpy as np
import theano
import theano.tensor as T
from lemontree.layers.layer import BaseLayer
from lemontree.layers.recurrent import BaseRecurrentLayer
from lemontree.layers.activation import Tanh


class ElmanRecurrentLayer(BaseRecurrentLayer):
    """
    This class implements simple recurrent layer.
    """
    def __init__(self, input_shape, output_shape,
                 out_activation = Tanh(),
                 gradient_steps=-1,
                 output_return_index=[-1],
                 precompute=False, unroll=False, backward=False, name=None):
        """
        This function initializes the class.
        Input is 3D tensor, output is 3D tensor.
        Do not use activation layer after this layer, since activation is already applied to output.

        Parameters
        ----------
        input_shape: tuple
            a tuple of single value, i.e., (input dim,)
        output_shape: tuple
            a tuple of single value, i.e., (output dim,)
        out_activation: activation, default: Tanh()
            a base activation layer which will be used as output activation.
        gradient_steps: int, default: -1
            an integer which indicates unroll of rnn for backward path.
            if -1, pass gradient through whole sequence length.
            if a positive integer, pass gradient back only for that much.
        output_return_index: list, default: [-1]
            a list of integers which output step should be saved.
            if [-1], only final output is returned.
            if none, return all steps through sequence.
        precompute: bool, default: False
            a bool value determine input precomputation.    
            for speedup, we can precompute input in return of increased memory usage.
        unroll: bool, default: False
            a bool value determine recurrent loop unrolling.
            for speedup, we can unroll and compile theano function,
            in return of increased memory usage and much increased compile time.
        backward: bool, default: False
            a bool value determine the direction of sequence.
            although using backward True, output will be original order.
        name: string
            a name of the class. 

        Returns
        -------
        None.
        """
        super(ElmanRecurrentLayer, self).__init__(gradient_steps, output_return_index,
                                                  precompute, unroll, backward, name)
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 1, '"input_shape" should be a tuple with single value.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 1, '"output_shape" should be a tuple with single value.'
        assert isinstance(out_activation, BaseLayer), '"out_activation" should be an activation layer itself.'

        # set members
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.out_activation = out_activation

        # create shared variables
        """
        Shared Variables
        ----------------
        W: 2D matrix
            shape is (input dim, output dim).
        W: 2D matrix
            shape is (output dim, output dim).
        b: 1D vector
            shape is (output dim,).
        """
        W = np.zeros((input_shape[0], output_shape[0])).astype(theano.config.floatX)  # input[t] to output[t]
        self.W = theano.shared(W, self.name + '_weight_W')
        self.W.tags = ['weight', self.name]
        U = np.zeros((output_shape[0], output_shape[0])).astype(theano.config.floatX)  # output[t-1] to output[t]
        self.U = theano.shared(U, self.name + '_weight_U')
        self.U.tags = ['weight', self.name]
        b = np.zeros(output_shape).astype(theano.config.floatX)
        self.b = theano.shared(b, self.name + '_bias')
        self.b.tags = ['bias', self.name]

    def get_output(self, input_, mask_, hidden_init):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input and output (hidden).
        
        Math Expression
        -------------------
        Y[t] = out_activation(dot(X[t], W) + dot(Y[t-1], U) + b)

        if precompute True, compute dot(X[t],W) for all steps first.
        if mask exist and 1, Y[t] = Y[t-1]

        Parameters
        ----------
        input_: TensorVariable
        mask_: TensorVariable
        hidden_init: TensorVariable

        Returns
        -------
        TensorVariable
        """
        # input_ are (n_batch, n_timesteps, n_features)
        # change to (n_timesteps, n_batch, n_features)
        input_ = input_.dimshuffle(1, 0, 2)
        # mask_ are (n_batch, n_timesteps)
        masks = masks.dimshuffle(1, 0, 'x')
        sequence_length = input_.shape[0]
        batch_num = input_.shape[1]

        # precompute input
        if self.precompute:
            additional_dims = tuple(input.shape[k] for k in range(2, input.ndim))  # (output_dim,)
            input = T.reshape(input, (sequence_length*batch_num,) + additional_dims)
            input = T.dot(input, self.W)
            additional_dims = tuple(input.shape[k] for k in range(1, input.ndim))  # (output_dim,)
            input = T.reshape(input, (sequence_length, batch_num,) + additional_dims)

        # step function
        def step(input_, hidden):
            if self.precompute:
                return self.out_activation.get_output(input_ + T.dot(hidden, self.U) + self.b)
            else:
                return self.out_activation.get_output(T.dot(input_, self.W) + T.dot(hidden, self.U) + self.b)

        # step function, with mask
        def step_masked(input_, mask_, hidden):
            hidden_computed = step(input_, hidden)
            return T.switch(mask_, hidden_computed, hidden)

        # main operation
        if self.unroll:
            counter = range(self.gradient_steps)
            if self.backward:
                counter = counter[::-1]  # reversed index
            iter_output = []
            outputs_info = [hidden_init]
            for index in counter:
                step_input = [input_[index], mask_[index]] + outputs_info
                step_output = step_masked(*step_input)
                iter_output.append(step_output)
                outputs_info = [iter_output[-1]]
            hidden_output = T.stack(iter_output, axis=0)

        else:
            hidden_output = theano.scan(fn=step_masked,
                                        sequences=[input_, mask_],
                                        outputs_info=[hidden_init],
                                        go_backwards=self.backward,
                                        n_steps = None,
                                        truncate_gradient=self.gradient_steps)[0]  # only need outputs, not updates

        # computed output are (n_timesteps, n_batch, n_features)
        # select only required
        if self.output_return_index is None:
            hidden_output_return = hidden_output
        else:
            hidden_output_return = hidden_output[self.output_return_index]
        # change to (n_batch, n_timesteps, n_features)
        hidden_output_return = hidden_output_return.dimshuffle(1, 0, *range(2, hidden_output_return.ndim))

        # backward order straight
        if self.backward:
            hidden_output_return = hidden_output_return[:, ::-1]

        return hidden_output_return

    def get_params(self):
        """
        This function overrides the parents' one.
        Returns interal layer parameters.

        Parameters
        ----------
        None.

        Returns
        -------
        list
            a list of shared variables used.
        """
        return [self.W, self.U, self.b]
