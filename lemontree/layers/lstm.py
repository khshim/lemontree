"""
This code includes LSTM, long short-term memory layer.
LSTM is a kind of RNN, which preserves temporal information for a very long term.
"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from lemontree.layers.layer import BaseLayer
from lemontree.layers.recurrent import BaseRecurrentLayer
from lemontree.layers.activation import Sigmoid, Tanh


class LSTMRecurrentLayer(BaseRecurrentLayer):
    """
    This class implements LSTM recurrent layer.
    """
    def __init__(self, input_shape, output_shape,
                 gate_activation = Sigmoid(),
                 cell_activation = Tanh(),
                 out_activation = Tanh(),
                 forget_bias_one=False,
                 peephole=False,
                 gradient_steps=-1,
                 output_return_index=[-1],
                 save_state_index=-1,
                 also_return_cell=False,
                 precompute=False, unroll=False, backward=False):
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
        gate_activation: activation, default: Sigmoid()
            a base activation layer which will be used as gate activation.
            use sigmoid or hard_sigmoid.
        cell_activation: activation, default: Tanh()
            a base activation layer which will be used to create cell candidate.
        out_activation: activation, default: Tanh()
            a base activation layer which will be used as output activation.
        forget_bias_one: bool, default: False
            a bool value determine using bias 1 as forget gate initialization.
        peephole: bool, default: False
            a bool value determine using peephole connection or not.
            if yes, peephole will be added to gates and candidate.
        gradient_steps: int, default: -1
            an integer which indicates unroll of rnn for backward path.
            if -1, pass gradient through whole sequence length.
            if a positive integer, pass gradient back only for that much.
        output_return_index: list, default: [-1]
            a list of integers which output step should be returned.
            if [-1], only final output is returned.
            if none, return all steps through sequence.
        save_state_index: int, default:-1
            an integers which output step should be saved.
            if -1, final state is saved.
        also_return_cell: bool, default: False
            a bool value determine cell return.
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
        """
        super(LSTMRecurrentLayer, self).__init__(gradient_steps, output_return_index, save_state_index,
                                                 precompute, unroll, backward)
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 1, '"input_shape" should be a tuple with single value.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 1, '"output_shape" should be a tuple with single value.'
        assert isinstance(gate_activation, BaseLayer), '"gate_activation" should be an activation layer itself.'
        assert isinstance(cell_activation, BaseLayer), '"cell_activation" should be an activation layer itself.'
        assert isinstance(out_activation, BaseLayer), '"out_activation" should be an activation layer itself.'
        assert isinstance(also_return_cell, bool), '"also_return_cell" should be a bool value whether returning cell or not.'
        assert isinstance(forget_bias_one, bool), '"forget_bias_one" should be a bool value whether using 1 as forget bias or not.'
        assert isinstance(peephole, bool), '"peephole" should be a bool value whether using peephole connection or not.'

        # set members
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.gate_activation = gate_activation
        self.cell_activation = cell_activation
        self.out_activation = out_activation
        self.forget_bias_one = forget_bias_one
        self.also_return_cell = also_return_cell
        self.peephole = peephole
        self.updates = OrderedDict()
        
    def set_shared(self):
        """
        This function overrides the parents' one.
        Set shared variables.

        Shared Variables
        ----------------
        W: 2D matrix
            shape is (input dim, output dim * 4).
        U: 2D matrix
            shape is (output dim, output dim * 4).
        b: 1D vector
            shape is (output dim * 4,).
        V: 2D matrix
            shape is (output dim, output dim * 3).

        cell_init: 2D matrix
            shape is (batch_size, output dim)
        hidden_init: 2D matrix
            shape is (batch_size, output dim)

        For fast computation, using merged (concatenated) matrix.
        W and U are computed by a single dot product, while V is not.
        """
        # order: forget, input, output, cell_candidate
        W_np = np.zeros((self.input_shape[0], self.output_shape[0] * 4)).astype(theano.config.floatX)  # input[t] to ~~
        self.W = theano.shared(W_np, self.name + '_weight_W')
        self.W.tags = ['weight', self.name]
        # order: forget, input, output, cell_candidate
        U_np = np.zeros((self.output_shape[0], self.output_shape[0] * 4)).astype(theano.config.floatX)  # output[t-1] to ~~
        self.U = theano.shared(U_np, self.name + '_weight_U')
        self.U.tags = ['weight', self.name]
        # order: forget, input, output, cell_candidate
        b_np = np.zeros((self.output_shape[0] * 4,)).astype(theano.config.floatX)
        if self.forget_bias_one:
            b_np[:self.output_shape[0]] = 1.0  # forget gate bias intialize to 1
        self.b = theano.shared(b_np, self.name + '_bias')
        self.b.tags = ['bias', self.name]

        if self.peephole:
            # order: forget, input, output
            V_np = np.ones((self.output_shape[0] * 3,)).astype(theano.config.floatX)  # cell[t-1] to ~~
            self.V = theano.shared(V_np, self.name + '_peephole_V')
            self.V.tags = ['peephole', self.name]

        init_np = np.zeros((self.batch_size, self.output_shape[0])).astype(theano.config.floatX)
        self.cell_init = theano.shared(init_np, self.name + '_cell_init')
        self.cell_init.tags = ['cell_init', self.name]
        self.hidden_init = theano.shared(init_np, self.name + '_hidden_init')
        self.hidden_init.tags = ['hidden_init', self.name]

    def set_shared_by(self, params):
        if self.peephole:
            self.W = params[0]
            self.U = params[1]
            self.b = params[2]
            self.V = params[3]
        else:
            self.W = params[0]
            self.U = params[1]
            self.b = params[2]

    def get_output(self, input_, mask_, reset_, cell_init_=None, hidden_init_=None):
        """
        This function overrides the parents' one.
        Creates symbolic function to compute output from an input and output (hidden).
        
        Math Expression
        -------------------
        f[t] = gate_activation(dot(X[t], Wf) + dot(Y[t-1], Uf) + bf)
        i[t] = gate_activation(dot(X[t], Wi) + dot(Y[t-1], Ui) + bi)
        o[t] = gate_activation(dot(X[t], Wo) + dot(Y[t-1], Uo) + bo)
        candidate = cell_activation(dot(X[t], Wc) + dot(Y[t-1], Uc) + bc)
        C[t] = C[t-1] * f[t] + candidate * i[t]
        Y[t] = out_activation(C[t]) * o[t]

        if precompute True, compute dot(X[t],W) for all steps first.
        if mask exist and 1, Y[t] = Y[t-1] and C[t] = C[t-1]

        if peephole

        f[t] = gate_activation(dot(X[t], Wf) + dot(Y[t-1], Uf) + dot(C[t-1], Vf) + bf)
        i[t] = gate_activation(dot(X[t], Wi) + dot(Y[t-1], Ui) + dot(C[t-1], Vi) + bi)
        o[t] = gate_activation(dot(X[t], Wo) + dot(Y[t-1], Uo) + dot(C[t], Vo) + bo)

        Parameters
        ----------
        input_: TensorVariable (batch_size, sequence_length, input_dim)
        mask_: TensorVariable  (batch_size, sequence_length)
        reset_: TensorVariable (batch_size)

        Returns
        -------
        TensorVariable
        """
        # input_ are (n_batch, n_timesteps, n_features)
        # change to (n_timesteps, n_batch, n_features)
        input_ = input_.dimshuffle(1, 0, *range(2, input_.ndim))
        # mask_ are (n_batch, n_timesteps)
        mask_ = mask_.dimshuffle(1, 0, 'x')
        sequence_length = input_.shape[0]

        cell_init = self.cell_init * reset_.dimshuffle(0, 'x')
        if cell_init_ is not None:
            cell_init = T.switch(reset_.dimshuffle(0, 'x'), cell_init, cell_init_)
        hidden_init = self.hidden_init * reset_.dimshuffle(0, 'x')
        if hidden_init_ is not None:
            hidden_init = T.switch(reset_.dimshuffle(0, 'x'), hidden_init, hidden_init_)

        # precompute input
        if self.precompute:
            additional_dims = tuple(input_.shape[k] for k in range(2, input_.ndim))  # (output_dim,)
            input_ = T.reshape(input_, (sequence_length*self.batch_size,) + additional_dims)
            input_ = T.dot(input_, self.W)
            additional_dims = tuple(input_.shape[k] for k in range(1, input_.ndim))  # (output_dim,)
            input_ = T.reshape(input_, (sequence_length, self.batch_size,) + additional_dims)

        # step function
        def step(input_, cell, hidden):
            if self.precompute:
                no_activation = input_ + T.dot(hidden, self.U) + self.b
            else:
                no_activation = T.dot(input_, self.W) + T.dot(hidden, self.U) + self.b

            if not self.peephole:
                forget_gate = self.gate_activation.get_output(no_activation[:, 0 * self.output_shape[0]:1 * self.output_shape[0]])
                input_gate = self.gate_activation.get_output(no_activation[:, 1 * self.output_shape[0]:2 * self.output_shape[0]])
                cell_candidate = self.cell_activation.get_output(no_activation[:, 3 * self.output_shape[0]:4 * self.output_shape[0]])
                output_gate = self.gate_activation.get_output(no_activation[:, 2 * self.output_shape[0]:3 * self.output_shape[0]])
                
            else:
                forget_gate = self.gate_activation.get_output(no_activation[:, 0 * self.output_shape[0]:1 * self.output_shape[0]]
                                            + cell * self.V[0 * self.output_shape[0]:1 * self.output_shape[0]])
                input_gate = self.gate_activation.get_output(no_activation[:, 1 * self.output_shape[0]:2 * self.output_shape[0]]
                                           + cell * self.V[1 * self.output_shape[0]:2 * self.output_shape[0]])
                cell_candidate = self.cell_activation.get_output(no_activation[:, 3 * self.output_shape[0]:4 * self.output_shape[0]])
                output_gate = self.gate_activation.get_output(no_activation[:, 2 * self.output_shape[0]:3 * self.output_shape[0]]
                                            + cell_candidate * self.V[2 * self.output_shape[0]:3 * self.output_shape[0]])

            cell_computed = forget_gate * cell + input_gate * cell_candidate
            hidden_computed = output_gate * self.out_activation.get_output(cell_computed)

            return cell_computed, hidden_computed

        # step function, with mask
        def step_masked(input_, mask_, cell, hidden):
            cell_computed, hidden_computed = step(input_, cell, hidden)
            return T.switch(mask_, cell_computed, cell), T.switch(mask_, hidden_computed, hidden)

        # main operation
        if self.unroll:
            counter = range(self.gradient_steps)
            if self.backward:
                counter = counter[::-1]  # reversed index
            iter_output_cell = []
            iter_output_hidden = []
            outputs_info = [cell_init, hidden_init]
            for index in counter:
                step_input = [input_[index], mask_[index]] + outputs_info
                step_output = step_masked(*step_input)
                iter_output_cell.append(step_output[0])
                iter_output_hidden.append(step_output[1])
                outputs_info = [iter_output_cell[-1], iter_output_hidden[-1]]
            cell_output = T.stack(iter_output_cell, axis=0)
            hidden_output = T.stack(iter_output_hidden, axis=0)

        else:
            cell_output, hidden_output = theano.scan(fn=step_masked,
                                                     sequences=[input_, mask_],
                                                     outputs_info=[cell_init, hidden_init],
                                                     go_backwards=self.backward,
                                                     n_steps = None,
                                                     truncate_gradient=self.gradient_steps)[0]  # only need outputs, not updates
        
        # computed output are (n_timesteps, n_batch, n_features)
        # select only required
        self.updates[self.cell_init] = cell_output[self.save_state_index]
        self.updates[self.hidden_init] = hidden_output[self.save_state_index]

        if self.output_return_index is None:
            cell_output_return = cell_output
            hidden_output_return = hidden_output
        else:
            cell_output_return = cell_output[self.output_return_index]
            hidden_output_return = hidden_output[self.output_return_index]
        # change to (n_batch, n_timesteps, n_features)
        cell_output_return = cell_output_return.dimshuffle(1, 0, *range(2, cell_output_return.ndim))
        hidden_output_return = hidden_output_return.dimshuffle(1, 0, *range(2, hidden_output_return.ndim))

        if self.backward:
            cell_output_return = cell_output_return[:, ::-1]
            hidden_output_return = hidden_output_return[:, ::-1]

        if self.also_return_cell:
            return cell_output_return, hidden_output_return
        else:
            return hidden_output_return

    def get_params(self):
        """
        This function overrides the parents' one.
        Returns interal layer parameters.

        Returns
        -------
        list
            a list of shared variables used.
        """
        if self.peephole:
            return [self.W, self.U, self.b, self.V]
        else:
            return [self.W, self.U, self.b]

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
