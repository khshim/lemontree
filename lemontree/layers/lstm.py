"""
This code includes LSTM, long short-term memory layer.
LSTM is a kind of RNN, which preserves temporal information for a very long term.
"""

import numpy as np
import theano
import theano.tensor as T
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
        super(LSTMRecurrentLayer, self).__init__(gradient_steps, output_return_index,
                                                 precompute, unroll, backward, name)
        # check asserts
        assert isinstance(input_shape, tuple) and len(input_shape) == 1, '"input_shape" should be a tuple with single value.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 1, '"output_shape" should be a tuple with single value.'
        assert isinstance(gate_activation, BaseLayer), '"gate_activation" should be an activation layer itself.'
        assert isinstance(cell_activation, BaseLayer), '"cell_activation" should be an activation layer itself.'
        assert isinstance(out_activation, BaseLayer), '"out_activation" should be an activation layer itself.'
        assert isinstance(forget_bias_one, bool), '"forget_bias_one" should be a bool value whether using 1 as forget bias or not.'
        assert isinstance(peephole, bool), '"peephole" should be a bool value whether using peephole connection or not.'

        # set members
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.gate_activation = gate_activation
        self.cell_activation = cell_activation
        self.out_activation = out_activation
        self.forget_bias_one = forget_bias_one
        self.peephole = peephole
        
        # create shared variables
        """
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

        For fast computation, using merged (concatenated) matrix.
        W and U are computed by a single dot product, while V is not.
        """
        # order: forget, input, output, cell_candidate
        W_np = np.zeros((input_shape[0], output_shape[0] * 4)).astype(theano.config.floatX)  # input[t] to ~~
        self.W = theano.shared(W_np, self.name + '_weight_W')
        self.W.tags = ['weight', self.name]
        # order: forget, input, output, cell_candidate
        U_np = np.zeros((output_shape[0], output_shape[0] * 4)).astype(theano.config.floatX)  # output[t-1] to ~~
        self.U = theano.shared(U_np, self.name + '_weight_U')
        self.U.tags = ['weight', self.name]
        # order: forget, input, output, cell_candidate
        b_np = np.zeros((output_shape[0] * 4,)).astype(theano.config.floatX)
        if self.forget_bias_one:
            b_np[:output_shape[0]] = 1.0  # forget gate bias intialize to 1
        self.b = theano.shared(b_np, self.name + '_bias')
        self.b.tags = ['bias', self.name]

        if self.peephole:
            # order: forget, input, output
            V_np = np.zeros((output_shape[0], output_shape[0] * 3)).astype(theano.config.floatX)  # cell[t-1] to ~~
            self.V = theano.shared(V_np, self.name + '_weight_V')
            self.V.tags = ['weight', self.name]

    def get_output(self, input_, mask_, cell_init, hidden_init):
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
        input_: TensorVariable
        mask_: TensorVariable
        cell_init: TensorVariable
        hidden_init: TensorVariable

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
        batch_num = input_.shape[1]

        # precompute input
        if self.precompute:
            additional_dims = tuple(input_.shape[k] for k in range(2, input_.ndim))  # (output_dim,)
            input_ = T.reshape(input_, (sequence_length*batch_num,) + additional_dims)
            input_ = T.dot(input_, self.W)
            additional_dims = tuple(input_.shape[k] for k in range(1, input_.ndim))  # (output_dim,)
            input_ = T.reshape(input_, (sequence_length, batch_num,) + additional_dims)

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
                                            + T.dot(cell, self.V[:, 0 * self.output_shape[0]:1 * self.output_shape[0]]))
                input_gate = self.gate_activation.get_output(no_activation[:, 1 * self.output_shape[0]:2 * self.output_shape[0]]
                                           + T.dot(cell, self.V[:, 1 * self.output_shape[0]:2 * self.output_shape[0]]))
                cell_candidate = self.cell_activation.get_output(no_activation[:, 3 * self.output_shape[0]:4 * self.output_shape[0]])
                output_gate = self.gate_activation.get_output(no_activation[:, 2 * self.output_shape[0]:3 * self.output_shape[0]]
                                            + T.dot(cell_candidate, self.V[:, 2 * self.output_shape[0]:3 * self.output_shape[0]]))

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

        return cell_output_return, hidden_output_return

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
        if self.peephole:
            return [self.W, self.U, self.b, self.V]
        else:
            return [self.W, self.U, self.b]


class DeepLSTMRecurrentLayer(BaseRecurrentLayer):
    """
    This class implements Deep LSTM recurrent layer.
    """
    def __init__(self, depth, input_shape, output_shape,
                 gate_activation = Sigmoid(),
                 cell_activation = Tanh(),
                 out_activation = Tanh(),
                 forget_bias_one=False,
                 peephole=False,
                 gradient_steps=-1,
                 output_return_index=[-1],
                 precompute=False, unroll=False, backward=False, name=None):
        """
        This function initializes the class.
        Input is 3D tensor, output is 3D tensor.
        Do not use activation layer after this layer, since activation is already applied to output.
        Currrently, we don't support precompute at this time.
        Also, we don't support differnt hidden dimension for each deep stack.
        i.e., all layer have same dimension.

        Parameters
        ----------
        depth: int
            an integer that indicates layer depth.
        input_shape: tuple
            a tuple of single value, i.e., (input dim,)
        output_shape: tupe
            a tuple of single value, i.e., (hidden dim,)
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
            a list of integers which output step should be saved.
            if [-1], only final output is returned.
            if none, return all steps through sequence.
        precompute: bool, default: False
            a bool value determine input precomputation.    
            although used True, we force precompute to False. (currently not available)
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
        super(DeepLSTMRecurrentLayer, self).__init__(gradient_steps, output_return_index,
                                                     False, unroll, backward, name)
        # check asserts
        assert isinstance(depth, int), '"depth" should be an integer of layer depth.'
        assert isinstance(input_shape, tuple) and len(input_shape) == 1, '"input_shape" should be a tuple with single value.'
        assert isinstance(output_shape, tuple) and len(output_shape) == 1, '"output_shape" should be a tuple with single value.'
        assert isinstance(gate_activation, BaseLayer), '"gate_activation" should be an activation layer itself.'
        assert isinstance(cell_activation, BaseLayer), '"cell_activation" should be an activation layer itself.'
        assert isinstance(out_activation, BaseLayer), '"out_activation" should be an activation layer itself.'
        assert isinstance(forget_bias_one, bool), '"forget_bias_one" should be a bool value whether using 1 as forget bias or not.'
        assert isinstance(peephole, bool), '"peephole" should be a bool value whether using peephole connection or not.'

        # set members
        self.depth = depth
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.gate_activation = gate_activation
        self.cell_activation = cell_activation
        self.out_activation = out_activation
        self.forget_bias_one = forget_bias_one
        self.peephole = peephole
        
        # create shared variables
        """
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

        For fast computation, using merged (concatenated) matrix.
        W and U are computed by a single dot product, while V is not.
        """
        self.W = []
        self.U = []
        self.b = []
        if self.peephole:
            self.V = []

        # order: forget, input, output, cell_candidate
        W_np = np.zeros((input_shape[0], output_shape[0] * 4)).astype(theano.config.floatX)  # input[t] to ~~
        # order: forget, input, output, cell_candidate
        U_np = np.zeros((output_shape[0], output_shape[0] * 4)).astype(theano.config.floatX)  # output[t-1] to ~~
        # order: forget, input, output, cell_candidate
        b_np = np.zeros((output_shape[0] * 4,)).astype(theano.config.floatX)
        if self.forget_bias_one:
            b_np[:output_shape[0]] = 1.0  # forget gate bias intialize to 1
        if self.peephole:
            # order: forget, input, output
            V_np = np.zeros((output_shape[0], output_shape[0] * 3)).astype(theano.config.floatX)  # cell[t-1] to ~~

        for dp in range(self.depth):   
            W = theano.shared(W_np, self.name + '_' + str(dp) + '_weight_W')
            W.tags = ['weight', self.name + '_' + str(dp)]
            self.W.append(W)
        
            U = theano.shared(U_np, self.name + '_' + str(dp) + '_weight_U')
            U.tags = ['weight', self.name + '_' + str(dp)]
            self.U.append(U)

            b = theano.shared(b_np, self.name + '_' + str(dp) + '_bias')
            b.tags = ['bias', self.name + '_' + str(dp)]
            self.b.append(b)

            if self.peephole:
                V = theano.shared(V_np, self.name + '_' + str(dp) + '_weight_V')
                V.tags = ['weight', self.name + '_' + str(dp)]
                self.V.append(V)

        assert len(self.W) == self.depth
        assert len(self.U) == self.depth
        assert len(self.b) == self.depth
        if self.peephole:
            assert len(self.V) == self.depth

    def get_output(self, input_, mask_, cells_init, hiddens_init):
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
        input_: TensorVariable
        mask_: TensorVariable
        cells_init: TensorVariable
        hiddens_init: TensorVariable

        Returns
        -------
        TensorVariable
        """
        # input_ are (n_batch, n_timesteps, n_features)
        # change to (n_timesteps, n_batch, n_features)
        input_ = input_.dimshuffle(1, 0, *range(2, input_.ndim))
        # mask_ are (n_batch, n_timesteps)
        # change to (n_timesteps, n_batch)
        mask_ = mask_.dimshuffle(1, 0, 'x')
        # cells_init are (n_batch, n_depth, n_features)
        # hiddens_init are (n_batch, n_depth, n_features)

        sequence_length = input_.shape[0]
        batch_num = input_.shape[1]

        # step function
        def step(input_, cells, hiddens):
            cells_computed = T.zeros_like(cells)
            hiddens_computed = T.zeros_like(hiddens)

            current_input = input_
            for dp in range(self.depth):

                no_activation = T.dot(current_input, self.W[dp]) + T.dot(hiddens[:,dp], self.U[dp]) + self.b[dp]

                if not self.peephole:
                    forget_gate = self.gate_activation.get_output(no_activation[:, 0 * self.output_shape[0]:1 * self.output_shape[0]])
                    input_gate = self.gate_activation.get_output(no_activation[:, 1 * self.output_shape[0]:2 * self.output_shape[0]])
                    cell_candidate = self.cell_activation.get_output(no_activation[:, 3 * self.output_shape[0]:4 * self.output_shape[0]])
                    output_gate = self.gate_activation.get_output(no_activation[:, 2 * self.output_shape[0]:3 * self.output_shape[0]])
                
                else:
                    forget_gate = self.gate_activation.get_output(no_activation[:, 0 * self.output_shape[0]:1 * self.output_shape[0]]
                                                + T.dot(cells[dp], self.V[dp][:, 0 * self.output_shape[0]:1 * self.output_shape[0]]))
                    input_gate = self.gate_activation.get_output(no_activation[:, 1 * self.output_shape[0]:2 * self.output_shape[0]]
                                               + T.dot(cells[dp], self.V[dp][:, 1 * self.output_shape[0]:2 * self.output_shape[0]]))
                    cell_candidate = self.cell_activation.get_output(no_activation[:, 3 * self.output_shape[0]:4 * self.output_shape[0]])
                    output_gate = self.gate_activation.get_output(no_activation[:, 2 * self.output_shape[0]:3 * self.output_shape[0]]
                                                + T.dot(cell_candidate, self.V[dp][:, 2 * self.output_shape[0]:3 * self.output_shape[0]]))

                cell_computed = forget_gate * cells[:,dp] + input_gate * cell_candidate
                hidden_computed = output_gate * self.out_activation.get_output(cell_computed)

                cells_computed = T.set_subtensor(cells_computed[dp], cell_computed)
                hiddens_computed = T.set_subtensor(hiddens_computed[dp], hidden_computed)
                current_input = hidden_computed

            return cells_computed, hiddens_computed

        # step function, with mask
        def step_masked(input_, mask_, cells, hiddens):
            cells_computed, hiddens_computed = step(input_, cells, hiddens)
            return T.switch(mask_, cells_computed, cells), T.switch(mask_, hiddens_computed, hiddens)

        # main operation
        if self.unroll:
            counter = range(self.gradient_steps)
            if self.backward:
                counter = counter[::-1]  # reversed index
            iter_output_cells = []
            iter_output_hiddens = []
            outputs_info = [cells_init, hiddens_init]
            for index in counter:
                step_input = [input_[index], mask_[index]] + outputs_info
                step_output = step_masked(*step_input)
                iter_output_cells.append(step_output[0])
                iter_output_hiddens.append(step_output[1])
                outputs_info = [iter_output_cells[-1], iter_output_hiddens[-1]]
            cells_output = T.stack(iter_output_cells, axis=0)
            hiddens_output = T.stack(iter_output_hiddens, axis=0)

        else:
            cells_output, hiddens_output = theano.scan(fn=step_masked,
                                                       sequences=[input_, mask_],
                                                       outputs_info=[cells_init, hiddens_init],
                                                       go_backwards=self.backward,
                                                       n_steps = None,
                                                       truncate_gradient=self.gradient_steps)[0]  # only need outputs, not updates
        
        # computed output are (n_timesteps, n_batch, n_depth, n_features)
        # select only required
        if self.output_return_index is None:
            cells_output_return = cells_output
            hiddens_output_return = hiddens_output
        else:
            cells_output_return = cells_output[self.output_return_index]
            hiddens_output_return = hiddens_output[self.output_return_index]
        # change to (n_batch, n_timesteps, n_depth, n_features)
        cells_output_return = cells_output_return.dimshuffle(1, 0, *range(2, cells_output_return.ndim))
        hiddens_output_return = hiddens_output_return.dimshuffle(1, 0, *range(2, hiddens_output_return.ndim))

        if self.backward:
            cells_output_return = cells_output_return[:, ::-1]
            hiddens_output_return = hiddens_output_return[:, ::-1]

        return cells_output_return, hiddens_output_return

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
        if self.peephole:
            return self.W + self.U + self.b + self.V
        else:
            return self.W + self.U + self.b
