import numpy as np
import theano
import theano.tensor as T
from lemontree.layers.lstm import LSTMRecurrentLayer
from lemontree.layers.activation import Linear

lstm = LSTMRecurrentLayer(input_shape=(3,),
                          output_shape=(4,),
                          gate_activation=Linear(),
                          cell_activation=Linear(),
                          out_activation=Linear(),
                          forget_bias_one=False,
                          peephole=False,
                          gradient_steps=-1,
                          output_return_index=None,
                          precompute=False,
                          name='lstm_test'
                          )

W = np.random.uniform(-1, 1, size=(3,16)).astype(theano.config.floatX)  # (input dim, output dim)
U = np.random.uniform(-1, 1, size=(4,16)).astype(theano.config.floatX)  # (output dim, output dim)
lstm.W.set_value(W)
lstm.U.set_value(U)

input_seq = np.random.uniform(-1, 1, size=(5,6,3))  # (n_batch, n_timestep, input dim)
#mask_seq = np.random.binomial(1, 0.5, size=(5,6))  # (n_batch, n_timestep)
mask_seq = np.ones((5,6), dtype='int32')
hidden_init = np.random.uniform(-1, 1, size=(5,4))  # (n_batch, output_dim)
cell_init = np.random.uniform(-1, 1, size=(5,4))  # (n_batch, output_dim)

sym_input = T.ftensor3('input_')
sym_mask = T.imatrix('mask_')
sym_hidden = T.fmatrix('hidden_init')
sym_cell = T.fmatrix('cell_init')

func = theano.function(inputs=[sym_input, sym_mask, sym_hidden, sym_cell],
                       outputs=lstm.get_output(sym_input, sym_mask, sym_cell, sym_hidden),
                       allow_input_downcast=True)

cell_seq, hidden_seq = func(input_seq, mask_seq, cell_init, hidden_init)
print(cell_seq.shape)
print(cell_seq)
print(hidden_seq.shape)
print(hidden_seq)

