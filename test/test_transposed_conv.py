import numpy as np
import theano
import theano.tensor as T

from lemontree.initializers import GlorotNormal
from lemontree.layers.convolution import TransposedConvolution3DLayer

# test convolution

x = T.ftensor4('X')
trans = TransposedConvolution3DLayer((1,6,6), (1,8,8), (5,5), 'half', (2,2))
trans.set_name('test_trans')
trans.set_shared()
trans.set_batch_size(1)
GlorotNormal().initialize_params([trans.W])
y = trans.get_output(x)
func_valid = theano.function([x],y, allow_input_downcast=True)

x_np = np.random.uniform(0, 1, size=(1,1,16,16))
y_np = func_valid(x_np)
W_np = trans.W.get_value()

print(x_np.shape)
print(x_np)
print(W_np.shape)
print(W_np)
print(y_np.shape)
print(y_np)
