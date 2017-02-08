import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams

# Test RandomStreams

rng = RandomStreams(123)
sample = rng.normal((3,3), 0, 1, dtype=theano.config.floatX)
func = theano.function([], sample, allow_input_downcast=True)

print('...RandomStreams')
for i in range(3):
    print(i, func())

# Test RNGMRG

mrg = MRG_RandomStreams(123)
sample2 = mrg.normal((3,3), 0, 1, dtype=theano.config.floatX)
func2 = theano.function([], sample2, allow_input_downcast=True)

print('...MRG_RandomStreams')
for i in range(3):
    print(i, func2())

# Test RandomStreams in scan op

def step():
    sample = rng.normal((3,3), 0, 1, dtype=theano.config.floatX)
    return sample

out, update = theano.scan(step, None, None, n_steps=5)
func3 = theano.function([], out, allow_input_downcast=True)

print('...RandomStreams in Scan')
for i in range(3):
    print(i, func3())

# Test RNGMRG in scan op

def step2():
    sample = mrg.normal((3,3), 0, 1, dtype=theano.config.floatX)
    return sample

out2, update2 = theano.scan(step2, None, None, n_steps=5)
func4 = theano.function([], out2, allow_input_downcast=True)

print('...MRG_RandomStreams in Scan')
for i in range(3):
    print(i, func4())

# Test RandomStreams in scan op with input

oi = T.fmatrix('ooii')

def step3(input_):
    sample = rng.normal(input_.shape, 0, 1, dtype=theano.config.floatX)
    return input_ * sample

out3, update3 = theano.scan(step3, None, [oi], n_steps=5)
func5 = theano.function([oi], out3, updates=update3, allow_input_downcast=True)

print('...RandomStreams in Scan using shape')
for i in range(3):
    print(i, func5(np.array([[1,2,3],[2,3,4],[3,4,5]])))