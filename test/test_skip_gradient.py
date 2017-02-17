import numpy as np
import theano
import theano.tensor as T

x = T.fmatrix('X')

w1_np = np.array([[0,1],[1,0]], dtype='float32')
w2_np = np.array([[0.1, 0.3], [-0.1, -0.5]], dtype='float32')

w1 = theano.shared(w1_np, 'W1')
w2 = theano.shared(w2_np, 'W2')

y = T.dot(x, w1) + w2

out = T.sum(y)
updates1 = {}
updates2 = {}
updates1[w1] = w1 - 0.1 * T.grad(out, w1)
updates2[w2] = w2 - 0.1 * T.grad(out, w2)

func1 = theano.function([x],[y, out], updates=updates1, allow_input_downcast=True)
func2 = theano.function([x],[y, out], updates=updates2, allow_input_downcast=True)

x_np = np.random.uniform(0, 1, (2,2))

print('Before')
print(w1.get_value())
print(w2.get_value())

print('Func1')
y_np, out_np = func1(x_np)
print(w1.get_value())
print(w2.get_value())

print('Func2')
y_np, out_np = func2(x_np)
print(w1.get_value())
print(w2.get_value())

