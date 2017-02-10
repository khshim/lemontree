import theano
import theano.tensor as T
import numpy as np

arr = np.array([1.2, 3.4, 5.6])
new_arr = np.array([2.1, 3.2, 4.3])

class C1(object):
    def __init__(self, member):
        self.member = member

class C2(object):
    def __init__(self, member):
        self.member = member

arr_shared = theano.shared(arr, 'ARR')

c1 = C1(arr_shared)
c2 = C2(arr_shared)

print(arr_shared.get_value())
print(c1.member.get_value())
print(c2.member.get_value())

arr_shared.set_value(new_arr)
print(arr_shared.get_value())
print(c1.member.get_value())
print(c2.member.get_value())

c1.member.set_value(arr)
print(arr_shared.get_value())
print(c1.member.get_value())
print(c2.member.get_value())