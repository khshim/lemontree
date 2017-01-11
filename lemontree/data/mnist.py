# Kyuhong Shim 2016

import numpy as np
import scipy.io

# MNIST data as matlab file.
# http://www.cs.nyu.edu/~roweis/data.html
# Download mnist_all.mat
# Mapping [0,255] to [0,1]


def uint_to_float(data):
    return np.asarray(data, dtype='float32')


def load_mnist(base_datapath, mode='flat'):
    # mnist = scipy.io.loadmat('data/dataset/mnist_all.mat')
    mnist = scipy.io.loadmat(base_datapath + 'mnist/mnist_all.mat')
    test0 = uint_to_float(mnist['test0']) / 255.0
    test1 = uint_to_float(mnist['test1']) / 255.0
    test2 = uint_to_float(mnist['test2']) / 255.0
    test3 = uint_to_float(mnist['test3']) / 255.0
    test4 = uint_to_float(mnist['test4']) / 255.0
    test5 = uint_to_float(mnist['test5']) / 255.0
    test6 = uint_to_float(mnist['test6']) / 255.0
    test7 = uint_to_float(mnist['test7']) / 255.0
    test8 = uint_to_float(mnist['test8']) / 255.0
    test9 = uint_to_float(mnist['test9']) / 255.0
    train0 = uint_to_float(mnist['train0']) / 255.0
    train1 = uint_to_float(mnist['train1']) / 255.0
    train2 = uint_to_float(mnist['train2']) / 255.0
    train3 = uint_to_float(mnist['train3']) / 255.0
    train4 = uint_to_float(mnist['train4']) / 255.0
    train5 = uint_to_float(mnist['train5']) / 255.0
    train6 = uint_to_float(mnist['train6']) / 255.0
    train7 = uint_to_float(mnist['train7']) / 255.0
    train8 = uint_to_float(mnist['train8']) / 255.0
    train9 = uint_to_float(mnist['train9']) / 255.0

    train_data = np.vstack((train0, train1, train2, train3, train4,
                            train5, train6, train7, train8, train9))
    test_data = np.vstack((test0, test1, test2, test3, test4,
                           test5, test6, test7, test8, test9))

    train_label = np.hstack((np.tile(np.array([0]), (train0.shape[0],)),
                             np.tile(np.array([1]), (train1.shape[0],)),
                             np.tile(np.array([2]), (train2.shape[0],)),
                             np.tile(np.array([3]), (train3.shape[0],)),
                             np.tile(np.array([4]), (train4.shape[0],)),
                             np.tile(np.array([5]), (train5.shape[0],)),
                             np.tile(np.array([6]), (train6.shape[0],)),
                             np.tile(np.array([7]), (train7.shape[0],)),
                             np.tile(np.array([8]), (train8.shape[0],)),
                             np.tile(np.array([9]), (train9.shape[0],))))
    test_label = np.hstack((np.tile(np.array([0]), (test0.shape[0],)),
                            np.tile(np.array([1]), (test1.shape[0],)),
                            np.tile(np.array([2]), (test2.shape[0],)),
                            np.tile(np.array([3]), (test3.shape[0],)),
                            np.tile(np.array([4]), (test4.shape[0],)),
                            np.tile(np.array([5]), (test5.shape[0],)),
                            np.tile(np.array([6]), (test6.shape[0],)),
                            np.tile(np.array([7]), (test7.shape[0],)),
                            np.tile(np.array([8]), (test8.shape[0],)),
                            np.tile(np.array([9]), (test9.shape[0],))))

    train_order = np.random.permutation(train_label.shape[0])
    test_order = np.random.permutation(test_label.shape[0])

    train_data = train_data[train_order]
    test_data = test_data[test_order]
    train_label = train_label[train_order]
    test_label = test_label[test_order]

    if mode == 'flat':
        pass
    elif mode == 'tensor':
        train_data = np.reshape(train_data, (train_data.shape[0], 1, 28, 28))
        test_data = np.reshape(test_data, (test_data.shape[0], 1, 28, 28))
    else:
        raise NotImplementedError('No such mode exist.')

    print('Train data shape: ', train_data.shape, train_data.dtype)
    print('Test data shape: ', test_data.shape, test_data.dtype)
    print('Train label shape: ', train_label.shape, train_label.dtype)
    print('Test label shape: ', test_label.shape, test_label.dtype)

    return train_data, train_label, test_data, test_label
