# Kyuhong Shim 2016

import numpy as np
import scipy.io

# CIFAR-10 data as matlab file.
# https://www.cs.toronto.edu/~kriz/cifar.html
# Download data_batch_1.mat ~ data_batch_5.mat, test_batch.mat
# GCN + ZCA whitening


def uint_to_float(data):
    return np.asarray(data, dtype='float32')


def uint_to_int(data):
    return np.asarray(data, dtype='int32')


def load_cifar10(base_datapath, mode='tensor'):
    cifar_train1 = scipy.io.loadmat(base_datapath + 'cifar10/data_batch_1.mat')
    cifar_train2 = scipy.io.loadmat(base_datapath + 'cifar10/data_batch_2.mat')
    cifar_train3 = scipy.io.loadmat(base_datapath + 'cifar10/data_batch_3.mat')
    cifar_train4 = scipy.io.loadmat(base_datapath + 'cifar10/data_batch_4.mat')
    cifar_train5 = scipy.io.loadmat(base_datapath + 'cifar10/data_batch_5.mat')
    cifar_test = scipy.io.loadmat(base_datapath + 'cifar10/test_batch.mat')
    train1_data = uint_to_float(cifar_train1['data']) / 255.0
    train2_data = uint_to_float(cifar_train2['data']) / 255.0
    train3_data = uint_to_float(cifar_train3['data']) / 255.0
    train4_data = uint_to_float(cifar_train4['data']) / 255.0
    train5_data = uint_to_float(cifar_train5['data']) / 255.0
    test_data = uint_to_float(cifar_test['data']) / 255.0
    train1_label = uint_to_int(cifar_train1['labels'])
    train2_label = uint_to_int(cifar_train2['labels'])
    train3_label = uint_to_int(cifar_train3['labels'])
    train4_label = uint_to_int(cifar_train4['labels'])
    train5_label = uint_to_int(cifar_train5['labels'])
    test_label = uint_to_int(cifar_test['labels'])

    train_data = np.vstack((train1_data, train2_data, train3_data,
                            train4_data, train5_data))
    train_label = np.vstack((train1_label, train2_label, train3_label,
                             train4_label, train5_label))

    train_label = np.squeeze(train_label)
    test_label = np.squeeze(test_label)

    train_order = np.random.permutation(train_label.shape[0])
    test_order = np.random.permutation(test_label.shape[0])

    train_data = train_data[train_order]
    test_data = test_data[test_order]
    train_label = train_label[train_order]
    test_label = test_label[test_order]

    if mode == 'tensor':
        train_data = np.reshape(train_data, (train_data.shape[0], 3, 32, 32))
        test_data = np.reshape(test_data, (test_data.shape[0], 3, 32, 32))
    elif mode == 'flat':
        pass
    else:
        raise NotImplementedError('No such mode exist')

    print('Train data shape: ', train_data.shape, train_data.dtype)
    print('Test data shape: ', test_data.shape, test_data.dtype)

    print('Train label shape: ', train_label.shape, train_label.dtype)
    print('Test label shape: ', test_label.shape, test_label.dtype)

    return train_data, train_label, test_data, test_label


def label_to_real(label):
    if label == 0:
        return 'airplane'
    elif label == 1:
        return 'automobile'
    elif label == 2:
        return 'bird'
    elif label == 3:
        return 'cat'
    elif label == 4:
        return 'deer'
    elif label == 5:
        return 'dog'
    elif label == 6:
        return 'frog'
    elif label == 7:
        return 'horse'
    elif label == 8:
        return 'ship'
    elif label == 9:
        return 'truck'
    else:
        raise NotImplementedError('No such label exist')
