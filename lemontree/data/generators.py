# Kyuhong Shim 2016

import numpy as np


class BaseGenerator(object):

    def __init__(self, name=None, batchsize=128):
        self.name = name
        self.batchsize = batchsize

    def initialize(self, data, label):
        self.data = data
        self.label = label
        self.ndata = len(data)  # data.shape[0]
        self.order = np.random.permutation(self.ndata)
        self.max_index = self.ndata // self.batchsize

    def shuffle(self):
        self.order = np.random.permutation(self.ndata)

    def change_batchsize(self, newbatchsize):
        self.batchsize = newbatchsize
        self.max_index = self.ndata // self.batchsize

    def get_minibatch(self, index):
        assert index <= self.max_index
        batch_data = self.data[self.order[self.batchsize * index: self.batchsize * (index+1)]]
        batch_label = self.label[self.order[self.batchsize * index: self.batchsize * (index+1)]]
        return (batch_data, batch_label)

    def get_fullbatch(self):
        return (self.data, self.label)


class ImageGenerator(BaseGenerator):

    def __init__(self, name=None, batchsize=128):
        super(ImageGenerator, self).__init__(name, batchsize)
        self.flip_lr = False
        self.flip_ud = False

    def set_flip_lr_true(self):
        if self.flip_lr:
            self.flip_lr = False
        else:
            self.flip_lr = True

    def set_flip_ud_true(self):
        if self.flip_ud:
            self.flip_ud = False
        else:
            self.flip_ud = True

    def rgb_to_yuv(self):
        # assume r, g, b order in channel
        new_data = np.zeros_like(self.data, self.data.dtype)
        new_data[:, 0, :, :] = 0.299 * self.data[:, 0, :, :] + 0.587 * self.data[:, 1, :, :] + 0.114 * self.data[:, 2, :, :]
        new_data[:, 1, :, :] = 0.5 - 0.168736 * self.data[:, 0, :, :] - 0.331364 * self.data[:, 1, :, :] + 0.5 * self.data[:, 2, :, :]
        new_data[:, 2, :, :] = 0.5 + 0.5 * self.data[:, 0, :, :] - 0.418688 * self.data[:, 1, :, :] - 0.081312 * self.data[:, 2, :, :]
        self.data = new_data

    def yuv_to_rgb(self):
        # assume y, u, v order (or y, cb, cr) in channel
        new_data = np.zeros_like(self.data, self.data.dtype)
        new_data[:, 0, :, :] = self.data[:, 0, :, :] + 1.402 * (self.data[:, 2, :, :] - 0.5)
        new_data[:, 1, :, :] = self.data[:, 0, :, :] - 0.34414 * (self.data[:, 1, :, :] - 0.5) - 0.71414 * (self.data[:, 2, :, :] - 0.5)
        new_data[:, 2, :, :] = self.data[:, 0, :, :] + 1.772 * (self.data[:, 1, :, :] - 0.5)
        self.data = new_data

    def gcn(self, mean=None, std=None):
        if mean is None and std is None:
            mean = np.mean(self.data, axis=0)
            std = np.std(self.data, axis=0)
        result = (self.data - mean) / (std + 1e-8)
        self.data = result
        return mean, std

    def lcn(self):
        if len(self.data.shape) == 4:
            flat = np.reshape(self.data, (self.data.shape[0], np.prod(self.data.shape[1:])))
        elif len(self.data.shape) == 2:
            flat = self.data
        else:
            raise NotImplementedError('Not yet implemented')

        mean = np.mean(flat, axis=1)
        flat = flat - mean[:, np.newaxis]
        # std = np.sqrt(np.sum(flat ** 2, axis = 1)) + 1e-8
        std = np.std(flat, axis=1)
        result = flat / std[:, np.newaxis]

        if len(self.data.shape) == 4:
            result = np.reshape(result, self.data.shape)
        self.data = result

    def zca(self, pc_matrix=None):
        flat = np.reshape(self.data, (self.data.shape[0], np.prod(self.data.shape[1:])))
        # print('Flatten shape: ', flat.shape)

        if pc_matrix is None:
            sigma = np.dot(flat.T, flat) / flat.shape[0]
            U, S, V = np.linalg.svd(sigma)
            newS = np.diag(1.0 / (np.sqrt(S) + 1e-8))
            self.pc_matrix = np.dot(np.dot(U, newS), np.transpose(U))
        else:
            self.pc_matrix = pc_matrix

        white = np.dot(flat, self.pc_matrix)
        result = np.reshape(white, self.data.shape)
        self.data = result
        return self.pc_matrix

    def get_minibatch(self, index):
        assert index <= self.max_index
        batch_data = self.data[self.order[self.batchsize * index: self.batchsize * (index+1)]]
        batch_label = self.label[self.order[self.batchsize * index: self.batchsize * (index+1)]]
        if self.flip_lr:
            random_choice = np.random.permutation(self.batchsize)[:self.batchsize//2]
            batch_data[random_choice] = batch_data[random_choice, :, :, ::-1]
        if self.flip_ud:
            random_choice = np.random.permutation(self.batchsize)[:self.batchsize//2]
            batch_data[random_choice] = batch_data[random_choice, :, ::-1]
        return (batch_data, batch_label)


class SequenceGenerator(BaseGenerator):

    def __init__(self, name=None, batchsize=128, sequence_length=50):
        super(SequenceGenerator, self).__init__(name, batchsize)
        self.sequence_length = sequence_length

    def initialize(self, data, label):
        self.data = []
        for i in range(len(data) // (self.batchsize * self.sequence_length)):
            self.data.append
        
    #def initialize(self, data, sort=False):
    #    if sort:
    #        self.data = sorted(data, key=len)  # list of sentences (strings)
    #    else:
    #        self.data = data
    #    self.ndata = len(data)
        
    #    if self.bucket > 0:
    #        assert sort  # To use bucketing, we should sort sentences
    #        self.bucket_key = []
    #        for ind in range(self.ndata // self.bucket, self.ndata, self.ndata // self.bucket):
    #            self.bucket_key.append(len(self.data[ind]))
    #        self.bucket_key = self.bucket_key[:self.bucket-1]



