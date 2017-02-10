"""
This code is an example of how to train CIFAR-10 classification.
The model is Wide Residual Networks (like).
See "Wide Residual Networks". WRN-16-8 is the structure below.
(Sergey Zagoruyko, Nikos Komodakis, 2016.)
"""

import time
import numpy as np
import theano
import theano.tensor as T

from lemontree.data.cifar10 import CIFAR10
from lemontree.generators.image import ImageGenerator
from lemontree.controls.history import HistoryWithEarlyStopping
from lemontree.controls.scheduler import LearningRateMultiplyScheduler
from lemontree.graphs.graph import SimpleGraph
from lemontree.layers.activation import ReLU, Softmax
from lemontree.layers.convolution import Convolution3DLayer
from lemontree.layers.pool import Pooling3DLayer, GlobalAveragePooling3DLayer
from lemontree.layers.dense import DenseLayer
from lemontree.layers.normalization import BatchNormalization1DLayer, BatchNormalization3DLayer
from lemontree.layers.shape import Flatten3DLayer
from lemontree.layers.merge import MergeAddLayer
from lemontree.layers.dropout import DropoutLayer
from lemontree.initializers import HeNormal
from lemontree.objectives import CategoricalAccuracy, CategoricalCrossentropy
from lemontree.optimizers import Adam, RMSprop
from lemontree.parameters import SimpleParameter
from lemontree.utils.param_utils import filter_params_by_tags, print_tags_in_params, print_params_num
from lemontree.utils.type_utils import merge_dicts
from lemontree.utils.graph_utils import get_inputs_of_variables
from lemontree.utils.data_utils import int_to_onehot

np.random.seed(9999)
base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
# base_datapath = 'D:/Dropbox/Project/data/'
# base_datapath = '/home/khshim/data/'
experiment_name = 'cifar10_wrn'

#================Prepare data================#

cifar10 = CIFAR10(base_datapath, 'tensor')
cifar10.split_train_valid(40000)
train_data, train_label = cifar10.get_fullbatch_train()
test_data, test_label = cifar10.get_fullbatch_test()
valid_data, valid_label = cifar10.get_fullbatch_valid()

train_gen = ImageGenerator([train_data, train_label], 128,
                          flip_lr=True, flip_ud=False, padding=(2,2,2,2), crop_size=(32,32), name='train')
test_gen = ImageGenerator([test_data, test_label], 128,
                          flip_lr=True, flip_ud=False, padding=(2,2,2,2), crop_size=(32,32), name='test')
valid_gen = ImageGenerator([valid_data, valid_label], 128,
                           flip_lr=True, flip_ud=False, padding=(2,2,2,2), crop_size=(32,32), name='valid')

train_global_mean, train_global_std = train_gen.gcn()
train_pc_matrix = train_gen.zca()
test_gen.gcn(train_global_mean, train_global_std)
test_gen.zca(train_pc_matrix)
valid_gen.gcn(train_global_mean, train_global_std)
valid_gen.zca(train_pc_matrix)

#================Build graph================#

x = T.ftensor4('X')
y = T.ivector('y')

graph = SimpleGraph(experiment_name, 128)
graph.add_layer(Convolution3DLayer((3,32,32), (16,32,32), (3,3), 'half', use_bias=False), is_start=True)  # 0
graph.add_layer(BatchNormalization3DLayer((16,32,32), 0.95))                                    # 1
graph.add_layer(ReLU())                                                                         # 2

graph.add_layer(Convolution3DLayer((16,32,32), (128,32,32), (3,3), 'half', use_bias=False))     # 3
graph.add_layer(BatchNormalization3DLayer((128,32,32), 0.95))                                   # 4
graph.add_layer(ReLU())                                                                         # 5
graph.add_layer(Convolution3DLayer((128,32,32), (128,32,32), (3,3), 'half'))                    # 6
graph.add_layer(Convolution3DLayer((16,32,32), (128,32,32), (1,1)), get_from=[-5])              # 7
graph.add_layer(MergeAddLayer((128,32,32)), get_from=[-1,-2])                                   # 8

graph.add_layer(BatchNormalization3DLayer((128,32,32), 0.95))                                   # 9
graph.add_layer(ReLU())                                                                         # 10
graph.add_layer(Convolution3DLayer((128,32,32), (128,32,32), (3,3), 'half', use_bias=False))    # 11
graph.add_layer(BatchNormalization3DLayer((128,32,32), 0.95))                                   # 12
graph.add_layer(ReLU())                                                                         # 13
graph.add_layer(Convolution3DLayer((128,32,32), (128,32,32), (3,3), 'half'))                    # 14
graph.add_layer(MergeAddLayer((128,32,32)), get_from=[-1,-7])                                   # 15

graph.add_layer(BatchNormalization3DLayer((128,32,32), 0.95))                                   # 16
graph.add_layer(ReLU())                                                                         # 17

graph.add_layer(Convolution3DLayer((128,32,32), (256,16,16), (3,3), 'half', (2,2), use_bias=False))     # 18
graph.add_layer(BatchNormalization3DLayer((256,16,16), 0.95))                                   # 19
graph.add_layer(ReLU())                                                                         # 20
graph.add_layer(Convolution3DLayer((256,16,16), (256,16,16), (3,3), 'half'))                    # 21
graph.add_layer(Convolution3DLayer((128,32,32), (256,16,16), (1,1), 'valid', (2,2)), get_from=[-5])     # 22
graph.add_layer(MergeAddLayer((256,16,16)), get_from=[-1,-2])                                   # 23

graph.add_layer(BatchNormalization3DLayer((256,16,16), 0.95))                                   # 24
graph.add_layer(ReLU())                                                                         # 25
graph.add_layer(Convolution3DLayer((256,16,16), (256,16,16), (3,3), 'half', use_bias=False))    # 26
graph.add_layer(BatchNormalization3DLayer((256,16,16), 0.95))                                   # 27
graph.add_layer(ReLU())                                                                         # 28
graph.add_layer(Convolution3DLayer((256,16,16), (256,16,16), (3,3), 'half'))                    # 29
graph.add_layer(MergeAddLayer((256,16,16)), get_from=[-1,-7])                                   # 30

graph.add_layer(BatchNormalization3DLayer((256,16,16), 0.95))                                   # 31
graph.add_layer(ReLU())                                                                         # 32

graph.add_layer(Convolution3DLayer((256,16,16), (512,8,8), (3,3), 'half', (2,2), use_bias=False))       # 33
graph.add_layer(BatchNormalization3DLayer((512,8,8), 0.95))                                     # 34
graph.add_layer(ReLU())                                                                         # 35
graph.add_layer(Convolution3DLayer((512,8,8), (512,8,8), (3,3), 'half'))                        # 36
graph.add_layer(Convolution3DLayer((256,16,16), (512,8,8), (1,1), 'valid', (2,2)), get_from=[-5])     # 37
graph.add_layer(MergeAddLayer((256,16,16)), get_from=[-1,-2])                                   # 38

graph.add_layer(BatchNormalization3DLayer((512,8,8), 0.95))                                     # 39
graph.add_layer(ReLU())                                                                         # 40
graph.add_layer(Convolution3DLayer((512,8,8), (512,8,8), (3,3), 'half', use_bias=False))        # 41
graph.add_layer(BatchNormalization3DLayer((512,8,8), 0.95))                                     # 42
graph.add_layer(ReLU())                                                                         # 43
graph.add_layer(Convolution3DLayer((512,8,8), (512,8,8), (3,3), 'half'))                        # 44
graph.add_layer(MergeAddLayer((512,8,8)), get_from=[-1,-7])                                     # 45

graph.add_layer(BatchNormalization3DLayer((512,8,8), 0.95))                                     # 46
graph.add_layer(ReLU())                                                                         # 47

graph.add_layer(GlobalAveragePooling3DLayer((512,8,8), (512,)))                                 # 48
graph.add_layer(DenseLayer((512,), (10,)))                                                      # 49
graph.add_layer(Softmax())                                                                      # 50

graph_output, graph_layers = graph.get_output({0:[x]}, -1, 0)

loss = CategoricalCrossentropy().get_output(graph_output, y)
accuracy = CategoricalAccuracy().get_output(graph_output, y)

graph_params = graph.get_params(graph_layers)
graph_updates = graph.get_updates(graph_layers)

#================Prepare arguments================#

HeNormal().initialize_params(filter_params_by_tags(graph_params, ['weight']))
# print_tags_in_params(graph_params)
print_params_num(graph_params)

optimizer = RMSprop(0.001)  # Adam(0.002)
optimizer_updates = optimizer.get_updates(loss, graph_params)
optimizer_params = optimizer.get_params()

total_params = optimizer_params + graph_params
total_updates = merge_dicts([optimizer_updates, graph_updates])

params_saver = SimpleParameter(total_params, experiment_name + '_params/')
params_saver.save_params()
# params_saver.load_params()

lr_scheduler = LearningRateMultiplyScheduler(optimizer.lr, 0.1)
hist = HistoryWithEarlyStopping(experiment_name + '_history/', 10, 5)
hist.add_keys(['train_accuracy',  'valid_accuracy', 'test_accuracy'])

#================Compile functions================#

outputs = [loss, accuracy]
graph_inputs = get_inputs_of_variables(outputs)

train_func = theano.function(inputs=graph_inputs,
                             outputs=outputs,
                             updates=total_updates,
                             allow_input_downcast=True)

test_func = theano.function(inputs=graph_inputs,
                            outputs=outputs,
                            allow_input_downcast=True)

test_func_output = theano.function(inputs=[x],
                                   outputs=T.argmax(graph_output, axis=-1),
                                   allow_input_downcast=True)


#================Convenient functions================#

def train_trainset():
    graph.change_flag(1)
    train_loss = []
    train_accuracy = []
    for index in range(train_gen.max_index):
        trainset = train_gen.get_minibatch(index)
        train_batch_loss, train_batch_accuracy = train_func(*trainset)
        train_loss.append(train_batch_loss)
        train_accuracy.append(train_batch_accuracy)
    hist.history['train_loss'].append(np.mean(np.asarray(train_loss)))
    hist.history['train_accuracy'].append(np.mean(np.asarray(train_accuracy)))

def test_validset():
    graph.change_flag(-1)
    valid_loss = []
    valid_accuracy = []
    for index in range(valid_gen.max_index):
        validset = valid_gen.get_minibatch(index)
        valid_batch_loss, valid_batch_accuracy = test_func(*validset)
        valid_loss.append(valid_batch_loss)
        valid_accuracy.append(valid_batch_accuracy)
    hist.history['valid_loss'].append(np.mean(np.asarray(valid_loss)))
    hist.history['valid_accuracy'].append(np.mean(np.asarray(valid_accuracy)))

def test_testset():
    graph.change_flag(-1)
    test_accuracy = []
    for index in range(test_gen.max_index):        
        confusion_matrix = np.zeros((128, 10)).astype('int32')
        for times in range(10):
            testset = test_gen.get_minibatch(index)  # re-sample again, same data, different preprocessing
            test_output = test_func_output(testset[0])
            test_output = int_to_onehot(test_output, 10)
            confusion_matrix += test_output
        testset = test_gen.get_minibatch(index)
        test_batch_answer = np.argmax(confusion_matrix, axis=-1)
        test_batch_accuracy = np.mean(np.equal(test_batch_answer, testset[1]))
        test_accuracy.append(test_batch_accuracy)
    hist.history['test_accuracy'].append(np.mean(np.asarray(test_accuracy)))

#================Train================#

change_lr = False
end_train = False
for epoch in range(1000):
    if end_train:
        params_saver.load_params()
        break
    if change_lr:
        params_saver.load_params()
        lr_scheduler.change_learningrate(epoch)
        optimizer.reset_params()
    train_gen.shuffle()

    print('...Epoch', epoch)
    start_time = time.clock()

    train_trainset()
    test_validset()

    end_time = time.clock()
    print('......time:', end_time - start_time)

    hist.print_history_of_epoch()
    checker = hist.check_earlystopping()
    if checker == 'save_param':
        params_saver.save_params()
        change_lr = False
        end_train = False
    elif checker == 'change_lr':
        change_lr = True
        end_train = False
    elif checker == 'end_train':
        change_lr = False
        end_train = True
    elif checker == 'keep_train':
        change_lr = False
        end_train = False
    else:
        raise NotImplementedError('Not supported checker type')

#================Test================#

test_testset()
best_loss, best_epoch = hist.best_loss_and_epoch_of_key('valid_loss')
hist.print_history_of_epoch(best_epoch, ['train_loss', 'train_accuracy', 'valid_loss', 'valid_accuracy'])
hist.print_history_of_epoch(-1, ['test_accuracy'])
hist.save_history_to_csv()