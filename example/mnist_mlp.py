"""
This code is an example of how to train MNIST classification.
The most easiest deep learning example.
Good to learn how to use LemonTree.
"""

import time
import numpy as np
import theano
import theano.tensor as T

from lemontree.data.mnist import MNIST
from lemontree.data.generators import SimpleGenerator
from lemontree.controls.history import HistoryWithEarlyStopping
from lemontree.controls.scheduler import LearningRateMultiplyScheduler
from lemontree.graphs.graph import SimpleGraph
from lemontree.layers.activation import ReLU, Softmax, GumbelSoftmax
from lemontree.layers.dense import DenseLayer
from lemontree.layers.normalization import BatchNormalization1DLayer
from lemontree.initializers import HeNormal, Constant
from lemontree.objectives import CategoricalAccuracy, CategoricalCrossentropy
from lemontree.optimizers import Adam
from lemontree.parameters import SimpleParameter
from lemontree.utils.param_utils import filter_params_by_tags, print_tags_in_params
from lemontree.utils.type_utils import merge_dicts
from lemontree.utils.graph_utils import get_inputs_of_variables

np.random.seed(9999)
base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
experiment_name = 'mnist_mlp'

mnist = MNIST(base_datapath, 'flat')
mnist.split_train_valid(50000)
train_data, train_label = mnist.get_fullbatch_train()
test_data, test_label = mnist.get_fullbatch_test()
valid_data, valid_label = mnist.get_fullbatch_valid()

train_gen = SimpleGenerator('train', 250)
train_gen.initialize(train_data, train_label)
test_gen = SimpleGenerator('test', 250)
test_gen.initialize(test_data, test_label)
valid_gen = SimpleGenerator('valid', 250)
valid_gen.initialize(valid_data, valid_label)

def build_model(input_, name):
    graph = SimpleGraph(name)
    graph.add_input(input_)
    graph.add_layers([DenseLayer((784,),(1024,), use_bias=False, name='dense1'),
                      BatchNormalization1DLayer((1024,), 0.9, name='bn1'),
                      ReLU(name='relu1'),
                      DenseLayer((1024,),(1024,), use_bias=False, name='dense2'),
                      BatchNormalization1DLayer((1024,), 0.9, name='bn2'),
                      ReLU(name='relu2'),
                      DenseLayer((1024,),(1024,), use_bias=False, name='dense3'),
                      BatchNormalization1DLayer((1024,), 0.9, name='bn3'),
                      ReLU(name='relu3'),
                      DenseLayer((1024,),(1024,), use_bias=False, name='dense4'),
                      BatchNormalization1DLayer((1024,), 0.9, name='bn4'),
                      ReLU(name='relu4'),
                      DenseLayer((1024,),(10,), name='dense5'),
                      GumbelSoftmax(name='softmax1')])

    loss = CategoricalCrossentropy().get_loss(graph.get_output(), y)
    accuracy = CategoricalAccuracy().get_loss(graph.get_output(), y)
    outputs = [loss, accuracy]

    return graph, outputs

def compile_train_function(outputs, updates):
    func = theano.function(inputs=get_inputs_of_variables(outputs),
                           outputs=outputs,
                           updates=updates,
                           allow_input_downcast=True)
    return func

def compile_test_function(outputs):
    func = theano.function(inputs=get_inputs_of_variables(outputs),
                           outputs=outputs,
                           allow_input_downcast=True)
    return func

    graph_params = graph.get_params()
    graph_updates = graph.get_updates()

def run_with_inputs(func, inputs):
    return func(*inputs)

x = T.fmatrix('X')
y = T.ivector('y')

HeNormal().initialize_params(filter_params_by_tags(graph_params, ['weight']))
print_tags_in_params(graph_params)

adam = Adam(0.002)
optimizer_updates = adam.get_updates(loss, graph_params)
optimizer_params = adam.get_params()

total_params = optimizer_params + graph_params
total_updates = merge_dicts([optimizer_updates, graph_updates])

params_saver = SimpleParameter(total_params, experiment_name + '_params/')
params_saver.save_params()

graph_inputs = get_inputs_of_variables([loss, accuracy])

train_func = compile_train_function([loss, accuracy], total_updates)
test_func = compile_test_function([loss, accuracy])

lr_scheduler = LearningRateMultiplyScheduler(adam.lr, 0.2)
hist = HistoryWithEarlyStopping(experiment_name + '_history/', 10, 5)
hist.add_keys(['train_accuracy',  'valid_accuracy', 'test_accuracy'])

change_lr = False
end_train = False
for epoch in range(1000):
    if end_train:
        params_saver.load_params()
        break
    if change_lr:
        params_saver.load_params()
        lr_scheduler.change_learningrate(epoch)
        Constant(0).initialize_params(filter_params_by_tags(total_params, ['optimizer_param']))
    train_gen.shuffle()

    print('...Epoch', epoch)
    start_time = time.clock()
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

graph.change_flag(-1)
test_loss = []
test_accuracy = []
for index in range(test_gen.max_index):
    testset = test_gen.get_minibatch(index)
    test_batch_loss, test_batch_accuracy = test_func(*testset)
    test_loss.append(test_batch_loss)
    test_accuracy.append(test_batch_accuracy)
hist.history['test_loss'].append(np.mean(np.asarray(test_loss)))
hist.history['test_accuracy'].append(np.mean(np.asarray(test_accuracy)))
hist.print_history_of_epoch()
hist.save_history_to_csv()

