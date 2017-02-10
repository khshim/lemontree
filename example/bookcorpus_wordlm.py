"""
This code is an example of how to train word level language model.
The model is single layer LSTM (500).
We use GloVe word embeddings. (400003 vocabulary)
"""

import time
import numpy as np
import theano
import theano.tensor as T

from lemontree.data.mnist import MNIST
from lemontree.data.glove import GloveData
from lemontree.data.bookcorpus import BookCorpusWordCorpus
from lemontree.graphs.graph import SimpleGraph
from lemontree.generators.sequence import WordLMGenerator
from lemontree.controls.history import HistoryWithEarlyStopping
from lemontree.controls.scheduler import LearningRateMultiplyScheduler
from lemontree.experimentals.dense_scp import TimeDistributedDenseLayerSCP
from lemontree.layers.activation import ReLU, Softmax
from lemontree.layers.dense import TimeDistributedDenseLayer, DenseLayer
from lemontree.layers.lstm import LSTMRecurrentLayer
from lemontree.initializers import GlorotNormal
from lemontree.objectives import CategoricalCrossentropy, CategoricalAccuracy, WordPerplexity
from lemontree.optimizers import RMSprop
from lemontree.parameters import SimpleParameter
from lemontree.utils.data_utils import sentence_bucketing, sentence_padding
from lemontree.utils.param_utils import filter_params_by_tags, print_tags_in_params, print_params_num
from lemontree.utils.graph_utils import get_inputs_of_variables
from lemontree.utils.type_utils import merge_dicts

np.random.seed(9999)
# base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
# base_datapath = 'D:/Dropbox/Project/data/'
base_datapath = '/home/khshim/data/'
experiment_name = 'bookcorpus_wordlm'

#================Prepare data================#

corpus1 = BookCorpusWordCorpus(base_datapath, 'books_large_p1_100000')  # pickle data
corpus2 = BookCorpusWordCorpus(base_datapath, 'books_large_p2_100000')  # pickle data
train_data = corpus1.train_data + corpus2.train_data
test_data = corpus1.test_data + corpus2.test_data
valid_data = corpus1.valid_data + corpus2.valid_data

print('Train data length:', len(train_data))
print('Test data length:', len(test_data))
print('Valid data length:', len(valid_data))

batch_size = 50
sequence_length = 20
stride_length = 10
buckets = [20,40,60,80,100]

train_data, train_mask = sentence_bucketing(train_data, buckets)
test_data, test_mask = sentence_bucketing(test_data, buckets)
valid_data, valid_mask = sentence_bucketing(valid_data, buckets)

glove = GloveData(base_datapath, load_pickle=True)  # pickle dict and embeddings
train_gens = []
test_gens = []
valid_gens = []
for bb in range(len(buckets)):
    if len(train_data[bb]) >= batch_size:
        train_gens.append(WordLMGenerator([train_data[bb], train_mask[bb]], glove, \
            sequence_length, stride_length, buckets[bb], batch_size))
    if len(test_data[bb]) >= batch_size:
        test_gens.append(WordLMGenerator([test_data[bb], test_mask[bb]], glove, \
            sequence_length, stride_length, buckets[bb], batch_size))
    if len(valid_data[bb]) >= batch_size:
        valid_gens.append(WordLMGenerator([valid_data[bb], valid_mask[bb]], glove, \
            sequence_length, stride_length, buckets[bb], batch_size))

#================Build graph================#

x = T.ftensor3('X')  # (batch_size, sequence_length, 300)
m = T.wmatrix('M')  # (batch_size, sequence_length)
y = T.imatrix('Y')  # (batch_size, sequence_length)
r = T.wvector('r')  # (batch_size,)

graph = SimpleGraph(experiment_name, batch_size)
graph.add_layer(LSTMRecurrentLayer(input_shape=(300,),
                                   output_shape=(1024,),
                                   forget_bias_one=True,
                                   peephole=True,
                                   output_return_index=None,
                                   save_state_index=stride_length-1,
                                   precompute=False,
                                   unroll=False,
                                   backward=False), is_start=True)
# graph.add_layer(TimeDistributedDenseLayer((1024,), (512,)))  # not much time difference, and less memory
graph.add_layer(DenseLayer((1024,), (512,)))
graph.add_layer(TimeDistributedDenseLayerSCP((512,), (glove.vocabulary,)))

graph_output, graph_layers = graph.get_output({0:[x,m,r], -1:[y,m]}, -1, 0)
loss = graph_output[0]
perplexity = graph_output[1]

graph_params = graph.get_params()
graph_updates = graph.get_updates()

#================Prepare arguments================#

GlorotNormal().initialize_params(filter_params_by_tags(graph_params, ['weight']))
print_params_num(graph_params)

optimizer = RMSprop(0.0001, clipnorm=5.0)
optimizer_updates = optimizer.get_updates(loss, graph_params)
optimizer_params = optimizer.get_params()

total_params = optimizer_params + graph_params
total_updates = merge_dicts([optimizer_updates, graph_updates])

params_saver = SimpleParameter(total_params, experiment_name + '_params/')
params_saver.save_params()

lr_scheduler = LearningRateMultiplyScheduler(optimizer.lr, 0.2)
hist = HistoryWithEarlyStopping(experiment_name + '_history/', 5, 5)
hist.add_keys(['train_perplexity', 'valid_perplexity', 'test_perplexity'])

#================Compile functions================#

outputs = [loss, perplexity]
graph_inputs = [x,m,y,r]

train_func = theano.function(inputs=graph_inputs,
                             outputs=outputs,
                             updates=total_updates,
                             allow_input_downcast=True)

test_func = theano.function(inputs=graph_inputs,
                            outputs=outputs,
                            allow_input_downcast=True)

#================Convenient functions================#


def train_trainset():
    train_loss = []
    train_perplexity = []
    start_time = time.clock()
    for i in range(len(train_gens)):
        train_gen = train_gens[i]
        for index in range(train_gen.max_index):
            # run minibatch
            for trainset in train_gen.get_minibatch(index):  # data, mask, label, reset
                data = train_gen.convert_to_vector(trainset[0])
                mask = trainset[1]
                label = trainset[2]
                reset = trainset[3]
                train_batch_loss, train_batch_perplexity = train_func(data, mask, label, reset)
                train_loss.append(train_batch_loss)
                train_perplexity.append(train_batch_perplexity)
                print('..........................', train_batch_loss, train_batch_perplexity)
            if index % 100 == 0 and index != 0:
                current_time = time.clock()
                print('......... minibatch index', index, 'of total index', train_gen.max_index, 'of generator', i)
                print('............ minibatch x 100 loss',  np.mean(np.asarray(train_loss[-100:])))  # only 100, 200... th loss
                print('............ minibatch x 100 perplexity', np.mean(np.asarray(train_perplexity[-100:])))  # only 100, 200... th loss
                print('......... minibatch x 100 time', current_time - start_time)
                start_time = current_time
        
    hist.history['train_loss'].append(np.mean(np.asarray(train_loss)))
    hist.history['train_perplexity'].append(np.mean(np.asarray(train_perplexity)))

def test_validset():
    valid_loss = []
    valid_perplexity = []
    start_time = time.clock()
    for i in range(len(valid_gens)):
        valid_gen = valid_gens[i]
        for index in range(valid_gen.max_index):
            # run minibatch
            for validset in valid_gen.get_minibatch(index):  # data, mask, label, reset
                data = valid_gen.convert_to_vector(validset[0])
                mask = validset[1]
                label = validset[2]
                reset = validset[3]
                valid_batch_loss, valid_batch_perplexity = valid_func(data, mask, label, reset)
                valid_loss.append(valid_batch_loss)
                valid_perplexity.append(valid_batch_perplexity)
            if index % 100 == 0 and index != 0:
                current_time = time.clock()
                print('......... minibatch index', index, 'of total index', valid_gen.max_index, 'of generator', i)
                print('............ minibatch x 100 loss',  np.mean(np.asarray(valid_loss[-100:])))  # only 100, 200... th loss
                print('............ minibatch x 100 perplexity', np.mean(np.asarray(valid_perplexity[-100:])))  # only 100, 200... th loss
                print('......... minibatch x 100 time', current_time - start_time)
                start_time = current_time
        
    hist.history['valid_loss'].append(np.mean(np.asarray(valid_loss)))
    hist.history['valid_perplexity'].append(np.mean(np.asarray(valid_perplexity)))

def test_testset():
    test_loss = []
    test_perplexity = []
    start_time = time.clock()
    for i in range(len(test_gens)):
        test_gen = test_gens[i]
        for index in range(test_gen.max_index):
            # run minibatch
            for testset in test_gen.get_minibatch(index):  # data, mask, label, reset
                data = test_gen.convert_to_vector(testset[0])
                mask = testset[1]
                label = testset[2]
                reset = testset[3]
                test_batch_loss, test_batch_perplexity = test_func(data, mask, label, reset)
                test_loss.append(test_batch_loss)
                test_perplexity.append(test_batch_perplexity)
            if index % 100 == 0 and index != 0:
                current_time = time.clock()
                print('......... minibatch index', index, 'of total index', test_gen.max_index, 'of generator', i)
                print('............ minibatch x 100 loss',  np.mean(np.asarray(test_loss[-100:])))  # only 100, 200... th loss
                print('............ minibatch x 100 perplexity', np.mean(np.asarray(test_perplexity[-100:])))  # only 100, 200... th loss
                print('......... minibatch x 100 time', current_time - start_time)
                start_time = current_time
        
    hist.history['test_loss'].append(np.mean(np.asarray(test_loss)))
    hist.history['test_perplexity'].append(np.mean(np.asarray(test_perplexity)))

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
        # optimizer.reset_params()
    for i in range(len(train_gens)):
        train_gens[i].shuffle()

    print('... Epoch', epoch)
    start_time = time.clock()

    train_trainset()
    test_validset()

    end_time = time.clock()
    print('...... time:', end_time - start_time)

    hist.print_history_of_epoch()
    checker = hist.check_earlystopping()
    if checker == 'save_param':
        hist.save_history_to_csv()
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
hist.print_history_of_epoch(best_epoch, ['train_loss', 'train_perplexity', 'valid_loss', 'valid_perplexity'])
best_loss, best_epoch = hist.best_loss_and_epoch_of_key('test_loss')
hist.print_history_of_epoch(best_epoch, ['test_loss', 'test_perplexity'])
hist.save_history_to_csv()
