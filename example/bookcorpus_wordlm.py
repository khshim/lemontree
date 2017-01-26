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
from lemontree.generators.sequence import GloVeWordLMGenerator
from lemontree.controls.history import HistoryWithEarlyStopping
from lemontree.controls.scheduler import LearningRateMultiplyScheduler
from lemontree.layers.activation import ReLU, Softmax
from lemontree.layers.dense import DenseLayer
from lemontree.layers.lstm import LSTMRecurrentLayer
from lemontree.initializers import GlorotNormal
from lemontree.objectives import CategoricalCrossentropy, CategoricalAccuracy
from lemontree.optimizers import RMSprop
from lemontree.parameters import SimpleParameter
from lemontree.utils.param_utils import filter_params_by_tags, print_tags_in_params
from lemontree.utils.graph_utils import get_inputs_of_variables

np.random.seed(9999)
# base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
base_datapath = 'D:/Dropbox/Project/data/'
# base_datapath = '/home/khshim/data/'
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
overlap_length = 10

cell_init = np.zeros((batch_size,500)).astype(theano.config.floatX)
hidden_init = np.zeros((batch_size,500)).astype(theano.config.floatX)

glove = GloveData(base_datapath, load_pickle=True)
train_gen = GloVeWordLMGenerator([train_data], glove, sequence_length, overlap_length, batch_size, 'train', 337)
test_gen = GloVeWordLMGenerator([test_data], glove, sequence_length, overlap_length, batch_size, 'test', 338)
valid_gen = GloVeWordLMGenerator([valid_data], glove, sequence_length, overlap_length, batch_size, 'valid', 339)

#================Build graph================#

x = T.ftensor3('X')  # (batch_size, sequence_length, 300)
m = T.imatrix('M')  # (batch_size, sequence_length)
y = T.imatrix('y')  # (batch_size, sequence_length)
ci = T.fmatrix('ci')  # (batch_size, 500)
hi = T.fmatrix('hi')  # (batch_size, 500)

# SimpleGraph class is too simple, so we don't use any explicit structure here.

# lstm
lstm = LSTMRecurrentLayer(input_shape=(300,),
                          output_shape=(500,),
                          forget_bias_one=True,
                          output_return_index=None,
                          precompute=False,
                          unroll=False,
                          peephole=False,
                          name='lstm1')

feature_cell, feature_hidden = lstm.get_output(x,m,ci,hi)  # (batch_size, sequence_length, 500)

# dense
dense = DenseLayer((500,), (glove.vocabulary,), target_cpu=False, name='dense1')
inter = dense.get_output(feature_hidden)  # (batch_size, sequence_length, 400003)
inter = T.reshape(inter, (inter.shape[0] * inter.shape[1], inter.shape[2]))  # (batch_size * sequence_length, 400003)
output = Softmax(name='softmax').get_output(inter)  # (batch_size * sequence_length, 400003)

# label and mask
label = T.reshape(y, (y.shape[0] * y.shape[1],))  # (batch_size * sequence_length,)
mask = T.reshape(m, (m.shape[0] * m.shape[1],))  # (batch_size * sequence_length,)

loss = CategoricalCrossentropy(True).get_loss(output, label, mask)
accuracy = CategoricalAccuracy().get_loss(output, label, mask)

graph_params = lstm.get_params() + dense.get_params()

#================Prepare arguments================#

GlorotNormal().initialize_params(filter_params_by_tags(graph_params, ['weight']))
print_tags_in_params(graph_params)

optimizer = RMSprop(0.0001, clipnorm=5.0)
optimizer_updates = optimizer.get_updates(loss, graph_params)
optimizer_params = optimizer.get_params()

total_params = optimizer_params + graph_params
total_updates = optimizer_updates  # no graph updates exist

params_saver = SimpleParameter(total_params, experiment_name + '_params/')
params_saver.save_params()

lr_scheduler = LearningRateMultiplyScheduler(optimizer.lr, 0.2)
hist = HistoryWithEarlyStopping(experiment_name + '_history/', 5, 5)
hist.add_keys(['train_accuracy', 'valid_accuracy', 'test_accuracy'])

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

#================Convenient functions================#

def train_trainset():
    train_loss = []
    train_accuracy = []
    for index in range(train_gen.max_index):
        trainset = train_gen.get_minibatch(index)
        train_batch_loss, train_batch_accuracy = train_func(trainset[0], trainset[1], cell_init, hidden_init, trainset[2])
        train_loss.append(train_batch_loss)
        train_accuracy.append(train_batch_accuracy)
    hist.history['train_loss'].append(np.mean(np.asarray(train_loss)))
    hist.history['train_accuracy'].append(np.mean(np.asarray(train_accuracy)))

def test_validset():
    valid_loss = []
    valid_accuracy = []
    for index in range(valid_gen.max_index):
        validset = valid_gen.get_minibatch(index)
        valid_batch_loss, valid_batch_accuracy = test_func(validset[0], validset[1], cell_init, hidden_init, validset[2])
        valid_loss.append(valid_batch_loss)
        valid_accuracy.append(valid_batch_accuracy)
    hist.history['valid_loss'].append(np.mean(np.asarray(valid_loss)))
    hist.history['valid_accuracy'].append(np.mean(np.asarray(valid_accuracy)))

def test_testset():
    test_loss = []
    test_accuracy = []
    for index in range(test_gen.max_index):
        testset = test_gen.get_minibatch(index)
        test_batch_loss, test_batch_accuracy = test_func(testset[0], testset[1], cell_init, hidden_init, testset[2])
        test_loss.append(test_batch_loss)
        test_accuracy.append(test_batch_accuracy)
    hist.history['test_loss'].append(np.mean(np.asarray(test_loss)))
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
        # optimizer.reset_params()
    train_gen.shuffle()

    print('... Epoch', epoch)
    start_time = time.clock()

    train_trainset()
    test_validset()

    end_time = time.clock()
    print('...... time:', end_time - start_time)

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
best_loss, best_epoch = hist.best_loss_and_epoch_of_key('test_loss')
hist.print_history_of_epoch(best_epoch, ['test_loss', 'test_accuracy'])
hist.save_history_to_csv()