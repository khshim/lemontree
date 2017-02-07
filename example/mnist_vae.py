"""
This code is an example of how to train MNIST Variational Autoencoder (VAE).
VAE includes reparameterization trick, trainer, and generator.
"""

import os
import time
import numpy as np
import theano
import theano.tensor as T
import scipy.misc
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG

from lemontree.data.mnist import MNIST
from lemontree.generators.generator import SimpleGenerator
from lemontree.controls.history import HistoryWithEarlyStopping
from lemontree.controls.scheduler import LearningRateMultiplyScheduler
from lemontree.graphs.graph import SimpleGraph
from lemontree.layers.activation import ReLU, Sigmoid, Tanh
from lemontree.layers.dense import DenseLayer
from lemontree.layers.normalization import BatchNormalization1DLayer
from lemontree.initializers import HeNormal
from lemontree.objectives import BinaryCrossentropy
from lemontree.optimizers import Adam
from lemontree.parameters import SimpleParameter
from lemontree.utils.param_utils import filter_params_by_tags, print_tags_in_params, print_params_num
from lemontree.utils.type_utils import merge_dicts
from lemontree.utils.graph_utils import get_inputs_of_variables
from lemontree.utils.data_utils import split_data

np.random.seed(9999)
base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
# base_datapath = 'D:/Dropbox/Project/data/'
# base_datapath = '/home/khshim/data/'
experiment_name = 'mnist_vae'

#================Prepare data================#

mnist = MNIST(base_datapath, 'flat')
train_data = mnist.train_data
train_data, valid_data = split_data(train_data, 0.9)
test_data = mnist.test_data
train_data = train_data * 2 - 1  # convert to [-1, 1]
train_gen = SimpleGenerator([train_data], 250, 'train')
valid_data = valid_data * 2 - 1  # convert to [-1, 1]
valid_gen = SimpleGenerator([valid_data], 250, 'valid')
test_data = test_data * 2 - 1  # convert to [-1, 1]
test_gen = SimpleGenerator([test_data], 250, 'test')

rng = MRG(9999)

#================Build graph================#

x = T.fmatrix('X')
z = T.fmatrix('Z')

# encoder

enc_dense1 = DenseLayer((784,), (1024,), use_bias=False, name='enc_dense1')
enc_bn1 = BatchNormalization1DLayer((1024,), name='enc_bn1')
enc_dense2 = DenseLayer((1024,), (1024,), use_bias=False,  name='enc_dense2')
enc_bn2 = BatchNormalization1DLayer((1024,), name='enc_bn2')
enc_dense_mv = DenseLayer((1024,), (256,), name='enc_dense_mv')

enc_feature = enc_dense1.get_output(x)
enc_feature = enc_bn1.get_output(enc_feature)
enc_feature= ReLU(0.1).get_output(enc_feature)
enc_feature = enc_dense2.get_output(enc_feature)
enc_feature = enc_bn2.get_output(enc_feature)
enc_feature = ReLU(0.1).get_output(enc_feature)
enc_feature = enc_dense_mv.get_output(enc_feature)

enc_mu = enc_feature[:, :128]
enc_logvar = enc_feature[:, 128:]

latent = enc_mu + T.sqrt(T.exp(enc_logvar)) * rng.normal((250, 128), 0, 1)

# decoder

dec_dense1 = DenseLayer((128,), (1024,), use_bias=False, name='dec_dense1')
dec_bn1 = BatchNormalization1DLayer((1024,), name='dec_bn1')
dec_dense2 = DenseLayer((1024,), (1024,), use_bias=False, name='dec_dense2')
dec_bn2 = BatchNormalization1DLayer((1024,), name='dec_bn2')
dec_dense3 = DenseLayer((1024,), (784,), name='dec_dense3')


dec_output = dec_dense1.get_output(latent)
dec_output = dec_bn1.get_output(dec_output)
dec_output = ReLU(0.1).get_output(dec_output)
dec_output = dec_dense2.get_output(dec_output)
dec_output = dec_bn2.get_output(dec_output)
dec_output = ReLU(0.1).get_output(dec_output)
dec_output = dec_dense3.get_output(dec_output)
dec_output = Tanh().get_output(dec_output)

graph_params = enc_dense1.get_params() + enc_bn1.get_params() + enc_dense2.get_params() + enc_bn2.get_params() + enc_dense_mv.get_params() +\
    dec_dense1.get_params() + dec_bn1.get_params() + dec_dense2.get_params() + dec_bn2.get_params() + dec_dense3.get_params()

latent_loss = 0.5 * T.mean(T.square(enc_mu) + T.exp(enc_logvar) - enc_logvar - 1)
reconstruct_loss = BinaryCrossentropy(True).get_loss((dec_output + 1) / 2, (x + 1) /2)
loss = latent_loss + reconstruct_loss

# generator

gen_output = dec_dense1.get_output(z)
gen_output = dec_bn1.get_output(gen_output)
gen_output = ReLU(0.1).get_output(gen_output)
gen_output = dec_dense2.get_output(gen_output)
gen_output = dec_bn2.get_output(gen_output)
gen_output = ReLU(0.1).get_output(gen_output)
gen_output = dec_dense3.get_output(gen_output)
gen_output = Tanh().get_output(gen_output)
gen_output = (gen_output + 1) / 2

#================Prepare arguments================#

HeNormal().initialize_params(filter_params_by_tags(graph_params, ['weight']))
print_tags_in_params(graph_params)
print_params_num(graph_params)

optimizer = Adam(0.001)
optimizer_updates = optimizer.get_updates(loss, graph_params)
optimizer_params = optimizer.get_params()

total_params = optimizer_params + graph_params
total_updates = optimizer_updates

params_saver = SimpleParameter(total_params, experiment_name + '_params/')
params_saver.save_params()

lr_scheduler = LearningRateMultiplyScheduler(optimizer.lr, 0.1)
hist = HistoryWithEarlyStopping(experiment_name + '_history/', 5, 5)
hist.add_keys(['train_latent_loss', 'train_reconstruct_loss'])
hist.add_keys(['valid_latent_loss', 'valid_reconstruct_loss'])
hist.add_keys(['test_latent_loss', 'test_reconstruct_loss'])

#================Compile functions================#

#test_func = theano.function(inputs=[x],
#                            outputs=[enc_mu, enc_var, latent],
#                            allow_input_downcast=True)
#test_output = test_func(train_gen.get_minibatch(0)[0])
#print(test_output[0])
#print(test_output[1])
#print(test_output[2])

train_func = theano.function(inputs=[x],
                             outputs=[loss, latent_loss, reconstruct_loss],
                             updates=total_updates,
                             allow_input_downcast=True)

test_func = theano.function(inputs=[x],
                            outputs=[loss, latent_loss, reconstruct_loss],
                            allow_input_downcast=True)

gen_func = theano.function(inputs=[z],
                           outputs=gen_output,
                           allow_input_downcast=True)

#================Convenient functions================#

def train_trainset():
    train_loss = []
    latent_loss = []
    reconstruct_loss = []
    for index in range(train_gen.max_index):
        trainset = train_gen.get_minibatch(index)
        train_batch_loss = train_func(trainset[0])
        train_loss.append(train_batch_loss[0])
        latent_loss.append(train_batch_loss[1])
        reconstruct_loss.append(train_batch_loss[2])
    hist.history['train_loss'].append(np.mean(np.asarray(train_loss)))
    hist.history['train_latent_loss'].append(np.mean(np.asarray(latent_loss)))
    hist.history['train_reconstruct_loss'].append(np.mean(np.asarray(reconstruct_loss)))

def test_validset():
    valid_loss = []
    valid_latent_loss = []
    valid_reconstruct_loss = []
    for index in range(valid_gen.max_index):
        validset = valid_gen.get_minibatch(index)
        valid_batch_loss = test_func(validset[0])
        valid_loss.append(valid_batch_loss[0])
        valid_latent_loss.append(valid_batch_loss[1])
        valid_reconstruct_loss.append(valid_batch_loss[2])
    hist.history['valid_loss'].append(np.mean(np.asarray(valid_loss)))
    hist.history['valid_latent_loss'].append(np.mean(np.asarray(valid_latent_loss)))
    hist.history['valid_reconstruct_loss'].append(np.mean(np.asarray(valid_reconstruct_loss)))

def test_testset():
    test_loss = []
    test_latent_loss = []
    test_reconstruct_loss = []
    for index in range(test_gen.max_index):
        testset = test_gen.get_minibatch(index)
        test_batch_loss = test_func(testset[0])
        test_loss.append(test_batch_loss[0])
        testlatent_loss.append(test_batch_loss[1])
        test_reconstruct_loss.append(test_batch_loss[2])
    hist.history['test_loss'].append(np.mean(np.asarray(test_loss)))
    hist.history['test_latent_loss'].append(np.mean(np.asarray(test_latent_loss)))
    hist.history['test_reconstruct_loss'].append(np.mean(np.asarray(test_reconstruct_loss)))

result_folder = experiment_name + '_result/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

def generate(epoch):
    random_z = np.random.normal(0, 1, (64, 128))
    generated = gen_func(random_z)
    manifold = np.zeros((28*8, 28*8), dtype=theano.config.floatX)
    for indx in range(8):
        for indy in range(8):
            current_img = np.reshape(generated[indx * 8 + indy], (28,28))
            manifold[indx * 28:(indx+1) * 28, indy * 28:(indy+1) * 28] = current_img
    manifold = np.asarray(manifold * 255, dtype='int32')
    manifold = scipy.misc.toimage(manifold, cmin=0, cmax=255)
    scipy.misc.imsave(result_folder + str(epoch) + '.png', manifold)

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

    print('...Epoch', epoch)
    start_time = time.clock()

    train_trainset()
    test_validset()
    generate(epoch)

    end_time = time.clock()
    print('......time:', end_time - start_time)

    hist.print_history_of_epoch()
    checker = hist.check_earlystopping('valid_loss')
    if checker == 'save_param':
        time.sleep(1)
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
hist.print_history_of_epoch(best_epoch, ['train_loss', 'valid_loss'])
hist.print_history_of_epoch(-1, ['test_loss'])
hist.save_history_to_csv()