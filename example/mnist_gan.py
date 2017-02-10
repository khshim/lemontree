"""
This code is an example of how to train MNIST classification.
The most easiest deep learning example.
Good to learn how to use LemonTree.
"""

import os
import time
import numpy as np
import theano
import theano.tensor as T
import scipy.misc

from lemontree.data.mnist import MNIST
from lemontree.generators.generator import SimpleGenerator
from lemontree.controls.history import HistoryWithEarlyStopping
from lemontree.controls.scheduler import LearningRateMultiplyScheduler
from lemontree.graphs.graph import SimpleGraph
from lemontree.layers.activation import ReLU, Softmax, Sigmoid
from lemontree.layers.dense import DenseLayer
from lemontree.layers.dropout import DropoutLayer
from lemontree.layers.convolution import Convolution3DLayer
from lemontree.layers.pool import Upscaling3DLayer
from lemontree.layers.shape import Flatten3DLayer, ReshapeLayer
from lemontree.layers.normalization import BatchNormalization1DLayer, BatchNormalization3DLayer
from lemontree.initializers import HeNormal
from lemontree.objectives import BinaryCrossentropy, CategoricalCrossentropy, CategoricalAccuracy
from lemontree.optimizers import Adam
from lemontree.parameters import SimpleParameter
from lemontree.utils.param_utils import filter_params_by_tags, print_tags_in_params, print_params_num
from lemontree.utils.type_utils import merge_dicts
from lemontree.utils.graph_utils import get_inputs_of_variables
from lemontree.utils.data_utils import int_to_onehot

np.random.seed(9999)
# base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
base_datapath = 'D:/Dropbox/Project/data/'
# base_datapath = '/home/khshim/data/'
experiment_name = 'mnist_mlp'

#================Prepare data================#

batch_size = 100

mnist = MNIST(base_datapath, 'tensor')
train_data, train_label = mnist.get_fullbatch_train()
train_gen = SimpleGenerator([train_data], batch_size, 'train')

#================Build graph================#

x = T.ftensor4('X')  # (batch_size, 784)
z = T.fmatrix('Z')  # (batch_size, 100)
y_one = T.ivector('Y1')  # (batch_size,)
y_zero = T.ivector('Y0')  # (batch_size,)

generator = SimpleGraph(experiment_name + '_gen', batch_size)
generator.add_layer(DenseLayer((100,),(39200,), use_bias=False), is_start=True)                    # 0
generator.add_layer(BatchNormalization1DLayer((39200,)))                                           # 1
generator.add_layer(ReLU())                                                                         # 2
generator.add_layer(ReshapeLayer((39200,), (200,14,14)))                                           # 3
generator.add_layer(Upscaling3DLayer((200,14,14), (200,28,28), (2,2)))                              # 4
generator.add_layer(Convolution3DLayer((200,28,28), (100,28,28), (3,3), 'half', use_bias=False))    # 5
generator.add_layer(BatchNormalization3DLayer((100,28,28)))                                         # 6
generator.add_layer(ReLU())                                                                         # 7
generator.add_layer(Convolution3DLayer((100,28,28), (50,28,28), (3,3), 'half', use_bias=False))     # 8
generator.add_layer(BatchNormalization3DLayer((50,28,28)))                                          # 9
generator.add_layer(ReLU())                                                                         # 10
generator.add_layer(Convolution3DLayer((50,28,28), (1,28,28), (1,1), 'valid', use_bias=False))      # 11                    
generator.add_layer(Sigmoid())                                                                      # 12

fake, fake_layers = generator.get_output({0:[z]}, -1, 0)

discriminator = SimpleGraph(experiment_name + '_disc', batch_size)
discriminator.add_layer(Convolution3DLayer((1,28,28), (200,14,14), (5,5), 'half', (2,2)), is_start=True)
discriminator.add_layer(ReLU(0.2))
discriminator.add_layer(DropoutLayer(0.25))
discriminator.add_layer(Convolution3DLayer((200,14,14), (500,7,7), (5,5), 'half', (2,2)))
discriminator.add_layer(ReLU(0.2))
discriminator.add_layer(DropoutLayer(0.25))
discriminator.add_layer(Flatten3DLayer((500,7,7), (24500,)))
discriminator.add_layer(DenseLayer((24500,), (256,)))
discriminator.add_layer(ReLU(0.2))
discriminator.add_layer(DropoutLayer(0.25))
discriminator.add_layer(DenseLayer((256,), (2,)))
discriminator.add_layer(Softmax())

fake_disc, fake_disc_layers = discriminator.get_output({0:[fake]}, -1, 0)
real_disc, real_disc_layers = discriminator.get_output({0:[x]}, -1, 0)

disc_loss_real = CategoricalCrossentropy(True).get_output(real_disc, y_one)
disc_acc_real = CategoricalAccuracy().get_output(real_disc, y_one)
disc_loss_fake = CategoricalCrossentropy(True).get_output(fake_disc, y_zero)
disc_acc_fake = CategoricalAccuracy().get_output(fake_disc, y_zero)
disc_loss = disc_loss_fake + disc_loss_real
gen_loss = CategoricalCrossentropy(True).get_output(fake_disc, y_one)
gen_acc = CategoricalAccuracy().get_output(fake_dsic, y_one)

generator_params = generator.get_params()
discriminator_params = discriminator.get_params()

#================Prepare arguments================#

HeNormal().initialize_params(filter_params_by_tags(generator_params, ['weight']))
print_params_num(generator_params)
HeNormal().initialize_params(filter_params_by_tags(discriminator_params, ['weight']))
print_params_num(discriminator_params)

gen_optimizer = Adam(0.0001)
gen_optimizer_updates = gen_optimizer.get_updates(gen_loss, generator_params)
gen_optimizer_params = gen_optimizer.get_params()

disc_optimizer = Adam(0.001)
disc_optimizer_updates = disc_optimizer.get_updates(disc_loss, discriminator_params)
disc_optimizer_params = disc_optimizer.get_params()

total_params = generator_params + discriminator_params + gen_optimizer_params + disc_optimizer_params

params_saver = SimpleParameter(total_params, experiment_name + '_params/')
params_saver.save_params()

#lr_scheduler = LearningRateMultiplyScheduler(optimizer.lr, 0.1)
hist = HistoryWithEarlyStopping(experiment_name + '_history/', 5, 5)
hist.add_keys(['gen_loss', 'disc_loss', 'disc_loss_real', 'disc_loss_fake'])
hist.add_keys(['gen_acc', 'disc_acc_real', 'disc_acc_fake'])

#================Compile functions================#

graph_outputs = [gen_loss, disc_loss, disc_loss_real, disc_loss_fake]
graph_inputs = [x,z]

gen_func = theano.function(inputs=[z, y_one],
                           outputs=[gen_loss, gen_acc],
                           updates=gen_optimizer_updates,
                           allow_input_downcast=True)

disc_func = theano.function(inputs=[x,z, y_zero, y_one],
                            outputs=[disc_loss, disc_loss_real, disc_loss_fake, disc_acc_real, disc_acc_fake],
                            updates=disc_optimizer_updates,
                            allow_input_downcast=True)

fake_image = theano.function(inputs=[z],
                             outputs=fake,
                             allow_input_downcast=True)


#================Convenient functions================#

y_one_np = np.ones((batch_size,))
y_zero_np = np.zeros((batch_size,))

def pretrain_disc():
    disc_loss = []
    disc_loss_real = []
    disc_loss_fake = []
    disc_acc_real = []
    disc_acc_fake = []
    for index in range(train_gen.max_index):
        trainset = train_gen.get_minibatch(index)
        data = trainset[0]
        latent = np.random.normal(0, 1, size=(batch_size, 100))
        train_outputs = disc_func(data, latent, y_zero_np, y_one_np)
        disc_loss.append(train_outputs[0])
        disc_loss_real.append(train_outputs[1])
        disc_loss_fake.append(train_outputs[2])
        disc_acc_real.append(train_outputs[3])
        disc_acc_fake.append(train_outputs[4])
    print('Pretrained disc_loss', np.mean(np.asarray(disc_loss)))
    print('Pretrained disc_loss_real', np.mean(np.asarray(disc_loss_real)))
    print('Pretrained disc_loss_fake', np.mean(np.asarray(disc_loss_fake)))
    print('Pretrained disc_acc_real', np.mean(np.asarray(disc_acc_real)))
    print('Pretrained disc_acc_fake', np.mean(np.asarray(disc_acc_fake)))

def train_trainset():
    gen_loss = []
    gen_acc = []
    disc_loss = []
    disc_loss_real = []
    disc_loss_fake = []
    disc_acc_real = []
    disc_acc_fake = []
    for index in range(train_gen.max_index):
        trainset = train_gen.get_minibatch(index)
        data = trainset[0]
        latent1 = np.random.normal(0, 1, size=(batch_size, 100))
        latent2 = np.random.normal(0, 1, size=(batch_size, 100))

        train1_outputs = disc_func(data, latent1, y_zero_np, y_one_np)
        train2_outputs = gen_func(latent2, y_one_np)

        gen_loss.append(train2_outputs[0])
        gen_acc.append(train2_outputs[1])
        disc_loss.append(train1_outputs[0])
        disc_loss_real.append(train1_outputs[1])
        disc_loss_fake.append(train1_outputs[2])
        disc_acc_real.append(train1_outputs[3])
        disc_acc_fake.append(train1_outputs[4])
    hist.history['gen_loss'].append(np.mean(np.asarray(gen_loss)))
    hist.history['gen_acc'].append(np.mean(np.asarray(gen_acc)))
    hist.history['disc_loss'].append(np.mean(np.asarray(disc_loss)))
    hist.history['disc_loss_real'].append(np.mean(np.asarray(disc_loss_real)))
    hist.history['disc_loss_fake'].append(np.mean(np.asarray(disc_loss_fake)))
    hist.history['disc_acc_real'].append(np.mean(np.asarray(disc_acc_real)))
    hist.history['disc_acc_fake'].append(np.mean(np.asarray(disc_acc_fake)))

result_folder = experiment_name + '_result/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

def generate_fake(epoch):
    page = np.zeros((10 * 28, 10 * 28)).astype('float32')
    latent = np.random.normal(0, 1, size=(batch_size, 100))
    fake = fake_image(latent)  # (100, 1, 28, 28)
    for a in range(10):
        for b in range(10):
            index = a * 10 + b
            image = np.reshape(fake[index], (28,28))
            page[a * 28:(a+1) * 28, b * 28:(b+1) * 28] = image
    page = np.asarray(page * 255, dtype='int32')
    page = scipy.misc.toimage(page, cmin=0, cmax=255)
    scipy.misc.imsave(result_folder + str(epoch) + '.png', page)

#================Train================#

change_lr = False
end_train = False

print('Pretrain start')
pretrain_disc()

for epoch in range(1000):
    #if end_train:
    #    params_saver.load_params()
    #    break
    #if change_lr:
    #    params_saver.load_params()
    #    lr_scheduler.change_learningrate(epoch)
    #    # optimizer.reset_params()
    train_gen.shuffle()

    print('...Epoch', epoch)
    start_time = time.clock()

    train_trainset()

    end_time = time.clock()
    print('......time:', end_time - start_time)

    hist.print_history_of_epoch()

    if epoch % 10 == 0:
        hist.save_history_to_csv()
        generate_fake(epoch)
    #checker = hist.check_earlystopping()
    #if checker == 'save_param':
    #    params_saver.save_params()
    #    change_lr = False
    #    end_train = False
    #elif checker == 'change_lr':
    #    change_lr = True
    #    end_train = False
    #elif checker == 'end_train':
    #    change_lr = False
    #    end_train = True
    #elif checker == 'keep_train':
    #    change_lr = False
    #    end_train = False
    #else:
    #    raise NotImplementedError('Not supported checker type')

