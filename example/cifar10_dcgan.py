import os
import time
import numpy as np
import theano
import theano.tensor as T
import scipy.misc
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG

from lemontree.data.cifar10 import CIFAR10
from lemontree.graphs.graph import SimpleGraph
from lemontree.generators.generator import SimpleGenerator
from lemontree.controls.history import SimpleHistory
from lemontree.graphs.graph import SimpleGraph
from lemontree.layers.activation import ReLU, Sigmoid, Tanh, Softmax
from lemontree.layers.dense import DenseLayer
from lemontree.layers.convolution import Convolution3DLayer, TransposedConvolution3DLayer
from lemontree.layers.pool import Upscaling3DLayer
from lemontree.layers.normalization import BatchNormalization3DLayer, BatchNormalization1DLayer
from lemontree.layers.dropout import DropoutLayer
from lemontree.layers.shape import ReshapeLayer, Flatten3DLayer
from lemontree.initializers import GlorotNormal, Normal
from lemontree.objectives import BinaryCrossentropy, BinaryAccuracy, CategoricalCrossentropy, CategoricalAccuracy
from lemontree.optimizers import Adam
from lemontree.parameters import SimpleParameter
from lemontree.utils.param_utils import filter_params_by_tags, print_tags_in_params, print_params_num
from lemontree.utils.type_utils import merge_dicts
from lemontree.utils.graph_utils import get_inputs_of_variables
from lemontree.utils.data_utils import split_data

np.random.seed(9999)
# base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
# base_datapath = 'D:/Dropbox/Project/data/'
base_datapath = '/home/khshim/data/'

batch_size = 100
experiment_name = 'cifar10_dcgan'

#================Prepare Data================#

cifar10 = CIFAR10(base_datapath, 'tensor')
train_data = cifar10.train_data
train_data = train_data * 2 - 1
train_gen = SimpleGenerator([train_data], batch_size, 'train')

#================Build Graph================#

z = T.fmatrix('Z')  # (batch_size, 100)
x = T.ftensor4('X')  # (batch_size, 3, 28, 28)

# You can use either upscaling + conv / transposed conv

generator = SimpleGraph(experiment_name + '_gen', batch_size)
generator.add_layer(DenseLayer((100,), (8192,), use_bias=False), is_start=True)
generator.add_layer(BatchNormalization1DLayer((8192,), 0.9))
generator.add_layer(ReLU())
generator.add_layer(ReshapeLayer((8192,), (512, 4, 4)))
#generator.add_layer(Upscaling3DLayer((512,4,4), (512,8,8), (2,2)))
#generator.add_layer(Convolution3DLayer((512,8,8), (256,8,8), (3,3), 'half', use_bias=False))
generator.add_layer(TransposedConvolution3DLayer((512,4,4), (256,8,8), (3,3), 'half', (2,2), use_bias=False))
generator.add_layer(BatchNormalization3DLayer((256,8,8), 0.9))
generator.add_layer(ReLU())
#generator.add_layer(Upscaling3DLayer((256,8,8), (256,16,16), (2,2)))
#generator.add_layer(Convolution3DLayer((256,16,16), (128,16,16), (3,3), 'half', use_bias=False))
generator.add_layer(TransposedConvolution3DLayer((256,8,8), (128,16,16), (3,3), 'half', (2,2), use_bias=False))
generator.add_layer(BatchNormalization3DLayer((128,16,16), 0.9))
generator.add_layer(ReLU())
#generator.add_layer(Upscaling3DLayer((128,16,16), (128,32,32), (2,2)))
#generator.add_layer(Convolution3DLayer((128,32,32), (3,32,32), (3,3), 'half')
generator.add_layer(TransposedConvolution3DLayer((128,16,16), (3,32,32), (3,3), 'half', (2,2)))
generator.add_layer(Tanh())

gen_params = generator.get_params()
gen_updates = generator.get_updates()

discriminator = SimpleGraph(experiment_name + '_disc', batch_size)
discriminator.add_layer(Convolution3DLayer((3,32,32), (128,16,16), (3,3), 'half', (2,2)), is_start=True)
discriminator.add_layer(ReLU(0.2))
discriminator.add_layer(Convolution3DLayer((128,16,16), (256,8,8), (3,3), 'half', (2,2), use_bias=False))
discriminator.add_layer(BatchNormalization3DLayer((256,8,8), 0.9))
discriminator.add_layer(ReLU(0.2))
discriminator.add_layer(Convolution3DLayer((256,8,8), (512,4,4), (3,3), 'half', (2,2), use_bias=False))
discriminator.add_layer(BatchNormalization3DLayer((512,4,4), 0.9))
discriminator.add_layer(ReLU(0.2))
discriminator.add_layer(Flatten3DLayer((512,4,4), (8192,)))
discriminator.add_layer(DenseLayer((8192,), (1,)))
discriminator.add_layer(Sigmoid())

disc_params = discriminator.get_params()
disc_updates = discriminator.get_updates()

#================Make Loss================#

zx, _ = generator.get_output({0:[z]}, -1, 0)

d_x, _ = discriminator.get_output({0:[x]}, -1, 0)
d_zx, _ = discriminator.get_output({0:[zx]}, -1, 0)

disc_loss_real = BinaryCrossentropy(True).get_output(d_x, 1)
disc_loss_fake = BinaryCrossentropy(True).get_output(d_zx, 0)
disc_loss = disc_loss_real + disc_loss_fake
gen_loss = BinaryCrossentropy(True).get_output(d_zx, 1)


#================Initialize================#

# GlorotNormal().initialize_params(filter_params_by_tags(gen_params, ['weight']))
# GlorotNormal().initialize_params(filter_params_by_tags(disc_params, ['weight']))
Normal(0, 0.02).initialize_params(filter_params_by_tags(gen_params, ['weight']))
Normal(0, 0.02).initialize_params(filter_params_by_tags(disc_params, ['weight']))

hist = SimpleHistory(experiment_name + '_history/')
hist.add_keys(['disc_loss', 'gen_loss', 'disc_loss_fake', 'disc_loss_real'])

#================Make Optimizer================#

disc_opt = Adam(0.0002, 0.5, 0.9)
gen_opt = Adam(0.0002, 0.5, 0.9)

disc_opt_updates = disc_opt.get_updates(disc_loss, disc_params)
disc_opt_params = disc_opt.get_params()
gen_opt_updates = gen_opt.get_updates(gen_loss, gen_params)
gen_opt_params = gen_opt.get_params()

#total_params = disc_params + gen_params + disc_opt_params + gen_opt_params
#params_saver = SimpleParameter(total_params, experiment_name + '_params/')
#params_saver.save_params()

#================Compile Functions================#

disc_func = theano.function(inputs=[x,z],
                            outputs=[gen_loss, disc_loss, disc_loss_real, disc_loss_fake],
                            updates=disc_opt_updates,
                            allow_input_downcast=True)

gen_func = theano.function(inputs=[x,z],
                           outputs=[gen_loss, disc_loss, disc_loss_real, disc_loss_fake],
                           updates=gen_opt_updates,
                           allow_input_downcast=True)

get_image = theano.function(inputs=[z],
                            outputs=zx,
                            allow_input_downcast=True)

#================Convenient Functions================#


def pretrain():
    dl = []
    dlr = []
    dlf = []
    for index in range(train_gen.max_index):
        data = train_gen.get_minibatch(index)[0]
        latent = np.random.uniform(-1, 1, (batch_size, 100))
        func_outputs = disc_func(data, latent, y_disc_np)
        dl.append(func_outputs[1])
        dlr.append(func_outputs[2])
        dlf.append(func_outputs[3])
    print('Pretrain disc_loss', np.mean(np.asarray(dl)))
    print('Pretrain disc_loss_real', np.mean(np.asarray(dlr)))
    print('Pretrain disc_loss_fake', np.mean(np.asarray(dlf)))

def train():
    dl = []
    gl = []
    dlr = []
    dlf = []
    for index in range(0, train_gen.max_index-1, 2):
        data = train_gen.get_minibatch(index)[0]
        latent = np.random.uniform(-1, 1, (batch_size, 100))
        func_outputs = gen_func(data, latent)
        dl.append(func_outputs[1])
        gl.append(func_outputs[0])
        dlr.append(func_outputs[2])
        dlf.append(func_outputs[3])

        data = train_gen.get_minibatch(index+1)[0]
        latent = np.random.uniform(-1, 1, (batch_size, 100))
        func_outputs = disc_func(data, latent)
        dl.append(func_outputs[1])
        gl.append(func_outputs[0])
        dlr.append(func_outputs[2])
        dlf.append(func_outputs[3])
        
    hist.history['disc_loss'].append(np.mean(np.asarray(dl)))
    hist.history['gen_loss'].append(np.mean(np.asarray(gl)))
    hist.history['disc_loss_real'].append(np.mean(np.asarray(dlr)))
    hist.history['disc_loss_fake'].append(np.mean(np.asarray(dlf)))

result_folder = experiment_name + '_result/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

def generate(epoch):
    latent = np.random.uniform(-1, 1, (batch_size, 100))
    generated = (get_image(latent) + 1) / 2
    manifold = np.zeros((32*8, 32*8, 3), dtype=theano.config.floatX)
    for indx in range(8):
        for indy in range(8):
            current_img = np.swapaxes(generated[indx * 8 + indy], 0, 2)
            current_img = np.swapaxes(current_img, 0, 1)
            manifold[indx * 32:(indx+1) * 32, indy * 32:(indy+1) * 32, :] = current_img
    manifold = np.asarray(manifold * 255, dtype='int32')
    manifold = scipy.misc.toimage(manifold, cmin=0, cmax=255)
    scipy.misc.imsave(result_folder + str(epoch) + '.png', manifold)


#================Train================#

# pretrain()

for epoch in range(200):

    train_gen.shuffle()

    print('... Epoch', epoch)
    start_time = time.clock()

    train()
    if epoch % 2 == 0:
        generate(epoch)

    end_time = time.clock()
    print('...... time:', end_time - start_time)

    hist.print_history_of_epoch()
    if epoch % 10 == 0:
        hist.save_history_to_csv()

#================Test================#