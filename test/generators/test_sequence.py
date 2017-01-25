"""
This code is test code.

File: lemontree.generators.sequence
Target: GloVeWordLMGenerator
"""

import numpy as np
from lemontree.data.glove import GloveData
from lemontree.generators.sequence import GloVeWordLMGenerator

np.random.seed(9999)
# base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
base_datapath = 'D:/Dropbox/Project/data/'
# base_datapath = '/home/khshim/data/'


glove = GloveData(base_datapath, load_pickle=True)
sentences = [[1,2,3,4],
             [2,3,4,5,6],
             [6,7,8,9,10,11,12,13],
             [14,15,16],
             [17,18],
             [19,20,21,22,23],
             [24,25,26,27,28,29],
             [30,31,32]]

generator = GloVeWordLMGenerator([sentences], glove,
                                 sequence_length=5,
                                 overlap_length=3,
                                 batch_size=4,
                                 name='test')

for epoch in range(3):
    generator.shuffle()
    for i in range(generator.max_index):
        print('Epoch', epoch, i, 'th')
        data, mask, label, reset = generator.get_minibatch(i)
        print('... data', data.shape)
        #print(data)
        print('... mask', mask.shape)
        print(mask)
        print('...label', label.shape)
        print(label)
        print('... reset', reset.shape)
        print(reset)