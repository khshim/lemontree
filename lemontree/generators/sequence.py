"""
This code includes sequence generators to make minibatch and shuffle.
Generator make data to feed into training function for every mini-batch.
Each generator can hold multiple data, such as data and mask.
Sequence Generator preprocess sequence before training and during training.
"""

import numpy as np
from lemontree.generators.generator import SimpleGenerator


class WordLMGenerator(SimpleGenerator):
    """
    This class get sequence data 3D tensor and generate mini-batch of words.
    Using GloVe word vector to make word embedding.
    Words are saved as integer.
    Words are converted to word vector while generating each mini-batch.
    Assume <EOS> and/or <SOS> are added while reading data.
    """
    def __init__(self, data_list, embedding, sequence_length=20, stride_length=1, data_length=None,
                 batch_size=128, name=None, seed=333):
        """
        This function initializes the class.
        Assume data is preprocessed and already padded.

        Parameters
        ----------
        data_list: list
            a list of data, each include equal number of data.
        embedding: GloveData, ...
            an embedding class.
        sequence_length: integer, default: 20
            an integer which indicates each length of the sentence.
            in otherwords, the number of timestep RNN gets each time.
        stride_length: integer, default: 1
            an integer which indicates the index of next mini-batch starts.
            if None, sequence_length == overlap length.
            i.e., next batch starts from next word.
        data_length: integer, default: None
            a length of each data.
        name: string
            a string name of the class.
        seed: int
            an integer value of numpy random generator.
        batch_size: int
            an integer value, which is the number of data in mini-batch.     

        Returns
        -------
        None.
        """
        super(WordLMGenerator, self).__init__(data_list, batch_size, name, seed)
        # check asserts
        assert isinstance(sequence_length, int), '"sequence_length" should be a positive integer.'
        if stride_length is None:
            stride_length = sequence_length
        if data_length is None:
            data_length = sequence_length
        else:
            assert isinstance(stride_length, int), '"stride_length" should be a positive integer.'
            assert stride_length <= sequence_length, '"stride_length" should be smaller than "sequence_length".'
            assert data_length >= sequence_length, '"data_length" should be larger than "sequence_length".'

        # set members
        self.embedding = embedding
        self.sequence_length = sequence_length
        self.stride_length = stride_length
        self.data_length = data_length
        if data_length == sequence_length:
            self.slice_per_batch = 1
        else:
            self.slice_per_batch = (self.data_length - self.sequence_length) // self.stride_length + 1  # label is 1 shorter than data
    

    def get_minibatch(self, index):
        """
        This function overrides parents' ones.
        Generates the mini batch data.
        Convert word indices to word embedding vector.

        Parameters
        ----------
        index: int
            an integer value that indicates which mini batch will be returned.

        Returns
        -------
        tuple
            a tuple of partial data in data list.
            (3D data, 2D mask, 2D label, 1D reset)
            if reset = 0, next batch cell/hidden initialization will be 0.
            if reset = 1 , next batch cell/hidden initialization
                next specific batch index will be continued data.
        """
        # check asserts
        assert index < self.max_index

        # prepare minibatch

        # (batch_size, data_length)
        minibatch_data = [self.data_list[0][i] for i in self.order[self.batch_size * index: self.batch_size * (index+1)]]
        minibatch_mask = [self.data_list[1][i] for i in self.order[self.batch_size * index: self.batch_size * (index+1)]]

        minibatch_data = np.asarray(minibatch_data, dtype='int32')
        minibatch_mask = np.asarray(minibatch_mask, dtype='int32')
        minibatch_mask = np.hstack([minibatch_mask[:, 1:], np.zeros((self.batch_size, 1))])  # same as label
        minibatch_label = np.hstack([minibatch_data[:, 1:], np.zeros((self.batch_size, 1))])

        for j in range(self.slice_per_batch):
            minibatch_data_slices = minibatch_data[:, j * self.stride_length: j * self.stride_length + self.sequence_length]
            minibatch_mask_slices = minibatch_mask[:, j * self.stride_length: j * self.stride_length + self.sequence_length]
            minibatch_label_slices = minibatch_label[:, j * self.stride_length: j * self.stride_length + self.sequence_length]
            if j == 0:
                minibatch_reset = np.zeros((self.batch_size,)).astype('int32')
            else:
                minibatch_reset = np.ones((self.batch_size,)).astype('int32')

            yield (minibatch_data_slices, minibatch_mask_slices, minibatch_label_slices, minibatch_reset)

    def convert_to_vector(self, data):
        new_data = []
        for i in range(self.batch_size):
            new_data.append(self.embedding.indices_to_vec(data[i]))
        return np.asarray(new_data, dtype='float32')

if __name__ == '__main__':
    worddata = [[1,2,3,4], [5,6,7,8,9], [10,11,12,13,14,15,16],[17,18,19,20]]
    from lemontree.utils.data_utils import sentence_padding
    worddata, wordmask = sentence_padding(worddata, 6)
    print('Data padded', worddata)
    print('Mask padded', wordmask)    

    from lemontree.data.glove import GloveData
    glove = GloveData('C:/Users/skhu2/Dropbox/Project/data/')

    gen = WordLMGenerator([worddata, wordmask], glove, 4, 1, 6, 2, 'gen')

    gen.shuffle()
    for i in range(gen.max_index):
        for dataset in gen.get_minibatch(i):
            converted = gen.convert_to_vector(dataset[0])
            print('...Data', dataset[0])
            print('...Data converted', converted.shape, converted)
            print('...Mask', dataset[1])
            print('...Label', dataset[2])
            print('...Reset', dataset[3])