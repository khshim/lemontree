"""
This code includes functions for reading Glove word embeddings and search.

Download
--------
Glove: Global vectors for word representation
http://nlp.stanford.edu/projects/glove/
Download 'Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors)'.
Unzip the file and let them as txt file.

Base datapath: '~~~/data/'
Additional folder structure: '~~~/data/glove/glove.6B.300d.txt'
"""

import time
import numpy as np


class GloveData(object):
    """
    This class use glove word vector set to make word embeddings.
    """
    def __init__(self, base_datapath, mode='6B.300d', seed=33):
        """
        This function initializes the class.
        Initialization contains loading and parsing of glove data.
        Either two way is possible.
        One is to save glove as dictonary type in memory.
        The other is to save glove separately, label and index to dictionary an data in ndarray.
        We choose second option to get (maybe slightly) better cache and memory usage.
        
        Parameters
        ----------
        base_datapath: string
            a string path where glove vector text is saved.
        mode: string, default: '6B.300d'
            a string which will be target glove vector.
            i.e., 'glove.6B.300d.txt'.
        seed: int
            an integer to randomly generate eos, sos, unk tokens.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(base_datapath, str), '"base_datapath" should be a string path.'
        assert isinstance(mode, str), '"mode" should be a string name for glove word vector.'

        # load
        glove_file = base_datapath + 'glove/glove.' + mode + '.txt'
        print('Glove load file:', glove_file)
        self.dict = {}
        self.embedding = []
        index = 0
        start_time = time.clock()
        with open(glove_file, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                word_and_vector = line.replace('\n', '').split(' ')
                word = word_and_vector[0]
                vector = word_and_vector[1:]
                self.dict[word] = index
                self.embedding.append(vector)
                index += 1
        end_time= time.clock()
        self.vocabulary = len(self.embedding)
        self.dimension = len(self.embedding[0])

        # set <eos>, <sos>, <unk> vector.
        rng = np.random.RandomState(seed)
        eos_vector = rng.uniform(-1, 1, (self.dimension,))
        sos_vector = rng.uniform(-1, 1, (self.dimension,))
        unk_vector = rng.uniform(-1, 1, (self.dimension,))
        self.dict['<EOS>'] = index
        self.dict['<SOS>'] = index + 1
        self.dict['<UNK>'] = index + 2
        self.embedding.append(eos_vector)
        self.embedding.append(sos_vector)
        self.embedding.append(unk_vector)

        # convert to numpy
        self.embedding = np.asarray(self.embedding, dtype='float32')  # move list to a single numpy array.
        print('Glove load time:', end_time - start_time)
        print('Glove data shape:', self.embedding.shape)     
        print('Glove total vocabulary:', self.vocabulary)

    def words_to_indices(self, words):
        """
        This function returns words to a list of given vocabulary.
        If a word does not exist, return <unk> vector.

        Parameters
        ----------
        words: list
            a list of word to convert.
            order is preserved.

        Return
        ------
        list
            a list of word indices, which indicates index of dictionary.
        """
        # check asserts
        assert isinstance(words, list), '"words" should be a list of word strings.'

        # get keys and return list
        words_indices = []
        index = 0
        for ww in words:  # preserve order
            if ww not in self.dict.keys():
                key = self.vocabulary + 2  # <unk>
            else:
                key = self.dict[ww]  # find index in dictionary
            words_indices.append(key)
            index += 1
        assert index == len(words)  # all words are included
        return words_indices
        
    def indices_to_vec(self, indices):
        """
        This function returns word embedding of given word index.
        Highly recommand to do "words_to_indices" first, to remove OOV words to <unk>.

        Parameters
        ----------
        indices: list
            a list of integers under vocabulary size.
            order is preserved.

        Return
        ------
        ndarray
            a matrix of shape (words, vector dimension)
        """
        # check asserts
        assert isinstance(indices, list), '"indices" should be a list of integers.'

        # get keys and make matrix
        words_embedding = np.zeros((len(indices), self.dimension)).astype('float32')
        index = 0
        for ii in indices:
            words_embedding[index, :] = self.embedding[ii, :]
            index += 1
        assert index == len(indices)  # all words are included
        return words_embedding

    def words_to_vec(self, words):
        """
        This function converts words into glove word vectors.
        Convenient function, which do only two function merging.

        Parameters
        ----------
        words: list
            a list of word to convert.
            order is preserved.

        Return
        ------
        ndarray
            a matrix of shape (words, vector dimension)
        """
        indices = self.words_to_indices(words)
        return self.indices_to_vec(indices)        


if __name__ == '__main__':
    base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
    glove = GloveData(base_datapath)
    # for glove.6B.300d.txt'
    print(' ' in glove.dict.keys())  # False
    print('\n' in glove.dict.keys())  # False
    print('.' in glove.dict.keys())  # True
    print(',' in glove.dict.keys())  # True
    print('?' in glove.dict.keys())  # True
    print('!' in glove.dict.keys())  # True
    print('the' in glove.dict.keys())  # True
    print('a' in glove.dict.keys())  # True
    print('<UNK>' in glove.dict.keys())  # False -> True
    print('<EOS>' in glove.dict.keys())  # False -> True
    print('<SOS>' in glove.dict.keys())  # False -> True
    print('"' in glove.dict.keys())  # True
    print('learn' in glove.dict.keys())  # True
    print('learning' in glove.dict.keys())  # True
    print('learned' in glove.dict.keys())  # True

    embedded = glove.words_to_vec(['i', 'have', 'kyuhong', 'cat'])
    print(embedded.shape)
    print(embedded)
    print(glove.embedding[glove.dict['<UNK>']])
    
    print('----------------------------------------')
    print(glove.embedding[glove.dict['learn']][:20])
    print(glove.embedding[glove.dict['learning']][:20])
    print(glove.embedding[glove.dict['learned']][:20])