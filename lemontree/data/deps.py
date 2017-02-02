"""
This code includes functions for reading deps word embeddings and search.

Download
--------
Dependency-Based Word Embeddings
https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
Download 'Dependency-Based[words]'.
Unzip the file.

Base datapath: '~~~/data/'
Additional folder structure: '~~~/data/deps/deps.words'
"""

import time
import pickle
import numpy as np


class DepsData(object):
    """
    This class use deps word vector set to make word embeddings.
    """
    def __init__(self, base_datapath, seed=33, load_pickle=True):
        """
        This function initializes the class.
        Initialization contains loading and parsing of deps data.
        Either two way is possible.
        One is to save deps as dictonary type in memory.
        The other is to save deps separately, label and index to dictionary an data in ndarray.
        We choose second option to get (maybe slightly) better cache and memory usage.
        
        Parameters
        ----------
        base_datapath: string
            a string path where deps vector text is saved.
        seed: int
            an integer to randomly generate eos, sos, unk tokens.
        load_pickle: bool, default: False
            a bool value to load precomputed embedding and dictionary.

        Returns
        -------
        None.
        """
        # check asserts
        assert isinstance(base_datapath, str), '"base_datapath" should be a string path.'
        assert isinstance(load_pickle, bool), '"load_pickle" should be a bool value.'
        
        # load
        start_time = time.clock()
        if load_pickle:
            dict_file = base_datapath + 'deps/deps_dict.pickle'
            print('Deps dict load file:', dict_file)
            with open(dict_file, 'rb') as f:
                self.dict = pickle.load(f)

            embedding_file = base_datapath + 'deps/deps_embedding.pickle'
            print('Deps embedding load file:', embedding_file)
            with open(embedding_file, 'rb') as f:
                self.embedding = pickle.load(f)
            self.vocabulary = self.embedding.shape[0]
            self.dimension = self.embedding.shape[1]
        else:
            deps_file = base_datapath + 'deps/deps.words'
            print('Deps load file:', deps_file)
            self.dict = {}
            self.embedding = []
            index = 0            
            with open(deps_file, 'r', encoding='utf-8') as f:
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
            self.vocabulary = len(self.embedding)
            
            # save pickle dump
            dict_file = base_datapath + 'deps/deps_dict.pickle'
            print('Deps dict save file:', dict_file)
            with open(dict_file, 'wb') as f:
                pickle.dump(self.dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            embedding_file = base_datapath + 'deps/deps_embedding.pickle'
            print('Deps embedding save file:', embedding_file)
            with open(embedding_file, 'wb') as f:
                pickle.dump(self.embedding, f, protocol=pickle.HIGHEST_PROTOCOL)

        # make reversed dictionay to find key by value
        self.dict_reverse = dict(zip(self.dict.values(), self.dict.keys()))

        end_time= time.clock()
        print('Deps load time:', end_time - start_time)
        print('Deps data shape:', self.embedding.shape)     
        print('Deps total vocabulary:', self.vocabulary)

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
                key = self.vocabulary -1 # <unk>
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
        This function converts words into deps word vectors.
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
    base_datapath = 'D:/Dropbox/Project/data/'
    deps = DepsData(base_datapath, load_pickle=True)
    # for deps.words'
    print(' ' in deps.dict.keys())  # False
    print('\n' in deps.dict.keys())  # False
    print('.' in deps.dict.keys())  # True
    print(',' in deps.dict.keys())  # True
    print('?' in deps.dict.keys())  # True
    print('!' in deps.dict.keys())  # True
    print('the' in deps.dict.keys())  # True
    print('The' in deps.dict.keys())  # False
    print('a' in deps.dict.keys())  # True
    print('A' in deps.dict.keys())  # False
    print('<UNK>' in deps.dict.keys())  # False -> True
    print('<EOS>' in deps.dict.keys())  # False -> True
    print('<SOS>' in deps.dict.keys())  # False -> True
    print('"' in deps.dict.keys())  # True
    print('learn' in deps.dict.keys())  # True
    print('learning' in deps.dict.keys())  # True
    print('learned' in deps.dict.keys())  # True

    embedded = deps.words_to_vec(['i', 'have', 'kyuhong', 'cat'])
    print(embedded.shape)
    print(embedded)
    print(deps.embedding[deps.dict['<UNK>']])