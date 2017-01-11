# Kyuhong Shim 2016

import numpy as np
import nltk

# Nietzsche text data.
# https://s3.amazonaws.com/text-datasets/nietzsche.txt
# Download nietzsche.txt, save as ANSI encoding.
# Change to character-level sequences.

# nltk.download()
# Download model -> punkt

def load_nietzsche(base_datapath, mode='character'):
    nietzsche = open(base_datapath + 'nietzsche/nietzsche.txt')
    corpus = nietzsche.read()
    if mode == 'character':
        print('Corpus length: ', len(corpus))
        sequences = list(bytearray(corpus.encode('utf-8')))        
    elif mode == 'word':
        sequences = nltk.word_tokenize(corpus)
    elif mode == 'sentence':
        corpus.replace('\n', '')
        sequences = nltk.tokenize.sent_tokenize(corpus)
    else:
        raise NotImplementedError('Not yet supported')
    print('Sequence length: ', len(sequences))
    next_sequences = sequences[1:] + [sequences[0]]
    return sequences, next_sequences  # return list of characters/words/sentences


if __name__ == '__main__':
    base_datapath = 'C:/Users/skhu2/Dropbox/Project/data/'
    sequences, next_sequences = load_nietzsche(base_datapath, mode = 'word')
    print(len(sequences), len(next_sequences))