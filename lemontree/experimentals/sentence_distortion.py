"""
This code implements sentence distortion to make wrong sentences.
Four distortion is introduced.
Shuffle (only one pair adjoint)
Delete (one)
Insert (one)
Shuffle (random)
"""

import numpy as np
from lemontree.data.glove import GloveData

def shuffle_random(words):
    num = len(words)
    how = np.random.permutation(num)
    new_words = []
    for i in range(num):
        new_words.append(words[how[i]])
    return new_words


def delete_one(words):
    num = len(words)
    new_words = words.copy()
    index = np.random.randint(0,num)
    new_words.pop(index)
    return new_words


def duplicate_one(words):
    num = len(words)
    new_words = words.copy()
    index = np.random.randint(0,num)
    new_words.insert(index, words[index])
    return new_words


def swap_pair(words):
    num = len(words)
    index = list(range(num))
    for i in range(0, num - 1, 2):
        index[i] = i+1
        index[i+1]=i
    return list(words[i] for i in index)
