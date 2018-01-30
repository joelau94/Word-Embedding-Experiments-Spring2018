'WordEmbs.pkl' contains the words and their corresponding embeddings from the dependency parser I trained as homework 4.

Usage:
# python code:

import cPickle as pkl
import numpy as np

data = pkl.load(open('WordEmbs.pkl','rb')) # load pickle object
word_list = data['words'] # a python list containing words in the same order as in embeddings (4807 words)
word_embedding = data['embeddings'] # a numpy array, which is the word embedding matrix (dimension: 4807 * 64)

# end of python code

Let's try t-SNE / clustering / PCA on this toy data. We may even only try 100 words to see what happens.
I'll try to export the word-embeddings from SyntaxNet model this week, but before that let's play our methods on the toy first.
