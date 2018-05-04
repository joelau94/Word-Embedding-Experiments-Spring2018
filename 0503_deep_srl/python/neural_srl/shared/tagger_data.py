from constants import UNKNOWN_TOKEN

import os
import numpy as np
import random
from collections import defaultdict
import cPickle as pkl

import pdb

def char_batch_input(words):
  '''
  :param words: list[list[str]]
  :return oov_char: padded oov_char array
  :return rnn_mask: oov word char rnn mask
  '''
  if len(words) == 0:
    return np.array([[]]), np.array([[]])

  max_word_len = max(map(len, words))
  rnn_mask = np.array([[1.]*len(w) + [0.]*(max_word_len-len(w)) for w in words])
  oov_char = np.array([w + [0.]*(max_word_len-len(w)) for w in words])

  return oov_char, rnn_mask

def _process_x_y_weights(length, x, y):
  weights = np.resize((y >= 0).astype(float), (length))
  x = np.resize(x, (length, x.shape[1]))
  y = np.resize(y, (length))

  return x, y, weights

def tensorize(sentence, max_length):
  """ Input:
      - sentence: The sentence is a tuple of lists (s1, s2, ..., sk)
            s1 is always a sequence of word ids.
            sk is always a sequence of label ids.
            s2 ... sk-1 are sequences of feature ids,
              such as predicate or supertag features.
      - max_length: The maximum length of sequences, used for padding.
  """
  x = np.array([t for t in zip(*sentence[:-1])])
  y = np.array(sentence[-1])
  x, y, weights = _process_x_y_weights(max_length, x, y)
  return x, np.absolute(y), len(sentence[0]), weights

def tensorize_with_rand_oov(sentence, max_length):
  x = np.array([t for t in zip(*sentence[:-1])])
  y = np.array(sentence[-1])
  oov_pos = np.random.randint(1, len(sentence[-1])-1)
  x, y, weights = _process_x_y_weights(max_length, x, y)
  return x, np.absolute(y), oov_pos, len(sentence[0]), weights

def tensorize_with_oov(sentence, unk_id, max_length):
  x = np.array([t for t in zip(*sentence[:-1])])
  y = np.array(sentence[-1])
  oov_pos = np.array([i for i, idx in enumerate(sentence[0]) if idx == unk_id])
  x, y, weights = _process_x_y_weights(max_length, x, y)
  return x, np.absolute(y), oov_pos, len(sentence[0]), weights

def tensorize_with_raw_rand_oov(sentence, raw_sent, max_length, c2i):
  x = np.array([t for t in zip(*sentence[:-1])])
  y = np.array(sentence[-1])
  oov_pos = np.random.randint(1, len(sentence[-1])-1)
  oov_char = [c2i[c] for c in raw_sent[oov_pos]]

  x, y, weights = _process_x_y_weights(max_length, x, y)
  return x, np.absolute(y), oov_pos, oov_char, len(sentence[0]), weights

def tensorize_with_raw_oov(sentence, raw_sent, unk_id, max_length, c2i):
  x = np.array([t for t in zip(*sentence[:-1])])
  y = np.array(sentence[-1])
  oov_pos = np.array([i for i, idx in enumerate(sentence[0]) if idx == unk_id])

  oov_char = [[c2i[c] for c in raw_sent[i]] for i in oov_pos]
  oov_char, rnn_mask = char_batch_input(oov_char)

  x, y, weights = _process_x_y_weights(max_length, x, y)
  return x, np.absolute(y), oov_pos, oov_char, rnn_mask, len(sentence[0]), weights
  
class TaggerData(object):

  def __init__(self, config, train_sents, dev_sents, word_dict, label_dict, embeddings, embedding_shapes,
         feature_dicts=None, raw_sents=None):
    ''' 
    '''
    self.max_train_length = config.max_train_length
    self.max_dev_length = max([len(s[0]) for s in dev_sents]) if len(dev_sents) > 0 else 0
    self.batch_size = config.batch_size
    self.use_se_marker = config.use_se_marker
    self.unk_id = word_dict.str2idx[UNKNOWN_TOKEN]
    
    self.train_sents = [s for s in train_sents if len(s[0]) <= self.max_train_length]
    self.dev_sents = dev_sents

    if raw_sents is not None:
      raw_train_sents, raw_dev_sents = raw_sents
      self.raw_train_sents = [map(str.lower, s) for s in raw_train_sents if len(s[0]) <= self.max_train_length]
      self.raw_dev_sents = [map(str.lower, s) for s in raw_dev_sents]
      # self.raw_train_sents: list[list[str]]

    self.word_dict = word_dict
    self.label_dict = label_dict
    self.embeddings = embeddings
    self.embedding_shapes = embedding_shapes
    self.feature_dicts = feature_dicts

    # train_tensors: list of sent [tuple: (x, y, len, mask)]: x, y, mask are np arrays
    self.train_tensors = [tensorize(s, self.max_train_length) for s in self.train_sents]
    self.dev_tensors =  [tensorize(s, self.max_dev_length) for s in self.dev_sents]

  def init_char(self):
    self.w2i = self.word_dict.str2idx
    words = [w for i, w in enumerate(self.word_dict.idx2str) if i != self.unk_id]
    self.i2c = [' '] + list(set(''.join(words)))
    self.c2i = defaultdict(int)
    for i, c in enumerate(self.i2c):
      self.c2i[c] = i

  def save_char(self, path):
    pkl.dump(self.c2i, open(os.path.join(path, 'c2i.pkl'), 'w+')) # save char mapping

  def load_char(self, path):
    self.c2i = pkl.load(open(os.path.join(path, 'c2i.pkl'), 'r'))
    
  def get_ctx_gemb_training_data(self, include_last_batch=False):
    """
    Get shuffled training samples, which does not contain unk
    Called at the beginning of each epoch.
    """
    unk_id = self.word_dict.unknown_id
    self.gemb_train_sents = [s for s in self.train_sents
        if len(s[0]) <= self.max_train_length and unk_id not in s]
    self.gemb_train_tensors = [tensorize_with_rand_oov(s, self.max_train_length) for s in self.gemb_train_sents]
    train_ids = range(len(self.gemb_train_sents))
    random.shuffle(train_ids)

    if not include_last_batch:
      num_batches = len(train_ids) // self.batch_size
      train_ids = train_ids[:num_batches * self.batch_size]
      
    num_samples = len(self.gemb_train_sents)
    tensors = [self.gemb_train_tensors[t] for t in train_ids] # shuffled train data
    batched_tensors = [tensors[i: min(i+self.batch_size, num_samples)]
               for i in xrange(0, num_samples, self.batch_size)]
    # batched_tensors: list of batch (list of sent (tuple: x, y, oov_pos, len, mask))
    results = [zip(*t) for t in batched_tensors]
    # results: list of batch [ tuple(list of x, list of y, list of oov_pos, list of len, list of mask) ]
    
    print("Extracted {} samples and {} batches for GEMB training.".format(num_samples, len(batched_tensors)))
    return results

  def get_mix_gemb_training_data(self, include_last_batch=False):
    """
    Get shuffled training samples, which does not contain unk
    Called at the beginning of each epoch.
    """
    unk_id = self.word_dict.unknown_id
    self.gemb_train_sents = [(s, r) for i, (s, r) in enumerate(zip(self.train_sents, self.raw_train_sents))
        if len(s[0]) <= self.max_train_length and unk_id not in s]
    self.gemb_train_tensors = [tensorize_with_raw_rand_oov(s, r, self.max_train_length, self.c2i)
                                for s, r in self.gemb_train_sents]
    train_ids = range(len(self.gemb_train_sents))
    random.shuffle(train_ids)

    if not include_last_batch:
      num_batches = len(train_ids) // self.batch_size
      train_ids = train_ids[:num_batches * self.batch_size]
      
    num_samples = len(self.gemb_train_sents)
    tensors = [self.gemb_train_tensors[t] for t in train_ids] # shuffled train data
    batched_tensors = [tensors[i: min(i+self.batch_size, num_samples)]
               for i in xrange(0, num_samples, self.batch_size)]
    # batched_tensors: list of batch (list of sent (tuple: x, y, oov_pos, oov_char, len, mask))
    results = [zip(*t) for t in batched_tensors]
    # results: list of batch [ tuple(list of x, list of y, list of oov_pos, list of oov_char, list of len, list of mask) ]
    
    print("Extracted {} samples and {} batches for GEMB training.".format(num_samples, len(batched_tensors)))
    return results

  def get_training_data(self, include_last_batch=False):
    """ Get shuffled training samples. Called at the beginning of each epoch.
    """
    # TODO: Speed up: Use variable size batches (different max length).  
    train_ids = range(len(self.train_sents))
    random.shuffle(train_ids)
    
    if not include_last_batch:
      num_batches = len(train_ids) // self.batch_size
      train_ids = train_ids[:num_batches * self.batch_size]
      
    num_samples = len(self.train_sents)
    tensors = [self.train_tensors[t] for t in train_ids] # shuffled train data
    batched_tensors = [tensors[i: min(i+self.batch_size, num_samples)]
               for i in xrange(0, num_samples, self.batch_size)]
    # batched_tensors: list of batch (list of sent (tuple: x, y, len, mask))
    results = [zip(*t) for t in batched_tensors]
    # results: list of batch [ tuple(list of x, list of y, list of len, list of mask) ]
    
    print("Extracted {} samples and {} batches.".format(num_samples, len(batched_tensors)))
    return results
  
  def get_development_data(self, batch_size=None):
    if batch_size is None:
      return [np.array(v) for v in zip(*self.dev_tensors)]
    
    num_samples = len(self.dev_sents)
    batched_tensors = [self.dev_tensors[i: min(i+ batch_size, num_samples)]
               for i in xrange(0, num_samples, batch_size)]
    return [zip(*t) for t in batched_tensors]
  
  def get_test_data(self, test_sentences, batch_size = None):
    max_len = max([len(s[0]) for s in test_sentences])
    num_samples = len(test_sentences)
    #print("Max sentence length: {} among {} samples.".format(max_len, num_samples))
    test_tensors =  [tensorize(s, max_len) for s in test_sentences]
    if batch_size is None:
      return [np.array(v) for v in zip(*test_tensors)]
    batched_tensors = [test_tensors[i: min(i+ batch_size, num_samples)]
               for i in xrange(0, num_samples, batch_size)]
    return [zip(*t) for t in batched_tensors]
  
  def get_ctx_gemb_development_data(self, batch_size=1):
    batch_size = 1 # batch size must be 1, because number of oov is not equal for each sent
    unk_id = self.word_dict.unknown_id
    self.gemb_dev_tensors = [tensorize_with_oov(s, unk_id, self.max_dev_length) for s in self.dev_sents]
    if batch_size is None:
      return [np.array(v) for v in zip(*self.gemb_dev_tensors)]
    
    num_samples = len(self.dev_sents)
    batched_tensors = [self.gemb_dev_tensors[i: min(i+ batch_size, num_samples)]
               for i in xrange(0, num_samples, batch_size)]
    return [zip(*t) for t in batched_tensors]
  
  def get_ctx_gemb_test_data(self, test_sentences, batch_size = None):
    batch_size = 1 # batch size must be 1, because number of oov is not equal for each sent
    unk_id = self.word_dict.unknown_id
    max_len = max([len(s[0]) for s in test_sentences])
    num_samples = len(test_sentences)
    #print("Max sentence length: {} among {} samples.".format(max_len, num_samples))
    test_tensors =  [tensorize_with_oov(s, unk_id, max_len) for s in test_sentences]
    if batch_size is None:
      return [np.array(v) for v in zip(*test_tensors)]
    batched_tensors = [test_tensors[i: min(i+ batch_size, num_samples)]
               for i in xrange(0, num_samples, batch_size)]
    return [zip(*t) for t in batched_tensors]

  def get_mix_gemb_development_data(self, batch_size=1):
    batch_size = 1 # batch size must be 1, because number of oov is not equal for each sent
    unk_id = self.word_dict.unknown_id
    self.gemb_dev_tensors = [tensorize_with_raw_oov(s, r, unk_id, self.max_dev_length, self.c2i)
                              for s, r in zip(self.dev_sents, self.raw_dev_sents)]
    if batch_size is None:
      return [np.array(v) for v in zip(*self.gemb_dev_tensors)]
    
    num_samples = len(self.dev_sents)
    batched_tensors = [self.gemb_dev_tensors[i: min(i+ batch_size, num_samples)]
               for i in xrange(0, num_samples, batch_size)]
    return [zip(*t) for t in batched_tensors]
  
  def get_mix_gemb_test_data(self, test_sentences, raw_test_sents, batch_size = None):
    batch_size = 1 # batch size must be 1, because number of oov is not equal for each sent
    unk_id = self.word_dict.unknown_id
    max_len = max([len(s[0]) for s in test_sentences])
    num_samples = len(test_sentences)
    #print("Max sentence length: {} among {} samples.".format(max_len, num_samples))
    test_tensors =  [tensorize_with_raw_oov(s, r, unk_id, max_len, self.c2i) for s, r in zip(test_sentences, raw_test_sents)]
    if batch_size is None:
      return [np.array(v) for v in zip(*test_tensors)]
    batched_tensors = [test_tensors[i: min(i+ batch_size, num_samples)]
               for i in xrange(0, num_samples, batch_size)]
    return [zip(*t) for t in batched_tensors]
