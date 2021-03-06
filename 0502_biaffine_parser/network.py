#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Timothy Dozat
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import pickle as pkl

import numpy as np
import tensorflow as tf

from lib import models
from lib import optimizers
from lib import rnn_cells

from configurable import Configurable
from vocab import Vocab
from dataset import Dataset


def replace_with_gemb(word_inputs, gembedding, oov_pos):
  word_inputs_new = np.copy(word_inputs)
  #print(len(gembedding))
  #print(oov_pos)
  for i, oov in zip(oov_pos, gembedding):
    word_inputs_new[0, i, :] = oov
  # for i, oov in enumerate(gembedding):
  #   word_inputs_new[0, i, :] = oov
  return word_inputs_new

#***************************************************************
class Network(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, model, *args, **kwargs):
    """"""
    
    if args:
      if len(args) > 1:
        raise TypeError('Parser takes at most one argument')
    
    kwargs['name'] = kwargs.pop('name', model.__name__)
    super(Network, self).__init__(*args, **kwargs)
    if not os.path.isdir(self.save_dir):
      os.mkdir(self.save_dir)
    with open(os.path.join(self.save_dir, 'config.cfg'), 'w') as f:
      self._config.write(f)
      
    self._global_step = tf.Variable(0., trainable=False)
    self._global_epoch = tf.Variable(0., trainable=False)
    self._model = model(self._config, global_step=self.global_step)
    
    self._vocabs = []
    vocab_files = [(self.word_file, 1, 'Words'),
                   (self.tag_file, [3, 4], 'Tags'),
                   (self.rel_file, 7, 'Rels')]
    for i, (vocab_file, index, name) in enumerate(vocab_files):
      vocab = Vocab(vocab_file, index, self._config,
                    name=name,
                    cased=self.cased if not i else True,
                    use_pretrained=(not i),
                    global_step=self.global_step)
      self._vocabs.append(vocab)
    
    self._trainset = Dataset(self.train_file, self._vocabs, model, self._config, name='Trainset')
    self._validset = Dataset(self.valid_file, self._vocabs, model, self._config, name='Validset')
    self._testset = Dataset(self.test_file, self._vocabs, model, self._config, name='Testset')
    self._train_gemb_set = Dataset(self.train_file, self._vocabs, model, self._config, name='Trainset')
    self._valid_gemb_set = Dataset(self.valid_file, self._vocabs, model, self._config, name='Validset')
    self._test_gemb_set = Dataset(self.test_file, self._vocabs, model, self._config, name='Testset')
    
    self._ops = self._gen_ops()
    self._save_vars = filter(lambda x: u'Pretrained' not in x.name, tf.global_variables())
    self.history = {
      'train_loss': [],
      'train_accuracy': [],
      'valid_loss': [],
      'valid_accuracy': [],
      'test_acuracy': 0
    }
    return
  
  #=============================================================
  def train_minibatches(self):
    """"""
    
    return self._trainset.get_minibatches(self.train_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs)

  def train_gemb_minibatches(self):
    """"""
    
    return self._train_gemb_set.get_minibatches_gemb_train(self.train_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs)
  def valid_gemb_minibatches(self):
    """"""
    
    return self._valid_gemb_set.get_minibatches_gemb_train(1,
                                          self.model.input_idxs,
                                          self.model.target_idxs,
                                          shuffle = False)
  

  #=============================================================
  def valid_minibatches(self):
    """"""
    
    return self._validset.get_minibatches(self.test_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs,
                                          shuffle=False)
  
  #=============================================================
  def test_minibatches(self):
    """"""
    
    return self._testset.get_minibatches(self.test_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs,
                                          shuffle=False)

  def test_gemb_minibatches(self):
    """"""
    
    return self._test_gemb_set.get_minibatches_gemb_test(self.test_batch_size,
                                          self.model.input_idxs,
                                          self.model.target_idxs,
                                          shuffle=False)
  
  #=============================================================
  # assumes the sess has already been initialized
  def train(self, sess):
    """"""
    
    save_path = os.path.join(self.save_dir, self.name.lower() + '-pretrained')
    saver = tf.train.Saver(self.save_vars, max_to_keep=1)
    
    n_bkts = self.n_bkts
    train_iters = self.train_iters
    print_every = self.print_every
    validate_every = self.validate_every
    save_every = self.save_every
    try:
      train_time = 0
      train_loss = 0
      n_train_sents = 0
      n_train_correct = 0
      n_train_tokens = 0
      n_train_iters = 0
      total_train_iters = sess.run(self.global_step)
      valid_time = 0
      valid_loss = 0
      valid_accuracy = 0
      while total_train_iters < train_iters:
        for j, (feed_dict, _) in enumerate(self.train_minibatches()):
          train_inputs = feed_dict[self._trainset.inputs]
          train_targets = feed_dict[self._trainset.targets]
          start_time = time.time()
          _, loss, n_correct, n_tokens = sess.run(self.ops['train_op'], feed_dict=feed_dict)
          train_time += time.time() - start_time
          train_loss += loss
          n_train_sents += len(train_targets)
          n_train_correct += n_correct
          n_train_tokens += n_tokens
          n_train_iters += 1
          total_train_iters += 1
          self.history['train_loss'].append(loss)
          self.history['train_accuracy'].append(100 * n_correct / n_tokens)
          if total_train_iters == 1 or total_train_iters % validate_every == 0:
            valid_time = 0
            valid_loss = 0
            n_valid_sents = 0
            n_valid_correct = 0
            n_valid_tokens = 0
            with open(os.path.join(self.save_dir, 'sanitycheck.txt'), 'w') as f:
              for k, (feed_dict, _) in enumerate(self.valid_minibatches()):
                inputs = feed_dict[self._validset.inputs]
                targets = feed_dict[self._validset.targets]
                start_time = time.time()
                loss, n_correct, n_tokens, predictions = sess.run(self.ops['valid_op'], feed_dict=feed_dict)
                valid_time += time.time() - start_time
                valid_loss += loss
                n_valid_sents += len(targets)
                n_valid_correct += n_correct
                n_valid_tokens += n_tokens
                self.model.sanity_check(inputs, targets, predictions, self._vocabs, f, feed_dict=feed_dict)
            valid_loss /= k+1
            valid_accuracy = 100 * n_valid_correct / n_valid_tokens
            valid_time = n_valid_sents / valid_time
            self.history['valid_loss'].append(valid_loss)
            self.history['valid_accuracy'].append(valid_accuracy)
          if print_every and total_train_iters % print_every == 0:
            train_loss /= n_train_iters
            train_accuracy = 100 * n_train_correct / n_train_tokens
            train_time = n_train_sents / train_time
            print('%6d) Train loss: %.4f    Train acc: %5.2f%%    Train rate: %6.1f sents/sec\n\tValid loss: %.4f    Valid acc: %5.2f%%    Valid rate: %6.1f sents/sec' % (total_train_iters, train_loss, train_accuracy, train_time, valid_loss, valid_accuracy, valid_time))
            train_time = 0
            train_loss = 0
            n_train_sents = 0
            n_train_correct = 0
            n_train_tokens = 0
            n_train_iters = 0
        sess.run(self._global_epoch.assign_add(1.))
        if save_every and (total_train_iters % save_every == 0):
          saver.save(sess, os.path.join(self.save_dir, self.name.lower() + '-trained'),
                     latest_filename=self.name.lower(),
                     global_step=self.global_epoch,
                     write_meta_graph=False)
          with open(os.path.join(self.save_dir, 'history.pkl'), 'w') as f:
            pkl.dump(self.history, f)
          self.test(sess, validate=True)
    except KeyboardInterrupt:
      try:
        raw_input('\nPress <Enter> to save or <Ctrl-C> to exit.')
      except:
        print('\r', end='')
        sys.exit(0)
    saver.save(sess, os.path.join(self.save_dir, self.name.lower() + '-trained'),
               latest_filename=self.name.lower(),
               global_step=self.global_epoch,
               write_meta_graph=False)
    with open(os.path.join(self.save_dir, 'history.pkl'), 'w') as f:
      pkl.dump(self.history, f)
    with open(os.path.join(self.save_dir, 'scores.txt'), 'w') as f:
      pass
    self.test(sess, validate=True)
    return


  # assumes the sess has already been initialized
  def train_gemb(self, sess):
    """"""
    
    save_path = os.path.join(self.save_dir, self.name.lower() + '-pretrained')
    saver = tf.train.Saver(self.save_vars, max_to_keep=1)
    
    n_bkts = self.n_bkts
    train_iters = self.train_iters
    print_every = self.print_every
    validate_every = self.validate_every
    save_every = self.save_every
    try:
      train_time = 0
      train_loss = 0
      n_train_sents = 0
      n_train_correct = 0
      n_train_tokens = 0
      n_train_iters = 0
      total_train_iters = sess.run(self.global_step)
      valid_time = 0
      valid_loss = 0
      valid_accuracy = 0
      while total_train_iters < train_iters:
        #print("Iteration:", total_train_iters)
        for j, (feed_dict, oov_pos, _) in enumerate(self.train_gemb_minibatches()):
          train_inputs = feed_dict[self._train_gemb_set.inputs] # useless
          train_targets = feed_dict[self._train_gemb_set.targets] # useless
          feed_dict.update({self.train_gemb_graph['oov_pos']: oov_pos})
          start_time = time.time()
          _, loss = sess.run(self.ops['train_gemb_op'], feed_dict=feed_dict)
          train_time += time.time() - start_time
          train_loss += loss
          n_train_sents += len(train_targets)
          n_train_iters += 1
          total_train_iters += 1
          self.history['train_loss'].append(loss)
          if total_train_iters == 1 or total_train_iters % validate_every == 0:
            valid_time = 0
            valid_loss = 0
            n_valid_sents = 0
            n_valid_correct = 0
            n_valid_tokens = 0
            with open(os.path.join(self.save_dir, 'sanitycheck.txt'), 'w') as f:
              for k, (feed_dict, oov_pos, _) in enumerate(self.valid_gemb_minibatches()):
                inputs = feed_dict[self._valid_gemb_set.inputs]
                targets = feed_dict[self._valid_gemb_set.targets]
                start_time = time.time()

                # get gemb
                feed_dict.update({self.valid_get_gemb_graph['oov_pos']: oov_pos})
                gembedding, word_inputs = sess.run([self.valid_get_gemb_graph['gembedding'],
                                                    self.valid_get_gemb_graph['word_inputs']],
                                                  feed_dict=feed_dict)
                #gembedding_new = replace_with_gemb(word_inputs, gembedding, oov_pos)
                gembedding_new = word_inputs
                # replace with gemb
                feed_dict.update({self.valid_gemb_graph['gembedding_new']: gembedding_new})
                loss, n_correct, n_tokens, predictions = sess.run(self.ops['valid_gemb_op'], feed_dict=feed_dict)
                #rel_logits = sess.run(self.valid_gemb_graph['rel_logits'], feed_dict=feed_dict)
                #print('rel_logits:{}', rel_logits.shape)

                valid_time += time.time() - start_time
                valid_loss += loss
                n_valid_sents += len(targets)
                n_valid_correct += n_correct
                n_valid_tokens += n_tokens
                self.model.sanity_check(inputs, targets, predictions, self._vocabs, f, feed_dict={
                        self._valid_gemb_set.inputs: inputs,
                        self._valid_gemb_set.targets: targets
                    })
            valid_loss /= k+1
            valid_accuracy = 100 * n_valid_correct / n_valid_tokens
            valid_time = n_valid_sents / valid_time
            self.history['valid_loss'].append(valid_loss)
            self.history['valid_accuracy'].append(valid_accuracy)
          if print_every and total_train_iters % print_every == 0:
            train_loss /= n_train_iters
            train_time = n_train_sents / train_time
            print('%6d) Train loss: %.4f    Train rate: %6.1f sents/sec\n\tValid loss: %.4f    Valid acc: %5.2f%%    Valid rate: %6.1f sents/sec' % (total_train_iters, train_loss, train_time, valid_loss, valid_accuracy, valid_time))
            train_time = 0
            train_loss = 0
            n_train_sents = 0
            n_train_iters = 0
        sess.run(self._global_epoch.assign_add(1.))
        if save_every and (total_train_iters % save_every == 0):
          saver.save(sess, os.path.join(self.save_dir, self.name.lower() + '-trained'),
                     latest_filename=self.name.lower(),
                     global_step=self.global_epoch,
                     write_meta_graph=False)
          with open(os.path.join(self.save_dir, 'history.pkl'), 'w') as f:
            pkl.dump(self.history, f)
          self.test(sess, validate=True)
    except KeyboardInterrupt:
      try:
        raw_input('\nPress <Enter> to save or <Ctrl-C> to exit.')
      except:
        print('\r', end='')
        sys.exit(0)
    saver.save(sess, os.path.join(self.save_dir, self.name.lower() + '-trained'),
               latest_filename=self.name.lower(),
               global_step=self.global_epoch,
               write_meta_graph=False)
    with open(os.path.join(self.save_dir, 'history.pkl'), 'w') as f:
      pkl.dump(self.history, f)
    with open(os.path.join(self.save_dir, 'scores.txt'), 'w') as f:
      pass
    self.test(sess, validate=True)
    return


  #=============================================================
  # TODO make this work if lines_per_buff isn't set to 0
  def test_with_gemb(self, sess, validate=False):
    """
    self.model is a class: Parser
    self._model is the instance of Parser class
    """
    if validate:
      filename = self.valid_file
      minibatches = self.valid_gemb_minibatches
      dataset = self._valid_gemb_set
      op = self.valid_gemb_graph['probabilities']
      get_gemb_op = [self.valid_get_gemb_graph['gembedding'], self.valid_get_gemb_graph['word_inputs']]
      get_gemb_graph = self.valid_get_gemb_graph
      graph = self.valid_gemb_graph
    else:
      filename = self.test_file
      minibatches = self.test_gemb_minibatches
      dataset = self._test_gemb_set
      op = self.test_gemb_graph['probabilities']
      get_gemb_op = [self.test_get_gemb_graph['gembedding'], self.test_get_gemb_graph['word_inputs']]
      get_gemb_graph = self.test_get_gemb_graph
      graph = self.test_gemb_graph

    all_predictions = [[]]
    all_sents = [[]]
    bkt_idx = 0
    for (feed_dict, oov_pos, sents) in minibatches():
      mb_inputs = feed_dict[dataset.inputs]
      mb_targets = feed_dict[dataset.targets]

      # get gemb
      feed_dict.update({get_gemb_graph['oov_pos']: oov_pos})
      gembedding, word_inputs = sess.run(get_gemb_op, feed_dict=feed_dict)

      # replace with gemb
      gembedding_new = replace_with_gemb(word_inputs, gembedding, oov_pos)
      #gembedding_new = word_inputs
      mb_probs = sess.run(op, feed_dict={
            dataset.inputs: mb_inputs,
            dataset.targets: mb_targets,
            graph['gembedding_new']: gembedding_new
        })
      all_predictions[-1].extend(self.model.validate(mb_inputs, mb_targets, mb_probs))
      all_sents[-1].extend(sents)
      
      if len(all_predictions[-1]) == len(dataset[bkt_idx]):
        bkt_idx += 1
        if bkt_idx < len(dataset._metabucket):
          all_predictions.append([])
          all_sents.append([])
    
    with open(os.path.join(self.save_dir, os.path.basename(filename)), 'w') as f:
      for bkt_idx, idx in dataset._metabucket.data:
        data = dataset._metabucket[bkt_idx].data[idx][1:]
        preds = all_predictions[bkt_idx][idx]
        words = all_sents[bkt_idx][idx]
        for i, (datum, word, pred) in enumerate(zip(data, words, preds)):
          tup = (
            i+1,
            word,
            self.tags[pred[3]] if pred[3] != -1 else self.tags[datum[2]],
            self.tags[pred[4]] if pred[4] != -1 else self.tags[datum[3]],
            str(pred[5]) if pred[5] != -1 else str(datum[4]),
            self.rels[pred[6]] if pred[6] != -1 else self.rels[datum[5]],
            str(pred[7]) if pred[7] != -1 else '_',
            self.rels[pred[8]] if pred[8] != -1 else '_',
          )
          f.write('%s\t%s\t_\t%s\t%s\t_\t%s\t%s\t%s\t%s\n' % tup)
        f.write('\n')
    with open(os.path.join(self.save_dir, 'scores.txt'), 'a') as f:
      s, _ = self.model.evaluate(os.path.join(self.save_dir, os.path.basename(filename)), punct=self.model.PUNCT)
      f.write(s)
    return


  def test(self, sess, validate=False):
    
    if validate:
      filename = self.valid_file
      minibatches = self.valid_minibatches
      dataset = self._validset
      op = self.valid_graph['probabilities']
    else:
      filename = self.test_file
      minibatches = self.test_minibatches
      dataset = self._testset
      op = self.test_graph['probabilities']
    
    all_predictions = [[]]
    all_sents = [[]]
    bkt_idx = 0
    for (feed_dict, sents) in minibatches():
      mb_inputs = feed_dict[dataset.inputs]
      mb_targets = feed_dict[dataset.targets]
      mb_probs = sess.run(op, feed_dict=feed_dict)
      all_predictions[-1].extend(self.model.validate(mb_inputs, mb_targets, mb_probs))
      all_sents[-1].extend(sents)
      if len(all_predictions[-1]) == len(dataset[bkt_idx]):
        bkt_idx += 1
        if bkt_idx < len(dataset._metabucket):
          all_predictions.append([])
          all_sents.append([])
    with open(os.path.join(self.save_dir, os.path.basename(filename)), 'w') as f:
      for bkt_idx, idx in dataset._metabucket.data:
        data = dataset._metabucket[bkt_idx].data[idx][1:]
        preds = all_predictions[bkt_idx][idx]
        words = all_sents[bkt_idx][idx]
        for i, (datum, word, pred) in enumerate(zip(data, words, preds)):
          tup = (
            i+1,
            word,
            self.tags[pred[3]] if pred[3] != -1 else self.tags[datum[2]],
            self.tags[pred[4]] if pred[4] != -1 else self.tags[datum[3]],
            str(pred[5]) if pred[5] != -1 else str(datum[4]),
            self.rels[pred[6]] if pred[6] != -1 else self.rels[datum[5]],
            str(pred[7]) if pred[7] != -1 else '_',
            self.rels[pred[8]] if pred[8] != -1 else '_',
          )
          f.write('%s\t%s\t_\t%s\t%s\t_\t%s\t%s\t%s\t%s\n' % tup)
        f.write('\n')
    with open(os.path.join(self.save_dir, 'scores.txt'), 'a') as f:
      s, _ = self.model.evaluate(os.path.join(self.save_dir, os.path.basename(filename)), punct=self.model.PUNCT)
      f.write(s)
    return

  
  #=============================================================
  def savefigs(self, sess, optimizer=False):
    """"""
    
    import gc
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    matdir = os.path.join(self.save_dir, 'matrices')
    if not os.path.isdir(matdir):
      os.mkdir(matdir)
    for var in self.save_vars:
      if optimizer or ('Optimizer' not in var.name):
        print(var.name)
        mat = sess.run(var)
        if len(mat.shape) == 1:
          mat = mat[None,:]
        plt.figure()
        try:
          plt.pcolor(mat, cmap='RdBu')
          plt.gca().invert_yaxis()
          plt.colorbar()
          plt.clim(vmin=-1, vmax=1)
          plt.title(var.name)
          plt.savefig(os.path.join(matdir, var.name.replace('/', '-')))
        except ValueError:
          pass
        plt.close()
        del mat
        gc.collect()
    
  #=============================================================
  def _gen_ops(self):
    """"""
    
    optimizer = optimizers.RadamOptimizer(self._config, global_step=self.global_step)
    optimizer2 = optimizers.RadamOptimizerMod(self._config, global_step=self.global_step)

    train_graph = self._model.build_graph(self._trainset)

    train_gemb_graph = self._model.build_graph(self._train_gemb_set, moving_params=optimizer)
    self._model.add_gemb_loss_graph(train_gemb_graph)
    
    train_op = optimizer.minimize(train_graph['loss'])
    train_gemb_op = optimizer2.minimize(train_gemb_graph['gemb_loss'],
        var_list=[var for var in tf.global_variables() if 'gemb/gemb_fc' in var.op.name])

    # These have to happen after optimizer.minimize is called

    # for baselines
    valid_graph = self._model.build_graph(self._validset, moving_params=optimizer)
    test_graph = self._model.build_graph(self._testset, moving_params=optimizer)

    # for gembs

    valid_get_gemb_graph = self._model.build_graph(self._valid_gemb_set, moving_params=optimizer)
    self._model.add_get_gemb_graph(valid_get_gemb_graph)
    test_get_gemb_graph = self._model.build_graph(self._test_gemb_set, moving_params=optimizer)
    self._model.add_get_gemb_graph(test_get_gemb_graph)

    valid_gemb_graph = self._model.build_test_gemb_graph(self._valid_gemb_set, moving_params=optimizer)
    test_gemb_graph = self._model.build_test_gemb_graph(self._test_gemb_set, moving_params=optimizer)

    self.train_graph = train_graph
    self.valid_graph = valid_graph
    self.test_graph = test_graph
    self.train_gemb_graph = train_gemb_graph
    self.valid_get_gemb_graph = valid_get_gemb_graph
    self.valid_gemb_graph = valid_gemb_graph
    self.test_get_gemb_graph = test_get_gemb_graph
    self.test_gemb_graph = test_gemb_graph
    
    ops = {}
    ops['train_op'] = [train_op,
                       train_graph['loss'],
                       train_graph['n_correct'],
                       train_graph['n_tokens']]
    ops['train_gemb_op'] = [train_gemb_op,
                       train_gemb_graph['gemb_loss']]
    ops['valid_op'] = [valid_graph['loss'],
                       valid_graph['n_correct'],
                       valid_graph['n_tokens'],
                       valid_graph['predictions']]
    ops['valid_gemb_op'] = [valid_gemb_graph['loss'],
                       valid_gemb_graph['n_correct'],
                       valid_gemb_graph['n_tokens'],
                       valid_gemb_graph['predictions']]
    ops['optimizer'] = optimizer

    
    
    return ops
    
  #=============================================================
  @property
  def global_step(self):
    return self._global_step
  @property
  def global_epoch(self):
    return self._global_epoch
  @property
  def model(self):
    return self._model
  @property
  def words(self):
    return self._vocabs[0]
  @property
  def tags(self):
    return self._vocabs[1]
  @property
  def rels(self):
    return self._vocabs[2]
  @property
  def ops(self):
    return self._ops
  @property
  def save_vars(self):
    return self._save_vars
  
#***************************************************************
if __name__ == '__main__':
  """"""
  
  import argparse
  
  argparser = argparse.ArgumentParser()
  argparser.add_argument('--test', action='store_true')
  argparser.add_argument('--load', action='store_true')
  argparser.add_argument('--model', default='Parser')
  argparser.add_argument('--matrix', action='store_true')
  argparser.add_argument('--train-gemb', action='store_true')
  argparser.add_argument('--test-gemb', action='store_true')
  
  args, extra_args = argparser.parse_known_args()
  cargs = {k: v for (k, v) in vars(Configurable.argparser.parse_args(extra_args)).iteritems() if v is not None}
  
  print('*** '+args.model+' ***')
  model = getattr(models, args.model)
  
  if 'save_dir' in cargs and os.path.isdir(cargs['save_dir']) and not (args.test or args.matrix or args.load):
    raw_input('Save directory already exists. Press <Enter> to overwrite or <Ctrl-C> to exit.')
  if (args.test or args.load or args.matrix) and 'save_dir' in cargs:
    cargs['config_file'] = os.path.join(cargs['save_dir'], 'config.cfg')
  network = Network(model, **cargs)
  os.system('echo Model: %s > %s/MODEL' % (network.model.__class__.__name__, network.save_dir))
  #print([v.name for v in network.save_vars])
  #config_proto.gpu_options.per_process_gpu_memory_fraction = network.per_process_gpu_memory_fraction
  config_proto = tf.ConfigProto()
  #config_proto.gpu_options.per_process_gpu_memory_fraction=0.3 # don't hog all vRAM
  config_proto.gpu_options.per_process_gpu_memory_fraction = network.per_process_gpu_memory_fraction
  with tf.Session(config=config_proto) as sess:
    sess.run(tf.global_variables_initializer())
    if not (args.test or args.test_gemb or args.matrix):
      if args.load:
        load_path = 'saves/preload_train_gemb'
        print("LOADING MODEL FROM:", load_path)
        
        saver = tf.train.Saver(var_list=network.save_vars)
        saver.restore(sess, tf.train.latest_checkpoint(load_path, latest_filename=network.name.lower()))
        if os.path.isfile(os.path.join(load_path, 'history.pkl')):
          with open(os.path.join(load_path, 'history.pkl')) as f:
            network.history = pkl.load(f)
      if args.train_gemb:
        network.train_gemb(sess)
      else:
        network.train(sess)
    elif args.matrix:
      saver = tf.train.Saver(var_list=network.save_vars)
      saver.restore(sess, tf.train.latest_checkpoint(network.save_dir, latest_filename=network.name.lower()))
      # TODO make this save pcolor plots of all matrices to a directory in save_dir
      #with tf.variable_scope('RNN0/BiRNN_FW/LSTMCell/Linear', reuse=True):
      #  pkl.dump(sess.run(tf.get_variable('Weights')), open('mat0.pkl', 'w'))
      #with tf.variable_scope('RNN1/BiRNN_FW/LSTMCell/Linear', reuse=True):
      #  pkl.dump(sess.run(tf.get_variable('Weights')), open('mat1.pkl', 'w'))
      #with tf.variable_scope('RNN2/BiRNN_FW/LSTMCell/Linear', reuse=True):
      #  pkl.dump(sess.run(tf.get_variable('Weights')), open('mat2.pkl', 'w'))
      #with tf.variable_scope('MLP/Linear', reuse=True):
      #  pkl.dump(sess.run(tf.get_variable('Weights')), open('mat3.pkl', 'w'))
      network.savefigs(sess)
    else:
      # os.system('echo Testing: >> %s/HEAD' % network.save_dir)
      # os.system('git rev-parse HEAD >> %s/HEAD' % network.save_dir)
      saver = tf.train.Saver(var_list=network.save_vars)
      #load_path = 'saves/preload_train_gemb'
      #saver.restore(sess, tf.train.latest_checkpoint(load_path, latest_filename=network.name.lower()))
      saver.restore(sess, tf.train.latest_checkpoint(network.save_dir, latest_filename=network.name.lower()))
      if args.test_gemb:
        network.test_with_gemb(sess, validate=True)
        start_time = time.time()
        network.test_with_gemb(sess, validate=False)
      else:
        network.test(sess, validate=True)
        start_time = time.time()
        network.test(sess, validate=False)
      print('Parsing took %f seconds' % (time.time() - start_time))
