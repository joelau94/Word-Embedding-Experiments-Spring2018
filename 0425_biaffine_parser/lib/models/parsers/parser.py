#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pdb

from vocab import Vocab
from lib.models.parsers.base_parser import BaseParser

#***************************************************************
class Parser(BaseParser):
  """"""
  
  #=============================================================
  def build_graph(self, dataset, moving_params=None):
    graph = {}
    
    vocabs = dataset.vocabs
    inputs = dataset.inputs
    targets = dataset.targets

    # for easy retrieval
    self.vocabs = vocabs # train/valid/dev are referring to the same vocabs object
    graph['inputs'] = inputs # input nodes are in different graph: train/valid/test
    
    reuse = tf.AUTO_REUSE
    self.tokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,0], vocabs[0].ROOT)), 2)
    self.sequence_lengths = tf.reshape(tf.reduce_sum(self.tokens_to_keep3D, [1, 2]), [-1,1])
    self.n_tokens = tf.reduce_sum(self.sequence_lengths)
    self.moving_params = moving_params
    

    word_inputs, pret_inputs = vocabs[0].embedding_lookup(inputs[:,:,0], inputs[:,:,1], moving_params=self.moving_params)
    tag_inputs = vocabs[1].embedding_lookup(inputs[:,:,2], moving_params=self.moving_params)
    if self.add_to_pretrained:
      word_inputs += pret_inputs

    graph['word_inputs'] = word_inputs

    if self.word_l2_reg > 0:
      unk_mask = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,1], vocabs[0].UNK)),2)
      word_loss = self.word_l2_reg*tf.nn.l2_loss((word_inputs - pret_inputs) * unk_mask)
    embed_inputs = self.embed_concat(word_inputs, tag_inputs)
    
    recur_states = [embed_inputs]
    for i in xrange(self.n_recur):
      with tf.variable_scope('RNN%d' % i, reuse=reuse):
        top_recur, _ = self.RNN(recur_states[-1])
        recur_states.append(top_recur)
    
    with tf.variable_scope('MLP', reuse=reuse):
      dep_mlp, head_mlp = self.MLP(recur_states[-1], self.class_mlp_size+self.attn_mlp_size, n_splits=2)
      dep_arc_mlp, dep_rel_mlp = dep_mlp[:,:,:self.attn_mlp_size], dep_mlp[:,:,self.attn_mlp_size:]
      head_arc_mlp, head_rel_mlp = head_mlp[:,:,:self.attn_mlp_size], head_mlp[:,:,self.attn_mlp_size:]
    
    with tf.variable_scope('Arcs', reuse=reuse):
      arc_logits = self.bilinear_classifier(dep_arc_mlp, head_arc_mlp)
      arc_output = self.output(arc_logits, targets[:,:,1])
      if moving_params is None:
        predictions = targets[:,:,1]
      else:
        predictions = arc_output['predictions']
    with tf.variable_scope('Rels', reuse=reuse):
      rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, len(vocabs[2]), predictions)
      rel_output = self.output(rel_logits, targets[:,:,2])
      rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond)
    
    graph['probabilities'] = tf.tuple([arc_output['probabilities'],
                                        rel_output['probabilities']])
    graph['predictions'] = tf.stack([arc_output['predictions'],
                                     rel_output['predictions']])
    graph['correct'] = arc_output['correct'] * rel_output['correct']
    graph['tokens'] = arc_output['tokens']
    graph['n_correct'] = tf.reduce_sum(graph['correct'])
    graph['n_tokens'] = self.n_tokens
    graph['accuracy'] = graph['n_correct'] / graph['n_tokens']
    graph['loss'] = arc_output['loss'] + rel_output['loss'] 
    if self.word_l2_reg > 0:
      graph['loss'] += word_loss

    graph['embed'] = embed_inputs
    graph['recur'] = recur_states
    graph['dep_arc'] = dep_arc_mlp
    graph['head_dep'] = head_arc_mlp
    graph['dep_rel'] = dep_rel_mlp
    graph['head_rel'] = head_rel_mlp
    graph['arc_logits'] = arc_logits
    graph['rel_logits'] = rel_logits

    return graph

  def add_gemb_loss_graph(self, graph):
    '''
    Build graph used for training GEMB
    recur_states_1: first layer of BiRNN, should pass in graph['recur'][1]
    '''
    oov_pos = tf.placeholder(dtype=tf.int32, shape=(None,), name='oov_pos')
    graph['oov_pos'] = oov_pos
    recur_states_1 = graph['recur'][1] # (batch, len, hidden_dim)
    recur_states_1_dimshuffle = tf.transpose(recur_states_1, [1,0,2]) # (len, batch, hidden_dim)
    #pdb.set_trace()
    # dim1 = tf.shape(recur_states_1)[0]
    # dim2 = tf.shape(recur_states_1)[1]
    # dim3 = tf.shape(recur_states_1)[2]

    # recur_states_1_reshaped = tf.reshape(recur_states_1,[dim1*dim2, dim3])
    ctx = tf.concat([tf.split(tf.gather(recur_states_1_dimshuffle, oov_pos-1), num_or_size_splits = 2, axis=-1)[0],
                    tf.split(tf.gather(recur_states_1_dimshuffle, oov_pos+1), num_or_size_splits = 2, axis=-1)[0]],
                    axis=-1)
    ctx = tf.transpose(ctx, [1,0,2])
                    
    # ctx_reshaped = tf.reshape(ctx, [-1,dim2,dim3])
    with tf.variable_scope('gemb'):
      feat = tf.contrib.layers.fully_connected(inputs=ctx,
            num_outputs=len(self.vocabs[0]._str2idx),
            activation_fn=None,
            scope='gemb_fc',
            reuse=tf.AUTO_REUSE)

    # for testing
    logits = tf.squeeze(feat)
    inputs_dimshuffle = tf.transpose(graph['inputs'], [1,0,2])
    labels = tf.one_hot(tf.squeeze(tf.split(tf.gather(inputs_dimshuffle, oov_pos),
                                            num_or_size_splits=3,
                                            axis=-1)[0]),
                          len(self.vocabs[0]._str2idx))
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    graph['gemb_loss'] = tf.reduce_mean(losses)


  def add_get_gemb_graph(self, graph):
    '''
    Build graph used for generating GEMB
    recur_states_1: first layer of BiRNN, should pass in graph['recur'][1]
    '''
    oov_pos = tf.placeholder(dtype=tf.int32, shape=(None,), name='oov_pos')
    graph['oov_pos'] = oov_pos
    embed_mat = self.vocabs[0].trainable_embeddings
    vocab_size = embed_mat.get_shape().as_list()[0]
    recur_states_1 = graph['recur'][1] # (batch=1, len, 2*hidden_dim)
    recur_states_1 = tf.squeeze(recur_states_1, axis=[0]) # (len, 2*hidden_dim)
    #pdb.set_trace()
    # dim1 = tf.shape(recur_states_1)[0]
    # dim2 = tf.shape(recur_states_1)[1]
    # dim3 = tf.shape(recur_states_1)[2]

    # recur_states_1_reshaped = tf.reshape(recur_states_1,[dim1*dim2, dim3])
    ctx = tf.concat([tf.split(tf.gather(recur_states_1, oov_pos-1), num_or_size_splits = 2, axis=-1)[0],
                    tf.split(tf.gather(recur_states_1, oov_pos+1), num_or_size_splits = 2, axis=-1)[0]],
                    axis=-1) # (oov, hidden_dim)

    with tf.variable_scope('gemb'):
      feat = tf.contrib.layers.fully_connected(inputs=ctx,
            num_outputs=len(self.vocabs[0]._str2idx),
            activation_fn=None,
            scope='gemb_fc',
            reuse=True) # (oov, vocab)

    gemb_scores = tf.nn.softmax(feat) # (oov, vocab)
    graph['gembedding'] = tf.reduce_sum(tf.expand_dims(gemb_scores, axis=-1) * embed_mat, axis=-2) # (oov, vocab, emb_dim)


  def build_test_gemb_graph(self, dataset, moving_params=None):
    graph = {}
    
    vocabs = dataset.vocabs
    inputs = dataset.inputs
    targets = dataset.targets

    # for easy retrieval
    self.vocabs = vocabs # train/valid/dev are referring to the same vocabs object
    graph['inputs'] = inputs # input nodes are in different graph: train/valid/test
    
    reuse = True
    self.tokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,0], vocabs[0].ROOT)), 2)
    self.sequence_lengths = tf.reshape(tf.reduce_sum(self.tokens_to_keep3D, [1, 2]), [-1,1])
    self.n_tokens = tf.reduce_sum(self.sequence_lengths)
    self.moving_params = moving_params
    

    word_inputs, pret_inputs = vocabs[0].embedding_lookup(inputs[:,:,0], inputs[:,:,1], moving_params=self.moving_params)
    tag_inputs = vocabs[1].embedding_lookup(inputs[:,:,2], moving_params=self.moving_params)
    if self.add_to_pretrained:
      word_inputs += pret_inputs

    graph['gembedding_new'] = tf.placeholder(dtype=tf.float32, shape=(None,None,vocabs[0].embed_size), name='gembedding_new')
    word_inputs = graph['gembedding_new']

    if self.word_l2_reg > 0:
      unk_mask = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,1], vocabs[0].UNK)),2)
      word_loss = self.word_l2_reg*tf.nn.l2_loss((word_inputs - pret_inputs) * unk_mask)
    embed_inputs = self.embed_concat(word_inputs, tag_inputs)
    
    recur_states = [embed_inputs]
    for i in xrange(self.n_recur):
      with tf.variable_scope('RNN%d' % i, reuse=reuse):
        top_recur, _ = self.RNN(recur_states[-1])
        recur_states.append(top_recur)
    
    with tf.variable_scope('MLP', reuse=reuse):
      dep_mlp, head_mlp = self.MLP(recur_states[-1], self.class_mlp_size+self.attn_mlp_size, n_splits=2)
      dep_arc_mlp, dep_rel_mlp = dep_mlp[:,:,:self.attn_mlp_size], dep_mlp[:,:,self.attn_mlp_size:]
      head_arc_mlp, head_rel_mlp = head_mlp[:,:,:self.attn_mlp_size], head_mlp[:,:,self.attn_mlp_size:]
    
    with tf.variable_scope('Arcs', reuse=reuse):
      arc_logits = self.bilinear_classifier(dep_arc_mlp, head_arc_mlp)
      arc_output = self.output(arc_logits, targets[:,:,1], gemb=True)
      if moving_params is None:
        predictions = targets[:,:,1]
      else:
        predictions = arc_output['predictions']
    with tf.variable_scope('Rels', reuse=reuse):
      rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, len(vocabs[2]), predictions)
      rel_logits = tf.squeeze(rel_logits)
      rel_output = self.output(rel_logits, targets[:,:,2], gemb=True)
      rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond)
    
    graph['probabilities'] = tf.tuple([arc_output['probabilities'],
                                        rel_output['probabilities']])
    graph['predictions'] = tf.stack([arc_output['predictions'],
                                     rel_output['predictions']])
    graph['correct'] = arc_output['correct'] * rel_output['correct']
    graph['tokens'] = arc_output['tokens']
    graph['n_correct'] = tf.reduce_sum(graph['correct'])
    graph['n_tokens'] = self.n_tokens
    graph['accuracy'] = graph['n_correct'] / graph['n_tokens']
    graph['loss'] = arc_output['loss'] + rel_output['loss'] 
    if self.word_l2_reg > 0:
      graph['loss'] += word_loss

    graph['embed'] = embed_inputs
    graph['recur'] = recur_states
    graph['dep_arc'] = dep_arc_mlp
    graph['head_dep'] = head_arc_mlp
    graph['dep_rel'] = dep_rel_mlp
    graph['head_rel'] = head_rel_mlp
    graph['arc_logits'] = arc_logits
    graph['rel_logits'] = rel_logits

    return graph

  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""
    
    parse_preds = self.parse_argmax(parse_probs, tokens_to_keep)
    rel_probs = rel_probs[np.arange(len(parse_preds)), parse_preds]
    rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
    return parse_preds, rel_preds
