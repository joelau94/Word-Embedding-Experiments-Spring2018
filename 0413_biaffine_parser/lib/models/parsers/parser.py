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
  def __call__(self, dataset, moving_params=None):
    """"""
    
    vocabs = dataset.vocabs
    inputs = dataset.inputs
    targets = dataset.targets

    # for easy retrieval
    self.vocabs = vocabs
    self.inputs = inputs
    
    reuse = (moving_params is not None)
    self.tokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,0], vocabs[0].ROOT)), 2)
    self.sequence_lengths = tf.reshape(tf.reduce_sum(self.tokens_to_keep3D, [1, 2]), [-1,1])
    self.n_tokens = tf.reduce_sum(self.sequence_lengths)
    self.moving_params = moving_params
    

    word_inputs, pret_inputs = vocabs[0].embedding_lookup(inputs[:,:,0], inputs[:,:,1], moving_params=self.moving_params)
    print("Word inputs : ", inputs)
    tag_inputs = vocabs[1].embedding_lookup(inputs[:,:,2], moving_params=self.moving_params)
    if self.add_to_pretrained:
      word_inputs += pret_inputs

    self.word_inputs = word_inputs # for easy retrieval and replacement

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
    
    output = {}
    output['probabilities'] = tf.tuple([arc_output['probabilities'],
                                        rel_output['probabilities']])
    output['predictions'] = tf.stack([arc_output['predictions'],
                                     rel_output['predictions']])
    output['correct'] = arc_output['correct'] * rel_output['correct']
    output['tokens'] = arc_output['tokens']
    output['n_correct'] = tf.reduce_sum(output['correct'])
    output['n_tokens'] = self.n_tokens
    output['accuracy'] = output['n_correct'] / output['n_tokens']
    output['loss'] = arc_output['loss'] + rel_output['loss'] 
    if self.word_l2_reg > 0:
      output['loss'] += word_loss
    
    output['embed'] = embed_inputs
    output['recur'] = recur_states
    output['dep_arc'] = dep_arc_mlp
    output['head_dep'] = head_arc_mlp
    output['dep_rel'] = dep_rel_mlp
    output['head_rel'] = head_rel_mlp
    output['arc_logits'] = arc_logits
    output['rel_logits'] = rel_logits

    self.output_val = output # for easy retrieval
    return output

  def gemb_graph(self):
    '''
    Build graph used for generating GEMB
    recur_states_1: first layer of BiRNN, should pass in output['recur'][1]
    '''
    self.oov_pos = tf.placeholder(dtype=tf.int32, shape=(None), name='oov_pos')
    embed_mat = self.vocabs[0].trainable_embeddings
    vocab_size = embed_mat.get_shape().as_list()[0]
    recur_states_1 = self.output_val['recur'][1]
    #pdb.set_trace()
    dim1 = tf.shape(recur_states_1)[0]
    dim2 = tf.shape(recur_states_1)[1]
    dim3 = tf.shape(recur_states_1)[2]

    recur_states_1_reshaped = tf.reshape(recur_states_1,[dim1*dim2, dim3])
    #recur_states_1_reshaped[self.oov_pos-1]
    
    ctx = tf.concat([tf.split(recur_states_1_reshaped[self.oov_pos-1], num_or_size_splits = 2, axis=-1)[0],
                    tf.split(recur_states_1_reshaped[self.oov_pos+1], num_or_size_splits = 2, axis=-1)[0]],
                    axis=-1)
                    
    ctx_reshaped = tf.reshape(ctx, tf.shape(recur_states_1))                
    with tf.variable_scope('gemb'):
      feat = tf.contrib.layers.fully_connected(inputs=ctx_reshaped,
            num_outputs=len(self.vocabs[0]._str2idx),
            activation_fn=None,
            scope='gemb_fc')

    # for testing
    self.gemb_scores = tf.nn.softmax(feat)
    self.gembedding = tf.reduce_sum(tf.expand_dims(self.gemb_scores, axis=-1) * embed_mat, axis=-2)

    # for training
    # print(tf.shape(feat)) # (3,)
    # print(feat.get_shape()) # (?, ?, 21679)
    # print(tf.shape(tf.squeeze(feat)))
    # print(feat.get_shape())
    # print(tf.shape(tf.one_hot(self.inputs[self.oov_pos,:,0], len(self.vocabs[0]._str2idx))))
    self.gemb_loss = tf.nn.softmax_cross_entropy_with_logits(logits = tf.transpose(tf.squeeze(feat)), labels = tf.one_hot(self.inputs[self.oov_pos,:,0], len(self.vocabs[0]._str2idx)))

  def insert_gemb_graph(self):
    '''
    Insert input node of gemb
    '''
    self.gembedding_new = tf.placeholder(dtype=tf.float32, shape=(None,None,None), name='gembedding_new')
    self.word_inputs = self.gembedding_new
  
  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""
    
    parse_preds = self.parse_argmax(parse_probs, tokens_to_keep)
    rel_probs = rel_probs[np.arange(len(parse_preds)), parse_preds]
    rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
    return parse_preds, rel_preds
