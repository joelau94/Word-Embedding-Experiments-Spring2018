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

import tensorflow as tf

from lib.optimizers.base_optimizer import BaseOptimizer

#***************************************************************
class RadamOptimizerMod(BaseOptimizer):
  """"""
  def minimize(self, loss, var_list, name=None):
    """"""
    
    # Error checking
    #var_list = tf.trainable_variables()
    for x_tm1 in var_list:
      if not isinstance(x_tm1, tf.Variable):
        raise TypeError("Argument is not a tf.Variable: %s" % x_tm1)
    if not var_list:
      raise ValueError("No variables to optimize")
    if loss.dtype.base_dtype != tf.float32:
      raise ValueError('Loss is not float32')
    
    # Compute gradients
    var_refs = [x_tm1._ref() for x_tm1 in var_list]
    grads = tf.gradients(loss, var_refs,
                                colocate_gradients_with_ops=True,
                                gate_gradients=True,
                                aggregation_method=2)
    for x_tm1, g_t in zip(var_list, grads):
      if g_t is not None:
        if x_tm1.dtype.base_dtype != tf.float32:
          raise ValueError('%s is not float32' % x_tm1.name)

    # Apply gradients
    with tf.control_dependencies(None):
      self._init_acc(var_list, grads)
    with tf.name_scope(values=[], name=name, default_name=self._name) as name:
      caches = filter(lambda cache: cache['g_t'] is not None, self._prepare(var_list, grads))
      for cache in caches:
        x_tm1, g_t = cache['x_tm1'], cache['g_t']
        with tf.name_scope("update_" + x_tm1.op.name), tf.device(x_tm1.device):
          if isinstance(g_t, tf.Tensor):
            cache['g_t'] = tf.where(tf.is_finite(g_t), g_t, tf.zeros_like(g_t))
            self._apply_dense(cache)
          else:
            cache['g_t'] = tf.where(tf.is_finite(g_t.values), g_t.values, tf.zeros_like(g_t.values))
            cache['idxs'] = g_t.indices
            self._apply_sparse(cache)
      with tf.control_dependencies([self._finish(caches)]):
        with tf.device(self.global_step.device):
          return tf.assign_add(self.global_step, 1, name=name).op

  #=============================================================
  def _init_acc(self, var_list, grads):
    """"""
    
    super(RadamOptimizerMod, self)._init_acc(var_list, grads)
    for x_tm1, g_t in zip(var_list, grads):
      if self.mu > 0:
        self.get_accumulator(x_tm1, 'm')
        shape = self.get_variable_shape(x_tm1)
        if isinstance(g_t, tf.Tensor):
          self.get_accumulator(x_tm1, 'm/tm1', [])
        else:
          self.get_accumulator(x_tm1, 'm/tm1', [shape[0]]+[1]*(len(shape)-1))
      if self.nu > 0:
        self.get_accumulator(x_tm1, 'v')
        shape = self.get_variable_shape(x_tm1)
        if isinstance(g_t, tf.Tensor):
          self.get_accumulator(x_tm1, 'v/tm1', [])
        else:
          self.get_accumulator(x_tm1, 'v/tm1', [shape[0]]+[1]*(len(shape)-1))
    return
  
  #=============================================================
  def _apply_dense(self, cache):
    """"""
    
    x_tm1, g_t = cache['x_tm1'], cache['g_t']
    updates = cache['updates']
    
    if self.mu > 0:
      m_t, t_m = self._dense_moving_average(x_tm1, g_t, 'm', beta=self.mu)
      m_bar_t = (1-self.gamma) * m_t + self.gamma * g_t
      updates.extend([m_t, t_m])
    else:
      m_bar_t = g_t
    
    if self.nu > 0:
      v_t, t_v = self._dense_moving_average(x_tm1, g_t**2, 'v', beta=self.nu)
      v_bar_t = tf.sqrt(v_t + self.epsilon)
      updates.extend([v_t, t_v])
    else:
      v_bar_t = 1
    
    s_t = self.learning_rate * m_bar_t / v_bar_t
    cache['s_t'] = s_t
    return cache
  
  #=============================================================
  def _apply_sparse(self, cache):
    """"""
    
    x_tm1, g_t, idxs = cache['x_tm1'], cache['g_t'], cache['idxs']
    idxs, idxs_ = tf.unique(idxs)
    g_t_ = tf.unsorted_segment_sum(g_t, idxs_, tf.size(idxs))
    updates = cache['updates']
    
    if self.mu > 0:
      m_t, t_m = self._sparse_moving_average(x_tm1, idxs, g_t_, 'm', beta=self.mu)
      m_t_ = tf.gather(m_t, idxs)
      m_bar_t_ = (1-self.gamma) * m_t_ + self.gamma * g_t_
      updates.extend([m_t, t_m])
    else:
      m_bar_t_ = g_t_
    
    if self.nu > 0:
      v_t, t_v = self._sparse_moving_average(x_tm1, idxs, g_t_**2, 'v', beta=self.nu)
      v_t_ = tf.gather(v_t, idxs)
      v_bar_t_ = tf.sqrt(v_t_ + self.epsilon)
      updates.extend([v_t, t_v])
    else:
      v_bar_t_ = 1
    
    s_t_ = self.learning_rate * m_bar_t_ / v_bar_t_
    cache['s_t'] = s_t_
    cache['g_t'] = g_t_
    cache['idxs'] = idxs
    return cache
  
