from optimizer import *
from layer import *
from gemb import *

from collections import OrderedDict
import itertools
import numpy
import theano
import theano.tensor as tensor

class BiLSTMTaggerModel(object):
  """ Constructs the network and builds the following Theano functions:
      - pred_function: Takes input and mask, returns prediction.
      - loss_function: Takes input, mask and labels, returns the final cross entropy loss (scalar).
  """
  def __init__(self, data, config, fast_predict=False):
    self.embedding_shapes = data.embedding_shapes;
    self.lstm_type = config.lstm_cell  
    self.lstm_hidden_size = int(config.lstm_hidden_size)
    self.num_lstm_layers = int(config.num_lstm_layers)
    self.max_grad_norm = float(config.max_grad_norm)

    self.vocab_size = data.word_dict.size()
    self.label_space_size = data.label_dict.size()
    self.unk_id = data.unk_id
    
    # Initialize layers and parameters
    self.embedding_layer = EmbeddingLayer(data.embedding_shapes, data.embeddings)    
    self.params = [p for p in self.embedding_layer.params]

    self.rnn_layers = [None] * self.num_lstm_layers
    for l in range(self.num_lstm_layers):
      input_dim = self.embedding_layer.output_size if l == 0 else self.lstm_hidden_size
      input_dropout = config.input_dropout_prob if (config.per_layer_dropout or l == 0) else 0.0
      recurrent_dropout = config.recurrent_dropout_prob
      
      self.rnn_layers[l] = get_rnn_layer(self.lstm_type)(input_dim,
                                 self.lstm_hidden_size,
                                 input_dropout_prob=input_dropout,
                                 recurrent_dropout_prob=recurrent_dropout,
                                 fast_predict=fast_predict,
                                 prefix='lstm_{}'.format(l))
      print (self.rnn_layers[l])
      self.params.extend(self.rnn_layers[l].params)
    
    self.softmax_layer = SoftmaxLayer(self.lstm_hidden_size, self.label_space_size)
    self.params.extend(self.softmax_layer.params)
    
    # Build model
    # Shape of x: [seq_len, batch_size, num_features]
    self.x0 = tensor.ltensor3('x')
    self.y0 = tensor.lmatrix('y')
    self.mask0 = tensor.matrix('mask', dtype=floatX)
    self.is_train = tensor.bscalar('is_train')
    
    self.x = self.x0.dimshuffle(1, 0, 2)
    self.y = self.y0.dimshuffle(1, 0)
    self.mask = self.mask0.dimshuffle(1, 0) 
    
    self.inputs = [None] * (self.num_lstm_layers + 1)
    self.inputs[0] = self.embedding_layer.connect(self.x)
    self.rev_mask = self.mask[::-1]
    
    for l, rnn in enumerate(self.rnn_layers):
      outputs = rnn.connect(self.inputs[l],
                  self.mask if l % 2 == 0 else self.rev_mask,
                  self.is_train)
      self.inputs[l+1] = outputs[::-1]
     
    self.scores, self.pred = self.softmax_layer.connect(self.inputs[-1])
    self.pred0 = self.pred.reshape([self.mask.shape[0], self.mask.shape[1]]).dimshuffle(1, 0)

  def add_ctx_gemb(self):
    # embedding_shapes[0]: word embedding shape; embedding_shapes[0][1]: num_words
    self.gemb = GembModel(self.lstm_hidden_size*2, self.embedding_layer.embedding_shapes[0][0])
    print('Vocab Size: {}'.format(self.embedding_layer.embedding_shapes[0][0]))

  def add_mix_gemb(self, char_vocab_size, char_embed_dim, char_hidden_dim):
    # embedding_shapes[0]: word embedding shape; embedding_shapes[0][1]: num_words
    self.gemb = GembModel(self.lstm_hidden_size*2 + char_hidden_dim*2, self.embedding_layer.embedding_shapes[0][0])
    self.gemb.add_char(char_vocab_size, char_embed_dim, char_hidden_dim)
    print('Vocab Size: {}'.format(self.embedding_layer.embedding_shapes[0][0]))

  def add_char_gemb(self, char_vocab_size, char_embed_dim, char_hidden_dim):
    # embedding_shapes[0]: word embedding shape; embedding_shapes[0][1]: num_words
    self.gemb = GembModel(char_hidden_dim*2, self.embedding_layer.embedding_shapes[0][0])
    self.gemb.add_char(char_vocab_size, char_embed_dim, char_hidden_dim)
    print('Vocab Size: {}'.format(self.embedding_layer.embedding_shapes[0][0]))

  '''Context GEMB'''

  def get_ctx_emb_function(self):
    """
    Return embeddings, with OOVs replaced by context-estimation
    Used at test time
    """
    oov_pos = tensor.lvector('oov_pos_pred')

    self.inputs = [None] * (self.num_lstm_layers + 1)
    self.inputs[0] = self.embedding_layer.connect(self.x)
    self.rev_mask = self.mask[::-1]
    
    for l, rnn in enumerate(self.rnn_layers):
      outputs = rnn.connect(self.inputs[l],
                  self.mask if l % 2 == 0 else self.rev_mask,
                  self.is_train)
      self.inputs[l+1] = outputs[::-1]

    fw_states = self.inputs[1] # (sent_len, batch, hidden_dim)
    bw_states = self.inputs[2] # (sent_len, batch, hidden_dim)
    emb_mat = self.embedding_layer.embeddings[0]

    feat = self.gemb.mlp.connect(tensor.concatenate([fw_states[oov_pos-1,:,:], bw_states[oov_pos+1,:,:]], axis=-1)) # (oov_num, batch, num_words)
    probs = tensor.nnet.softmax(feat.reshape([feat.shape[0]*feat.shape[1], feat.shape[2]])) # (oov_num*batch, num_words)
    emb_reweight = probs.dimshuffle(0,1,'x') * emb_mat # (oov_num*batch, num_words, emb_dim)
    gembedding = emb_reweight.sum(axis=1).reshape([feat.shape[0], feat.shape[1], -1]) # ??? (oov_num, batch, emb_dim)

    return theano.function([self.x0, self.mask0, oov_pos], [gembedding, self.inputs[0]],
                name='ctx_gemb_pred',
                on_unused_input='warn',
                givens=({self.is_train: numpy.cast['int8'](1)}))
    # done for now

  def get_ctx_gemb_loss_function(self):
    oov_pos = tensor.lvector('oov_pos')

    oov_pos_x = oov_pos.flatten()
    oov_pos_y = tensor.arange(oov_pos_x.shape[0])

    fw_states = self.inputs[1] # (sent_len, batch, hidden_dim)
    bw_states = self.inputs[2] # (sent_len, batch, hidden_dim)
    emb_mat = self.embedding_layer.embeddings[0]

    preact = tensor.concatenate([fw_states[oov_pos_x-1,oov_pos_y,:], bw_states[oov_pos_x+1,oov_pos_y,:]], axis=-1)
    feat = self.gemb.mlp.connect(preact) # (oov_num, batch, num_words)
    
    # Bug: reshapre requires the parameters to be integers.
    #   How do we evaliuate "feat.shape" to get ints?
    probs = tensor.nnet.softmax(feat) # (oov_num*batch, num_words) oov_num=1 fixed
    log_probs = tensor.log(probs)

    loss = CrossEntropyLoss().connect(inputs=log_probs, weights=None, labels=self.x[oov_pos_x,oov_pos_y,0].reshape([-1,1]))

    grads = gradient_clipping(tensor.grad(loss, self.gemb.params),
                  self.max_grad_norm)
    updates = adadelta(self.gemb.params, grads)

    return theano.function([self.x0, self.mask0, oov_pos], loss,
                 name='f_ctx_gemb_loss',
                 updates=updates,
                 on_unused_input='warn',
                 givens=({self.is_train: numpy.cast['int8'](1)}))

  '''Mix GEMB'''

  def get_mix_emb_function(self):
    """
    Return embeddings, with OOVs replaced by context-estimation
    Used at test time
    """
    oov_pos = tensor.lvector('oov_pos_pred')
    oov_char = tensor.lmatrix('oov_char_pred')
    rnn_mask = tensor.matrix('rnn_mask_pred', dtype=floatX)

    self.inputs = [None] * (self.num_lstm_layers + 1)
    self.inputs[0] = self.embedding_layer.connect(self.x)
    self.rev_mask = self.mask[::-1]
    
    for l, rnn in enumerate(self.rnn_layers):
      outputs = rnn.connect(self.inputs[l],
                  self.mask if l % 2 == 0 else self.rev_mask,
                  self.is_train)
      self.inputs[l+1] = outputs[::-1]

    fw_states = self.inputs[1] # (sent_len, batch=1, hidden_dim)
    bw_states = self.inputs[2] # (sent_len, batch=1, hidden_dim)
    emb_mat = self.embedding_layer.embeddings[0]

    ctx_preact = tensor.concatenate([fw_states[oov_pos-1,:,:], bw_states[oov_pos+1,:,:]], axis=-1)
    # (oov_num, batch=1, 2*hidden_dim)

    char_states = self.gemb.char_rnn.connect(oov_char, rnn_mask) # (oov_num, 2*char_hidden_dim)
    char_preact = char_states.dimshuffle((0,'x',1))

    preact = tensor.concatenate([ctx_preact, char_preact], axis=-1)

    feat = self.gemb.mlp.connect(preact) # (oov_num, batch=1, num_words)
    probs = tensor.nnet.softmax(feat.reshape([feat.shape[0]*feat.shape[1], feat.shape[2]])) # (oov_num*batch, num_words)
    emb_reweight = probs.dimshuffle(0,1,'x') * emb_mat # (oov_num*batch, num_words, emb_dim)
    gembedding = emb_reweight.sum(axis=1).reshape([feat.shape[0], feat.shape[1], -1]) # ??? (oov_num, batch, emb_dim)

    return theano.function([self.x0, self.mask0, oov_pos, oov_char, rnn_mask], [gembedding, self.inputs[0]],
                name='mix_gemb_pred',
                on_unused_input='warn',
                givens=({self.is_train: numpy.cast['int8'](1)}))
    # done for now

  def get_mix_gemb_loss_function(self):
    oov_pos = tensor.lvector('oov_pos')
    oov_char = tensor.lmatrix('oov_char')
    rnn_mask = tensor.matrix('rnn_mask', dtype=floatX)

    oov_pos_x = oov_pos.flatten()
    oov_pos_y = tensor.arange(oov_pos_x.shape[0])

    fw_states = self.inputs[1] # (sent_len, batch, hidden_dim)
    bw_states = self.inputs[2] # (sent_len, batch, hidden_dim)
    emb_mat = self.embedding_layer.embeddings[0]

    ctx_preact = tensor.concatenate([fw_states[oov_pos_x-1,oov_pos_y,:], bw_states[oov_pos_x+1,oov_pos_y,:]], axis=-1)
    # (batch, 2*hidden_dim)

    char_states = self.gemb.char_rnn.connect(oov_char, rnn_mask) # (batch, 2*char_hidden_dim)
    char_preact = char_states

    preact = tensor.concatenate([ctx_preact, char_preact], axis=-1)

    feat = self.gemb.mlp.connect(preact) # (batch, num_words)

    probs = tensor.nnet.softmax(feat) # (oov_num*batch, num_words) oov_num=1 fixed
    log_probs = tensor.log(probs)

    loss = CrossEntropyLoss().connect(inputs=log_probs, weights=None, labels=self.x[oov_pos_x,oov_pos_y,0].reshape([-1,1]))

    grads = gradient_clipping(tensor.grad(loss, self.gemb.params),
                  self.max_grad_norm)
    updates = adadelta(self.gemb.params, grads)

    return theano.function([self.x0, self.mask0, oov_pos, oov_char, rnn_mask], loss,
                 name='f_mix_gemb_loss',
                 updates=updates,
                 on_unused_input='warn',
                 givens=({self.is_train: numpy.cast['int8'](1)}))

  '''Char GEMB'''

  def get_char_emb_function(self):
    """
    Return embeddings, with OOVs replaced by context-estimation
    Used at test time
    """
    oov_char = tensor.lmatrix('oov_char_pred')
    rnn_mask = tensor.matrix('rnn_mask_pred', dtype=floatX)

    self.inputs = [None] * (self.num_lstm_layers + 1)
    self.inputs[0] = self.embedding_layer.connect(self.x)
    self.rev_mask = self.mask[::-1]

    emb_mat = self.embedding_layer.embeddings[0]
    char_states = self.gemb.char_rnn.connect(oov_char, rnn_mask) # (oov_num, 2*char_hidden_dim)
    char_preact = char_states.dimshuffle((0,'x',1))

    feat = self.gemb.mlp.connect(char_preact) # (oov_num, batch=1, num_words)
    probs = tensor.nnet.softmax(feat.reshape([feat.shape[0]*feat.shape[1], feat.shape[2]])) # (oov_num*batch, num_words)
    emb_reweight = probs.dimshuffle(0,1,'x') * emb_mat # (oov_num*batch, num_words, emb_dim)
    gembedding = emb_reweight.sum(axis=1).reshape([feat.shape[0], feat.shape[1], -1]) # ??? (oov_num, batch, emb_dim)

    return theano.function([self.x0, self.mask0, oov_char, rnn_mask], [gembedding, self.inputs[0]],
                name='char_gemb_pred',
                on_unused_input='warn',
                givens=({self.is_train: numpy.cast['int8'](1)}))
    # done for now

  def get_char_gemb_loss_function(self):
    oov_pos = tensor.lvector('oov_pos')
    oov_char = tensor.lmatrix('oov_char')
    rnn_mask = tensor.matrix('rnn_mask', dtype=floatX)

    oov_pos_x = oov_pos.flatten()
    oov_pos_y = tensor.arange(oov_pos_x.shape[0])

    emb_mat = self.embedding_layer.embeddings[0]

    char_states = self.gemb.char_rnn.connect(oov_char, rnn_mask) # (batch, 2*char_hidden_dim)
    char_preact = char_states

    feat = self.gemb.mlp.connect(char_preact) # (batch, num_words)

    probs = tensor.nnet.softmax(feat) # (oov_num*batch, num_words) oov_num=1 fixed
    log_probs = tensor.log(probs)

    loss = CrossEntropyLoss().connect(inputs=log_probs, weights=None, labels=self.x[oov_pos_x,oov_pos_y,0].reshape([-1,1]))

    grads = gradient_clipping(tensor.grad(loss, self.gemb.params),
                  self.max_grad_norm)
    updates = adadelta(self.gemb.params, grads)

    return theano.function([self.x0, self.mask0, oov_pos, oov_char, rnn_mask], loss,
                 name='f_char_gemb_loss',
                 updates=updates,
                 on_unused_input='warn',
                 givens=({self.is_train: numpy.cast['int8'](1)}))

  '''GEMB testing'''
  def get_distribution_by_gemb_function(self):
    """ Return predictions and scores of shape [batch_size, time_steps, label space size].
        Used at test time.
    """
    inputs_0 = tensor.ftensor3('inputs_0')

    self.inputs = [None] * (self.num_lstm_layers + 1)
    self.inputs[0] = inputs_0
    self.rev_mask = self.mask[::-1]
    
    for l, rnn in enumerate(self.rnn_layers):
      outputs = rnn.connect(self.inputs[l],
                  self.mask if l % 2 == 0 else self.rev_mask,
                  self.is_train)
      self.inputs[l+1] = outputs[::-1]

    self.scores, self.pred = self.softmax_layer.connect(self.inputs[-1])
    self.pred0 = self.pred.reshape([self.mask.shape[0], self.mask.shape[1]]).dimshuffle(1, 0)

     # (sent_len, batch_size, label_space_size) --> (batch_size, sent_len, label_space_size)
    scores0 = self.scores.reshape([self.inputs[0].shape[0], self.inputs[0].shape[1],
                     self.label_space_size]).dimshuffle(1, 0, 2)
                      
    return theano.function([inputs_0, self.mask0], [self.pred0, scores0],
                 name='f_ctx_gemb_pred',
                 allow_input_downcast=True,
                 on_unused_input='warn',
                 givens=({self.is_train:  numpy.cast['int8'](0)}))

  '''GEMB validation'''
  def get_eval_with_gemb_function(self):
    inputs_0 = tensor.ftensor3('inputs_0')

    self.inputs = [None] * (self.num_lstm_layers + 1)
    self.inputs[0] = inputs_0
    self.rev_mask = self.mask[::-1]
    
    for l, rnn in enumerate(self.rnn_layers):
      outputs = rnn.connect(self.inputs[l],
                  self.mask if l % 2 == 0 else self.rev_mask,
                  self.is_train)
      self.inputs[l+1] = outputs[::-1]

    self.scores, self.pred = self.softmax_layer.connect(self.inputs[-1])
    self.pred0 = self.pred.reshape([self.mask.shape[0], self.mask.shape[1]]).dimshuffle(1, 0)

     # (sent_len, batch_size, label_space_size) --> (batch_size, sent_len, label_space_size)
    scores0 = self.scores.reshape([self.inputs[0].shape[0], self.inputs[0].shape[1],
                     self.label_space_size]).dimshuffle(1, 0, 2)

    loss = CrossEntropyLoss().connect(self.scores, self.mask, self.y)
    return theano.function([inputs_0, self.mask0, self.y0], [self.pred0, loss],
                 name='f_gemb_eval',
                 allow_input_downcast=True,
                 on_unused_input='warn',
                 givens=({self.is_train:  numpy.cast['int8'](0)}))

  '''Original tagger'''

  def get_eval_function(self):  
    """ We should feed in non-dimshuffled inputs x0, mask0 and y0.
        Used for tracking Dev loss at training time.
    """
    loss = CrossEntropyLoss().connect(self.scores, self.mask, self.y)
    return theano.function([self.x0, self.mask0, self.y0], [self.pred0, loss],
                 name='f_eval',
                 allow_input_downcast=True,
                 on_unused_input='warn',
                 givens=({self.is_train:  numpy.cast['int8'](0)}))
    
  def get_distribution_function(self):
    """ Return predictions and scores of shape [batch_size, time_steps, label space size].
        Used at test time.
    """
    scores0 = self.scores.reshape([self.x.shape[0], self.x.shape[1],
                     self.label_space_size]).dimshuffle(1, 0, 2)
                      
    return theano.function([self.x0, self.mask0], [self.pred0, scores0],
                 name='f_pred',
                 allow_input_downcast=True,
                 on_unused_input='warn',
                 givens=({self.is_train:  numpy.cast['int8'](0)}))
  
  def get_loss_function(self):
    """ We should feed in non-dimshuffled inputs x0, mask0 and y0.
    """
    loss = CrossEntropyLoss().connect(self.scores, self.mask, self.y)
    grads = gradient_clipping(tensor.grad(loss, self.params),
                  self.max_grad_norm)
    updates = adadelta(self.params, grads)

    return theano.function([self.x0, self.mask0, self.y0], loss,
                 name='f_loss',
                 updates=updates,
                 on_unused_input='warn',
                 givens=({self.is_train: numpy.cast['int8'](1)}))
  
  def save(self, filepath):
    """ Save model parameters to file.
    """
    all_params = OrderedDict([(param.name, param.get_value()) for param in self.params])
    numpy.savez(filepath, **all_params)
    print('Saved model to: {}'.format(filepath))

  def load(self, filepath):
    """ Load model parameters from file.
    """
    all_params = numpy.load(filepath)
    for param in self.params:
      if param.name in all_params:
        vals = all_params[param.name]
        if param.name.startswith('embedding') and self.embedding_shapes[0][0] > vals.shape[0]:
          # Expand to new vocabulary.
          print self.embedding_shapes[0][0], vals.shape[0]
          new_vals = numpy.concatenate((vals, param.get_value()[vals.shape[0]:, :]), axis=0)
          param.set_value(new_vals)
        else:
          param.set_value(vals)
    print('Loaded model from: {}'.format(filepath))
