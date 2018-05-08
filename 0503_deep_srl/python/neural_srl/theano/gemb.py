from layer import *
from util import *
import numpy as np
from collections import OrderedDict


def _p(pp, name):
  return '%s_%s' % (pp, name)


def split(_x, n):
    # only support 3d and 2d tensors
    input_size = _x.shape[-1]
    output_size = input_size/n
    output = []
    if _x.ndim == 3:
        for i in range(n):
            output.append(_x[:, :, i * output_size : (i+1) * output_size])
        return output
    else:
        for i in range(n):
            output.append(_x[:, i * output_size : (i+1) * output_size])
        return output

class MLPLayer(object):
    def __init__(self, input_dim, output_dim, prefix='MLP', use_bias=True):
        self.W = get_variable(_p(prefix, 'W'), [input_dim, output_dim],
                          random_normal_initializer(0.0, 0.01))
        self.params = [self.W]
        self.use_bias = use_bias
        if self.use_bias:
            self.b = get_variable(_p(prefix, 'b'), [output_dim],
                                all_zero_initializer())
            self.params += [self.b]

    def connect(self, inputs):
        output = tensor.dot(inputs, self.W)
        if self.use_bias:
            output += self.b
        return output

class EmbeddingLookup(object):

    def __init__(self, vocab_size, input_dim, output_dim, prefix='CharEmbed'):
        self.params = []
        self.W = get_variable(_p(prefix, 'W'), [input_dim, output_dim],
                          random_normal_initializer(0.0, 0.01))
        self.b = get_variable(_p(prefix, 'b'), [output_dim],
                              all_zero_initializer())
        self.params = [self.W, self.b]

    def connect(self, index):
        return self.W[index] + self.b


class UniGruEncoder(object):

    def __init__(self, embedding_dim, hidden_dim, prefix='GRU'):
        self.params = []
        self.hidden_dim = hidden_dim

        self.W_hzr = MLPLayer(embedding_dim, 3*hidden_dim, prefix=prefix+'_W_hzr')
        self.params += self.W_hzr.params

        self.U_zr = MLPLayer(hidden_dim, 2*hidden_dim, prefix=prefix+'_U_zr', use_bias=False)
        self.params += self.U_zr.params

        self.U_h = MLPLayer(hidden_dim, hidden_dim, prefix=prefix+'_U_h', use_bias=False)
        self.params += self.U_h.params

    def step(self, weighted_inputs, prev_h, mask=None):

        h_input, z_input, r_input = split(weighted_inputs, 3)

        z_hidden, r_hidden = split(self.U_zr.connect(prev_h), 2)

        z = tensor.nnet.sigmoid(z_input + z_hidden)
        r = tensor.nnet.sigmoid(r_input + r_hidden)

        h_hidden = self.U_h.connect(r*prev_h)

        proposed_h = tensor.tanh(h_input + h_hidden)

        h = (1.-z) * prev_h + z * proposed_h

        if mask is not None:
            mask = mask.dimshuffle(0, 'x')
            return mask * h + (1.-mask) * prev_h
        else:
            return h

    def connect(self, inputs, sent_len, init_state=None, batch_size=1, mask=None):

        init_state = tensor.zeros((batch_size, self.hidden_dim), dtype='float32')
        # init_state = TT.alloc(np.float32(0.), batch_size, self.hidden_dim)

        weighted_inputs = self.W_hzr.connect(inputs).reshape( (sent_len, batch_size, 3*self.hidden_dim) )

        if mask is not None:
            sequences = [weighted_inputs, mask]
            fn = lambda x, m, h : self.step(x, h, mask=m)
        else:
            sequences = [weighted_inputs]
            fn = lambda x, h : self.step(x, h)

        results, updates = theano.scan(fn,
                            sequences=sequences,
                            outputs_info=[init_state])

        return results


class BiGruEncoder(object): # TODO: start from here

    def __init__(self, vocab_size, embedding_dim, hidden_dim, prefix='BiGRU'):
        super(BiGruEncoder, self).__init__()
        self.params = []
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedder = EmbeddingLookup(self.vocab_size, self.embedding_dim, self.hidden_dim, prefix=prefix+'_embedder')
        self.params += self.embedder.params

        self.forward_gru = UniGruEncoder(self.embedding_dim, self.hidden_dim, prefix=prefix+'_forward_gru')
        self.params += self.forward_gru.params

        self.backward_gru = UniGruEncoder(self.embedding_dim, self.hidden_dim, prefix=prefix+'_backward_gru')
        self.params += self.backward_gru.params

    def connect(self, inputs, inputs_mask=None):

        sent_len = inputs.shape[0]
        batch_size = inputs.shape[1]

        embedding = self.embedder.connect(inputs) #(sent_len, batch_size, embedding_dim)

        if inputs_mask is not None:
            forward_context = self.forward_gru.connect(embedding,
                                            sent_len=sent_len,
                                            batch_size=batch_size,
                                            mask=inputs_mask) 
            #(sent_len, batch_size, hidden_dim)
            backward_context = self.backward_gru.connect(embedding[::-1],
                                            sent_len=sent_len,
                                            batch_size=batch_size,
                                            mask=inputs_mask[::-1])
        else:
            forward_context = self.forward_gru.connect(embedding,
                                            sent_len=sent_len,
                                            batch_size=batch_size) 
            #(sent_len, batch_size, hidden_dim)
            backward_context = self.backward_gru.connect(embedding[::-1],
                                            sent_len=sent_len,
                                            batch_size=batch_size)

        # context = tensor.concatenate([ forward_context, backward_context[::-1] ], axis=2) #(sent_len, batch_size, 2*hidden_dim)

        # return context

        # mask is built in this way: mask * h + (1.-mask) * prev_h
        # so that the final state of each batch gets copied through max length
        fw_final_state = forward_context[-1]
        bw_final_state = backward_context[-1]
        final_states = tensor.concatenate([ fw_final_state, bw_final_state ], axis=-1) # (batch, hidden_dim)
	return final_states

class GembModel(object):
    """docstring for GembModel"""
    def __init__(self, input_dim, output_dim):
        #super(GembModel, self).__init__()
        self.params = []
        self.mlp = MLPLayer(input_dim, output_dim)
        self.params.extend(self.mlp.params)

    def add_char(self, vocab_size, embed_dim, hidden_dim):
        '''
        :param vocab_size: number of char
        :param embed_dim: dimension of char embedding
        :param hidden_dim: dimension of hidden states in char rnn
        '''
        self.char_rnn = BiGruEncoder(vocab_size, embed_dim, hidden_dim, prefix='ChrRNN')
        self.params.extend(self.char_rnn.params)
    
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


def replace_with_gemb(inputs_0, gembedding, oov_pos):
    emb_dim = gembedding.shape[2]
    inputs_0_new = np.copy(inputs_0)
    for i, oov in enumerate(gembedding):
        for j, batch in enumerate(oov):
            inputs_0_new[oov_pos[i],j,:emb_dim] = batch
    return inputs_0_new.astype("float32")
