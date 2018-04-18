from layer import *
from util import *
import numpy as np


def _p(pp, name):
  return '%s_%s' % (pp, name)


class MLPLayer(object):
    def __init__(self, input_dim, output_dim, prefix='MLP'):
        self.W = get_variable(_p(prefix, 'W'), [input_dim, output_dim],
                          random_normal_initializer(0.0, 0.01))
        self.b = get_variable(_p(prefix, 'b'), [output_dim],
                              all_zero_initializer())
        self.params = [self.W, self.b]

    def connect(self, inputs):
        output = tensor.dot(inputs, self.W) + self.b
        return output


class GembModel(object):
    """docstring for GembModel"""
    def __init__(self, input_dim, output_dim):
        #super(GembModel, self).__init__()
        self.mlp = MLPLayer(input_dim, output_dim)
        self.params = self.mlp.params
    
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
    return inputs_0_new