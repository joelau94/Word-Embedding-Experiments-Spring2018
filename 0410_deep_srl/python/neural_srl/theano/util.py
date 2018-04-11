import numpy

import theano
from theano import tensor
from theano import config as theano_config
from neural_srl.shared.numpy_utils import random_normal_initializer

floatX = theano_config.floatX

def numpy_floatX(data):
    return numpy.asarray(data, dtype=floatX)

def get_variable(name, shape, initializer=None, dtype=floatX):
    if initializer != None:
        param = initializer(shape, dtype)
    else:
        param = random_normal_initializer()(shape, dtype)

    return theano.shared(value=param, name=name, borrow=True)

def softmax3d(energy):
    energy_reshape = energy.reshape( (energy.shape[0]*energy.shape[1], energy.shape[2]) )
    return tensor.nnet.softmax(energy_reshape).reshape( (energy.shape[0], energy.shape[1], energy.shape[2]) )