import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test
from keras.layers import recurrent, embeddings, decomposition
from keras.layers.recurrent import LSTM, GRU
from keras.layers.decomposition import BetaLayer, GammaLayer, EpsLayer
from keras.models import Sequential
from keras.layers.core import Masking, Dense, Lambda
from keras.layers.wrappers import TimeDistributed
from keras import regularizers
from keras.utils.test_utils import keras_test

from keras import backend as K

nb_samples, timesteps, embedding_dim, output_dim, num_classes = 2, 5, 4, 3, 2
embedding_num = 12

def test_EpsLayer():
    model = Sequential()
    model.add(GRU(input_shape=(timesteps, embedding_dim), output_dim = output_dim, \
            return_sequences = True, return_all_states = True))
    model.add(EpsLayer())
    model.add(TimeDistributed(Dense(output_dim = num_classes)))
    model.add(TimeDistributed(Lambda(lambda x : K.exp(x))))

    model.compile(optimizer='sgd', loss='mse')
    out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
    assert(out.shape == (nb_samples, timesteps, num_classes))

def test_BetaLayer():
    model = Sequential()
    model.add(LSTM(input_shape=(timesteps, embedding_dim), output_dim = output_dim, \
            return_sequences = True, return_all_states = True))
    model.add(BetaLayer())
    model.add(TimeDistributed(Dense(output_dim = num_classes)))
    model.add(TimeDistributed(Lambda(lambda x : K.exp(x))))
    
    model.compile(optimizer='sgd', loss='mse')
    out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
    assert(out.shape == (nb_samples, timesteps, num_classes))

def test_GammaLayer():
    model = Sequential()
    model.add(LSTM(input_shape=(timesteps, embedding_dim), output_dim = output_dim, \
            return_sequences = True, return_all_states = True))
    model.add(GammaLayer())
    model.add(TimeDistributed(Dense(output_dim = num_classes)))
    model.add(TimeDistributed(Lambda(lambda x : K.exp(x))))
    
    model.compile(optimizer='sgd', loss='mse')
    out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
    assert(out.shape == (nb_samples, timesteps, num_classes))

if __name__ == '__main__':
    pytest.main([__file__])
