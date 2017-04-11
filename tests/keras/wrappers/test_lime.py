import pytest
import numpy as np

from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.layers.core import Masking, Dense, Lambda
from keras.layers.lime import *
from keras.layers.embeddings import Embedding

from keras import backend as K

nb_samples, timesteps, embedding_dim, output_dim, num_classes = 2, 7, 6, 3, 4
embedding_num = 12

def test_textlime():
    m = Sequential()
    m.add(Embedding(input_dim = embedding_num, output_dim = embedding_dim, mask_zero = True))
    m.add(LSTM(output_dim = output_dim))
    m.add(Dense(output_dim = num_classes, activation = "softmax"))
    m.compile("sgd", "categorical_crossentropy")
   
    lime = TextLime(model = m)
    lime_result = lime.call(np.random.choice(embedding_num,(nb_samples,timesteps)))

    assert lime_result.shape == (nb_samples, timesteps, num_classes)
    
    lime = TextLime(model = m, sampler = LimeTextSamplerRandom(10, 7))
    lime_result = lime.call(np.random.choice(embedding_num,(nb_samples,timesteps)))

    assert lime_result.shape == (nb_samples, timesteps, num_classes)

if __name__ == '__main__':
    pytest.main([__file__])

