# IDEE: Erasure Layer / Decomposition layer as secondary output (tied weights!), then give training goal!!!

import pytest
import numpy as np
np.random.seed(123)
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test
from keras.layers import recurrent, embeddings, Embedding, Dropout, Input
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, Model
from keras.layers.core import Masking, Dense
from keras.layers.wrappers import *
from keras import regularizers
from keras.utils.test_utils import keras_test

from keras import backend as K

nb_samples, timesteps, embedding_dim, units, num_classes, vocab_size = 2, 5, 6, 150, 40, 3
embedding_num = 12

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def test_EmbeddingWrapper():

    model = Sequential()
    model.add(EmbeddingWrapper(Embedding(input_dim = vocab_size, output_dim = embedding_dim, mask_zero = True), 
        scope = "local", mode = None))
    model.compile('sgd', 'mse')
    out = model.predict(np.random.random((nb_samples, timesteps)))
    assert(out.shape == (nb_samples, timesteps, embedding_dim))
    
    model = Sequential()
    model.add(EmbeddingWrapper(Embedding(input_dim = vocab_size, output_dim = embedding_dim, mask_zero = False), 
        scope = "local", mode = "l2"))
    model.compile('sgd', 'mse')
    out = model.predict(np.random.random((nb_samples, timesteps)))
    assert(out.shape == (nb_samples, timesteps))
    
    model = Sequential()
    model.add(EmbeddingWrapper(Embedding(input_dim = vocab_size, output_dim = embedding_dim, mask_zero = True), 
        scope = "global", mode = "l1"))
    model.compile('sgd', 'mse')
    out = model.predict(np.random.random((nb_samples, timesteps)))
    assert(out.shape == (nb_samples, timesteps))

def test_unit_tests_EmbeddingWrapper():
    [emb0, emb1, emb2] = [np.array([1,2]), np.array([4,6]), np.array([4,4])]
    embedding_weights = [np.array([emb0, emb1, emb2])]
    X = np.array([[0,1,2,0]])
    
    mean = (emb0 + emb1 + emb2 + emb0) / 4
    minus0, minus1, minus2 = emb0 - mean, emb1 - mean, emb2 - mean
    
    model = Sequential()
    model.add(EmbeddingWrapper(Embedding(input_dim = 3, output_dim = 2, mask_zero = False), \
    	mode = None, scope = "local", weights = embedding_weights))
    model.compile('sgd', 'mse')

    pred = model.predict(X)[0]
    assert np.allclose(pred, np.array([minus0, minus1, minus2, minus0]))
    
    #test masking
    model = Sequential()
    model.add(EmbeddingWrapper(Embedding(input_dim = 3, output_dim = 2, mask_zero = True), \
    	mode = None, scope = "local", weights = embedding_weights))
    model.compile('sgd', 'mse')

    pred = model.predict(X)[0]
    assert np.allclose(pred, np.array([np.zeros_like(minus0), minus1, minus2, np.zeros_like(minus0)]))

    #test l2 norm
    model = Sequential()
    model.add(EmbeddingWrapper(Embedding(input_dim = 3, output_dim = 2, mask_zero = False), \
    	mode = "l2", scope = "local", weights = embedding_weights))
    model.compile('sgd', 'mse')

    pred = model.predict(X)[0]
    assert np.allclose(pred, np.array([np.linalg.norm(x, ord = 2) for x in (minus0, minus1, minus2, minus0)]))
    
    #test l1 norm
    model = Sequential()
    model.add(EmbeddingWrapper(Embedding(input_dim = 3, output_dim = 2, mask_zero = False), \
    	mode = "l1", scope = "local", weights = embedding_weights))
    model.compile('sgd', 'mse')

    pred = model.predict(X)[0]
    assert np.allclose(pred, np.array([np.linalg.norm(x, ord = 1) for x in (minus0, minus1, minus2, minus0)]))
    
    #test masked l2 norm
    model = Sequential()
    model.add(EmbeddingWrapper(Embedding(input_dim = 3, output_dim = 2, mask_zero = True), \
    	mode = "l2", scope = "local", weights = embedding_weights))
    model.compile('sgd', 'mse')

    pred = model.predict(X)[0]
    assert np.allclose(pred, np.array([0] + [np.linalg.norm(x, ord = 2) for x in (minus1, minus2)] + [0]))
    
    #test global
    mean = (emb0 + emb1 + emb2) / 3
    minus0, minus1, minus2 = emb0 - mean, emb1 - mean, emb2 - mean

    model = Sequential()
    model.add(EmbeddingWrapper(Embedding(input_dim = 3, output_dim = 2, mask_zero = True), \
    	mode = "l2", scope = "global", weights = embedding_weights))
    model.compile('sgd', 'mse')

    pred = model.predict(X)[0]
    assert np.allclose(pred, np.array([0] + [np.linalg.norm(x, ord = 2) for x in (minus1, minus2)] + [0]))




'''


def test_GradientWrapper():
    model = Sequential()
    model.add(GradientWrapper(GRU(units = units, return_sequences = False), input_shape = (None, embedding_dim), mode = "dot"))
    model.compile(optimizer='sgd', loss='mse')
    out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
    assert(out.shape == (nb_samples, timesteps, units))
    
    model = Sequential()
    model.add(Embedding(input_dim=5, output_dim = embedding_dim, mask_zero = True))
    inner_model = Sequential()
    inner_model.add(GRU(units = units, return_sequences = False, input_shape = (None, embedding_dim)))
    #inner_model.add(Dropout(0.5))
    inner_model.add(Dense(units = num_classes, activation = "linear"))
    model.add(GradientWrapper(inner_model, mode = "l1"))
    model.compile(optimizer='sgd', loss='mse')
    a = np.array([[1,2,3,0,0], [1,3,0,0,0]])
    out = model.predict(a)
    assert(out.shape == (nb_samples, timesteps, num_classes))
    assert(out[0,2,0] != 0)
    assert(out[0,3,0] == 0)
    
    inp = Input((None,))
    emb = Embedding(input_dim=5, output_dim = embedding_dim, mask_zero = True)
    gru = GRU(units = units, return_sequences = False, input_shape = (None, embedding_dim))
    wrap = GradientWrapper(gru, mode = None)
    dense = Dense(units = num_classes, activation = "linear")

    inner = Sequential()
    inner.add(gru)
    inner.add(dense)
    wrap2 = GradientWrapper(inner, mode = None)

    outp = dense(wrap(emb(inp)))
    outp2 = wrap2(emb(inp))
    model = Model([inp], [outp])
    model2 = Model([inp], [outp2])
    print("predicting model1")
    out = model.predict(a)
    print("predicting model2")
    out2 = model2.predict(a)
    print("model1")
    print(out)
    print("model2")
    print(out2)
    assert (0)



def test_ErasureWrapper():
    for i in range(1,4):
        model = Sequential()
        model.add(ErasureWrapper(GRU(units = units, return_sequences = False), input_shape = (None, embedding_dim), ngram = i))
        model.add(TimeDistributed(Dense(num_classes, activation = "exp")))
        model.compile(optimizer='sgd', loss='mse')
        out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
        assert(out.shape == (nb_samples, timesteps - i + 1, num_classes))
    
    # with return_sequences = True
    for i in range(1,4):
        model = Sequential()
        model.add(ErasureWrapper(GRU(units = units, return_sequences = True), input_shape = (None, embedding_dim), ngram = i))
        model.add(TimeDistributed(TimeDistributed(Dense(num_classes, activation = "exp"))))
        model.compile(optimizer='sgd', loss='mse')
        out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
        assert(out.shape == (nb_samples, timesteps - i + 1, timesteps, num_classes))

def test_BetaLayer():
    for i in range(1,4):
        for layer in (GRU, LSTM):
            model = Sequential()
            model.add(BetaDecomposition(layer(units = units), ngram = i, input_shape = (None, embedding_dim)))
            model.add(TimeDistributed(Dense(units = num_classes, activation = "exp")))
            model.compile(optimizer='sgd', loss='mse')
            out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
            assert(out.shape == (nb_samples, timesteps - i + 1, num_classes))

def test_GammaLayer():
    for i in range(1,4):
        for layer in (GRU, LSTM):
            model = Sequential()
            model.add(GammaDecomposition(layer(units = units), ngram = i, input_shape = (None, embedding_dim)))
            model.add(TimeDistributed(Dense(units = num_classes, activation = "exp")))
            model.compile(optimizer='sgd', loss='mse')
            out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
            assert(out.shape == (nb_samples, timesteps - i + 1, num_classes))

def test_return_sequences():
    for i in range(1,4):
        for outer_layer in (BetaDecomposition, GammaDecomposition):
            for inner_layer in (LSTM, GRU):
                model = Sequential()
                model.add(outer_layer(inner_layer(units = units, return_sequences = True), ngram = i, input_shape=(None, embedding_dim)))
                model.add(TimeDistributed(TimeDistributed(Dense(units = num_classes, activation = "exp"))))
    
                model.compile(optimizer='sgd', loss='mse')
                out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
                assert(out.shape == (nb_samples, timesteps - i + 1, timesteps, num_classes))


def test_bidirectional():
    for i in range(1,4):
        for outer_layer in (BetaDecomposition, GammaDecomposition):
            for inner_layer in (LSTM, GRU):
                model = Sequential()
                model.add(Bidirectional(outer_layer(inner_layer(units = units), ngram = i), input_shape=(None, embedding_dim)))
                model.add(TimeDistributed(Dense(units = num_classes, activation = "exp")))
                model.compile("sgd", "mse")
                out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
                assert out.shape == (nb_samples, timesteps - i + 1, num_classes)
    
        
def test_unit_tests_DecompositionLSTM():

    x1 = np.array([1,2,1])
    x2 = np.array([0,1,1])
    x3 = np.array([1,1,1])
    
    X = np.stack([x1,x2,x3])
    X = np.stack([X])
    
    Wout = np.array([[2,0,0],[0,1,1]])
    bout = np.zeros((3,))


    # LSTM
    h0 = np.array([0,0])
    c0 = np.array([0,0])
        
    Wi = np.array([[0,0], [0,1], [0,1]])
    Ui = np.array([[0,1], [1,0]])
        
    Wf = np.array([[2,0], [0,2], [0,1]])
    Uf = np.array([[0,2], [1,2]])
        
    Wo = np.array([[1,0], [0,0], [0,1]])
    Uo = np.array([[0,2], [1,1]])
        
    Wc = np.array([[1,3], [0,0], [0,1]])
    Uc = np.array([[0,1], [1,1]])
        
    i1 = sigmoid(np.dot(x1, Wi) + np.dot(h0, Ui))
    f1 = sigmoid(np.dot(x1, Wf) + np.dot(h0, Uf))
    o1 = sigmoid(np.dot(x1, Wo) + np.dot(h0, Uo))
    h_tilde1 = np.tanh(np.dot(x1, Wc) + np.dot(h0, Uc))
        
    c1 = f1 * c0 + i1 * h_tilde1
    h1 = o1 * np.tanh(c1)
        
    i2 = sigmoid(np.dot(x2, Wi) + np.dot(h1, Ui))
    f2 = sigmoid(np.dot(x2, Wf) + np.dot(h1, Uf))
    o2 = sigmoid(np.dot(x2, Wo) + np.dot(h1, Uo))
    h_tilde2 = np.tanh(np.dot(x2, Wc) + np.dot(h1, Uc))

    c2 = f2 * c1 + i2 * h_tilde2
    h2 = o2 * np.tanh(c2)

    i3 = sigmoid(np.dot(x3, Wi) + np.dot(h2, Ui))
    f3 = sigmoid(np.dot(x3, Wf) + np.dot(h2, Uf))
    o3 = sigmoid(np.dot(x3, Wo) + np.dot(h2, Uo))
    h_tilde3 = np.tanh(np.dot(x3, Wc) + np.dot(h2, Uc))

    c3 = f3 * c2 + i3 * h_tilde3
    h3 = o3 * np.tanh(c3)
    
    bi = np.zeros((2,))
    bc = np.zeros((2,))
    bf = np.zeros((2,))
    bo = np.zeros((2,))

    W = [np.concatenate([Wi, Wf, Wc, Wo], -1), np.concatenate([Ui, Uf, Uc, Uo], -1), np.concatenate([bi, bf, bc, bo], -1)]

    beta1 = np.exp(np.dot(o3 * (np.tanh(c1) - np.tanh(c0)), Wout))
    beta2 = np.exp(np.dot(o3 * (np.tanh(c2) - np.tanh(c1)), Wout))
    beta3 = np.exp(np.dot(o3 * (np.tanh(c3) - np.tanh(c2)), Wout))
    
    m = Sequential()
    m.add(BetaDecomposition(LSTM(units = 2, recurrent_activation = "sigmoid"), input_shape = (None, 3), weights = W))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]

    assert np.allclose(np.array([beta1, beta2, beta3]), pred)
    
    gamma1 = np.exp(np.dot(o3 * (np.tanh(f2 * f3 * c1) - np.tanh(f1 * f2 * f3 * c0)), Wout))
    gamma2 = np.exp(np.dot(o3 * (np.tanh(f3 * c2) - np.tanh(f2 * f3 * c1)), Wout))
    gamma3 = np.exp(np.dot(o3 * (np.tanh(c3) - np.tanh(f3 * c2)), Wout))
    
    m = Sequential()
    m.add(GammaDecomposition(LSTM(units = 2, recurrent_activation = "sigmoid"), input_shape = (None, 3), weights = W))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([gamma1, gamma2, gamma3]), pred)
    

    mbeta1 = np.exp(np.dot(o2 * (np.tanh(c1) - np.tanh(c0)), Wout))
    mbeta2 = np.exp(np.dot(o2 * (np.tanh(c2) - np.tanh(c1)), Wout))
    
    m = Sequential()
    m.add(Masking(input_shape=(None,3), mask_value = 1))
    m.add(BetaDecomposition(LSTM(units = 2, recurrent_activation = "sigmoid"), input_shape = (None, 3), weights = W))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")

    pred = m.predict(X)[0]

    assert np.allclose(np.array([mbeta1, mbeta2, np.ones_like(mbeta2)]), pred)
    
    mgamma1 = np.exp(np.dot(o2 * (np.tanh(f2 * c1) - np.tanh(f1*f2*c0)), Wout))
    mgamma2 = np.exp(np.dot(o2 * (np.tanh(c2) - np.tanh(f2 * c1)), Wout))

    m = Sequential()
    m.add(Masking(input_shape=(None,3), mask_value = 1))
    m.add(GammaDecomposition(LSTM(units = 2, recurrent_activation = "sigmoid"), input_shape = (None, 3), weights = W))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([mgamma1, mgamma2, np.ones_like(mgamma2)]), pred)
    
    # with bigram
    
    bbeta2 = np.exp(np.dot(o3 * (np.tanh(c2) - np.tanh(c0)), Wout))
    bbeta3 = np.exp(np.dot(o3 * (np.tanh(c3) - np.tanh(c1)), Wout))

    m = Sequential()
    m.add(BetaDecomposition(LSTM(units = 2, recurrent_activation = "sigmoid"), input_shape = (None, 3), weights = W, ngram = 2))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([bbeta2, bbeta3]), pred)
    
    bgamma2 = np.exp(np.dot(o3 * (np.tanh(f3 * c2) - np.tanh(f3 * f2 * f1 * c0)), Wout))
    bgamma3 = np.exp(np.dot(o3 * (np.tanh(c3) - np.tanh(f3 * f2 * c1)), Wout))

    m = Sequential()
    m.add(GammaDecomposition(LSTM(units = 2, recurrent_activation = "sigmoid"), input_shape = (None, 3), weights = W, ngram = 2))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([bgamma2, bgamma3]), pred)

    tbeta = np.exp(np.dot(o3 * np.tanh(c3), Wout))
    

def test_unit_tests_DecompositionGRU():

    x1 = np.array([1,2,1])
    x2 = np.array([0,1,1])
    x3 = np.array([1,1,1])
    
    X = np.stack([x1,x2,x3])
    X = np.stack([X])
    
    Wout = np.array([[2,0,0],[0,1,1]])
    bout = np.zeros((3,))

    # GRU
    h0 = np.array([0,0])
    c0 = np.array([0,0])
    
    Wz = np.array([[0,0], [0,1], [0,1]])
    Uz = np.array([[0,1], [1,0]])

    Wr = np.array([[2,0], [0,2], [0,1]])
    Ur = np.array([[0,2], [1,2]])

    Wh = np.array([[1,0], [0,0], [0,1]])
    Uh = np.array([[0,2], [1,1]])
    
    z1 = sigmoid(np.dot(x1, Wz) + np.dot(h0, Uz))
    r1 = sigmoid(np.dot(x1, Wr) + np.dot(h0, Ur))
    h_tilde1 = np.tanh(np.dot(x1, Wh) + np.dot(r1 * h0, Uh))
    h1 = (1 - z1) * h_tilde1 + z1 * h0

    z2 = sigmoid(np.dot(x2, Wz) + np.dot(h1, Uz))
    r2 = sigmoid(np.dot(x2, Wr) + np.dot(h1, Ur))
    h_tilde2 = np.tanh(np.dot(x2, Wh) + np.dot(r2 * h1, Uh))
    h2 = (1 - z2) * h_tilde2 + z2 * h1

    z3 = sigmoid(np.dot(x3, Wz) + np.dot(h2, Uz))
    r3 = sigmoid(np.dot(x3, Wr) + np.dot(h2, Ur))
    h_tilde3 = np.tanh(np.dot(x3, Wh) + np.dot(r3 * h2, Uh))
    h3 = (1 - z3) * h_tilde3 + z3 * h2

    bz = np.zeros((2,))
    br = np.zeros((2,))
    bh = np.zeros((2,))

    W = [np.concatenate([Wz, Wr, Wh], -1), np.concatenate([Uz, Ur, Uh], -1), np.concatenate([bz, br, bh], -1)]

    delta1 = np.exp(np.dot(h1-h0, Wout))
    delta2 = np.exp(np.dot(h2-h1, Wout))
    delta3 = np.exp(np.dot(h3-h2, Wout))
    
    m = Sequential()
    m.add(BetaDecomposition(GRU(units = 2, recurrent_activation = "sigmoid"), input_shape = (None, 3), weights = W))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]

    assert np.allclose(np.array([delta1, delta2, delta3]), pred)
    
    omega1 = np.exp(np.dot((z3 * z2 * h1) - (z3 * z2 * z1 * h0), Wout))
    omega2 = np.exp(np.dot((z3 * h2) - (z3 * z2 * h1), Wout))
    omega3 = np.exp(np.dot(h3 - (z3 * h2), Wout))
    
    m = Sequential()
    m.add(GammaDecomposition(GRU(units = 2, recurrent_activation = "sigmoid", weights = W), input_shape = (None, 3)))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]

    assert np.allclose(np.array([omega1, omega2, omega3]), pred)
    
    mdelta1 = np.exp(np.dot(h1 - h0, Wout))
    mdelta2 = np.exp(np.dot(h2 - h1, Wout))

    m = Sequential()
    m.add(Masking(input_shape=(None,3), mask_value = 1))
    m.add(BetaDecomposition(GRU(units = 2, recurrent_activation = "sigmoid"), input_shape = (None, 3), weights = W))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([mdelta1, mdelta2, np.ones_like(mdelta2)]), pred)
    
    momega1 = np.exp(np.dot((z2 * h1) - (z2 * z1 * h0), Wout))
    momega2 = np.exp(np.dot(h2 - (z2 * h1), Wout))

    m = Sequential()
    m.add(Masking(input_shape=(None,3), mask_value = 1))
    m.add(GammaDecomposition(GRU(units = 2, recurrent_activation = "sigmoid"), input_shape = (None, 3), weights = W))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([momega1, momega2, np.ones_like(momega2)]), pred)


    # with bigram
    
    bdelta2 = np.exp(np.dot(h2 - h0, Wout))
    bdelta3 = np.exp(np.dot(h3 - h1, Wout))

    m = Sequential()
    m.add(BetaDecomposition(GRU(units = 2, recurrent_activation = "sigmoid"), input_shape = (None, 3), weights = W, ngram = 2))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([bdelta2, bdelta3]), pred)
    
    bomega2 = np.exp(np.dot((z3 * h2) - (z3 * z2 * z1 * h0), Wout))
    bomega3 = np.exp(np.dot(h3 - (z3 * z2 * h1), Wout))

    m = Sequential()
    m.add(GammaDecomposition(GRU(units = 2, recurrent_activation = "sigmoid"), input_shape = (None, 3), weights = W, ngram = 2))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([bomega2, bomega3]), pred)

    tdelta = np.exp(np.dot(h3, Wout))
    
    m = Sequential()
    m.add(BetaDecomposition(GRU(units = 2, recurrent_activation = "sigmoid"), input_shape = (None, 3), weights = W, ngram = 3))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([tdelta]), pred)
    
    m = Sequential()
    m.add(GammaDecomposition(GRU(units = 2, recurrent_activation = "sigmoid"), input_shape = (None, 3), weights = W, ngram = 3))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([tdelta]), pred)


def test_unit_tests_Erasure():

    x1 = np.array([1,2,1,2])
    x2 = np.array([0,1,1,1])
    x3 = np.array([1,1,1,1])
    
    X = np.stack([x1,x2,x3])
    X = np.stack([X,X,X])
    
    Wout = np.array([[2,0,0],[0,1,1]])
    bout = np.zeros((3,))

    # LSTM
    
    h0 = np.array([0,0])
    c0 = np.array([0,0])
        
    Wi = np.array([[0,0], [0,1], [0,1],[1,6]])
    Ui = np.array([[0,1], [1,0]])
        
    Wf = np.array([[2,0], [0,2], [0,1],[1,4]])
    Uf = np.array([[0,2], [1,2]])
        
    Wo = np.array([[1,0], [0,0], [0,1],[1,8]])
    Uo = np.array([[0,2], [1,1]])
        
    Wc = np.array([[1,3], [0,0], [0,1],[1,4]])
    Uc = np.array([[0,1], [1,1]])
    
    bi = np.zeros((2,))
    bc = np.zeros((2,))
    bf = np.zeros((2,))
    bo = np.zeros((2,))

    W = [np.concatenate([Wi, Wf, Wc, Wo], -1), np.concatenate([Ui, Uf, Uc, Uo], -1), np.concatenate([bi, bf, bc, bo], -1)]

    # with nothing missing
    
    model = Sequential()
    model.add(LSTM(units = 2, weights = W, input_shape = (None,4)))
    model.compile(optimizer='sgd', loss='mse')
    outnorm = model.predict(np.array([[x1, x2, x3]]))
    
    # with x missing
    
    out1 = model.predict(np.array([[np.zeros_like(x1), x2, x3]]))
    out2 = model.predict(np.array([[x1, np.zeros_like(x2), x3]]))
    out3 = model.predict(np.array([[x1, x2, np.zeros_like(x3)]]))
    
    model = Sequential()
    model.add(ErasureWrapper(LSTM(units = 2), weights = W, input_shape=(None,4)))
    model.compile(optimizer='sgd', loss='mse')
    outerasure = model.predict(np.array([[x1,x2,x3],[x1,x2,x3]]))[0]

    assert(np.allclose(np.array([outnorm-out1, outnorm-out2, outnorm-out3]).squeeze(), outerasure))
    
    # GRU

    Wz = np.array([[0,0], [0,1], [0,1],[1,2]])
    Uz = np.array([[0,1], [1,0]])

    Wr = np.array([[2,0], [0,2], [0,1],[1,1]])
    Ur = np.array([[0,2], [1,2]])

    Wh = np.array([[1,0], [0,0], [0,1],[1,1]])
    Uh = np.array([[0,2], [1,1]])
    
    z1 = sigmoid(np.dot(x1, Wz) + np.dot(h0, Uz))
    r1 = sigmoid(np.dot(x1, Wr) + np.dot(h0, Ur))
    h_tilde1 = np.tanh(np.dot(x1, Wh) + np.dot(r1 * h0, Uh))
    h1 = (1 - z1) * h_tilde1 + z1 * h0

    z2 = sigmoid(np.dot(x2, Wz) + np.dot(h1, Uz))
    r2 = sigmoid(np.dot(x2, Wr) + np.dot(h1, Ur))
    h_tilde2 = np.tanh(np.dot(x2, Wh) + np.dot(r2 * h1, Uh))
    h2 = (1 - z2) * h_tilde2 + z2 * h1

    z3 = sigmoid(np.dot(x3, Wz) + np.dot(h2, Uz))
    r3 = sigmoid(np.dot(x3, Wr) + np.dot(h2, Ur))
    h_tilde3 = np.tanh(np.dot(x3, Wh) + np.dot(r3 * h2, Uh))
    h3 = (1 - z3) * h_tilde3 + z3 * h2

    bz = np.zeros((2,))
    br = np.zeros((2,))
    bh = np.zeros((2,))

    W = [np.concatenate([Wz, Wr, Wh], -1), np.concatenate([Uz, Ur, Uh], -1), np.concatenate([bz, br, bh], -1)]
    Wb = [np.concatenate([Wz, Wr, Wh], -1), np.concatenate([np.array([[4,5],[6,7]]), Ur, Uh], -1), np.concatenate([bz, br, bh], -1)]
    
    # with nothing missing
    
    model = Sequential()
    model.add(GRU(units = 2, weights = W, input_shape = (None, 4)))
    model.compile(optimizer='sgd', loss='mse')
    outnorm = model.predict(np.array([[x1, x2, x3]]))
    
    # with x missing
    
    out1 = model.predict(np.array([[np.zeros_like(x1), x2, x3]]))
    out2 = model.predict(np.array([[x1, np.zeros_like(x2), x3]]))
    out3 = model.predict(np.array([[x1, x2, np.zeros_like(x3)]]))
    
    out12 = model.predict(np.array([[np.zeros_like(x1), np.zeros_like(x2), x3]]))
    out23 = model.predict(np.array([[x1, np.zeros_like(x2), np.zeros_like(x3)]]))
    
    
    model = Sequential()
    model.add(ErasureWrapper(GRU(units = 2, weights = W), input_shape=(3,4), ngram = 1))
    model.compile(optimizer='sgd', loss='mse')
    outerasure = model.predict(np.array([[x1,x2,x3]]))

    assert(np.allclose(np.array([outnorm-out1, outnorm-out2, outnorm-out3]).squeeze(), outerasure))
    
    # with 2 x missing
    
    model = Sequential()
    model.add(ErasureWrapper(GRU(units = 2), weights = W, input_shape=(None,4), ngram = 2))
    model.compile(optimizer='sgd', loss='mse')
    outerasure = model.predict(np.array([[x1,x2,x3]]))
    assert(np.allclose(np.array([outnorm-out12, outnorm-out23]).squeeze(), outerasure))
    
    
    # with bidirectional
    
    model = Sequential()
    model.add(Bidirectional(GRU(units = 2), weights = W + Wb, input_shape = (3,4)))
    model.compile(optimizer='sgd', loss='mse')
    outnorm = model.predict(np.array([[x1, x2, x3]]))
    
    # with x missing
    
    out1 = model.predict(np.array([[np.zeros_like(x1), x2, x3]]))
    out2 = model.predict(np.array([[x1, np.zeros_like(x2), x3]]))
    out3 = model.predict(np.array([[x1, x2, np.zeros_like(x3)]]))
    
    model = Sequential()
    model.add(ErasureWrapper(Bidirectional(GRU(units = 2), weights = W + Wb), ngram = 1, input_shape = (3,4)))
    model.compile(optimizer='sgd', loss='mse')
    outerasurebi = model.predict(np.array([[x1,x2,x3]]))

    assert(np.allclose(np.array([outnorm-out1, outnorm-out2, outnorm - out3]).squeeze(), outerasurebi))
    
    
    # with masking
    
    model = Sequential()
    model.add(Masking(mask_value = 0, input_shape = (3,4)))
    model.add(GRU(units = 2, weights = W))
    model.compile(optimizer='sgd', loss='mse')
    outnormbi = model.predict(np.array([[x1, x2, np.zeros_like(x3)]]))
    
    out1 = model.predict(np.array([[np.zeros_like(x1), x2, np.zeros_like(x3)]]))
    out2 = model.predict(np.array([[x1, np.zeros_like(x2), np.zeros_like(x3)]]))
    out3 = model.predict(np.array([[x1, x2, np.zeros_like(x3)]]))
    
    model = Sequential()
    model.add(Masking(0, input_shape = (3,4)))
    model.add(ErasureWrapper(GRU(units = 2), weights = W, ngram = 1))
    model.compile(optimizer='sgd', loss='mse')
    outerasurebi = model.predict(np.array([[x1,x2,np.zeros_like(x3)]]))
    
    # with bidir + masking
    
    model = Sequential()
    model.add(Masking(mask_value = 0, input_shape = (3,4)))
    model.add(Bidirectional(GRU(units = 2), weights = W + Wb))
    model.compile(optimizer='sgd', loss='mse')
    outnormbi = model.predict(np.array([[x1, x2, np.zeros_like(x3)]]))
    
    out1 = model.predict(np.array([[np.zeros_like(x1), x2, np.zeros_like(x3)]]))
    out2 = model.predict(np.array([[x1, np.zeros_like(x2), np.zeros_like(x3)]]))
    out3 = model.predict(np.array([[x1, x2, np.zeros_like(x3)]]))
    
    model = Sequential()
    model.add(Masking(0, input_shape = (3,4)))
    model.add(ErasureWrapper(Bidirectional(GRU(units = 2), weights = W + Wb), ngram = 1))
    model.compile(optimizer='sgd', loss='mse')
    outerasurebi = model.predict(np.array([[x1,x2,np.zeros_like(x3)]]))
    
    # with bidir + masking
    WW = np.array([[1,2,3],[4,5,6],[6,5,4],[1,1,2]])
    bb = np.array([1,2,2])
    Wemb = [np.random.random((5,4))]
    
    model = Sequential()
    model.add(Embedding(input_dim = 5, output_dim = 4, mask_zero = True, weights = Wemb))
    model.add(Bidirectional(GRU(units = 2), weights = W + Wb))
    model.add(Dense(units=3, weights = [WW, bb], use_bias = True, activation = "linear"))
    model.compile(optimizer='sgd', loss='mse')
    outnormbi = model.predict(np.array([[1,2,0]]))
    
    out1 = model.predict(np.array([[0,2,0]]))
    out2 = model.predict(np.array([[1,0,0]]))
    out3 = model.predict(np.array([[1,2,0]]))
    
    model = Sequential()
    model.add(Embedding(input_dim = 5, output_dim = 4, mask_zero = True, weights = Wemb))
    model.add(ErasureWrapper(Bidirectional(GRU(units = 2), weights = W + Wb), ngram = 1))
    model.add(TimeDistributed(Dense(units=3, activation = "linear", use_bias = False), weights = [WW]))
    model.compile(optimizer='sgd', loss='mse')
    outerasurebi = model.predict(np.array([[1,2,0]]))

    assert(np.allclose(np.array([outnormbi-out1, outnormbi-out2, outnormbi - out3]).squeeze(), outerasurebi))
    
def test_unit_tests_DecompositionLSTM_bidirectional():

    x1 = np.array([1,2,1])
    x2 = np.array([0,1,1])
    x3 = np.array([1,1,1])
    
    X = np.stack([x1,x2,x3])
    X = np.stack([X])
    
    Woutf = np.array([[2,0,0],[0,1,1]])
    Woutb = np.array([[6,7,8],[4,5,3]])
    Wout = np.concatenate([Woutf, Woutb], axis = 0)
    bout = np.zeros((3,))


    # Forward
    h0 = np.array([0,0])
    c0 = np.array([0,0])
        
    Wif = np.array([[0,0], [0,1], [0,1]])
    Uif = np.array([[0,1], [1,0]])
        
    Wff = np.array([[2,0], [0,2], [0,1]])
    Uff = np.array([[0,2], [1,2]])
        
    Wof = np.array([[1,0], [0,0], [0,1]])
    Uof = np.array([[0,2], [1,1]])
        
    Wcf = np.array([[1,3], [0,0], [0,1]])
    Ucf = np.array([[0,1], [1,1]])
        
    i1f = sigmoid(np.dot(x1, Wif) + np.dot(h0, Uif))
    f1f = sigmoid(np.dot(x1, Wff) + np.dot(h0, Uff))
    o1f = sigmoid(np.dot(x1, Wof) + np.dot(h0, Uof))
    h_tilde1f = np.tanh(np.dot(x1, Wcf) + np.dot(h0, Ucf))
        
    c1f = f1f * c0 + i1f * h_tilde1f
    h1f = o1f * np.tanh(c1f)
        
    i2f = sigmoid(np.dot(x2, Wif) + np.dot(h1f, Uif))
    f2f = sigmoid(np.dot(x2, Wff) + np.dot(h1f, Uff))
    o2f = sigmoid(np.dot(x2, Wof) + np.dot(h1f, Uof))
    h_tilde2f = np.tanh(np.dot(x2, Wcf) + np.dot(h1f, Ucf))

    c2f = f2f * c1f + i2f * h_tilde2f
    h2f = o2f * np.tanh(c2f)

    i3f = sigmoid(np.dot(x3, Wif) + np.dot(h2f, Uif))
    f3f = sigmoid(np.dot(x3, Wff) + np.dot(h2f, Uff))
    o3f = sigmoid(np.dot(x3, Wof) + np.dot(h2f, Uof))
    h_tilde3f = np.tanh(np.dot(x3, Wcf) + np.dot(h2f, Ucf))

    c3f = f3f * c2f + i3f * h_tilde3f
    h3f = o3f * np.tanh(c3f)
    
    bi = np.zeros((2,))
    bc = np.zeros((2,))
    bf = np.zeros((2,))
    bo = np.zeros((2,))

    
    # Backward
    h4 = np.array([0,0])
    c4 = np.array([0,0])
        
    Wib = np.array([[0,0], [0,1], [0,1]])
    Uib = np.array([[0,1], [4,0]])
        
    Wfb = np.array([[2,0], [5,2], [0,1]])
    Ufb = np.array([[0,2], [1,2]])
        
    Wob = np.array([[1,0], [0,6], [0,1]])
    Uob = np.array([[0,2], [1,1]])
        
    Wcb = np.array([[1,3], [0,0], [0,3]])
    Ucb = np.array([[0,1], [1,1]])
        
    i3b = sigmoid(np.dot(x3, Wib) + np.dot(h4, Uib))
    f3b = sigmoid(np.dot(x3, Wfb) + np.dot(h4, Ufb))
    o3b = sigmoid(np.dot(x3, Wob) + np.dot(h4, Uob))
    h_tilde3b = np.tanh(np.dot(x3, Wcb) + np.dot(h4, Ucb))
        
    c3b = f3b * c4 + i3b * h_tilde3b
    h3b = o3b * np.tanh(c3b)
        
    i2b = sigmoid(np.dot(x2, Wib) + np.dot(h3b, Uib))
    f2b = sigmoid(np.dot(x2, Wfb) + np.dot(h3b, Ufb))
    o2b = sigmoid(np.dot(x2, Wob) + np.dot(h3b, Uob))
    h_tilde2b = np.tanh(np.dot(x2, Wcb) + np.dot(h3b, Ucb))

    c2b = f2b * c3b + i2b * h_tilde2b
    h2b = o2b * np.tanh(c2b)

    i1b = sigmoid(np.dot(x1, Wib) + np.dot(h2b, Uib))
    f1b = sigmoid(np.dot(x1, Wfb) + np.dot(h2b, Ufb))
    o1b = sigmoid(np.dot(x1, Wob) + np.dot(h2b, Uob))
    h_tilde1b = np.tanh(np.dot(x1, Wcb) + np.dot(h2b, Ucb))

    c1b = f1b * c2b + i1b * h_tilde1b
    h1b = o1b * np.tanh(c1b)
    
    bi = np.zeros((2,))
    bc = np.zeros((2,))
    bf = np.zeros((2,))
    bo = np.zeros((2,))
    
    Wf = [np.concatenate([Wif, Wff, Wcf, Wof], -1), np.concatenate([Uif, Uff, Ucf, Uof], -1), np.concatenate([bi, bf, bc, bo], -1)]
    Wb = [np.concatenate([Wib, Wfb, Wcb, Wob], -1), np.concatenate([Uib, Ufb, Ucb, Uob], -1), np.concatenate([bi, bf, bc, bo], -1)]
    
    rbeta1b = o1b * (np.tanh(c1b) - np.tanh(c2b))
    rbeta2b = o1b * (np.tanh(c2b) - np.tanh(c3b))
    rbeta3b = o1b * (np.tanh(c3b) - np.tanh(c4))

    rbeta1f = o3f * (np.tanh(c1f) - np.tanh(c0))
    rbeta2f = o3f * (np.tanh(c2f) - np.tanh(c1f))
    rbeta3f = o3f * (np.tanh(c3f) - np.tanh(c2f))
    
    rbeta1 = np.concatenate([rbeta1f, rbeta1b])
    rbeta2 = np.concatenate([rbeta2f, rbeta2b])
    rbeta3 = np.concatenate([rbeta3f, rbeta3b])
    
    beta1 = np.exp(np.dot(rbeta1, Wout))
    beta2 = np.exp(np.dot(rbeta2, Wout))
    beta3 = np.exp(np.dot(rbeta3, Wout))
    
    m = Sequential()
    m.add(BetaDecomposition(LSTM(units = 2, recurrent_activation = "sigmoid", go_backwards = True), weights = Wb, input_shape = (None, 3)))
    m.compile("sgd", "mse")
    pred = m.predict(X)[0]
    
    assert(np.allclose(pred, np.array([rbeta3b, rbeta2b, rbeta1b])))

    m = Sequential()
    m.add(Bidirectional(BetaDecomposition(LSTM(units = 2, recurrent_activation = "sigmoid")), weights = Wf + Wb, input_shape = (None, 3)))
    #m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile("sgd", "mse")
    pred = m.predict(X)[0]
    
    

    assert(np.allclose(pred, np.array([rbeta1, rbeta2, rbeta3])))
    
    # gamma
    
    rgamma1b = o1b * (np.tanh(c1b) - np.tanh(c2b * f1b))
    rgamma2b = o1b * (np.tanh(c2b * f1b) - np.tanh(c3b * f2b * f1b))
    rgamma3b = o1b * (np.tanh(c3b * f2b * f1b) - np.tanh(c4 * f3b * f2b * f1b))

    rgamma1f = o3f * (np.tanh(c1f * f2f * f3f) - np.tanh(c0 * f1f * f2f * f3f))
    rgamma2f = o3f * (np.tanh(c2f * f3f) - np.tanh(c1f * f2f * f3f))
    rgamma3f = o3f * (np.tanh(c3f) - np.tanh(c2f * f3f))

    rgamma1 = np.concatenate([rgamma1f, rgamma1b])
    rgamma2 = np.concatenate([rgamma2f, rgamma2b])
    rgamma3 = np.concatenate([rgamma3f, rgamma3b])
   
    gamma1 = np.exp(np.dot(rgamma1, Wout))
    gamma2 = np.exp(np.dot(rgamma2, Wout))
    gamma3 = np.exp(np.dot(rgamma3, Wout))
    
    m = Sequential()
    m.add(Bidirectional(GammaDecomposition(LSTM(units = 2, recurrent_activation = "sigmoid")), weights = Wf + Wb, input_shape = (None, 3)))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp", weights = [Wout, bout])))
    m.compile("sgd", "mse")
    pred = m.predict(X)[0]
    
    assert(np.allclose(pred, np.array([gamma1, gamma2, gamma3])))

    
    # does masking work?
    
    i2b = sigmoid(np.dot(x2, Wib) + np.dot(h4, Uib))
    f2b = sigmoid(np.dot(x2, Wfb) + np.dot(h4, Ufb))
    o2b = sigmoid(np.dot(x2, Wob) + np.dot(h4, Uob))
    h_tilde2b = np.tanh(np.dot(x2, Wcb) + np.dot(h4, Ucb))

    c2b = f2b * c4 + i2b * h_tilde2b
    h2b = o2b * np.tanh(c2b)

    i1b = sigmoid(np.dot(x1, Wib) + np.dot(h2b, Uib))
    f1b = sigmoid(np.dot(x1, Wfb) + np.dot(h2b, Ufb))
    o1b = sigmoid(np.dot(x1, Wob) + np.dot(h2b, Uob))
    h_tilde1b = np.tanh(np.dot(x1, Wcb) + np.dot(h2b, Ucb))

    c1b = f1b * c2b + i1b * h_tilde1b
    h1b = o1b * np.tanh(c1b)
    
    mbeta1 = np.exp(np.dot(np.concatenate([o2f * (np.tanh(c1f) - np.tanh(c0)), \
            o1b * (np.tanh(c1b) - np.tanh(c2b))]), Wout))
    mbeta2 = np.exp(np.dot(np.concatenate([o2f * (np.tanh(c2f) - np.tanh(c1f)), \
            o1b * (np.tanh(c2b) - np.tanh(c4))]), Wout))
    
    mgamma1 = np.exp(np.dot(np.concatenate([o2f * (np.tanh(c1f * f2f) - np.tanh(c0 * f1f * f2f)), \
            o1b * (np.tanh(c1b) - np.tanh(c2b * f1b))]), Wout))
    mgamma2 = np.exp(np.dot(np.concatenate([o2f * (np.tanh(c2f) - np.tanh(c1f * f2f)), \
            o1b * (np.tanh(c2b * f1b) - np.tanh(c4 * f2b * f1b))]), Wout))
    
    m = Sequential()
    m.add(Masking(1, input_shape = (None, 3)))
    m.add(Bidirectional(BetaDecomposition(LSTM(units = 2, recurrent_activation = "sigmoid")), \
            weights = Wf + Wb, merge_mode = "concat"))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert(np.allclose(pred, np.array([mbeta1, mbeta2, np.ones_like(mbeta2)])))
    
    m = Sequential()
    m.add(Masking(1, input_shape = (None, 3)))
    m.add(Bidirectional(GammaDecomposition(LSTM(units = 2, recurrent_activation = "sigmoid")), \
            weights = Wf + Wb, merge_mode = "concat"))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]

    assert(np.allclose(pred, np.array([mgamma1, mgamma2, np.ones_like(mgamma2)])))

def test_unit_tests_DecompositionGRU_bidirectional():

    x1 = np.array([1,2,1])
    x2 = np.array([0,1,1])
    x3 = np.array([1,1,1])
    
    X = np.stack([x1,x2,x3])
    X = np.stack([X])
    
    Woutf = np.array([[2,0,0],[0,1,1]])
    Woutb = np.array([[6,7,8],[4,5,3]])
    Wout = np.concatenate([Woutf, Woutb], axis = 0)
    bout = np.zeros((3,))


    # Forward
    h0 = np.array([0,0])
        
    Wzf = np.array([[0,0], [0,1], [0,1]])
    Uzf = np.array([[0,1], [1,0]])
        
    Wrf = np.array([[2,0], [0,2], [0,1]])
    Urf = np.array([[0,2], [1,2]])
        
    Whf = np.array([[1,0], [0,0], [0,1]])
    Uhf = np.array([[0,2], [1,1]])
        
    z1f = sigmoid(np.dot(x1, Wzf) + np.dot(h0, Uzf))
    r1f = sigmoid(np.dot(x1, Wrf) + np.dot(h0, Urf))
    h_tilde1f = np.tanh(np.dot(x1, Whf) + np.dot(r1f * h0, Uhf))

    h1f = z1f * h0 + (1-z1f) * h_tilde1f
    
    z2f = sigmoid(np.dot(x2, Wzf) + np.dot(h1f, Uzf))
    r2f = sigmoid(np.dot(x2, Wrf) + np.dot(h1f, Urf))
    h_tilde2f = np.tanh(np.dot(x2, Whf) + np.dot(r2f * h1f, Uhf))

    h2f = z2f * h1f + (1-z2f) * h_tilde2f
   
    z3f = sigmoid(np.dot(x3, Wzf) + np.dot(h2f, Uzf))
    r3f = sigmoid(np.dot(x3, Wrf) + np.dot(h2f, Urf))
    h_tilde3f = np.tanh(np.dot(x3, Whf) + np.dot(r3f * h2f, Uhf))

    h3f = z3f * h2f + (1-z3f) * h_tilde3f

    bz = np.zeros((2,))
    br = np.zeros((2,))
    bh = np.zeros((2,))

    
    # Backward
    h4 = np.array([0,0])
        
    Wzb = np.array([[0,0], [0,1], [0,1]])
    Uzb = np.array([[0,1], [4,0]])
        
    Wrb = np.array([[2,0], [5,2], [0,1]])
    Urb = np.array([[0,2], [1,2]])
        
    Whb = np.array([[1,0], [0,6], [0,1]])
    Uhb = np.array([[0,2], [1,1]])
        
    z3b = sigmoid(np.dot(x3, Wzb) + np.dot(h4, Uzb))
    r3b = sigmoid(np.dot(x3, Wrb) + np.dot(h4, Urb))
    h_tilde3b = np.tanh(np.dot(x3, Whb) + np.dot(h4 * r3b, Uhb))
        
    h3b = z3b * h4 + (1-z3b) * h_tilde3b
    
    z2b = sigmoid(np.dot(x2, Wzb) + np.dot(h3b, Uzb))
    r2b = sigmoid(np.dot(x2, Wrb) + np.dot(h3b, Urb))
    h_tilde2b = np.tanh(np.dot(x2, Whb) + np.dot(h3b * r2b, Uhb))
        
    h2b = z2b * h3b + (1-z2b) * h_tilde2b
    
    z1b = sigmoid(np.dot(x1, Wzb) + np.dot(h2b, Uzb))
    r1b = sigmoid(np.dot(x1, Wrb) + np.dot(h2b, Urb))
    h_tilde1b = np.tanh(np.dot(x1, Whb) + np.dot(h2b * r1b, Uhb))
        
    h1b = z1b * h2b + (1-z1b) * h_tilde1b

    bz = np.zeros((2,))
    br = np.zeros((2,))
    bh = np.zeros((2,))
    
    Wf = [np.concatenate([Wzf, Wrf, Whf], -1), np.concatenate([Uzf, Urf, Uhf], -1), np.concatenate([bz, br, bh], -1)]
    Wb = [np.concatenate([Wzb, Wrb, Whb], -1), np.concatenate([Uzb, Urb, Uhb], -1), np.concatenate([bz, br, bh], -1)]
    

    rbeta1b = h1b - h2b
    rbeta2b = h2b - h3b
    rbeta3b = h3b - h4

    rbeta1f = h1f - h0
    rbeta2f = h2f - h1f
    rbeta3f = h3f - h2f
    
    rbeta1 = np.concatenate([rbeta1f, rbeta1b])
    rbeta2 = np.concatenate([rbeta2f, rbeta2b])
    rbeta3 = np.concatenate([rbeta3f, rbeta3b])
    
    beta1 = np.exp(np.dot(rbeta1, Wout))
    beta2 = np.exp(np.dot(rbeta2, Wout))
    beta3 = np.exp(np.dot(rbeta3, Wout))

    m = Sequential()
    m.add(Bidirectional(BetaDecomposition(GRU(units = 2, recurrent_activation = "sigmoid")), weights = Wf + Wb, input_shape = (None, 3)))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile("sgd", "mse")
    pred = m.predict(X)[0]
    
    assert(np.allclose(pred, np.array([beta1, beta2, beta3])))
    
    # gamma
    
    rgamma1b = h1b - h2b * z1b
    rgamma2b = h2b * z1b - h3b * z2b * z1b
    rgamma3b = h3b * z2b * z1b - h4 * z3b * z2b * z1b

    rgamma1f = h1f * z2f * z3f - h0 * z1f * z2f * z3f
    rgamma2f = h2f * z3f - h1f * z2f * z3f
    rgamma3f = h3f - h2f * z3f

    rgamma1 = np.concatenate([rgamma1f, rgamma1b])
    rgamma2 = np.concatenate([rgamma2f, rgamma2b])
    rgamma3 = np.concatenate([rgamma3f, rgamma3b])
   
    gamma1 = np.exp(np.dot(rgamma1, Wout))
    gamma2 = np.exp(np.dot(rgamma2, Wout))
    gamma3 = np.exp(np.dot(rgamma3, Wout))
    
    m = Sequential()
    m.add(Bidirectional(GammaDecomposition(GRU(units = 2, recurrent_activation = "sigmoid")), weights = Wf + Wb, input_shape = (None, 3)))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp", weights = [Wout, bout])))
    m.compile("sgd", "mse")
    pred = m.predict(X)[0]

    assert(np.allclose(pred, np.array([gamma1, gamma2, gamma3])))

    
    # does masking work?
    
    z2b = sigmoid(np.dot(x2, Wzb) + np.dot(h4, Uzb))
    r2b = sigmoid(np.dot(x2, Wrb) + np.dot(h4, Urb))
    h_tilde2b = np.tanh(np.dot(x2, Whb) + np.dot(h4 * r2b, Uhb))

    h2b = z2b * h4 + (1-z2b) * h_tilde2b
    
    z1b = sigmoid(np.dot(x1, Wzb) + np.dot(h2b, Uzb))
    r1b = sigmoid(np.dot(x1, Wrb) + np.dot(h2b, Urb))
    h_tilde1b = np.tanh(np.dot(x1, Whb) + np.dot(h2b * r1b, Uhb))

    h1b = z1b * h2b + (1-z1b) * h_tilde1b

    mbeta1 = np.exp(np.dot(np.concatenate([h1f - h0, h1b - h2b]), Wout))
    mbeta2 = np.exp(np.dot(np.concatenate([h2f - h1f, h2b - h4]), Wout))
    
    mgamma1 = np.exp(np.dot(np.concatenate([h1f * z2f - h0 * z1f * z2f, h1b - h2b * z1b]), Wout))
    mgamma2 = np.exp(np.dot(np.concatenate([h2f - h1f * z2f, h2b * z1b - h4 * z2b * z1b]), Wout))
    
    m = Sequential()
    m.add(Masking(1, input_shape = (None, 3)))
    m.add(Bidirectional(BetaDecomposition(GRU(units = 2, recurrent_activation = "sigmoid")), \
            weights = Wf + Wb, merge_mode = "concat"))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]

    assert(np.allclose(pred, np.array([mbeta1, mbeta2, np.ones_like(mbeta2)])))
    
    m = Sequential()
    m.add(Masking(1, input_shape = (None, 3)))
    m.add(Bidirectional(GammaDecomposition(GRU(units = 2, recurrent_activation = "sigmoid")), \
            weights = Wf + Wb, merge_mode = "concat"))
    m.add(TimeDistributed(Dense(units = 3, activation = "exp"), weights = [Wout, bout]))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]

    assert(np.allclose(pred, np.array([mgamma1, mgamma2, np.ones_like(mgamma2)])))
'''
if __name__ == '__main__':
    pytest.main([__file__])
