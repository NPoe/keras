import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test
from keras.layers import recurrent, embeddings, decomposition, Input
from keras.layers.recurrent import LSTM, GRU
from keras.layers.decomposition import *
from keras.models import Sequential, Model
from keras.layers.core import Masking, Dense, Lambda, Merge
from keras.layers.wrappers import TimeDistributed, Bidirectional, ErasureWrapper
from keras import regularizers
from keras.utils.test_utils import keras_test

from keras import backend as K

nb_samples, timesteps, embedding_dim, output_dim, num_classes = 2, 5, 6, 3, 4
embedding_num = 12

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def test_ErasureWrapper():
    for i in range(1,4):
        model = Sequential()
        model.add(ErasureWrapper(GRU(output_dim = output_dim, \
                return_sequences = False, return_all_states = False), input_shape = (timesteps, embedding_dim), ngram = i))

        model.compile(optimizer='sgd', loss='mse')
        out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
        assert(out.shape == (nb_samples, timesteps - i + 1, output_dim))

def test_GammaLayerGRU():
    for i in range(1,4):
        model = Sequential()
        model.add(GRU(input_shape=(timesteps, embedding_dim), output_dim = output_dim, \
                return_sequences = True, return_all_states = True))
        model.add(GammaLayerGRU(ngram = i))
        model.add(TimeDistributed(Dense(output_dim = num_classes, activation = "exp")))

        model.compile(optimizer='sgd', loss='mse')
        out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
        assert(out.shape == (nb_samples, timesteps - i + 1, num_classes))

def test_GammaLayerGRUBidirectional():
    
    inp = Input(shape = (timesteps, embedding_dim))
    gru = Bidirectional(GRU(input_shape = (timesteps, embedding_dim), output_dim = output_dim,
            return_sequences = True, return_all_states = True), merge_mode = "concat")(inp)
    gamma = Bidirectional(GammaLayerGRU(), input_mode = "split", merge_mode = "concat")(gru)

    outp = TimeDistributed(Dense(output_dim = num_classes, activation = "exp"))(gamma)

    model = Model([inp], [gamma, outp])
    model.compile(optimizer = 'sgd', loss = 'mse')
    out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
    assert(out[0].shape == (nb_samples, timesteps, output_dim * 2))
    assert(out[1].shape == (nb_samples, timesteps, num_classes))


def test_BetaLayerGRU():
    for i in range(1,4):
        model = Sequential()
        model.add(GRU(input_shape=(timesteps, embedding_dim), output_dim = output_dim, \
                return_sequences = True, return_all_states = True))
        model.add(BetaLayerGRU(ngram = i))
        model.add(TimeDistributed(Dense(output_dim = num_classes, activation = "exp")))

        model.compile(optimizer='sgd', loss='mse')
        out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
        assert(out.shape == (nb_samples, timesteps - i + 1, num_classes))

def test_BetaLayerLSTM():
    for i in range(1,4):
        model = Sequential()
        model.add(LSTM(input_shape=(timesteps, embedding_dim), output_dim = output_dim, \
                return_sequences = True, return_all_states = True))
        model.add(BetaLayerLSTM(ngram = i))
        model.add(TimeDistributed(Dense(output_dim = num_classes, activation = "exp")))
    
        model.compile(optimizer='sgd', loss='mse')
        out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
        assert(out.shape == (nb_samples, timesteps - i + 1, num_classes))

def test_GammaLayerLSTM():
    for i in range(1,4):
        model = Sequential()
        model.add(LSTM(input_shape=(timesteps, embedding_dim), output_dim = output_dim, \
                return_sequences = True, return_all_states = True))
        model.add(GammaLayerLSTM(ngram = i))
        model.add(TimeDistributed(Dense(output_dim = num_classes, activation = "exp")))
    
        model.compile(optimizer='sgd', loss='mse')
        out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
        assert(out.shape == (nb_samples, timesteps - i + 1, num_classes))


def test_unit_tests_Decomposition():

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

    W = [Wi, Ui, bi, Wc, Uc, bc, Wf, Uf, bf, Wo, Uo, bo]

    m = Sequential()
    m.add(LSTM(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]

    assert np.allclose(np.array([h1, h2, h3]), pred[:,0,:])
    assert np.allclose(np.array([c1, c2, c3]), pred[:,1,:])
    assert np.allclose(np.array([i1, i2, i3]), pred[:,2,:])
    assert np.allclose(np.array([f1, f2, f3]), pred[:,3,:])
    assert np.allclose(np.array([o1, o2, o3]), pred[:,4,:])
    assert np.allclose(np.array([h_tilde1, h_tilde2, h_tilde3]), pred[:,5,:])
    
    beta1 = np.exp(np.dot(o3 * (np.tanh(c1) - np.tanh(c0)), Wout))
    beta2 = np.exp(np.dot(o3 * (np.tanh(c2) - np.tanh(c1)), Wout))
    beta3 = np.exp(np.dot(o3 * (np.tanh(c3) - np.tanh(c2)), Wout))
    
    m = Sequential()
    m.add(LSTM(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(BetaLayerLSTM(activation = "tanh"))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([beta1, beta2, beta3]), pred)
    
    gamma1 = np.exp(np.dot(o3 * (np.tanh(f2 * f3 * c1) - np.tanh(f1 * f2 * f3 * c0)), Wout))
    gamma2 = np.exp(np.dot(o3 * (np.tanh(f3 * c2) - np.tanh(f2 * f3 * c1)), Wout))
    gamma3 = np.exp(np.dot(o3 * (np.tanh(c3) - np.tanh(f3 * c2)), Wout))
    
    m = Sequential()
    m.add(LSTM(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(GammaLayerLSTM(activation = "tanh"))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([gamma1, gamma2, gamma3]), pred)
    

    mbeta1 = np.exp(np.dot(o2 * (np.tanh(c1) - np.tanh(c0)), Wout))
    mbeta2 = np.exp(np.dot(o2 * (np.tanh(c2) - np.tanh(c1)), Wout))
    
    m = Sequential()
    m.add(Masking(input_shape=(None,3), mask_value = 1))
    m.add(LSTM(output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(BetaLayerLSTM(activation = "tanh"))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")

    pred = m.predict(X)[0]

    assert np.allclose(np.array([mbeta1, mbeta2, mbeta2]), pred)
    
    mgamma1 = np.exp(np.dot(o2 * (np.tanh(f2 * c1) - np.tanh(f1*f2*c0)), Wout))
    mgamma2 = np.exp(np.dot(o2 * (np.tanh(c2) - np.tanh(f2 * c1)), Wout))

    m = Sequential()
    m.add(Masking(input_shape=(None,3), mask_value = 1))
    m.add(LSTM(output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(GammaLayerLSTM(activation = "tanh"))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")

    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([mgamma1, mgamma2, mgamma2]), pred)
    
    # with bigram
    
    bbeta2 = np.exp(np.dot(o3 * (np.tanh(c2) - np.tanh(c0)), Wout))
    bbeta3 = np.exp(np.dot(o3 * (np.tanh(c3) - np.tanh(c1)), Wout))

    m = Sequential()
    m.add(LSTM(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(BetaLayerLSTM(ngram = 2))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([bbeta2, bbeta3]), pred)
    
    bgamma2 = np.exp(np.dot(o3 * (np.tanh(f3 * c2) - np.tanh(f3 * f2 * f1 * c0)), Wout))
    bgamma3 = np.exp(np.dot(o3 * (np.tanh(c3) - np.tanh(f3 * f2 * c1)), Wout))

    m = Sequential()
    m.add(LSTM(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(GammaLayerLSTM(ngram = 2))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([bgamma2, bgamma3]), pred)

    tbeta = np.exp(np.dot(o3 * np.tanh(c3), Wout))
    
    m = Sequential()
    m.add(LSTM(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(BetaLayerLSTM(ngram = 3))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([tbeta]), pred)
    
    m = Sequential()
    m.add(LSTM(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(GammaLayerLSTM(ngram = 3))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([tbeta]), pred)
    

    # GRU
    
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

    W = [Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh]

    m = Sequential()
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]

    assert np.allclose(np.array([h1, h2, h3]), pred[:,0,:])
    assert np.allclose(np.array([z1, z2, z3]), pred[:,1,:])
    assert np.allclose(np.array([r1, r2, r3]), pred[:,2,:])
    assert np.allclose(np.array([h_tilde1, h_tilde2, h_tilde3]), pred[:,3,:])
    
    
    delta1 = np.exp(np.dot(h1-h0, Wout))
    delta2 = np.exp(np.dot(h2-h1, Wout))
    delta3 = np.exp(np.dot(h3-h2, Wout))
    
    m = Sequential()
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(BetaLayerGRU())
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]

    assert np.allclose(np.array([delta1, delta2, delta3]), pred)
    
    omega1 = np.exp(np.dot((z3 * z2 * h1) - (z3 * z2 * z1 * h0), Wout))
    omega2 = np.exp(np.dot((z3 * h2) - (z3 * z2 * h1), Wout))
    omega3 = np.exp(np.dot(h3 - (z3 * h2), Wout))
    
    m = Sequential()
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(GammaLayerGRU())
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]

    assert np.allclose(np.array([omega1, omega2, omega3]), pred)
    
    mdelta1 = np.exp(np.dot(h1 - h0, Wout))
    mdelta2 = np.exp(np.dot(h2 - h1, Wout))

    m = Sequential()
    m.add(Masking(input_shape=(None,3), mask_value = 1))
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(BetaLayerGRU())
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([mdelta1, mdelta2, mdelta2]), pred)
    
    momega1 = np.exp(np.dot((z2 * h1) - (z2 * z1 * h0), Wout))
    momega2 = np.exp(np.dot(h2 - (z2 * h1), Wout))

    m = Sequential()
    m.add(Masking(input_shape=(None,3), mask_value = 1))
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(GammaLayerGRU())
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([momega1, momega2, momega2]), pred)


    # with bigram
    
    bdelta2 = np.exp(np.dot(h2 - h0, Wout))
    bdelta3 = np.exp(np.dot(h3 - h1, Wout))

    m = Sequential()
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(BetaLayerGRU(ngram = 2))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([bdelta2, bdelta3]), pred)
    
    bomega2 = np.exp(np.dot((z3 * h2) - (z3 * z2 * z1 * h0), Wout))
    bomega3 = np.exp(np.dot(h3 - (z3 * z2 * h1), Wout))

    m = Sequential()
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(GammaLayerGRU(ngram = 2))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([bomega2, bomega3]), pred)

    tdelta = np.exp(np.dot(h3, Wout))
    m = Sequential()
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(BetaLayerGRU(ngram = 3))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([tdelta]), pred)
    
    m = Sequential()
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(GammaLayerGRU(ngram = 3))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "exp", weights = [Wout, bout])))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([tdelta]), pred)


def test_unit_tests_Erasure():

    x1 = np.array([1,2,1])
    x2 = np.array([0,1,1])
    x3 = np.array([1,1,1])
    
    X = np.stack([x1,x2,x3])
    X = np.stack([X,X,X])
    
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
    
    bi = np.zeros((2,))
    bc = np.zeros((2,))
    bf = np.zeros((2,))
    bo = np.zeros((2,))

    W = [Wi, Ui, bi, Wc, Uc, bc, Wf, Uf, bf, Wo, Uo, bo]

    # with nothing missing
    
    model = Sequential()
    model.add(Masking(input_shape=(3,3), mask_value = 0))
    model.add(LSTM(output_dim = 2, return_sequences = False, return_all_states = False, weights = W))
    model.compile(optimizer='sgd', loss='mse')
    outnorm = model.predict(np.array([[x1, x2, x3]]))
    
    # with x missing
    
    out1 = model.predict(np.array([[np.zeros_like(x1), x2, x3]]))
    out2 = model.predict(np.array([[x1, np.zeros_like(x2), x3]]))
    out3 = model.predict(np.array([[x1, x2, np.zeros_like(x3)]]))
    
    model = Sequential()
    model.add(ErasureWrapper(LSTM(output_dim = 2, weights = W), input_shape=(3,3)))
    model.compile(optimizer='sgd', loss='mse')
    outerasure = model.predict(np.array([[x1,x2,x3]]))

    assert(np.allclose(np.array([outnorm-out1, outnorm-out2, outnorm-out3]).squeeze(), outerasure))
    
    # GRU

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

    W = [Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh]
    
    # with nothing missing
    
    model = Sequential()
    model.add(Masking(input_shape=(3,3), mask_value = 0))
    model.add(GRU(output_dim = 2, return_sequences = False, return_all_states = False, weights = W))
    model.compile(optimizer='sgd', loss='mse')
    outnorm = model.predict(np.array([[x1, x2, x3]]))
    
    # with x missing
    
    out1 = model.predict(np.array([[np.zeros_like(x1), x2, x3]]))
    out2 = model.predict(np.array([[x1, np.zeros_like(x2), x3]]))
    out3 = model.predict(np.array([[x1, x2, np.zeros_like(x3)]]))
    
    # with 2 x missing

    out12 = model.predict(np.array([[np.zeros_like(x1), np.zeros_like(x2), x3]]))
    out23 = model.predict(np.array([[x1, np.zeros_like(x2), np.zeros_like(x3)]]))
    
    model = Sequential()
    model.add(ErasureWrapper(GRU(output_dim = 2, weights = W), input_shape=(3,3), ngram = 1))
    model.compile(optimizer='sgd', loss='mse')
    outerasure = model.predict(np.array([[x1,x2,x3]]))

    assert(np.allclose(np.array([outnorm-out1, outnorm-out2, outnorm-out3]).squeeze(), outerasure))
    
    model = Sequential()
    model.add(ErasureWrapper(GRU(output_dim = 2, weights = W), input_shape=(3,3), ngram = 2))
    model.compile(optimizer='sgd', loss='mse')
    outerasure = model.predict(np.array([[x1,x2,x3]]))

    assert(np.allclose(np.array([outnorm-out12, outnorm-out23]).squeeze(), outerasure))

def test_unit_tests_Decomposition_bidirectional():

    x1 = np.array([1,2,1])
    x2 = np.array([0,1,1])
    x3 = np.array([1,1,1])
    
    X = np.stack([x1,x2,x3])
    X = np.stack([X])
    
    Wout = np.array([[2,0,0],[0,1,1],[6,7,8],[4,5,3]])
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

    Wf = [Wif, Uif, bi, Wcf, Ucf, bc, Wff, Uff, bf, Wof, Uof, bo]
    
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
    
    Wb = [Wib, Uib, bi, Wcb, Ucb, bc, Wfb, Ufb, bf, Wob, Uob, bo]

    inp = Input(shape=(3,3))
   
    LSTM_bidir = Bidirectional(LSTM(input_shape= (None, 3), output_dim = 2,
        return_sequences = True, return_all_states = True, activation = "tanh", 
        inner_activation = "sigmoid"), weights = Wf + Wb, merge_mode = "concat")(inp)

    beta_bidir = Bidirectional(BetaLayerLSTM(), input_mode = "split", merge_mode = "concat")(LSTM_bidir)

    outp = TimeDistributed(Dense(output_dim = 3, weights = [Wout, bout], activation = "exp"))(beta_bidir)
    m = Model([inp], [outp])
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    beta1 = np.exp(np.dot(np.concatenate([o3f * (np.tanh(c1f) - np.tanh(c0)), o1b * (np.tanh(c1b) - np.tanh(c2b))]), Wout))
    beta2 = np.exp(np.dot(np.concatenate([o3f * (np.tanh(c2f) - np.tanh(c1f)), o1b * (np.tanh(c2b) - np.tanh(c3b))]), Wout))
    beta3 = np.exp(np.dot(np.concatenate([o3f * (np.tanh(c3f) - np.tanh(c2f)), o1b * (np.tanh(c3b) - np.tanh(c4))]), Wout))

    assert(np.allclose(pred, np.array([beta1, beta2, beta3])))

    gamma_bidir = Bidirectional(GammaLayerLSTM(), input_mode = "split", merge_mode = "concat")(LSTM_bidir)
    outp_g = TimeDistributed(Dense(output_dim = 3, weights = [Wout, bout], activation = "exp"))(gamma_bidir)

    m_g = Model([inp], [outp_g])
    m_g.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred_g = m_g.predict(X)[0]
    
    gamma1 = np.exp(np.dot(np.concatenate([o3f * (np.tanh(c1f * f2f * f3f) - np.tanh(c0 * f1f * f2f * f3f)), \
            o1b * (np.tanh(c1b) - np.tanh(c2b * f1b))]), Wout))
    gamma2 = np.exp(np.dot(np.concatenate([o3f * (np.tanh(c2f * f3f) - np.tanh(c1f * f2f * f3f)), \
            o1b * (np.tanh(c2b * f1b) - np.tanh(c3b * f2b * f1b))]), Wout))
    gamma3 = np.exp(np.dot(np.concatenate([o3f * (np.tanh(c3f) - np.tanh(c2f * f3f)), \
            o1b * (np.tanh(c3b * f2b * f1b) - np.tanh(c4 * f3b * f2b * f1b))]), Wout))
    
    assert(np.allclose(pred_g, np.array([gamma1, gamma2, gamma3])))
    

    # does masking work?
    masked = Masking(1)(inp)
    LSTM_bidir = Bidirectional(LSTM(input_shape= (None, 3), output_dim = 2,
        return_sequences = True, return_all_states = True, activation = "tanh", 
        inner_activation = "sigmoid"), weights = Wf + Wb, merge_mode = "concat")(masked)
    gamma_bidir = Bidirectional(GammaLayerLSTM(), input_mode = "split", merge_mode = "concat")(LSTM_bidir)
    outp_mg = TimeDistributed(Dense(output_dim = 3, weights = [Wout, bout], activation = "exp"))(gamma_bidir)

    m_mg = Model([inp], [outp_mg])
    pred_mg = m_mg.predict(X)[0]
    
    
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
    
    
    mgamma1 = np.exp(np.dot(np.concatenate([o2f * (np.tanh(c1f * f2f) - np.tanh(c0)), \
            o1b * (np.tanh(c1b) - np.tanh(c2b * f1b))]), Wout))
    mgamma2 = np.exp(np.dot(np.concatenate([o2f * (np.tanh(c2f) - np.tanh(c1f * f2f)), \
            o1b * (np.tanh(c2b * f1b) - np.tanh(c4))]), Wout))


    assert(np.allclose(pred_mg[:2], np.array([mgamma1, mgamma2])))





if __name__ == '__main__':
    pytest.main([__file__])
