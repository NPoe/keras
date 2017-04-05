import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test
from keras.layers import recurrent, embeddings, decomposition
from keras.layers.recurrent import LSTM, GRU
from keras.layers.decomposition import *
from keras.models import Sequential
from keras.layers.core import Masking, Dense, Lambda
from keras.layers.wrappers import TimeDistributed
from keras import regularizers
from keras.utils.test_utils import keras_test

from keras import backend as K

nb_samples, timesteps, embedding_dim, output_dim, num_classes = 2, 5, 6, 3, 4
embedding_num = 12

def test_ErasureLayer():
    model = Sequential()
    model.add(ErasureLayer(GRU(output_dim = output_dim, \
            return_sequences = False, return_all_states = False), input_shape = (timesteps, embedding_dim)))

    model.compile(optimizer='sgd', loss='mse')
    out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
    assert(out.shape == (nb_samples, timesteps, output_dim))

'''
def test_OmegaLayer():
    model = Sequential()
    model.add(GRU(input_shape=(timesteps, embedding_dim), output_dim = output_dim, \
            return_sequences = True, return_all_states = True))
    model.add(OmegaLayer())
    model.add(TimeDistributed(Dense(output_dim = num_classes, activation = "linear")))
    model.add(TimeDistributed(Lambda(lambda x : K.exp(x))))

    model.compile(optimizer='sgd', loss='mse')
    out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
    assert(out.shape == (nb_samples, timesteps, num_classes))

def test_DeltaLayer():
    model = Sequential()
    model.add(GRU(input_shape=(timesteps, embedding_dim), output_dim = output_dim, \
            return_sequences = True, return_all_states = True))
    model.add(DeltaLayer())
    model.add(TimeDistributed(Dense(output_dim = num_classes, activation = "linear")))
    model.add(TimeDistributed(Lambda(lambda x : K.exp(x))))

    model.compile(optimizer='sgd', loss='mse')
    out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
    assert(out.shape == (nb_samples, timesteps, num_classes))

def test_PhiLayer():
    model = Sequential()
    model.add(LSTM(input_shape=(timesteps, embedding_dim), output_dim = output_dim, \
            return_sequences = True, return_all_states = True))
    model.add(PhiLayer())
    model.add(TimeDistributed(Dense(output_dim = num_classes, activation = "linear")))
    model.add(TimeDistributed(Lambda(lambda x : K.exp(x))))

    model.compile(optimizer='sgd', loss='mse')
    out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
    assert(out.shape == (nb_samples, timesteps, num_classes))
def test_EpsLayer():
    model = Sequential()
    model.add(GRU(input_shape=(timesteps, embedding_dim), output_dim = output_dim, \
            return_sequences = True, return_all_states = True))
    model.add(EpsLayer())
    model.add(TimeDistributed(Dense(output_dim = num_classes, activation = "linear")))
    model.add(TimeDistributed(Lambda(lambda x : K.exp(x))))

    model.compile(optimizer='sgd', loss='mse')
    out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
    assert(out.shape == (nb_samples, timesteps, num_classes))

def test_BetaLayer():
    model = Sequential()
    model.add(LSTM(input_shape=(timesteps, embedding_dim), output_dim = output_dim, \
            return_sequences = True, return_all_states = True))
    model.add(BetaLayer())
    model.add(TimeDistributed(Dense(output_dim = num_classes, activation = "linear")))
    model.add(TimeDistributed(Lambda(lambda x : K.exp(x))))
    
    model.compile(optimizer='sgd', loss='mse')
    out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
    assert(out.shape == (nb_samples, timesteps, num_classes))

def test_GammaLayer():
    model = Sequential()
    model.add(LSTM(input_shape=(timesteps, embedding_dim), output_dim = output_dim, \
            return_sequences = True, return_all_states = True))
    model.add(GammaLayer())
    model.add(TimeDistributed(Dense(output_dim = num_classes, activation = "linear")))
    model.add(TimeDistributed(Lambda(lambda x : K.exp(x))))
    
    model.compile(optimizer='sgd', loss='mse')
    out = model.predict(np.random.random((nb_samples, timesteps, embedding_dim)))
    assert(out.shape == (nb_samples, timesteps, num_classes))

def test_unit_tests_Decomposition():

    x1 = np.array([1,2,1])
    x2 = np.array([0,1,1])
    x3 = np.array([1,1,1])
    
    X = np.stack([x1,x2,x3])
    X = np.stack([X])
    
    Wout = np.array([[2,0,0],[0,1,1]])
    bout = np.zeros((3,))

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

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
    m.add(BetaLayer(activation = "tanh"))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "linear", weights = [Wout, bout])))
    m.add(TimeDistributed(Lambda(lambda x:K.exp(x))))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([beta1, beta2, beta3]), pred)
    
    gamma1 = np.exp(np.dot(o3 * (np.tanh(f2 * f3 * c1) - np.tanh(f1 * f2 * f3 * c0)), Wout))
    gamma2 = np.exp(np.dot(o3 * (np.tanh(f3 * c2) - np.tanh(f2 * f3 * c1)), Wout))
    gamma3 = np.exp(np.dot(o3 * (np.tanh(c3) - np.tanh(f3 * c2)), Wout))
    
    m = Sequential()
    m.add(LSTM(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(GammaLayer(activation = "tanh"))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "linear", weights = [Wout, bout])))
    m.add(TimeDistributed(Lambda(lambda x:K.exp(x))))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([gamma1, gamma2, gamma3]), pred)
    
    phi1 = np.exp(np.dot(o3 * np.tanh(f2 * f3 * i1 * h_tilde1), Wout))
    phi2 = np.exp(np.dot(o3 * np.tanh(f3 * i2 * h_tilde2), Wout))
    phi3 = np.exp(np.dot(o3 * np.tanh(i3 * h_tilde3), Wout))
    
    m = Sequential()
    m.add(LSTM(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(PhiLayer(activation = "tanh"))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "linear", weights = [Wout, bout])))
    m.add(TimeDistributed(Lambda(lambda x:K.exp(x))))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([phi1, phi2, phi3]), pred)
    
    mbeta1 = np.exp(np.dot(o2 * (np.tanh(c1) - np.tanh(c0)), Wout))
    mbeta2 = np.exp(np.dot(o2 * (np.tanh(c2) - np.tanh(c1)), Wout))
    mbeta3 = np.ones_like(mbeta2)
    
    m = Sequential()
    m.add(Masking(input_shape=(None,3), mask_value = 1))
    m.add(LSTM(output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(BetaLayer(activation = "tanh"))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "linear", weights = [Wout, bout])))
    m.add(TimeDistributed(Lambda(lambda x:K.exp(x))))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")

    pred = m.predict(X)[0]

    assert np.allclose(np.array([mbeta1, mbeta2, mbeta3]), pred)
    
    mgamma1 = np.exp(np.dot(o2 * (np.tanh(f2 * c1) - np.tanh(f1*f2*c0)), Wout))
    mgamma2 = np.exp(np.dot(o2 * (np.tanh(c2) - np.tanh(f2 * c1)), Wout))
    mgamma3 = np.ones_like(mgamma2)

    m = Sequential()
    m.add(Masking(input_shape=(None,3), mask_value = 1))
    m.add(LSTM(output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(GammaLayer(activation = "tanh"))
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "linear", weights = [Wout, bout])))
    m.add(TimeDistributed(Lambda(lambda x:K.exp(x))))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")

    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([mgamma1, mgamma2, mgamma3]), pred)

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
    
    eps1 = np.exp(np.dot(z3 * z2 * (1-z1) * h_tilde1, Wout))
    eps2 = np.exp(np.dot(z3 * (1-z2) * h_tilde2, Wout))
    eps3 = np.exp(np.dot((1-z3) * h_tilde3, Wout))
    
    
    m = Sequential()
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(EpsLayer())
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "linear", weights = [Wout, bout])))
    m.add(TimeDistributed(Lambda(lambda x:K.exp(x))))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]

    assert np.allclose(np.array([eps1, eps2, eps3]), pred)
    
    
    delta1 = np.exp(np.dot(h1-h0, Wout))
    delta2 = np.exp(np.dot(h2-h1, Wout))
    delta3 = np.exp(np.dot(h3-h2, Wout))
    
    m = Sequential()
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(DeltaLayer())
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "linear", weights = [Wout, bout])))
    m.add(TimeDistributed(Lambda(lambda x:K.exp(x))))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]

    assert np.allclose(np.array([delta1, delta2, delta3]), pred)
    
    omega1 = np.exp(np.dot((z3 * z2 * h1) - (z3 * z2 * z1 * h0), Wout))
    omega2 = np.exp(np.dot((z3 * h2) - (z3 * z2 * h1), Wout))
    omega3 = np.exp(np.dot(h3 - (z3 * h2), Wout))
    
    m = Sequential()
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(OmegaLayer())
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "linear", weights = [Wout, bout])))
    m.add(TimeDistributed(Lambda(lambda x:K.exp(x))))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]

    assert np.allclose(np.array([omega1, omega2, omega3]), pred)
    
    meps1 = np.exp(np.dot(z2 * (1-z1) * h_tilde1, Wout))
    meps2 = np.exp(np.dot((1-z2) * h_tilde2, Wout))
    meps3 = np.ones_like(meps2)

    m = Sequential()
    m.add(Masking(input_shape=(None,3), mask_value = 1))
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(EpsLayer())
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "linear", weights = [Wout, bout])))
    m.add(TimeDistributed(Lambda(lambda x:K.exp(x))))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([meps1, meps2, meps3]), pred)
    
    mdelta1 = np.exp(np.dot(h1 - h0, Wout))
    mdelta2 = np.exp(np.dot(h2 - h1, Wout))
    mdelta3 = np.ones_like(mdelta2)

    m = Sequential()
    m.add(Masking(input_shape=(None,3), mask_value = 1))
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(DeltaLayer())
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "linear", weights = [Wout, bout])))
    m.add(TimeDistributed(Lambda(lambda x:K.exp(x))))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([mdelta1, mdelta2, mdelta3]), pred)
    
    
    momega1 = np.exp(np.dot((z2 * h1) - (z2 * z1 * h0), Wout))
    momega2 = np.exp(np.dot(h2 - (z2 * h1), Wout))
    momega3 = np.ones_like(momega2)

    m = Sequential()
    m.add(Masking(input_shape=(None,3), mask_value = 1))
    m.add(GRU(input_shape = (None, 3), output_dim = 2, weights = W, inner_activation = "sigmoid", activation = "tanh", \
            return_sequences = True, return_all_states = True))
    m.add(OmegaLayer())
    m.add(TimeDistributed(Dense(output_dim = 3, activation = "linear", weights = [Wout, bout])))
    m.add(TimeDistributed(Lambda(lambda x:K.exp(x))))
    m.compile(loss = "categorical_crossentropy", optimizer = "adagrad")
    pred = m.predict(X)[0]
    
    assert np.allclose(np.array([momega1, momega2, momega3]), pred)
'''

def test_unit_tests_Erasure():

    x1 = np.array([1,2,1])
    x2 = np.array([0,1,1])
    x3 = np.array([1,1,1])
    
    X = np.stack([x1,x2,x3])
    X = np.stack([X,X,X])
    
    Wout = np.array([[2,0,0],[0,1,1]])
    bout = np.zeros((3,))

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

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
    model.add(ErasureLayer(LSTM(output_dim = 2, weights = W), input_shape=(3,3)))
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
    
    model = Sequential()
    model.add(ErasureLayer(GRU(output_dim = 2, weights = W), input_shape=(3,3)))
    model.compile(optimizer='sgd', loss='mse')
    outerasure = model.predict(np.array([[x1,x2,x3]]))

    assert(np.allclose(np.array([outnorm-out1, outnorm-out2, outnorm-out3]).squeeze(), outerasure))

if __name__ == '__main__':
    pytest.main([__file__])
