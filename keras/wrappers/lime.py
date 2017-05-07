from __future__ import absolute_import

import numpy as np

from ..models import Sequential
from ..layers import Dense
from ..callbacks import EarlyStopping
from ..utils.generic_utils import Progbar


class Lime:
    def __init__(self, model, **kwargs):
        self.model = model

    def get_output_shape_for(self, input_shape):
        raise NotImplementedError()

    def call(self, X, mask = None, verbose = 1):
        l = []
        samples = X.shape[0]
        progbar = Progbar(target = samples, verbose = verbose)
        for i in range(samples):
            l.append(self._lime(X[i]))
            progbar.update(current = i+1)
        if verbose:
            print("")
        return np.array(l)

    def _lime(self, x, mask = None):
        raise NotImplementedError()


class TextLime(Lime):
    def __init__(self, model, 
        mode = "random", 
        pad = None, 
        nb_samples = 500, 
        minlength = 2, 
        maxlength = 7, 
        loss = "binary_crossentropy"):
	
        self.pad = pad
        self.loss = loss
        self.mode = mode
        self.minlength = minlength
        self.maxlength = maxlength
        self.nb_samples = nb_samples

        assert len(model.output_shape) == 2 # samples, #classes
        assert len(model.input_shape) >= 2 # samples, timesteps, ...
        assert self.loss in ("mse", "binary_crossentropy")
        assert self.minlength <= self.maxlength
        assert self.minlength > 0
        assert self.maxlength > 0
        assert self.mode in ("random", "fixed")
	
        super(TextLime, self).__init__(model)

    def get_output_shape_for(self, input_shape):
        return input_shape[:2] + (self.model.output_shape[-1],) # samples, timesteps, #classes
    
    def all_samples(self, inputlength):
        maxlength = min(self.maxlength, inputlength)
        minlength = min(self.minlength, inputlength)

        return sum([[(s,l) for s in range(0, inputlength - l)] for l in range(minlength, maxlength)], [])
    
    def find_unpadded_length(self, x):
        if not self.pad is None:
            _x_len = 0
            for i in range(x.shape[0]):
                if x[i] != self.pad:
                    _x_len = i
            return _x_len

        else:
            return x.shape[0]

    def _lime(self, x):
        
        assert len(x.shape) == 1
        
        x_len = self.find_unpadded_length(x)
        samples = self.all_samples(x_len)
        nb_orig_samples = len(samples)
        
        if self.mode == "random":
            np.random.shuffle(samples)
            samples = samples[:min(len(samples), self.nb_samples)]
        
        X_s = np.stack([x[:x_len] for _ in range(len(samples))], axis = 0)

        masks = []

        for start, length in samples:
            masks.append([0] * start + [1] * length + [0] * (x_len - start - length))

        X_s *= np.array(masks)
        X_binary = X_s > 0 # 1 everywhere where we have not masked, 0 elsewhere

        p_s = self.model.predict(X_s, verbose = 0)

        weights = []

        MIN_DELTA = {"mse": 0.0001, "binary_crossentropy": 0.001}

        for cl in range(self.model.output_shape[-1]):
            if self.loss == "binary_crossentropy":
                y = p_s.argmax(axis = 1) == cl # 1 for samples where self.model has predicted cl, 0 elsewhere
            elif self.loss == "mse":
                y = p_s[:,cl]


            simple_model = Sequential()
            simple_model.add(Dense(input_shape = (x_len,), units = 1, activation = "sigmoid", use_bias = False))
            simple_model.compile(loss = self.loss, optimizer = "rmsprop", metrics = ["accuracy"])
            simple_model.fit(X_binary, y, verbose = 0, epochs = 2000, \
                    callbacks = [EarlyStopping(monitor="loss", min_delta = MIN_DELTA[self.loss], patience = 5)])

            weights_cl = simple_model.layers[-1].weights[0].container.storage[0].squeeze()
            weights_cl = np.concatenate([weights_cl, np.zeros((x.shape[0] - x_len,))], axis = 0)
            weights.append(weights_cl)
            
        weights = np.array(weights).transpose() # timesteps, #classes
        return weights


