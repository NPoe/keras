from __future__ import absolute_import

import numpy as np

from ..models import Sequential
from ..layers import Dense
from ..callbacks import EarlyStopping
from ..utils.generic_utils import Progbar
from ..regularizers import l1

class Lime:
    def __init__(self, model, **kwargs):
        self.model = model

    def get_output_shape_for(self, input_shape):
        raise NotImplementedError()

    def call(self, X, mask = None, verbose = 1):
        l = []
        samples = X.shape[0]
        progbar = Progbar(target = samples, verbose = verbose)
        self.compile_lime_model(X)
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
        mode = "full", 
        pad = 0, 
        nb_samples = 10000, 
        minlength = 2, 
        maxlength = 7, 
        loss = "binary_crossentropy",
	activation = "sigmoid"):
	
        self.pad = pad
        self.loss = loss
        self.activation = activation
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
        assert self.mode in ("random", "full")
	
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

    def compile_lime_model(self, X):
        self.lime_model = Sequential()
        self.lime_model.add(Dense(input_shape=(X.shape[1],), units = 1, activation = self.activation, 
            use_bias = False, kernel_regularizer = l1()))
        self.lime_model.compile(loss = self.loss, optimizer = "rmsprop", metrics = ["accuracy"])
        self.initial_weights = self.lime_model.get_weights()

    def _lime(self, x):
        
        assert len(x.shape) == 1
        
        x_len = self.find_unpadded_length(x)
        samples = self.all_samples(x_len)
        nb_orig_samples = len(samples)
        
        if self.mode == "random":
            idx = np.random.choice(range(len(samples)), min(len(samples), self.nb_samples), replace = False)
            samples = [samples[i] for i in idx]
        
        masks = np.array([[0] * start + [1] * length + [0] * (x.shape[0] - start - length) for start, length in samples])
        
        input_original = []
	
        for mask in masks:
            nonzero = mask.nonzero()[0]
            relevant = np.concatenate([x[nonzero], np.ones((self.maxlength - len(nonzero),), dtype = 'int') * self.pad])
            input_original.append(relevant)
	    
        scores = self.model.predict(np.array(input_original), verbose = 0)
        
        weights = []

        for cl in range(self.model.output_shape[-1]):
            if self.loss == "binary_crossentropy":
                y = scores.argmax(axis = 1) == cl # 1 for samples where self.model has predicted cl, 0 elsewhere
                delta = 0.0001
            elif self.loss == "mse":
                y = scores[:,cl] # return the raw scores for cl
                delta = 0.00001
            
            self.lime_model.set_weights(self.initial_weights) # reset weights

            self.lime_model.fit(masks, y, verbose = 0, epochs = 10000, shuffle = True, \
                    callbacks = [EarlyStopping(monitor="loss", min_delta = delta, patience = 10)])

            weights_cl = self.lime_model.get_weights()[0].squeeze()[:x_len]
            weights_cl = np.concatenate([weights_cl, np.zeros((x.shape[0] - x_len,))], axis = 0)
            weights.append(weights_cl)
            
        weights = np.array(weights).transpose() # timesteps, #classes
        return weights


