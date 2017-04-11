from __future__ import absolute_import

import numpy as np

from ..models import Sequential
from ..layers import Dense
from ..callbacks import EarlyStopping
from ..utils.generic_utils import Progbar


class LimeTextSamplerNgram:
    def __init__(self, ngram):
        self.ngram = ngram
        assert self.ngram >= 1

    def get_starts_and_lengths(self, length):
        samples = []
        ngram = min(self.ngram, length)
        for i in range(length - ngram + 1):
            samples.append((i, ngram))
        return samples

class LimeTextSamplerRandom:
    def __init__(self, nb_samples, maxlength):
        self.nb_samples = nb_samples
        self.maxlength = maxlength

    def get_starts_and_lengths(self, length):
        samples = []
        maxlength = min(self.maxlength, length)

        for _ in range(self.nb_samples):
            start = np.random.randint(0, length + 1 - maxlength)
            duration = np.random.randint(1, maxlength + 1)
            samples.append((start, duration))
        return samples
        

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
    def __init__(self, model, sampler = LimeTextSamplerNgram(3), pad = 0, **kwargs):
        self.sampler = sampler
        self.pad = pad
        assert len(model.output_shape) == 2 # samples, #classes
        assert len(model.input_shape) >= 2 # samples, timesteps, ...
        super(TextLime, self).__init__(model, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape[:2] + (self.model.output_shape[-1],) # samples, timesteps, #classes

    def _lime(self, x):
        
        assert len(x.shape) == 1
        
        x_len = x.shape[0]
        if not self.pad is None:
            x_len = 0
            _x_len = 0
            for i in range(x.shape[0]):
                _x_len += 1
                if x[i] != self.pad:
                    x_len = _x_len

        samples = self.sampler.get_starts_and_lengths(x_len)
        X_s = np.stack([x[:x_len] for _ in range(len(samples))], axis = 0)

        masks = []

        for start, length in samples:
            masks.append([0] * start + [1] * length + [0] * (x_len - start - length))

        X_s *= np.array(masks)
        p_s = self.model.predict(X_s, verbose = 0)
        y_s = p_s.argmax(axis = 1)
        X_binary = X_s > 0


        weights = []

        for cl in range(self.model.output_shape[-1]):
            y_cl = y_s == cl # 1 for samples where self.model has predicted cl, 0 elsewhere
            
            simple_model = Sequential()
            simple_model.add(Dense(input_shape = (x_len,), output_dim = 1, activation = "sigmoid", bias = False))
            simple_model.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])
            simple_model.fit(X_binary, y_cl, verbose = 0, nb_epoch = 10000, \
                    callbacks = [EarlyStopping(monitor="loss", min_delta = 0.001, patience = 3)])

            weights_cl = simple_model.layers[-1].weights[0].container.storage[0].squeeze()
            weights_cl = np.concatenate([weights_cl, np.zeros((x.shape[0] - x_len,))], axis = 0)
            weights.append(weights_cl)
       
        weights = np.array(weights).transpose() # timesteps, #classes
        return weights


