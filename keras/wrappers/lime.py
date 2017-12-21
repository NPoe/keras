from __future__ import absolute_import

import numpy as np

from ..models import Sequential
from ..layers import Dense
from ..callbacks import EarlyStopping
from ..utils.generic_utils import Progbar
from ..regularizers import l1
from ..optimizers import SGD

class Lime:
    def __init__(self, model, **kwargs):
        self.model = model

    def get_output_shape_for(self, input_shape):
        raise NotImplementedError()

    def call(self, X, mask = None, verbose = 1, out = None):
        l = []
        samples = X.shape[0]
        if verbose:
            print("LIME with", self.mode, "mode")
            if self.mode == "random":
                print ("N =", self.nb_samples)
        progbar = Progbar(target = samples, verbose = verbose)
        self.compile_lime_model(X)
        for i in range(samples):
            if out is None: o = None
            else: o = out[i]
            l.append(self._lime(X[i], out = o))
            progbar.update(current = i+1)
        if verbose:
            print("")
        return np.array(l)

    def _lime(self, x, mask = None):
        raise NotImplementedError()


class TextLime(Lime):
    def __init__(self, model, 
        mode = "random", 
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
        maxlength = min(self.maxlength, inputlength + 1)
        minlength = min(self.minlength, inputlength)

        return sum([[(s,l) for s in range(0, inputlength + 1 - l)] for l in range(minlength, maxlength)], [])
    
    def find_unpadded_length(self, x):
        if not self.pad is None:
            _x_len = 0
            for i in range(x.shape[0]):
                if x[i] != self.pad:
                    _x_len = i+1
            return _x_len

        else:
            return x.shape[0]

    def compile_lime_model(self, X):
        self.lime_model = Sequential()
        np.random.seed(12345)
        self.lime_model.add(Dense(input_shape=(X.shape[1],), units = 1, activation = self.activation, 
            use_bias = False, kernel_regularizer = l1()))
        self.lime_model.compile(loss = self.loss, optimizer = SGD(momentum = 0.9, nesterov = True), metrics = ["accuracy"])
        self.initial_weights = self.lime_model.get_weights()

    def _lime(self, x, out = None):
        
        assert len(x.shape) == 1
        
        x_len = self.find_unpadded_length(x)
        numclasses = self.model.output_shape[-1]
        
        if x_len == 0:
            return np.zeros((x.shape[0], numclasses))
	
        samples = self.all_samples(x_len)
        nb_orig_samples = len(samples)
        
        if self.mode == "random":
            idx = np.random.choice(range(len(samples)), min(len(samples), self.nb_samples), replace = False)
            samples = [samples[i] for i in idx]
        
        masks_by_length = {}
        for start, length in samples:
            if not length in masks_by_length:
                masks_by_length[length] = []
            masks_by_length[length].append(np.array([0] * start + [1] * length + [0] * (x.shape[0] - start - length)))


        all_masks = []
        all_scores = []

        for length in masks_by_length.keys():
            input_original = np.stack([x[mask.nonzero()[0]] for mask in masks_by_length[length]], axis = 0)
            masks = np.stack(masks_by_length[length], axis = 0)
            scores = self.model.predict(input_original, verbose = 0)
            
            all_scores.append(scores)
            all_masks.append(masks)
        
        # masks & scores are now ordered by length; this does not matter since fit() will shuffle them
        all_scores = np.concatenate(all_scores, axis = 0)
        all_masks = np.concatenate(all_masks, axis = 0)
        all_weights = []

        if out is None: classes = list(range(numclasses))
        
        else: classes = classes = [out]
        
        for cl in classes:
            if self.loss == "binary_crossentropy":
                y = all_scores.argmax(axis = 1) == cl # 1 for samples where self.model has predicted cl, 0 elsewhere
            elif self.loss == "mse":
                y = all_scores[:,cl] # return the raw scores for cl
            
            self.lime_model.set_weights(self.initial_weights) # reset weights

            self.lime_model.fit(all_masks, y, verbose = 0, epochs = 1000, batch_size = all_masks.shape[0], shuffle = True)
            all_weights.append(self.lime_model.get_weights()[0][:x_len])
            
        all_weights = np.concatenate(all_weights, axis = 1)
        all_weights = np.concatenate([all_weights, np.zeros((x.shape[0] - x_len, len(classes)))], axis = 0)
        return all_weights


