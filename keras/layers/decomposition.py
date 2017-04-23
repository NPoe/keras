# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from .. import backend as K
from .. import activations
from ..engine import Layer, InputSpec

# idea: try bidirectional for Decomposition & Erasure
# multi-timestep output for Decomposition layers (Erasure should already work)

class DecompositionLayer(Layer):
    def __init__(self, ngram = 1, return_sequences = False, go_backwards = False):
        self.supports_masking = True
        self.ngram = ngram
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        assert self.ngram >= 1
        super(DecompositionLayer, self).__init__()

    def get_output_shape_for(self, input_shape):
        assert len(input_shape) >= 4 # samples, timesteps, states (c|h|o...), hidden_size ...
        if not input_shape[1] is None:
            return (input_shape[0], input_shape[1] - self.ngram + 1) + input_shape[3:]
        return (input_shape[0], input_shape[1]) + input_shape[3:]

class LSTMDecompositionLayer(DecompositionLayer):
    def __init__(self, activation = "tanh", **kwargs):    
        self.activation = activation
        super(LSTMDecompositionLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        assert x.ndim >= 4, 'Input should be at least 4D.'

        l = self.ngram - 1

        if not mask is None:
            mask = mask[:,l:]

        cAct = activations.get(self.activation)(self.prep_c_sequence(x, mask)) 
        oT = x[:,0 - int(not self.go_backwards),4,:]

        def _step(c, states):
            assert len(states) == self.ngram
            return oT * (c - states[0]), list(states[1:]) + [c]

        _, outputs, _, _ = K.rnn(_step, cAct[:,l:], \
                initial_states = [K.zeros_like(cAct[:,0])] + [cAct[:,i] for i in range(l)], \
                mask = mask, go_backwards = self.go_backwards)

        return outputs

class GRUDecompositionLayer(DecompositionLayer):
    def call(self, x, mask=None):
        assert x.ndim >= 4, 'Input should be at least 4D.'

        l = self.ngram - 1
        
        if not mask is None:
            mask = mask[:,l:]
        
        h = self.prep_h_sequence(x, mask)
        
        def _step(h, states):
            assert len(states) == self.ngram
            return h - states[0], list(states[1:]) + [h]

        _, outputs, _, _ = K.rnn(_step, h[:,l:], \
                initial_states = [K.zeros_like(h[:,0])] + [h[:,i] for i in range(l)], \
                mask = mask, go_backwards = self.go_backwards)
        
        return outputs

class BetaLayerGRU(GRUDecompositionLayer):
    def prep_h_sequence(self, x, mask = None):
        return x[:,:,0,:]

class GammaLayerGRU(GRUDecompositionLayer):
    def prep_h_sequence(self, x, mask = None):
        z = x[:,:,1,:]
        h = x[:,:,0,:]

        def _step(_z, _states):
            return _states[0], [_z * _states[0]]

        _, outz, _, _ = K.rnn(_step, z, initial_states = [K.ones_like(z[:,1])], mask = mask, go_backwards = not self.go_backwards)

        if not self.go_backwards:
            outz = outz[:,::-1]

        return outz * h

class BetaLayerLSTM(LSTMDecompositionLayer):
    def prep_c_sequence(self, x, mask = None):
        return x[:,:,1,:]

class GammaLayerLSTM(LSTMDecompositionLayer):
    def prep_c_sequence(self, x, mask = None):
        f = x[:,:,3,:]
        c = x[:,:,1,:]

        def _step(_f, _states):
            return _states[0], [_f * _states[0]]

        _, outf, _, _ = K.rnn(_step, f, initial_states = [K.ones_like(f[:,1])], mask = mask, go_backwards = not self.go_backwards)

        if not self.go_backwards:
            outf = outf[:,::-1]
       
        return outf * c
