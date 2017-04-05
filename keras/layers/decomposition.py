# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from .wrappers import Wrapper
from .. import backend as K
from .. import activations
from ..engine import Layer, InputSpec

class ErasureLayer(Wrapper):
    def __init__(self, layer, **kwargs):
        self.supports_masking = True
        super(ErasureLayer, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
            
        super(ErasureLayer, self).build()

    def call(self, x, mask = None):
        input_shape = K.int_shape(x)
       
        if mask is None:
            mask = K.ones_like(x[:,:,0])
        
        orig_score = self.layer.call(x, mask)
        m = K.stack([mask for _ in range(input_shape[1])], 1) # samples, timesteps (erasure), timesteps (rnn)
        anti_eye = K.variable(np.eye(input_shape[1]) == 0) # sqare matrix with 0 on diagonal and 1 elsewhere

        knock_m = m*anti_eye # knock out ones on the diagonals (zeros remain zeros in any case)

        def step(_mask, const):
            _x = const[0]
            _orig_score = const[1]
            return _orig_score - self.layer.call(_x, mask = _mask), []

        _, outputs, _, _ = K.rnn(step, knock_m, initial_states=[], constants = [x, orig_score], unroll=False, mask = mask)
        
        return outputs

    def get_output_shape_for(self, input_shape):
        return tuple(list(input_shape[0:2]) + list(self.layer.get_output_shape_for(input_shape)[1:]))

class DecompositionLayer(Layer):
    def __init__(self):
        self.supports_masking = True
        super(DecompositionLayer, self).__init__()
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[3])

class LSTMDecompositionLayer(DecompositionLayer):
    def __init__(self, activation = "tanh"):    
        self.activation = activation
        super(LSTMDecompositionLayer, self).__init__()

    def call(self, x, mask=None):
        assert x.ndim >= 4, 'Input should be at least 4D.'
        
        cAct = activations.get(self.activation)(self.prep_c_sequence(x, mask))
        
        oT = x[:,-1,4,:]

        def _step(c, states):
            prev_c = states[0]
            return oT * (c - prev_c), [c]

        _, outputs, _, _ = K.rnn(_step, cAct, initial_states = [K.zeros_like(cAct[:,1])], mask = mask)

        return outputs

class GRUDecompositionLayer(DecompositionLayer):
    def call(self, x, mask=None):
        assert x.ndim >= 4, 'Input should be at least 4D.'
        
        h = self.prep_h_sequence(x, mask)
        
        def _step(h, states):
            prev_h = states[0]
            return h - prev_h, [h]

        _, outputs, _, _ = K.rnn(_step, h, initial_states = [K.zeros_like(h[:,1])], mask = mask)
        
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

        _, outz, _, _ = K.rnn(_step, z, initial_states = [K.ones_like(z[:,1])], mask = mask, go_backwards = True)

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

        _, outf, _, _ = K.rnn(_step, f, initial_states = [K.ones_like(f[:,1])], mask = mask, go_backwards = True)

        outf = outf[:,::-1]
       
        return outf * c
