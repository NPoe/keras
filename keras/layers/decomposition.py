# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from .wrappers import Wrapper
from .. import backend as K
from .. import activations
from ..engine import Layer, InputSpec

import theano
import theano.tensor as T

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

        m = m*anti_eye # knock out ones on the diagonals (zeros remain zeros in any case)

        def step(_mask, const):
            _x = const[0]
            _orig_score = const[1]
            return _orig_score - self.layer.call(_x, mask = _mask), []

        _, outputs, _, _ = K.rnn(step, m, initial_states=[], constants = [x, orig_score], unroll=False)
        
        return outputs

    def get_output_shape_for(self, input_shape):
        return tuple(list(input_shape[0:2]) + list(self.layer.get_output_shape_for(input_shape)[1:]))


class DecompositionLayer(Layer):
    def __init__(self, activation = "tanh", ngram = 1):
        self.activation = activation
        self.ngram = ngram
        self.supports_masking = True
        super(DecompositionLayer, self).__init__()
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[3])

class LSTMDecompositionLayer(DecompositionLayer):
    def call(self, x, mask=None):
        assert x.ndim >= 4, 'Input should be at least 4D.'
        
        h = x[:,:,0,:]

        axes = [1,0] + list(range(2, h.ndim))
        
        if mask is None: mask = K.ones_like(h)

        if mask.ndim == h.ndim - 1:
            mask = K.expand_dims(mask)

        assert mask.ndim == h.ndim
        mask = mask.dimshuffle(axes)
        
        
        cAct = activations.get(self.activation)(self.prep_c_sequence(x, mask)) / self.ngram
        
        oT = x[:,-1,4,:]

        cActLeft = K.concatenate([K.zeros_like(cAct[:self.ngram]), cAct[:(-1) * self.ngram]], axis = 0)
        cActRight = cAct

        initial_beta = K.zeros_like(cAct[-1])
        def _step(cActLeft, cActRight, mask, beta):
            return T.switch(mask, cActRight - cActLeft, 0)

        outputs, _ = theano.scan(_step, sequences = [cActLeft, cActRight, mask], outputs_info = [initial_beta], non_sequences = [], go_backwards = False)


        if isinstance(outputs, list):
            outputs = outputs[0]

        outputs = T.squeeze(outputs)
        axes = [1,0] + list(range(2, outputs.ndim))
        outputs = outputs * oT
        outputs = outputs.dimshuffle(axes)
        return outputs

class GRUDecompositionLayer(DecompositionLayer):
    def call(self, x, mask=None):
        assert x.ndim >= 4, 'Input should be at least 4D.'
        
        h = x[:,:,0,:]

        axes = [1,0] + list(range(2, h.ndim))
        
        if mask is None: mask = K.ones_like(h)

        if mask.ndim == h.ndim - 1:
            mask = K.expand_dims(mask)

        assert mask.ndim == h.ndim
        mask = mask.dimshuffle(axes)
        
        
        h = self.prep_h_sequence(x, mask)
        
        hLeft = K.concatenate([K.zeros_like(h[:self.ngram]), h[:(-1) * self.ngram]], axis = 0)
        hRight = h

        initial_beta = K.zeros_like(h[-1])
        def _step(hLeft, hRight, mask, beta):
            return T.switch(mask, hRight - hLeft, 0)

        outputs, _ = theano.scan(_step, sequences = [hLeft, hRight, mask], outputs_info = [initial_beta], non_sequences = [], go_backwards = False)


        if isinstance(outputs, list):
            outputs = outputs[0]

        outputs = T.squeeze(outputs)
        axes = [1,0] + list(range(2, outputs.ndim))
        outputs = outputs.dimshuffle(axes)
        return outputs


class DeltaLayer(GRUDecompositionLayer):
    def prep_h_sequence(self, x, mask):
        h = x[:,:,0,:]
        h = T.squeeze(h)
        axes = [1,0] + list(range(2, h.ndim))
        return h.dimshuffle(axes)

class OmegaLayer(GRUDecompositionLayer):
    def prep_h_sequence(self, x, mask):
        z = x[:,:,1,:]
        
        axes = [1,0] + list(range(2, z.ndim))
        z = z.dimshuffle(axes)
        
        initial_o_z = K.ones_like(z[-1])
        initial_z = K.ones_like(z[-1])
        
        def _step(z, mask, gamma, next_z):
            return next_z, T.switch(mask, z*next_z, next_z)
            
        outz, _ = theano.scan(_step, sequences = [z, mask], outputs_info = [initial_o_z, initial_z], non_sequences = [], go_backwards = True)
        
        if isinstance(outz, list):
            outz = outz[0]
        
        outz = outz[::-1]
        outz = T.squeeze(outz)
        
        h = x[:,:,0,:]
        h = T.squeeze(h)
        h = h.dimshuffle(axes)

        return outz * h

    

class BetaLayer(LSTMDecompositionLayer):
    def prep_c_sequence(self, x, mask):
        c = x[:,:,1,:]
        c = T.squeeze(c)
        axes = [1,0] + list(range(2, c.ndim))
        return c.dimshuffle(axes)

class GammaLayer(LSTMDecompositionLayer):
    def prep_c_sequence(self, x, mask):
        f = x[:,:,3,:]
        
        axes = [1,0] + list(range(2, f.ndim))
        f = f.dimshuffle(axes)
        
        initial_o_f = K.ones_like(f[-1])
        initial_f = K.ones_like(f[-1])
        
        def _step(f, mask, gamma, next_f):
            return next_f, T.switch(mask, f*next_f, next_f)
            
        outf, _ = theano.scan(_step, sequences = [f, mask], outputs_info = [initial_o_f, initial_f], non_sequences = [], go_backwards = True)
        
        if isinstance(outf, list):
            outf = outf[0]
        
        outf = outf[::-1]
        outf = T.squeeze(outf)
        
        c = x[:,:,1,:]
        c = T.squeeze(c)
        c = c.dimshuffle(axes)

        return outf * c


class EpsLayer(DecompositionLayer):
    def call(self, x, mask = None):
        z = x[:,:,1,:]
        h_tilde = x[:,:,3,:]
        ret_z = K.ones_like(z)-z
        
        axes = [1,0] + list(range(2, z.ndim))
        z = z.dimshuffle(axes)
        
        initial_o_z = K.ones_like(z[-1])
        initial_z = K.ones_like(z[-1])
        
        if mask is None:
            mask = K.ones_like(z)
        
        else:
            if mask.ndim == z.ndim - 1:
                mask = K.expand_dims(mask)
            assert mask.ndim == z.ndim
            mask = mask.dimshuffle(axes)
            
        def _step(z, mask, gamma, next_z):
            return T.switch(mask, next_z, 0), T.switch(mask, z*next_z, next_z)
            
        outz, _ = theano.scan(_step, sequences = [z, mask], outputs_info = [initial_o_z, initial_z], non_sequences = [], go_backwards = True)

        if isinstance(outz, list):
            outz = outz[0]
        
        outz = outz[::-1]
        outz = outz.dimshuffle(axes)
        return T.squeeze(outz * ret_z * h_tilde)


class PhiLayer(DecompositionLayer):
    def call(self, x, mask = None):
        f = x[:,:,3,:]
        i = x[:,:,2,:]
        h_tilde = x[:,:,5,:]
        oT = x[:,-1,4,:]
        
        axes = [1,0] + list(range(2, f.ndim))
        f = f.dimshuffle(axes)
        i = i.dimshuffle(axes)
        h_tilde = h_tilde.dimshuffle(axes)

        initial_o_f = K.ones_like(f[-1])
        initial_f = K.ones_like(f[-1])
        
        if mask is None:
            mask = K.ones_like(f)
        
        else:
            if mask.ndim == f.ndim - 1:
                mask = K.expand_dims(mask)
            assert mask.ndim == f.ndim
            mask = mask.dimshuffle(axes)
            
        def _step(f, mask, gamma, next_f):
            return T.switch(mask, next_f, 0), T.switch(mask, f*next_f, next_f)
            
        outf, _ = theano.scan(_step, sequences = [f, mask], outputs_info = [initial_o_f, initial_f], non_sequences = [], go_backwards = True)

        if isinstance(outf, list):
            outf = outf[0]
        
        outf = outf[::-1]
        cAct = activations.get(self.activation)(outf * i * h_tilde)
        
        out = cAct * oT
        out = out.dimshuffle(axes)
        
        return T.squeeze(out)


