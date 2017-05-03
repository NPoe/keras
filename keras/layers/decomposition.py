# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from .. import backend as K
from .. import activations
from ..engine import Layer, InputSpec

# idea: try bidirectional for Decomposition & Erasure
# multi-timestep output for Decomposition layers (Erasure should already work)

class DecompositionLayer(Layer):
	def __init__(self, input_dim = None, return_sequences = False, input_length = None, ngram = 1, 
			go_backwards = False, stateful = False, **kwargs):
		
		self.supports_masking = True
		self.stateful = False
		self.return_sequences = return_sequences
		
		self.ngram = ngram
		self.return_sequences = return_sequences
		self.go_backwards = go_backwards
		
		self.input_spec = [InputSpec(ndim=4)]
		self.input_dim = input_dim
		self.input_length = input_length
		if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)
		
		assert self.ngram >= 1
		super(DecompositionLayer, self).__init__(**kwargs)

	def get_output_shape_for(self, input_shape):
		assert len(input_shape) >= 4 # samples, timesteps, states (c|h|o...), hidden_size ...
		timesteps_causer = input_shape[1]
		if not input_shape[1] is None:
			timesteps_causer = input_shape[1] - self.ngram + 1

		if self.return_sequences:
			return (input_shape[0], timesteps_causer, input_shape[1]) + input_shape[3:]
		return (input_shape[0], timesteps_causer) + input_shape[3:]
	
	def get_config(self):
		config = {'return_sequences': self.return_sequences,
				  'go_backwards': self.go_backwards,
                                  'stateful': self.stateful, 'ngram': self.ngram}
		
		config['input_dim'] = self.input_dim
		config['input_length'] = self.input_length

		base_config = super(DecompositionLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def call(self, x, mask=None):
		ndim = x.ndim
		if not ndim >= 4:
			raise Exception("Input should be at least 4D")

		sequence = self.get_sequence(x)
		
		if mask is None:
			mask = K.ones_like(x[tuple([slice(None), slice(None)] + [0 for _ in range(ndim - 2)])]) # (samples, timesteps)

		if self.go_backwards:
			sequence = sequence[:,::-1]
			mask = mask[:,::-1]

		if self.return_sequences:
			mask_stacked = K.square_stack(mask, 1) # (samples, causees, causers)
			sequence_stacked = K.square_stack(sequence, 1) # (samples, causees, causers, ...)

			triangle = K.tril_from_vec(mask[0])
			mask_stacked = mask_stacked * triangle
		else:
			sequence_stacked = K.expand_dims(sequence, 1) # (samples, 1, causers, ...)
			mask_stacked = K.expand_dims(mask, 1) # (samples, 1, causers)
	
		sequence_stacked = self.prep_sequence(sequence_stacked, mask_stacked, x)
		
		for i in range(ndim - 3):
			mask_stacked = K.expand_dims(mask_stacked, -1)
		if self.go_backwards:
			mask_stacked = mask_stacked[:,::-1,::-1]
			sequence_stacked = sequence_stacked[:,:,::-1]

		(left, right) = self.shift_sequence(sequence_stacked)
		(left_mask, right_mask) = self.shift_sequence(mask_stacked)

		subtraction = right - left
		subtraction = subtraction * right_mask
	
		out = self.finalize(subtraction, x)
		if not self.return_sequences:
			out = K.squeeze(out, 1) # samples, causers, ...
		else:
			axes = [0, 2, 1] + list(range(3, ndim))
			out = K.permute_dimensions(out, axes) # samples, causers, causees, ...

		return out

	def shift_sequence(self, sequence):
		sequence_left = K.concatenate([K.zeros_like(sequence[:,:,:1]), sequence[:,:,:(-self.ngram)]], axis = 2)
		sequence_right = sequence[:,:,(self.ngram-1):]
		return (sequence_left, sequence_right)

class GRUDecompositionLayer(DecompositionLayer):
	def finalize(self, sequence, x):
		return sequence

	def get_sequence(self, x):
		return x[:,:,0] # hidden vector

class LSTMDecompositionLayer(DecompositionLayer):
	def __init__(self, activation = "tanh", **kwargs):	
		self.activation = activation
		super(LSTMDecompositionLayer, self).__init__(**kwargs)

	def get_sequence(self, x):
		return x[:,:,1] # memory cell

	def finalize(self, sequence, x):
		o = x[:,:,4]
		o = K.expand_dims(o, 2) # samples, causers, 1, ...

		if self.return_sequences:
			return sequence * o
		
		else:
			oT = o[:,1 - 2 * int(not self.go_backwards)] # samples, 1, 1, ...
			return sequence * K.expand_dims(oT, 1)


class BetaLayerGRU(GRUDecompositionLayer):
	def prep_sequence(self, sequence_stacked, mask_stacked, x):
		return sequence_stacked

class GammaLayerGRU(GRUDecompositionLayer):
	def prep_sequence(self, sequence_stacked, mask_stacked, x):
		z = x[:,:,1]
		
		def __step(__z, __states):
			return __states[0], [__z * __states[0]]

		def _step(_mask, _states):
			_, outz, _, _ = K.rnn(
				__step, 
				z, 
				initial_states = [K.ones_like(z[:,1])], 
				mask = _mask, 
				go_backwards = not self.go_backwards)
			
			outz = outz[:,::-1]
			return outz, []

		_, out, _, _ = K.rnn(
			_step,
			mask_stacked,
			initial_states = [])

		return out * sequence_stacked

class BetaLayerLSTM(LSTMDecompositionLayer):
	def prep_sequence(self, sequence_stacked, mask_stacked, x):
		return activations.get(self.activation)(sequence_stacked)

class GammaLayerLSTM(LSTMDecompositionLayer):
	def prep_sequence(self, sequence_stacked, mask_stacked, x):
		f = x[:,:,3]

		def __step(__f, __states):
			return __states[0], [__f * __states[0]]

		def _step(_mask, _states):
			_, outf, _, _ = K.rnn(
				__step, 
				f, 
				initial_states = [K.ones_like(f[:,1])], 
				mask = _mask, 
				go_backwards = not self.go_backwards)

			outf = outf[:,::-1]
			return outf, []

		_, out, _, _ = K.rnn(
			_step,
			mask_stacked,
			initial_states = [])
	   
		return activations.get(self.activation)(out * sequence_stacked)
