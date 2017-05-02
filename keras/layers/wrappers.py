import copy
import numpy as np
from ..engine import Layer
from ..engine import InputSpec
from .. import backend as K


class Wrapper(Layer):
	"""Abstract wrapper base class.
	"""

	def __init__(self, layer, **kwargs):
		self.layer = layer
		self.uses_learning_phase = layer.uses_learning_phase
		super(Wrapper, self).__init__(**kwargs)

	def build(self, input_shape=None):
		# Assumes that self.layer is already set.
		# Should be called at the end of .build() in the children classes.
		self.trainable_weights = getattr(self.layer, 'trainable_weights', [])
		self.non_trainable_weights = getattr(self.layer, 'non_trainable_weights', [])
		self.updates = getattr(self.layer, 'updates', [])
		self.losses = getattr(self.layer, 'losses', [])
		self.constraints = getattr(self.layer, 'constraints', {})

	def get_weights(self):
		weights = self.layer.get_weights()
		return weights

	def set_weights(self, weights):
		self.layer.set_weights(weights)

	def get_config(self):
		config = {'layer': {'class_name': self.layer.__class__.__name__,
							'config': self.layer.get_config()}}
		base_config = super(Wrapper, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	@classmethod
	def from_config(cls, config):
		from keras.utils.layer_utils import layer_from_config
		layer = layer_from_config(config.pop('layer'))
		return cls(layer, **config)


class ErasureWrapper(Wrapper):
	def __init__(self, layer, ngram = 1, **kwargs):
		self.supports_masking = True
		self.ngram = ngram
		super(ErasureWrapper, self).__init__(layer, **kwargs)

	def build(self, input_shape):
		assert len(input_shape) >= 3
		self.input_spec = [InputSpec(shape=input_shape)]
		if not self.layer.built:
			self.layer.build(input_shape)
			self.layer.built = True
			
		super(ErasureWrapper, self).build()

	def call(self, x, mask = None):
		ndim = x.ndim
	   
	if mask is None:
			mask = K.ones_like(x[tuple([slice(None), slice(None)] + [0 for _ in range(2, ndim)])])
		
		orig_score = self.layer.call(x, mask)
		mask_stacked = K.square_stack(mask, 1) # samples, timesteps (erasure), timesteps (rnn)
		
	diag = K.diag_from_vec(mask[0])
	for i in range(1, self.ngram):
		diag = diag + K.diag_from_vec(mask[0], offset = 1)

	diag = K.expand_dims(diag, 0) # 1, timesteps (erasure), timesteps (rnn)
		
	diag = diag[:,(self.ngram - 1):]
		mask_stacked = mask_stacked[:,(self.ngram - 1):]
	   
		anti_diag = diag == 0 # sqare matrix with 0 on diagonal and 1 elsewhere
		knock_m = m*anti_eye # knock out ones on the diagonals (zeros remain zeros in any case)
		
		def step(_mask, _):
			return orig_score - self.layer.call(x, mask = _mask), []

		_, outputs, _, _ = K.rnn(step, knock_m, initial_states=[]) # samples, causers, ...
	   
		return outputs

	def get_output_shape_for(self, input_shape):
		timesteps_causer = input_shape[1]
	if not input_shape[1] is None:
		timesteps_causer = timesteps_causer - self.ngram + 1
		return (input_shape[0], timesteps_causer) + self.layer.get_output_shape_for(input_shape)[1:]


class TimeDistributed(Wrapper):
	"""This wrapper allows to apply a layer to every temporal slice of an input.

	The input should be at least 3D, and the dimension of index one
	will be considered to be the temporal dimension.

	Consider a batch of 32 samples,
	where each sample is a sequence of 10 vectors of 16 dimensions.
	The batch input shape of the layer is then `(32, 10, 16)`,
	and the `input_shape`, not including the samples dimension, is `(10, 16)`.

	You can then use `TimeDistributed` to apply a `Dense` layer
	to each of the 10 timesteps, independently:

	```python
		# as the first layer in a model
		model = Sequential()
		model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
		# now model.output_shape == (None, 10, 8)

		# subsequent layers: no need for input_shape
		model.add(TimeDistributed(Dense(32)))
		# now model.output_shape == (None, 10, 32)
	```

	The output will then have shape `(32, 10, 8)`.

	`TimeDistributed` can be used with arbitrary layers, not just `Dense`,
	for instance with a `Convolution2D` layer:

	```python
		model = Sequential()
		model.add(TimeDistributed(Convolution2D(64, 3, 3),
								  input_shape=(10, 3, 299, 299)))
	```

	# Arguments
		layer: a layer instance.
	"""

	def __init__(self, layer, **kwargs):
		self.supports_masking = True
		super(TimeDistributed, self).__init__(layer, **kwargs)

	def build(self, input_shape):
		assert len(input_shape) >= 3
		self.input_spec = [InputSpec(shape=input_shape)]
		child_input_shape = (input_shape[0],) + input_shape[2:]
		if not self.layer.built:
			self.layer.build(child_input_shape)
			self.layer.built = True
		super(TimeDistributed, self).build()

	def get_output_shape_for(self, input_shape):
		child_input_shape = (input_shape[0],) + input_shape[2:]
		child_output_shape = self.layer.get_output_shape_for(child_input_shape)
		timesteps = input_shape[1]
		return (child_output_shape[0], timesteps) + child_output_shape[1:]

	def call(self, inputs, mask=None):
		input_shape = K.int_shape(inputs)
		if input_shape[0]:
			# batch size matters, use rnn-based implementation
			def step(x, _):
				output = self.layer.call(x)
				return output, []

			_, outputs, _, _ = K.rnn(step, inputs,
								  initial_states=[],
								  input_length=input_shape[1],
								  unroll=False)
			y = outputs
		else:
			# no batch size specified, therefore the layer will be able
			# to process batches of any size
			# we can go with reshape-based implementation for performance
			input_length = input_shape[1]
			if not input_length:
				input_length = K.shape(inputs)[1]
			# (nb_samples * timesteps, ...)
			inputs = K.reshape(inputs, (-1,) + input_shape[2:])
			y = self.layer.call(inputs)  # (nb_samples * timesteps, ...)
			# (nb_samples, timesteps, ...)
			output_shape = self.get_output_shape_for(input_shape)
			y = K.reshape(y, (-1, input_length) + output_shape[2:])

		# Apply activity regularizer if any:
		if (hasattr(self.layer, 'activity_regularizer') and
		   self.layer.activity_regularizer is not None):
			regularization_loss = self.layer.activity_regularizer(y)
			self.add_loss(regularization_loss, inputs)
		return y


class Bidirectional(Wrapper):
	"""Bidirectional wrapper for RNNs.

	# Arguments
		layer: `Recurrent` instance.
		merge_mode: Mode by which outputs of the
			forward and backward RNNs will be combined.
			One of {'sum', 'mul', 'concat', 'ave', None}.
			If None, the outputs will not be combined,
			they will be returned as a list.

	# Examples

	```python
		model = Sequential()
		model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
		model.add(Bidirectional(LSTM(10)))
		model.add(Dense(5))
		model.add(Activation('softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	```
	"""

	def __init__(self, layer, merge_mode='concat', input_mode='same', weights=None, **kwargs):
		if merge_mode not in ['sum', 'mul', 'ave', 'concat', None]:
			raise ValueError('Invalid merge mode. '
							 'Merge mode should be one of '
							 '{"sum", "mul", "ave", "concat", None}')
		if input_mode not in ['same', 'split']:
			raise ValueError('Invalid input mode. '
							 'Input mode should be one of '
							 '{"same", "split"}')

		self.forward_layer = copy.copy(layer)
		config = layer.get_config()
		config['go_backwards'] = not config['go_backwards']
		self.backward_layer = layer.__class__.from_config(config)
		self.forward_layer.name = 'forward_' + self.forward_layer.name
		self.backward_layer.name = 'backward_' + self.backward_layer.name
		self.merge_mode = merge_mode
		self.input_mode = input_mode
		if weights:
			nw = len(weights)
			self.forward_layer.initial_weights = weights[:nw // 2]
			self.backward_layer.initial_weights = weights[nw // 2:]
		self.stateful = layer.stateful
		self.return_sequences = layer.return_sequences
		self.supports_masking = True
		super(Bidirectional, self).__init__(layer, **kwargs)

	def get_weights(self):
		return self.forward_layer.get_weights() + self.backward_layer.get_weights()

	def set_weights(self, weights):
		nw = len(weights)
		self.forward_layer.set_weights(weights[:nw // 2])
		self.backward_layer.set_weights(weights[nw // 2:])

	def get_output_shape_for(self, input_shape):
		basic_shape = self.forward_layer.get_output_shape_for(input_shape)
		if self.merge_mode in ['sum', 'ave', 'mul']:
			if self.input_mode == "split":
				return tuple(basic_shape[:-1]) + (basic_shape[-1] // 2,)
			return self.forward_layer.get_output_shape_for(input_shape)
		elif self.merge_mode == 'concat':
			if self.input_mode == "same":
				return tuple(basic_shape[:-1]) + (basic_shape[-1] * 2,)
			return basic_shape
		elif self.merge_mode is None:
			if self.input_mode == "split":
				return [tuple(basic_shape[:-1]) + (basic_shape[-1] // 2)] * 2
			return [self.forward_layer.get_output_shape_for(input_shape)] * 2

	def call(self, inputs, mask=None):

		if self.input_mode == "same":
			y = self.forward_layer.call(inputs, mask)
			y_rev = self.backward_layer.call(inputs, mask)
		elif self.input_mode == "split":
			length = K.int_shape(inputs)[-1]
			assert length % 2 == 0
			y = self.forward_layer.call(inputs[...,:length//2], mask)
			y_rev = self.backward_layer.call(inputs[...,length//2:], mask)

		if self.return_sequences:
			y_rev = K.reverse(y_rev, 1)
		if self.merge_mode == 'concat':
			return K.concatenate([y, y_rev])
		elif self.merge_mode == 'sum':
			return y + y_rev
		elif self.merge_mode == 'ave':
			return (y + y_rev) / 2
		elif self.merge_mode == 'mul':
			return y * y_rev
		elif self.merge_mode is None:
			return [y, y_rev]

	def reset_states(self):
		self.forward_layer.reset_states()
		self.backward_layer.reset_states()

	def build(self, input_shape):
		self.forward_layer.build(input_shape)
		self.backward_layer.build(input_shape)

	def compute_mask(self, input, mask):
		if self.return_sequences:
			if not self.merge_mode:
				return [mask, mask]
			else:
				return mask
		else:
			return None

	@property
	def trainable_weights(self):
		if hasattr(self.forward_layer, 'trainable_weights'):
			return (self.forward_layer.trainable_weights +
					self.backward_layer.trainable_weights)
		return []

	@property
	def non_trainable_weights(self):
		if hasattr(self.forward_layer, 'non_trainable_weights'):
			return (self.forward_layer.non_trainable_weights +
					self.backward_layer.non_trainable_weights)
		return []

	@property
	def updates(self):
		if hasattr(self.forward_layer, 'updates'):
			return self.forward_layer.updates + self.backward_layer.updates
		return []

	@property
	def losses(self):
		if hasattr(self.forward_layer, 'losses'):
			return self.forward_layer.losses + self.backward_layer.losses
		return []

	@property
	def constraints(self):
		constraints = {}
		if hasattr(self.forward_layer, 'constraints'):
			constraints.update(self.forward_layer.constraints)
			constraints.update(self.backward_layer.constraints)
		return constraints

	def get_config(self):
		config = {"merge_mode": self.merge_mode}
		base_config = super(Bidirectional, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
