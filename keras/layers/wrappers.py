# -*- coding: utf-8 -*-
from __future__ import absolute_import

import copy
import numpy as np
import inspect
from .recurrent import GRU, LSTM
from ..engine import Layer
from ..engine import InputSpec
from .. import activations
from .. import backend as K

class Wrapper(Layer):
    """Abstract wrapper base class.

    Wrappers take another layer and augment it in various ways.
    Do not use this class as a layer, it is only an abstract base class.
    Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.

    # Arguments
        layer: The layer to be wrapped.
    """

    def __init__(self, layer, **kwargs):
        super(Wrapper, self).__init__(**kwargs)
        self.layer = layer
        if self._initial_weights:
            self.layer._set_initial_weights(self._initial_weights)
        elif self.layer._get_initial_weights():
            self._initial_weights = self.layer._get_initial_weights()

    def build(self, input_shape=None):
        built = True

    def _get_initial_weights(self):
        if self._initial_weights:
            return self._initial_weights
        return self.layer._get_initial_weights()

    @property
    def activity_regularizer(self):
        if hasattr(self.layer, 'activity_regularizer'):
            return self.layer.activity_regularizer
        else:
            return None

    @property
    def trainable_weights(self):
        return self.layer.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.layer.non_trainable_weights

    @property
    def updates(self):
        if hasattr(self.layer, 'updates'):
            return self.layer.updates
        return []

    def get_updates_for(self, inputs=None):
        if inputs is None:
            updates = self.layer.get_updates_for(None)
            return updates + super(Wrapper, self).get_updates_for(None)
        return super(Wrapper, self).get_updates_for(inputs)

    @property
    def losses(self):
        if hasattr(self.layer, 'losses'):
            return self.layer.losses
        return []

    def get_losses_for(self, inputs=None):
        if inputs is None:
            losses = self.layer.get_losses_for(None)
            return losses + super(Wrapper, self).get_losses_for(None)
        return super(Wrapper, self).get_losses_for(inputs)

    @property
    def constraints(self):
        return self.layer.constraints

    def get_weights(self):
        return self.layer.get_weights()

    def set_weights(self, weights):
        self.layer.set_weights(weights)

    def get_config(self):
        config = {'layer': {'class_name': self.layer.__class__.__name__,
                            'config': self.layer.get_config()}}
        base_config = super(Wrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from . import deserialize as deserialize_layer
        layer = deserialize_layer(config.pop('layer'), custom_objects=custom_objects)
        return cls(layer, **config)


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
    ```

    The output will then have shape `(32, 10, 8)`.

    In subsequent layers, there is no need for the `input_shape`:

    ```python
        model.add(TimeDistributed(Dense(32)))
        # now model.output_shape == (None, 10, 32)
    ```

    The output will then have shape `(32, 10, 32)`.

    `TimeDistributed` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:

    ```python
        model = Sequential()
        model.add(TimeDistributed(Conv2D(64, (3, 3)),
                                  input_shape=(10, 299, 299, 3)))
    ```

    # Arguments
        layer: a layer instance.
    """

    def __init__(self, layer, implementation = 0, **kwargs):
        super(TimeDistributed, self).__init__(layer, **kwargs)
        self.supports_masking = True
        self.implementation = implementation

    def build(self, input_shape):
        super(TimeDistributed, self).build()
        assert len(input_shape) >= 3
        self.input_spec = InputSpec(shape=input_shape)
        child_input_shape = (input_shape[0],) + input_shape[2:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True
        self.built = True

    def compute_output_shape(self, input_shape):
        child_input_shape = (input_shape[0],) + input_shape[2:]
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        timesteps = input_shape[1]
        return (child_output_shape[0], timesteps) + child_output_shape[1:]


    def call(self, inputs, mask=None):
        
        if self.implementation == 1:
        
            def step(x, _):
                output = self.layer.call(x)
                return output, []

            _, y, _, _ = K.rnn(step, inputs, initial_states=[])

        else:
            input_shape = K.int_shape(inputs)
            input_length = input_shape[1]
            if not input_length:
                input_length = K.shape(inputs)[1]
            # Shape: (num_samples * timesteps, ...)
            inputs = K.reshape(inputs, (-1,) + input_shape[2:])
            y = self.layer.call(inputs)  # (num_samples * timesteps, ...)
            # Shape: (num_samples, timesteps, ...)
            output_shape = self.compute_output_shape(input_shape)
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

    # Raises
        ValueError: In case of invalid `merge_mode` argument.

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

    def __init__(self, layer, merge_mode='concat', input_mode = 'same', **kwargs):
        super(Bidirectional, self).__init__(layer, **kwargs)
        
        if merge_mode not in ['sum', 'mul', 'ave', 'concat', None]:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"sum", "mul", "ave", "concat", None}')
        if input_mode not in ['same', 'split']:
            raise ValueError('Invalid input mode. '
                    'Input mode should be one of '
                    '{"same", "split"}')

        self.forward_layer = copy.deepcopy(layer)
        config = layer.get_config()
        config['go_backwards'] = not config['go_backwards']
        self.backward_layer = layer.__class__.from_config(config)
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
        
        self.merge_mode = merge_mode
        self.input_mode = input_mode
        self.stateful = layer.stateful
        self.return_sequences = layer.return_sequences
        self.supports_masking = True
        
        if self._initial_weights:
            nw = len(self._initial_weights)
            self.forward_layer._set_initial_weights(self._initial_weights[:nw // 2])
            self.backward_layer._set_initial_weights(self._initial_weights[nw // 2:])
        elif self.forward_layer._get_initial_weights():
            self._initial_weights = self.forward_layer._get_initial_weights() + self.backward_layer._get_initial_weights()
    
    def _get_initial_weights(self):
        if self._initial_weights:
            return self._initial_weights
        return self.forward_layer._get_initial_weights() + self.backward_layer._get_initial_weights()


    def get_weights(self):
        return self.forward_layer.get_weights() + self.backward_layer.get_weights()

    def set_weights(self, weights):
        nw = len(weights)
        self.forward_layer.set_weights(weights[:nw // 2])
        self.backward_layer.set_weights(weights[nw // 2:])

    def compute_output_shape(self, input_shape):
        basic_shape = self.forward_layer.compute_output_shape(input_shape)
        if self.merge_mode in ['sum', 'ave', 'mul']:
            if self.input_mode == "split":
                return tuple(basic_shape[:-1]) + (basic_shape[-1] // 2,)
            elif self.input_mode == "same":
                return basic_shape
        elif self.merge_mode == 'concat':
            if self.input_mode == "same":
                return tuple(basic_shape[:-1]) + (basic_shape[-1] * 2,)
            elif self.input_mode == 'split':
                return basic_shape
        elif self.merge_mode is None:
            if self.input_mode == "split":
                return [tuple(basic_shape[:-1]) + (basic_shape[-1] // 2)] * 2
            elif self.input_mode == "same":
                return [basic_shape] * 2
        
    def call(self, inputs, training=None, mask=None):
        kwargs = {}
        func_args = inspect.getargspec(self.layer.call).args
        if 'training' in func_args:
            kwargs['training'] = training
        if 'mask' in func_args:
            kwargs['mask'] = mask
        
        if self.input_mode == "same":
            y = self.forward_layer.call(inputs, **kwargs)
            y_rev = self.backward_layer.call(inputs, **kwargs)
        elif self.input_mode == "split":
            length = K.int_shape(inputs)[-1]
            y = self.forward_layer.call(inputs[...,:length//2], **kwargs)
            y_rev = self.backward_layer.call(inputs[...,length//2:], **kwargs)

        if self.return_sequences:
            y_rev = K.reverse(y_rev, 1)
        if self.merge_mode == 'concat':
            output = K.concatenate([y, y_rev])
        elif self.merge_mode == 'sum':
            output = y + y_rev
        elif self.merge_mode == 'ave':
            output = (y + y_rev) / 2
        elif self.merge_mode == 'mul':
            output = y * y_rev
        elif self.merge_mode is None:
            output = [y, y_rev]

        # Properly set learning phase
        if hasattr(self.layer, 'recurrent_dropout'):
            if 0 < self.layer.dropout + self.layer.recurrent_dropout:
                if self.merge_mode is None:
                    for out in output:
                        out._uses_learning_phase = True
                else:
                    output._uses_learning_phase = True

        return output

    def reset_states(self):
        self.forward_layer.reset_states()
        self.backward_layer.reset_states()

    def build(self, input_shape):
        with K.name_scope(self.forward_layer.name):
            self.forward_layer.build(input_shape)
            self.forward_layer.built = True
        with K.name_scope(self.backward_layer.name):
            self.backward_layer.build(input_shape)
            self.backward_layer.built = True
        
        self.built = True

    def compute_mask(self, inputs, mask):
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
        config = {'merge_mode': self.merge_mode}
        base_config = super(Bidirectional, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ErasureWrapper(Wrapper):
	def __init__(self, layer, ngram = 1, **kwargs):
		super(ErasureWrapper, self).__init__(layer, **kwargs)
		self.supports_masking = True
		self.ngram = ngram

	def build(self, input_shape):
		super(ErasureWrapper, self).build()
		assert len(input_shape) >= 3
		self.input_spec = [InputSpec(shape=input_shape)]
		self.layer.build(input_shape)
		self.layer.built = True
		self.built = True


	def call(self, x, mask = None):
		if mask is None:
			mask = K.ones_like(x[tuple([slice(None), slice(None)] + [0 for _ in range(2, x.ndim)])])
		
		orig_score = self.layer.call(x, mask)
		mask_stacked = K.square_stack(mask, 1) # samples, timesteps (erasure), timesteps (rnn)
		
		diag = K.diag_from_vec(mask[0])
		for i in range(1, self.ngram):
			diag = diag + K.diag_from_vec(mask[0], offset = 1)
			
		if self.ngram > 1:
			diag = diag[:(-1)*(self.ngram - 1)] # timesteps(erasure, causer), timesteps(rnn, causee?)
		diag = K.expand_dims(diag, 0) # 1, timesteps (erasure), timesteps (rnn)
		mask_stacked = mask_stacked[:,(self.ngram - 1):]
		
		anti_diag = -1 * (diag - 1) # sqare matrix with 0 on diagonal and 1 elsewhere
		knock_mask_stacked = mask_stacked*anti_diag # knock out ones on the diagonals (zeros remain zeros in any case)
		
		def step(_mask, _):
			return orig_score - self.layer.call(x, mask = _mask), []
		
		_, outputs, _, _ = K.rnn(step, knock_mask_stacked, initial_states=[]) # samples, causers, ...
		return outputs
		
	def compute_output_shape(self, input_shape):
		timesteps_causer = input_shape[1]
		if not input_shape[1] is None:
			timesteps_causer = timesteps_causer - self.ngram + 1
		
		return (input_shape[0], timesteps_causer) + self.layer.compute_output_shape(input_shape)[1:]

class Decomposition(Wrapper):
        """Abstract decomposition layer base class.
        Decomposition layers decompose the output of LSTMs and GRUs into relevance contributions by individual timesteps.
        Do not use this class as a layer, it is only an abstract base class.

        See also https://arxiv.org/pdf/1702.02540.
                # Arguments
                layer: The layer to be wrapped. Must be a GRU or LSTM.
                return_square_sequences:
                ngram:
                :

        """

        def __init__(self, 
                layer,
                ngram = 1, 
                go_backwards = False,
                stateful = False,
                return_sequences = True,
                **kwargs):

            super(Decomposition, self).__init__(layer, **kwargs)
            
            self.supports_masking = True
            self.return_sequences = True
            
            self.ngram = ngram

            assert isinstance(self.layer, LSTM) or isinstance(self.layer, GRU)
            assert self.ngram >= 1

            self.layer.go_backwards = go_backwards or self.layer.go_backwards
            self.go_backwards = self.layer.go_backwards

            self.activation = self.layer.activation
            self.stateful = self.layer.stateful
            
            self.return_square_sequences = self.layer.return_sequences
       

        def build(self, input_shape):
            super(Decomposition, self).build()
            self.input_spec = InputSpec(shape=input_shape)
            self.layer.build(input_shape)
            self.layer.built = True
            self.built = True

        
        def compute_output_shape(self, input_shape):
            basic_shape = self.layer.compute_output_shape(input_shape)
            timesteps_causer = input_shape[1]
            if not input_shape[1] is None:
                timesteps_causer = input_shape[1] - self.ngram + 1

            return (basic_shape[0], timesteps_causer) + basic_shape[1:]

        
        def get_config(self):
            config = {'layer': {'class_name': self.layer.__class__.__name__, 'config': self.layer.get_config()},
                    'return_square_sequences': self.return_square_sequences,
                    'return_sequences': self.return_sequences,
                    'go_backwards': self.go_backwards,
                    'stateful': self.stateful, 
                    'ngram': self.ngram}

            base_config = super(Decomposition, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

        
        def call(self, inputs, mask=None):
            
            _, _, _, states = self.layer._call(inputs, mask = mask)
            
            if isinstance(self.layer, GRU):
                cells = states[0] # hidden states
                forget_gates = states[1] # z gates

            elif isinstance(self.layer, LSTM):
                cells = states[1] # memory cells
                forget_gates = states[3] # forget gates
                out_gates = states[4] # out gates
           
            if mask is None:
                mask = K.ones_like(cells[tuple([slice(None), slice(None)] + [0 for _ in range(2, cells.ndim)])]) # (samples, timesteps)

            if self.go_backwards:
                mask = mask[:,::-1]

            if self.return_square_sequences:
                mask_stacked = K.square_stack(mask, 1) # (samples, causees, causers)
                cells_stacked = K.square_stack(cells, 1) # (samples, causees, causers, ...)

                mask_stacked *= K.tril_from_vec(mask[0]) # (timesteps, timesteps) lower triangle
            
            else:
                cells_stacked = K.expand_dims(cells, 1) # (samples, 1, causers, ...)
                mask_stacked = K.expand_dims(mask, 1) # (samples, 1, causers)
                
            cells_stacked *= self.compute_forget_mask(mask_stacked, forget_gates)

            if isinstance(self.layer, LSTM):
                cells_stacked =  activations.get(self.activation)(cells_stacked)
            
            for i in range(2, cells.ndim):
                mask_stacked = K.expand_dims(mask_stacked, -1) # get mask to the same number of dimensions as the output

            (left, right) = self.shift_sequence(cells_stacked)
            (_, right_mask) = self.shift_sequence(mask_stacked)
            
            differences = (right - left) * right_mask
            
            if isinstance(self.layer, LSTM):
                differences *= self.get_out_gates(out_gates)
                
            if self.return_square_sequences:
                axes = [0, 2, 1] + list(range(3, cells.ndim + 1))
                return K.permute_dimensions(differences, axes) # samples, causers, causees, ...
            else:    
                return K.squeeze(differences, 1) # samples, causers, ...

        def shift_sequence(self, sequence):
            sequence_left = K.concatenate([K.zeros_like(sequence[:,:,:1]), sequence[:,:,:(-self.ngram)]], axis = 2)
            sequence_right = sequence[:,:,(self.ngram-1):]
            return (sequence_left, sequence_right)

        def get_out_gates(self, out_gates):
            out_gates = K.expand_dims(out_gates, 2) # samples, causers, 1, ...

            if self.return_square_sequences: 
                return out_gates

            else:
                return K.expand_dims(out_gates[:,-1], 1)


class BetaDecomposition(Decomposition):
    def compute_forget_mask(self, mask_stacked, forget_gates):
        return 1

class GammaDecomposition(Decomposition):
    def compute_forget_mask(self, mask_stacked, forget_gates):

        def __step(__f, __states):
            return __states[0], [__f * __states[0]]

        def _step(_mask, _states):
            _, outf, _, _ = K.rnn(
                __step, 
                forget_gates, 
                initial_states = [K.ones_like(forget_gates[:,1])], 
                mask = _mask,
                go_backwards = True)

            return outf[:,::-1], []

        _, out, _, _ = K.rnn(
            _step,
            mask_stacked,
            initial_states = [])

        return out
