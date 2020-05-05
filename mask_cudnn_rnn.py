from keras.layers import Layer, GRU, CuDNNGRU
import keras.backend as K


def mask_cudnn_gru(units: int,
                   use_cudnn: bool,
                   return_sequences: bool = False,
                   return_state: bool = False,
                   **kwargs):
    if (use_cudnn):
        return MaskCuDNNGRU(units,
                            return_sequences=return_sequences,
                            return_state=return_state,
                            **kwargs)
    else:
        return GRU(units,
                   activation='tanh',
                   recurrent_activation='sigmoid',
                   return_sequences=return_sequences,
                   return_state=return_state,
                   **kwargs)


class MaskCuDNNGRU(Layer):
    rnn = None
    remove = None

    def __init__(self,
                 units: int,
                 return_sequences: bool = False,
                 return_state: bool = False,
                 **kwargs):
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        if (self.return_sequences):
            print(
                'MaskCuDNNRNN :: Warning! return_sequences is not fully support masking. All time steps will be calculated. The masking will send to output with same size.'
            )

        self.rnn = CuDNNGRU(self.units,
                            return_sequences=True,
                            return_state=False)
        super(MaskCuDNNGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskCuDNNGRU, self).build(input_shape)

    def call(self, x, mask=None, initial_state=None):
        x = self.rnn(x, initial_state=initial_state, mask=None)

        if (self.return_sequences and (not self.return_state)):
            return x

        if (mask is not None):
            index = K.cast(mask, 'uint8')
            index = K.cumsum(index, 1) + index
            index = K.equal(index, K.max(index, 1, True))

            index = K.cast(
                K.repeat_elements(K.expand_dims(index, -1), self.units, -1),
                K.floatx())
            state = K.sum(x * index, 1)
        else:
            state = x[:, -1]

        if (self.return_sequences):
            return [x, state]
        if (self.return_state):
            return [state, state]
        return state

    def compute_output_shape(self, input_shape):
        if (len(input_shape) != 3):
            raise ValueError('Input of MaskCuDNNRNN must be 3 dims, recieved',
                             len(input_shape), 'dim(s) = ', input_shape, '.')
        seq = (input_shape[0], input_shape[1],
               self.units) if (self.return_sequences) else (input_shape[0],
                                                            self.units)

        if (self.return_state):
            return [seq, (input_shape[0], self.units)]
        return seq

    def compute_mask(self, input, mask=None):
        if (self.return_sequences and (mask is not None)):
            return mask
        return None
