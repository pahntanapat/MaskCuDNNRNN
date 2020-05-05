# MaskCuDNNRNN
Custom layer for CuDNN RNN which are compatible with masking

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Concept of MaskCuDNNGRU layer
.............
............
...........

### Prerequisites

The MaskCuDNNRNN requires ...
1. [Keras](https://keras.io/#installation)
2. [GPU driver](https://www.nvidia.com/Download)
3. [CUDA](https://developer.nvidia.com/cuda-downloads)
4. [CuDNN](https://developer.nvidia.com/cudnn)

If No. 2 - 4 are not completely installed, you can use only CPU-compatible GRU layer.
<br/>

### Installing

Now, we do not distribute it on any channel. Please clone the git or copy the [mask_cudnn_rnn.py](https://github.com/pahntanapat/MaskCuDNNRNN/blob/master/mask_cudnn_rnn.py).
<br/>
<br/>

## Usages

### MaskCuDNNGRU <a name="MaskCuDNNGRU"></a>
```
MaskCuDNNGRU(units: int, return_sequences: bool = False, return_state: bool = False, **kwargs)
```
Masking-compatible CuDNNGRU

#### Argument
* **units**: Positive integer, dimensionality of the output space.
* **return_sequences**: boolean, if True, return the full sequence, else, return the last output.
* **return_state**: boolean, whether to return the last state in addition to the output.
* **\*\*kwargs**: other optional arguments, same to Keras' [CuDNNGRU](https://keras.io/layers/recurrent/#cudnngru)


#### Example

This is an example of sequence-to-sequence model.
```
from keras.layer import Embedding, Input, Dense
from mask_cudnn_rnn import MaskCuDNNGRU

embed = Embedding(vocab_size + 1, vector_size,  mask_zero=True)
encoder = MaskCuDNNGRU(units=16, return_sequences = False, return_state = True, name = 'encoder')

decoder_1 = MaskCuDNNGRU(units=16, return_sequences = True, return_state = False, name = 'decoder_1')
decoder_2 = Bidirectional(MaskCuDNNGRU(units=32, return_sequences = False, return_state = False) name = 'decoder_2')
fc = Dense(vocab_size, activation='softmax')

# Create Training model

input_enc = Input(shape=(None,))
input_dec = Input(shape=(None,))

x = embed(input_enc)
_, state = encoder(x)

x = decoder_1(input_dec, initial_state=state)
x = decoder_2(x)
x = fc(x)

train_model = Model([input_enc, input_dec], [x])
# .....
```


### mask_cudnn_gru
```
mask_cudnn_gru(units: int, use_cudnn: bool, return_sequences: bool = False, return_state: bool = False, **kwargs)
```
This function can create [MaskCuDNNGRU](#MaskCuDNNGRU) and CuDNN-compatible [conventional GRU](https://keras.io/layers/recurrent/#gru). Both layers can exchange the weights together. [https://stackoverflow.com/questions/52900017/keras-loading-model-built-with-cudnnlstm-on-host-without-gpu]

#### Argument
* **units**: Positive integer, dimensionality of the output space.
* **use_cudnn**: boolean, If True, return [MaskCuDNNGRU](#MaskCuDNNGRU), else, return [GRU](https://keras.io/layers/recurrent/#gru).
* **return_sequences**: boolean, if True, return the full sequence, else, return the last output.
* **return_state**: boolean, whether to return the last state in addition to the output.
* **\*\*kwargs**: other optional arguments, same to Keras' [CuDNNGRU](https://keras.io/layers/recurrent/#cudnngru)

#### Example
Like the above example, only change some line

```
# .....
from mask_cudnn_rnn import mask_cudnn_gru

use_cudnn = True # False

# ...
# ...

encoder = MaskCuDNNGRU(units=16, use_cudnn=use_cudnn, return_sequences = False, return_state = True, name = 'encoder')

# ...
# ...
```
<br/>



## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## Authors

* **Tanapat Kahabodeekanokkul** - *Author* - [pahntanapat](https://gist.github.com/pahntanapat)


## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* Billie Thompson, [PurpleBooth/README-Template.md](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2) :: Great README template
* etc
