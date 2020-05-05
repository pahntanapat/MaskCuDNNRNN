# MaskCuDNNRNN
Custom layer for CuDNN RNN which are compatible with masking

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The MaskCuDNNRNN requires ...
1. [Keras](https://keras.io/#installation)
2. [GPU driver](https://www.nvidia.com/Download)
3. [CUDA](https://developer.nvidia.com/cuda-downloads)
4. [CuDNN](https://developer.nvidia.com/cudnn)

If No. 2 - 4 are not completely installed, you can use only CPU-compatible GRU layer.


### Installing

Now, we do not distribute it on any channel. Please clone the git or copy the [mask_cudnn_rnn.py](https://github.com/pahntanapat/MaskCuDNNRNN/blob/master/mask_cudnn_rnn.py).


## Usages

### MaskCuDNNGRU <a name="MaskCuDNNGRU"></a>
```
MaskCuDNNGRU(units: int, return_sequences: bool = False, return_state: bool = False, **kwargs)
```
Masking-compatible CuDNNGRU

#### Argument
* units: Positive integer, dimensionality of the output space.
* return_sequences: boolean, if True, return the full sequence, else, return the last output.
* return_state: boolean, whether to return the last state in addition to the output.
* **kwargs: other optional arguments, same to Keras' [CuDNNGRU](https://keras.io/layers/recurrent/#cudnngru)


#### Example
```
from keras.layer import Embedding, Input
from mask_cudnn_rnn import MaskCuDNNGRU

embed = Embedding(vocab_size+1, vector_size,  mask_zero=True)
encoder = MaskCuDNNGRU()

decoder_1 = MaskCuDNNGRU()
decoder_2 = MaskCuDNNGRU()
fc = Dense()
```


### mask_cudnn_gru
```
mask_cudnn_gru(units: int, use_cudnn: bool, return_sequences: bool = False, return_state: bool = False, **kwargs)
```
This function can create [MaskCuDNNGRU](#MaskCuDNNGRU) and CuDNN-compatible [conventional GRU](https://keras.io/layers/recurrent/#gru).

#### Argument
* units: Positive integer, dimensionality of the output space.
* use_cudnn: boolean, If True, return [MaskCuDNNGRU](#MaskCuDNNGRU), else, return [GRU](https://keras.io/layers/recurrent/#gru).
* return_sequences: boolean, if True, return the full sequence, else, return the last output.
* return_state: boolean, whether to return the last state in addition to the output.
* **kwargs: other optional arguments, same to Keras' [CuDNNGRU](https://keras.io/layers/recurrent/#cudnngru)

#### Example
```
from mask_cudnn_rnn import MaskCuDNNGRU

mask_gru = MaskCuDNNGRU()
```




## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## Authors

* **Tanapat Kahabodeekanokkul** - *Author* - [pahntanapat](https://gist.github.com/pahntanapat)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.


## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* Billie Thompson, [PurpleBooth/README-Template.md](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2) :: Great README template
* etc
