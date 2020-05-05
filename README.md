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

### MaskCuDNNGRU
```
MaskCuDNNGRU(units: int, return_sequences: bool = False, return_state: bool = False, **kwargs)
```

[](https://keras.io/layers/recurrent/#cudnngru)

```
from mask_cudnn_rnn import MaskCuDNNGRU

mask_gru = MaskCuDNNGRU()
```

#### For example
```
from keras.layer import Embedding
from mask_cudnn_rnn import MaskCuDNNGRU

Embedding(vocab_size+1, vector_size,  mask_zero=True)
```
### Return stage and sequence


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
