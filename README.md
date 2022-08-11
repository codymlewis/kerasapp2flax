# Keras Applications to JAX

Flax linen implementations of keras applications neural network models. Also
includes a parameter translator to get the pretrained weights provided in the
tensorflow keras applications library.

## Running the translator

First install [JAX](https://github.com/google/jax), then run
`pip install -U -r requirements.txt` in the cloned directory of this repository.

Finally, execute the translator with `python main.py --model <MODEL>` specifying the
model weights you want to translate.

## Implemented Models

- The DenseNet variations
- The ResNet-RS variations
- Inception-V3
- MobileNetV2
