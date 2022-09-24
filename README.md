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
- The ResNet variations
- The ResNet-V2 variations
- The ResNet-RS variations
- Inception-V3
- MobileNetV2
- The EfficientNet variations

## Translation Notes

For this section I note each of the quirks that I happened across in translation.

In a lot of cases there is a near one-to-one mapping between tensorflow and the flax library,
however there are many edge cases created mainly by the difference between object oriented and
functional design. I am only noting those edge cases.

In the following, code samples will use the [flax.linen](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html)
library aliased as `nn`, the [jax.numpy](https://jax.readthedocs.io/en/latest/jax.numpy.html) library aliased as
`jnp`, and the [einops](https://einops.rocks/) library without an alias. We use `b` to refer to the batch size,
`h` to the height, `w` to width, and `c` or `d` to refer to the channels/depth.


Note: there is a bias towards the 2D variations of these layers.


### Generally Translating Parameters

By carefully ensuring that the names of layers match across to two models involved in the translation process,
I found that it is mostly simple to translate the parameters by using fuzzy matching to rearrange the keras 
parameters into a dictionary of equivalent shape to the flax parameters.

Note that processing the parameter names to remove the underscores, remove the ':0', and/or make them entirely 
lower-case often helps with the matching process.

### Add

The add layer is very simple in jax-based machine learning, simply add together the two arrays as normal.

```python
#  x and y are jax.numpy arrays, perhaps the output of another nn layer
x + y
```

### Multiply

Similar to the add layer, multiply is very simple, just multiply the arrays as normal.

```python
#  x and y are jax.numpy arrays, perhaps the output of another nn layer
x * y
```

### Padding

Padding layers are replaced with the jax.numpy.pad function, where padding is applied as intended to the middle
dimensions of the array using the tuple specification.

```python
# Here x has the dimensions of (b, h, w, c)
x = jnp.pad(x, ((0, 0), (p1, p1), (p2, p2), (0, 0)))
# The resulting x has the dimensions of (b, p1 + h + p1, p2 + w + p2, c)
```

### Concatenate

Concat is replaced with the jax.numpy.concatenate function, where the first argument is a tuple of arrays to 
concatenate, and then the axis can be specified as normal.

```python
x = jnp.concatenate((x, y), axis=3)
```


### Global Average Pooling

Global average pooling is a bit interesting as it is more of a function than a true layer due a lack of
parameters. In jax, we can do this function by taking the mean of the input across the intermediate dimensions
outputting a 2D array with (b, d) dimensions.

```python
# x has the dimensions of (b, h, w, d) for this example
x = einops.reduce(x, 'b h w d -> b d', 'mean')
```

### Convolutions

Convolutional layers are mostly one-to-one between the libraries, but the main difference to note is that the
default padding in keras is "valid" while in flax it is "SAME".

```python
x = nn.Conv(filters, kernel, padding="VALID")(x)
```

### Batch Normalization

Again, batch normalization is mostly one-to-one with keras, where the main difference is that the flax version is
stateless, meaning the addition of the `use_running_average` and the need to explicitly track the batch
statistics during training.

```
x = nn.BatchNorm(use_running_average=not train, axis=axis)(x)
```

When translating parameters, it must be noted that batch normalization involves both parameters and batch
statistics. The parameters in keras called gamma and beta which correspond to the scale and bias parameters in
flax. The batch statistics have names close enough to be reliably fuzzy matched.

### Depthwise Convolutions

Depthwise convolutions are easily one of the most complicated layers to translate, due to involving some obscure
parameters and the need to rearrange parameters between the two libraries. Firstly, to create a depthwise
convolution in flax, the number of parameters must be equal to the cardinality of the depth dimension, then
the `feature_group_count` parameter must also be set to the cardinality of the depth dimension.

```python
x = nn.Conv(x.shape[-1], kernel, feature_group_count=x.shape[-1])(x)
```

A depth multiplier can also be constructed by multiplying the number of filters by the desired depth multiplier.

```python
x = nn.Conv(x.shape[-1] * depth_multiplier, kernel, feature_group_count=x.shape[-1])(x)
```

Otherwise, standard convolutional layer hyperparameters directly correspond between the libraries. 

While the output of the layers will hold the same shape and rotation as the corresponding keras layer output,
the parameters differ greatly. Firstly, in the case of a depth multiplier of 1, the last two dimensions of the
parameters are transposed.

```python
dw_conv_kernel_params = einops.rearrange(tf_dw_conv.get_weights()[0], 'b h w d -> b h d w')
```

When the depth multipler is greater than 1, flax places the final two dimensions of the corresponding keras
parameters into the final dimension and sets the second last one to be 1 dimensional. The translation is thus:

```python
dw_conv_kernel_params = einops.rearrange(tff.get_weights()[0], 'b h d dm -> b h 1 (d dm)')
```
