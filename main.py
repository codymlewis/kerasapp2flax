import tensorflow as tf
import jax
import jaxlib
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from fuzzywuzzy import process

import models.resnetrs


def copy_dict(d):
    copy = {}
    for key, value in d.items():
        if type(value) is FrozenDict:
            copy[key] = copy_dict(value)
        else:
            copy[key] = value
    return copy


def update_multidict(multidict, keys, value, km):
    if len(keys):
        multidict[keys[0]] = update_multidict(multidict[keys[0]], keys[1:], value, km)
    else:
        return value
    return multidict


if __name__ == "__main__":
    jax_params = models.resnetrs.ResNetRS50(1000).init(jax.random.PRNGKey(0), jnp.zeros((1, 224, 224, 3)))
    tf_model = tf.keras.applications.ResNetRS50()
    tf_params = {w.name: jnp.array(w.value().numpy()) for w in tf_model.weights}
    final_params = copy_dict(jax_params)
    for k in tf_params.keys():
        k_orig = k
        k = k.replace('__', '_')
        key = []
        if "batch_norm" in k:
            if 'gamma' in k or 'beta' in k:
                continue
            key.append('batch_stats')
        else:
            key.append('params')
        jkp = jax_params[key[-1]]
        while type(jkp) is not jaxlib.xla_extension.DeviceArray:
            key.append(process.extractOne(k, jkp.keys())[0])
            jkp = jkp[key[-1]]
        final_params = update_multidict(final_params, key, tf_params[k_orig], k)
    model = models.resnetrs.ResNetRS50(1000)
    model.apply(final_params, jnp.zeros((1, 224, 224, 3)), train=False, rngs={'dropout': jax.random.PRNGKey(0)})
