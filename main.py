import argparse
import logging

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


def update_multidict(multidict, keys, value):
    if len(keys):
        multidict[keys[0]] = update_multidict(multidict[keys[0]], keys[1:], value)
    else:
        return value
    return multidict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate the keras applications weights to flax weights.")
    parser.add_argument("--model", type=str, default="ResNetRS50", help="Model to translate the weights of.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    jax_variables = getattr(models.resnetrs, args.model)(1000).init(
        jax.random.PRNGKey(0), jnp.zeros((1, 224, 224, 3))
    )
    tf_model = getattr(tf.keras.applications, args.model)()
    tf_variables = {w.name: jnp.array(w.value().numpy()) for w in tf_model.weights}
    final_variables = copy_dict(jax_variables)
    for k in tf_variables.keys():
        k_orig = k
        k = k.replace('__', '_')
        key = []
        if "batch_norm" in k:
            if 'gamma' in k or 'beta' in k:
                continue
            key.append('batch_stats')
        else:
            key.append('params')
        jkp = jax_variables[key[-1]]
        while type(jkp) is not jaxlib.xla_extension.DeviceArray:
            key.append(process.extractOne(k, jkp.keys())[0])
            jkp = jkp[key[-1]]
            logging.info(f"Matched {k} to {key[-1]}")
        final_variables = update_multidict(final_variables, key, tf_variables[k_orig])
    model = getattr(models.resnetrs, args.model)(1000)
    print(jnp.argmax(model.apply(final_variables, jnp.zeros((1, 224, 224, 3)), train=False, rngs={'dropout': jax.random.PRNGKey(0)})))
