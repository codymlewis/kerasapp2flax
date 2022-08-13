import argparse
from difflib import SequenceMatcher
import logging
import re
import os

import numpy as np
import scipy.spatial.distance
import tensorflow as tf
import jax
import jaxlib
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax import serialization
from fuzzywuzzy import fuzz, process
import einops

import models

# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')


def copy_dict(d):
    copy = {}
    for key, value in d.items():
        if type(value) is FrozenDict:
            copy[key] = copy_dict(value)
        else:
            copy[key] = value
    return copy


def update_multidict(multidict, keys, value, fk=None):
    if len(keys):
        if len(keys) == 1:
            k = keys[0]
            if multidict[k].shape != value.shape:
                logging.error(f"NON MATCH {fk=} md orig: {multidict[k].shape}, value shape: {value.shape}")
        multidict[keys[0]] = update_multidict(multidict[keys[0]], keys[1:], value, keys if fk is None else fk)
    else:
        return value
    return multidict


def best_match(key, choices):
    key = key.replace('_', '')
    winner, swinner = None, 0
    for (c, s) in choices:
        if ((pc := c.replace('_', '')) in key or key in pc) and \
                (winner is None or len(c) > len(winner)) and s >= swinner:
            winner = c
            swinner = s
    if winner is None:
        winner, _ = choices[0]
    return winner


def inception_scorer(key: str, query: str) -> int:
    key_orig, query_orig = key, query
    key.replace('batch_normalization', 'BN')
    key = key.replace('_', '').replace('/', '').replace(':0', '').lower()
    query = query.replace('_', '').replace('/', '').replace(':0', '').lower()
    k = np.array([int(b) for b in bytes(key, 'utf-8')])
    k = np.pad(k, (0, 150 - len(k)))
    q = np.array([int(b) for b in bytes(query, 'utf-8')])
    q = np.pad(q, (0, 150 - len(q)))
    score = int(100 * (1 - scipy.spatial.distance.hamming(k, q)))
    score += SequenceMatcher(None, key, query).find_longest_match().size
    if re.match(r'.*_\d.*', key_orig) and re.match(r'.*_\d.*', query_orig):
        if int(re.findall('_\d\d?', key_orig)[0][1:]) == int(re.findall('_\d\d?', query_orig)[0][1:]):
            score += 100
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate the keras applications weights to flax weights.")
    parser.add_argument("--model", type=str, default="ResNetRS50", help="Model to translate the weights of.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    print("Downloading and translating model...")
    scorer = inception_scorer if 'inception' in args.model.lower() else fuzz.WRatio
    jax_variables = getattr(models, args.model)().init(
        jax.random.PRNGKey(0), jnp.zeros((1, 224, 224, 3))
    )
    tf_model = getattr(tf.keras.applications, args.model)()
    tf_variables = {w.name: jnp.array(w.value().numpy()) for w in tf_model.weights}
    final_variables = copy_dict(jax_variables)
    for k in tf_variables.keys():
        k_orig = k
        k = k.replace('__', '_')
        key = []
        if "batch_norm" in k.lower() or "bn" in k.lower():
            if 'gamma' in k or 'beta' in k:
                key.append('params')
                k = k.replace('gamma', 'scale')
                k = k.replace('beta', 'bias')
            else:
                key.append('batch_stats')
        else:
            key.append('params')
        jkp = jax_variables[key[-1]]
        if args.model == "MobileNetV2":
            if (m := re.match(r'block_\d\d?', k)):
                block_id = re.findall(r'\d\d?', m.group())[0]
                key.append(f'InvertedResBlock_{block_id}')
                jkp = jkp[key[-1]]
            elif re.match('expanded_conv_', k):
                key.append('InvertedResBlock_0')
                jkp = jkp[key[-1]]
        while type(jkp) is not jaxlib.xla_extension.DeviceArray:
            matches = process.extract(k, jkp.keys(), scorer=scorer)
            key.append(best_match(k, matches))
            jkp = jkp[key[-1]]
            logging.info(f"Matched {k} to {key[-1]}")
        if "depthwise" in k_orig and key[0] != "batch_stats":
            tf_vars = einops.rearrange(tf_variables[k_orig], 'b h c w -> b h w c')
        else:
            tf_vars = tf_variables[k_orig]
        final_variables = update_multidict(final_variables, key, tf_vars)
    
    os.makedirs('weights', exist_ok=True)
    with open((fn := f"weights/{args.model}.variables"), 'wb') as f:
        f.write(serialization.to_bytes(final_variables))
    print(f"Written pretrained variables to {fn}")

    # Model testing
#    x = np.random.uniform(size=(100, 224, 224, 3))
#    print("JAX Model")
#    logits = getattr(models, args.model)().apply(final_variables, x, train=False)
#    print(jnp.argmax(logits, axis=-1))
#
#    print("tf model")
#    tf_logits = tf_model(x).numpy()
#    print(jnp.argmax(tf_logits, axis=-1))
#    print(f"Match: {(jnp.argmax(logits, axis=-1) == jnp.argmax(tf_logits, axis=-1)).all()}")
