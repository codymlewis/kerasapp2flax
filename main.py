import argparse
from difflib import SequenceMatcher
import logging
import re

import numpy as np
import scipy.spatial.distance
import tensorflow as tf
import jax
import jaxlib
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from fuzzywuzzy import fuzz, process

import models.resnetrs


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
                print(f"NON MATCH {fk=} md orig: {multidict[k].shape}, value shape: {value.shape}")
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

    scorer = inception_scorer if 'inception' in args.model.lower() else fuzz.WRatio
    jax_variables = getattr(models, args.model)(1000).init(
        jax.random.PRNGKey(0), jnp.zeros((1, 224, 224, 3))
    )
    tf_model = getattr(tf.keras.applications, args.model)()
    tf_variables = {w.name: jnp.array(w.value().numpy()) for w in tf_model.weights}
    final_variables = copy_dict(jax_variables)
    for k in tf_variables.keys():
        k_orig = k
        k = k.replace('__', '_')
        key = []
        if "batch_norm" in k or "bn" in k:
            if 'gamma' in k or 'beta' in k:
                continue
            key.append('batch_stats')
        else:
            key.append('params')
        jkp = jax_variables[key[-1]]
        while type(jkp) is not jaxlib.xla_extension.DeviceArray:
            matches = process.extract(k, jkp.keys(), scorer=scorer)
            key.append(best_match(k, matches))
            jkp = jkp[key[-1]]
            logging.info(f"Matched {k} to {key[-1]}")
        final_variables = update_multidict(final_variables, key, tf_variables[k_orig])
    model = getattr(models, args.model)(1000)
    print(jnp.argmax(model.apply(final_variables, jnp.zeros((1, 224, 224, 3)), train=False, rngs={'dropout': jax.random.PRNGKey(0)})))
