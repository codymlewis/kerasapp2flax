import unittest

import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import orbax.checkpoint as ocp
import chex
from parameterized import parameterized

import models


# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')


def model_outputs(model_name):
        flax_model = getattr(models, model_name)()
        empty_state = TrainState.create(
            apply_fn=flax_model.apply,
            params=flax_model.init(jax.random.PRNGKey(0), jnp.zeros((1, 224, 224, 3))),
            tx=optax.set_to_zero(),
        )
        train_state = ocp.PyTreeCheckpointer().restore(f"weights/{model_name}", item=empty_state)
        del empty_state

        x = np.random.uniform(high=1.0, size=(100, 224, 224, 3))
        logits = train_state.apply_fn(train_state.params, x, train=False)

        tf_model = getattr(tf.keras.applications, model_name)()
        tf_logits = tf_model(x, training=False).numpy()
        return jnp.argmax(logits, axis=-1), jnp.argmax(tf_logits, axis=-1)


class TestModels(unittest.TestCase):
    @parameterized.expand([
        ("DenseNet121"),
        ("DenseNet169"),
        ("DenseNet201"),
        # ("InceptionV3"),
        # ("MobileNetV2"),
        # ("ResNetRS50"),
        # ("ResNetRS101"),
        # ("ResNetRS152"),
        # ("ResNetRS270"),
        # ("ResNetRS350"),
        # ("ResNetRS420"),
    ])
    def test_Model(self, model_name):
        chex.assert_trees_all_close(*model_outputs(model_name))



if __name__ == "__main__":
    unittest.main()