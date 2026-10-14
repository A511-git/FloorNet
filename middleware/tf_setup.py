"""
TensorFlow 2.x / Kaggle GPU setup for legacy FloorNet (TF1 graph + slim + tflearn).

Call configure() before importing train.py or RecordReader.
PyTorch/CUDA on Kaggle is unused; this enables TensorFlow on the GPU.
"""

from __future__ import print_function

import sys
import types

_CONFIGURED = False


def _install_tflearn_shim():
    import tensorflow.compat.v1 as tf

    def batch_normalization(inputs, **kwargs):
        import tf_slim as slim

        return slim.batch_norm(inputs, fused=True)

    def max_pool_2d(inputs, kernel_size, strides=1, padding="valid"):
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = int(kernel_size[0]), int(kernel_size[1])
        pad = padding.upper() if isinstance(padding, str) else "VALID"
        return tf.nn.max_pool(
            inputs,
            ksize=[1, kh, kw, 1],
            strides=[1, int(strides), int(strides), 1],
            padding=pad,
        )

    class _Config(object):
        @staticmethod
        def init_training_mode():
            return None

    _is_training = [False]

    @staticmethod
    def is_training(mode=None):
        if mode is not None:
            _is_training[0] = bool(mode)
        return _is_training[0]

    class _Layers(object):
        normalization = types.SimpleNamespace(batch_normalization=batch_normalization)
        max_pool_2d = staticmethod(max_pool_2d)

    class _TflearnShim(object):
        layers = _Layers()
        config = _Config()
        is_training = is_training

    sys.modules["tflearn"] = _TflearnShim()


def _patch_contrib_slim():
    import tensorflow as tf
    import tf_slim as slim

    if not hasattr(tf, "contrib"):
        tf.contrib = types.ModuleType("contrib")
    tf.contrib.slim = slim


def configure(enable_gpu_growth=True):
    global _CONFIGURED
    if _CONFIGURED:
        return
    import tensorflow as tf

    tf.compat.v1.disable_v2_behavior()
    _install_tflearn_shim()
    _patch_contrib_slim()

    if enable_gpu_growth:
        try:
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    _CONFIGURED = True
