"""
TensorFlow 2.x / Kaggle GPU setup for legacy FloorNet (TF1 graph + slim + tflearn).

Call configure() before importing train.py or RecordReader.
PyTorch/CUDA on Kaggle is unused; this enables TensorFlow on the GPU.
"""

from __future__ import print_function

import sys
import types

_CONFIGURED = False
_IS_TRAINING = [False]


def _batch_normalization(inputs, **kwargs):
    import tf_slim as slim

    kwargs.setdefault("is_training", _IS_TRAINING[0])
    kwargs.setdefault("fused", True)
    return slim.batch_norm(inputs, **kwargs)


def _max_pool_2d(inputs, kernel_size, strides=1, padding="valid", **kwargs):
    import tensorflow.compat.v1 as tf

    if isinstance(kernel_size, (int, float)):
        kh, kw = int(kernel_size), int(kernel_size)
    else:
        kh, kw = int(kernel_size[0]), int(kernel_size[1])

    if isinstance(strides, (int, float)):
        sh, sw = int(strides), int(strides)
    else:
        sh, sw = int(strides[0]), int(strides[1])

    pad = padding.upper() if isinstance(padding, str) else "VALID"
    name = kwargs.get("name", None)
    return tf.nn.max_pool(
        inputs,
        ksize=[1, kh, kw, 1],
        strides=[1, sh, sw, 1],
        padding=pad,
        name=name,
    )


def _is_training(mode=None, session=None):
    if mode is not None:
        _IS_TRAINING[0] = bool(mode)
    return _IS_TRAINING[0]


class _ConfigShim(object):
    @staticmethod
    def init_training_mode():
        return None


class _LayersShim(object):
    def __init__(self):
        self.normalization = types.SimpleNamespace(batch_normalization=_batch_normalization)
        self.conv = types.SimpleNamespace(max_pool_2d=_max_pool_2d)
        self.max_pool_2d = _max_pool_2d


def _install_tflearn_shim():
    if "tflearn" in sys.modules and getattr(sys.modules["tflearn"], "_floornet_shim", False):
        return

    shim = types.ModuleType("tflearn")
    shim._floornet_shim = True
    shim.layers = _LayersShim()
    shim.config = _ConfigShim()
    shim.is_training = _is_training
    shim.init_graph = lambda *args, **kwargs: None

    sys.modules["tflearn"] = shim
    sys.modules["tflearn.layers"] = shim.layers
    sys.modules["tflearn.layers.normalization"] = shim.layers.normalization
    sys.modules["tflearn.layers.conv"] = shim.layers.conv
    sys.modules["tflearn.config"] = shim.config


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
