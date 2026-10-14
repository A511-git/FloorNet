"""Verify TensorFlow + tf-slim for inference (prints actionable errors)."""

from __future__ import print_function

import sys


def verify_tensorflow_backend(repo_root=None):
    """
    Returns (ok: bool, message: str).
    Tries the same imports infer_worker uses.
    """
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from middleware.tf_setup import configure

        configure()
        import tensorflow as tf
        import tf_slim  # noqa: F401
        # Same import chain as infer_worker (must not use tensorflow.contrib.slim on TF2)
        import train  # noqa: F401
        from train import build_graph  # noqa: F401

        version = getattr(tf, "__version__", "unknown")
        gpus = []
        try:
            gpus = tf.config.list_physical_devices("GPU")
        except Exception:
            pass
        gpu_note = "GPU: %s" % (gpus if gpus else "none (CPU only)")
        return True, "TensorFlow %s — %s" % (version, gpu_note)
    except ImportError as e:
        return False, "ImportError: %s" % (e,)
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e)
