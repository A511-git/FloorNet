#!/usr/bin/env python
"""Kaggle/local: verify TensorFlow stack before infer.py.  python check_tf.py"""
from __future__ import print_function

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from middleware.deps_check import verify_tensorflow_backend

ok, msg = verify_tensorflow_backend(ROOT)
print(msg)
sys.exit(0 if ok else 1)
