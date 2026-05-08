"""Multiprocessing start method defaults for Gymnasium vector env workers."""

from __future__ import annotations

import os
import sys

_VALID = frozenset({"spawn", "fork", "forkserver"})


def default_vec_env_context() -> str:
    """Choose AsyncVectorEnv multiprocessing context.

    - ``VEC_ENV_MP_CONTEXT`` env may override (spawn|fork|forkserver).
    - Windows: ``spawn`` (required).
    - Linux / POSIX (non-mac): ``fork`` (lowest worker startup / IPC vs ``spawn``; use ``forkserver``
      if fork is unsafe for your stack).
    - macOS: ``spawn`` (safer with Obj-C runtimes / GUI stacks).
    """
    env_v = os.environ.get("VEC_ENV_MP_CONTEXT", "").strip().lower()
    if env_v in _VALID:
        return env_v
    if sys.platform == "win32":
        return "spawn"
    if sys.platform == "darwin":
        return "spawn"
    return "fork"
