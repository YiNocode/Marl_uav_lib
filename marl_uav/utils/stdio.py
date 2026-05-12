"""Process stdio helpers for Windows-safe Unicode logs."""

from __future__ import annotations

import os
import sys


def configure_utf8_stdio() -> None:
    """Prefer UTF-8 stdio and replace unencodable characters.

    Genesis prints Unicode banner/status lines during initialization. On some
    Windows shells Python defaults to GBK/cp936, which can crash logging before
    training reaches TensorBoard initialization.
    """

    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def utf8_subprocess_env() -> dict[str, str]:
    """Return an environment dict that makes child Python stdio UTF-8."""

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env
