#!/usr/bin/env python3
"""Quick sanity checks for mysupertools.multiply without external deps.

Run from repo root:
    python Tarek Djaker/ayouboa30_optimized/tools/validate_mysupertools.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PKG_PATH = os.path.join(ROOT, "module2", "mysupertools")

sys.path.insert(0, PKG_PATH)

from mysupertools.tool.operation_a_b import multiply  # type: ignore  # noqa: E402


def main() -> int:
    assert multiply(4, 5) == 20
    assert multiply(-1, 5) == -5
    assert multiply("a", 5) == "error"
    assert multiply(None, 5) == "error"
    print("mysupertools.multiply OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

