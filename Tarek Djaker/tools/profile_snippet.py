#!/usr/bin/env python3
"""Print a small code snippet to initialize notebooks with your profile.

Usage:
    python tools/profile_snippet.py
"""
from __future__ import annotations

import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lib = root / "lib"

snippet = f"""
# --- Tarek Djaker notebook profile ---
import sys, os
sys.path.append(r"{lib}")
from tarek_profile import nb_init, profile_banner
nb_init()
profile_banner(title=__name__ if '__name__' in globals() else None)
# -------------------------------------
""".lstrip()

print(snippet)

