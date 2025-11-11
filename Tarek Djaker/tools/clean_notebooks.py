#!/usr/bin/env python3
"""Strip outputs and execution counts from Jupyter notebooks under a root.

Usage:
    python tools/clean_notebooks.py [ROOT]

If ROOT is omitted, the script cleans the parent directory of this file.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict


def clean_notebook(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb: Dict[str, Any] = json.load(f)
    except Exception:
        return False

    cells = nb.get("cells")
    if isinstance(cells, list):
        for cell in cells:
            if isinstance(cell, dict):
                cell["outputs"] = []
                if "execution_count" in cell:
                    cell["execution_count"] = None

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")
        return True
    except Exception:
        return False


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    changed = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".ipynb"):
                nb_path = os.path.join(dirpath, name)
                if clean_notebook(nb_path):
                    changed += 1
                    print(f"Cleaned: {nb_path}")
    print(f"Done. Notebooks cleaned: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

