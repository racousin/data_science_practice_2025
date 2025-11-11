# Profile: Tarek Djaker

Personal workspace with cleaned modules and helper tools.

What’s included
- Modules populated and cleaned: `module1` .. `module9` (notebook outputs stripped; generated results removed).
- `module2/mysupertools`: package with type hints and robust numeric checks.
- Tools for maintenance under `tools/`.
- `.gitignore` to keep artifacts out of version control.

Identity
- `module1/user` contains: `Tarek Djaker,Tarek,Djaker ` (username, first name, last name).

Tools
- PowerShell cleaner: `tools/clean_repo.ps1`
  - Cleans notebook outputs and removes generated CSVs.
  - Usage (PowerShell): `./tools/clean_repo.ps1` or `./tools/clean_repo.ps1 -Root <path>`
- Python cleaner: `tools/clean_notebooks.py`
  - Strips outputs from `.ipynb` files without extra dependencies.
  - Usage: `python tools/clean_notebooks.py` or `python tools/clean_notebooks.py <path>`
- Quick check for mysupertools: `tools/validate_mysupertools.py`
  - Runs simple assertions for `multiply` locally.
  - Usage: `python tools/validate_mysupertools.py`

Notes
- All CSV results (submission/predictions) have been removed; regenerate as needed.
- Keep improvements ethical and focused on readability, robustness, and reproducibility.

## Personalization

This profile includes a small helper library and settings unique to Tarek Djaker:

- Config: `profile.json` defines identity, random seed, display and plot preferences.
- Library: `lib/tarek_profile` provides utilities to initialize notebooks consistently.
- Plot style: `lib/tarek_profile/mplstyles/tarek.mplstyle` for a distinctive visual style.

Use in notebooks

- Quick snippet (prints to console): `python tools/profile_snippet.py`
- Or paste this at the top of your notebook:

  ```python
  import sys
  sys.path.append(r"./lib")   # adjust if running from another cwd
  from tarek_profile import nb_init, profile_banner
  nb_init()
  profile_banner(title=None)
  ```

Optional: auto-insert header cell into existing notebooks

- PowerShell: `./tools/add_profile_header.ps1` (adds a small init cell at the top)
- Dry-run: `./tools/add_profile_header.ps1 -WhatIf`
