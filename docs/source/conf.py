import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, os.path.abspath("../../NuLattice"))

project = 'NuLattice'
copyright = '2025, M. Rothman, B. Johnson-Toth, G. Hagen, M. Heinz, T. Papenbrock'
author = 'M. Rothman, B. Johnson-Toth, G. Hagen, M. Heinz, T. Papenbrock'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']
pygments_style = 'sphinx'

html_theme_options = {
    'body_max_width' : 'none',
    'page_width': 'auto',
}
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": True,
    "member-order": 'bysource'
}
