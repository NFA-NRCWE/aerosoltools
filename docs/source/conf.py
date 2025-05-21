import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'aerosoltools'
author = 'NRCWE community'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'nbsphinx',
]
autosummary_generate = True
nbsphinx_execute = 'auto'
exclude_patterns = ['_build', '**.ipynb_checkpoints']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_typehints = "description"
