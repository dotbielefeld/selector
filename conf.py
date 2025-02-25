import os
import sys

# Add the module path (if needed)
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
project = 'selector'
copyright = '2024, Dimitri Weiss'
author = 'Dimitri Weiss'
release = '0.0.1'

# -- Sphinx Extensions ------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
]

# -- Mocking Missing Modules ------------------------------------------------
# This prevents Sphinx from failing when importing unavailable dependencies
#autodoc_mock_imports = ["selector", "selector.run_ac", "irace"]

# Alternatively, you can mock them explicitly using unittest.mock
#MOCK_MODULES = ["selector", "selector.run_ac"]
#for mod_name in MOCK_MODULES:
#    sys.modules[mod_name] = mock.Mock()

# MOCK_MODULES = ["selector", "selector.run_ac"]
# sys.modules.update((mod, mock.Mock()) for mod in MOCK_MODULES)

# autodoc_mock_imports = ["selector", "selector.run_ac", "irace", "ac", "up_lpg",
#                        "smac"]

# -- HTML Output ------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
templates_path = ['_templates']
html_static_path = ['_static']
html_build_dir = '$READTHEDOCS_OUTPUT/html/'
