"""Configuration of sphinx."""
import os
import sys
from unittest.mock import MagicMock


class FakeRay:
    def remote(self, *args, **kwargs):
        def decorator(func):
            return func  # Simply return the function without modification
        return decorator


# Completely replace "ray" with a fake module, since it breaks sphinx
sys.modules["ray"] = MagicMock(remote=FakeRay().remote)

# Add the module path (if needed)
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
project = 'selector'
copyright = '2024, Dimitri Weiss'
author = 'Dimitri Weiss'
release = '0.0.1'

# -- Sphinx Extensions ------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',  # Extracts docstrings
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',  # Supports Google & NumPy docstrings
    'sphinx.ext.autosummary',  # Generates summary tables
    'sphinx.ext.viewcode',  # Adds links to source code
    'sphinxcontrib.bibtex',  # bibtex
]

bibtex_bibfiles = ["references.bib"]

autosummary_generate = True  # Auto-generate `_autosummary` docs

autodoc_default_options = {
    'members': True,
    'undoc-members': True,  # Include members without docstrings
    'show-inheritance': True,
    # "special-members": "__init__",  # Ensure constructors are documented
    # "exclude-members": "__weakref__",  # Avoid unnecessary members
    "inherited-members": False,
    # "autodoc_inherit_docstring": False,
    "exclude-members": "default",
    "special-members": "default",

}

autodoc_class_signature = "explicit"

exclude_patterns = [
    '_autosummary/selector.wrapper.rst',
    '_autosummary/selector.test_ray.rst',
    '_autosummary/selector.main.rst',
]
autodoc_mock_imports = ["json", "json.JSONEncoder"]

# -- HTML Output ------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
templates_path = ['_templates']
html_static_path = ['_static']
html_build_dir = '$READTHEDOCS_OUTPUT/html/'


def skip_classes(app, what, name, obj, skip, options):
    # List of class names to exclude
    excluded_classes = ["GGApp",
                        "GGAppBaggingRegressor",
                        "GGAppRandomForestRegressor",
                        "GGAppRegressorMixin",
                        "BaggingRegressor",
                        "LocalSearch",
                        "FlexInstanceSet",
                        "InstanceSet",
                        "LoadOptionsFromFile",
                        "dummy_task",
                        "tae_from_aclib"]

    if what == "class" and name in excluded_classes:
        print(f"Skipping class: {name}")  # Debugging output
        return True  # Force exclusion
    
    # Also exclude their methods and attributes
    if what in ["method", "attribute"] and hasattr(obj, "__qualname__"):
        for cls in excluded_classes:
            if obj.__qualname__.startswith(cls):
                print(f"Skipping method/attribute: {obj.__qualname__}")
                return True

    return skip


def setup(app):
    # app.connect("autodoc-process-docstring", unwrap_ray_classes)
    app.connect("autodoc-skip-member", skip_classes)  # Keep your existing function
