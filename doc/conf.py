#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# xarray-simlab documentation build configuration file, created by
# sphinx-quickstart on Sun Jun 11 14:04:24 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import sys
import os

print("python exec:", sys.executable)
print("sys.path:", sys.path)
try:
    import numpy
    print("numpy: %s, %s" % (numpy.__version__, numpy.__file__))
except ImportError:
    print("no numpy")
try:
    import xarray
    print("xarray: %s, %s" % (xarray.__version__, xarray.__file__))
except ImportError:
    print("no xarray")

import xrsimlab
print("xrsimlab: %s, %s" % (xrsimlab.__version__, xrsimlab.__file__))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',
    # 'numpydoc',
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'nbsphinx',
]

extlinks = {
    'issue': ('https://github.com/benbovy/xarray-simlab/issues/%s', 'GH'),
    'pull': ('https://github.com/benbovy/xarray-simlab/pull/%s', 'PR')
}

autosummary_generate = True

napoleon_numpy_docstring = True
# numpydoc_class_members_toctree = True
# numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'xarray-simlab'
copyright = '2017, xarray-simlab Developers'
author = 'Benoit Bovy and xarray-simlab Developers'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = xrsimlab.__version__
# The full version, including alpha/beta/rc tags.
release = xrsimlab.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
# language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from
# docs.readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# otherwise, readthedocs.org uses their theme by default, so no need
# to specify it

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'xarray-simlab-doc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'xarray-simlab.tex', 'xarray-simlab Documentation',
     'xarray-simlab Developers', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'xarray-simlab', 'xarray-simlab Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'xarray-simlab', 'xarray-simlab Documentation',
     author, 'xarray-simlab', 'xarray extension for computer model simulations',
     'Miscellaneous'),
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3.6/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'xarray': ('http://xarray.pydata.org/en/stable/', None)
}


# -- Extend autodoc for xarray accessors

# The code below has been taken from pandas v0.20.2
# (https://github.com/pandas-dev/pandas) - License BSD 3-clause

import sphinx
from sphinx.util import rpartition
from sphinx.ext.autodoc import Documenter, MethodDocumenter, AttributeDocumenter
from sphinx.ext.autosummary import Autosummary


class AccessorDocumenter(MethodDocumenter):
    """
    Specialized Documenter subclass for accessors.
    """

    objtype = 'accessor'
    directivetype = 'method'

    # lower than MethodDocumenter so this is not chosen for normal methods
    priority = 0.6

    def format_signature(self):
        # this method gives an error/warning for the accessors, therefore
        # overriding it (accessor has no arguments)
        return ''


class AccessorLevelDocumenter(Documenter):
    """
    Specialized Documenter subclass for objects on accessor level (methods,
    attributes).
    """

    # This is the simple straightforward version
    # modname is None, base the last elements (eg 'hour')
    # and path the part before (eg 'Series.dt')
    # def resolve_name(self, modname, parents, path, base):
    #     modname = 'pandas'
    #     mod_cls = path.rstrip('.')
    #     mod_cls = mod_cls.split('.')
    #
    #     return modname, mod_cls + [base]

    def resolve_name(self, modname, parents, path, base):
        if modname is None:
            if path:
                mod_cls = path.rstrip('.')
            else:
                mod_cls = None
                # if documenting a class-level object without path,
                # there must be a current class, either from a parent
                # auto directive ...
                mod_cls = self.env.temp_data.get('autodoc:class')
                # ... or from a class directive
                if mod_cls is None:
                    mod_cls = self.env.temp_data.get('py:class')
                # ... if still None, there's no way to know
                if mod_cls is None:
                    return None, []
            # HACK: this is added in comparison to ClassLevelDocumenter
            # mod_cls still exists of class.accessor, so an extra
            # rpartition is needed
            modname, accessor = rpartition(mod_cls, '.')
            modname, cls = rpartition(modname, '.')
            parents = [cls, accessor]
            # if the module name is still missing, get it like above
            if not modname:
                modname = self.env.temp_data.get('autodoc:module')
            if not modname:
                if sphinx.__version__ > '1.3':
                    modname = self.env.ref_context.get('py:module')
                else:
                    modname = self.env.temp_data.get('py:module')
            # ... else, it stays None, which means invalid
        return modname, parents + [base]


class AccessorAttributeDocumenter(AccessorLevelDocumenter, AttributeDocumenter):

    objtype = 'accessorattribute'
    directivetype = 'attribute'

    # lower than AttributeDocumenter so this is not chosen for normal attributes
    priority = 0.6


class AccessorMethodDocumenter(AccessorLevelDocumenter, MethodDocumenter):

    objtype = 'accessormethod'
    directivetype = 'method'

    # lower than MethodDocumenter so this is not chosen for normal methods
    priority = 0.6


class AccessorCallableDocumenter(AccessorLevelDocumenter, MethodDocumenter):
    """
    This documenter lets us removes .__call__ from the method signature for
    callable accessors like Series.plot
    """
    objtype = 'accessorcallable'
    directivetype = 'method'

    # lower than MethodDocumenter; otherwise the doc build prints warnings
    priority = 0.5

    def format_name(self):
        return MethodDocumenter.format_name(self).rstrip('.__call__')


def setup(app):
    app.add_autodocumenter(AccessorDocumenter)
    app.add_autodocumenter(AccessorAttributeDocumenter)
    app.add_autodocumenter(AccessorMethodDocumenter)
    app.add_autodocumenter(AccessorCallableDocumenter)
