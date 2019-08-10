"""
@author:    Patrik Purgai
@copyright: Copyright 2019, supervised-translation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.07.03.
"""

# pylint: disable=no-name-in-module
# pylint: disable=import-error

from distutils.core import setup
from Cython.Build import cythonize

import numpy
import os

project_path = os.path.abspath(os.path.dirname(__file__))
collates_path = os.path.join(project_path, 'src', 'collate.pyx')
tokenize_path = os.path.join(project_path, 'src', 'tokenize.pyx')

setup(
    ext_modules=cythonize([
        collates_path, tokenize_path
    ], annotate=True, language='c++'),
    include_dirs=[numpy.get_include()],
)
