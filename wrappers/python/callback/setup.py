# setup.py
from distutils.core import setup, Extension

import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include  = numpy.get_numpy_include()

pcback = Extension('pycback',
                   sources = ['python_caller.c'],
                   include_dirs = [
                       numpy_include,
                   ],
                   language='c',
                   extra_compile_args=['-std=c99'])


setup(name = "pycback",
      version = "1.0",
      author = "Alex Gorodetsky",
      description = "Utilities for Callback Functions",
      ext_modules=[pcback]
)

# ~/Software/c3_installed/lib/c3/
