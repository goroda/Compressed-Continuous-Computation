# setup.py
from distutils.core import setup, Extension

import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include  = numpy.get_numpy_include()

    
c3 = Extension('_c3',
                ['c3.i'],
                include_dirs = [],
                define_macros =[],
                undef_macros = [],
                language='c',
                library_dirs = ['/home/aagorod/Software/c3/lib'],
                libraries = ['c3']
                )

setup(
      name = "c3",
      version = "1.0",
      ext_modules=[c3]
)
# ~/Software/c3_installed/lib/c3/
