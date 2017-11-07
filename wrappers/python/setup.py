# setup.py
from distutils.core import setup, Extension

import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include  = numpy.get_numpy_include()

    
# c3 = Extension('_c3',
#                 ['c3.i'],
#                include_dirs = [
#                    numpy_include,
#                    '../../include',
#                    '../../src/lib_array',
#                    '../../src/lib_clinalg',
#                    '../../src/lib_funcs',
#                    '../../src/lib_optimization',
#                    '../../src/lib_linalg',
#                    '../../src/lib_probability',
#                    '../../src/lib_quadrature',
#                    '../../src/lib_stringmanip',
#                    '../../src/lib_superlearn',
#                    '../../src/lib_interface',
#                ],
#                define_macros =[('COMPILE_WITH_PYTHON',None)],
#                undef_macros = [],
#                language='c',
#                runtime_library_dirs=['../../build/src'],
#                library_dirs = ['../../build/src'],
#                # extra_link_args=['-Wl,-R/Users/aagorod/Software/c3/build/src'],
#                # library_dirs = ['/Users/aagorod/Software/c3/lib'],
#                libraries = ['c3'],
#                extra_compile_args = ['-std=c99'],
# )

pcback = Extension('pycback',
                   sources = ['python_caller.c'],
                   include_dirs = [
                       numpy_include,
                   ],
                   language='c',
                   extra_compile_args=['-std=c99'])


setup(
      name = "c3",
      version = "1.0",
      # ext_modules=[c3,pcback]
      ext_modules=[pcback]
)

# ~/Software/c3_installed/lib/c3/
