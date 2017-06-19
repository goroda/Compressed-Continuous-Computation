from distutils.core import setup, Extension

import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include  = numpy.get_numpy_include()

    
module = Extension('pycback',
                   sources = ['python_caller.c'],
                   include_dirs = [
                       numpy_include,
                       '../../../src/lib_funcs'
                   ],
                   library_dirs = ['../../lib'],
                   libraries=['c3'],
                   language='c',
                   extra_compile_args=['-std=c99'])

setup(name='pycback', version='1.0', ext_modules=[module])
