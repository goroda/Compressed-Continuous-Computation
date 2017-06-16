from distutils.core import setup, Extension

module = Extension('pycback',sources = ['python_caller.c'], extra_compile_args=['-std=c99'])

setup(name='pycback', version='1.0', ext_modules=[module])
