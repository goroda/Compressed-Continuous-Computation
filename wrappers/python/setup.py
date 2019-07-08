# adapted from https://stackoverflow.com/a/48015772

import os
import pathlib

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig

import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include  = numpy.get_numpy_include()

class CMakeExtension(Extension):

    def __init__(self, name):
        "dont invoke the original build_ext for this sepcial extension"
        super().__init__(name, sources=[])

class build_ext(build_ext_orig):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        # config = 'Debug' if self.debug else 'Release'
        # cmake_args = [
        #     '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
        #     '-DCMAKE_BUILD_TYPE=' + config
        # ]

        # example of build args
        build_args = [
            # '--config', config,
            '--', '-j4'
        ]

        os.chdir(str(build_temp))
        # self.spawn(['cmake', str(cwd)] + cmake_args)
        self.spawn(['cmake', str(cwd)])
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        os.chdir(str(cwd))

setup(
    name='spam',
    version='0.1',
    packages=['c3py'],
    ext_modules=[CMakeExtension('../..')],
    cmdclass={
        'build_ext': build_ext
    }
    
)
    
# c3ext = Extension("C3Ext",
#                   sources =["c3swig/c3.i"],
#                   swig_opts=["-py3", "-I../../src"],
#                   include_dirs=["../../src"],
#                   libraries="c3")

# setup(
#       name = "c3",
#       version = "1.0",
#       ext_modules=[c3ext]
# )
# c3 = Extension('_c3',
#                sources=['c3swig/c3_wrap.c']
#                ['c3.i'],
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


# setup(
#     name="c3py",
#     version="1.0",
#     packages=find_packages(),
    
#     py_modules=["c3py.py"],

#     author="Alex Gorodetsky",
#     author_email="alex@alexgorodetsky.com",
#     description="This is the Compressed Continuous Computation library in Python",
#     keywords="highd-approximation machine-learning uncertainty-quantification",
#     url="",
#     project_urls={
#         "Source Code": "https://github.com/goroda/compressed-continuous-computation"
#     },
#     classifiers=[
#         'License:BSD'
#     ]
    
# )


# # pcback = Extension('pycback',
# #                    sources = ['python_caller.c'],
# #                    include_dirs = [
# #                        numpy_include,
# #                    ],
# #                    language='c',
# #                    extra_compile_args=['-std=c99'])


# ~/Software/c3_installed/lib/c3/
