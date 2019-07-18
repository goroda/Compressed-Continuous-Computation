# adapted from https://stackoverflow.com/a/48015772
import sys
if sys.platform == 'darwin':
    from distutils import sysconfig
    vars = sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')
    my_link_args=["-Wl,-rpath,@loader_path"]
    my_runtime_dirs = []
else:
    #my_link_args=["-Xlinker","-Wl,-rpath,$ORIGIN"]
    my_link_args=[]#"-Xlinker","-Wl,-rpath,$ORIGIN"]
    my_runtime_dirs=['$ORIGIN/']

import os
import pathlib

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig

import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include  = numpy.get_numpy_include()

PACKAGE_NAME = "c3py"
SOURCE_DIR = pathlib.Path().absolute()

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory,"README.md"), encoding='utf-8') as f:
    long_description = f.read()

class CMakeExtension(Extension):
    """
    An extension to run the cmake build

    This simply overrides the base extension class so that setuptools
    doesnt try to build your sources for you
    """

    def __init__(self, name, sources=[]):
        print("Add Extension ", name)
        super().__init__(name=name, sources=sources)

class build_ext(build_ext_orig):

    def run(self):
        for ext in self.extensions:
            if ext.name == '_c3_notsure':
                self.build_cmake(ext)

            if ext.name == "_c3":
                self.build_swig(ext)
                
        super().run()

    def build_swig(self, ext):
        self.announce("Building Swig", level=3)
        ext.library_dirs = [os.path.join("build", s) for s in os.listdir("build") if os.path.splitext(s)[0].startswith('lib')]
        
        print(ext.library_dirs)
        print(ext.runtime_library_dirs)
        # exit(1)
        # super().build(ext)
    
    def build_cmake(self, ext):

        ext.extra_compile_args=['-install_name @rpath/libc3.dylib']
        # print(ext.extra_compile_args)        
        # exit(1)

        print("Creating Tempory File")
        cwd = pathlib.Path().absolute()
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [ '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
                       '-DCMAKE_BUILD_TYPE=' + config,
                       '-DBUILD_TESTS=OFF',
                       '-DBUILD_EXAMPLES=OFF',
                       '-DBUILD_PROFILING=OFF',
                       '-DBUILD_BENCHMARKS=OFF',                    
                       '-DBUILD_UTILITIES=OFF',
                       # '-DMAKE_PYTHON_WRAPPERS=ON'
        ]

        # example of build args
        build_args = [
            '--config', config,
            '--', '-j4'
        ]

        print("Temporary file = ", build_temp)
        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        os.chdir(str(cwd))

c3py = Extension("_c3",
                 sources =["wrappers/python/c3swig/c3.i",
                 ],
                 swig_opts=["-py3",
                             "-Ic3"],
                 # define_macros =[('COMPILE_WITH_PYTHON',True)],
                 undef_macros = [],
                 language='c',                 
                 runtime_library_dirs= my_runtime_dirs, 
                 # library_dirs=[''],                 
                 include_dirs=["c3/lib_clinalg", 
                               "c3/lib_interface",
                               "c3/lib_funcs",
                               "c3/lib_linalg",
                               "c3/lib_array",
                               "c3/lib_fft",
                               "c3/lib_quadrature",
                               "c3/lib_optimization",
                               "c3/lib_stringmanip",
                               "c3/lib_superlearn",
                               "c3/lib_tensor",                               
                               "c3/lib_probability",
                               numpy_include,
                 ],
                 extra_compile_args = ['-std=c99'],
                 extra_link_args =  my_link_args, 
                 libraries=["c3"]
)

setup(name='c3py',
      author="Alex Gorodetsky",
      author_email="alex@alexgorodetsky.com", 
      version='0.0.5',
      description="Compressed Continuous Computation Library in Python",
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords="highd-approximation machine-learning uncertainty-quantification tensors",
      url="https://github.com/goroda/compressed-continuous-computation",
      package_dir={'': 'wrappers/python/c3py'},
      py_modules=['c3py'],
      ext_modules=[CMakeExtension(name="_c3_notsure"),
                   c3py],
      cmdclass={
          'build_ext' : build_ext,
      },
      install_requires=[
          'numpy', 'pathlib'
      ],
      # license='BSD',
      classifiers=[
          "Natural Language :: English",
          "Programming Language :: C",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3 :: Only",
          "License :: OSI Approved :: BSD License",
          "Topic :: Scientific/Engineering :: Mathematics"
      ]
)

