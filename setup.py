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
       # ext.runtime_library_dirs = [os.path.join("build", s) for s in os.listdir("build") if os.path.splitext(s)[0].startswith('lib')]
        # ext.runtime_library_dirs = ["@loader_path"]
        
        print(ext.library_dirs)
        print(ext.runtime_library_dirs)
        # exit(1)
        # super().build(ext)
    
    def build_cmake(self, ext):

        ext.extra_compile_args=['-install_name @rpath/libc3.dylib']
        # print(ext.extra_compile_args)        
        # exit(1)
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
                       '-DMAKE_PYTHON_WRAPPERS=ON']

        # example of build args
        build_args = [
            '--config', config,
            '--', '-j4'
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        os.chdir(str(cwd))

pcback = Extension('pycback',
                   sources = ['wrappers/python/callback/python_caller.c'],
                   include_dirs = [
                       numpy_include,
                   ],
                   language='c',
                   extra_compile_args=['-std=c99'])

c3py = Extension("_c3",
                 sources =["wrappers/python/c3swig/c3.i",
                 ],
                 swig_opts=["-py3",
                             "-Ic3"],
                 define_macros =[('COMPILE_WITH_PYTHON',True)],
                 undef_macros = [],
                 language='c',                 
                 # runtime_library_dirs=['build/lib.macosx-10.6-x86_64-3.6/'],
                 # library_dirs=['build/lib.macosx-10.6-x86_64-3.6/'],
                 # runtime_library_dirs=[os.path.join("build", s) for s in os.listdir("build") if os.path.splitext(s)[0].startswith('lib')],
                 # library_dirs=[os.path.join("build", s) for s in os.listdir("build") if os.path.splitext(s)[0].startswith('lib')],
                 # runtime_library_dirs=["@loader_path/"],
                 # library_dirs=['build/lib*'],
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
                 extra_compile_args = ['-std=c99'],#, "-Wl,-rpath=@executable_path"],
                 # extra_link_args =  ["-Wl,-rpath,'$ORIGIN'"],
                 # extra_link_args =  ["-rpath=$ORIGIN"],
                 extra_link_args =  my_link_args, #[-Wl,-rpath,@loader_path"],
                 # extra_link_args =  ["-R$ORIGIN"],
                 libraries=["c3"]
)

setup(name='c3py',
      author="Alex Gorodetsky",
      author_email="alex@alexgorodetsky.com", 
      version='0.1',
      description="Compressed Continuous Computation library in Python",
      keywords="highd-approximation machine-learning uncertainty-quantification tensors",
      package_dir={'': 'wrappers/python/c3py'},
      py_modules=['c3py'],
      ext_modules=[CMakeExtension(name="_c3_notsure"),
                   pcback, c3py],
      cmdclass={
          'build_ext' : build_ext,
      },
      install_requires=[
          'numpy',
      ],
      license='BSD',
      classifiers=[
          "License :: BSD",
          "Natural Language :: English",
          "Programming Language :: C",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3,"
          "Topic :: Approximation"
      ]
)

