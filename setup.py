# adapted from https://stackoverflow.com/a/48015772
import sys
if sys.platform == 'darwin':
    from distutils import sysconfig
    vars = sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')
    
import os
import pathlib

# from distutils.command.install_data import install_data
# from setuptools import find_packages, setup, Extension
# from setuptools.command.build_ext import build_ext
# from setuptools.command.install_lib import install_lib
# from setuptools.command.install_scripts import install_scripts
# import struct

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


# class InstallCMakeLibsData(install_data):
#     """
#     A wrapper to get the install data into the egg-info

#     Listing the installed files in the egg-info guarantees that
#     all of the package files will be uninstalled when the user uninstalls
#     your package through pip
#     """
#     def run(self):
#         """
#         Outfiles are the libraries that were built using cmake
#         """

#         self.outfiles = self.distribution.data_files

# class InstallCMakeLibs(install_lib):
#     """
#     Get the libraries from the parent distribution, use those as the outfiles

#     Skip building anything; everything is already build, forward libraries to
#     the installation step
#     """

#     def run(self):
#         """
#         Copy libraries from the bin directory and place them as appropriate
#         """

#         self.announce("Moving library files", level=3)

#         # We have already built the libraries in the previous build_ext step

#         self.skip_build = True

#         bin_dir = self.distribution.bin_dir

#         # Depending on the files that are generated from your cmake
#         # build chain, you may need to change the below code, such that
#         # your files are moved to the appropriate location when the installation is run

#         libs = [os.path.join(bin_dir, _lib) for _lib in
#                 os.listdir(bin_dir) if
#                 os.path.isfile(os.path.join(bin_dir, _lib)) and
#                 os.path.splitext(_lib)[1] in [".dll", ".so"]
#                 and not (_lib.startswith("python") or _lib.startswith(PACKAGE_NAME))]

#         for lib in libs:
#             shutil.move(lib, os.path.join(self.build_dir,
#                                           os.path.basename(lib)))
    

#         self.distribution.data_files = [os.path.join(self.install_dir, os.path.basename(lib))
#                                         for lib in libs]

#         # Must be forced to run after adding the libs to data_files
#         self.distribution.run_command("install_data")
#         super().run()

# class InstallCMakeScripts(install_scripts):
#     """
#     Install the scripts in the build dir
#     """

#     def run(self):
#         """
#         Copy the required directory to the build directory and super().run()
#         """

#         self.announce("Moving scripts files", level=3)

#         # Scripts were already built in a previous step

#         self.skip_build = True

#         bin_dir = self.distribution.bin_dir

#         scripts_dirs = [os.path.join(bin_dir, _dir) for _dir in
#                         os.listdir(bin_dir) if
#                         os.path.isdir(os.path.join(bin_dir, _dir))]

#         for scripts_dir in scripts_dirs:

#             shutil.move(scripts_dir,
#                         os.path.join(self.build_dir,
#                                      os.path.basename(scripts_dir)))

#         # Mark the scripts for installation, adding them to 
#         # distribution.scripts seems to ensure that the setuptools' record 
#         # writer appends them to installed-files.txt in the package's egg-info

#         self.distribution.scripts = scripts_dirs

#         super().run()

# class BuildCMakeExt(build_ext):
#     """
#     Builds using cmake instead of the python setuptools implicit build
#     """

#     def run(self):
#         """
#         Perform build_cmake before doing the 'normal' stuff
#         """

#         for extension in self.extensions:

#             if extension.name == '_c3':

#                 self.build_cmake(extension)

#         super().run()

#     def build_cmake(self, extension):
#         """
#         The steps required to build the extension
#         """

#         self.announce("Preparing the build environment", level=3)

#         build_dir = pathlib.Path(self.build_temp)

#         extension_path = pathlib.Path(self.get_ext_fullpath(extension.name))

#         os.makedirs(build_dir, exist_ok=True)
#         os.makedirs(extension_path.parent.absolute(), exist_ok=True)

#         # Now that the necessary directories are created, build

#         self.announce("Configuring cmake project", level=3)

#         # Change your cmake arguments below as necessary
#         # Below is just an example set of arguments for building Blender as a Python module

        
#         self.spawn(['cmake', '-H'+str(SOURCE_DIR),
#                     '-B'+self.build_temp,
#                     '-DBUILD_TESTS=OFF',
#                     '-DBUILD_EXAMPLES=OFF',
#                     '-DBUILD_PROFILING=OFF',
#                     '-DBUILD_BENCHMARKS=OFF',                    
#                     '-DBUILD_UTILITIES=OFF',
#                     '-DMAKE_PYTHON_WRAPPERS=ON'
#         ])
                    

#         self.announce("Building binaries", level=3)

        # self.spawn(["cmake", "--build", self.build_temp, "--target", "c3", '--','-j4'])
                    # "--config", "Release"])

        # Build finished, now copy the files into the copy directory
        # The copy directory is the parent directory of the extension (.pyd)

        # self.announce("Moving built python module", level=3)

        # bin_dir = os.path.join(build_dir, 'bin', 'Release')
        # self.distribution.bin_dir = bin_dir

        # pyd_path = [os.path.join(bin_dir, _pyd) for _pyd in
        #             os.listdir(bin_dir) if
        #             os.path.isfile(os.path.join(bin_dir, _pyd)) and
        #             os.path.splitext(_pyd)[0].startswith(PACKAGE_NAME) and
        #             os.path.splitext(_pyd)[1] in [".pyd", ".so"]][0]

        # shutil.move(pyd_path, extension_path)

        # After build_ext is run, the following commands will run:
        # 
        # install_lib
        # install_scripts
        # 
        # These commands are subclassed above to avoid pitfalls that
        # setuptools tries to impose when installing these, as it usually
        # wants to build those libs and scripts as well or move them to a
        # different place. See comments above for additional information
        
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
                           # "c3/lib_array/array.c",
                           # "c3/lib_superlearn/learning_options.c",
                           # "c3/lib_superlearn/objective_functions.c",
                           # "c3/lib_superlearn/parameterization.c",
                           # "c3/lib_superlearn/regress.c",
                           # "c3/lib_superlearn/superlearn.c",
                           # "c3/lib_superlearn/superlearn_util.c",
                           # "c3/lib_interface/approximate.c",
                           # "c3/lib_clinalg/diffusion.c",
                           # "c3/lib_clinalg/dmrg.c",
                           # "c3/lib_clinalg/dmrgprod.c",
                           # "c3/lib_clinalg/ft.c",
                           # "c3/lib_clinalg/indmanage.c",
                           # "c3/lib_clinalg/iterative.c",
                           # "c3/lib_clinalg/qmarray.c",
                           # "c3/lib_clinalg/quasimatrix.c",
                           # "c3/lib_fft/fft.c",
                           # "c3/lib_funcs/fapprox.c",
                           # "c3/lib_funcs/functions.c",
                           # "c3/lib_funcs/fwrap.c",
                           # "c3/lib_funcs/hpoly.c",
                           # "c3/lib_funcs/fourier.c",        
                           # "c3/lib_funcs/constelm.c",
                           # "c3/lib_funcs/kernels.c",
                           # "c3/lib_funcs/monitoring.c",
                           # "c3/lib_funcs/piecewisepoly.c",
                           # "c3/lib_funcs/pivoting.c",
                           # "c3/lib_funcs/polynomials.c",
                           # "c3/lib_funcs/space.c",
                           # "c3/lib_linalg/linalg.c",
                           # "c3/lib_linalg/matrix_util.c",
                           # "c3/lib_optimization/optimization.c",
                           # "c3/lib_quadrature/legquadrules2.c",
                           # "c3/lib_quadrature/legquadrules.c",
                           # "c3/lib_quadrature/quadrature.c",
                           # "c3/lib_stringmanip/stringmanip.c",
                           # "c3/lib_tensdecomp/candecomp.c",
                           # "c3/lib_tensdecomp/convert.c",
                           # "c3/lib_tensdecomp/cross.c",
                           # "c3/lib_tensdecomp/tensortrain.c",
                           # "c3/lib_tensdecomp/tt_integrate.c",
                           # "c3/lib_tensdecomp/tt_multilinalg.c",
                           # "c3/lib_tensor/tensor.c",
                           # "c3/lib_probability/probability.c",
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
                 # runtime_library_dirs=['$ORIGIN/'],
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
                 extra_link_args =  ["-Wl,-rpath,@loader_path"],
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

