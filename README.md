# Compressed Continuous Computation (C3)
> Computing with functions

The Compressed Continuous Computation (C3) package is intended to make it easy to perform continuous linear and multilinear algebra with multidimensional functions. It works by representing multidimensional functions in a low-rank format. Common tasks include taking "matrix" decompositions of vector- or matrix-valued functions, adding or multiplying functions together, integrating multidimensional functions, and much much more. The following is a sampling of capabilities
* Adaptive approximation of a black-box model (specified as a function pointer)
* Regression of a model from data
* Both linear and nonlinear approximation 
* Approximation in polynomial, piecewise polynomial, linear element, and radial basis function spaces
* General adaptive integration schemes 
* Differentiation
* Multiplication 
* Addition
* Rounding 
* Computing Jacobians and Hessians
* UQ
  1) Expectation and variance
  2) Sobol sensitivities

In addition to the above capabilities, which are unique to the C3 package, I also have general optimization routines including
* BFGS
* LBFGS
* Gradient descent
* Stochastic Gradient with ADAM 


Documentation of most functions is provided by Doxygen here: 
http://alexgorodetsky.com/c3doc/html/

### Installation / Getting started

The dependencies for this code are
   1) BLAS
   2) LAPACK
   3) SWIG (if building non-C interfaces)

## From Source
```shell
git clone https://github.com/goroda/Compressed-Continuous-Computation.git c3
cd c3
mkdir build
cd build
cmake ..
make
```

This will install all shared libraries into c3/build/src. The main shared library is libc3, the rest are all submodules. To install to a particular location use

``` shell
cmake .. -DCMAKE_INSTALL_PREFIX=/your/choice
make install
```

## Python interface

You can install the python interface using the pip utility through

``` shell
pip install c3py
```

One can obtain some examples in the pyexamples subdirectory
``` shell
python pywrappers/pytest.py
```

## Configuration Options

#### BUILD_STATIC_LIB
Default: `OFF'

Using this option can toggle whether or not static or shared libraries should be built.

**Note: This option cannot be set to ON if building the python wrapper**

#### BUILD_SUB_LIBS
Default: `OFF'

Using this option can toggle whether or not to build each sub-library into its own library

#### BUILD_TESTS
Default: `OFF'

Using this option can toggle whether or not to build unit tests

#### BUILD_EXAMPLES
Default: `OFF'

Using this option can toggle whether or not to compile exampels

#### BUILD_PROFILING
Default: `OFF'

Using this option can toggle whether or not to compile the profiling executables

#### BUILD_BENCHMARKS
Default: `OFF'

Using this option can toggle whether or not to compile the benchmarks tests

#### BUILD_UTILITIES
Default: `OFF'

Using this option can toggle whether or not to compile the utilities


## Systems I have tested on

1) Mac OS X with clang version 8.0  
2) Ubuntu with gcc version 5.0


## Solutions to some possible problems

### Error: Unable to find 'python.swg'

On Mac OS X, if SWIG is installed with macports using
```shell
sudo port install swig
```
then the above error might occur. To remedy this error install the swig-python package
```shell
sudo port install swig-python
```

### Numpy errors

Sometimes you may see the following errors

``` shell
_frozen_importlib:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 216, got 192
```

or

``` shell
RuntimeError: The current Numpy installation ('/Users/alex/anaconda3/envs/pytorch/lib/python3.6/site-packages/numpy/__init__.py') fails to pass simple sanity checks. This can be caused for example by incorrect BLAS library being linked in, or by mixing package managers (pip, conda, apt, ...). Search closed numpy issues for similar problems.
```

One way that I have found (https://stackoverflow.com/a/47975375) that seems to solve this is to upgrade numpy by running the following command. I am really not sure why this works ...

``` shell
sudo pip install numpy --upgrade --ignore-installed
```

## Coding practices

I aim to document (with Doxygen) every function available to the user and provide a unit test. Furthermore, I won't push code to the master branch that has memory leaks. I am constantly looking for more suggestions for improving the robustness of the code if any issues are encountered. 

## Contributing

Please open a Github issue to ask a question, report a bug, or to request features.
To contribute, fork the repository and setup a branch.

Author: [Alex A. Gorodetsky](https://www.alexgorodetsky.com)  
Contact: [goroda@umich.edu](mailto:goroda@umich.edu)  
Copyright (c) 2014-2016, Massachusetts Institute of Technology  
Copyright (c) 2016-2017, Sandia National Laboratories  
Copyright (c) 2018, University of Michigan  
License: BSD

