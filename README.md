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

## Installation / Getting started

The dependencies for this code are
   1) BLAS
   2) LAPACK
   3) SWIG (if building non-C interfaces)

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

### Python interface

I have created a partial python interface to the library. This library requires SWIG. To compile and install the python wrappers see the CMake option MAKE_PYTHON_WRAPPERS below. 

The modules will be created in wrappers/python. I have created a FunctionTrain class in the wrappers/python/c3py.py.

To enable proper access to the python library add the following to your environmental variables
``` shell
export C3HOME=~/Software/c3
export PYC3=${C3HOME}/wrappers/python
export PYTHONPATH=$PYTHONPATH:${PYC3}
```

Then, one can run the examples from the root c3 directory as 
``` shell
python wrappers/python/pytest.py
```

## Configuration Options

#### BUILD_STATIC_LIB
Default: `OFF'

Using this option can toggle whether or not static or shared libraries should be built.

**Note: This option cannot be set to ON if building the python wrapper**

#### BUILD_SUB_LIBS
Default: `OFF'

Using this option can toggle whether or not to build each sub-library into its own library

#### MAKE_PYTHON_WRAPPERS
Default: `OFF'

Using this option can toggle whether or not to compile the python wrappers. To specify specific python installations use the CMake options defined [here](https://cmake.org/cmake/help/v3.0/module/FindPythonLibs.html).

After specifying this option, the commands to compile the library and wrappers are
``` shell
make
make PyWrapper
```

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

