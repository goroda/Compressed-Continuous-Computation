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
  1) Expectations and Variances
  2) Sobol sensitivities

In addition to the above capabilities, which are unique to the C3 package, I also have general optimization routines including
* BFGS
* LBFGS
* Gradient descent
* Stochastic Gradient with ADAM 


For more details see the website at 

http://www.alexgorodetsky.com/c3/html/

## Installation / Getting started

The dependencies for this code are
   1) BLAS
   2) LAPACK
   3) SWIG (if building non-C interfaces)

```
git clone https://github.com/goroda/Compressed-Continuous-Computation.git c3
cd c3
mkdir build
cd build
cmake ..
make
```

This will install all shared libraries into c3/lib. The main shared library is libc3, the rest are all submodules. To install to a particular location use

```
cmake .. -DCMAKE_INSTALL_PREFIX=/your/choice
make install
```

### Python interface

I have created a partial python interface to the library. This library requires SWIG. To compile and install the python wrappers see the CMake option MAKE_PYTHON_WRAPPERS below. 

The modules will be created in wrappers/python. I have created a FunctionTrain class in the wrappers/python/c3py.py.

To run an example in python first make sure that the c3 library is on your path. For example, do
```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:<path_to_c3_lib>
```
on a Linux system, or 
```
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:<path_to_c3_lib>
```
on Mac OS X. For example, if the library is installed in a default location the path would be  /path_to_c3/lib.

Then, one can run the examples by
```
cd wrappers/python
python pytest.py

```

## Configuration Options

#### BUILD_STATIC_LIB
Default: `OFF'

Using this option can toggle whether or not static or shared libraries should be built.

** Note: This option cannot be set to ON if building the python wrapper **

#### BUILD_SUB_LIBS
Default: `OFF'

Using this option can toggle whether or not to build each sub-library into its own library

#### MAKE_PYTHON_WRAPPERS
Default: `OFF'

Using this option can toggle whether or not to compile the python wrappers. To specify specific python installations use the CMake options defined [here](https://cmake.org/cmake/help/v3.0/module/FindPythonLibs.html).

After specifying this option one can build the python modules using

```
make
make PyWrapper
```

## Systems I have tested on

Mac OS X with clang version 8.0
Ubuntu with gcc version 5.0

## Coding practices

I aim to document (with Doxygen) every function available to the user and provide a unit test. Furthermore, I won't push code to the master branch that has memory leaks. I am constantly looking for more suggestions for improving the robustness of the code if any issues are encountered. 

## Contributing

Please open a Github issue to ask a question, report a bug, or to request features.
To contribute, fork the repository and setup a branch.

Author: Alex A. Gorodetsky

Contact: goroda@mit.edu

Copyright (c) 2014-2016, Massachusetts Institute of Technology

Copyright (c) 2016-2017, Sandia National Laboratories

License: BSD

