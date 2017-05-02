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

In addition to the above capabilities, which are unique to the C3 package, I also have general optimization routines including
* BFGS
* LBFGS
* Gradient descent


For more details see the website at 

http://www.alexgorodetsky.com/c3/html/


## Installation / Getting started

The dependencies for this code are
   1) BLAS
   2) LAPACK

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

I have created a simple python interface to the library. It has an interface to some simple operations. This library requires SWIG and the following commands 

```
cd wrappers/python     # changes directory to the swig interface file
./run.sh               # uses SWIG to create a python interface 
python pytest.py       # run exmaple simple script that performs some operations
```

I have created a FunctionTrain class in the wrappers/python/c3py.py.

Note: run.sh uses Python 3. If you need 2.7 then remove the -py3 flag in the script.

## Configuration Options

#### BUILD_STATIC_LIB
Default: `OFF'

Using this option can toggle whether or not static or shared libraries should be built



## Systems I have tested on

Mac OS X with clang
Ubuntu with gcc

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

