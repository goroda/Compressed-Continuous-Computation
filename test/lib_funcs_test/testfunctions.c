// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016, Sandia Corporation

// This file is part of the Compressed Continuous Computation (C3) Library
// Author: Alex A. Gorodetsky 
// Contact: goroda@mit.edu

// All rights reserved.

// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice, 
//    this list of conditions and the following disclaimer in the documentation 
//    and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its contributors 
//    may be used to endorse or promote products derived from this software 
//    without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//Code


// first function

#include "testfunctions.h"
#include <assert.h>
#include <math.h>

int Sin3xTx2(size_t N, const double * x, double * out, void * args)
{
    for (size_t ii = 0; ii < N; ii++ ){
        out[ii] = pow(x[ii],2)+1.0*sin(3.0 * x[ii]);
    }
    if (args != NULL){
        int * count = args;
        *count += N;
    }
    return 0;
}

double funcderiv(double x, void * args){
    assert ( args == NULL );
    return 3.0 * cos(3.0 * x) + 2.0 * x;
}

// second function
int powX2(size_t N, const double * x, double * out, void * args)
{
    (void)(args);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = pow(x[ii],2);
    }
    return 0;
}

// third function
int TwoPowX3(size_t N, const double * x, double * out, void * args)
{
    (void)(args);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = 2.0 * pow(x[ii],3.0);
    }
    return 0;
}

// 6th function
int polyroots(size_t N, const double * x, double * out, void * args)
{
    (void)(args);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = (x[ii] - 2.0) * (x[ii] - 1.0) * x[ii] *
                  (x[ii] + 3.0) * (x[ii] - 1.0);
    }
    return 0;
}

// 7th function
int maxminpoly(size_t N, const double * x, double * out, void * args)
{
    (void)(args);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = sin(3.14159 * x[ii]);
    }
    return 0;
}
