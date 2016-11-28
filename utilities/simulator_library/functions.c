// This file is part of the Compressed Continuous Computation (C3) toolbox
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

// Rosenbrock function

#include <math.h>

#include "functions.h"

void rosen_brock_func(size_t N, const double * x, double * out, void * arg)
{
    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        double f1 = 10.0 * (x[ii*2+1] - pow(x[ii*2+0],2));
        double f2 = (1.0 - x[ii*2+0]);
        out[ii] = pow(f1,2) + pow(f2,2);
    }
}

void sin_sum2d(size_t N, const double * x, double * out, void * arg)
{

    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = sin(x[ii*2+0] + x[ii*2+1]);
    }
}

void sin_sum5d(size_t N, const double * x, double * out, void * arg)
{

    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        double temp = 0.0;
        for (size_t jj = 0; jj < 5; jj++){
            temp += x[ii*5+jj];
        }
        out[ii] = sin(temp);
    }
}

void sin_sum10d(size_t N, const double * x, double * out, void * arg)
{

    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        double temp = 0.0;
        for (size_t jj = 0; jj < 10; jj++){
            temp += x[ii*10+jj];
        }
        out[ii] = sin(temp);
    }
}

// other functionsx
size_t function_get_dim(void * arg)
{
    struct Function * p = arg;
    return p->dim;
}

void function_eval(size_t N, const double * x, double * out, void * arg)
{
    struct Function * p = arg;
    p->eval(N,x,out,arg);
}


struct Function funcs[34];
size_t num_funcs;
void create_functions()
{
    num_funcs = 4;
    funcs[0].dim = 2;
    funcs[0].eval = rosen_brock_func;

    funcs[1].dim = 2;
    funcs[1].eval = sin_sum2d;

    funcs[2].dim = 5;
    funcs[2].eval = sin_sum5d;

    funcs[3].dim = 10;
    funcs[3].eval = sin_sum10d;
}
