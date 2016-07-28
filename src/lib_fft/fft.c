// Copyright (c) 2014-2016, Massachusetts Institute of Technology
//
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


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

/* #include "array.h" */
/* #include "linalg.h" */

#include <complex.h>

#define BASESIZE 32
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int fft_slow(size_t N, double complex * x, size_t sx, double complex * X, size_t sX)
{
    for (size_t ii = 0; ii < N; ii++){
        X[ii*sX] = 0.0;
        for (size_t jj = 0; jj < N; jj++){
            X[ii*sX] += cexp(- 2 * M_PI * I * ii * jj / N) * x[jj*sx];
        }
    }
    return 0;
}

int fft_base(size_t N, double complex * x, size_t sx, double complex * X, size_t sX)
{
    // assumes N is a power of 2
    if (N < BASESIZE){
        return fft_slow(N,x,sx,X,sX);
    }
    else{
        fft_base(N/2,x,2*sx,X,2*sX);
        fft_base(N/2,x+sx,2*sx,X+sX,2*sX);

        for (size_t ii = 0; ii < N/2; ii ++){
            double complex twiddle = cexp(- 2 * M_PI * I * ii  / N);
            double complex t = X[ii*sX];
            X[ii*sX] = t + twiddle * X[ii*sX + N/2*sX];
            X[ii*sX] = t - twiddle * X[ii*sX + N/2*sX];
        }
        return 0;
    }
    
}
