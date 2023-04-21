// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016-2017 Sandia Corporation
// Copyright (c) 2017 NTESS, LLC.

// This file is part of the Compressed Continuous Computation (C3) Library
// Author: Alex A. Gorodetsky 
// Contact: alex@alexgorodetsky.com

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
#include <assert.h>
#include <complex.h>

#define BASESIZE 2
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/***********************************************************//**
    Compute DFT
***************************************************************/
int fft_slow(size_t N, const double complex * xin, size_t sx, 
             double complex * xout, size_t sX)
{
    for (size_t ii = 0; ii < N; ii++){
        xout[ii*sX] = 0.0;
        for (size_t jj = 0; jj < N; jj++){
            xout[ii*sX] += cexp(- 2 * M_PI * (double complex)I * ii * jj / N)
                                    * xin[jj*sx];
        }
    }
    return 0;
}

/***********************************************************//**
    Compute IDFT
***************************************************************/
int ifft_slow(size_t N, const double complex * xin, size_t sx, 
              double complex * xout, size_t sX)
{
    for (size_t ii = 0; ii < N; ii++){
        xout[ii*sX] = 0.0;
        for (size_t jj = 0; jj < N; jj++){
            xout[ii*sX] += cexp( 2 * M_PI * (double complex)I * ii * jj / N)
                                     * xin[jj*sx];
        }
        xout[ii*sX] /= (double)N;
    }
    return 0;
}




/***********************************************************//**
   FFT                                                            
***************************************************************/
int fft_base(size_t N, const double complex * x, size_t sx, 
             double complex * X, size_t sX)
{
    // assumes N is a power of 2
    if (N <= BASESIZE){
        int res = fft_slow(N,x,sx,X,sX);
        return res;
    }
    else{
        fft_base(N/2,x,2*sx,X,2*sX);
        fft_base(N/2,x+sx,2*sx,X+sX,2*sX);

        double complex * e = malloc(N/2 * sizeof(double complex));
        double complex * o = malloc(N/2 * sizeof(double complex));

        for (size_t ii = 0; ii < N/2; ii++){
            e[ii] = X[ii*2*sX];
            o[ii] = X[ii*2*sX+sX];
        }

        for (size_t ii = 0; ii < N/2; ii ++){
            double complex twiddle = cexp(- 2 * M_PI * (double complex)I * ii  / N);
            /* printf("ii=%zu, twiddle = (%G,%G)\n",ii,creal(twiddle),cimag(twiddle)); */
            
            /* printf("\t ind = %zu, %zu\n",ii*2*sX, ii*2*sX+sX); */
            /* printf("\te = (%G,%G), o = (%G,%G)\n", creal(e),cimag(e), creal(o),cimag(o)); */
            X[ii*sX] = e[ii] + twiddle * o[ii];
            X[ii*sX + N/2*sX] = e[ii] - twiddle * o[ii];
        }
        free(e); e = NULL;
        free(o); o = NULL;

        return 0;
    }
    
}

/***********************************************************//**
   IFFT                                                            
***************************************************************/
int ifft_base(size_t N, double complex * x, size_t sx, 
             double complex * X, size_t sX)
{
    for (size_t ii = 0; ii < N; ii++){
        x[ii] = conj(x[ii*sX]);
    }
    int res = fft_base(N,x,sx,X,sX);
    for (size_t ii = 0; ii < N; ii++){
        x[ii*sX] = conj(x[ii*sX]);
        X[ii*sX] = conj(X[ii*sX]);
        X[ii*sX] /= (double)N;
    }
    return res;
}

/***********************************************************//**
   Check if a number is power of two.

   Algorithm #10 from FFT   
   http://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/ ***************************************************************/
static int isPowerOfTwo (size_t x)
{
  return ((x != 0) && ((x & (~x + 1)) == x));
}

int is2(size_t x)
{
    while (((x % 2) == 0) && x > 1) 
        x /= 2;
    return (x == 1);
}

int fft(size_t N, const double complex * x, size_t sx, 
        double complex * X, size_t sX)
{
    assert (sx == 1);
    assert (sX == 1);
    int res = 1;
    if (isPowerOfTwo(N)){
    /* if (is2(N)){ */
        /* return 1; */
        res = fft_base(N,x,sx,X,sX);
    }
    else{
        fprintf(stderr, "fft_base can only be called with power of 2, not %zu\n",N);
        /* printf("is2(N) = %d\n", isPowerOfTwo(N)); */
    }
    return res;
}

int ifft(size_t N, double complex * x, size_t sx, 
        double complex * X, size_t sX)
{
    assert (sx == 1);
    assert (sX == 1);
    int res = 1;
    if (isPowerOfTwo(N)){
        /* return 1; */
        res = ifft_base(N,x,sx,X,sX);
    }
    else{
        fprintf(stderr, "fft_base can only be called with power of 2\n");
    }
        
    return res;
}


int cheb_vals_to_coeff(size_t nvals, const double * vals, double * coeff)
{
    
    size_t Nin = nvals-1;
    /* if (is2(Nin) == 0){ */
    if (!isPowerOfTwo(Nin)){        
        fprintf(stderr,"Non-power of 2 cheb_coeff_to_vals is not yet implemented\n");
        return 1;
    }
    double complex * xin = malloc(2*Nin * sizeof(double complex));
    double complex * xout = malloc(2*Nin * sizeof(double complex));
    xin[0] = vals[0];
    xout[0] = 0.0;
    xout[Nin] = 0.0;
    for (size_t ii = 1; ii < Nin; ii++){
        /* printf("ii=%zu 2*Nin-ii=%zu\n",ii,2*Nin-ii); */
        xin[ii] = vals[ii];
        xin[2*Nin-ii] = vals[ii];

        xout[ii] = 0.0;
        xout[ii+Nin] = 0.0;
    }
    xin[nvals-1] = vals[nvals-1];
    
    int res = fft_base(2*Nin,xin,1,xout,1);
    
    /* for (size_t ii = 0; ii < 2*Nin; ii++){ */
    /*     printf("X[%zu]=%G \n",ii,creal(xout[ii])); */
    /* } */
    for (size_t ii = 0; ii < nvals; ii++){
        coeff[ii] = creal(xout[nvals-1-ii]) / (double) Nin;
        
    }
    coeff[0] /= 2.0;
    coeff[nvals-1] /= 2.0;

    free(xin); xin = NULL;
    free(xout); xout = NULL;

    return res;
}
