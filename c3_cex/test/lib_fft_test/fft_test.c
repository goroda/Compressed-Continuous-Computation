// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016-2017 Sandia Corporation

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
#include <float.h>
#include <complex.h>
#include <time.h>
#include "CuTest.h"

#include "array.h"
#include "fft.h"

void Test_fft_slow(CuTest * tc){

    printf("Testing function: fft_slow\n");
    complex double x[] = {1, 1, 1, 1, 0, 0, 0, 0};
    complex double X[8];
    size_t N = 8;
    int res = fft_slow(N,x,1,X,1);
    CuAssertIntEquals(tc,0,res);

    CuAssertDblEquals(tc,4.0,creal(X[0]),1e-15);
    CuAssertDblEquals(tc,0.0,cimag(X[0]),1e-15);
    
    CuAssertDblEquals(tc,1.0,creal(X[1]),1e-15);
    CuAssertDblEquals(tc,-2.414142,cimag(X[1]),1e-4);

    CuAssertDblEquals(tc,0.0,creal(X[2]),1e-15);
    CuAssertDblEquals(tc,0.0,cimag(X[2]),1e-15);

    CuAssertDblEquals(tc,1.0,creal(X[3]),1e-15);
    CuAssertDblEquals(tc,-0.414214,cimag(X[3]),1e-4);

    CuAssertDblEquals(tc,0.0,creal(X[4]),1e-15);
    CuAssertDblEquals(tc,0.0,cimag(X[4]),1e-15);

    CuAssertDblEquals(tc,1.0,creal(X[5]),1e-15);
    CuAssertDblEquals(tc,0.414214,cimag(X[5]),1e-4);

    CuAssertDblEquals(tc,0.0,creal(X[6]),1e-15);
    CuAssertDblEquals(tc,0.0,cimag(X[6]),1e-15);

    CuAssertDblEquals(tc,1.0,creal(X[7]),1e-15);
    CuAssertDblEquals(tc,2.414214,cimag(X[7]),1e-4);

    complex double xi[8];
    res = ifft_slow(N,X,1,xi,1);
    CuAssertIntEquals(tc,0,res);
    for (size_t ii = 0; ii < N; ii++){
        CuAssertDblEquals(tc,creal(x[ii]),creal(xi[ii]),1e-14);
        CuAssertDblEquals(tc,0.0,cimag(xi[ii]),1e-15);
    }
}

void Test_fft_base(CuTest * tc){

    printf("Testing function: fft_base\n");
    complex double x[] = {1, 1, 1, 1, 0, 0, 0, 0};
    complex double X[8];
    size_t N = 8;
    int res = fft_base(N,x,1,X,1);
    CuAssertIntEquals(tc,0,res);

    /* printf("------------\n"); */
    /* for (size_t ii = 0; ii < N; ii++){ */
    /*     printf("x[%zu] = (%G,%G)\n",ii,creal(X[ii]),cimag(X[ii])); */
    /* } */

    
    CuAssertDblEquals(tc,4.0,creal(X[0]),1e-15);
    CuAssertDblEquals(tc,0.0,cimag(X[0]),1e-15);
    
    CuAssertDblEquals(tc,1.0,creal(X[1]),1e-15);
    CuAssertDblEquals(tc,-2.414142,cimag(X[1]),1e-4);

    CuAssertDblEquals(tc,0.0,creal(X[2]),1e-15);
    CuAssertDblEquals(tc,0.0,cimag(X[2]),1e-15);

    CuAssertDblEquals(tc,1.0,creal(X[3]),1e-15);
    CuAssertDblEquals(tc,-0.414214,cimag(X[3]),1e-4);

    CuAssertDblEquals(tc,0.0,creal(X[4]),1e-15);
    CuAssertDblEquals(tc,0.0,cimag(X[4]),1e-15);

    CuAssertDblEquals(tc,1.0,creal(X[5]),1e-15);
    CuAssertDblEquals(tc,0.414214,cimag(X[5]),1e-4);

    CuAssertDblEquals(tc,0.0,creal(X[6]),1e-15);
    CuAssertDblEquals(tc,0.0,cimag(X[6]),1e-15);

    CuAssertDblEquals(tc,1.0,creal(X[7]),1e-15);
    CuAssertDblEquals(tc,2.414214,cimag(X[7]),1e-4);

    complex double xi[8];
    res = ifft_base(N,X,1,xi,1);
    CuAssertIntEquals(tc,0,res);
    
    /* printf("------------\n"); */
    /* for (size_t ii = 0; ii < N; ii++){ */
    /*     printf("x[%zu] = (%G,%G)\n",ii,creal(xi[ii]),cimag(xi[ii])); */
    /* } */
    for (size_t ii = 0; ii < N; ii++){
        CuAssertDblEquals(tc,creal(x[ii]),creal(xi[ii]),1e-14);
        CuAssertDblEquals(tc,0.0,cimag(xi[ii]),1e-15);
    }
}

void Test_fft(CuTest * tc){

    printf("Testing function: fft (1)\n");
    complex double x[] = {1, 1, 1, 1, 0, 0, 0, 0, 0.0};
    complex double X[9];
    size_t N = 9;
    int res = fft(N,x,1,X,1);
    CuAssertIntEquals(tc,1,res);

}

void Test_fft2(CuTest * tc){

    printf("Testing function: fft (2)\n");
    complex double x[] = {1, 1, 1, 1, 0, 0, 
                          0, 0};
    complex double X[8];
    size_t N = 8;
    int res = fft(N,x,1,X,1);
    CuAssertIntEquals(tc,0,res);

    complex double xi[8];
    res = ifft(N,X,1,xi,1);

    CuAssertIntEquals(tc,0,res);
    for (size_t ii = 0; ii < N; ii++){
        CuAssertDblEquals(tc,creal(x[ii]),creal(xi[ii]),1e-14);
        CuAssertDblEquals(tc,0.0,cimag(xi[ii]),1e-14);
    }
}

void Test_fft_base_big(CuTest * tc){

    printf("Testing function: fft for timing purposes \n");

    // 2^20

    /* FILE * fp = fopen("fft_scaling.txt","w+"); */
    /* assert (fp != NULL); */
    size_t N = 1;
    for (size_t ll = 0; ll < 18; ll++){
    /* for (size_t ll = 0; ll < 1; ll++){ */
    /* size_t N = 1024*16; */
        N = N * 2;
        printf("ll = %zu, N = %zu\n", ll, N);

        double complex *x = malloc(N * sizeof(double complex));
        double complex *X = malloc(N * sizeof(double complex));
        
        size_t nrand = 100;
        clock_t tic = clock();
        for (size_t jj = 0; jj < nrand; jj++){
            for (size_t ii = 0; ii < N; ii++){
                x[ii] = randn();
            }
            int res = fft(N, x, 1, X, 1);
            /* int res = fft_base(N,x,1,X,1); */
            CuAssertIntEquals(tc,0,res);
        }
        clock_t toc = clock();
        double avg = (double)(toc - tic) / CLOCKS_PER_SEC / (double) nrand;
        printf("Average (over %zu) time to perform size %zu DFT is %G seconds \n",nrand,N,avg);
        printf("N = %zu, time = %G\n",N,avg);
        /* fprintf(fp,"%zu %G\n",N,avg); */
        free(x); x = NULL;
        free(X); X = NULL;
    }
    /* fclose(fp); */
}

void Test_fft_ifft(CuTest * tc){

    printf("Testing function: fft and ifft \n");

    // 2^20

    size_t N = 1;
    for (size_t ll = 0; ll < 10; ll++){
        N = N * 2;
        /* printf("ll = %zu, N = %zu\n", ll, N); */

        double complex *x = malloc(N * sizeof(double complex));
        double complex *xout = malloc(N * sizeof(double complex));
        double complex *X = malloc(N * sizeof(double complex));
        
        size_t nrand = 100;
        for (size_t jj = 0; jj < nrand; jj++){
            for (size_t ii = 0; ii < N; ii++){
                x[ii] = randn();
            }
            int res = fft(N, x, 1, X, 1);
            /* int res = fft_base(N,x,1,X,1); */
            CuAssertIntEquals(tc,0,res);
            res = ifft(N, X, 1, xout, 1);

            for (size_t ii = 0; ii < N; ii++){
                double diff = cabs(xout[ii] - x[ii]);
                /* printf("diff = %3.5G\n", diff); */
                CuAssertDblEquals(tc, 0.0, diff, 1e-14);
            }
                
        }

        free(x); x = NULL;
        free(xout); xout = NULL;
        free(X); X = NULL;
    }
}

void Test_cheb_vals_to_coeff(CuTest * tc){

    printf("Testing function: vals_to_coeff \n");

    double vals[3] = {6.0, 2.0, 2.0};
    size_t nvals = 3;
    double coeff[3];
    
    int res = cheb_vals_to_coeff(nvals,vals,coeff);
    CuAssertIntEquals(tc,0,res);
    
    /* dprint(nvals,coeff); */

}

CuSuite * FFTGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_fft_slow);
    SUITE_ADD_TEST(suite, Test_fft_base);
    SUITE_ADD_TEST(suite, Test_fft);
    SUITE_ADD_TEST(suite, Test_fft2);
    SUITE_ADD_TEST(suite, Test_fft_base_big);
    SUITE_ADD_TEST(suite, Test_fft_ifft);
 
    /* SUITE_ADD_TEST(suite, Test_cheb_vals_to_coeff); */

    return suite;
}
