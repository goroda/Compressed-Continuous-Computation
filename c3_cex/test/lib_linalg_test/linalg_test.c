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







#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "array.h"
#include "linalg.h"

#include "CuTest.h"

void Test_qr(CuTest * tc)
{
    printf("Testing Function: qr\n");
    size_t N = 7;
    size_t M = 5;
    double m[35] = { 0.41, 0.03, 0.73, 0.21, 0.52,
                     0.09, 0.64, 0.72, 0.61, 0.75,
                     0.98, 0.30, 0.67, 0.72, 0.42,
                     0.58, 0.19, 0.97, 0.25, 0.14,
                     0.68, 0.54, 0.87, 0.74, 0.78,
                     0.83, 0.88, 0.56, 0.59, 0.36,
                     0.60, 0.19, 0.71, 0.78, 0.25 };
    
    qr(N,M,m, N);
    
    double out[25];

    double m2[35];
    memmove(m2, m, 35 * sizeof(double));
    cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans, M, M, N, 1.0, m, N, m2, N, 0.0, out, M);
    
    size_t ii, jj;
    for (ii = 0; ii < M; ii++){
        for (jj = 0; jj < M; jj++){
            if (ii == jj){
                CuAssertDblEquals(tc, 1.0, out[ii + jj * M], 1e-14);
            }
            else{
                CuAssertDblEquals(tc, 0.0, out[ii+ jj * M], 1e-14);
            }
        }
    }
}

void Test_pinv(CuTest * tc)
{
    printf("Testing Function: pinv\n");
    size_t N = 7;
    size_t M = 5;
    double m[35] = { 0.41, 0.03, 0.73, 0.21, 0.52,
                     0.09, 0.64, 0.72, 0.61, 0.75,
                     0.98, 0.30, 0.67, 0.72, 0.42,
                     0.58, 0.19, 0.97, 0.25, 0.14,
                     0.68, 0.54, 0.87, 0.74, 0.78,
                     0.83, 0.88, 0.56, 0.59, 0.36,
                     0.60, 0.19, 0.71, 0.78, 0.25 };
    
    double m2[35];
    memmove(m2, m, 35 * sizeof(double));

    double ainv[35];
    size_t rank = pinv(N,M, N, m2, ainv, 1e-10);
    
    CuAssertIntEquals(tc, 5, rank);
    
    double out[25];
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, M, M, N,
                    1.0, ainv, M, m, N, 0.0, out, M);
    
    size_t ii, jj;
    for (ii = 0; ii < M; ii++){
        for (jj = 0; jj < M; jj++){
            if (ii == jj){
                CuAssertDblEquals(tc, 1.0, out[ii + jj * M], 1e-14);
            }
            else{
                CuAssertDblEquals(tc, 0.0, out[ii+ jj * M], 1e-14);
            }
        }
    }
}


void Test_rq_with_rmult(CuTest * tc)
{
    printf("Testing Function: rq_with_rmult\n");
    size_t N = 5;
    size_t M = 7;
    double m[35] = { 0.41, 0.03, 0.73, 0.21, 0.52,
                     0.09, 0.64, 0.72, 0.61, 0.75,
                     0.98, 0.30, 0.67, 0.72, 0.42,
                     0.58, 0.19, 0.97, 0.25, 0.14,
                     0.68, 0.54, 0.87, 0.74, 0.78,
                     0.83, 0.88, 0.56, 0.59, 0.36,
                     0.60, 0.19, 0.71, 0.78, 0.25 };
    
    size_t ii, jj;
    double r[25];
    double m2[35];
    memmove(m2,m, 35*sizeof(double));
    memset(r, 0, 25 * sizeof(double));
    for (ii=0;ii<N;ii++){r[ii*N+ii] = 1.0;}
    rq_with_rmult(N,M, m, N, N, N, r, N);
    
    double out[35];
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, M, N, 
                    1.0, r, N, m, N ,0.0,out,N);

    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < M; jj++){
            CuAssertDblEquals(tc, out[ii+ jj * N], m2[ii + jj * N], 1e-14);
        }
    }
    
}

void Test_svd(CuTest * tc)
{
    printf("Testing Function: svd \n");
    
    size_t n = 5; // number of rows
    size_t m = 4; // number of columns
    double a[20];
    double u[25];
    double vt[16];
    double s[4];
    
    a[0] = 1.0; a[1] = 2.0; a[2] = 2.5; a[3] = 4.0; a[4] = -2.0; 
    a[5] = -1.0; a[6] = 3.2; a[7] = 1.5; a[8] = 1.0; a[9] = -3.0; 
    a[10] = 0.5; a[11] = 1.2; a[12] = 4.5; a[13] = 2.0; a[14] = -4.0; 
    a[15] = -0.2; a[16] = -1.2; a[17] = -2.5; a[18] = 3.0; a[19] = -5.0; 
    
    svd(n,m,n,a,u,s,vt);
    
    CuAssertDblEquals(tc, 9.5996, s[0], 1e-2);
    CuAssertDblEquals(tc, 5.727, s[1], 1e-2);
    CuAssertDblEquals(tc, 2.806, s[2], 1e-2);
    CuAssertDblEquals(tc, 2.36, s[3], 1e-2);

    // first column of vt
    CuAssertDblEquals(tc, -0.526, vt[0], 1e-2);
    CuAssertDblEquals(tc, 0.187, vt[1], 1e-2);
    CuAssertDblEquals(tc, -0.549, vt[2], 1e-2);
    CuAssertDblEquals(tc, 0.622, vt[3], 1e-2);

    // last column of vt
    CuAssertDblEquals(tc, -0.390, vt[0+3*m], 1e-2);
    CuAssertDblEquals(tc, -0.918, vt[1+3*m], 1e-2);
    CuAssertDblEquals(tc, -0.009, vt[2+3*m], 1e-2);
    CuAssertDblEquals(tc, -0.064, vt[3+3*m], 1e-2);

    // first column of u
    CuAssertDblEquals(tc, -0.035, u[0], 1e-2);
    CuAssertDblEquals(tc, -0.281, u[1], 1e-2);
    CuAssertDblEquals(tc, -0.394, u[2], 1e-2);
    CuAssertDblEquals(tc, -0.516, u[3], 1e-2);
    CuAssertDblEquals(tc, 0.706, u[4], 1e-2);

    // last column of ut
    CuAssertDblEquals(tc, 0.858, u[0+4*n], 1e-2);
    CuAssertDblEquals(tc, 0.279, u[1+4*n], 1e-2);
    CuAssertDblEquals(tc, -0.1999, u[2+4*n], 1e-2);
    CuAssertDblEquals(tc, -0.328, u[3+4*n], 1e-2);
    CuAssertDblEquals(tc, -0.198, u[4+4*n], 1e-2);

}

void Test_vec_kron(CuTest * tc)
{
    printf("Testing Function: vec_kron\n");

    double m[6] = { 0.41, 0.03, 
                    0.73, 0.09, 
                    0.64, 0.72}; // 2 x 3
    double m2[12] = { 0.98, 0.30, 0.67, 0.72,
                      0.58, 0.19, 0.97 , 0.25,
                      0.33, 0.01, 0.2, 0.45}; // 4 x 3
    
    double vec[9] = { 0.2, 0.4, 0.6, 0.3, 0.5, 0.7, 0.1, 0.05 };
    
    double out[9];
    memset(out, 0.0, 9);
    vec_kron(2,3,m,2,4,3,m2,4,vec,0.0, out);
    double outs[9] = {0.40703, 0.364065, 0.139685, 
                      0.75409, 0.667695, 0.256555,
                      1.17592, 0.92676,  0.36244};
    
    size_t ii;
    for (ii = 0; ii < 9; ii++){
        CuAssertDblEquals(tc, outs[ii], out[ii], 1e-4);
    }
}


void Test_maxvol_rhs(CuTest * tc)
{
    printf("Testing Function: maxvol_rhs\n");
    size_t N = 7;
    size_t M = 4;
    double m[28] = { 0.41, 0.03, 0.73, 0.21, 
                 0.09, 0.64, 0.72, 0.61,
                 0.98, 0.30, 0.67, 0.72,
                 0.58, 0.19, 0.97, 0.25,
                 0.68, 0.54, 0.87, 0.74,
                 0.83, 0.88, 0.56, 0.59,
                 0.60, 0.19, 0.71, 0.78};

    double ainv[16];
    size_t rows[4];
    int success = maxvol_rhs(m,N,M,rows,ainv);
    
    double arows[16];
    size_t ii,jj;
    for (ii = 0; ii < M; ii++){
        for (jj = 0; jj < M; jj++){
            arows[jj + ii*M] = m[rows[jj] + ii*N];
        }
    }
    
    double out[16];
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, M, M, 
                    1.0, arows, M, ainv, M ,0.0,out,M);
    
    for (ii = 0; ii < M; ii++){
        for (jj = 0; jj < M; jj++){
            if (ii == jj){
                CuAssertDblEquals(tc, 1.0, out[ii + jj * M], 1e-14);
            }
            else{
                CuAssertDblEquals(tc, 0.0, out[ii+ jj * M], 1e-14);
            }
        }
    }
    CuAssertIntEquals(tc, 0, success);
}

void Test_kron_col(CuTest * tc)
{
    printf("Testing Function: kron_col\n");

    size_t n1 = 2;
    //int n2 = 3;
    size_t n3 = 4;
    //int n4 = 5;
    double m[6] = { 0.41, 0.03, 
                    0.73, 0.09, 
                    0.64, 0.72};
    double m2[20] = { 0.98, 0.30, 0.67, 0.72,
                      0.58, 0.19, 0.97, 0.25,
                      0.68, 0.54, 0.87, 0.74,
                      0.83, 0.88, 0.56, 0.59,
                      0.60, 0.19, 0.71, 0.78};
    
    // first 2 x 2  (top) ... x ... 3 x 3 (upper left)
    double out[36];
    kron_col(2,2,m,n1,3,3,m2,n3,out,6);
    
    double os[36] = {0.4018,  0.123 ,  0.2747,  0.0294,  0.009 ,  0.0201,
                     0.2378,  0.0779,  0.3977,  0.0174,  0.0057,  0.0291,
                     0.2788,  0.2214,  0.3567,  0.0204,  0.0162,  0.0261,
                     0.7154,  0.219 ,  0.4891,  0.0882,  0.027 ,  0.0603,
                     0.4234,  0.1387,  0.7081,  0.0522,  0.0171,  0.0873,
                     0.4964,  0.3942,  0.6351,  0.0612,  0.0486,  0.0783};
    int ii;
    for (ii = 0; ii < 36; ii++){
        CuAssertDblEquals(tc, os[ii], out[ii], 1e-3);
    }
}


void Test_linear_ls(CuTest * tc)
{
    printf("Testing Function: linear_ls\n");

    size_t nparam = 5;
    double param[5] = { 0.2, 0.3, 0.9, -0.2, 0.1};

    size_t nrows = 8;
    double * x = calloc_double(nparam*nrows);
    double * y = calloc_double(nrows);
    for (size_t ii = 0; ii < nrows; ii++){
        for (size_t jj = 0; jj < nparam; jj++){
            x[jj*nrows+ii] = randn();
            y[ii] += param[jj]*x[jj*nrows+ii];
        }
    }

    double * sol = calloc_double(nparam);
    linear_ls(nrows,nparam,x,y,sol);

    /* dprint(nparam,sol); */
    for (size_t ii = 0; ii < nparam; ii++){
        CuAssertDblEquals(tc,param[ii],sol[ii],1e-14);
    }

    free(x); x = NULL;
    free(y); y = NULL;
    free(sol); sol = NULL;
}

CuSuite * LinalgGetSuite(){
    //printf("----------------------------\n");

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_rq_with_rmult);
    SUITE_ADD_TEST(suite, Test_qr);
    SUITE_ADD_TEST(suite, Test_svd);
    SUITE_ADD_TEST(suite, Test_pinv);
    SUITE_ADD_TEST(suite, Test_vec_kron);
    SUITE_ADD_TEST(suite, Test_maxvol_rhs);
    SUITE_ADD_TEST(suite, Test_kron_col);
    SUITE_ADD_TEST(suite, Test_linear_ls);
    return suite;
}
