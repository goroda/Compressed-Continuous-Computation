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
#include <assert.h>
#include <time.h>

#include "array.h"
#include "CuTest.h"

#include "lib_funcs.h"
#include "lib_clinalg.h"
#include "lib_linalg.h"

typedef struct orth_poly_expansion opoly_t;

void Test_qmarray_orth1d_columns(CuTest * tc)
{
    printf("Testing function: qmarray_orth1d_columns\n");

    enum poly_type ptype = LEGENDRE;
    struct Qmarray * Q = qmarray_orth1d_columns(POLYNOMIAL,
            &ptype, 2,2,-1.0,1.0);
    
    CuAssertIntEquals(tc, 2, Q->nrows);
    CuAssertIntEquals(tc, 2, Q->nrows);

    struct Quasimatrix * q1 = qmarray_extract_column(Q,0);
    struct Quasimatrix * q2 = qmarray_extract_column(Q,1);
    
    double test1 = quasimatrix_inner(q1,q1);
    CuAssertDblEquals(tc,1.0,test1,1e-14);
    double test2 = quasimatrix_inner(q2,q2);
    CuAssertDblEquals(tc,1.0,test2,1e-14);
    double test3 = quasimatrix_inner(q1,q2);
    CuAssertDblEquals(tc,0.0,test3,1e-14);

    quasimatrix_free(q1); q1 = NULL;
    quasimatrix_free(q2); q2 = NULL;
    qmarray_free(Q); Q = NULL;
}

/* void Test_qmarray_householder(CuTest * tc){ */

/*     printf("Testing function: qmarray_householder (1/4)\n"); */

/*     double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     void * args[4] = {&c, &c2, &c3, &c4}; */


/*     enum poly_type p = LEGENDRE; */
/*     struct Qmarray * A = qmarray_approx1d( */
/*                         2, 2, funcs, args, PIECEWISE, &p, -1.0, 1.0, NULL); */
    

/*     struct Qmarray * Acopy = qmarray_copy(A); */

/*     struct Quasimatrix * a1 = qmarray_extract_column(A,0); */
/*     struct Quasimatrix * a2 = qmarray_extract_column(A,1); */

/*     struct Quasimatrix * a3 = qmarray_extract_column(Acopy,0); */
/*     struct Quasimatrix * a4 = qmarray_extract_column(Acopy,1); */
    
/*     struct Quasimatrix * temp = NULL; */
/*     double diff; */
/*     temp = quasimatrix_daxpby(1.0, a1,-1.0, a3); */
/*     diff = quasimatrix_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-15); */
/*     quasimatrix_free(temp); temp = NULL; */

/*     temp = quasimatrix_daxpby(1.0, a2, -1.0, a4); */
/*     diff = quasimatrix_norm(temp); */
/*     quasimatrix_free(temp); temp = NULL; */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-15); */

/*     quasimatrix_free(a1); */
/*     quasimatrix_free(a2); */
/*     quasimatrix_free(a3); */
/*     quasimatrix_free(a4); */

    
/*     double * R = calloc_double(2*2); */

/*     struct Qmarray * Q = qmarray_householder_simple("QR", A,R); */
/*     /\* printf("got QR\n"); *\/ */

/*     CuAssertIntEquals(tc,2,Q->nrows); */
/*     CuAssertIntEquals(tc,2,Q->ncols); */

/*     // test orthogonality */
/*     struct Quasimatrix * q1a = qmarray_extract_column(Q,0); */
/*     struct Quasimatrix * q2a = qmarray_extract_column(Q,1); */
/*     double test1 = quasimatrix_inner(q1a,q1a); */
/*     CuAssertDblEquals(tc,1.0,test1,1e-13); */
/*     double test2 = quasimatrix_inner(q2a,q2a); */
/*     CuAssertDblEquals(tc,1.0,test2,1e-13); */
/*     double test3 = quasimatrix_inner(q1a,q2a); */
/*     CuAssertDblEquals(tc,0.0,test3,1e-13); */

/*     quasimatrix_free(q1a); q1a = NULL; */
/*     quasimatrix_free(q2a); q2a = NULL; */

/*     //dprint2d_col(2,2,R); */
    
/*      // testt equivalence */
/*     struct Qmarray * Anew = qmam(Q,R,2); */
    
/*     struct Quasimatrix * q1 = qmarray_extract_column(Anew,0); */
/*     struct Quasimatrix * q2 = qmarray_extract_column(Anew,1); */

/*     struct Quasimatrix * q3 = qmarray_extract_column(Acopy,0); */
/*     struct Quasimatrix * q4 = qmarray_extract_column(Acopy,1); */
    
/*     temp = quasimatrix_daxpby(1.0, q1,-1.0, q3); */
/*     diff = quasimatrix_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-13); */
/*     quasimatrix_free(temp); */

/*     temp = quasimatrix_daxpby(1.0, q2, -1.0, q4); */
/*     diff = quasimatrix_norm(temp); */
/*     quasimatrix_free(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-13); */

/*     quasimatrix_free(q1); */
/*     quasimatrix_free(q2); */
/*     quasimatrix_free(q3); */
/*     quasimatrix_free(q4); */
/*     qmarray_free(Anew); */

/*     qmarray_free(A); */
/*     qmarray_free(Q); */
/*     qmarray_free(Acopy); */
/*     free(R); */
/* } */

/* void Test_qmarray_householder2(CuTest * tc){ */

/*     printf("Testing function: qmarray_householder (2/4)\n"); */

/*     double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     void * args[4] = {&c, &c2, &c3, &c4}; */


/*     enum poly_type p = LEGENDRE; */
/*     struct Qmarray * A = qmarray_approx1d( */
/*                         1, 4, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL); */
/*     double * R = calloc_double(4*4); */
/*     struct Qmarray * Q = qmarray_householder_simple("QR", A,R); */


/*     struct Quasimatrix * A2 = quasimatrix_approx1d( */
/*                         4, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL); */
/*     double * R2 = calloc_double(4*4); */
/*     struct Quasimatrix * Q2 = quasimatrix_householder_simple(A2,R2); */
    
 
/*     /\* */
/*     printf("=========================\n"); */
/*     printf("R=\n"); */
/*     dprint2d_col(4,4,R); */

/*     printf("R2=\n"); */
/*     dprint2d_col(4,4,R2); */
/*     *\/ */

/*     CuAssertDblEquals(tc, 0.0, norm2diff(R,R2,16),1e-14); */

/*     struct GenericFunction * temp = NULL; */
/*     double diff; */
/*     struct GenericFunction *f1,*f2; */
/*     f1 = qmarray_get_func(Q,0,0); */
/*     f2 = quasimatrix_get_func(Q2,0); */
/*     temp = generic_function_daxpby(1,f1,-1.0,f2); */
/*     diff = generic_function_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff ,1e-14); */
/*     generic_function_free(temp); */

/*     f1 = qmarray_get_func(Q,0,1); */
/*     f2 = quasimatrix_get_func(Q2,1); */
/*     temp = generic_function_daxpby(1,f1,-1.0,f2); */
/*     diff = generic_function_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff ,1e-14); */
/*     generic_function_free(temp); */
    
/*     f1 = qmarray_get_func(Q,0,2); */
/*     f2 = quasimatrix_get_func(Q2,2); */
/*     temp = generic_function_daxpby(1,f1,-1.0,f2); */
/*     diff = generic_function_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff ,1e-14); */
/*     generic_function_free(temp); */

/*     f1 = qmarray_get_func(Q,0,3); */
/*     f2 = quasimatrix_get_func(Q2,3); */
/*     temp = generic_function_daxpby(1,f1,-1.0,f2); */
/*     diff = generic_function_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff ,1e-14); */
/*     generic_function_free(temp); */

/*     qmarray_free(A); */
/*     qmarray_free(Q); */
/*     quasimatrix_free(A2); */
/*     quasimatrix_free(Q2); */
/*     free(R); */
/*     free(R2); */
/* } */

/* void Test_qmarray_householder3(CuTest * tc){ */

/*     printf("Testing function: qmarray_householder (3/4)\n"); */

/*     double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func3}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     void * args[4] = {&c, &c2, &c3, &c4}; */

/*     enum poly_type p = LEGENDRE; */
/*     struct Qmarray * A = qmarray_approx1d( */
/*                         1, 4, funcs, args, PIECEWISE, &p, -1.0, 1.0, NULL); */
/*     size_t N = 100; */
/*     double * xtest = linspace(-1.0, 1.0, N); */
/*     double temp1, temp2, err; */
/*     size_t ii,jj; */
/*     for (ii = 0; ii < 4; ii++){ */
/*         err = 0.0; */
/*         for (jj = 0; jj < N; jj++){ */
/*             temp1 = funcs[ii](xtest[jj],&c); */
/*             temp2 = generic_function_1d_eval(A->funcs[ii],xtest[jj]); */
/*             err += fabs(temp1-temp2); */
/*         } */
/*         //printf("err= %3.15G\n",err); */
/*         CuAssertDblEquals(tc,0.0,err,1e-6); */
/*     } */

/*     double * R = calloc_double(4*4); */
/*     struct Qmarray * Q = qmarray_householder_simple("QR", A,R); */

/*     struct Quasimatrix * A2 = quasimatrix_approx1d( */
/*                         4, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL); */
/*     for (ii = 0; ii < 4; ii++){ */
/*         err = 0.0; */
/*         for (jj = 0; jj < N; jj++){ */
/*             temp1 = funcs[ii](xtest[jj],&c); */
/*             struct GenericFunction * gf = quasimatrix_get_func(A2,ii); */
/*             temp2 = generic_function_1d_eval(gf,xtest[jj]); */
/*             err += fabs(temp1-temp2); */
/*         } */
/*         //printf("err= %3.15G\n",err); */
/*         CuAssertDblEquals(tc,0.0,err,1e-11); */
/*     } */
/*     free(xtest); xtest = NULL; */

/*     double * R2 = calloc_double(4*4); */
/*     struct Quasimatrix * Q2 = quasimatrix_householder_simple(A2,R2); */

/*     struct GenericFunction *f1,*f2,*f3,*f4; */
/*     f1 = qmarray_get_func(Q,0,0); */
/*     f2 = qmarray_get_func(Q,0,1); */
/*     f3 = qmarray_get_func(Q,0,2); */
/*     f4 = qmarray_get_func(Q,0,3); */

/*     double inner; */
/*     inner = generic_function_inner(f1,f1); */
/*     CuAssertDblEquals(tc,1.0,inner,1e-13); */
/*     inner = generic_function_inner(f1,f2); */
/*     CuAssertDblEquals(tc,0.0,inner,1e-13); */
/*     inner = generic_function_inner(f1,f3); */
/*     CuAssertDblEquals(tc,0.0,inner,1e-13); */
/*     inner = generic_function_inner(f1,f4); */
/*     CuAssertDblEquals(tc,0.0,inner,1e-12); */

/*     inner = generic_function_inner(f2,f2); */
/*     CuAssertDblEquals(tc,1.0,inner,1e-13); */
/*     inner = generic_function_inner(f2,f3); */
/*     CuAssertDblEquals(tc,0.0,inner,1e-13); */
/*     inner = generic_function_inner(f2,f4); */
/*     CuAssertDblEquals(tc,0.0,inner,1e-13); */

/*     inner = generic_function_inner(f3,f3); */
/*     CuAssertDblEquals(tc,1.0,inner,1e-13); */
/*     inner = generic_function_inner(f3,f4); */
/*     CuAssertDblEquals(tc,0.0,inner,1e-13); */

/*     inner = generic_function_inner(f4,f4); */
/*     CuAssertDblEquals(tc,1.0,inner,1e-11); */

/*     f1 = quasimatrix_get_func(Q2,0); */
/*     f2 = quasimatrix_get_func(Q2,1); */
/*     f3 = quasimatrix_get_func(Q2,2); */
/*     f4 = quasimatrix_get_func(Q2,3); */

    
/*     inner = generic_function_inner(f1,f1); */
/*     CuAssertDblEquals(tc,1.0,inner,1e-13); */
/*     inner = generic_function_inner(f1,f2); */
/*     CuAssertDblEquals(tc,0.0,inner,1e-13); */
/*     inner = generic_function_inner(f1,f3); */
/*     CuAssertDblEquals(tc,0.0,inner,1e-13); */
/*     inner = generic_function_inner(f1,f4); */
/*     CuAssertDblEquals(tc,0.0,inner,1e-12); */

/*     inner = generic_function_inner(f2,f2); */
/*     CuAssertDblEquals(tc,1.0,inner,1e-13); */
/*     inner = generic_function_inner(f2,f3); */
/*     CuAssertDblEquals(tc,0.0,inner,1e-13); */
/*     inner = generic_function_inner(f2,f4); */
/*     CuAssertDblEquals(tc,0.0,inner,1e-13); */

/*     inner = generic_function_inner(f3,f3); */
/*     CuAssertDblEquals(tc,1.0,inner,1e-13); */
/*     inner = generic_function_inner(f3,f4); */
/*     CuAssertDblEquals(tc,0.0,inner,1e-13); */

/*     inner = generic_function_inner(f4,f4); */
/*     CuAssertDblEquals(tc,1.0,inner,1e-13); */

/*     /\* */
/*     printf("=========================\n"); */
/*     printf("R=\n"); */
/*     dprint2d_col(4,4,R); */

/*     printf("R2=\n"); */
/*     dprint2d_col(4,4,R2); */
/*     *\/ */

/*     CuAssertDblEquals(tc, 0.0, norm2diff(R,R2,16),1e-13); */

/*     struct GenericFunction * temp = NULL; */
/*     double diff; */

/*     f1 = qmarray_get_func(Q,0,0); */
/*     f2 = quasimatrix_get_func(Q2,0); */
/*     temp = generic_function_daxpby(1,f1,-1.0,f2); */
/*     diff = generic_function_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff ,1e-13); */
/*     generic_function_free(temp); */

/*     f1 = qmarray_get_func(Q,0,1); */
/*     f2 = quasimatrix_get_func(Q2,1); */
/*     temp = generic_function_daxpby(1,f1,-1.0,f2); */
/*     diff = generic_function_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff ,1e-13); */
/*     generic_function_free(temp); */

/*     f1 = qmarray_get_func(Q,0,2); */
/*     f2 = quasimatrix_get_func(Q2,2); */
/*     temp = generic_function_daxpby(1,f1,-1.0,f2); */
/*     diff = generic_function_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff ,1e-13); */
/*     generic_function_free(temp); */

/*     //temp = generic_function_daxpby(1,Q->funcs[3],-1.0,Q2->funcs[3]); */
/*    // diff = generic_function_norm(temp); */
/*     //CuAssertDblEquals(tc, 0.0, diff ,1e-14); */
/*    // generic_function_free(temp); */


/*     qmarray_free(A); */
/*     qmarray_free(Q); */
/*     quasimatrix_free(A2); */
/*     quasimatrix_free(Q2); */
/*     free(R); */
/*     free(R2); */
/* } */

/* void Test_qmarray_householder4(CuTest * tc){ */

/*     printf("Testing function: qmarray_householder (4/4)\n"); */

/*     double (*funcs [4])(double, void *) = {&func, &func3, &func3, &func3}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     void * args[4] = {&c, &c2, &c3, &c4}; */


/*     enum poly_type p = LEGENDRE; */
/*     struct Qmarray * A = qmarray_approx1d( */
/*                         1, 4, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL); */
/*     double * R = calloc_double(4*4); */
/*     struct Qmarray * Q = qmarray_householder_simple("QR", A,R); */


/*     struct Quasimatrix * A2 = quasimatrix_approx1d( */
/*                         4, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL); */
/*     double * R2 = calloc_double(4*4); */
/*     struct Quasimatrix * Q2 = quasimatrix_householder_simple(A2,R2); */
    
 
/*     /\* */
/*     printf("=========================\n"); */
/*     printf("R=\n"); */
/*     dprint2d_col(4,4,R); */

/*     printf("R2=\n"); */
/*     dprint2d_col(4,4,R2); */
/*     *\/ */

/*     CuAssertDblEquals(tc, 0.0, norm2diff(R,R2,16),1e-14); */

/*     struct GenericFunction * temp = NULL; */
/*     double diff; */
/*     struct GenericFunction *f1,*f2; */

/*     f1 = qmarray_get_func(Q,0,0); */
/*     f2 = quasimatrix_get_func(Q2,0); */
/*     temp = generic_function_daxpby(1,f1,-1.0,f2); */
/*     diff = generic_function_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff ,1e-14); */
/*     generic_function_free(temp); */

/*     f1 = qmarray_get_func(Q,0,1); */
/*     f2 = quasimatrix_get_func(Q2,1); */
/*     temp = generic_function_daxpby(1,f1,-1.0,f2); */
/*     diff = generic_function_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff ,1e-14); */
/*     generic_function_free(temp); */

/*     f1 = qmarray_get_func(Q,0,2); */
/*     f2 = quasimatrix_get_func(Q2,2); */
/*     temp = generic_function_daxpby(1,f1,-1.0,f2); */
/*     diff = generic_function_norm(temp); */
/*     //print_generic_function(Q->funcs[2],0,NULL); */
/*     //print_generic_function(Q2->funcs[2],0,NULL); */
/*     CuAssertDblEquals(tc, 0.0, diff ,1e-14); */
/*     generic_function_free(temp); */

/*     f1 = qmarray_get_func(Q,0,3); */
/*     f2 = quasimatrix_get_func(Q2,3); */
/*     temp = generic_function_daxpby(1,f1,-1.0,f2); */
/*     diff = generic_function_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff ,1e-14); */
/*     generic_function_free(temp); */


/*     qmarray_free(A); */
/*     qmarray_free(Q); */
/*     quasimatrix_free(A2); */
/*     quasimatrix_free(Q2); */
/*     free(R); */
/*     free(R2); */
/* } */

/* void Test_qmarray_householder_linelm(CuTest * tc){ */
    
/*     // printf("\n\n\n\n\n\n\n\n\n\n"); */
/*     printf("Testing function: qmarray_householder for linelm\n"); */

/*     double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     void * args[4] = {&c, &c2, &c3, &c4}; */

/*     struct Qmarray* T = qmarray_orth1d_columns(LINELM,NULL,2,2,-1.0,1.0); */
/*     double * tmat= qmatqma_integrate(T,T); */
/*     CuAssertDblEquals(tc,1.0,tmat[0],1e-14); */
/*     CuAssertDblEquals(tc,0.0,tmat[1],1e-14); */
/*     CuAssertDblEquals(tc,0.0,tmat[2],1e-14); */
/*     CuAssertDblEquals(tc,1.0,tmat[3],1e-14); */
/* //    printf("tmat = \n"); */
/* //    dprint2d_col(2,2,tmat); */
/*     qmarray_free(T); T = NULL; */
/*     free(tmat); tmat = NULL; */
    
/*     double *x = linspace(-1.0,1.0,5); */
/*     struct LinElemExpAopts * aopts = lin_elem_exp_aopts_alloc(5,x); */
/*     free(x); x= NULL; */

/*     size_t nr = 2; */
/*     size_t nc = 2; */
/*     struct Qmarray * A = qmarray_approx1d( */
/*         nr, nc, funcs, args, LINELM, NULL, -1.0, 1.0, aopts); */
    
/*     struct Qmarray * Acopy = qmarray_copy(A); */
    
/*     double * R = calloc_double(nc*nc); */
/* //    printf("lets go\n"); */
/*     struct Qmarray * Q = qmarray_householder_simple("QR",Acopy,R); */
/* //    print_qmarray(Q,0,NULL); */
/* //    printf("done\n"); */

/* //    print_qmarray(A,0,NULL); */
/*     struct Qmarray * Anew = qmam(Q,R,nc); */

/* //    printf("Q (rows,cols) = (%zu,%zu)\n",Q->nrows,Q->ncols); */
/* //    printf("compute Q^TQ\n"); */
/*     double * qmat = qmatqma_integrate(Q,Q); */
/* //    printf("q is \n"); */
/* //    print_qmarray(Q,0,NULL); */
/* //    dprint2d_col(nc,nc,qmat); */

/* //    printf("norm A = %G\n",qmarray_norm2(A)); */
/* //    printf("norm Q = %G\n",qmarray_norm2(Q)); */
/* //    printf("R is \n"); */
/* //    dprint2d_col(nc,nc,R); */

/*     struct GenericFunction *f1,*f2; */
/*     f1 = qmarray_get_func(A,0,0); */
/*     f2 = qmarray_get_func(Anew,0,0); */
/*     double diff1=generic_function_norm2diff(f1,f2); */
/*     f1 = qmarray_get_func(A,0,1); */
/*     f2 = qmarray_get_func(Anew,0,1); */
/*     double diff2=generic_function_norm2diff(f1,f2); */
/*     f1 = qmarray_get_func(A,1,0); */
/*     f2 = qmarray_get_func(Anew,1,0); */
/*     double diff3=generic_function_norm2diff(f1,f2); */
/*     f1 = qmarray_get_func(A,1,1); */
/*     f2 = qmarray_get_func(Anew,1,1); */
/*     double diff4=generic_function_norm2diff(f1,f2); */

/*     //printf("diffs = %3.15G,%3.15G,%3.15G,%3.15G\n",diff1,diff2,diff3,diff4); */

/* //    assert(1 == 0); */
/* //    assert (fabs(qmat[0] - 1.0) < 1e-10); */
/* //    assert (diff1 < 1e-5); */
/* //    print_qmarray(Anew,0,NULL) */
/*     CuAssertDblEquals(tc,1.0,qmat[0],1e-14); */
/*     CuAssertDblEquals(tc,0.0,qmat[1],1e-14); */
/*     CuAssertDblEquals(tc,0.0,qmat[2],1e-14); */
/*     CuAssertDblEquals(tc,1.0,qmat[3],1e-14); */


/*     CuAssertDblEquals(tc,0.0,diff1,1e-14); */
/*     CuAssertDblEquals(tc,0.0,diff2,1e-14); */
/*     CuAssertDblEquals(tc,0.0,diff3,1e-14); */
/*     CuAssertDblEquals(tc,0.0,diff4,1e-14); */

/*     lin_elem_exp_aopts_free(aopts); */
/*     qmarray_free(Anew); Anew = NULL; */
/*     free(R); R = NULL; */
/*     free(qmat); qmat = NULL; */
/*     qmarray_free(Q); Q = NULL; */
/*     qmarray_free(A); */
/*     qmarray_free(Acopy); */
/* } */

/* void Test_qmarray_householder_hermite1(CuTest * tc){ */
    
/*     // printf("\n\n\n\n\n\n\n\n\n\n"); */
/*     printf("Testing function: qmarray_householder for hermite (1)\n"); */

/*     double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     void * args[4] = {&c, &c2, &c3, &c4}; */

/*     enum function_class fc = POLYNOMIAL; */
/*     enum poly_type ptype = HERMITE; */
/*     struct Qmarray* T = qmarray_orth1d_columns(fc,&ptype,2,2,-DBL_MAX,DBL_MAX); */
/*     double * tmat= qmatqma_integrate(T,T); */
/*     CuAssertDblEquals(tc,1.0,tmat[0],1e-14); */
/*     CuAssertDblEquals(tc,0.0,tmat[1],1e-14); */
/*     CuAssertDblEquals(tc,0.0,tmat[2],1e-14); */
/*     CuAssertDblEquals(tc,1.0,tmat[3],1e-14); */
/* //    printf("tmat = \n"); */
/* //    dprint2d_col(2,2,tmat); */
/*     qmarray_free(T); T = NULL; */
/*     free(tmat); tmat = NULL; */
    
/*     size_t nr = 2; */
/*     size_t nc = 2; */
/*     struct Qmarray * A = qmarray_approx1d( */
/*         nr, nc, funcs, args, fc, &ptype, -1.0, 1.0, NULL); */
    
/*     struct Qmarray * Acopy = qmarray_copy(A); */
    
/*     double * R = calloc_double(nc*nc); */
/*     struct Qmarray * Q = qmarray_householder_simple("QR",Acopy,R); */
/*     struct Qmarray * Anew = qmam(Q,R,nc); */

/*     double * qmat = qmatqma_integrate(Q,Q); */

/*     struct GenericFunction *f1,*f2; */
/*     f1 = qmarray_get_func(A,0,0); */
/*     f2 = qmarray_get_func(Anew,0,0); */
/*     double diff1=generic_function_norm2diff(f1,f2); */
/*     f1 = qmarray_get_func(A,0,1); */
/*     f2 = qmarray_get_func(Anew,0,1); */
/*     double diff2=generic_function_norm2diff(f1,f2); */
/*     f1 = qmarray_get_func(A,1,0); */
/*     f2 = qmarray_get_func(Anew,1,0); */
/*     double diff3=generic_function_norm2diff(f1,f2); */
/*     f1 = qmarray_get_func(A,1,1); */
/*     f2 = qmarray_get_func(Anew,1,1); */
/*     double diff4=generic_function_norm2diff(f1,f2); */

/*     CuAssertDblEquals(tc,1.0,qmat[0],1e-14); */
/*     CuAssertDblEquals(tc,0.0,qmat[1],1e-14); */
/*     CuAssertDblEquals(tc,0.0,qmat[2],1e-14); */
/*     CuAssertDblEquals(tc,1.0,qmat[3],1e-14); */

/*     CuAssertDblEquals(tc,0.0,diff1,1e-13); */
/*     CuAssertDblEquals(tc,0.0,diff2,1e-13); */
/*     CuAssertDblEquals(tc,0.0,diff3,1e-13); */
/*     CuAssertDblEquals(tc,0.0,diff4,1e-10); */

/*     qmarray_free(Anew); Anew = NULL; */
/*     free(R); R = NULL; */
/*     free(qmat); qmat = NULL; */
/*     qmarray_free(Q); Q = NULL; */
/*     qmarray_free(A); */
/*     qmarray_free(Acopy); */
/* } */

/* void Test_qmarray_qr1(CuTest * tc) */
/* { */
/*     printf("Testing function: qmarray_qr (1/3)\n"); */
    
/*     double lb = -2.0; */
/*     double ub = 3.0; */
/*     size_t r1 = 5; */
/*     size_t r2 = 7; */
/*     size_t maxorder = 10; */

/*     struct Qmarray * A = qmarray_poly_randu(LEGENDRE,r1,r2, */
/*                                             maxorder,lb,ub); */
/*     struct Qmarray * Acopy = qmarray_copy(A); */

/*     struct Qmarray * Q = NULL; */
/*     double * R = NULL; */
/*     qmarray_qr(A,&Q,&R); */
/*     //printf("got it \n"); */
    
/*     //printf("R = \n"); */
/*     //dprint2d_col(r2,r2,R); */
    
/*     //printf("Check ortho\n"); */
/*     double * mat = qmatqma_integrate(Q,Q); */
/*     size_t ii,jj; */
/*     for (ii = 0; ii < r2; ii++){ */
/*         for (jj = 0; jj < r2; jj++){ */
/*             if (ii == jj){ */
/*                CuAssertDblEquals(tc,1.0,mat[ii*r2+jj],1e-14); */
/*             } */
/*             else{ */
/*                CuAssertDblEquals(tc,0.0,mat[ii*r2+jj],1e-14); */
/*             } */
/*             if (jj > ii){ */
/*                 CuAssertDblEquals(tc,0.0,R[ii*r2+jj],1e-14); */
/*             } */
/*         } */
/*     } */
/*     //dprint2d_col(r2,r2,mat); */
/*     free(mat); mat = NULL; */

/*     struct Qmarray * QR = qmam(Q,R,r2); */
    
/*     double diff = qmarray_norm2diff(QR,Acopy); */
/*     //printf("diff = %G\n",diff); */
/*     CuAssertDblEquals(tc,0.0,diff*diff,1e-14); */
    
/*     qmarray_free(A); A = NULL; */
/*     qmarray_free(Acopy); Acopy = NULL; */
/*     qmarray_free(Q); Q = NULL; */
/*     qmarray_free(QR); QR = NULL; */
/*     free(R); R = NULL; */
/* } */

/* void Test_qmarray_qr2(CuTest * tc) */
/* { */
/*     printf("Testing function: qmarray_qr (2/3)\n"); */
    
/*     double lb = -2.0; */
/*     double ub = 3.0; */
/*     size_t r1 = 7; */
/*     size_t r2 = 5; */
/*     size_t maxorder = 10; */

/*     struct Qmarray * A = qmarray_poly_randu(LEGENDRE, */
/*                                             r1,r2,maxorder, */
/*                                             lb,ub); */
/*     struct Qmarray * Acopy = qmarray_copy(A); */

/*     struct Qmarray * Q = NULL; */
/*     double * R = NULL; */
/*     qmarray_qr(A,&Q,&R); */
    
/*     double * mat = qmatqma_integrate(Q,Q); */
/*     size_t ii,jj; */
/*     for (ii = 0; ii < r2; ii++){ */
/*         for (jj = 0; jj < r2; jj++){ */
/*             if (ii == jj){ */
/*                CuAssertDblEquals(tc,1.0,mat[ii*r2+jj],1e-14); */
/*             } */
/*             else{ */
/*                CuAssertDblEquals(tc,0.0,mat[ii*r2+jj],1e-14); */
/*             } */
/*             if (jj > ii){ */
/*                 CuAssertDblEquals(tc,0.0,R[ii*r2+jj],1e-14); */
/*             } */
/*         } */
/*     } */
/*     //dprint2d_col(r2,r2,mat); */
/*     free(mat); mat = NULL; */

/*     struct Qmarray * QR = qmam(Q,R,r2); */
    
/*     double diff = qmarray_norm2diff(QR,Acopy); */
/*     //printf("diff = %G\n",diff); */
/*     CuAssertDblEquals(tc,0.0,diff*diff,1e-14); */
    
/*     qmarray_free(A); A = NULL; */
/*     qmarray_free(Acopy); Acopy = NULL; */
/*     qmarray_free(Q); Q = NULL; */
/*     qmarray_free(QR); QR = NULL; */
/*     free(R); R = NULL; */
/* } */

/* void Test_qmarray_qr3(CuTest * tc){ */

/*     printf("Testing function: qmarray_qr (3/3)\n"); */

/*     double (*funcs [4])(double, void *) = {&func, &func2, &func2, &func3}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     void * args[4] = {&c, &c2, &c3, &c4}; */

/*     enum poly_type p = LEGENDRE; */
/*     struct Qmarray * A =  */
/*         qmarray_approx1d(1, 4, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL); */
    
/*     struct Qmarray * Acopy = qmarray_copy(A); */
    
/*     //size_t r1 = 1; */
/*     size_t r2 = 4; */

/*     struct Qmarray * Q = NULL; */
/*     double * R = NULL; */
/*     qmarray_qr(A,&Q,&R); */
/*     //printf("R = \n"); */
/*     //dprint2d_col(r2,r2,R); */
    
/*     double * mat = qmatqma_integrate(Q,Q); */
/*     //printf("mat = \n"); */
/*     //dprint2d_col(r2,r2,mat); */

/*     size_t ii,jj; */
/*     for (ii = 0; ii < r2; ii++){ */
/*         for (jj = 0; jj < r2; jj++){ */
/*             if (ii == jj){ */
/*                CuAssertDblEquals(tc,1.0,mat[ii*r2+jj],1e-14); */
/*             } */
/*             else{ */
/*                CuAssertDblEquals(tc,0.0,mat[ii*r2+jj],1e-14); */
/*             } */
/*             if (jj > ii){ */
/*                 CuAssertDblEquals(tc,0.0,R[ii*r2+jj],1e-14); */
/*             } */
/*         } */
/*     } */
/*     free(mat); mat = NULL; */

/*     struct Qmarray * QR = qmam(Q,R,r2); */
    
/*     double diff = qmarray_norm2diff(QR,Acopy); */
/*     CuAssertDblEquals(tc,0.0,diff*diff,1e-14); */
    
/*     qmarray_free(A); A = NULL; */
/*     qmarray_free(Acopy); Acopy = NULL; */
/*     qmarray_free(Q); Q = NULL; */
/*     qmarray_free(QR); QR = NULL; */
/*     free(R); R = NULL; */
/* } */

/* void Test_qmarray_lq(CuTest * tc) */
/* { */
/*     printf("Testing function: qmarray_lq (1/3)\n"); */
    
/*     double lb = -2.0; */
/*     double ub = 3.0; */
/*     size_t r1 = 5; */
/*     size_t r2 = 7; */
/*     size_t maxorder = 10; */

/*     struct Qmarray * A = qmarray_poly_randu(LEGENDRE,r1,r2, */
/*                                             maxorder,lb,ub); */
/*     struct Qmarray * Acopy = qmarray_copy(A); */

/*     struct Qmarray * Q = NULL; */
/*     double * L = NULL; */
/*     qmarray_lq(A,&Q,&L); */
/*     //printf("got it \n"); */
    
/*     //printf("R = \n"); */
/*     //dprint2d_col(r2,r2,R); */
    
/*     //printf("Check ortho\n"); */
/*     double * mat = qmaqmat_integrate(Q,Q); */
/*     size_t ii,jj; */
/*     for (ii = 0; ii < r1; ii++){ */
/*         for (jj = 0; jj < r1; jj++){ */
/*             if (ii == jj){ */
/*                 CuAssertDblEquals(tc,1.0,mat[ii*r1+jj],1e-14); */
/*             } */
/*             else{ */
/*                 CuAssertDblEquals(tc,0.0,mat[ii*r1+jj],1e-14); */
/*             } */
/*             if (jj < ii){ */
/*                 CuAssertDblEquals(tc,0.0,L[ii*r1+jj],1e-14); */
/*             } */
/*         } */
/*     } */
/*     //dprint2d_col(r2,r2,mat); */
/*     free(mat); mat = NULL; */

/*     struct Qmarray * LQ = mqma(L,Q,r1); */
    
/*     double diff = qmarray_norm2diff(LQ,Acopy); */
/*     //printf("diff = %G\n",diff); */
/*     CuAssertDblEquals(tc,0.0,diff*diff,1e-14); */
    
/*     qmarray_free(A); A = NULL; */
/*     qmarray_free(Acopy); Acopy = NULL; */
/*     qmarray_free(Q); Q = NULL; */
/*     qmarray_free(LQ); LQ = NULL; */
/*     free(L); L = NULL; */
/* } */

/* void Test_qmarray_householder_rows(CuTest * tc){ */

/*     printf("Testing function: qmarray_householder_rows \n"); */

/*     double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     void * args[4] = {&c, &c2, &c3, &c4}; */


/*     enum poly_type p = LEGENDRE; */
/*     struct Qmarray * A = qmarray_approx1d( */
/*                         2, 2, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL); */
    

/*     struct Qmarray * Acopy = qmarray_copy(A); */

/*     struct Quasimatrix * a1 = qmarray_extract_row(A,0); */
/*     struct Quasimatrix * a2 = qmarray_extract_row(A,1); */

/*     struct Quasimatrix * a3 = qmarray_extract_row(Acopy,0); */
/*     struct Quasimatrix * a4 = qmarray_extract_row(Acopy,1); */
    
/*     struct Quasimatrix * temp = NULL; */
/*     double diff; */
/*     temp = quasimatrix_daxpby(1.0, a1,-1.0, a3); */
/*     diff = quasimatrix_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-15); */
/*     quasimatrix_free(temp); temp = NULL; */

/*     temp = quasimatrix_daxpby(1.0, a2, -1.0, a4); */
/*     diff = quasimatrix_norm(temp); */
/*     quasimatrix_free(temp); temp = NULL; */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-15); */

/*     quasimatrix_free(a1); */
/*     quasimatrix_free(a2); */
/*     quasimatrix_free(a3); */
/*     quasimatrix_free(a4); */

    
/*     double * R = calloc_double(2*2); */

/*     struct Qmarray * Q = qmarray_householder_simple("LQ", A,R); */

/*     CuAssertIntEquals(tc,2,Q->nrows); */
/*     CuAssertIntEquals(tc,2,Q->ncols); */

/*     // test orthogonality */
/*     struct Quasimatrix * q1a = qmarray_extract_row(Q,0); */
/*     struct Quasimatrix * q2a = qmarray_extract_row(Q,1); */
/*     double test1 = quasimatrix_inner(q1a,q1a); */
/*     CuAssertDblEquals(tc,1.0,test1,1e-14); */
/*     double test2 = quasimatrix_inner(q2a,q2a); */
/*     CuAssertDblEquals(tc,1.0,test2,1e-14); */
/*     double test3 = quasimatrix_inner(q1a,q2a); */
/*     CuAssertDblEquals(tc,0.0,test3,1e-14); */

/*     quasimatrix_free(q1a); q1a = NULL; */
/*     quasimatrix_free(q2a); q2a = NULL; */

/*     //dprint2d_col(2,2,R); */
    
/*      // testt equivalence */
/*     struct Qmarray * Anew = mqma(R,Q,2); */
    
/*     struct Quasimatrix * q1 = qmarray_extract_row(Anew,0); */
/*     struct Quasimatrix * q2 = qmarray_extract_row(Anew,1); */

/*     struct Quasimatrix * q3 = qmarray_extract_row(Acopy,0); */
/*     struct Quasimatrix * q4 = qmarray_extract_row(Acopy,1); */
    
/*     temp = quasimatrix_daxpby(1.0, q1,-1.0, q3); */
/*     diff = quasimatrix_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-12); */
/*     quasimatrix_free(temp); */

/*     temp = quasimatrix_daxpby(1.0, q2, -1.0, q4); */
/*     diff = quasimatrix_norm(temp); */
/*     quasimatrix_free(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-12); */

/*     quasimatrix_free(q1); */
/*     quasimatrix_free(q2); */
/*     quasimatrix_free(q3); */
/*     quasimatrix_free(q4); */
/*     qmarray_free(Anew); */

/*     qmarray_free(A); */
/*     qmarray_free(Q); */
/*     qmarray_free(Acopy); */
/*     free(R); */
/* } */
/* void Test_qmarray_householder_rows_hermite(CuTest * tc){ */

/*     printf("Testing function: qmarray_householder_rows with hermite polynomials \n"); */

/*     double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     void * args[4] = {&c, &c2, &c3, &c4}; */


/*     enum poly_type p = HERMITE; */
/*     struct Qmarray * A = qmarray_approx1d( */
/*                         2, 2, funcs, args, POLYNOMIAL, &p, */
/*                         -DBL_MAX, DBL_MAX, NULL); */
    
/*     struct Qmarray * Acopy = qmarray_copy(A); */

/*     double * R = calloc_double(2*2); */

/*     struct Qmarray * Q = qmarray_householder_simple("LQ", Acopy,R); */

/*     CuAssertIntEquals(tc,2,Q->nrows); */
/*     CuAssertIntEquals(tc,2,Q->ncols); */
/*     struct Qmarray * Anew = mqma(R,Q,2); */
/*     double * qmat = qmaqmat_integrate(Q,Q); */

/*     double diff1=generic_function_norm2diff(A->funcs[0],Anew->funcs[0]); */
/*     double diff2=generic_function_norm2diff(A->funcs[1],Anew->funcs[1]); */
/*     double diff3=generic_function_norm2diff(A->funcs[2],Anew->funcs[2]); */
/*     double diff4=generic_function_norm2diff(A->funcs[3],Anew->funcs[3]); */

/*     CuAssertDblEquals(tc,1.0,qmat[0],1e-14); */
/*     CuAssertDblEquals(tc,0.0,qmat[1],1e-14); */
/*     CuAssertDblEquals(tc,0.0,qmat[2],1e-14); */
/*     CuAssertDblEquals(tc,1.0,qmat[3],1e-14); */

/*     /\* print_qmarray(A,0,NULL); *\/ */
/*     /\* printf("*************\n"); *\/ */
/*     /\* print_qmarray(Anew,0,NULL) *\/; */
    
/*     CuAssertDblEquals(tc,0.0,diff1,1e-14); */
/*     CuAssertDblEquals(tc,0.0,diff2,1e-12); */
/*     CuAssertDblEquals(tc,0.0,diff3,1e-12); */
/*     CuAssertDblEquals(tc,0.0,diff4,1e-12); */
    
/*     qmarray_free(Anew); Anew = NULL; */
/*     free(R); R = NULL; */
/*     free(qmat); qmat = NULL; */
/*     qmarray_free(Q); Q = NULL; */
/*     qmarray_free(A); A = NULL; */
/*     qmarray_free(Acopy); Acopy = NULL; */

/* } */

/* void Test_qmarray_householder_rowslinelm(CuTest * tc){ */
    
/*     // printf("\n\n\n\n\n\n\n\n\n\n"); */
/*     printf("Testing function: qmarray_householder LQ for linelm\n"); */

/*     double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     void * args[4] = {&c, &c2, &c3, &c4}; */

/*     struct Qmarray* T = qmarray_orth1d_rows(LINELM,NULL,2,2,-1.0,1.0); */
/*     double * tmat= qmaqmat_integrate(T,T); */
/*     CuAssertDblEquals(tc,1.0,tmat[0],1e-14); */
/*     CuAssertDblEquals(tc,0.0,tmat[1],1e-14); */
/*     CuAssertDblEquals(tc,0.0,tmat[2],1e-14); */
/*     CuAssertDblEquals(tc,1.0,tmat[3],1e-14); */
/* //    printf("tmat = \n"); */
/* //    dprint2d_col(2,2,tmat); */
/*     qmarray_free(T); T = NULL; */
/*     free(tmat); tmat = NULL; */

/*     double *x = linspace(-1.0,1.0,5); */
/*     struct LinElemExpAopts * aopts = lin_elem_exp_aopts_alloc(5,x); */
/*     free(x); x= NULL; */
    
/*     size_t nr = 2; */
/*     size_t nc = 2; */
/*     struct Qmarray * A = qmarray_approx1d( */
/*         nr, nc, funcs, args, LINELM, NULL, -1.0, 1.0, aopts); */
/*     lin_elem_exp_aopts_free(aopts); */

/*     struct Qmarray * Acopy = qmarray_copy(A); */
    
/*     double * R = calloc_double(nr*nr); */

/*     struct Qmarray * Q = qmarray_householder_simple("LQ",Acopy,R); */

/*     struct Qmarray * Anew = mqma(R,Q,nr); */

/*     double * qmat = qmaqmat_integrate(Q,Q); */

/*     double diff1=generic_function_norm2diff(A->funcs[0],Anew->funcs[0]); */
/*     double diff2=generic_function_norm2diff(A->funcs[1],Anew->funcs[1]); */
/*     double diff3=generic_function_norm2diff(A->funcs[2],Anew->funcs[2]); */
/*     double diff4=generic_function_norm2diff(A->funcs[3],Anew->funcs[3]); */

/*     CuAssertDblEquals(tc,1.0,qmat[0],1e-14); */
/*     CuAssertDblEquals(tc,0.0,qmat[1],1e-14); */
/*     CuAssertDblEquals(tc,0.0,qmat[2],1e-14); */
/*     CuAssertDblEquals(tc,1.0,qmat[3],1e-14); */

/*     CuAssertDblEquals(tc,0.0,diff1,1e-14); */
/*     CuAssertDblEquals(tc,0.0,diff2,1e-14); */
/*     CuAssertDblEquals(tc,0.0,diff3,1e-14); */
/*     CuAssertDblEquals(tc,0.0,diff4,1e-14); */
    
/*     qmarray_free(Anew); Anew = NULL; */
/*     free(R); R = NULL; */
/*     free(qmat); qmat = NULL; */
/*     qmarray_free(Q); Q = NULL; */
/*     qmarray_free(A); */
/*     qmarray_free(Acopy); */
/* } */

/* void Test_qmarray_lu1d(CuTest * tc){ */

/*     printf("Testing function: qmarray_lu1d (1/2)\n"); */
/*     double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     void * args[4] = {&c, &c2, &c3, &c4}; */


/*     enum poly_type p = LEGENDRE; */
/*     struct Qmarray * A = qmarray_approx1d( */
/*                         2, 2, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL); */
    

/*     struct Qmarray * L = qmarray_alloc(2,2); */

/*     struct Qmarray * Acopy = qmarray_copy(A); */

/*     double * U = calloc_double(2*2); */
/*     size_t * pivi = calloc_size_t(2); */
/*     double * pivx = calloc_double(2); */
/*     qmarray_lu1d(A,L,U,pivi,pivx,NULL); */
    
/*     double eval; */
    
/*     //print_qmarray(A,0,NULL); */
/*     // check pivots */
/*     //printf("U = \n"); */
/*     //dprint2d_col(2,2,U); */
/*     eval = generic_function_1d_eval(L->funcs[2+ pivi[0]], pivx[0]); */
/*     //printf("eval = %G\n",eval); */
/*     CuAssertDblEquals(tc, 0.0, eval, 1e-13); */
    
/*     struct Qmarray * Comb = qmam(L,U,2); */
/*     double difff = qmarray_norm2diff(Comb,Acopy); */
/*     //printf("difff = %G\n",difff); */
/*     CuAssertDblEquals(tc,difff,0,1e-14); */
    
/*     //exit(1); */
/*     qmarray_free(Acopy); */
/*     qmarray_free(A); */
/*     qmarray_free(Comb); */
/*     qmarray_free(L); */
/*     free(U); */
/*     free(pivx); */
/*     free(pivi); */
/* } */

/* void Test_qmarray_lu1d2(CuTest * tc){ */

/*     printf("Testing function: qmarray_lu1d (2/2)\n"); */
/*     //this is column ordered when convertest to Qmarray */
/*     double (*funcs [6])(double, void *) = {&func,  &func4, &func,  */
/*                                            &func4, &func5, &func6}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     struct counter c5; c5.N = 0; */
/*     struct counter c6; c6.N = 0; */
/*     void * args[6] = {&c, &c2, &c3, &c4, &c5, &c6}; */


/*     enum poly_type p = LEGENDRE; */
/*     struct Qmarray * A = qmarray_approx1d( */
/*                         2, 3, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL); */
/*     //printf("A = (%zu,%zu)\n",A->nrows,A->ncols); */

/*     struct Qmarray * L = qmarray_alloc(2,3); */

/*     struct Qmarray * Acopy = qmarray_copy(A); */

/*     double * U = calloc_double(3*3); */
/*     size_t * pivi = calloc_size_t(3); */
/*     double * pivx = calloc_double(3); */
/*     qmarray_lu1d(A,L,U,pivi,pivx,NULL); */
    
/*     double eval; */
    
/*     // check pivots */
/*     size_t ii,jj; */
/*     for (ii = 0; ii < 3; ii++){ */
/*         for (jj = 0; jj < ii; jj++){ */
/*             eval = generic_function_1d_eval(L->funcs[2*ii+pivi[jj]], pivx[jj]); */
/*             CuAssertDblEquals(tc,0.0,eval,1e-14); */
/*         } */

/*         eval = generic_function_1d_eval(L->funcs[2*ii+pivi[ii]], pivx[ii]); */
/*         CuAssertDblEquals(tc,1.0,eval,1e-14); */
/*     } */
    
/*     struct Qmarray * Comb = qmam(L,U,3); */
/*     double difff = qmarray_norm2diff(Comb,Acopy); */
/*     CuAssertDblEquals(tc,difff,0,1e-13); */
    
/*     //exit(1); */
/*     qmarray_free(Acopy); */
/*     qmarray_free(A); */
/*     qmarray_free(Comb); */
/*     qmarray_free(L); */
/*     free(U); */
/*     free(pivx); */
/*     free(pivi); */
/* } */

/* void Test_qmarray_lu1d_hermite(CuTest * tc){ */

/*     printf("Testing function: qmarray_lu1d with hermite (1)\n"); */
/*     //this is column ordered when convertest to Qmarray */
/*     double (*funcs [6])(double, void *) = {&func,  &func4, &func6,  */
/*                                            &func4, &func5, &func3}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     struct counter c5; c5.N = 0; */
/*     struct counter c6; c6.N = 0; */
/*     void * args[6] = {&c, &c2, &c3, &c4, &c5, &c6}; */


/*     enum poly_type p = HERMITE; */
/*     struct Qmarray * A = qmarray_approx1d( */
/*         2, 3, funcs, args, POLYNOMIAL, &p, -DBL_MAX, DBL_MAX, NULL); */
/*     //printf("A = (%zu,%zu)\n",A->nrows,A->ncols); */

/*     struct Qmarray * L = qmarray_alloc(2,3); */

/*     struct Qmarray * Acopy = qmarray_copy(A); */

/*     double * U = calloc_double(3*3); */
/*     size_t * pivi = calloc_size_t(3); */
/*     double * pivx = calloc_double(3); */

/*     size_t nopt = 60; */
/*     double * xopt = linspace(-10.0,10.0,nopt); */
/*     struct c3Vector * c3v = c3vector_alloc(nopt,xopt); */
/*     qmarray_lu1d(A,L,U,pivi,pivx,c3v); */
/*     free(xopt); xopt = NULL; */
/*     c3vector_free(c3v); c3v = NULL; */
    
/*     double eval; */

/*     /\* printf("pivots "); *\/ */
/*     /\* iprint_sz(3,pivi); *\/ */
/*     /\* printf("pivot x"); *\/ */
/*     /\* dprint(3,pivx); *\/ */
/*     //dprint2d_col(3,3,U); */
/*     // check pivots */
/*     size_t ii,jj; */
/*     for (ii = 0; ii < 3; ii++){ */
/*         for (jj = 0; jj < ii; jj++){ */
/*             eval = generic_function_1d_eval(L->funcs[2*ii+pivi[jj]], pivx[jj]); */
/* //            double nt = generic_function_array_norm(2,1,L->funcs+2*ii); */
/*             //printf("nt = %G\n",nt); */
/*             CuAssertDblEquals(tc,0.0,eval,1e-13); */
/*         } */

/*         eval = generic_function_1d_eval(L->funcs[2*ii+pivi[ii]], pivx[ii]); */
/*         CuAssertDblEquals(tc,1.0,eval,1e-14); */
/*     } */
    
/*     struct Qmarray * Comb = qmam(L,U,3); */
/*     double diff = qmarray_norm2diff(Comb,Acopy); */
/*     double norm1 = qmarray_norm2(Acopy); */
/*     //printf("diff=%G, reldiff=%G\n",diff,diff/norm1); */
/*     CuAssertDblEquals(tc,0.0,diff/norm1,1e-13); */
    
/*     //exit(1); */
/*     qmarray_free(Acopy); */
/*     qmarray_free(A); */
/*     qmarray_free(Comb); */
/*     qmarray_free(L); */
/*     free(U); */
/*     free(pivx); */
/*     free(pivi); */
/* } */

/* void Test_qmarray_lu1d_linelm(CuTest * tc){ */

/*     printf("Testing function: qmarray_lu1d with linelm \n"); */
/*     //this is column ordered when convertest to Qmarray */
/*     double (*funcs [6])(double, void *) = {&func,  &func4, &func,  */
/*                                            &func4, &func5, &func6}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     struct counter c5; c5.N = 0; */
/*     struct counter c6; c6.N = 0; */
/*     void * args[6] = {&c, &c2, &c3, &c4, &c5, &c6}; */

/*     struct Qmarray * A = qmarray_approx1d( */
/*                         2, 3, funcs, args, LINELM, NULL, -1.0, 1.0, NULL); */
/*     //printf("A = (%zu,%zu)\n",A->nrows,A->ncols); */

/*     struct Qmarray * L = qmarray_alloc(2,3); */

/*     struct Qmarray * Acopy = qmarray_copy(A); */

/*     double * U = calloc_double(3*3); */
/*     size_t * pivi = calloc_size_t(3); */
/*     double * pivx = calloc_double(3); */
/*     qmarray_lu1d(A,L,U,pivi,pivx,NULL); */
    
/*     double eval; */
    
/*     //print_qmarray(A,0,NULL); */
/*     // check pivots */
/*     //printf("U = \n"); */
/*     //dprint2d_col(2,2,U); */
/*     size_t ii,jj; */
/*     for (ii = 0; ii < 3; ii++){ */
/*         //printf("Checking column %zu \n",ii); */
/*         //printf("---------------\n"); */
/*         for (jj = 0; jj < ii; jj++){ */
/*             //printf("Should have zero at (%zu,%G)\n",pivi[jj],pivx[jj]); */
/*             eval = generic_function_1d_eval(L->funcs[2*ii+pivi[jj]], pivx[jj]); */
/*             CuAssertDblEquals(tc,0.0,eval,1e-14); */
/*             //printf("eval = %G\n",eval); */
/*         } */
/*         //printf("Should have one at (%zu,%G)\n",pivi[ii],pivx[ii]); */
/*         eval = generic_function_1d_eval(L->funcs[2*ii+pivi[ii]], pivx[ii]); */
/*         CuAssertDblEquals(tc,1.0,eval,1e-14); */
/*         //printf("eval = %G\n",eval); */
/*     } */
/*     /\* */
/*     eval = generic_function_1d_eval(L->funcs[2+ pivi[0]], pivx[0]); */
/*     printf("eval = %G\n",eval); */
/*     eval = generic_function_1d_eval(L->funcs[4+ pivi[1]], pivx[1]); */
/*     printf("eval = %G\n",eval); */
/*     eval = generic_function_1d_eval(L->funcs[4+ pivi[0]], pivx[0]); */
/*     printf("eval = %G\n",eval); */
/*     *\/ */

/*     //CuAssertDblEquals(tc, 0.0, eval, 1e-13); */
    
/*     struct Qmarray * Comb = qmam(L,U,3); */
/*     double difff = qmarray_norm2diff(Comb,Acopy); */
/*     //printf("difff = %G\n",difff); */
/*     CuAssertDblEquals(tc,difff,0,1e-13); */
    
/*     //exit(1); */
/*     qmarray_free(Acopy); */
/*     qmarray_free(A); */
/*     qmarray_free(Comb); */
/*     qmarray_free(L); */
/*     free(U); */
/*     free(pivx); */
/*     free(pivi); */
/* } */

/* void Test_qmarray_maxvol1d(CuTest * tc){ */

/*     printf("Testing function: qmarray_maxvol1d (1/2) \n"); */

/*     double (*funcs [6])(double, void *) =  */
/*         {&func, &func2, &func3, &func4, */
/*          &func5, &func6}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     struct counter c5; c5.N = 0; */
/*     struct counter c6; c6.N = 0; */
/*     void * args[6] = {&c, &c2, &c3, &c4, &c5, &c6}; */


/*     enum poly_type p = LEGENDRE; */
/*     struct Qmarray * A =  */
/*         qmarray_approx1d( */
/*             3, 2, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL); */
    

/*     double * Asinv = calloc_double(2*2); */
/*     size_t * pivi = calloc_size_t(2); */
/*     double * pivx= calloc_double(2); */

/*     qmarray_maxvol1d(A,Asinv,pivi,pivx,NULL); */
     
/*     /\* */
/*     printf("pivots at = \n"); */
/*     iprint_sz(3,pivi);  */
/*     dprint(3,pivx); */
/*     *\/ */

/*     struct Qmarray * B = qmam(A,Asinv,2); */
/*     double maxval, maxloc; */
/*     size_t maxrow, maxcol; */
/*     qmarray_absmax1d(B,&maxloc,&maxrow, &maxcol, &maxval,NULL); */
/*     //printf("Less = %d", 1.0+1e-2 > maxval); */
/*     CuAssertIntEquals(tc, 1, (1.0+1e-2) > maxval); */
/*     qmarray_free(B); */

/*     qmarray_free(A); */
/*     free(Asinv); */
/*     free(pivx); */
/*     free(pivi); */
/* } */

/* void Test_qmarray_maxvol1d2(CuTest * tc){ */

/*     printf("Testing function: qmarray_maxvol1d (2/2) \n"); */

/*     double (*funcs [6])(double, void *) = */
/*         {&func, &func2, &func3, &func4, */
/*          &func4, &func4}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     struct counter c5; c5.N = 0; */
/*     struct counter c6; c6.N = 0; */
/*     void * args[6] = {&c, &c2, &c3, &c4, &c5, &c6}; */


/*     enum poly_type p = LEGENDRE; */
/*     struct Qmarray * A =  */
/*         qmarray_approx1d( */
/*             1, 6, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL); */
    

/*     double * Asinv = calloc_double(6*6); */
/*     size_t * pivi = calloc_size_t(6); */
/*     double * pivx= calloc_double(6); */

/*     qmarray_maxvol1d(A,Asinv,pivi,pivx,NULL); */
     
/*     /\* */
/*     printf("pivots at = \n"); */
/*     iprint_sz(6,pivi);  */
/*     dprint(6,pivx); */
/*     *\/ */

/*     struct Qmarray * B = qmam(A,Asinv,2); */
/*     double maxval, maxloc; */
/*     size_t maxrow, maxcol; */
/*     qmarray_absmax1d(B,&maxloc,&maxrow, &maxcol, &maxval,NULL); */
/*     //printf("Less = %d", 1.0+1e-2 > maxval); */
/*     CuAssertIntEquals(tc, 1, (1.0+1e-2) > maxval); */
/*     qmarray_free(B); */

/*     qmarray_free(A); */
/*     free(Asinv); */
/*     free(pivx); */
/*     free(pivi); */
/* } */

/* void Test_qmarray_maxvol1d_hermite1(CuTest * tc){ */

/*     printf("Testing function: qmarray_maxvol1d with hermite poly (1) \n"); */

/*     double (*funcs [6])(double, void *) = */
/*         {&func, &func2, &func3, &func4, */
/*          &func4, &func4}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     struct counter c5; c5.N = 0; */
/*     struct counter c6; c6.N = 0; */
/*     void * args[6] = {&c, &c2, &c3, &c4, &c5, &c6}; */


/*     enum poly_type p = HERMITE; */
/*     struct Qmarray * A =  */
/*         qmarray_approx1d( */
/*             1, 6, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL); */
    

/*     double * Asinv = calloc_double(6*6); */
/*     size_t * pivi = calloc_size_t(6); */
/*     double * pivx= calloc_double(6); */

/*     size_t nopt = 40; */
/*     double * xopt = linspace(-10.0,10.0,nopt); */
/*     struct c3Vector * c3v = c3vector_alloc(nopt,xopt); */
/*     qmarray_maxvol1d(A,Asinv,pivi,pivx,c3v); */
     
/*     /\* */
/*     printf("pivots at = \n"); */
/*     iprint_sz(6,pivi);  */
/*     dprint(6,pivx); */
/*     *\/ */

/*     struct Qmarray * B = qmam(A,Asinv,2); */
/*     double maxval, maxloc; */
/*     size_t maxrow, maxcol; */
/*     qmarray_absmax1d(B,&maxloc,&maxrow,&maxcol,&maxval,c3v); */
/*     //printf("Less = %d", 1.0+1e-2 > maxval); */
/*     CuAssertIntEquals(tc, 1, (1.0+1e-2) > maxval); */

/*     free(xopt); xopt = NULL; */
/*     c3vector_free(c3v); c3v = NULL; */
/*     qmarray_free(B); */
/*     qmarray_free(A); */
/*     free(Asinv); */
/*     free(pivx); */
/*     free(pivi); */
/* } */

/* void Test_qmarray_maxvol1d_linelm(CuTest * tc){ */

/*     printf("Testing function: qmarray_maxvol1d linelm (1)\n"); */

/*     double (*funcs [6])(double, void *) =  */
/*         {&func, &func2, &func3, &func4, */
/*          &func5, &func6}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     struct counter c5; c5.N = 0; */
/*     struct counter c6; c6.N = 0; */
/*     void * args[6] = {&c, &c2, &c3, &c4, &c5, &c6}; */

/*     struct Qmarray * A =  */
/*         qmarray_approx1d(3, 2, funcs, args, LINELM,  */
/*                          NULL, -1.0, 1.0, NULL); */

/*     unsigned char * text = NULL; */
/*     size_t size; */
/*     qmarray_serialize(NULL,A,&size); */
/*     text = malloc(size * sizeof(unsigned char)); */
/*     qmarray_serialize(text,A,NULL); */
    
/*     struct Qmarray * C = NULL; */
/*     qmarray_deserialize(text,&C); */
/*     free(text); text = NULL; */

/*     double diff = qmarray_norm2diff(A,C); */
/*     CuAssertDblEquals(tc,0.0,diff,1e-10); */
/*     qmarray_free(C); C = NULL; */

    
/*     double * Asinv = calloc_double(2*2); */
/*     size_t * pivi = calloc_size_t(2); */
/*     double * pivx= calloc_double(2); */

/*     qmarray_maxvol1d(A,Asinv,pivi,pivx,NULL); */
     
/*     /\* */
/*     printf("pivots at = \n"); */
/*     iprint_sz(3,pivi);  */
/*     dprint(3,pivx); */
/*     *\/ */

/*     struct Qmarray * B = qmam(A,Asinv,2); */
/*     double maxval, maxloc; */
/*     size_t maxrow, maxcol; */
/*     qmarray_absmax1d(B,&maxloc,&maxrow, &maxcol, &maxval,NULL); */
/*     //printf("Less = %d", 1.0+1e-2 > maxval); */
/*     CuAssertIntEquals(tc, 1, (1.0+1e-2) > maxval); */
/*     qmarray_free(B); */

/*     qmarray_free(A); */
/*     free(Asinv); */
/*     free(pivx); */
/*     free(pivi); */
/* } */

/* void Test_qmarray_svd(CuTest * tc){ */

/*     printf("Testing function: qmarray_svd \n"); */

/*     double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     void * args[4] = {&c, &c2, &c3, &c4}; */


/*     enum poly_type p = LEGENDRE; */
/*     struct Qmarray * A = qmarray_approx1d( */
/*                         2, 2, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL); */

/*     struct Qmarray * Acopy = qmarray_copy(A); */

/*     double * vt = calloc_double(2*2); */
/*     double * s = calloc_double(2); */
/*     struct Qmarray * Q = NULL; */

/* //    printf("compute SVD of qmarray\n"); */
/*     qmarray_svd(A,&Q,s,vt); */
/* //    printf("done computing!\n"); */

/*     CuAssertIntEquals(tc,2,Q->nrows); */
/*     CuAssertIntEquals(tc,2,Q->ncols); */

/*     // test orthogonality */

/*     struct Quasimatrix * q1a = qmarray_extract_column(Q,0); */
/*     struct Quasimatrix * q2a = qmarray_extract_column(Q,1); */
/*     double test1 = quasimatrix_inner(q1a,q1a); */
/*     CuAssertDblEquals(tc,1.0,test1,1e-14); */
/*     double test2 = quasimatrix_inner(q2a,q2a); */
/*     CuAssertDblEquals(tc,1.0,test2,1e-14); */
/*     double test3 = quasimatrix_inner(q1a,q2a); */
/*     CuAssertDblEquals(tc,0.0,test3,1e-14); */

/*     quasimatrix_free(q1a); q1a = NULL; */
/*     quasimatrix_free(q2a); q2a = NULL; */

/*     //dprint2d_col(2,2,R); */
    
/*      // testt equivalence */
/*     struct Quasimatrix * temp = NULL; */
/*     double * comb = calloc_double(2*2); */

/*     double * sdiag = diag(2, s); */
    
/*     cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, 2, 2, 2, 1.0,  */
/*                     sdiag, 2, vt, 2, 0.0, comb, 2); */
/*     //comb = dgemm */
/*     free(s); */
/*     free(sdiag); */
/*     free(vt); */

/*     //printf("on the quasimatrix portion\n"); */
/*     double diff; */
/*     struct Qmarray * Anew = qmam(Q,comb,2); */
/*     free(comb); */
    
/*     struct Quasimatrix * q1 = qmarray_extract_column(Anew,0); */
/*     struct Quasimatrix * q2 = qmarray_extract_column(Anew,1); */

/*     struct Quasimatrix * q3 = qmarray_extract_column(Acopy,0); */
/*     struct Quasimatrix * q4 = qmarray_extract_column(Acopy,1); */
    
/*     temp = quasimatrix_daxpby(1.0, q1,-1.0, q3); */
/*     diff = quasimatrix_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-12); */
/*     quasimatrix_free(temp); */

/*     temp = quasimatrix_daxpby(1.0, q2, -1.0, q4); */
/*     diff = quasimatrix_norm(temp); */
/*     quasimatrix_free(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-12); */
    
/*     quasimatrix_free(q1); */
/*     quasimatrix_free(q2); */
/*     quasimatrix_free(q3); */
/*     quasimatrix_free(q4); */
/*     qmarray_free(Anew); */

/*     qmarray_free(A); */
/*     qmarray_free(Q); */
/*     qmarray_free(Acopy); */
/* } */

/* void Test_qmarray_serialize(CuTest * tc){ */

/*     printf("Testing function: (de)qmarray_serialize\n"); */

/*     double (*funcs [6])(double, void *) = {&func, &func2, &func3, &func4, */
/*                                             &func5, &func6}; */
/*     struct counter c; c.N = 0; */
/*     struct counter c2; c2.N = 0; */
/*     struct counter c3; c3.N = 0; */
/*     struct counter c4; c4.N = 0; */
/*     struct counter c5; c5.N = 0; */
/*     struct counter c6; c6.N = 0; */
/*     void * args[6] = {&c, &c2, &c3, &c4, &c5, &c6}; */


/*     enum poly_type p = LEGENDRE; */
/*     struct Qmarray * A = qmarray_approx1d( */
/*                         3, 2, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL); */
    
/*     unsigned char * text = NULL; */
/*     size_t size; */
/*     qmarray_serialize(NULL,A,&size); */
/*     text = malloc(size * sizeof(unsigned char)); */
/*     qmarray_serialize(text,A,NULL); */
    

/*     struct Qmarray * B = NULL; */
/*     qmarray_deserialize(text,&B); */
/*     free(text); text = NULL; */

/*     CuAssertIntEquals(tc,3,B->nrows); */
/*     CuAssertIntEquals(tc,2,B->ncols); */

/*     struct GenericFunction * temp; */
/*     double diff; */

/*     temp = generic_function_daxpby(1.0, A->funcs[0],-1.0,B->funcs[0]); */
/*     diff = generic_function_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-15); */
/*     generic_function_free(temp); */

/*     temp = generic_function_daxpby(1.0, A->funcs[1],-1.0,B->funcs[1]); */
/*     diff = generic_function_norm(temp); */
/*     generic_function_free(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-13); */

/*     temp = generic_function_daxpby(1.0, A->funcs[2],-1.0,B->funcs[2]); */
/*     diff = generic_function_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-13); */
/*     generic_function_free(temp); */

/*     temp = generic_function_daxpby(1.0, A->funcs[3],-1.0,B->funcs[3]); */
/*     diff = generic_function_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-15); */
/*     generic_function_free(temp); */

/*     temp = generic_function_daxpby(1.0, A->funcs[4],-1.0,B->funcs[4]); */
/*     diff = generic_function_norm(temp); */
/*     generic_function_free(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-13); */

/*     temp = generic_function_daxpby(1.0, A->funcs[5],-1.0,B->funcs[5]); */
/*     diff = generic_function_norm(temp); */
/*     CuAssertDblEquals(tc, 0.0, diff, 1e-13); */
/*     generic_function_free(temp); */

/*     qmarray_free(A); */
/*     qmarray_free(B); */
/* } */


CuSuite * CLinalgQmarrayGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_qmarray_orth1d_columns);
    /* SUITE_ADD_TEST(suite, Test_qmarray_householder); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_householder2); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_householder3); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_householder4); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_householder_linelm); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_householder_hermite1); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_qr1); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_qr2); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_qr3); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_lq); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_householder_rows); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_householder_rows_hermite); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_householder_rowslinelm); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_lu1d); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_lu1d2); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_lu1d_hermite); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_lu1d_linelm); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_maxvol1d); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_maxvol1d2); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_maxvol1d_hermite1); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_maxvol1d_linelm); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_svd); */
    /* SUITE_ADD_TEST(suite, Test_qmarray_serialize); */
    
    return suite;
}
