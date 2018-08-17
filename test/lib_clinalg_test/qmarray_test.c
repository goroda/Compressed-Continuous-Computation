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
#include <assert.h>
#include <time.h>

#include "array.h"

#include "CuTest.h"
#include "testfunctions.h"

#include "lib_funcs.h"
#include "lib_linalg.h"

#include "quasimatrix.h"
#include "qmarray.h"


static void qmarray_test_col_orth(CuTest * tc, struct Qmarray * A, double tol)
{
    size_t ncols = qmarray_get_ncols(A);
    for (size_t ii = 0; ii < ncols; ii++){
        for (size_t jj = 0; jj < ncols; jj++){
            struct Quasimatrix * q1a = qmarray_extract_column(A,ii);
            struct Quasimatrix * q2a = qmarray_extract_column(A,jj);
            if (ii == jj){
                double test2 = quasimatrix_inner(q1a,q2a);
                CuAssertDblEquals(tc,1.0,test2,tol);                
            }
            else{
                double test1 = quasimatrix_inner(q1a,q2a);
                CuAssertDblEquals(tc,0.0,test1,tol);
            }
            quasimatrix_free(q1a); q1a = NULL;
            quasimatrix_free(q2a); q2a = NULL;
        }
    }
}

static void qmarray_test_row_orth(CuTest * tc, struct Qmarray * A, double tol)
{
    size_t nrows = qmarray_get_nrows(A);
    for (size_t ii = 0; ii < nrows; ii++){
        for (size_t jj = 0; jj < nrows; jj++){
            struct Quasimatrix * q1a = qmarray_extract_row(A,ii);
            struct Quasimatrix * q2a = qmarray_extract_row(A,jj);
            if (ii == jj){
                double test2 = quasimatrix_inner(q1a,q2a);
                CuAssertDblEquals(tc,1.0,test2,tol);                
            }
            else{
                double test1 = quasimatrix_inner(q1a,q2a);
                CuAssertDblEquals(tc,0.0,test1,tol);
            }
            quasimatrix_free(q1a); q1a = NULL;
            quasimatrix_free(q2a); q2a = NULL;
        }
    }
}

static void quasimatrix_test_col_orth(CuTest * tc, struct Quasimatrix * A, double tol)
{
    size_t ncols = quasimatrix_get_size(A);
    double inner;
    for (size_t ii = 0; ii < ncols; ii++){
        for (size_t jj = 0; jj < ncols; jj++){
            struct GenericFunction * f1 = quasimatrix_get_func(A,ii);
            struct GenericFunction * f2 = quasimatrix_get_func(A,jj);
            if (ii == jj){
                inner = generic_function_inner(f1,f2);
                CuAssertDblEquals(tc,1.0,inner,tol);                
            }
            else{
                inner = generic_function_inner(f1,f2);
                CuAssertDblEquals(tc,0.0,inner,tol);                
            }
        }
    }
}

static void 
qmarray_test_equality1(CuTest * tc, struct Qmarray * A,
                       struct Qmarray * B, double tol)
{
    size_t nrowsA = qmarray_get_nrows(A);
    size_t ncolsA = qmarray_get_ncols(A);

    size_t nrowsB = qmarray_get_nrows(B);
    size_t ncolsB = qmarray_get_ncols(B);

    CuAssertIntEquals(tc,nrowsA,nrowsB);
    CuAssertIntEquals(tc,ncolsA,ncolsB);

    for (size_t ii = 0; ii < nrowsA; ii++){
        /* printf("Testing row %zu\n", ii); */
        for (size_t jj = 0; jj < ncolsA; jj++){
            /* printf("\t Testing col %zu\n", jj); */
            struct GenericFunction * f1 = qmarray_get_func(A, ii, jj);
            struct GenericFunction * f2 = qmarray_get_func(B, ii, jj);
            double diff = generic_function_norm2diff(f1, f2);
            /* printf("Function 1 \n"); */
            /* print_generic_function(f1, 2, NULL, stdout); */
            /* printf("Function 2 \n"); */
            /* print_generic_function(f2, 2, NULL, stdout); */
            /* printf("diff = %G\n", diff); */
            CuAssertDblEquals(tc, 0.0, diff, tol);
        }
    }
    
}

static void 
qmarray_test_equality2(CuTest * tc, struct Qmarray * A,
                       struct Qmarray * B, double tol)
{
    size_t nrowsA = qmarray_get_nrows(A);
    size_t ncolsA = qmarray_get_ncols(A);

    size_t nrowsB = qmarray_get_nrows(B);
    size_t ncolsB = qmarray_get_ncols(B);

    CuAssertIntEquals(tc,nrowsA,nrowsB);
    CuAssertIntEquals(tc,ncolsA,ncolsB);

    for (size_t ii = 0; ii < nrowsA; ii++){
        for (size_t jj = 0; jj < ncolsA; jj++){
            struct GenericFunction *f,*g,*fg=NULL;
            f = qmarray_get_func(A,ii,jj);
            g = qmarray_get_func(B,ii,jj);
            fg = generic_function_daxpby(1,f,-1.0,g);
            double diff = generic_function_norm(fg);
            CuAssertDblEquals(tc,0.0,diff,tol);
            generic_function_free(fg);
        }
    }
}

static void 
qmarray_quasimatrix_test_equality1(CuTest * tc, struct Qmarray * A,
                                   struct Quasimatrix * B, double tol)
{
    size_t nrowsA = qmarray_get_nrows(A);
    size_t ncolsA = qmarray_get_ncols(A);

    size_t ncolsB = quasimatrix_get_size(B);

    CuAssertIntEquals(tc,nrowsA,1);
    CuAssertIntEquals(tc,ncolsA,ncolsB);

    for (size_t ii = 0; ii < ncolsA; ii++){
        struct GenericFunction * temp = NULL;
        double diff;
        struct GenericFunction *f1, *f2;
        f1 = qmarray_get_func(A,0,ii);
        f2 = quasimatrix_get_func(B,ii);
        temp = generic_function_daxpby(1,f1,-1.0,f2);
        diff = generic_function_norm(temp);
        CuAssertDblEquals(tc, 0.0, diff,tol);
        generic_function_free(temp);
    }
}

////////////////////////////////////////////////////////////////////////////////


void Test_qmarray_serialize(CuTest * tc){

    printf("Testing function: (de)qmarray_serialize\n");

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,6);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);
    fwrap_set_func_array(fw,4,func5,NULL);
    fwrap_set_func_array(fw,5,func6,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct Qmarray * A = qmarray_approx1d(3,2,qmopts,fw);

    unsigned char * text = NULL;
    size_t size;
    qmarray_serialize(NULL,A,&size);
    text = malloc(size * sizeof(unsigned char));
    qmarray_serialize(text,A,NULL);
    
    struct Qmarray * B = NULL;
    qmarray_deserialize(text,&B);
    free(text); text = NULL;

    size_t nrows = qmarray_get_nrows(B);
    size_t ncols = qmarray_get_ncols(B);

    CuAssertIntEquals(tc,3,nrows);
    CuAssertIntEquals(tc,2,ncols);

    qmarray_test_equality2(tc,A,B,1e-15);

    qmarray_free(A);
    qmarray_free(B);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
    fwrap_destroy(fw);
}

void Test_qmarray_savetxt(CuTest * tc){

    printf("Testing function: qmarray_savetxt and _loadtxt\n");

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,6);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);
    fwrap_set_func_array(fw,4,func5,NULL);
    fwrap_set_func_array(fw,5,func6,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct Qmarray * A = qmarray_approx1d(3,2,qmopts,fw);

    FILE * fp = fopen("qmarraytest.txt","w+");
    size_t prec = 21;
    qmarray_savetxt(A,fp,prec);
    
    struct Qmarray * B = NULL;
    rewind(fp);
    B = qmarray_loadtxt(fp);
    fclose(fp);

    size_t nrows = qmarray_get_nrows(B);
    size_t ncols = qmarray_get_ncols(B);

    CuAssertIntEquals(tc,3,nrows);
    CuAssertIntEquals(tc,2,ncols);

    qmarray_test_equality2(tc,A,B,1e-15);

    qmarray_free(A);
    qmarray_free(B);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
    fwrap_destroy(fw);
}

void Test_qmarray_orth1d_columns(CuTest * tc)
{
    printf("Testing function: qmarray_orth1d_columns\n");
    
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);

    struct Qmarray * Q = qmarray_orth1d_columns(2,2,qmopts);
    
    size_t nrows = qmarray_get_nrows(Q);
    size_t ncols = qmarray_get_ncols(Q);
    CuAssertIntEquals(tc, 2, nrows);
    CuAssertIntEquals(tc, 2, ncols);

    qmarray_test_col_orth(tc,Q,1e-14);

    qmarray_free(Q); Q = NULL;
    one_approx_opts_free(qmopts);
    ope_opts_free(opts);
}

void Test_qmarray_orth1d_columns_fourier(CuTest * tc)
{
    printf("Testing function: qmarray_orth1d_columns with fourier\n");
    
    struct OpeOpts * opts = ope_opts_alloc(FOURIER);
    ope_opts_set_start(opts, 65);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);

    struct Qmarray * Q = qmarray_orth1d_columns(2,2,qmopts);
    
    size_t nrows = qmarray_get_nrows(Q);
    size_t ncols = qmarray_get_ncols(Q);
    CuAssertIntEquals(tc, 2, nrows);
    CuAssertIntEquals(tc, 2, ncols);

    qmarray_test_col_orth(tc,Q,1e-14);

    qmarray_free(Q); Q = NULL;
    one_approx_opts_free(qmopts);
    ope_opts_free(opts);
}

void Test_qmarray_orth1d_rows_fourier(CuTest * tc)
{
    printf("Testing function: qmarray_orth1d_rows with fourier\n");
    
    struct OpeOpts * opts = ope_opts_alloc(FOURIER);
    ope_opts_set_start(opts, 65);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);

    struct Qmarray * Q = qmarray_orth1d_rows(2,2,qmopts);
    
    size_t nrows = qmarray_get_nrows(Q);
    size_t ncols = qmarray_get_ncols(Q);
    CuAssertIntEquals(tc, 2, nrows);
    CuAssertIntEquals(tc, 2, ncols);

    qmarray_test_row_orth(tc,Q,1e-14);

    qmarray_free(Q); Q = NULL;
    one_approx_opts_free(qmopts);
    ope_opts_free(opts);
}

void Test_qmarray_householder(CuTest * tc){

    printf("Testing function: qmarray_householder (1/4)\n");

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,-1.0,1.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(PIECEWISE, opts);

    struct Qmarray * A = qmarray_approx1d(2,2,qmopts,fw);
    CuAssertIntEquals(tc,2,qmarray_get_nrows(A));
    CuAssertIntEquals(tc,2,qmarray_get_ncols(A));
    /* print_qmarray(A, 1, NULL, stdout); */


    /* printf("\n\n********* copy *****************\n\n"); */
    struct Qmarray * Acopy = qmarray_copy(A);
    CuAssertIntEquals(tc,2,qmarray_get_nrows(Acopy));
    CuAssertIntEquals(tc,2,qmarray_get_ncols(Acopy));
    /* print_qmarray(Acopy, 2, NULL, stdout); */
    qmarray_test_equality1(tc,A,Acopy,3e-14);

    
    /* printf("Do householdering simple\n"); */
    double * R = calloc_double(2*2);
    struct Qmarray * Q = qmarray_householder_simple("QR",A,R,qmopts);

    
    size_t nrows = qmarray_get_nrows(Q);
    size_t ncols = qmarray_get_ncols(Q);
    CuAssertIntEquals(tc,2,nrows);
    CuAssertIntEquals(tc,2,ncols);

    qmarray_test_col_orth(tc,Q,4e-14);
    
    // test equivalence
    struct Qmarray * Anew = qmam(Q,R,2);
    qmarray_test_equality1(tc,Anew,Acopy,4e-14);
    qmarray_free(A);
    qmarray_free(Q);
    free(R);    
    qmarray_free(Acopy);
    qmarray_free(Anew);

    fwrap_destroy(fw);
    pw_poly_opts_free(opts);
    one_approx_opts_free(qmopts);
}

void Test_qmarray_householder2(CuTest * tc){

    printf("Testing function: qmarray_householder (2/4)\n");

   // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);

    struct Qmarray * A = qmarray_approx1d(1,4,qmopts,fw);
    double * R = calloc_double(4*4);
    struct Qmarray * Q = qmarray_householder_simple("QR",A,R,qmopts);
    
    enum function_class fc[4] = {POLYNOMIAL, POLYNOMIAL, POLYNOMIAL, POLYNOMIAL};
    struct Quasimatrix * A2 = quasimatrix_approx1d(4,fw,fc,opts);
    double * R2 = calloc_double(4*4);
    struct Quasimatrix * Q2 = quasimatrix_householder_simple(A2,R2,opts);
    
    CuAssertDblEquals(tc, 0.0, norm2diff(R,R2,16),1e-14);

    struct GenericFunction * temp = NULL;
    double diff;
    struct GenericFunction *f1,*f2;
    f1 = qmarray_get_func(Q,0,0);
    f2 = quasimatrix_get_func(Q2,0);
    temp = generic_function_daxpby(1,f1,-1.0,f2);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff ,1e-14);
    generic_function_free(temp);

    f1 = qmarray_get_func(Q,0,1);
    f2 = quasimatrix_get_func(Q2,1);
    temp = generic_function_daxpby(1,f1,-1.0,f2);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff ,1e-14);
    generic_function_free(temp);
    
    f1 = qmarray_get_func(Q,0,2);
    f2 = quasimatrix_get_func(Q2,2);
    temp = generic_function_daxpby(1,f1,-1.0,f2);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff ,1e-14);
    generic_function_free(temp);

    f1 = qmarray_get_func(Q,0,3);
    f2 = quasimatrix_get_func(Q2,3);
    temp = generic_function_daxpby(1,f1,-1.0,f2);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff ,1e-14);
    generic_function_free(temp);

    qmarray_free(A);
    qmarray_free(Q);
    quasimatrix_free(A2);
    quasimatrix_free(Q2);
    free(R);
    free(R2);
    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
}

void Test_qmarray_householder2_fourier(CuTest * tc){

    printf("Testing function: qmarray_householder_fourier (1/2)\n");

   // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,funcp,NULL);
    fwrap_set_func_array(fw,2,func2p,NULL);
    fwrap_set_func_array(fw,3,func3p,NULL);

    struct OpeOpts * opts = ope_opts_alloc(FOURIER);
    ope_opts_set_lb(opts, 0-1);
    ope_opts_set_ub(opts, 1);
    ope_opts_set_start(opts, 65);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);

    struct Qmarray * A = qmarray_approx1d(1,4,qmopts,fw);
    struct Qmarray * B = qmarray_copy(A);

    double diff = qmarray_norm2diff(A, B);

    CuAssertDblEquals(tc, 0, diff, 1e-14);
    double * R = calloc_double(4*4);
    struct Qmarray * Q = qmarray_householder_simple("QR",A,R,qmopts);
    /* dprint2d_col(4, 4, R); */

    struct Qmarray * Ah= qmam(Q, R, 1);
    diff = qmarray_norm2diff(Ah, B);
    CuAssertDblEquals(tc, 0, diff, 1e-14);
    /* printf("diff = %3.5G\n", diff); */

    /* printf("norm qmarray2 = %3.5G\n", qmarray_norm2(Q)); */
    double inner;
    struct GenericFunction *f1,*f2;

    for (size_t ii = 0; ii < 4; ii++){
        f1 = qmarray_get_func(Q, 0, ii);
        for (size_t jj = 0; jj < 4; jj++){
            f2 = qmarray_get_func(Q, 0, jj);
            inner = generic_function_inner(f1, f2);
            if (ii != jj){
                CuAssertDblEquals(tc, 0.0, inner ,1e-14);
            }
            else{
                CuAssertDblEquals(tc, 1.0, inner ,1e-14);
            }
        }
    }


    qmarray_free(A);
    qmarray_free(B);
    qmarray_free(Ah);
    qmarray_free(Q);
    free(R);
    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
}

void Test_qmarray_householder2_fourier2(CuTest * tc){

    printf("Testing function: qmarray_householder_fourier (2/2)\n");

   // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,funcp,NULL);
    fwrap_set_func_array(fw,2,func2p,NULL);
    fwrap_set_func_array(fw,3,func3p,NULL);

    struct OpeOpts * opts = ope_opts_alloc(FOURIER);
    ope_opts_set_lb(opts, 0-1);
    ope_opts_set_ub(opts, 1);
    ope_opts_set_start(opts, 65);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);

    struct Qmarray * A = qmarray_approx1d(2,2,qmopts,fw);
    struct Qmarray * B = qmarray_copy(A);

    double diff = qmarray_norm2diff(A, B);
    CuAssertDblEquals(tc, 0, diff, 1e-14);
    
    double * R = calloc_double(2*2);
    struct Qmarray * Q = qmarray_householder_simple("QR",A,R,qmopts);

    /* printf("Qmarray rows = %zu\n", Q->nrows); */
    /* printf("Qmarray cols = %zu\n", Q->ncols); */
    /* printf("\n\n"); */
    /* dprint2d_col(2, 2, R); */

    struct Qmarray * Ah= qmam(Q, R, 2);
    diff = qmarray_norm2diff(Ah, B);

    CuAssertDblEquals(tc, 0, diff, 1e-14);


    struct GenericFunction *f1,*f2;
    f1 = qmarray_get_func(Ah, 0, 1);
    f2 = qmarray_get_func(B, 0, 1);    

    diff = generic_function_norm2diff(f1, f2);
    CuAssertDblEquals(tc, 0, diff, 1e-14);

    double inner;
    
    f1 = qmarray_get_func(Q, 0, 0);
    f2 = qmarray_get_func(Q, 0, 1);
    inner = generic_function_inner(f1, f2);

    f1 = qmarray_get_func(Q, 1, 0);
    f2 = qmarray_get_func(Q, 1, 1);
    inner += generic_function_inner(f1, f2);
    CuAssertDblEquals(tc, 0, inner, 1e-15);


    f1 = qmarray_get_func(Q, 0, 0);
    f2 = qmarray_get_func(Q, 1, 0);
    inner = generic_function_inner(f1, f1) + generic_function_inner(f2, f2);
    CuAssertDblEquals(tc, 1, inner, 1e-15);

    f1 = qmarray_get_func(Q, 0, 1);
    f2 = qmarray_get_func(Q, 1, 1);
    inner = generic_function_inner(f1, f1) + generic_function_inner(f2, f2);
    CuAssertDblEquals(tc, 1, inner, 1e-15);

    qmarray_free(A);
    qmarray_free(B);
    qmarray_free(Ah);
    qmarray_free(Q);
    free(R);
    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
}


void Test_qmarray_householder3(CuTest * tc){

    printf("Testing function: qmarray_householder (3/4)\n");

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct PwPolyOpts * pwopts = pw_poly_opts_alloc(LEGENDRE,-1.0,1.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(PIECEWISE,pwopts);
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);

    struct Qmarray * A = qmarray_approx1d(1,4,qmopts,fw);
    size_t N = 100;
    double * xtest = linspace(-1.0, 1.0, N);
    double temp1, temp2, err;
    size_t ii,jj;
    for (ii = 0; ii < 4; ii++){
        fwrap_set_which_eval(fw,ii);
        err = 0.0;
        struct GenericFunction * f = qmarray_get_func(A,0,ii);
        for (jj = 0; jj < N; jj++){
            fwrap_eval(1,xtest+jj,&temp1,fw);
            temp2 = generic_function_1d_eval(f,xtest[jj]);
            err += fabs(temp1-temp2);
        }
        //printf("err= %3.15G\n",err);
        CuAssertDblEquals(tc,0.0,err,1e-6);
    }
    struct Qmarray * Acopy = qmarray_copy(A);
    qmarray_test_equality1(tc,A,Acopy,2e-13);

    double * R = calloc_double(4*4);
    struct Qmarray * Q = qmarray_householder_simple("QR",A,R,qmopts);
    qmarray_test_col_orth(tc,Q,1e-13);
    
    enum function_class fc[4] = {POLYNOMIAL,POLYNOMIAL,POLYNOMIAL,POLYNOMIAL};
    struct Quasimatrix * A2 = quasimatrix_approx1d(4,fw,fc,opts);

    for (ii = 0; ii < 4; ii++){
        fwrap_set_which_eval(fw,ii);
        struct GenericFunction * gf = quasimatrix_get_func(A2,ii);
        err = 0.0;
        for (jj = 0; jj < N; jj++){
            fwrap_eval(1,xtest+jj,&temp1,fw);

            temp2 = generic_function_1d_eval(gf,xtest[jj]);
            err += fabs(temp1-temp2);
        }
        //printf("err= %3.15G\n",err);
        CuAssertDblEquals(tc,0.0,err,1e-11);
    }
    free(xtest); xtest = NULL;

    double * R2 = calloc_double(4*4);
    struct Quasimatrix * Q2 = quasimatrix_householder_simple(A2,R2,opts);
    quasimatrix_test_col_orth(tc,Q2,2e-13);


    CuAssertDblEquals(tc, 0.0, norm2diff(R,R2,16),2e-13);
    qmarray_quasimatrix_test_equality1(tc,Q,Q2,2e-13);

    qmarray_free(A);
    qmarray_free(Acopy);
    qmarray_free(Q);
    quasimatrix_free(A2);
    quasimatrix_free(Q2);
    free(R);
    free(R2);
    fwrap_destroy(fw);
    ope_opts_free(opts);
    pw_poly_opts_free(pwopts);
    one_approx_opts_free(qmopts);
}

void Test_qmarray_householder4(CuTest * tc){

    printf("Testing function: qmarray_householder (4/4)\n");

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func3,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func3,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);

    struct Qmarray * A = qmarray_approx1d(1,4,qmopts,fw);
    double * R = calloc_double(4*4);
    struct Qmarray * Q = qmarray_householder_simple("QR",A,R,qmopts);
    qmarray_test_col_orth(tc,Q,1e-13);

    enum function_class fc[4] = {POLYNOMIAL,POLYNOMIAL,POLYNOMIAL,POLYNOMIAL};
    struct Quasimatrix * A2 = quasimatrix_approx1d(4,fw,fc,opts);
    double * R2 = calloc_double(4*4);
    struct Quasimatrix * Q2 = quasimatrix_householder_simple(A2,R2,opts);
    quasimatrix_test_col_orth(tc,Q2,1e-13);
    
    CuAssertDblEquals(tc, 0.0, norm2diff(R,R2,16),1e-14);
    qmarray_quasimatrix_test_equality1(tc,Q,Q2,1e-15);

    qmarray_free(A);
    qmarray_free(Q);
    quasimatrix_free(A2);
    quasimatrix_free(Q2);
    free(R);
    free(R2);
    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
}

void Test_qmarray_householder_hermite1(CuTest * tc){
    
    // printf("\n\n\n\n\n\n\n\n\n\n");
    printf("Testing function: qmarray_householder for hermite (1)\n");

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func3,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func3,NULL);

    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);

    struct Qmarray* T = qmarray_orth1d_columns(2,2,qmopts);
    qmarray_test_col_orth(tc,T,1e-13);

    struct Qmarray * A = qmarray_approx1d(2,2,qmopts,fw);
    double * R = calloc_double(2*2);
    struct Qmarray * Acopy = qmarray_copy(A);
    struct Qmarray * Q = qmarray_householder_simple("QR",A,R,qmopts);
    qmarray_test_col_orth(tc,Q,1e-13);

    struct Qmarray * Anew = qmam(Q,R,2);
    qmarray_test_equality2(tc,Acopy,Anew,1e-14);

    qmarray_free(Anew); Anew = NULL;
    free(R); R = NULL;
    qmarray_free(Q); Q = NULL;
    qmarray_free(A);
    qmarray_free(Acopy);
    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
    qmarray_free(T);
}

void Test_qmarray_qr1(CuTest * tc)
{
    printf("Testing function: qmarray_qr (1/3)\n");
    
    double lb = -2.0;
    double ub = 3.0;
    size_t r1 = 5;
    size_t r2 = 7;
    size_t maxorder = 10;

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);

    struct Qmarray * A = qmarray_poly_randu(LEGENDRE,r1,r2,
                                            maxorder,lb,ub);
    struct Qmarray * Acopy = qmarray_copy(A);

    struct Qmarray * Q = NULL;
    double * R = NULL;
    qmarray_qr(A,&Q,&R,qmopts);
    qmarray_test_col_orth(tc,Q,1e-13);

    struct Qmarray * QR = qmam(Q,R,r2);
    double diff = qmarray_norm2diff(QR,Acopy);
    CuAssertDblEquals(tc,0.0,diff*diff,1e-14);
    
    qmarray_free(A); A = NULL;
    qmarray_free(Acopy); Acopy = NULL;
    qmarray_free(Q); Q = NULL;
    qmarray_free(QR); QR = NULL;
    free(R); R = NULL;
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
}

void Test_qmarray_qr2(CuTest * tc)
{
    printf("Testing function: qmarray_qr (2/3)\n");
    
    double lb = -2.0;
    double ub = 3.0;
    size_t r1 = 7;
    size_t r2 = 5;
    size_t maxorder = 10;

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);

    struct Qmarray * A = qmarray_poly_randu(LEGENDRE,r1,r2,
                                            maxorder,lb,ub);
    struct Qmarray * Acopy = qmarray_copy(A);

    struct Qmarray * Q = NULL;
    double * R = NULL;
    qmarray_qr(A,&Q,&R,qmopts);
    qmarray_test_col_orth(tc,Q,1e-13);

    struct Qmarray * QR = qmam(Q,R,r2);
    double diff = qmarray_norm2diff(QR,Acopy);
    CuAssertDblEquals(tc,0.0,diff*diff,1e-14);
    
    qmarray_free(A); A = NULL;
    qmarray_free(Acopy); Acopy = NULL;
    qmarray_free(Q); Q = NULL;
    qmarray_free(QR); QR = NULL;
    free(R); R = NULL;
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
}

void Test_qmarray_qr3(CuTest * tc){

    printf("Testing function: qmarray_qr (3/3)\n");

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func2,NULL);
    fwrap_set_func_array(fw,3,func3,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);

    struct Qmarray * A = qmarray_approx1d(1,4,qmopts,fw);
    struct Qmarray * Acopy = qmarray_copy(A);

    struct Qmarray * Q = NULL;
    double * R = NULL;
    qmarray_qr(A,&Q,&R,qmopts);
    qmarray_test_col_orth(tc,Q,1e-13);

    struct Qmarray * QR = qmam(Q,R,4);
    double diff = qmarray_norm2diff(QR,Acopy);
    CuAssertDblEquals(tc,0.0,diff*diff,1e-14);

    free(R); R = NULL;
    qmarray_free(Q); Q = NULL;
    qmarray_free(A);
    qmarray_free(QR); 
    qmarray_free(Acopy);
    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);

}

void Test_qmarray_lq(CuTest * tc)
{
    printf("Testing function: qmarray_lq (1/3)\n");
    
    double lb = -2.0;
    double ub = 3.0;
    size_t r1 = 5;
    size_t r2 = 7;
    size_t maxorder = 10;

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);

    struct Qmarray * A = qmarray_poly_randu(LEGENDRE,r1,r2,
                                            maxorder,lb,ub);
    struct Qmarray * Acopy = qmarray_copy(A);

    struct Qmarray * Q = NULL;
    double * L = NULL;
    qmarray_lq(A,&Q,&L,qmopts);
    qmarray_test_row_orth(tc,Q,1e-14);

    struct Qmarray * LQ = mqma(L,Q,r1);
    double diff = qmarray_norm2diff(LQ,Acopy);
    CuAssertDblEquals(tc,0.0,diff*diff,1e-14);
    
    qmarray_free(A); A = NULL;
    qmarray_free(Acopy); Acopy = NULL;
    qmarray_free(Q); Q = NULL;
    qmarray_free(LQ); LQ = NULL;
    free(L); L = NULL;
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
}


void Test_qmarray_householder_rows(CuTest * tc){

    printf("Testing function: qmarray_householder_rows \n");

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);

    struct Qmarray * A = qmarray_approx1d(2,2,qmopts,fw);
    struct Qmarray * Acopy = qmarray_copy(A);
    qmarray_test_equality1(tc,A,Acopy,1e-15);
   
    double * R = calloc_double(2*2);
    struct Qmarray * Q = qmarray_householder_simple("LQ",A,R,qmopts);

    size_t nrows = qmarray_get_nrows(Q);
    size_t ncols = qmarray_get_ncols(Q);
    CuAssertIntEquals(tc,2,nrows);
    CuAssertIntEquals(tc,2,ncols);
    qmarray_test_row_orth(tc,Q,1e-14);

     // testt equivalence
    struct Qmarray * Anew = mqma(R,Q,2);
    qmarray_test_equality2(tc,Anew,Acopy,1e-14);
    

    qmarray_free(Anew);
    qmarray_free(A);
    qmarray_free(Q);
    qmarray_free(Acopy);
    free(R);
    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
}

void Test_qmarray_householder_rows_hermite(CuTest * tc){

    printf("Testing function: qmarray_householder_rows with hermite polynomials \n");
    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);

    struct Qmarray * A = qmarray_approx1d(2,2,qmopts,fw);
    struct Qmarray * Acopy = qmarray_copy(A);

    double * R = calloc_double(2*2);
    struct Qmarray * Q = qmarray_householder_simple("LQ",A,R,qmopts);


    size_t nrows = qmarray_get_nrows(Q);
    size_t ncols = qmarray_get_ncols(Q);
    CuAssertIntEquals(tc,2,nrows);
    CuAssertIntEquals(tc,2,ncols);
    qmarray_test_row_orth(tc,Q,1e-15);

     // testt equivalence
    struct Qmarray * Anew = mqma(R,Q,2);
    qmarray_test_equality2(tc,Anew,Acopy,1e-13);

    qmarray_free(Anew); Anew = NULL;
    free(R); R = NULL;
    qmarray_free(Q); Q = NULL;
    qmarray_free(A); A = NULL;
    qmarray_free(Acopy); Acopy = NULL;
    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
}

void Test_qmarray_householder_constelm(CuTest * tc){
    
    // printf("\n\n\n\n\n\n\n\n\n\n");
    printf("Testing function: qmarray_householder for constelm\n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    double * x = linspace(-1.0,1.0,5);
    struct ConstElemExpAopts * opts = const_elem_exp_aopts_alloc(5,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(CONSTELM,opts);

    struct Qmarray * T = qmarray_orth1d_columns(2,2,qmopts);
    qmarray_test_col_orth(tc,T,1e-14);
    
    struct Qmarray * A = qmarray_approx1d(2,2,qmopts,fw);
    struct Qmarray * Acopy = qmarray_copy(A);

    size_t nc = 2;
    double * R = calloc_double(nc*nc);
    struct Qmarray * Q = qmarray_householder_simple("QR",Acopy,R,qmopts);
    qmarray_test_col_orth(tc,Q,1e-14);

    struct Qmarray * Anew = qmam(Q,R,nc);
    qmarray_test_equality2(tc,A,Anew,1e-15);

    qmarray_free(T);
    free(x); x = NULL;
    const_elem_exp_aopts_free(opts);
    one_approx_opts_free(qmopts);
    fwrap_destroy(fw);
    qmarray_free(Anew); Anew = NULL;
    free(R); R = NULL;
    qmarray_free(Q); Q = NULL;
    qmarray_free(A);
    qmarray_free(Acopy);
}

void Test_qmarray_householder_rowsconstelm(CuTest * tc){
    
    // printf("\n\n\n\n\n\n\n\n\n\n");
    printf("Testing function: qmarray_householder LQ for constelm\n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    double *x = linspace(-1.0,1.0,5);
    struct ConstElemExpAopts * opts = const_elem_exp_aopts_alloc(5,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(CONSTELM,opts);

    struct Qmarray * T = qmarray_orth1d_rows(2,2,qmopts);
    qmarray_test_row_orth(tc,T,1e-14);
    
    struct Qmarray * A = qmarray_approx1d(2,2,qmopts,fw);
    struct Qmarray * Acopy = qmarray_copy(A);

    size_t nr = 2;
    double * R = calloc_double(nr*nr);
    struct Qmarray * Q = qmarray_householder_simple("LQ",Acopy,R,qmopts);
    qmarray_test_row_orth(tc,Q,1e-14);

    struct Qmarray * Anew = mqma(R,Q,nr);
    qmarray_test_equality2(tc,A,Anew,1e-15);

    free(x); x = NULL;
    qmarray_free(T);
    const_elem_exp_aopts_free(opts);
    one_approx_opts_free(qmopts);
    fwrap_destroy(fw);
    qmarray_free(Anew); Anew = NULL;
    free(R); R = NULL;
    qmarray_free(Q); Q = NULL;
    qmarray_free(A);
    qmarray_free(Acopy);
}

void Test_qmarray_householder_linelm(CuTest * tc){
    
    // printf("\n\n\n\n\n\n\n\n\n\n");
    printf("Testing function: qmarray_householder for linelm\n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    double *x = linspace(-1.0,1.0,5);
    struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc(5,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(LINELM,opts);

    struct Qmarray * T = qmarray_orth1d_columns(2,2,qmopts);
    qmarray_test_col_orth(tc,T,1e-14);
    
    struct Qmarray * A = qmarray_approx1d(2,2,qmopts,fw);
    struct Qmarray * Acopy = qmarray_copy(A);

    size_t nc = 2;
    double * R = calloc_double(nc*nc);
    struct Qmarray * Q = qmarray_householder_simple("QR",Acopy,R,qmopts);
    qmarray_test_col_orth(tc,Q,1e-14);

    struct Qmarray * Anew = qmam(Q,R,nc);
    qmarray_test_equality2(tc,A,Anew,1e-15);

    qmarray_free(T);
    free(x); x = NULL;
    lin_elem_exp_aopts_free(opts);
    one_approx_opts_free(qmopts);
    fwrap_destroy(fw);
    qmarray_free(Anew); Anew = NULL;
    free(R); R = NULL;
    qmarray_free(Q); Q = NULL;
    qmarray_free(A);
    qmarray_free(Acopy);
}

void Test_qmarray_householder_rowslinelm(CuTest * tc){
    
    // printf("\n\n\n\n\n\n\n\n\n\n");
    printf("Testing function: qmarray_householder LQ for linelm\n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    double *x = linspace(-1.0,1.0,5);
    struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc(5,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(LINELM,opts);

    struct Qmarray * T = qmarray_orth1d_rows(2,2,qmopts);
    qmarray_test_row_orth(tc,T,1e-14);
    
    struct Qmarray * A = qmarray_approx1d(2,2,qmopts,fw);
    struct Qmarray * Acopy = qmarray_copy(A);

    size_t nr = 2;
    double * R = calloc_double(nr*nr);
    struct Qmarray * Q = qmarray_householder_simple("LQ",Acopy,R,qmopts);
    qmarray_test_row_orth(tc,Q,1e-14);

    struct Qmarray * Anew = mqma(R,Q,nr);
    qmarray_test_equality2(tc,A,Anew,1e-15);

    free(x); x = NULL;
    qmarray_free(T);
    lin_elem_exp_aopts_free(opts);
    one_approx_opts_free(qmopts);
    fwrap_destroy(fw);
    qmarray_free(Anew); Anew = NULL;
    free(R); R = NULL;
    qmarray_free(Q); Q = NULL;
    qmarray_free(A);
    qmarray_free(Acopy);
}

void Test_qmarray_svd(CuTest * tc){

    printf("Testing function: qmarray_svd \n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct Qmarray * A = qmarray_approx1d(2,2,qmopts,fw);
    struct Qmarray * Acopy = qmarray_copy(A);

    double * vt = calloc_double(2*2);
    double * s = calloc_double(2);
    struct Qmarray * Q = NULL;
    qmarray_svd(A,&Q,s,vt,qmopts);

    size_t nrows = qmarray_get_nrows(Q);
    size_t ncols = qmarray_get_ncols(Q);
    CuAssertIntEquals(tc,2,nrows);
    CuAssertIntEquals(tc,2,ncols);

    // test orthogonality
    qmarray_test_col_orth(tc,Q,1e-14);
    
     // testt equivalence
    double * comb = calloc_double(2*2);
    double * sdiag = diag(2, s);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, 2, 2, 2, 1.0,
                    sdiag, 2, vt, 2, 0.0, comb, 2);
    //comb = dgemm
    free(s);
    free(sdiag);
    free(vt);

    //printf("on the quasimatrix portion\n");
    struct Qmarray * Anew = qmam(Q,comb,2);
    free(comb);
    qmarray_test_equality2(tc,Acopy,Anew,1e-12);
    

    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
    qmarray_free(Anew);
    qmarray_free(A);
    qmarray_free(Q);
    qmarray_free(Acopy);
}

void Test_qmarray_lu1d(CuTest * tc){

    printf("Testing function: qmarray_lu1d (1/2)\n");
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct Qmarray * A = qmarray_approx1d(2,2,qmopts,fw);
    struct Qmarray * Acopy = qmarray_copy(A);


    struct Qmarray * L = qmarray_alloc(2,2);
    double * U = calloc_double(2*2);
    size_t * pivi = calloc_size_t(2);
    double * pivx = calloc_double(2);
    qmarray_lu1d(A,L,U,pivi,pivx,qmopts,NULL);
    

    struct GenericFunction * lf = qmarray_get_func(L,pivi[0],1);
    double eval = generic_function_1d_eval(lf,pivx[0]);
    CuAssertDblEquals(tc, 0.0, eval, 1e-13);
    
    /* eval = generic_function_1d_eval(L->funcs[2+ pivi[0]], pivx[0]); */
    /* //printf("eval = %G\n",eval); */
    /* CuAssertDblEquals(tc, 0.0, eval, 1e-13); */
    
    struct Qmarray * Comb = qmam(L,U,2);
    double difff = qmarray_norm2diff(Comb,Acopy);
    CuAssertDblEquals(tc,difff,0,1e-14);
    
    //exit(1);
    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
    qmarray_free(Acopy);
    qmarray_free(A);
    qmarray_free(Comb);
    qmarray_free(L);
    free(U);
    free(pivx);
    free(pivi);
}

void Test_qmarray_lu1d2(CuTest * tc){

    printf("Testing function: qmarray_lu1d (2/2)\n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,6);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func4,NULL);
    fwrap_set_func_array(fw,2,func,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);
    fwrap_set_func_array(fw,4,func5,NULL);
    fwrap_set_func_array(fw,5,func6,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct Qmarray * A = qmarray_approx1d(2,3,qmopts,fw);
    struct Qmarray * Acopy = qmarray_copy(A);

    struct Qmarray * L = qmarray_alloc(2,3);
    double * U = calloc_double(3*3);
    size_t * pivi = calloc_size_t(3);
    double * pivx = calloc_double(3);
    qmarray_lu1d(A,L,U,pivi,pivx,qmopts,NULL);
    

    // check pivots
    size_t ii,jj;
    double eval;
    struct GenericFunction * lf;
    for (ii = 0; ii < 3; ii++){
        for (jj = 0; jj < ii; jj++){
            /* eval = generic_function_1d_eval(L->funcs[2*ii+pivi[jj]], pivx[jj]); */
            lf = qmarray_get_func(L,pivi[jj],ii);
            eval = generic_function_1d_eval(lf,pivx[jj]);
            CuAssertDblEquals(tc,0.0,eval,1e-14);
        }

        /* eval = generic_function_1d_eval(L->funcs[2*ii+pivi[ii]], pivx[ii]); */
        lf = qmarray_get_func(L,pivi[ii],ii);
        eval = generic_function_1d_eval(lf,pivx[ii]);
        CuAssertDblEquals(tc,1.0,eval,1e-14);
    }
    
    struct Qmarray * Comb = qmam(L,U,3);
    double difff = qmarray_norm2diff(Comb,Acopy);
    CuAssertDblEquals(tc,difff,0,1e-13);
    
    //exit(1);
    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
    qmarray_free(Acopy);
    qmarray_free(A);
    qmarray_free(Comb);
    qmarray_free(L);
    free(U);
    free(pivx);
    free(pivi);
}

void Test_qmarray_lu1d_hermite(CuTest * tc){

    printf("Testing function: qmarray_lu1d with hermite (1)\n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,6);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func4,NULL);
    fwrap_set_func_array(fw,2,func6,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);
    fwrap_set_func_array(fw,4,func5,NULL);
    fwrap_set_func_array(fw,5,func3,NULL);

    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct Qmarray * A = qmarray_approx1d(2,3,qmopts,fw);
    struct Qmarray * Acopy = qmarray_copy(A);

    struct Qmarray * L = qmarray_alloc(2,3);
    double * U = calloc_double(3*3);
    size_t * pivi = calloc_size_t(3);
    double * pivx = calloc_double(3);
    size_t nopt = 300;
    double * xopt = linspace(-10.0,10.0,nopt);
    struct c3Vector * c3v = c3vector_alloc(nopt,xopt);
    qmarray_lu1d(A,L,U,pivi,pivx,qmopts,c3v);
    free(xopt); xopt = NULL;
    c3vector_free(c3v); c3v = NULL;

    // check pivots
    size_t ii,jj;
    double eval;
    struct GenericFunction * lf;
    for (ii = 0; ii < 3; ii++){
        for (jj = 0; jj < ii; jj++){
            lf = qmarray_get_func(L,pivi[jj],ii);
            eval = generic_function_1d_eval(lf,pivx[jj]);
            CuAssertDblEquals(tc,0.0,eval,1e-14);
        }

        lf = qmarray_get_func(L,pivi[ii],ii);
        eval = generic_function_1d_eval(lf,pivx[ii]);
        CuAssertDblEquals(tc,1.0,eval,1e-14);
    }
    
    struct Qmarray * Comb = qmam(L,U,3);
    double difff = qmarray_norm2diff(Comb,Acopy);
    CuAssertDblEquals(tc,difff,0,1e-5);
    
    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
    qmarray_free(Acopy);
    qmarray_free(A);
    qmarray_free(Comb);
    qmarray_free(L);
    free(U);
    free(pivx);
    free(pivi);
}

void Test_qmarray_lu1d_constelm(CuTest * tc){

    printf("Testing function: qmarray_lu1d with constelm \n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,6);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func4,NULL);
    fwrap_set_func_array(fw,2,func,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);
    fwrap_set_func_array(fw,4,func5,NULL);
    fwrap_set_func_array(fw,5,func6,NULL);

    double * x = linspace(-1.0,1.0,10);
    struct ConstElemExpAopts * opts = const_elem_exp_aopts_alloc(10,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(CONSTELM,opts);
    
    struct Qmarray * A = qmarray_approx1d(2,3,qmopts,fw);
    struct Qmarray * L = qmarray_alloc(2,3);
    struct Qmarray * Acopy = qmarray_copy(A);

    double * U = calloc_double(3*3);
    size_t * pivi = calloc_size_t(3);
    double * pivx = calloc_double(3);
    qmarray_lu1d(A,L,U,pivi,pivx,qmopts,NULL);
    
    // check pivots
    size_t ii,jj;
    double eval;
    struct GenericFunction * lf;
    for (ii = 0; ii < 3; ii++){
        for (jj = 0; jj < ii; jj++){
            lf = qmarray_get_func(L,pivi[jj],ii);
            eval = generic_function_1d_eval(lf,pivx[jj]);
            CuAssertDblEquals(tc,0.0,eval,1e-14);
        }

        lf = qmarray_get_func(L,pivi[ii],ii);
        eval = generic_function_1d_eval(lf,pivx[ii]);
        CuAssertDblEquals(tc,1.0,eval,1e-14);
    }

    struct Qmarray * Comb = qmam(L,U,3);
    double difff = qmarray_norm2diff(Comb,Acopy);
    CuAssertDblEquals(tc,difff,0,1e-13);

    fwrap_destroy(fw);
    free(x);
    one_approx_opts_free(qmopts);
    const_elem_exp_aopts_free(opts);
    qmarray_free(Acopy);
    qmarray_free(A);
    qmarray_free(Comb);
    qmarray_free(L);
    free(U);
    free(pivx);
    free(pivi);
}

void Test_qmarray_lu1d_linelm(CuTest * tc){

    printf("Testing function: qmarray_lu1d with linelm \n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,6);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func4,NULL);
    fwrap_set_func_array(fw,2,func,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);
    fwrap_set_func_array(fw,4,func5,NULL);
    fwrap_set_func_array(fw,5,func6,NULL);

    double * x = linspace(-1.0,1.0,10);
    struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc(10,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(LINELM,opts);
    
    struct Qmarray * A = qmarray_approx1d(2,3,qmopts,fw);
    struct Qmarray * L = qmarray_alloc(2,3);
    struct Qmarray * Acopy = qmarray_copy(A);

    double * U = calloc_double(3*3);
    size_t * pivi = calloc_size_t(3);
    double * pivx = calloc_double(3);
    qmarray_lu1d(A,L,U,pivi,pivx,qmopts,NULL);
    
    // check pivots
    size_t ii,jj;
    double eval;
    struct GenericFunction * lf;
    for (ii = 0; ii < 3; ii++){
        for (jj = 0; jj < ii; jj++){
            lf = qmarray_get_func(L,pivi[jj],ii);
            eval = generic_function_1d_eval(lf,pivx[jj]);
            CuAssertDblEquals(tc,0.0,eval,1e-14);
        }

        lf = qmarray_get_func(L,pivi[ii],ii);
        eval = generic_function_1d_eval(lf,pivx[ii]);
        CuAssertDblEquals(tc,1.0,eval,1e-14);
    }

    struct Qmarray * Comb = qmam(L,U,3);
    double difff = qmarray_norm2diff(Comb,Acopy);
    CuAssertDblEquals(tc,difff,0,1e-13);

    fwrap_destroy(fw);
    free(x);
    one_approx_opts_free(qmopts);
    lin_elem_exp_aopts_free(opts);
    qmarray_free(Acopy);
    qmarray_free(A);
    qmarray_free(Comb);
    qmarray_free(L);
    free(U);
    free(pivx);
    free(pivi);
}

void Test_qmarray_maxvol1d(CuTest * tc){

    printf("Testing function: qmarray_maxvol1d (1/2) \n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,6);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);
    fwrap_set_func_array(fw,4,func5,NULL);
    fwrap_set_func_array(fw,5,func6,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct Qmarray * A = qmarray_approx1d(3,2,qmopts,fw);
    
    double * Asinv = calloc_double(2*2);
    size_t * pivi = calloc_size_t(2);
    double * pivx= calloc_double(2);
    qmarray_maxvol1d(A,Asinv,pivi,pivx,qmopts,NULL);
     
    struct Qmarray * B = qmam(A,Asinv,2);
    double maxval, maxloc;
    size_t maxrow, maxcol;
    qmarray_absmax1d(B,&maxloc,&maxrow, &maxcol, &maxval,NULL);
    //printf("Less = %d", 1.0+1e-2 > maxval);
    CuAssertIntEquals(tc, 1, (1.0+1e-2) > maxval);
    qmarray_free(B);

    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
    qmarray_free(A);
    free(Asinv);
    free(pivx);
    free(pivi);
}

void Test_qmarray_maxvol1d2(CuTest * tc){

    printf("Testing function: qmarray_maxvol1d (2/2) \n");
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,6);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);
    fwrap_set_func_array(fw,4,func4,NULL);
    fwrap_set_func_array(fw,5,func4,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct Qmarray * A = qmarray_approx1d(1,6,qmopts,fw);

    double * Asinv = calloc_double(6*6);
    size_t * pivi = calloc_size_t(6);
    double * pivx= calloc_double(6);
    qmarray_maxvol1d(A,Asinv,pivi,pivx,qmopts,NULL);
    
    struct Qmarray * B = qmam(A,Asinv,6);
    double maxval, maxloc;
    size_t maxrow, maxcol;
    qmarray_absmax1d(B,&maxloc,&maxrow, &maxcol, &maxval,NULL);
    //printf("Less = %d", 1.0+1e-2 > maxval);
    CuAssertIntEquals(tc, 1, (1.0+1e-2) > maxval);
    qmarray_free(B);

    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
    qmarray_free(A);
    free(Asinv);
    free(pivx);
    free(pivi);
}

void Test_qmarray_maxvol1d_hermite1(CuTest * tc){

    printf("Testing function: qmarray_maxvol1d with hermite poly (1) \n");
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,6);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);
    fwrap_set_func_array(fw,4,func4,NULL);
    fwrap_set_func_array(fw,5,func4,NULL);

    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct Qmarray * A = qmarray_approx1d(1,6,qmopts,fw);

    
    double * Asinv = calloc_double(6*6);
    size_t * pivi = calloc_size_t(6);
    double * pivx= calloc_double(6);
    size_t nopt = 100;
    double * xopt = linspace(-5.0,5.0,nopt);
    struct c3Vector * c3v = c3vector_alloc(nopt,xopt);
    qmarray_maxvol1d(A,Asinv,pivi,pivx,qmopts,c3v);
     
    /* printf("pivots at = \n"); */
    /* iprint_sz(6,pivi); */
    /* dprint(6,pivx); */

    struct Qmarray * B = qmam(A,Asinv,6);
    double maxval, maxloc;
    size_t maxrow, maxcol;
    qmarray_absmax1d(B,&maxloc,&maxrow,&maxcol,&maxval,c3v);
    /* printf("maxval = %G\n",maxval); */
    /* printf("Less = %d", 1.0+1e-2 > maxval); */
    CuAssertIntEquals(tc, 1, (1.0+1e-2) > maxval);

    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
    free(xopt); xopt = NULL;
    c3vector_free(c3v); c3v = NULL;
    qmarray_free(B);
    qmarray_free(A);
    free(Asinv);
    free(pivx);
    free(pivi);
}

void Test_qmarray_maxvol1d_constelm(CuTest * tc){

    printf("Testing function: qmarray_maxvol1d constelm (1)\n");
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,6);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func4,NULL);
    fwrap_set_func_array(fw,2,func,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);
    fwrap_set_func_array(fw,4,func5,NULL);
    fwrap_set_func_array(fw,5,func6,NULL);

    double * x = linspace(-1.0,1.0,10);
    struct ConstElemExpAopts * opts = const_elem_exp_aopts_alloc(10,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(CONSTELM,opts);
    struct Qmarray * A = qmarray_approx1d(3,2,qmopts,fw);

    unsigned char * text = NULL;
    size_t size;
    qmarray_serialize(NULL,A,&size);
    text = malloc(size * sizeof(unsigned char));
    qmarray_serialize(text,A,NULL);
    
    struct Qmarray * C = NULL;
    qmarray_deserialize(text,&C);
    free(text); text = NULL;

    double diff = qmarray_norm2diff(A,C);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    qmarray_free(C); C = NULL;

    
    double * Asinv = calloc_double(2*2);
    size_t * pivi = calloc_size_t(2);
    double * pivx= calloc_double(2);
    qmarray_maxvol1d(A,Asinv,pivi,pivx,qmopts,NULL);
     
    /*
    printf("pivots at = \n");
    iprint_sz(3,pivi);
    dprint(3,pivx);
    */

    struct Qmarray * B = qmam(A,Asinv,2);
    double maxval, maxloc;
    size_t maxrow, maxcol;
    qmarray_absmax1d(B,&maxloc,&maxrow, &maxcol, &maxval,NULL);
    //printf("Less = %d", 1.0+1e-2 > maxval);
    CuAssertIntEquals(tc, 1, (1.0+1e-2) > maxval);


    fwrap_destroy(fw);
    one_approx_opts_free(qmopts);
    const_elem_exp_aopts_free(opts);
    free(x); x= NULL;
    qmarray_free(B);
    qmarray_free(A);
    free(Asinv);
    free(pivx);
    free(pivi);
}

void Test_qmarray_maxvol1d_linelm(CuTest * tc){

    printf("Testing function: qmarray_maxvol1d linelm (1)\n");
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,6);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func4,NULL);
    fwrap_set_func_array(fw,2,func,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);
    fwrap_set_func_array(fw,4,func5,NULL);
    fwrap_set_func_array(fw,5,func6,NULL);

    double * x = linspace(-1.0,1.0,10);
    struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc(10,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(LINELM,opts);
    struct Qmarray * A = qmarray_approx1d(3,2,qmopts,fw);

    unsigned char * text = NULL;
    size_t size;
    qmarray_serialize(NULL,A,&size);
    text = malloc(size * sizeof(unsigned char));
    qmarray_serialize(text,A,NULL);
    
    struct Qmarray * C = NULL;
    qmarray_deserialize(text,&C);
    free(text); text = NULL;

    double diff = qmarray_norm2diff(A,C);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    qmarray_free(C); C = NULL;

    
    double * Asinv = calloc_double(2*2);
    size_t * pivi = calloc_size_t(2);
    double * pivx= calloc_double(2);
    qmarray_maxvol1d(A,Asinv,pivi,pivx,qmopts,NULL);
     
    /*
    printf("pivots at = \n");
    iprint_sz(3,pivi);
    dprint(3,pivx);
    */

    struct Qmarray * B = qmam(A,Asinv,2);
    double maxval, maxloc;
    size_t maxrow, maxcol;
    qmarray_absmax1d(B,&maxloc,&maxrow, &maxcol, &maxval,NULL);
    //printf("Less = %d", 1.0+1e-2 > maxval);
    CuAssertIntEquals(tc, 1, (1.0+1e-2) > maxval);


    fwrap_destroy(fw);
    one_approx_opts_free(qmopts);
    lin_elem_exp_aopts_free(opts);
    free(x); x= NULL;
    qmarray_free(B);
    qmarray_free(A);
    free(Asinv);
    free(pivx);
    free(pivi);
}

void Test_fast_mat_kron(CuTest * tc)
{

    printf("Testing Function: fast_mat_kron \n");
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;
    size_t r11 = 5;
    size_t r12 = 6;
    size_t r21 = 7;
    size_t r22 = 8;
    size_t k = 5;
    double diff; 
    
    struct Qmarray * mat1 = qmarray_poly_randu(LEGENDRE,r11,r12,
                                               maxorder,lb,ub);
    struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,
                                               maxorder,lb,ub);

    double * mat = drandu(k*r11*r21);

    struct Qmarray * mat3 = qmarray_kron(mat1,mat2);
    struct Qmarray * shouldbe = mqma(mat,mat3,k);
    struct Qmarray * is = qmarray_mat_kron(k,mat,mat1,mat2);

    diff = qmarray_norm2diff(shouldbe,is);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    free(mat); mat = NULL;
    qmarray_free(mat1); mat1 = NULL;
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(mat3); mat3 = NULL;
    qmarray_free(shouldbe); shouldbe = NULL;
    qmarray_free(is); is = NULL;
}

void Test_fast_kron_mat(CuTest * tc)
{

    printf("Testing Function: fast_kron_mat \n");
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;
    size_t r11 = 3;
    size_t r12 = 4;
    size_t r21 = 5;
    size_t r22 = 6;
    size_t k = 2;
    double diff; 
    
    struct Qmarray * mat1 = 
        qmarray_poly_randu(LEGENDRE,r11,r12,maxorder,lb,ub);
    struct Qmarray * mat2 = 
        qmarray_poly_randu(LEGENDRE,r21,r22,maxorder,lb,ub);

    double * mat = drandu(k*r12*r22);

    struct Qmarray * mat3 = qmarray_kron(mat1,mat2);
    struct Qmarray * shouldbe = qmam(mat3,mat,k);
    struct Qmarray * is = qmarray_kron_mat(k,mat,mat1,mat2);

    diff = qmarray_norm2diff(shouldbe,is);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    free(mat); mat = NULL;
    qmarray_free(mat1); mat1 = NULL;
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(mat3); mat3 = NULL;
    qmarray_free(shouldbe); shouldbe = NULL;
    qmarray_free(is); is = NULL;
}

void Test_block_kron_mat1(CuTest * tc)
{
    printf("Testing Function: block_kron_mat (1/6) \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;

    size_t nblocks = 5;
    size_t rl1[5] = {3, 6, 9, 1, 6};
    size_t rl2[5] = {2, 4, 2, 5, 3};
    size_t sum_rl1 = 0;
    size_t sum_rl2 = 0;
    struct Qmarray * mat1[5];
    

    size_t r21 = 7;
    size_t r22 = 3;
    size_t k = 8;
    double diff; 
    

    size_t ii;
    for (ii = 0; ii < nblocks; ii++){
        sum_rl1 += rl1[ii];
        sum_rl2 += rl2[ii];
        mat1[ii] = qmarray_poly_randu(LEGENDRE,rl1[ii],rl2[ii],
                                      maxorder,lb,ub);
    }
    struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,
                                               maxorder,lb,ub);
    double * mat = drandu(k*sum_rl1*r21);

    struct Qmarray * is = qmarray_alloc(k, sum_rl2* r22);
    qmarray_block_kron_mat('D',1,nblocks,mat1,mat2,k,mat,is);


    struct Qmarray * big1 = qmarray_blockdiag(mat1[0],mat1[1]);
    struct Qmarray * big2 = qmarray_blockdiag(big1,mat1[2]);
    struct Qmarray * big3 = qmarray_blockdiag(big2,mat1[3]);
    struct Qmarray * big4 = qmarray_blockdiag(big3,mat1[4]);

    struct Qmarray * mid = qmarray_kron(big4, mat2);
    struct Qmarray * shouldbe = mqma(mat,mid,k);

    struct Qmarray * is2 = qmarray_mat_kron(k,mat,big4,mat2);
    diff = qmarray_norm2diff(shouldbe,is2);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    diff = qmarray_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff,1e-10);

    qmarray_free(big1);
    qmarray_free(big2);
    qmarray_free(big3);
    qmarray_free(big4);
    qmarray_free(mid);
    qmarray_free(shouldbe);
    qmarray_free(is2);

    free(mat); mat = NULL;
    for (ii = 0; ii < nblocks; ii++){
        qmarray_free(mat1[ii]); mat1[ii] = NULL;
    }
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(is); is = NULL;

    //qmarray_free(mat3); mat3 = NULL;
    //qmarray_free(shouldbe); shouldbe = NULL;
}

void Test_block_kron_mat2(CuTest * tc)
{
    printf("Testing Function: block_kron_mat (2/6) \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;

    size_t nblocks = 5;
    size_t rl1[5] = {4, 4, 4, 4, 4};
    size_t rl2[5] = {2, 4, 2, 5, 3};
    size_t sum_rl1 = 4;
    size_t sum_rl2 = 0;
    struct Qmarray * mat1[5];
    

    size_t r21 = 7;
    size_t r22 = 3;
    size_t k = 8;
    double diff; 
    

    size_t ii;
    for (ii = 0; ii < nblocks; ii++){
        sum_rl2 += rl2[ii];
        mat1[ii] = qmarray_poly_randu(LEGENDRE,rl1[ii],rl2[ii],maxorder,lb,ub);
    }
    struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,
                                               maxorder,lb,ub);
    double * mat = drandu(k*sum_rl1*r21);

    struct Qmarray * is = qmarray_alloc(k, sum_rl2* r22);
    qmarray_block_kron_mat('R',1,nblocks,mat1,mat2,k,mat,is);


    struct Qmarray * big1 = qmarray_stackh(mat1[0],mat1[1]);
    struct Qmarray * big2 = qmarray_stackh(big1,mat1[2]);
    struct Qmarray * big3 = qmarray_stackh(big2,mat1[3]);
    struct Qmarray * big4 = qmarray_stackh(big3,mat1[4]);

    struct Qmarray * mid = qmarray_kron(big4, mat2);
    struct Qmarray * shouldbe = mqma(mat,mid,k);

    //struct Qmarray * is2 = qmarray_mat_kron(k,mat,big4,mat2);
    //diff = qmarray_norm2diff(shouldbe,is2);
    //CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    diff = qmarray_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff,1e-10);

    qmarray_free(big1);
    qmarray_free(big2);
    qmarray_free(big3);
    qmarray_free(big4);
    qmarray_free(mid);
    qmarray_free(shouldbe);
    //qmarray_free(is2);

    free(mat); mat = NULL;
    for (ii = 0; ii < nblocks; ii++){
        qmarray_free(mat1[ii]); mat1[ii] = NULL;
    }
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(is); is = NULL;

    //qmarray_free(mat3); mat3 = NULL;
    //qmarray_free(shouldbe); shouldbe = NULL;

}

void Test_block_kron_mat3(CuTest * tc)
{
    printf("Testing Function: block_kron_mat (3/6) \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;

    size_t nblocks = 5;
    size_t rl1[5] = {3, 6, 9, 1, 6};
    size_t rl2[5] = {4, 4, 4, 4, 4};
    size_t sum_rl1 = 0;
    size_t sum_rl2 = 4;
    struct Qmarray * mat1[5];
    

    size_t r21 = 7;
    size_t r22 = 3;
    size_t k = 8;
    double diff; 
    
    size_t ii;
    for (ii = 0; ii < nblocks; ii++){
        sum_rl1 += rl1[ii];
        mat1[ii] = qmarray_poly_randu(LEGENDRE,rl1[ii],rl2[ii],
                                      maxorder,lb,ub);
    }
    struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,
                                               maxorder,lb,ub);
    double * mat = drandu(k*sum_rl1*r21);

    struct Qmarray * is = qmarray_alloc(k, sum_rl2 * r22);
    qmarray_block_kron_mat('C',1,nblocks,mat1,mat2,k,mat,is);
    
    struct Qmarray * big1 = qmarray_stackv(mat1[0],mat1[1]);
    struct Qmarray * big2 = qmarray_stackv(big1,mat1[2]);
    struct Qmarray * big3 = qmarray_stackv(big2,mat1[3]);
    struct Qmarray * big4 = qmarray_stackv(big3,mat1[4]);

    struct Qmarray * mid = qmarray_kron(big4, mat2);
    struct Qmarray * shouldbe = mqma(mat,mid,k);

    diff = qmarray_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff,1e-10);

    qmarray_free(big1);
    qmarray_free(big2);
    qmarray_free(big3);
    qmarray_free(big4);
    qmarray_free(mid);
    qmarray_free(shouldbe);

    free(mat); mat = NULL;
    for (ii = 0; ii < nblocks; ii++){
        qmarray_free(mat1[ii]); mat1[ii] = NULL;
    }
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(is); is = NULL;
}

void Test_block_kron_mat4(CuTest * tc)
{
    printf("Testing Function: block_kron_mat (4/6) \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;

    size_t nblocks = 5;
    size_t rl1[5] = {3, 6, 9, 1, 6};
    size_t rl2[5] = {2, 4, 6, 5, 3};
    size_t sum_rl1 = 0;
    size_t sum_rl2 = 0;
    struct Qmarray * mat1[5];
    

    size_t r21 = 7;
    size_t r22 = 3;
    size_t k = 8;
    double diff; 
    
    size_t ii;
    for (ii = 0; ii < nblocks; ii++){
        sum_rl1 += rl1[ii];
        sum_rl2 += rl2[ii];
        mat1[ii] = qmarray_poly_randu(LEGENDRE,rl1[ii],rl2[ii],
                                      maxorder,lb,ub);
    }
    struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,
                                               maxorder,lb,ub);
    double * mat = drandu(k*sum_rl2*r22);

    struct Qmarray * is = qmarray_alloc(sum_rl1 * r21,k);
    qmarray_block_kron_mat('D',0,nblocks,mat1,mat2,k,mat,is);


    struct Qmarray * big1 = qmarray_blockdiag(mat1[0],mat1[1]);
    struct Qmarray * big2 = qmarray_blockdiag(big1,mat1[2]);
    struct Qmarray * big3 = qmarray_blockdiag(big2,mat1[3]);
    struct Qmarray * big4 = qmarray_blockdiag(big3,mat1[4]);

    struct Qmarray * mid = qmarray_kron(big4, mat2);
    struct Qmarray * shouldbe = qmam(mid,mat,k);

    struct Qmarray * is2 = qmarray_kron_mat(k,mat,big4,mat2);
    diff = qmarray_norm2diff(shouldbe,is2);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    diff = qmarray_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff,1e-10);

    qmarray_free(big1);
    qmarray_free(big2);
    qmarray_free(big3);
    qmarray_free(big4);
    qmarray_free(mid);
    qmarray_free(shouldbe);
    qmarray_free(is2);

    free(mat); mat = NULL;
    for (ii = 0; ii < nblocks; ii++){
        qmarray_free(mat1[ii]); mat1[ii] = NULL;
    }
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(is); is = NULL;

    //qmarray_free(mat3); mat3 = NULL;
    //qmarray_free(shouldbe); shouldbe = NULL;

}


void Test_block_kron_mat5(CuTest * tc)
{
    printf("Testing Function: block_kron_mat (5/6) \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;

    size_t nblocks = 5;
    size_t rl1[5] = {2, 2, 2, 2, 2};
    size_t rl2[5] = {2, 4, 2, 5, 3};
    size_t sum_rl1 = 2;
    size_t sum_rl2 = 0;
    struct Qmarray * mat1[5];
    

    size_t r21 = 7;
    size_t r22 = 3;
    size_t k = 8;
    double diff; 
    
    size_t ii;
    for (ii = 0; ii < nblocks; ii++){
        sum_rl2 += rl2[ii];
        mat1[ii] = qmarray_poly_randu(LEGENDRE,rl1[ii],rl2[ii],
                                      maxorder,lb,ub);
    }
    struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,
                                               maxorder,lb,ub);
    double * mat = drandu(k*sum_rl2*r22);

    struct Qmarray * is = qmarray_alloc(sum_rl1 * r21,k);
    qmarray_block_kron_mat('R',0,nblocks,mat1,mat2,k,mat,is);

    struct Qmarray * big1 = qmarray_stackh(mat1[0],mat1[1]);
    struct Qmarray * big2 = qmarray_stackh(big1,mat1[2]);
    struct Qmarray * big3 = qmarray_stackh(big2,mat1[3]);
    struct Qmarray * big4 = qmarray_stackh(big3,mat1[4]);

    struct Qmarray * mid = qmarray_kron(big4, mat2);
    struct Qmarray * shouldbe = qmam(mid,mat,k);

    struct Qmarray * is2 = qmarray_kron_mat(k,mat,big4,mat2);
    diff = qmarray_norm2diff(shouldbe,is2);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    diff = qmarray_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff,1e-10);

    qmarray_free(big1);
    qmarray_free(big2);
    qmarray_free(big3);
    qmarray_free(big4);
    qmarray_free(mid);
    qmarray_free(shouldbe);
    qmarray_free(is2);

    free(mat); mat = NULL;
    for (ii = 0; ii < nblocks; ii++){
        qmarray_free(mat1[ii]); mat1[ii] = NULL;
    }
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(is); is = NULL;

    //qmarray_free(mat3); mat3 = NULL;
    //qmarray_free(shouldbe); shouldbe = NULL;

}

void Test_block_kron_mat6(CuTest * tc)
{
    printf("Testing Function: block_kron_mat (6/6) \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;

    size_t nblocks = 5;
    size_t rl1[5] = {3, 6, 9, 1, 6};
    size_t rl2[5] = {2, 2, 2, 2, 2};
    size_t sum_rl1 = 0;
    size_t sum_rl2 = 2;
    struct Qmarray * mat1[5];
    

    size_t r21 = 7;
    size_t r22 = 3;
    size_t k = 8;
    double diff; 
    
    size_t ii;
    for (ii = 0; ii < nblocks; ii++){
        sum_rl1 += rl1[ii];
        mat1[ii] = qmarray_poly_randu(LEGENDRE,rl1[ii],rl2[ii],
                                      maxorder,lb,ub);
    }
    struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,
                                               maxorder,lb,ub);
    double * mat = drandu(k*sum_rl2*r22);

    struct Qmarray * is = qmarray_alloc(sum_rl1 * r21,k);
    qmarray_block_kron_mat('C',0,nblocks,mat1,mat2,k,mat,is);


    struct Qmarray * big1 = qmarray_stackv(mat1[0],mat1[1]);
    struct Qmarray * big2 = qmarray_stackv(big1,mat1[2]);
    struct Qmarray * big3 = qmarray_stackv(big2,mat1[3]);
    struct Qmarray * big4 = qmarray_stackv(big3,mat1[4]);

    struct Qmarray * mid = qmarray_kron(big4, mat2);
    struct Qmarray * shouldbe = qmam(mid,mat,k);

    struct Qmarray * is2 = qmarray_kron_mat(k,mat,big4,mat2);
    diff = qmarray_norm2diff(shouldbe,is2);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    diff = qmarray_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff,1e-10);

    qmarray_free(big1);
    qmarray_free(big2);
    qmarray_free(big3);
    qmarray_free(big4);
    qmarray_free(mid);
    qmarray_free(shouldbe);
    qmarray_free(is2);

    free(mat); mat = NULL;
    for (ii = 0; ii < nblocks; ii++){
        qmarray_free(mat1[ii]); mat1[ii] = NULL;
    }
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(is); is = NULL;

    //qmarray_free(mat3); mat3 = NULL;
    //qmarray_free(shouldbe); shouldbe = NULL;

}

CuSuite * CLinalgQmarrayGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_qmarray_serialize);
    SUITE_ADD_TEST(suite, Test_qmarray_savetxt);
    
    // a bunch of QR tests
    SUITE_ADD_TEST(suite, Test_qmarray_orth1d_columns);
    SUITE_ADD_TEST(suite, Test_qmarray_orth1d_columns_fourier);
    SUITE_ADD_TEST(suite, Test_qmarray_orth1d_rows_fourier);
    SUITE_ADD_TEST(suite, Test_qmarray_householder2_fourier);
    SUITE_ADD_TEST(suite, Test_qmarray_householder2_fourier2);
    
    SUITE_ADD_TEST(suite, Test_qmarray_householder);
    SUITE_ADD_TEST(suite, Test_qmarray_householder2);
    SUITE_ADD_TEST(suite, Test_qmarray_householder3);
    SUITE_ADD_TEST(suite, Test_qmarray_householder4);
    SUITE_ADD_TEST(suite, Test_qmarray_householder_hermite1);
    SUITE_ADD_TEST(suite, Test_qmarray_qr1);
    SUITE_ADD_TEST(suite, Test_qmarray_qr2);
    SUITE_ADD_TEST(suite, Test_qmarray_qr3);
    SUITE_ADD_TEST(suite, Test_qmarray_lq);
    SUITE_ADD_TEST(suite, Test_qmarray_householder_rows);
    SUITE_ADD_TEST(suite, Test_qmarray_householder_rows_hermite);
    SUITE_ADD_TEST(suite, Test_qmarray_householder_constelm);
    SUITE_ADD_TEST(suite, Test_qmarray_householder_rowsconstelm);
    SUITE_ADD_TEST(suite, Test_qmarray_householder_linelm);
    SUITE_ADD_TEST(suite, Test_qmarray_householder_rowslinelm);
    SUITE_ADD_TEST(suite, Test_qmarray_svd);

    SUITE_ADD_TEST(suite, Test_qmarray_lu1d);
    SUITE_ADD_TEST(suite, Test_qmarray_lu1d2);
    SUITE_ADD_TEST(suite, Test_qmarray_lu1d_hermite);
    SUITE_ADD_TEST(suite, Test_qmarray_lu1d_constelm);
    SUITE_ADD_TEST(suite, Test_qmarray_lu1d_linelm);
    
    SUITE_ADD_TEST(suite, Test_qmarray_maxvol1d);
    SUITE_ADD_TEST(suite, Test_qmarray_maxvol1d2);
    SUITE_ADD_TEST(suite, Test_qmarray_maxvol1d_hermite1);
    SUITE_ADD_TEST(suite, Test_qmarray_maxvol1d_constelm);
    SUITE_ADD_TEST(suite, Test_qmarray_maxvol1d_linelm);

    SUITE_ADD_TEST(suite,Test_fast_mat_kron);
    SUITE_ADD_TEST(suite,Test_fast_kron_mat);
    SUITE_ADD_TEST(suite,Test_block_kron_mat1);
    SUITE_ADD_TEST(suite,Test_block_kron_mat2);
    SUITE_ADD_TEST(suite,Test_block_kron_mat3);
    SUITE_ADD_TEST(suite,Test_block_kron_mat4);
    SUITE_ADD_TEST(suite,Test_block_kron_mat5);
    SUITE_ADD_TEST(suite,Test_block_kron_mat6);
    
    return suite;
}
