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
#include "lib_clinalg.h"
#include "c3_interface.h"

void Test_ftapprox_grad(CuTest * tc)
{
    printf("Testing Function: function_train_gradient\n");
    
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcGrad,NULL);
    // set function monitor

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-10.0);
    ope_opts_set_ub(opts,10.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 5;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(-1.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);
     
    struct FT1DArray * ftg = function_train_gradient(ft);

    double pt[4] = {2.0, -3.1456, 1.0, 0.0};
    double * grad = ft1d_array_eval(ftg,pt);

    CuAssertDblEquals(tc, pt[1], grad[0],1e-12);
    CuAssertDblEquals(tc, pt[0], grad[1],1e-12);
    CuAssertDblEquals(tc, pt[3], grad[2],1e-12);
    CuAssertDblEquals(tc, pt[2], grad[3],1e-12);
    
    free(grad); grad = NULL;
    function_train_free(ft); ft = NULL; 
    ft1d_array_free(ftg); ftg = NULL;
    fwrap_destroy(fw);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
}

void Test_ft1d_array_serialize(CuTest * tc)
{
    printf("Testing Function: ft1d_array_(de)serialize\n");
    
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcGrad,NULL);
    // set function monitor

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-10.0);
    ope_opts_set_ub(opts,10.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 5;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(-1.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);
    struct FT1DArray * ftg = function_train_gradient(ft);

    unsigned char * text = NULL;
    size_t size;
    ft1d_array_serialize(NULL,ftg,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    ft1d_array_serialize(text,ftg,NULL);

    struct FT1DArray * ftgg = NULL;
    //printf("derserializing ft\n");
    ft1d_array_deserialize(text, &ftgg);

    double pt[4] = {2.0, -3.1456, 1.0, 0.0};
    double * grad = ft1d_array_eval(ftgg,pt);

    CuAssertDblEquals(tc, pt[1], grad[0],1e-12);
    CuAssertDblEquals(tc, pt[0], grad[1],1e-12);
    CuAssertDblEquals(tc, pt[3], grad[2],1e-12);
    CuAssertDblEquals(tc, pt[2], grad[3],1e-12);
    
    free(grad); grad = NULL;
    function_train_free(ft); ft = NULL; 
    ft1d_array_free(ftgg); ftgg = NULL;
    ft1d_array_free(ftg); ftg = NULL;
    free(text);
    fwrap_destroy(fw);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
}

void Test_ftapprox_hess(CuTest * tc)
{
    printf("Testing Function: function_train_hessian\n");
    size_t dim = 3;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcHess,NULL);
    // set function monitor

    double lb = -2.0;
    double ub = 2.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 5;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(-1.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);
    
    /* printf("ranks are\n"); */
    /* iprint_sz(dim+1,ft->ranks); */
    size_t N = 10;
    double * xtest = linspace(-2.0,2.0,N);
    double err = 0.0;
    double den = 0.0;
    double eval;
    double ptt[3];
    
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            for (size_t kk = 0; kk < N; kk++){
                //for (ll = 0; ll < N; ll++){
                    ptt[0] = xtest[ii]; ptt[1] = xtest[jj]; 
                    ptt[2] = xtest[kk]; //ptt[3] = xtest[ll];
                    funcHess(1,ptt,&eval,NULL);
                    den += pow(eval,2.0);
                    err += pow(eval - 
                               function_train_eval(ft,ptt),2.0);
                    //printf("err=%G\n",err);
               // }
            }
        }
    }
    err = sqrt(err/den);
    CuAssertDblEquals(tc,0.0,err,1e-12);
    free(xtest); xtest = NULL;

    struct FT1DArray * fth = function_train_hessian(ft);
    double pt[3] = {1.8, -1.0, 1.0};//, 0.5};
    double * hess = ft1d_array_eval(fth,pt);
    
    //dprint2d_col(3,3,hess);
    CuAssertDblEquals(tc, 2.0*pt[2], hess[0],1e-6);
    CuAssertDblEquals(tc, 0.0, hess[1],1e-6);
    CuAssertDblEquals(tc, 2.0*pt[0], hess[2],1e-6);
    CuAssertDblEquals(tc, hess[1], hess[3],1e-8);
    CuAssertDblEquals(tc, 0.0, hess[4], 1e-6);
    CuAssertDblEquals(tc, 4.0*pow(pt[2],3), hess[5], 1e-6);
    CuAssertDblEquals(tc, hess[2], hess[6], 1e-6);
    CuAssertDblEquals(tc, hess[5], hess[7], 1e-6);
    CuAssertDblEquals(tc, 12.0*pt[1]*pow(pt[2],2), hess[8],1e-6);
    
    free(hess); hess = NULL;
    function_train_free(ft); ft = NULL; 
    ft1d_array_free(fth); fth = NULL;
    fwrap_destroy(fw);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
}


void Test_ftapprox_hess2(CuTest * tc)
{
    printf("Testing Function: function_train_hessian (piecewise-poly)\n");
    size_t dim = 3;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcHess,NULL);
    // set function monitor

    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,-2.0,2.0);
    pw_poly_opts_set_maxorder(opts,5);
    pw_poly_opts_set_coeffs_check(opts,2);
    pw_poly_opts_set_tol(opts,1e-3);
    /* pw_poly_opts_set_minsize(opts,4); */
    pw_poly_opts_set_minsize(opts,1e-2);
    /* pw_poly_opts_set_nregions(opts,5); */
    pw_poly_opts_set_nregions(opts,2);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(PIECEWISE,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 3;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(-1.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);
    
    /* printf("ranks are\n"); */
    /* iprint_sz(dim+1,ft->ranks); */
    size_t N = 10;
    double * xtest = linspace(-2.0,2.0,N);
    double err = 0.0;
    double den = 0.0;
    double eval;
    double ptt[3];
    
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            for (size_t kk = 0; kk < N; kk++){
                //for (ll = 0; ll < N; ll++){
                    ptt[0] = xtest[ii]; ptt[1] = xtest[jj]; 
                    ptt[2] = xtest[kk]; //ptt[3] = xtest[ll];
                    funcHess(1,ptt,&eval,NULL);
                    den += pow(eval,2.0);
                    err += pow(eval - 
                               function_train_eval(ft,ptt),2.0);
                    //printf("err=%G\n",err);
               // }
            }
        }
    }
    err = sqrt(err/den);
    CuAssertDblEquals(tc,0.0,err,1e-12);
    free(xtest); xtest = NULL;

    /* printf("hess\n"); */
    struct FT1DArray * fth = function_train_hessian(ft);
    double pt[3] = {1.8, -1.0, 1.0};//, 0.5};
    double * hess = ft1d_array_eval(fth,pt);
    /* printf("got it\n"); */
    
    //dprint2d_col(3,3,hess);
    CuAssertDblEquals(tc, 2.0*pt[2], hess[0],1e-6);
    CuAssertDblEquals(tc, 0.0, hess[1],1e-6);
    CuAssertDblEquals(tc, 2.0*pt[0], hess[2],1e-6);
    CuAssertDblEquals(tc, hess[1], hess[3],1e-8);
    CuAssertDblEquals(tc, 0.0, hess[4], 1e-6);
    CuAssertDblEquals(tc, 4.0*pow(pt[2],3), hess[5], 1e-6);
    CuAssertDblEquals(tc, hess[2], hess[6], 1e-6);
    CuAssertDblEquals(tc, hess[5], hess[7], 1e-6);
    CuAssertDblEquals(tc, 12.0*pt[1]*pow(pt[2],2), hess[8],1e-6);
    
    free(hess); hess = NULL;
    function_train_free(ft); ft = NULL; 
    ft1d_array_free(fth); fth = NULL;
    fwrap_destroy(fw);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
}

CuSuite * CLinalgFuncTrainArrayGetSuite()
{
    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_ftapprox_grad);
    SUITE_ADD_TEST(suite,Test_ft1d_array_serialize);
    SUITE_ADD_TEST(suite, Test_ftapprox_hess);
    SUITE_ADD_TEST(suite, Test_ftapprox_hess2);
    return suite;
}
