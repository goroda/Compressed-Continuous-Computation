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


static void all_opts_free(
    struct Fwrap * fw,
    struct OpeOpts * opts,
    struct OneApproxOpts * qmopts,
    struct MultiApproxOpts * fopts)
{
    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
    multi_approx_opts_free(fopts);
}


void Test_function_train_grad_eval(CuTest * tc)
{
    printf("Testing Function: function_train_gradient_eval\n");
    
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcGrad,NULL);
    // set function monitor

    double lb = -10.0;
    double ub = 10.0;
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
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);

    double grad_eval[4];
    double pt[4] = {2.0, -3.1456, 1.0, 0.0};
    function_train_gradient_eval(ft,pt,grad_eval);

    double eval = function_train_eval(ft,pt);
    double eval_should = pt[0]*pt[1] + pt[2]*pt[3];
    CuAssertDblEquals(tc, eval_should, eval,1e-12);

    
    CuAssertDblEquals(tc, pt[1], grad_eval[0],1e-12);
    CuAssertDblEquals(tc, pt[0], grad_eval[1],1e-12);
    CuAssertDblEquals(tc, pt[3], grad_eval[2],1e-12);
    CuAssertDblEquals(tc, pt[2], grad_eval[3],1e-12);
    
    function_train_free(ft); ft = NULL; 
    fwrap_destroy(fw);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
}

void Test_function_train_grad_eval2(CuTest * tc)
{
    printf("Testing Function: function_train_gradient_eval (2)\n");
    
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcCheck2,NULL);
    // set function monitor

    double lb = -10.0;
    double ub = 10.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 1;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(-1.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    c3approx_set_cross_tol(c3a,1e-12);
    c3approx_set_round_tol(c3a,1e-12);
    c3approx_set_cross_maxiter(c3a,12);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);

    /* iprint_sz(dim+1,function_train_get_ranks(ft)); */
    double grad_eval[4];
    double pt[4] = {2.0, -3.1456, 1.0, 0.2};
    function_train_gradient_eval(ft,pt,grad_eval);
        
    double eval = function_train_eval(ft,pt);
    double eval_should;
    funcCheck2(1,pt,&eval_should,NULL);
    CuAssertDblEquals(tc, 0.0, (eval_should-eval)/eval_should,1e-7);


    double g1_should = 2*pt[0]*pt[1]*pt[1];
    double g2_should = 2.0*pt[0]*pt[1]*pt[0]  + sin(pt[3]);
    double g3_should = pt[3];
    double g4_should = pt[2] + pt[1]*cos(pt[3]);
    CuAssertDblEquals(tc, 0.0, (grad_eval[0]-g1_should)/g1_should,1e-6);
    CuAssertDblEquals(tc, 0.0, (grad_eval[1]-g2_should)/g2_should,1e-6);
    CuAssertDblEquals(tc, 0.0, (grad_eval[2]-g3_should)/g3_should,1e-6);
    CuAssertDblEquals(tc, 0.0, (grad_eval[3]-g4_should)/g4_should,2e-6);
    
    
    /* CuAssertDblEquals(tc, pt[3], grad_eval[2],1e-12); */
    /* CuAssertDblEquals(tc, pt[2], grad_eval[3],1e-12); */
    
    function_train_free(ft); ft = NULL; 
    fwrap_destroy(fw);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
}

void Test_function_train_grad_eval3(CuTest * tc)
{
    printf("Testing Function: function_train_gradient_eval (3)\n");
    
    size_t dim = 5;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcCheck3,NULL);
    // set function monitor

    double lb = -1.0;
    double ub = 1.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 2;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(lb,ub,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    c3approx_set_cross_tol(c3a,1e-12);
    c3approx_set_round_tol(c3a,1e-12);
    c3approx_set_cross_maxiter(c3a,12);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);

    /* iprint_sz(dim+1,function_train_get_ranks(ft)); */
    double grad_eval[5];
    double pt[5] = {-1.39480E-01, -3.89252E-01, -1.31999E-01, -8.02793E-01, 8.31607E-01};
    function_train_gradient_eval(ft,pt,grad_eval);

    double g_should = cos(pt[0]+pt[1]+pt[2]+pt[3]+pt[4]);
    for (size_t ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,g_should,grad_eval[ii],1e-4);
    }

    size_t N = 100;
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }
        double eval_ft = function_train_eval(ft,pt);
        double eval_true;
        funcCheck3(1,pt,&eval_true,NULL);
        CuAssertDblEquals(tc,eval_true,eval_ft,1e-5);
    }
    
    
    function_train_free(ft); ft = NULL; 
    fwrap_destroy(fw);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
}


void Test_function_train_initsum(CuTest * tc){

    printf("Testing function: function_train_initsum \n");

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(4);
    multi_approx_opts_set_all_same(fopts,qmopts);
   
    struct FunctionTrain * ft = function_train_initsum(fopts,fw);
    size_t * ranks = function_train_get_ranks(ft);
    for (size_t ii = 1; ii < 4; ii++ ){
        CuAssertIntEquals(tc,2,ranks[ii]);
    }

    double pt[4];
    double val, tval; 
    
    size_t N = 20;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;

    size_t ii,jj,kk,ll;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    double tval1, tval2, tval3, tval4;
                    func(1,pt,&tval1,NULL);
                    func2(1,pt+1,&tval2,NULL);
                    func3(1,pt+2,&tval3,NULL);
                    func4(1,pt+3,&tval4,NULL);
                    tval = tval1 + tval2 + tval3 + tval4;
                    val = function_train_eval(ft,pt);
                    den += pow(tval,2.0);
                    err += pow(tval-val,2.0);
                }
            }
        }
    }
    err = err/den;
    CuAssertDblEquals(tc,0.0,err,1e-15);
    free(xtest);

    function_train_free(ft);
    all_opts_free(fw,opts,qmopts,fopts);
}   

void Test_function_train_linear(CuTest * tc)
{
    printf("Testing Function: function_train_linear \n");

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(3);
    multi_approx_opts_set_all_same(fopts,qmopts);
    
    double slope[3] = {1.0, 2.0, 3.0};
    double offset[3] = {0.0, 0.0, 0.0};
    struct FunctionTrain * f =function_train_linear(slope,1,offset,1,fopts);

    double pt[3] = { -0.1, 0.4, 0.2 };
    double eval = function_train_eval(f,pt);
    CuAssertDblEquals(tc, 1.3, eval, 1e-14);
    
    pt[0] = 0.8; pt[1] = -0.2; pt[2] = 0.3;
    eval = function_train_eval(f,pt);
    CuAssertDblEquals(tc, 1.3, eval, 1e-14);

    pt[0] = -0.8; pt[1] = 1.0; pt[2] = -0.01;
    eval = function_train_eval(f,pt);
    CuAssertDblEquals(tc, 1.17, eval, 1e-14);
    
    function_train_free(f);
    all_opts_free(NULL,opts,qmopts,fopts);
}

void Test_function_train_quadratic(CuTest * tc)
{
    printf("Testing Function: function_train_quadratic (1/2)\n");
    size_t dim = 3;
    double lb = -3.12;
    double ub = 2.21;

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);

    double * quad = calloc_double(dim * dim);
    double * coeff = calloc_double(dim);
    size_t ii,jj,kk;
    for (ii = 0; ii < dim; ii++){
        coeff[ii] = randu();
        for (jj = 0; jj < dim; jj++){
            quad[ii*dim+jj] = randu();
        }
    }

    struct FunctionTrain * f = function_train_quadratic(quad,coeff,fopts);

    size_t N = 10;
    double * xtest = linspace(lb,ub,N);
    double * pt = calloc_double(dim);
    size_t ll, mm;
    double should, is;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N;  jj++){
            for (kk = 0; kk < N; kk++){
                pt[0] = xtest[ii]; pt[1] = xtest[jj]; pt[2] = xtest[kk];
                should = 0.0;
                for (ll = 0; ll< dim; ll++){
                    for (mm = 0; mm < dim; mm++){
                        should += (pt[ll]-coeff[ll])*quad[mm*dim+ll]*(pt[mm]-coeff[mm]);
                    }
                }
                //printf("should=%G\n",should);
                is = function_train_eval(f,pt);
                //printf("is=%G\n",is);
                CuAssertDblEquals(tc,should,is,1e-12);
            }
        }
    }
    free(xtest);
    free(pt);
    free(quad);
    free(coeff);
    
    function_train_free(f);
    all_opts_free(NULL,opts,qmopts,fopts);
}

void Test_function_train_quadratic2(CuTest * tc)
{
    printf("Testing Function: function_train_quadratic (2/2)\n");
    size_t dim = 4;
    double lb = -1.32;
    double ub = 6.0;

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);

    double * quad = calloc_double(dim * dim);
    double * coeff = calloc_double(dim);
    size_t ii,jj,kk,zz;
    for (ii = 0; ii < dim; ii++){
        coeff[ii] = randu();
        for (jj = 0; jj < dim; jj++){
            quad[ii*dim+jj] = randu();
        }
    }

    struct FunctionTrain * f = function_train_quadratic(quad,coeff,fopts);

    size_t N = 10;
    double * xtest = linspace(lb,ub,N);
    double * pt = calloc_double(dim);
    size_t ll, mm;
    double should, is;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N;  jj++){
            for (kk = 0; kk < N; kk++){
                for (zz = 0; zz < N; zz++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; pt[2] = xtest[kk]; pt[3] = xtest[zz];
                    should = 0.0;
                    for (ll = 0; ll< dim; ll++){
                        for (mm = 0; mm < dim; mm++){
                            should += (pt[ll]-coeff[ll])*quad[mm*dim+ll]*(pt[mm]-coeff[mm]);
                        }
                    }
                    //printf("should=%G\n",should);
                    is = function_train_eval(f,pt);
                    //printf("is=%G\n",is);
                    CuAssertDblEquals(tc,should,is,1e-12);
                }
            }
        }
    }
    
    free(xtest);
    free(pt);
    free(quad);
    free(coeff);
    function_train_free(f);
    all_opts_free(NULL,opts,qmopts,fopts);
}

void Test_function_train_sum_function_train_round(CuTest * tc)
{
    printf("Testing Function: function_train_sum and ft_round \n");

    size_t dim = 3;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);
    
    double coeffs[3] = {1.0, 2.0, 3.0};
    double off[3] = {0.0,0.0,0.0};
    struct FunctionTrain * a = function_train_linear(coeffs,1,off,1,fopts);

    double coeffsb[3] = {1.5, -0.2, 3.310};
    double offb[3]    = {0.0, 0.0, 0.0};
    struct FunctionTrain * b = function_train_linear(coeffsb,1,offb,1,fopts);
    
    struct FunctionTrain * c = function_train_sum(a,b);
    size_t * ranks = function_train_get_ranks(c);
    CuAssertIntEquals(tc,1,ranks[0]);
    CuAssertIntEquals(tc,4,ranks[1]);
    CuAssertIntEquals(tc,4,ranks[2]);
    CuAssertIntEquals(tc,1,ranks[3]);

    double pt[3];
    double eval, evals;
    
    pt[0] = -0.1; pt[1] = 0.4; pt[2]=0.2; 
    eval = function_train_eval(c,pt);
    evals = -0.1*(1.0 + 1.5) + 0.4*(2.0-0.2) + 0.2*(3.0 + 3.31);
    CuAssertDblEquals(tc, evals, eval, 1e-14);
    
    pt[0] = 0.8; pt[1] = -0.2; pt[2] = 0.3;
    evals = 0.8*(1.0 + 1.5) - 0.2*(2.0-0.2) + 0.3*(3.0 + 3.31);
    eval = function_train_eval(c,pt);
    CuAssertDblEquals(tc, evals, eval, 1e-14);

    pt[0] = -0.8; pt[1] = 1.0; pt[2] = -0.01;
    evals = -0.8*(1.0 + 1.5) + 1.0*(2.0-0.2) - 0.01*(3.0 + 3.31);
    eval = function_train_eval(c,pt);
    CuAssertDblEquals(tc, evals, eval,1e-14);
    
    struct FunctionTrain * d = function_train_round(c, 1e-10,fopts);
    ranks = function_train_get_ranks(d);
    CuAssertIntEquals(tc,1,ranks[0]);
    CuAssertIntEquals(tc,2,ranks[1]);
    CuAssertIntEquals(tc,2,ranks[2]);
    CuAssertIntEquals(tc,1,ranks[3]);

    pt[0] = -0.1; pt[1] = 0.4; pt[2]=0.2; 
    eval = function_train_eval(d,pt);
    evals = -0.1*(1.0 + 1.5) + 0.4*(2.0-0.2) + 0.2*(3.0 + 3.31);
    CuAssertDblEquals(tc, evals, eval, 1e-14);
    
    pt[0] = 0.8; pt[1] = -0.2; pt[2] = 0.3;
    evals = 0.8*(1.0 + 1.5) - 0.2*(2.0-0.2) + 0.3*(3.0 + 3.31);
    eval = function_train_eval(d,pt);
    CuAssertDblEquals(tc, evals, eval, 1e-14);

    pt[0] = -0.8; pt[1] = 1.0; pt[2] = -0.01;
    evals = -0.8*(1.0 + 1.5) + 1.0*(2.0-0.2) - 0.01*(3.0 + 3.31);
    eval = function_train_eval(d,pt);
    CuAssertDblEquals(tc, evals, eval,1e-14);
    
    function_train_free(a);
    function_train_free(b);
    function_train_free(c);
    function_train_free(d);
    all_opts_free(NULL,opts,qmopts,fopts);
}

void Test_function_train_scale(CuTest * tc)
{
    printf("Testing Function: function_train_scale \n");
    size_t dim = 4;
    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);

    struct FunctionTrain * ft = function_train_initsum(fopts,fw);

    double pt[4];
    double val, tval;
    double tval1,tval2,tval3,tval4;
    double scale = 4.0;
    function_train_scale(ft,scale);
    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;

    size_t ii,jj,kk,ll;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    func(1,pt,&tval1,NULL);
                    func2(1,pt+1,&tval2,NULL);
                    func3(1,pt+2,&tval3,NULL);
                    func4(1,pt+3,&tval4,NULL);
                    tval = tval1 + tval2 + tval3 + tval4;
                    tval = tval * scale;
                    val = function_train_eval(ft,pt);
                    den += pow(tval,2.0);
                    err += pow(tval-val,2.0);
                }
            }
        }
    }
    err = err/den;
    CuAssertDblEquals(tc,0.0,err,1e-15);
    free(xtest);
    
    all_opts_free(fw,opts,qmopts,fopts);
    function_train_free(ft);
}

void Test_function_train_product(CuTest * tc)
{
    printf("Testing Function: function_train_product \n");
    size_t dim = 4;

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw2,4);
    fwrap_set_func_array(fw2,0,func2,NULL);
    fwrap_set_func_array(fw2,1,func5,NULL);
    fwrap_set_func_array(fw2,2,func4,NULL);
    fwrap_set_func_array(fw2,3,func6,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);

    
    struct FunctionTrain * ft = function_train_initsum(fopts,fw);
    struct FunctionTrain * gt = function_train_initsum(fopts,fw2);
    struct FunctionTrain * ft2 =  function_train_product(ft,gt);

    double pt[4];
    double val, tval1,tval2; 
    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;

    size_t ii,jj,kk,ll;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    tval1 =  function_train_eval(ft,pt);
                    tval2 =  function_train_eval(gt,pt);
                    val = function_train_eval(ft2,pt);
                    den += pow(tval1*tval2,2.0);
                    err += pow(tval1*tval2-val,2.0);
                }
            }
        }
    }
    err = err/den;

    CuAssertDblEquals(tc,0.0,err,1e-15);
    
    free(xtest);
    function_train_free(ft);
    function_train_free(gt);
    function_train_free(ft2);
    fwrap_destroy(fw2);
    all_opts_free(fw,opts,qmopts,fopts);
}

void Test_function_train_product_cheb(CuTest * tc)
{
    printf("Testing Function: function_train_product_cheb \n");
    size_t dim = 4;

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw2,4);
    fwrap_set_func_array(fw2,0,func2,NULL);
    fwrap_set_func_array(fw2,1,func5,NULL);
    fwrap_set_func_array(fw2,2,func4,NULL);
    fwrap_set_func_array(fw2,3,func6,NULL);

    struct OpeOpts * opts = ope_opts_alloc(CHEBYSHEV);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);

    
    struct FunctionTrain * ft = function_train_initsum(fopts,fw);
    struct FunctionTrain * gt = function_train_initsum(fopts,fw2);
    struct FunctionTrain * ft2 =  function_train_product(ft,gt);

    double pt[4];
    double val, tval1,tval2; 
    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;

    size_t ii,jj,kk,ll;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    tval1 =  function_train_eval(ft,pt);
                    tval2 =  function_train_eval(gt,pt);
                    val = function_train_eval(ft2,pt);
                    den += pow(tval1*tval2,2.0);
                    err += pow(tval1*tval2-val,2.0);
                }
            }
        }
    }
    err = err/den;

    CuAssertDblEquals(tc,0.0,err,1e-15);
    
    free(xtest);
    function_train_free(ft);
    function_train_free(gt);
    function_train_free(ft2);
    fwrap_destroy(fw2);
    all_opts_free(fw,opts,qmopts,fopts);
}


void Test_function_train_integrate(CuTest * tc)
{
    printf("Testing Function: function_train_integrate \n");
    size_t dim = 4;

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,0.0);
    struct OpeOpts * opts2 = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts2,-1.0);
    struct OpeOpts * opts3 = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts3,-5.0);
    struct OpeOpts * opts4 = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts4,-5.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct OneApproxOpts * qmopts2 = one_approx_opts_alloc(POLYNOMIAL,opts2);
    struct OneApproxOpts * qmopts3 = one_approx_opts_alloc(POLYNOMIAL,opts3);
    struct OneApproxOpts * qmopts4 = one_approx_opts_alloc(POLYNOMIAL,opts4);

    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_dim(fopts,0,qmopts);
    multi_approx_opts_set_dim(fopts,1,qmopts2);
    multi_approx_opts_set_dim(fopts,2,qmopts3);
    multi_approx_opts_set_dim(fopts,3,qmopts4);

    struct FunctionTrain * ft = function_train_initsum(fopts,fw);
    double out =  function_train_integrate(ft);
    
    double shouldbe = 110376.0/5.0;
    double rel_error = pow(out-shouldbe,2)/fabs(shouldbe);
    CuAssertDblEquals(tc, 0.0 ,rel_error,1e-15);

    all_opts_free(fw,opts,qmopts,fopts);
    all_opts_free(NULL,opts2,qmopts2,NULL);
    all_opts_free(NULL,opts3,qmopts3,NULL);
    all_opts_free(NULL,opts4,qmopts4,NULL);
    function_train_free(ft);
}

void Test_function_train_integrate_cheb(CuTest * tc)
{
    printf("Testing Function: function_train_integrate_cheb \n");
    size_t dim = 4;

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct OpeOpts * opts = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_lb(opts,0.0);
    struct OpeOpts * opts2 = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_lb(opts2,-1.0);
    struct OpeOpts * opts3 = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_lb(opts3,-5.0);
    struct OpeOpts * opts4 = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_lb(opts4,-5.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct OneApproxOpts * qmopts2 = one_approx_opts_alloc(POLYNOMIAL,opts2);
    struct OneApproxOpts * qmopts3 = one_approx_opts_alloc(POLYNOMIAL,opts3);
    struct OneApproxOpts * qmopts4 = one_approx_opts_alloc(POLYNOMIAL,opts4);

    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_dim(fopts,0,qmopts);
    multi_approx_opts_set_dim(fopts,1,qmopts2);
    multi_approx_opts_set_dim(fopts,2,qmopts3);
    multi_approx_opts_set_dim(fopts,3,qmopts4);

    struct FunctionTrain * ft = function_train_initsum(fopts,fw);
    double out =  function_train_integrate(ft);
    
    double shouldbe = 110376.0/5.0;
    double rel_error = pow(out-shouldbe,2)/fabs(shouldbe);
    CuAssertDblEquals(tc, 0.0 ,rel_error,1e-15);

    all_opts_free(fw,opts,qmopts,fopts);
    all_opts_free(NULL,opts2,qmopts2,NULL);
    all_opts_free(NULL,opts3,qmopts3,NULL);
    all_opts_free(NULL,opts4,qmopts4,NULL);
    function_train_free(ft);
}

void Test_function_train_integrate_weighted(CuTest * tc)
{
    printf("Testing Function: function_train_integrate_weighted \n");
    size_t dim = 4;

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    double lb1 = 0.0;
    double lb2 = -1.0;
    double lb3 = -5.0;
    double lb4 = -5.0;
    double ub = 1.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb1);
    struct OpeOpts * opts2 = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts2,lb2);
    struct OpeOpts * opts3 = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts3,lb3);
    struct OpeOpts * opts4 = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts4,lb4);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct OneApproxOpts * qmopts2 = one_approx_opts_alloc(POLYNOMIAL,opts2);
    struct OneApproxOpts * qmopts3 = one_approx_opts_alloc(POLYNOMIAL,opts3);
    struct OneApproxOpts * qmopts4 = one_approx_opts_alloc(POLYNOMIAL,opts4);

    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_dim(fopts,0,qmopts);
    multi_approx_opts_set_dim(fopts,1,qmopts2);
    multi_approx_opts_set_dim(fopts,2,qmopts3);
    multi_approx_opts_set_dim(fopts,3,qmopts4);

    struct FunctionTrain * ft = function_train_initsum(fopts,fw);
    double out =  function_train_integrate_weighted(ft);


    double shouldbe = 110376.0/5.0 / (ub-lb1) / (ub-lb2) / 
                          (ub-lb3) / (ub-lb4);
    /* printf(" Is: %G\n",out); */
    /* printf(" Should: %G\n",shouldbe); */
    double rel_error = pow(out-shouldbe,2)/fabs(shouldbe);
    CuAssertDblEquals(tc, 0.0 ,rel_error,1e-15);

    all_opts_free(fw,opts,qmopts,fopts);
    all_opts_free(NULL,opts2,qmopts2,NULL);
    all_opts_free(NULL,opts3,qmopts3,NULL);
    all_opts_free(NULL,opts4,qmopts4,NULL);
    function_train_free(ft);
}

void Test_function_train_integrate_weighted_subset(CuTest * tc)
{
    printf("Testing Function: function_train_integrate_weighted_subset \n");
    size_t dim = 4;

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func2,NULL);
    fwrap_set_func_array(fw,1,func3,NULL);
    fwrap_set_func_array(fw,2,func4,NULL);
    fwrap_set_func_array(fw,3,func,NULL);

    double lb1 = 0.0;
    double lb2 = -1.0;
    double lb3 = -5.0;
    double lb4 = -5.0;
    double ub = 1.0;
    
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb1);
    struct OpeOpts * opts2 = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts2,lb2);
    struct OpeOpts * opts3 = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts3,lb3);
    struct OpeOpts * opts4 = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts4,lb4);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct OneApproxOpts * qmopts2 = one_approx_opts_alloc(POLYNOMIAL,opts2);
    struct OneApproxOpts * qmopts3 = one_approx_opts_alloc(POLYNOMIAL,opts3);
    struct OneApproxOpts * qmopts4 = one_approx_opts_alloc(POLYNOMIAL,opts4);

    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_dim(fopts,0,qmopts);
    multi_approx_opts_set_dim(fopts,1,qmopts2);
    multi_approx_opts_set_dim(fopts,2,qmopts3);
    multi_approx_opts_set_dim(fopts,3,qmopts4);

    struct FunctionTrain * ft = function_train_initsum(fopts,fw);

    double oned_integrals[4];
    oned_integrals[0] = (ub*ub - lb1*lb1)/2.0;
    oned_integrals[1] = (ub*ub*ub/3.0 - cos(3.14159*ub)/3.14159) -
        lb2*lb2*lb2/3.0 + cos(3.14159*lb2)/3.14159;
    oned_integrals[2] = (3.0*ub*ub*ub*ub*ub/5.0 - ub*ub*ub*2/3) -
        (3.0*lb3*lb3*lb3*lb3*lb3/5.0 - lb3*lb3*lb3*2.0/3.0);
    oned_integrals[3] = (ub - lb4);

    oned_integrals[0] /= (ub-lb1);
    oned_integrals[1] /= (ub-lb2);
    oned_integrals[2] /= (ub-lb3);
    oned_integrals[3] /= (ub-lb4);
    

    /* printf("oned_integrals = "); */
    /* dprint(4,oned_integrals); */

    /* printf("\n\n\n"); */
    // first check integrating out one variable
    size_t ndim_contract = 1;
    size_t on_dim=0;
    double pt[3];
    struct FunctionTrain * ft_1;
    double eval_ft,e1,e2,e3,should_be,rel_error;
    
    pt[0] = (ub-lb2)*0.2 + lb2;
    pt[1] = (ub-lb3)*0.9 + lb3;
    pt[2] = (ub-lb4)*0.4 * lb4;

    ft_1 = function_train_integrate_weighted_subset(ft,ndim_contract,&on_dim);
    eval_ft  = function_train_eval(ft_1,pt);
    function_train_free(ft_1); ft_1 = NULL;
    
    func3(1,pt,&e1,NULL);
    func4(1,pt+1,&e2,NULL);
    func(1,pt+2,&e3,NULL);
    should_be = e1 + e2 + e3 + oned_integrals[0];

    /* printf(" Is: %G\n",eval_ft); */
    /* printf(" Should: %G\n",should_be); */
    rel_error = pow(eval_ft-should_be,2)/fabs(should_be);
    CuAssertDblEquals(tc, 0.0 ,rel_error,1e-15);

    /* printf("\n\n\n"); */

    on_dim = 1;
    pt[0] = (ub-lb1)*0.1 + lb1;
    pt[1] = (ub-lb3)*0.9 + lb3;
    pt[2] = (ub-lb4)*0.4 * lb4;

    ft_1 = function_train_integrate_weighted_subset(ft,ndim_contract,&on_dim);
    eval_ft  = function_train_eval(ft_1,pt);
    function_train_free(ft_1); ft_1 = NULL;

    func2(1,pt,&e1,NULL);
    func4(1,pt+1,&e2,NULL);
    func(1,pt+2,&e3,NULL);
    should_be = e1 + e2 + e3 + oned_integrals[on_dim];

    /* printf(" Is: %G\n",eval_ft); */
    /* printf(" Should: %G\n",should_be); */
    rel_error = pow(eval_ft-should_be,2)/fabs(should_be);
    CuAssertDblEquals(tc, 0.0 ,rel_error,1e-13);
    
    on_dim = 2;
    pt[0] = (ub-lb1)*0.1 + lb1;
    pt[1] = (ub-lb2)*0.9 + lb2;
    pt[2] = (ub-lb4)*0.4 * lb4;

    ft_1 = function_train_integrate_weighted_subset(ft,ndim_contract,&on_dim);
    eval_ft  = function_train_eval(ft_1,pt);
    function_train_free(ft_1); ft_1 = NULL;

    func2(1,pt,&e1,NULL);
    func3(1,pt+1,&e2,NULL);
    func(1,pt+2,&e3,NULL);
    should_be = e1 + e2 + e3 + oned_integrals[on_dim];

    /* printf(" Is: %G\n",eval_ft); */
    /* printf(" Should: %G\n",should_be); */
    rel_error = pow(eval_ft-should_be,2)/fabs(should_be);
    CuAssertDblEquals(tc, 0.0 ,rel_error,1e-13);
    
    on_dim = 3;
    pt[0] = (ub-lb1)*0.1 + lb1;
    pt[1] = (ub-lb2)*0.9 + lb2;
    pt[2] = (ub-lb3)*0.4 + lb3;

    ft_1 = function_train_integrate_weighted_subset(ft,ndim_contract,&on_dim);
    eval_ft  = function_train_eval(ft_1,pt);
    function_train_free(ft_1); ft_1 = NULL;

    func2(1,pt,&e1,NULL);
    func3(1,pt+1,&e2,NULL);
    func4(1,pt+2,&e3,NULL);
    should_be = e1 + e2 + e3 + oned_integrals[on_dim];

    /* printf(" Is: %G\n",eval_ft); */
    /* printf(" Should: %G\n",should_be); */
    rel_error = pow(eval_ft-should_be,2)/fabs(should_be);
    CuAssertDblEquals(tc, 0.0 ,rel_error,1e-13);


    // now check all but one
    ndim_contract = 3;
    size_t on_dim_mult[3];
    double pt_one ;
    on_dim_mult[0] = 0;
    on_dim_mult[1] = 1;
    on_dim_mult[2] = 2;

    pt_one = (ub-lb4)*0.4 + lb4;

    ft_1 = function_train_integrate_weighted_subset(ft,ndim_contract,on_dim_mult);
    eval_ft  = function_train_eval(ft_1,&pt_one);
    function_train_free(ft_1); ft_1 = NULL;

    func(1,&pt_one,&e1,NULL);
    should_be = oned_integrals[0] + oned_integrals[1] + oned_integrals[2] + e1;
    /* printf(" Is: %G\n",eval_ft); */
    /* printf(" Should: %G\n",should_be); */
    rel_error = pow(eval_ft-should_be,2)/fabs(should_be);
    CuAssertDblEquals(tc, 0.0 ,rel_error,1e-13);

    on_dim_mult[0] = 0;
    on_dim_mult[1] = 1;
    on_dim_mult[2] = 3;

    pt_one = (ub-lb3)*0.4 + lb3;

    ft_1 = function_train_integrate_weighted_subset(ft,ndim_contract,on_dim_mult);
    eval_ft  = function_train_eval(ft_1,&pt_one);
    function_train_free(ft_1); ft_1 = NULL;

    func4(1,&pt_one,&e1,NULL);
    should_be = oned_integrals[0] + oned_integrals[1] + oned_integrals[3] + e1;
    /* printf(" Is: %G\n",eval_ft); */
    /* printf(" Should: %G\n",should_be); */
    rel_error = pow(eval_ft-should_be,2)/fabs(should_be);
    CuAssertDblEquals(tc, 0.0 ,rel_error,1e-13);

    on_dim_mult[0] = 0;
    on_dim_mult[1] = 2;
    on_dim_mult[2] = 3;

    pt_one = (ub-lb2)*0.4 + lb2;

    ft_1 = function_train_integrate_weighted_subset(ft,ndim_contract,on_dim_mult);
    eval_ft  = function_train_eval(ft_1,&pt_one);
    function_train_free(ft_1); ft_1 = NULL;

    func3(1,&pt_one,&e1,NULL);
    should_be = oned_integrals[0] + oned_integrals[2] + oned_integrals[3] + e1;
    /* printf(" Is: %G\n",eval_ft); */
    /* printf(" Should: %G\n",should_be); */
    rel_error = pow(eval_ft-should_be,2)/fabs(should_be);
    CuAssertDblEquals(tc, 0.0 ,rel_error,1e-13);


    on_dim_mult[0] = 1;
    on_dim_mult[1] = 2;
    on_dim_mult[2] = 3;

    pt_one = (ub-lb1)*0.4 + lb1;

    ft_1 = function_train_integrate_weighted_subset(ft,ndim_contract,on_dim_mult);
    eval_ft  = function_train_eval(ft_1,&pt_one);
    function_train_free(ft_1); ft_1 = NULL;

    func2(1,&pt_one,&e1,NULL);
    should_be = oned_integrals[1] + oned_integrals[2] + oned_integrals[3] + e1;
    /* printf(" Is: %G\n",eval_ft); */
    /* printf(" Should: %G\n",should_be); */
    rel_error = pow(eval_ft-should_be,2)/fabs(should_be);
    CuAssertDblEquals(tc, 0.0 ,rel_error,1e-13);

    all_opts_free(fw,opts,qmopts,fopts);
    all_opts_free(NULL,opts2,qmopts2,NULL);
    all_opts_free(NULL,opts3,qmopts3,NULL);
    all_opts_free(NULL,opts4,qmopts4,NULL);
    function_train_free(ft);
}


void Test_function_train_inner(CuTest * tc)
{
    printf("Testing Function: function_train_inner \n");
    size_t dim = 4;

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw2,4);
    fwrap_set_func_array(fw2,0,func6,NULL);
    fwrap_set_func_array(fw2,1,func5,NULL);
    fwrap_set_func_array(fw2,2,func4,NULL);
    fwrap_set_func_array(fw2,3,func3,NULL);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);

    struct FunctionTrain * ft = function_train_initsum(fopts,fw);
    struct FunctionTrain * gt = function_train_initsum(fopts,fw2);
    struct FunctionTrain * ft2 =  function_train_product(gt,ft);
    
    double int1 = function_train_integrate(ft2);
    double int2 = function_train_inner(gt,ft);
    
    double relerr = pow(int1-int2,2)/pow(int1,2);
    CuAssertDblEquals(tc,0.0,relerr,1e-13);
    
    function_train_free(ft);
    function_train_free(ft2);
    function_train_free(gt);
    fwrap_destroy(fw2);
    all_opts_free(fw,opts,qmopts,fopts);
}

void Test_function_train_inner_cheb(CuTest * tc)
{
    printf("Testing Function: function_train_inner_cheb \n");
    size_t dim = 4;

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw2,4);
    fwrap_set_func_array(fw2,0,func6,NULL);
    fwrap_set_func_array(fw2,1,func5,NULL);
    fwrap_set_func_array(fw2,2,func4,NULL);
    fwrap_set_func_array(fw2,3,func3,NULL);

    struct OpeOpts * opts = ope_opts_alloc(CHEBYSHEV);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);

    struct FunctionTrain * ft = function_train_initsum(fopts,fw);
    struct FunctionTrain * gt = function_train_initsum(fopts,fw2);
    struct FunctionTrain * ft2 =  function_train_product(gt,ft);
    
    double int1 = function_train_integrate(ft2);
    double int2 = function_train_inner(gt,ft);
    
    double relerr = pow(int1-int2,2)/pow(int1,2);
    CuAssertDblEquals(tc,0.0,relerr,1e-13);
    
    function_train_free(ft);
    function_train_free(ft2);
    function_train_free(gt);
    fwrap_destroy(fw2);
    all_opts_free(fw,opts,qmopts,fopts);
}

void Test_function_train_inner_linelm(CuTest * tc)
{
    printf("Testing Function: function_train_inner with linelm \n");
    size_t dim = 4;

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,4);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    fwrap_set_func_array(fw,3,func4,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw2,4);
    fwrap_set_func_array(fw2,0,func6,NULL);
    fwrap_set_func_array(fw2,1,func5,NULL);
    fwrap_set_func_array(fw2,2,func4,NULL);
    fwrap_set_func_array(fw2,3,func3,NULL);

    size_t N = 10;
    double * x = linspace(-1.0,1.0,N);
    struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc(N,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(LINELM,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);

    struct FunctionTrain * ft = function_train_initsum(fopts,fw);
    struct FunctionTrain * gt = function_train_initsum(fopts,fw2);
    struct FunctionTrain * ft2 =  function_train_product(gt,ft);
    
    double int1 = function_train_integrate(ft2);
    double int2 = function_train_inner(gt,ft);
    
    double relerr = pow(int1-int2,2)/pow(int1,2);
    CuAssertDblEquals(tc,0.0,relerr,1e-13);
    free(x); x = NULL;
    function_train_free(ft);
    function_train_free(ft2);
    function_train_free(gt);
    fwrap_destroy(fw2);
    fwrap_destroy(fw);
    lin_elem_exp_aopts_free(opts);
    one_approx_opts_free(qmopts);
    multi_approx_opts_free(fopts);
}

void Test_ftapprox_cross(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross  (1/4)\n");
    size_t dim = 2;

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,2);
    fwrap_set_func_array(fw,0,funcnda,NULL);
    fwrap_set_func_array(fw,1,funcndb,NULL);

    // two funcnd1 is funcda + funcdb
    struct Fwrap * fw2 = fwrap_create(2,"general-vec");
    fwrap_set_fvec(fw2,funcnd1,NULL);
 
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);

    struct FunctionTrain * ftref = function_train_initsum(fopts,fw);
    
    double * yr[2];
    yr[1] = calloc_double(3);
    yr[1][0] = -1.0;
    yr[1][1] =  0.0;
    yr[1][2] =  1.0;
    yr[0] = calloc_double(3);

    size_t init_rank = 3;
    struct FtCrossArgs * fca = ft_cross_args_alloc(dim,init_rank);
    ft_cross_args_set_cross_tol(fca,1e-5);
    ft_cross_args_set_maxiter(fca,10);
    ft_cross_args_set_verbose(fca,0);
    size_t * rank = ft_cross_args_get_ranks(fca);

    struct FiberOptArgs * optim = fiber_opt_args_init(dim);

    struct CrossIndex * isl[2];
    struct CrossIndex * isr[2];
    cross_index_array_initialize(dim,isl,1,0,NULL,NULL,sizeof(double));
    cross_index_array_initialize(dim,isr,0,1,rank,(void **)yr,sizeof(double));

    struct FunctionTrain * ft = ftapprox_cross(fw2,fca,isl,isr,fopts,optim,ftref);
    
    size_t N = 20;
    double * xtest = linspace(-1,1,N);
    size_t ii,jj;
    double err = 0.0;
    double den = 0.0;
    double val;
    double pt[2];
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            pt[0] = xtest[ii];
            pt[1] = xtest[jj];
            funcnd1(1,pt,&val,NULL);
            den += pow(val,2);
            err += pow(val - function_train_eval(ft,pt),2);
        }
    }
    
    err /= den;
    CuAssertDblEquals(tc,0.0,err,1e-15);
    free(xtest);

    cross_index_free(isl[1]);
    cross_index_free(isr[0]);
    all_opts_free(fw,opts,qmopts,fopts);
    fwrap_destroy(fw2);
    ft_cross_args_free(fca);
    fiber_opt_args_free(optim);
    function_train_free(ft);
    function_train_free(ftref);
    free(yr[0]);
    free(yr[1]);

}

void Test_ftapprox_cross_fourier(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross (fourier) (1)\n");
    size_t dim = 2;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,gauss2d,NULL);

    double lb = -2.0;
    double ub = 2.0;
    size_t n =  32;
    struct OpeOpts * opts = ope_opts_alloc(FOURIER);
    ope_opts_set_lb(opts, lb);
    ope_opts_set_ub(opts, ub);
    ope_opts_set_start(opts, n+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 8;
    double ** start = malloc_dd(dim);
    // optimization stuff
    size_t N = 100;
    double * x = linspace(lb,ub,N);
    struct c3Vector * optnodes = c3vector_alloc(N,x);
    free(x); x = NULL;
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        c3approx_set_opt_opts_dim(c3a,ii,optnodes);
        start[ii] = linspace(-0.5,0.5,init_rank);
       
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    struct MultiApproxOpts * aopts = c3approx_get_approx_args(c3a);    
    c3approx_set_round_tol(c3a, 1e-8);
    
    struct FunctionTrain * ftr = c3approx_do_cross(c3a,fw,1);

    struct FunctionTrain * ft = function_train_copy(ftr);

    double diff = function_train_relnorm2diff(ft,ftr);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    /* printf("diff = %3.5G\n", diff); */

    struct FunctionTrain * ftrr = function_train_round(ftr, 1e-8, aopts);
    diff = function_train_relnorm2diff(ft, ftrr);
    /* printf("diff again = %3.5G\n", diff); */
    CuAssertDblEquals(tc,0.0,diff,1e-7);
    function_train_free(ftrr); ftrr = NULL;
    function_train_free(ftr); ftr = NULL;

    
    /* printf("final ranks = "); iprint_sz(dim+1, function_train_get_ranks(ft)); */
    N = 20;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double den = 0.0;
    double val;
    double pt[2];
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            pt[0] = xtest[ii]; pt[1] = xtest[jj];
            gauss2d(1,pt,&val,NULL);
            den += pow(val,2.0);
            err += pow(val-function_train_eval(ft,pt),2.0);
            /* double diff = pow(val-function_train_eval(ft,pt),2.0);             */
            /* printf("diff = %3.5G, val = %3.5G\n", diff, val); */
        }
    }
    err = err/den;
    CuAssertDblEquals(tc,0.0,err,1e-3);
    free(xtest);

    /* fprintf(stderr, "WARNING SERIALIZATION OF FOURIER NOT IMPLEMENTED\n"); */
    // make sure serialization works
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    /* printf("Number of bytes = %zu\n", size); */
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    /* printf("derserializing ft\n"); */
    function_train_deserialize(text, &ftd);

    /* printf("ft dim = %zu\n", ftd->dim); */
    CuAssertIntEquals(tc, dim, ftd->dim);

    size_t * ranks = function_train_get_ranks(ftd);
    size_t * ranks1 = function_train_get_ranks(ft);
    /* printf("ft ranks = "); iprint_sz(dim+1, function_train_get_ranks(ft)); */
    /* printf("ft ranks = "); iprint_sz(dim+1, ranks); */
    for (size_t ii = 0; ii < dim+1; ii++){
        CuAssertIntEquals(tc, ranks[ii], ranks1[ii]);
    }

    double inte = function_train_integrate(ftd);
    double intf = function_train_integrate(ft);
    CuAssertDblEquals(tc, intf, inte, 1e-14);

    double norm1 = function_train_norm2(ftd);
    double norm2 = function_train_norm2(ft);
    CuAssertDblEquals(tc, norm1, norm2, 1e-14);
    /* printf("%3.5G, %3.5G\n", inte, intf); */

    diff = function_train_relnorm2diff(ft,ftd);
    CuAssertDblEquals(tc,0.0,diff,1e-10);

    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;

    c3approx_destroy(c3a);
    c3vector_free(optnodes);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
    function_train_free(ft);

    fwrap_destroy(fw);
}

void Test_ftapprox_cross_fourier2(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross for fourier (2) \n");

    // set function
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funch1,NULL);

    size_t n = 8;
    struct OpeOpts * opts = ope_opts_alloc(FOURIER);
    ope_opts_set_lb(opts, -3);
    ope_opts_set_ub(opts, 3);
    ope_opts_set_start(opts, n+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 2;
    double ** start = malloc_dd(dim);
    // optimization stuff
    size_t N = 100;
    double * x = linspace(-3.0,3.0,N);
    struct c3Vector * optnodes = c3vector_alloc(N,x);
    free(x); x = NULL;
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        c3approx_set_opt_opts_dim(c3a,ii,optnodes);
        start[ii] = linspace(-1.0,1.0,init_rank);
       
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    c3approx_set_adapt_maxrank_all(c3a, 7);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);

    N = 10;
    double * xtest = linspace(-2.0,2.0,N);
    double err = 0.0;
    double den = 0.0;
    double val;
    double pt[4];
    
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            for (size_t kk = 0; kk < N; kk++){
                for (size_t ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    funch1(1,pt,&val,NULL);
                    den += pow(val,2.0);
                    err += pow(val-function_train_eval(ft,pt),2.0);
                    //printf("err=%G\n",err);
                }
            }
        }
    }
    err = err/den;
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc,0.0,err,1e-1);
    //CuAssertDblEquals(tc,0.0,0.0,1e-15);

    /* make sure serialization works */
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double diff = function_train_relnorm2diff(ft,ftd);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;

    c3approx_destroy(c3a);
    c3vector_free(optnodes);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
    function_train_free(ft);
    free(xtest);
    fwrap_destroy(fw);
}


void Test_ftapprox_cross_cheby(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross_cheby  (1/4)\n");
    size_t dim = 2;

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,2);
    fwrap_set_func_array(fw,0,funcnda,NULL);
    fwrap_set_func_array(fw,1,funcndb,NULL);

    // two funcnd1 is funcda + funcdb
    struct Fwrap * fw2 = fwrap_create(2,"general-vec");
    fwrap_set_fvec(fw2,funcnd1,NULL);
 
    struct OpeOpts * opts = ope_opts_alloc(CHEBYSHEV);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);

    struct FunctionTrain * ftref = function_train_initsum(fopts,fw);
    
    double * yr[2];
    yr[1] = calloc_double(3);
    yr[1][0] = -1.0;
    yr[1][1] =  0.0;
    yr[1][2] =  1.0;
    yr[0] = calloc_double(3);

    size_t init_rank = 3;
    struct FtCrossArgs * fca = ft_cross_args_alloc(dim,init_rank);
    ft_cross_args_set_cross_tol(fca,1e-5);
    ft_cross_args_set_maxiter(fca,10);
    ft_cross_args_set_verbose(fca,0);
    size_t * rank = ft_cross_args_get_ranks(fca);

    struct FiberOptArgs * optim = fiber_opt_args_init(dim);

    struct CrossIndex * isl[2];
    struct CrossIndex * isr[2];
    cross_index_array_initialize(dim,isl,1,0,NULL,NULL,sizeof(double));
    cross_index_array_initialize(dim,isr,0,1,rank,(void **)yr,sizeof(double));

    struct FunctionTrain * ft = ftapprox_cross(fw2,fca,isl,isr,fopts,optim,ftref);
    
    size_t N = 20;
    double * xtest = linspace(-1,1,N);
    size_t ii,jj;
    double err = 0.0;
    double den = 0.0;
    double val;
    double pt[2];
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            pt[0] = xtest[ii];
            pt[1] = xtest[jj];
            funcnd1(1,pt,&val,NULL);
            den += pow(val,2);
            err += pow(val - function_train_eval(ft,pt),2);
        }
    }
    
    err /= den;
    CuAssertDblEquals(tc,0.0,err,1e-15);
    free(xtest);

    cross_index_free(isl[1]);
    cross_index_free(isr[0]);
    all_opts_free(fw,opts,qmopts,fopts);
    fwrap_destroy(fw2);
    ft_cross_args_free(fca);
    fiber_opt_args_free(optim);
    function_train_free(ft);
    function_train_free(ftref);
    free(yr[0]);
    free(yr[1]);

}

void Test_ftapprox_cross2(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross  (2/4)\n");
     
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcnd2,NULL);
    // set function monitor

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
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

    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;
    double val;
    double pt[4];
    
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            for (size_t kk = 0; kk < N; kk++){
                for (size_t ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj];
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    funcnd2(1,pt,&val,NULL);
                    den += pow(val,2.0);
                    err += pow(val-function_train_eval(ft,pt),2.0);
                }
            }
        }
    }
    err = err/den;
    CuAssertDblEquals(tc,0.0,err,1e-10);
    free(xtest);

    function_train_free(ft);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim, start);
    fwrap_destroy(fw);
}

void Test_ftapprox_cross2_cheby(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross_cheby  (2/4)\n");
     
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcnd2,NULL);
    // set function monitor

    struct OpeOpts * opts = ope_opts_alloc(CHEBYSHEV);
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

    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;
    double val;
    double pt[4];
    
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            for (size_t kk = 0; kk < N; kk++){
                for (size_t ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj];
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    funcnd2(1,pt,&val,NULL);
                    den += pow(val,2.0);
                    err += pow(val-function_train_eval(ft,pt),2.0);
                }
            }
        }
    }
    err = err/den;
    CuAssertDblEquals(tc,0.0,err,1e-10);
    free(xtest);

    function_train_free(ft);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim, start);
    fwrap_destroy(fw);
}

void Test_ftapprox_cross3(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross  (3/4)\n");
    size_t dim = 2;

    // reference function
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,0.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);
    double slopes[2] = {0.5, 0.5};
    double offset[2] = {0.0, 0.0};
    struct FunctionTrain * ftref = function_train_linear(slopes,1,offset,1,fopts);
    all_opts_free(NULL,opts,qmopts,fopts);

    // functions
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,disc2d,NULL);
//    fwrap_set_monitoring(fw,1);

    size_t start_ranks = 2;
    struct FiberOptArgs * opt = fiber_opt_args_init(dim);
    struct FtCrossArgs * fca = ft_cross_args_alloc(dim,start_ranks);
    ft_cross_args_set_cross_tol(fca,1e-8);
    ft_cross_args_set_maxiter(fca,5);
    ft_cross_args_set_verbose(fca,0);

    double * yr[2];
    yr[1] = calloc_double(2);
    yr[1][0] = 0.3;
    yr[1][1] =  0.0;
    yr[0] = calloc_double(2);

    struct CrossIndex * isl[2];
    struct CrossIndex * isr[2];
    size_t * ranks = ft_cross_args_get_ranks(fca);
    cross_index_array_initialize(dim,isl,1,0,NULL,NULL,sizeof(double));
    cross_index_array_initialize(dim,isr,0,1,ranks,(void**)yr,sizeof(double));

    struct PwPolyOpts * aopts = pw_poly_opts_alloc(LEGENDRE,0.0,1.0);
    pw_poly_opts_set_maxorder(aopts,8);
    pw_poly_opts_set_coeffs_check(aopts,2);
    pw_poly_opts_set_tol(aopts,1e-10);
    pw_poly_opts_set_minsize(aopts,1e-8);
    pw_poly_opts_set_nregions(aopts,5);

    qmopts = one_approx_opts_alloc(PIECEWISE,aopts);
    fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);

    struct FunctionTrain * ft = ftapprox_cross(fw,fca,isl,isr,fopts,opt,ftref);


    free(yr[0]);
    free(yr[1]);

    cross_index_free(isr[0]);
    cross_index_free(isl[1]);
            
    double v1, v2;
    size_t ii,jj;
    size_t N1 = 80;
    size_t N2 = 80;
    double * xtest = linspace(0.0,1.0,N1);
    double * ytest = linspace(0.0,1.0,N2);

    double out1=0.0;
    double den=0.0;
    double pt[2];
    for (ii = 0; ii < N1; ii++){
        for (jj = 0; jj < N2; jj++){
            pt[0] = xtest[ii]; pt[1] = ytest[jj];
            disc2d(1,pt,&v1,NULL);
            v2 = function_train_eval(ft,pt);
            den += pow(v1,2.0);
            out1 += pow(v1-v2,2.0);
            //printf("f(%G,%G) = %G, pred = %G\n",pt[0],pt[1],v1,v2);
        }
    }
    free(xtest);
    free(ytest);

    double err = sqrt(out1/den);

    CuAssertDblEquals(tc,0.0,err,9e-10);

    //

    ft_cross_args_free(fca);
    fwrap_destroy(fw);
    multi_approx_opts_free(fopts);
    one_approx_opts_free(qmopts);
    fiber_opt_args_free(opt);
    pw_poly_opts_free(aopts);
    function_train_free(ftref);
    function_train_free(ft);

}

void Test_ftapprox_cross3_cheby(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross_cheby  (3/4)\n");
    size_t dim = 2;

    // reference function
    struct OpeOpts * opts = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_lb(opts,0.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);
    double slopes[2] = {0.5, 0.5};
    double offset[2] = {0.0, 0.0};
    struct FunctionTrain * ftref = function_train_linear(slopes,1,offset,1,fopts);
    all_opts_free(NULL,opts,qmopts,fopts);

    // functions
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,disc2d,NULL);
//    fwrap_set_monitoring(fw,1);

    size_t start_ranks = 2;
    struct FiberOptArgs * opt = fiber_opt_args_init(dim);
    struct FtCrossArgs * fca = ft_cross_args_alloc(dim,start_ranks);
    ft_cross_args_set_cross_tol(fca,1e-8);
    ft_cross_args_set_maxiter(fca,5);
    ft_cross_args_set_verbose(fca,0);

    double * yr[2];
    yr[1] = calloc_double(2);
    yr[1][0] = 0.3;
    yr[1][1] =  0.0;
    yr[0] = calloc_double(2);

    struct CrossIndex * isl[2];
    struct CrossIndex * isr[2];
    size_t * ranks = ft_cross_args_get_ranks(fca);
    cross_index_array_initialize(dim,isl,1,0,NULL,NULL,sizeof(double));
    cross_index_array_initialize(dim,isr,0,1,ranks,(void**)yr,sizeof(double));

    struct PwPolyOpts * aopts = pw_poly_opts_alloc(LEGENDRE,0.0,1.0);
    pw_poly_opts_set_maxorder(aopts,8);
    pw_poly_opts_set_coeffs_check(aopts,2);
    pw_poly_opts_set_tol(aopts,1e-10);
    pw_poly_opts_set_minsize(aopts,1e-8);
    pw_poly_opts_set_nregions(aopts,5);

    qmopts = one_approx_opts_alloc(PIECEWISE,aopts);
    fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);

    struct FunctionTrain * ft = ftapprox_cross(fw,fca,isl,isr,fopts,opt,ftref);


    free(yr[0]);
    free(yr[1]);

    cross_index_free(isr[0]);
    cross_index_free(isl[1]);
            
    double v1, v2;
    size_t ii,jj;
    size_t N1 = 80;
    size_t N2 = 80;
    double * xtest = linspace(0.0,1.0,N1);
    double * ytest = linspace(0.0,1.0,N2);

    double out1=0.0;
    double den=0.0;
    double pt[2];
    for (ii = 0; ii < N1; ii++){
        for (jj = 0; jj < N2; jj++){
            pt[0] = xtest[ii]; pt[1] = ytest[jj];
            disc2d(1,pt,&v1,NULL);
            v2 = function_train_eval(ft,pt);
            den += pow(v1,2.0);
            out1 += pow(v1-v2,2.0);
            //printf("f(%G,%G) = %G, pred = %G\n",pt[0],pt[1],v1,v2);
        }
    }
    free(xtest);
    free(ytest);

    double err = sqrt(out1/den);

    CuAssertDblEquals(tc,0.0,err,9e-10);

    //

    ft_cross_args_free(fca);
    fwrap_destroy(fw);
    multi_approx_opts_free(fopts);
    one_approx_opts_free(qmopts);
    fiber_opt_args_free(opt);
    pw_poly_opts_free(aopts);
    function_train_free(ftref);
    function_train_free(ft);

}

void Test_ftapprox_cross4(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross  (4/4)\n");

    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcH4,NULL);
    // set function monitor

    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,-1.0,1.0);
    pw_poly_opts_set_maxorder(opts,7);
    pw_poly_opts_set_coeffs_check(opts,2);
    pw_poly_opts_set_tol(opts,1e-3);
    pw_poly_opts_set_minsize(opts,1e-2);
    pw_poly_opts_set_nregions(opts,2);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(PIECEWISE,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 2;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(-1.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);

    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;
    double val;
    double pt[4];
    
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            for (size_t kk = 0; kk < N; kk++){
                for (size_t ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    funcH4(1,pt,&val,NULL);
                    den += pow(val,2.0);
                    err += pow(val-function_train_eval(ft,pt),2.0);
                    //printf("err=%G\n",err);
                }
            }
        }
    }
    
    err = err/den;
    CuAssertDblEquals(tc,0.0,err,1e-10);
    free(xtest);

    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
    function_train_free(ft);
    fwrap_destroy(fw);
}


void Test_ftapprox_cross4_cheby(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross_cheby  (4/4)\n");
     
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcH4,NULL);
    /* fwrap_set_fvec(fw,funcnd2,NULL); */
    // set function monitor

    struct OpeOpts * opts = ope_opts_alloc(CHEBYSHEV);
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

    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;
    double val;
    double pt[4];
    
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            for (size_t kk = 0; kk < N; kk++){
                for (size_t ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj];
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    funcH4(1,pt,&val,NULL);
                    /* funcnd2(1,pt,&val,NULL); */
                    den += pow(val,2.0);
                    err += pow(val-function_train_eval(ft,pt),2.0);
                }
            }
        }
    }
    err = err/den;
    CuAssertDblEquals(tc,0.0,err,1e-10);
    free(xtest);

    function_train_free(ft);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim, start);
    fwrap_destroy(fw);
}


void Test_function_train_eval_co_peruturb(CuTest * tc)
{
    printf("Testing Function: function_train_eval_co_perturb \n");

    // set function
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcnd2,NULL);
    
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
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

    double pt[4] = {0.5, 0.2 ,0.3, 0.8};
    double pert[8] = { 0.3, 0.6, 0.1, 0.9, 0.4, 0.6, -0.2, -0.4};
    double evals[8];
    double val = function_train_eval_co_perturb(ft,pt,pert,evals);

    double valshould;
    funcnd2(1,pt,&valshould,NULL);
    CuAssertDblEquals(tc,valshould,val,1e-13);
    
    double evals_should[8];
    double pt2[4] = {pt[0],pt[1],pt[2],pt[3]};
    
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        pt2[ii] = pert[2*ii];
        funcnd2(1,pt2,&valshould,NULL);
        evals_should[2*ii] = valshould;

        CuAssertDblEquals(tc,evals_should[2*ii],evals[2*ii],1e-13);
        
        pt2[ii] = pert[2*ii+1];
        funcnd2(1,pt2,&valshould,NULL);
        evals_should[2*ii+1] = valshould;
        CuAssertDblEquals(tc,evals_should[2*ii+1],evals[2*ii+1],1e-13);

        pt2[ii] = pt[ii];
    }

    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
    function_train_free(ft);
    fwrap_destroy(fw);
}

void Test_ftapprox_cross_hermite1(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross for hermite (1) \n");

    // set function
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funch1,NULL);
    
    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 2;
    double ** start = malloc_dd(dim);
    // optimization stuff
    size_t N = 100;
    double * x = linspace(-10.0,10.0,N);
    struct c3Vector * optnodes = c3vector_alloc(N,x);
    free(x); x = NULL;
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        c3approx_set_opt_opts_dim(c3a,ii,optnodes);
        start[ii] = linspace(-1.0,1.0,init_rank);
       
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);

    N = 10;
    double * xtest = linspace(-2.0,2.0,N);
    double err = 0.0;
    double den = 0.0;
    double val;
    double pt[4];
    
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            for (size_t kk = 0; kk < N; kk++){
                for (size_t ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    funch1(1,pt,&val,NULL);
                    den += pow(val,2.0);
                    err += pow(val-function_train_eval(ft,pt),2.0);
                    //printf("err=%G\n",err);
                }
            }
        }
    }
    err = err/den;
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc,0.0,err,1e-10);
    //CuAssertDblEquals(tc,0.0,0.0,1e-15);

    // make sure serialization works
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double diff = function_train_relnorm2diff(ft,ftd);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;

    c3approx_destroy(c3a);
    c3vector_free(optnodes);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
    function_train_free(ft);
    free(xtest);
    fwrap_destroy(fw);
}


void Test_ftapprox_cross_hermite2(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross for hermite (2) \n");
    // set function
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funch2,NULL);
    
    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 15;
    double ** start = malloc_dd(dim);
    // optimization stuff
    size_t N = 100;
    double * x = linspace(-7.0,7.0,N);
    struct c3Vector * optnodes = c3vector_alloc(N,x);
    free(x); x = NULL;
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        c3approx_set_opt_opts_dim(c3a,ii,optnodes);
        start[ii] = linspace(-10.0,10.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);

    size_t * ranks = function_train_get_ranks(ft);
    for (size_t ii = 1; ii < dim; ii++){
        CuAssertIntEquals(tc,15,ranks[ii]);
    }
    /* printf("ranks are "); iprint_sz(5,ranks); */
    double err = 0.0;
    double den = 0.0;
    double pt[4];
    
    size_t nsamples = 10000;
    for (size_t ii = 0; ii < nsamples; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randn();
        }
        double eval;
        funch2(1,pt,&eval,NULL);
        double eval2 = function_train_eval(ft,pt);
        double diff = eval- eval2;
            
        den += pow(eval,2.0);
        err += pow(diff,2);
        /* printf("pt = "); dprint(dim,pt); */
        /* printf("eval = %G, eval2=%G,diff=%G\n",eval,eval2,diff); */
        /* if (fabs(diff) > 1e-1){ */
        /*     exit(1); */
        /* } */
    }
    err = err/den;
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc,0.0,err,1e-14);

    // make sure serialization works
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double diff = function_train_relnorm2diff(ft,ftd);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;

    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    c3vector_free(optnodes);
    free_dd(dim,start);
    function_train_free(ft);
    fwrap_destroy(fw);
}

void Test_ftapprox_cross_constelm1_eval_fiber(CuTest * tc)
{
    printf("Testing Function: function_train_eval_fiber_ind for constelm \n");
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcnd2,NULL);
    // set function monitor

    size_t N = 20;
    double * x = linspace(-1.0,1.0,N);
    struct ConstElemExpAopts * opts = const_elem_exp_aopts_alloc(N,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(CONSTELM,opts);    
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


    double pt[4];
    double val;
    size_t fixed_ind[4] = {3, 12, 14, 9};
    double out[20];    
    size_t ind[20];
    for (size_t ii = 0; ii < N; ii++){
        ind[ii] = ii;
    }

    for (size_t ll = 0; ll < dim; ll++){
        size_t dim_vary = ll;    
        function_train_eval_fiber_ind(ft,fixed_ind,N,ind,dim_vary,out);
        /* printf("\n\n\n"); */

        for (size_t ii = 0; ii < N; ii++){
            pt[0] = x[fixed_ind[0]]; 
            pt[1] = x[fixed_ind[1]]; 
            pt[2] = x[fixed_ind[2]]; 
            pt[3] = x[fixed_ind[3]]; 

            pt[dim_vary] = x[ind[ii]];
            /* pt[dim_vary] = x[ii]; */

            val = function_train_eval(ft,pt);
            /* funcnd2(1,pt,&val,NULL); */
        
            /* printf("ii = %zu\n",ii); */
            /* printf("\t (val,should) = (%G,%G)\n",out[ii],val); */
            double err = out[ii] - val;
            CuAssertDblEquals(tc,0.0,err,1e-10);
        }
    }

    function_train_free(ft);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts); 
    free_dd(dim, start);
    free(x);
    fwrap_destroy(fw);
}


void Test_ftapprox_cross_constelm1(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross for constelm (1) \n");
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcnd2,NULL);
    // set function monitor

    size_t N = 20;
    double * x = linspace(-1.0,1.0,N);
    struct ConstElemExpAopts * opts = const_elem_exp_aopts_alloc(N,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(CONSTELM,opts);    
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

    N = 30;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;
    double val;
    double pt[4];
    
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            for (size_t kk = 0; kk < N; kk++){
                for (size_t ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj];
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    funcnd2(1,pt,&val,NULL);
                    den += pow(val,2.0);
                    double eval = function_train_eval(ft,pt);
                    /* printf("eval = %G val = %G\n",eval,val); */
                    err += pow(val-eval,2.0);
                }
            }
        }
    }
    /* printf("err = %G den = %G\n",err,den); */
    err = err/den;
    CuAssertDblEquals(tc,0.0,err,1e-2);
    free(xtest);

    // make sure serialization works
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double diff = function_train_relnorm2diff(ft,ftd);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;

    function_train_free(ft);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts); 
    free_dd(dim, start);
    free(x);
    fwrap_destroy(fw);
}

void Test_ftapprox_cross_constelm2(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross for constelm (2) \n");
    size_t dim = 6;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,func_not_all,NULL);
    // set function monitor

    size_t Nx = 10000;
    double * x = linspace(-1.0,1.0,Nx);
    struct ConstElemExpAopts * opts = const_elem_exp_aopts_alloc(Nx,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(CONSTELM,opts);        
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


    //printf("finished !\n");
    double pt[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double eval = function_train_eval(ft,pt);
    CuAssertDblEquals(tc,0.0,eval,1e-3);
    
    // make sure serialization works
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double diff = function_train_relnorm2diff(ft,ftd);
    /* printf("diff = %E\n",diff); */
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;

    function_train_free(ft);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim, start);
    free(x);
    fwrap_destroy(fw);

}

void Test_ftapprox_cross_constelm3(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross for constelm (3) \n");
    size_t dim = 2;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,quad2d,NULL);
    // set function monitor

    size_t N = 2000;
    double * x = linspace(-2.0,2.0,N);
    struct ConstElemExpAopts * opts = const_elem_exp_aopts_alloc(N,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(CONSTELM,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 4;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(-2.0,2.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);

    N = 30;
    double * xtest = linspace(-2.0,2.0,N);
    double err = 0.0;
    double den = 0.0;
    double val;
    double pt[2];
    
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            pt[0] = xtest[ii]; pt[1] = xtest[jj];
            quad2d(1,pt,&val,NULL);
            den += pow(val,2.0);
            double eval = function_train_eval(ft,pt);
            /* printf("eval = %G val = %G\n",eval,val); */
            err += pow(val-eval,2.0);
        }
    }
    /* printf("err = %G den = %G\n",err,den); */
    /* printf("norm = %G\n",function_train_norm2(ft)); */
    pt[0] = 0.5;
    pt[1] = -0.3;
    err = err/den;
    /* printf("eval = %G\n",function_train_eval(ft,pt)); */
    /* printf("\t Eval true = %3.15G\n",pt[0]*pt[0] + pt[1]*pt[1]); */
    CuAssertDblEquals(tc,0.0,err,1e-2);
    free(xtest);

    // make sure serialization works
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double diff = function_train_relnorm2diff(ft,ftd);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;

    function_train_free(ft);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts); 
    free_dd(dim, start);
    free(x);
    fwrap_destroy(fw);
}




void Test_ftapprox_cross_linelm1(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross for linelm (1) \n");
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcnd2,NULL);
    // set function monitor

    size_t N = 80;
    double * x = linspace(-1.0,1.0,N);
    struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc(N,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(LINELM,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 5;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(-1.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    c3approx_set_cross_tol(c3a, 1e-12);
    c3approx_set_cross_maxiter(c3a, 15);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);

    N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;
    double val;
    double pt[4];
    
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            for (size_t kk = 0; kk < N; kk++){
                for (size_t ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj];
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    funcnd2(1,pt,&val,NULL);
                    den += pow(val,2.0);
                    err += pow(val-function_train_eval(ft,pt),2.0);
                }
            }
        }
    }
    err = err/den;
    CuAssertDblEquals(tc,0.0,err,1e-10);
    free(xtest);

    // make sure serialization works
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double diff = function_train_relnorm2diff(ft,ftd);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;

    function_train_free(ft);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts); 
    free_dd(dim, start);
    free(x);
    fwrap_destroy(fw);
}


void Test_ftapprox_cross_linelm1_eval_fiber(CuTest * tc)
{
    printf("Testing Function: function_train_eval_fiber_ind for linelm \n");
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcnd2,NULL);
    // set function monitor

    size_t N = 20;
    double * x = linspace(-1.0,1.0,N);
    struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc(N,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(LINELM,opts);    
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


    double pt[4];
    double val;
    size_t fixed_ind[4] = {3, 12, 14, 9};
    double out[20];    
    size_t ind[20];
    for (size_t ii = 0; ii < N; ii++){
        ind[ii] = ii;
    }

    for (size_t ll = 0; ll < dim; ll++){
        size_t dim_vary = ll;    
        function_train_eval_fiber_ind(ft,fixed_ind,N,ind,dim_vary,out);
        /* printf("\n\n\n"); */

        for (size_t ii = 0; ii < N; ii++){
            pt[0] = x[fixed_ind[0]]; 
            pt[1] = x[fixed_ind[1]]; 
            pt[2] = x[fixed_ind[2]]; 
            pt[3] = x[fixed_ind[3]]; 

            pt[dim_vary] = x[ind[ii]];
            /* pt[dim_vary] = x[ii]; */

            val = function_train_eval(ft,pt);
            /* funcnd2(1,pt,&val,NULL); */
        
            /* printf("ii = %zu\n",ii); */
            /* printf("\t (val,should) = (%G,%G)\n",out[ii],val); */
            double err = out[ii] - val;
            CuAssertDblEquals(tc,0.0,err,1e-10);
        }
    }

    function_train_free(ft);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts); 
    free_dd(dim, start);
    free(x);
    fwrap_destroy(fw);
}

void Test_ftapprox_cross_linelm2(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross for linelm (2) \n");

    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcnd2,NULL);
    // set function monitor

    double lb = -3.0;
    double ub = 2.0;
    double delta = 1e-2;
    double hmin = 1e-2;
    struct LinElemExpAopts * opts = NULL;
    opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,lb,ub,delta,hmin);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(LINELM,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 3;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(lb,ub,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    c3approx_set_round_tol(c3a, 1e-8);
    /* c3approx_set_cross_tol(c3a, 1e-10); */
    c3approx_set_cross_maxiter(c3a, 5);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);

    size_t N = 40;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double den = 0.0;
    double val;
    double pt[4];
    
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            for (size_t kk = 0; kk < N; kk++){
                for (size_t ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj];
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    funcnd2(1,pt,&val,NULL);
                    den += pow(val,2.0);
                    double diff = pow(val-function_train_eval(ft,pt),2.0);
                    /* printf("%3.5g, %3.5g, %3.5g, %3.5g = %3.5E\n", */
                    /*        pt[0], pt[1], pt[2], pt[3], diff); */
                    err += diff;
                }
            }
        }
    }
    err = err/den;
    CuAssertDblEquals(tc,0.0,err,1e-10);
    free(xtest);

    // make sure serialization works
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double diff = function_train_relnorm2diff(ft,ftd);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;

    function_train_free(ft);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim, start);
    fwrap_destroy(fw);
}

void Test_ftapprox_cross_linelm3(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross for linelm (3) \n");
    size_t dim = 6;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,func_not_all,NULL);
    // set function monitor

    size_t N = 150;
    double * x = linspace(-1.0,1.0,N);
    struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc(N,x);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(LINELM,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 4;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(-1.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    c3approx_set_cross_maxiter(c3a, 15);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);


    //printf("finished !\n");
    double pt[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double eval = function_train_eval(ft,pt);
    CuAssertDblEquals(tc,0.0,eval,1e-13);
    
    // make sure serialization works
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double diff = function_train_relnorm2diff(ft,ftd);
    /* printf("diff = %E\n",diff); */
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;

    function_train_free(ft);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim, start);
    free(x);
    fwrap_destroy(fw);
}

void Test_sin10dint(CuTest * tc)
{
    printf("Testing Function: integration of sin10d AND (de)serialization\n");
    size_t dim = 10;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,sin10d,NULL);
    // set function monitor

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,0.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 2;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(0.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    c3approx_set_cross_tol(c3a,1e-5);
    c3approx_set_cross_maxiter(c3a,10);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);
       
    
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double intval = function_train_integrate(ftd);
    
    double should = -0.62993525905472629935874873250680615583558172687;

    double relerr = fabs(intval-should)/fabs(should);
    //printf("Relative error of integrating 10 dimensional sin = %G\n",relerr);
    CuAssertDblEquals(tc,0.0,relerr,1e-12);

    free(text);
    function_train_free(ft);
    function_train_free(ftd);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
    fwrap_destroy(fw);
}

void Test_sin10dint_savetxt(CuTest * tc)
{
    printf("Testing Function: integration of sin10d AND _savetxt _loadtxt\n");
    size_t dim = 10;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,sin10d,NULL);
    // set function monitor

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,0.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 2;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(0.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    c3approx_set_cross_tol(c3a,1e-5);
    c3approx_set_cross_maxiter(c3a,10);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);
       
    
    FILE * fp = fopen("fttest.txt","w+");
    size_t prec = 21;
    function_train_savetxt(ft,fp,prec);

    struct FunctionTrain * ftd = NULL;
    rewind(fp);
    ftd = function_train_loadtxt(fp);

    double intval = function_train_integrate(ftd);
    
    double should = -0.62993525905472629935874873250680615583558172687;

    double relerr = fabs(intval-should)/fabs(should);
    //printf("Relative error of integrating 10 dimensional sin = %G\n",relerr);
    CuAssertDblEquals(tc,0.0,relerr,1e-12);

    fclose(fp);
    function_train_free(ft);
    function_train_free(ftd);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
    fwrap_destroy(fw);
}

void Test_sin100dint(CuTest * tc)
{
    printf("Testing Function: integration of sin100d\n");
    size_t dim = 100;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,sin100d,NULL);
    // set function monitor

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,0.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 2;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(0.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    c3approx_set_cross_tol(c3a,1e-5);
    c3approx_set_cross_maxiter(c3a,10);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);

    double intval = function_train_integrate(ft);
    double should = -0.00392679526107635150777939525615131307695379649361;

    double relerr = fabs(intval-should)/fabs(should);
    printf("Relative error of integrating 100 dimensional sin = %G\n",relerr);
    CuAssertDblEquals(tc,0.0,relerr,1e-10);

    function_train_free(ft);
    fwrap_destroy(fw);
    one_approx_opts_free_deep(&qmopts);
    c3approx_destroy(c3a);
    free_dd(dim,start);
}

void Test_sin1000dint(CuTest * tc)
{
    printf("Testing Function: integration of sin1000d\n");
       
    size_t dim = 1000;
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,sin1000d,NULL);
    // set function monitor

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,0.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 2;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(0.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    c3approx_set_cross_tol(c3a,1e-5);
    c3approx_set_cross_maxiter(c3a,10);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);

    double intval = function_train_integrate(ft);
    double should = -2.6375125156875276773939642726964969819689605535e-19;

    double relerr = fabs(intval-should)/fabs(should);
    printf("Relative error of integrating 1000 dimensional sin = %G\n",relerr);
    CuAssertDblEquals(tc,0.0,relerr,1e-10);

    function_train_free(ft);
    fwrap_destroy(fw);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);    
}

CuSuite * CLinalgFuncTrainGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_function_train_grad_eval);
    SUITE_ADD_TEST(suite, Test_function_train_grad_eval2);
    SUITE_ADD_TEST(suite, Test_function_train_grad_eval3);
    SUITE_ADD_TEST(suite, Test_function_train_initsum);
    SUITE_ADD_TEST(suite, Test_function_train_linear);
    SUITE_ADD_TEST(suite, Test_function_train_quadratic);
    SUITE_ADD_TEST(suite, Test_function_train_quadratic2);
    SUITE_ADD_TEST(suite, Test_function_train_sum_function_train_round);
    SUITE_ADD_TEST(suite, Test_function_train_scale);
    SUITE_ADD_TEST(suite, Test_function_train_product);
    SUITE_ADD_TEST(suite, Test_function_train_product_cheb);
    SUITE_ADD_TEST(suite, Test_function_train_integrate);
    SUITE_ADD_TEST(suite, Test_function_train_integrate_cheb);
    SUITE_ADD_TEST(suite, Test_function_train_integrate_weighted);
    SUITE_ADD_TEST(suite, Test_function_train_integrate_weighted_subset);
    SUITE_ADD_TEST(suite, Test_function_train_inner);
    SUITE_ADD_TEST(suite, Test_function_train_inner_cheb);
    SUITE_ADD_TEST(suite, Test_function_train_inner_linelm);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_fourier);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_fourier2);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_cheby);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross2);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross2_cheby);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross3);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross3_cheby);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross4);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross4_cheby);

    SUITE_ADD_TEST(suite, Test_function_train_eval_co_peruturb);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_hermite1);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_hermite2);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_constelm1);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_constelm1_eval_fiber);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_constelm2);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_constelm3);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_linelm1);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_linelm1_eval_fiber);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_linelm2);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_linelm3);
    SUITE_ADD_TEST(suite, Test_sin10dint);
    SUITE_ADD_TEST(suite, Test_sin10dint_savetxt);
    SUITE_ADD_TEST(suite, Test_sin100dint);
    
    /* SUITE_ADD_TEST(suite, Test_sin1000dint); */
    return suite;
}


void Test_CrossIndexing(CuTest * tc)
{
   printf("Testing Function: general cross indexing functions (uncomment print statements for visual test)\n");
   size_t d = 1;
   struct CrossIndex * ci = cross_index_alloc(d);
   size_t N = 10;
   double * pts = linspace(-2.0,2.0,N);
   for (size_t ii = 0; ii < N; ii++){
       cross_index_add_index(ci,d,&(pts[ii]),sizeof(double));
   }

   CuAssertIntEquals(tc,N,ci->n);
//   print_cross_index(ci);
   
   size_t N2 = 7;
   double * pts2 = linspace(-1.5,1.5,N);
   size_t Ntot = 14;
   int newfirst = 1;
   struct CrossIndex * ci2 = cross_index_create_nested(newfirst,0,Ntot,N2,pts2,ci);
   CuAssertIntEquals(tc,Ntot,ci2->n);
//   print_cross_index(ci2);

   struct CrossIndex * ci3 = cross_index_create_nested(newfirst,1,Ntot,N2,pts2,ci2);
   CuAssertIntEquals(tc,Ntot,ci3->n);
//   printf("\n\n\nci3\n");
//   print_cross_index(ci3);

   newfirst = 0;
   struct CrossIndex * ci4 = cross_index_create_nested(newfirst,1,Ntot,N2,pts2,ci2);
   CuAssertIntEquals(tc,Ntot,ci4->n);
//   printf("\n\n\nci4\n");
//   print_cross_index(ci4);

   size_t ind[5] = {1, 3, 0, 3, 2};
   double nx[5] = {0.2, -0.8, 0.3, -1.0, 0.2};
   struct CrossIndex * ci5 = cross_index_create_nested_ind(0,5,ind,nx,ci4);
   CuAssertIntEquals(tc,5,ci5->n);
//   print_cross_index(ci5);

   double ** vals = cross_index_merge_wspace(ci3,ci4);
//   printf("merged\n");
   for (size_t ii = 0; ii < Ntot*Ntot; ii++){
//       dprint(7,vals[ii]);
       free(vals[ii]); vals[ii] = NULL;
   }
   free(vals);
   
   cross_index_free(ci); ci = NULL;
   cross_index_free(ci2); ci2 = NULL;
   cross_index_free(ci3); ci3 = NULL;
   cross_index_free(ci4); ci4 = NULL;
   cross_index_free(ci5); ci5 = NULL;
   free(pts); pts = NULL;
   free(pts2); pts2 = NULL;

}

CuSuite * CLinalgCrossIndGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_CrossIndexing);

    return suite;
 }


