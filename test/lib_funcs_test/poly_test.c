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

#include "CuTest.h"
#include "testfunctions.h"

#include "array.h"
#include "lib_linalg.h"
#include "lib_optimization.h"
#include "lib_funcs.h"

typedef struct OrthPolyExpansion* opoly_t;
#define POLY_EVAL orth_poly_expansion_eval
#define POLY_FREE orth_poly_expansion_free

static void
compute_error(double lb,double ub, size_t N, opoly_t cpoly,
              double (*func)(double,void*), void* arg,
              double * abs_err, double * func_norm)
{
    double * xtest = linspace(lb,ub,N);
    size_t ii;
    *abs_err = 0.0;
    *func_norm = 0.0;
    for (ii = 0; ii < N; ii++){
        *abs_err += pow(POLY_EVAL(cpoly,xtest[ii]) - func(xtest[ii],arg),2);
        *func_norm += pow(func(xtest[ii],arg),2);
    }
    free(xtest); xtest = NULL;
}

static void
compute_error_vec(double lb,double ub, size_t N, opoly_t cpoly,
                  int (*func)(size_t, const double *,double *,void*),
                  void* arg,
                  double * abs_err, double * func_norm)
{
    double * xtest = linspace(lb,ub,N);
    size_t ii;
    *abs_err = 0.0;
    *func_norm = 0.0;
    double val;
    for (ii = 0; ii < N; ii++){
        func(1,xtest+ii,&val,arg);
        *abs_err += pow(POLY_EVAL(cpoly,xtest[ii]) - val,2);
        *func_norm += pow(val,2);
    }
    free(xtest); xtest = NULL;
}


void Test_cheb_approx(CuTest * tc){

    printf("Testing function: cheb_approx\n");
    
    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    // approximation
    size_t N = 50;
    double lb=-1.0,ub=1.0;
    opoly_t cpoly = orth_poly_expansion_init(CHEBYSHEV,N,lb,ub);
    int res = orth_poly_expansion_approx_vec(cpoly,fw);
    CuAssertIntEquals(tc,0,res);

    // compute error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,cpoly,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    fwrap_destroy(fw);
    POLY_FREE(cpoly);
    
}

void Test_cheb_approx_nonnormal(CuTest * tc){

    printf("Testing function: cheb_approx on (a,b)\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    size_t N = 50;
    double lb = -2,ub = 3;
    opoly_t cpoly = orth_poly_expansion_init(CHEBYSHEV,N,lb,ub);
    orth_poly_expansion_approx_vec(cpoly,fw);

    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,cpoly,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);
    
    POLY_FREE(cpoly);
    fwrap_destroy(fw);
}

void Test_cheb_approx_adapt(CuTest * tc){

    printf("Testing function: cheb_approx_adapt\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    // approximation
    double lb = -1.0, ub = 1.0;
    struct OpeOpts * opts = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-8);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);

    // error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,cpoly,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);
    
    POLY_FREE(cpoly);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}

void Test_cheb_approx_adapt_weird(CuTest * tc){

    printf("Testing function: cheb_approx_adapt on (a,b)\n");
    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    // approximation
    double lb = -2.0, ub = -1.0;
    struct OpeOpts * opts = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-8);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);

    // error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,cpoly,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);
    
    POLY_FREE(cpoly);
    ope_opts_free(opts);
    fwrap_destroy(fw);

}

void Test_cheb_integrate(CuTest * tc){

    printf("Testing function: cheb_integrate2\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,powX2,NULL);

    // approximation
    double lb = -2.0, ub = 3.0;
    struct OpeOpts * opts = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-8);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);

    // error
    double intshould = (pow(ub,3) - pow(lb,3))/3;
    double intis = cheb_integrate2(cpoly);
    CuAssertDblEquals(tc, intshould, intis, 1e-13);
    
    POLY_FREE(cpoly);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}

void Test_cheb_inner(CuTest * tc){

    printf("Testing function: orth_poly_expansion_inner with chebyshev poly \n");

    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,TwoPowX3,NULL);

    double lb = -2.0, ub = 3.0;
    /* double lb = -1.0, ub = 1.0; */
    struct OpeOpts * opts = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-10);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw1);
    opoly_t cpoly2 = orth_poly_expansion_approx_adapt(opts,fw2);
    
    double intshould = (pow(ub,6) - pow(lb,6))/3;
    double intis = orth_poly_expansion_inner(cpoly,cpoly2);
    CuAssertDblEquals(tc, intshould, intis, 1e-10);
    
    POLY_FREE(cpoly);
    POLY_FREE(cpoly2);
    ope_opts_free(opts);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
}

void Test_cheb_norm(CuTest * tc){
    
    printf("Testing function: orth_poly_expansion_norm with chebyshev poly\n");

    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);
    
    double lb = -2, ub = 3.0;
    struct OpeOpts * opts = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-8);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);

    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw1);
    
    double intshould = (pow(ub,5) - pow(lb,5))/5;
    double intis = orth_poly_expansion_norm(cpoly);
    CuAssertDblEquals(tc, sqrt(intshould), intis, 1e-10);
    
    POLY_FREE(cpoly);
    ope_opts_free(opts);
    fwrap_destroy(fw1);
}

/* void Test_cheb_product(CuTest * tc){ */

/*     printf("Testing function: orth_poly_expansion_product with chebyshev poly \n"); */
    
/*     // function */
/*     struct Fwrap * fw1 = fwrap_create(1,"general-vec"); */
/*     fwrap_set_fvec(fw1,powX2,NULL); */

/*     struct Fwrap * fw2 = fwrap_create(1,"general-vec"); */
/*     fwrap_set_fvec(fw2,TwoPowX3,NULL); */

/*     // approximation */
/*     double lb = -3.0, ub = 2.0; */
/*     struct OpeOpts * opts = ope_opts_alloc(CHEBYSHEV); */
/*     ope_opts_set_start(opts,10); */
/*     ope_opts_set_coeffs_check(opts,4); */
/*     ope_opts_set_tol(opts,1e-10); */
/*     ope_opts_set_lb(opts,lb); */
/*     ope_opts_set_ub(opts,ub); */
    
/*     opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw1); */
/*     opoly_t cpoly2 = orth_poly_expansion_approx_adapt(opts,fw2); */
/*     opoly_t cpoly3 = orth_poly_expansion_prod(cpoly,cpoly2); */

/*     size_t N = 100; */
/*     double * pts = linspace(lb,ub,N); */
/*     size_t ii; */
/*     for (ii = 0; ii < N; ii++){ */
/*         double eval1 = POLY_EVAL(cpoly3,pts[ii]); */
/*         double eval2 = POLY_EVAL(cpoly,pts[ii]) * */
/*                         POLY_EVAL(cpoly2,pts[ii]); */
/*         double diff= fabs(eval1-eval2); */
/*         CuAssertDblEquals(tc, 0.0, diff, 1e-10); */
/*     } */
/*     free(pts); pts = NULL; */
    
/*     POLY_FREE(cpoly); */
/*     POLY_FREE(cpoly2); */
/*     POLY_FREE(cpoly3); */
/*     ope_opts_free(opts); */
/*     fwrap_destroy(fw1); */
/*     fwrap_destroy(fw2); */
/* } */

void Test_cheb_orth_poly_expansion_create_with_params_and_grad(CuTest * tc){
    
    printf("Testing functions: orth_poly_expansion_create_with_params and grad with chebyshev poly\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * opts = ope_opts_alloc(CHEBYSHEV);
    struct OrthPolyExpansion * ope = NULL;
    ope = orth_poly_expansion_create_with_params(opts,nparams,params);


    double grad[10];
    double xloc = 0.4;
    int res = 
        orth_poly_expansion_param_grad_eval(ope,1,&xloc,grad);
    CuAssertIntEquals(tc,0,res);


    // numerical derivative
    struct OrthPolyExpansion * ope1 = NULL;
    struct OrthPolyExpansion * ope2 = NULL;

    size_t dim = nparams;
    double * x1 = calloc_double(dim);
    double * x2 = calloc_double(dim);
    for (size_t ii = 0; ii < dim; ii++){
        x1[ii] = params[ii];
        x2[ii] = params[ii];
    }
    
    double diff = 0.0;
    double v1,v2;
    double norm = 0.0;
    double eps = 1e-8;
    for (size_t ii = 0; ii < dim; ii++){
        x1[ii] += eps;
        x2[ii] -= eps;
        ope1 = orth_poly_expansion_create_with_params(opts,nparams,x1);
        v1 = orth_poly_expansion_eval(ope1,xloc);

        ope2 = orth_poly_expansion_create_with_params(opts,nparams,x2);
        v2 = orth_poly_expansion_eval(ope2,xloc);

        double diff_iter = pow( (v1-v2)/2.0/eps - grad[ii], 2 );
        /* printf("current diff = %G\n",diff_iter); */
        /* printf("\t norm = %G\n",grad[ii]); */
        diff += diff_iter;
        norm += pow( (v1-v2)/2.0/eps,2);
        
        x1[ii] -= eps;
        x2[ii] += eps;

        orth_poly_expansion_free(ope1); ope1 = NULL;
        orth_poly_expansion_free(ope2); ope2 = NULL;
    }
    if (norm > 1){
        diff /= norm;
    }
    free(x1); x1 = NULL;
    free(x2); x2 = NULL;
    CuAssertDblEquals(tc,0.0,diff,1e-7);
    
    ope_opts_free(opts); opts = NULL;
    orth_poly_expansion_free(ope); ope = NULL;
}


CuSuite * ChebGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_cheb_approx);
    SUITE_ADD_TEST(suite, Test_cheb_approx_nonnormal); 
    SUITE_ADD_TEST(suite, Test_cheb_approx_adapt); 
    SUITE_ADD_TEST(suite, Test_cheb_approx_adapt_weird); 
    SUITE_ADD_TEST(suite, Test_cheb_integrate); 
    SUITE_ADD_TEST(suite, Test_cheb_inner);
    SUITE_ADD_TEST(suite, Test_cheb_norm);
    SUITE_ADD_TEST(suite, Test_cheb_orth_poly_expansion_create_with_params_and_grad);
    /* SUITE_ADD_TEST(suite, Test_cheb_product);  */

    return suite;
}

///////////////////////////////////////////////////////////////////////////

void Test_legendre_approx(CuTest * tc){

    printf("Testing function: legendre_approx\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    // approximation
    size_t N = 50;
    double lb=-1.0,ub=1.0;
    opoly_t cpoly = orth_poly_expansion_init(LEGENDRE,N,lb,ub);
    int res = orth_poly_expansion_approx_vec(cpoly,fw);
    CuAssertIntEquals(tc,0,res);

    // compute error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,cpoly,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    fwrap_destroy(fw);
    POLY_FREE(cpoly);
}

void Test_legendre_approx_nonnormal(CuTest * tc){

    printf("Testing function: legendre_approx on (a,b)\n");

        // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    // approximation
    size_t N = 50;
    double lb = -2,ub = 3;
    opoly_t cpoly = orth_poly_expansion_init(LEGENDRE,N,lb,ub);
    int res = orth_poly_expansion_approx_vec(cpoly,fw);
    CuAssertIntEquals(tc,0,res);

    // compute error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,cpoly,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    fwrap_destroy(fw);
    POLY_FREE(cpoly);
}

void Test_legendre_approx_adapt(CuTest * tc){

    printf("Testing function: legendre_approx_adapt\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    // approximation
    double lb = -1.0, ub = 1.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-8);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);

    // error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,cpoly,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);
    
    POLY_FREE(cpoly);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}

void Test_legendre_approx_adapt_weird(CuTest * tc){

    printf("Testing function: legendre_approx_adapt on (a,b)\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    // approximation
    double lb = -2.0, ub = -1.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-8);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);

    // error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,cpoly,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);
    
    POLY_FREE(cpoly);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}

void Test_legendre_derivative_consistency(CuTest * tc)
{
    printf("Testing functions: legen_deriv and legen_deriv_upto  on (a,b)\n");

    size_t order = 10;
    double x = 0.5;
    double * derivvals = orth_poly_deriv_upto(LEGENDRE,order,x);
     
    size_t ii;
    for (ii = 0; ii < order+1; ii++){
        double val = deriv_legen(x,ii);
        CuAssertDblEquals(tc,val, derivvals[ii],1e-14);
    }
    free(derivvals); derivvals = NULL;
}

void Test_legendre_derivative(CuTest * tc){

    printf("Testing function: orth_poly_expansion_deriv  on (a,b)\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    // approximation
    double lb = -2.0, ub = -1.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-8);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);
    opoly_t der = orth_poly_expansion_deriv(cpoly);
    
    // error
    double abs_err;
    double func_norm;
    compute_error(lb,ub,1000,der,funcderiv,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);
    
    POLY_FREE(cpoly);
    POLY_FREE(der);
    ope_opts_free(opts);
    fwrap_destroy(fw);
    
}

void Test_legendre_integrate(CuTest * tc){

    printf("Testing function: legendre_integrate\n");
    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,powX2,NULL);

    // approximation
    double lb = -2.0, ub = -3.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-8);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);

    double intshould = (pow(ub,3) - pow(lb,3))/3;
    double intis = legendre_integrate(cpoly);
    CuAssertDblEquals(tc, intshould, intis, 1e-13);
    
    POLY_FREE(cpoly);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}

void Test_legendre_integrate_weighted(CuTest * tc){

    printf("Testing function: legendre_integrate_weighted\n");
    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,x3minusx,NULL);

    // approximation
    double lb = -3.0, ub = 9.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-8);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);

    double ubterm = 3.0/4.0 * pow(ub,4) - 1.0/2.0 * pow(ub,2);
    double lbterm = 3.0/4.0 * pow(lb,4) - 1.0/2.0 * pow(lb,2);
    double intshould = (ubterm-lbterm)/(ub-lb);
    double intis = orth_poly_expansion_integrate_weighted(cpoly);
    CuAssertDblEquals(tc, intshould, intis, 1e-13);
    
    POLY_FREE(cpoly);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}


void Test_legendre_inner(CuTest * tc){

    printf("Testing function: orth_poly_expansion_inner with legendre poly \n");

    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,TwoPowX3,NULL);

    double lb = -2.0, ub = 3.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-10);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw1);
    opoly_t cpoly2 = orth_poly_expansion_approx_adapt(opts,fw2);
    
    double intshould = (pow(ub,6) - pow(lb,6))/3;
    double intis = orth_poly_expansion_inner(cpoly,cpoly2);
    CuAssertDblEquals(tc, intshould, intis, 1e-10);
    
    POLY_FREE(cpoly);
    POLY_FREE(cpoly2);
    ope_opts_free(opts);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
}

void Test_legendre_inner_w(CuTest * tc){

    printf("Testing function: orth_poly_expansion_inner with legendre poly \n");

    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,TwoPowX3,NULL);

    double lb = -2.0, ub = 3.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-10);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw1);
    opoly_t cpoly2 = orth_poly_expansion_approx_adapt(opts,fw2);
    
    double intshould = (pow(ub,6) - pow(lb,6))/3/5;
    double intis = orth_poly_expansion_inner_w(cpoly,cpoly2);
    CuAssertDblEquals(tc, intshould, intis, 1e-10);
    
    POLY_FREE(cpoly);
    POLY_FREE(cpoly2);
    ope_opts_free(opts);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
}

void Test_legendre_norm_w(CuTest * tc){
    
    printf("Testing function: orth_poly_expansion_norm_w with legendre poly\n");

    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);
    
    double lb = -2, ub = 3.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-8);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);

    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw1);
    
    double intshould = sqrt((pow(ub,5) - pow(lb,5))/5/5);
    double intis = orth_poly_expansion_norm_w(cpoly);
    CuAssertDblEquals(tc, sqrt(intshould), intis, 1e-10);
    
    POLY_FREE(cpoly);
    ope_opts_free(opts);
    fwrap_destroy(fw1);
}

void Test_legendre_norm(CuTest * tc){
    
    printf("Testing function: orth_poly_expansion_norm with legendre poly\n");

        // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);
    
    double lb = -2, ub = 3.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-8);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);

    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw1);
    
    double intshould = (pow(ub,5) - pow(lb,5))/5.0;
    double intis = orth_poly_expansion_norm(cpoly);
    CuAssertDblEquals(tc, sqrt(intshould), intis, 1e-10);
    
    POLY_FREE(cpoly);
    ope_opts_free(opts);
    fwrap_destroy(fw1);
}

void Test_legendre_product(CuTest * tc){

    printf("Testing function: orth_poly_expansion_product with legendre poly \n");
    
    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,TwoPowX3,NULL);

    // approximation
    double lb = -3.0, ub = 2.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-10);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw1);
    opoly_t cpoly2 = orth_poly_expansion_approx_adapt(opts,fw2);
    opoly_t cpoly3 = orth_poly_expansion_prod(cpoly,cpoly2);

    size_t N = 100;
    double * pts = linspace(lb,ub,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = POLY_EVAL(cpoly3,pts[ii]);
        double eval2 = POLY_EVAL(cpoly,pts[ii]) *
                        POLY_EVAL(cpoly2,pts[ii]);
        double diff= fabs(eval1-eval2);
        CuAssertDblEquals(tc, 0.0, diff, 1e-10);
    }
    free(pts); pts = NULL;
    
    POLY_FREE(cpoly);
    POLY_FREE(cpoly2);
    POLY_FREE(cpoly3);
    ope_opts_free(opts);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
}

void Test_legendre_axpy(CuTest * tc){

    printf("Testing function: orth_poly_expansion_axpy with legendre poly \n");

    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,TwoPowX3,NULL);

    // approximation
    double lb = -3.0, ub = 2.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-10);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw1);
    opoly_t cpoly2 = orth_poly_expansion_approx_adapt(opts,fw2);
        
    
    int success = orth_poly_expansion_axpy(2.0,cpoly2,cpoly);
    CuAssertIntEquals(tc,0,success);
    
    size_t N = 100;
    double * pts = linspace(lb,ub,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = POLY_EVAL(cpoly,pts[ii]);
        double eval2a,eval2b;
        TwoPowX3(1,pts+ii,&eval2a,NULL);
        powX2(1,pts+ii,&eval2b,NULL);
        double eval2 = 2.0 * eval2a + eval2b;
        double diff= fabs(eval1-eval2);
        CuAssertDblEquals(tc, 0.0, diff, 1e-10);
    }
    free(pts); pts = NULL;
    
    POLY_FREE(cpoly);
    POLY_FREE(cpoly2);
    ope_opts_free(opts);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
}

void Test_legendre_orth_poly_expansion_create_with_params_and_grad(CuTest * tc){
    
    printf("Testing functions: orth_poly_expansion_create_with_params and grad with legendre poly\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OrthPolyExpansion * ope = NULL;
    ope = orth_poly_expansion_create_with_params(opts,nparams,params);


    double grad[10];
    double xloc = 0.4;
    int res = 
        orth_poly_expansion_param_grad_eval(ope,1,&xloc,grad);
    CuAssertIntEquals(tc,0,res);


    // numerical derivative
    struct OrthPolyExpansion * ope1 = NULL;
    struct OrthPolyExpansion * ope2 = NULL;

    size_t dim = nparams;
    double * x1 = calloc_double(dim);
    double * x2 = calloc_double(dim);
    for (size_t ii = 0; ii < dim; ii++){
        x1[ii] = params[ii];
        x2[ii] = params[ii];
    }
    
    double diff = 0.0;
    double v1,v2;
    double norm = 0.0;
    double eps = 1e-8;
    for (size_t ii = 0; ii < dim; ii++){
        x1[ii] += eps;
        x2[ii] -= eps;
        ope1 = orth_poly_expansion_create_with_params(opts,nparams,x1);
        v1 = orth_poly_expansion_eval(ope1,xloc);

        ope2 = orth_poly_expansion_create_with_params(opts,nparams,x2);
        v2 = orth_poly_expansion_eval(ope2,xloc);

        double diff_iter = pow( (v1-v2)/2.0/eps - grad[ii], 2 );
        /* printf("current diff = %G\n",diff_iter); */
        /* printf("\t norm = %G\n",grad[ii]); */
        diff += diff_iter;
        norm += pow( (v1-v2)/2.0/eps,2);
        
        x1[ii] -= eps;
        x2[ii] += eps;

        orth_poly_expansion_free(ope1); ope1 = NULL;
        orth_poly_expansion_free(ope2); ope2 = NULL;
    }
    if (norm > 1){
        diff /= norm;
    }
    free(x1); x1 = NULL;
    free(x2); x2 = NULL;
    CuAssertDblEquals(tc,0.0,diff,1e-7);
    
    ope_opts_free(opts); opts = NULL;
    orth_poly_expansion_free(ope); ope = NULL;
}


CuSuite * LegGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_legendre_approx);
    SUITE_ADD_TEST(suite, Test_legendre_approx_nonnormal);
    SUITE_ADD_TEST(suite, Test_legendre_approx_adapt);
    SUITE_ADD_TEST(suite, Test_legendre_approx_adapt_weird);
    SUITE_ADD_TEST(suite, Test_legendre_derivative);
    SUITE_ADD_TEST(suite, Test_legendre_derivative_consistency);
    SUITE_ADD_TEST(suite, Test_legendre_integrate);
    SUITE_ADD_TEST(suite, Test_legendre_integrate_weighted);
    SUITE_ADD_TEST(suite, Test_legendre_inner);
    SUITE_ADD_TEST(suite, Test_legendre_inner_w);
    SUITE_ADD_TEST(suite, Test_legendre_norm_w);
    SUITE_ADD_TEST(suite, Test_legendre_norm);
    SUITE_ADD_TEST(suite, Test_legendre_product);
    SUITE_ADD_TEST(suite, Test_legendre_axpy);
    SUITE_ADD_TEST(suite, Test_legendre_orth_poly_expansion_create_with_params_and_grad);

    return suite;
}

int fherm1(size_t N, const double * x, double * out, void * arg)
{
    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        out[ii]= x[ii] + x[ii]*x[ii];
    }
    return 0;     
}

void Test_hermite_approx(CuTest * tc){

    printf("Testing function: hermite_approx\n");
    
    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,fherm1,NULL);

    // approximation
    size_t N = 20;
    opoly_t cpoly = orth_poly_expansion_init(HERMITE,N,-DBL_MAX,DBL_MAX);
    int res = orth_poly_expansion_approx_vec(cpoly,fw);
    CuAssertIntEquals(tc,0,res);

    // compute error
    double abs_err;
    double func_norm;
    compute_error_vec(-1.0,1.0,1000,cpoly,fherm1,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    fwrap_destroy(fw);
    POLY_FREE(cpoly);
}

int fherm2(size_t N, const double * x, double * out, void * arg)
{
    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        out[ii]= sin(2.0*x[ii]);
    }
    return 0;     
}


void Test_hermite_approx_adapt(CuTest * tc)
{
    printf("Testing function: hermite_approx_adapt\n");
    
    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,fherm2,NULL);

    // approximation
    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-15);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);

    // error
    double abs_err;
    double func_norm;
    compute_error_vec(-1.0,1.0,1000,cpoly,fherm2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-10);
    
    POLY_FREE(cpoly);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}

/* void Test_hermite_derivative_consistency(CuTest * tc) */
/* { */
/*     printf("Testing functions: legen_deriv and legen_deriv_upto  on (a,b)\n"); */

/*     size_t order = 10; */
/*     double x = 0.5; */
/*     double * derivvals = orth_poly_deriv_upto(HERMITE,order,x); */
     
/*     size_t ii; */
/*     for (ii = 0; ii < order+1; ii++){ */
/*         double val = deriv_legen(x,ii); */
/*         //printf("consistency ii=%zu\n",ii); */
/*         //printf("in arr = %G, loner = %G \n ", val, derivvals[ii]); */
/*         CuAssertDblEquals(tc,val, derivvals[ii],1e-14); */
/*         //printf("got it\n"); */
/*     } */
/*     free(derivvals); derivvals = NULL; */
/* } */

/* void Test_hermite_derivative(CuTest * tc){ */

/*     printf("Testing function: orth_poly_expansion_deriv  on (a,b)\n"); */
/*     double lb = -2.0; */
/*     double ub = -1.0; */

/*     struct OpeOpts opts; */
/*     opts.start_num = 10; */
/*     opts.coeffs_check= 4; */
/*     opts.tol = 1e-9; */

/*     struct counter c; */
/*     c.N = 0; */
/*     opoly_t  cpoly = orth_poly_expansion_approx_adapt(func, &c,  */
/*                             HERMITE,lb,ub, &opts); */
    
/*     opoly_t  der = orth_poly_expansion_deriv(cpoly); */

/*     size_t N = 100; */
/*     double * xtest = linspace(lb,ub,N); */
/*     size_t ii; */
/*     double err = 0.0; */
/*     double errNorm = 0.0; */
/*     for (ii = 0; ii < N; ii++){ */
/*         err += pow(POLY_EVAL(der,xtest[ii]) - funcderiv(xtest[ii], NULL),2); */
/*         errNorm += pow(funcderiv(xtest[ii],NULL),2); */

/*         //printf("pt= %G err = %G \n",xtest[ii], err); */
/*     } */
/*     //printf("num polys adapted=%zu\n",cpoly->num_poly); */
/*     err = err / errNorm; */
/*     //printf("err = %G\n",err); */
/*     CuAssertDblEquals(tc, 0.0, err, 1e-12); */
/*     POLY_FREE(cpoly); */
/*     POLY_FREE(der); */
/*     free(xtest); */
/* } */

int fherm3(size_t N, const double * x, double * out, void * arg)
{
    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        out[ii]= sin(2*x[ii]+3) + 3*pow(x[ii],3);
    }
    return 0;     
}

void Test_hermite_integrate(CuTest * tc){

    printf("Testing function: hermite_integrate\n");
    
    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,fherm3,NULL);

    // approximation
    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-8);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);

    double intshould = sin(3)/exp(2);
    double intis = hermite_integrate(cpoly);
    CuAssertDblEquals(tc, intshould, intis, 1e-13);
    
    POLY_FREE(cpoly);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}


int fherm4(size_t N, const double * x, double * out, void * arg)
{
    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = sin(2.0*x[ii]+3.0); 
    }
    return 0;     
}

int fherm5(size_t N, const double * x, double * out, void * arg)
{
    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = 3*pow(x[ii],3);
    }
    return 0;     
}

void Test_hermite_inner(CuTest * tc){

    printf("Testing function: orth_poly_expansion_inner with hermite poly \n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,fherm4,NULL);
    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,fherm5,NULL);

    // approximation
    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-15);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);
    opoly_t cpoly2 = orth_poly_expansion_approx_adapt(opts,fw2);

    double intshould = -6.0*cos(3)/exp(2.0);
    double intis = orth_poly_expansion_inner(cpoly,cpoly2);
    CuAssertDblEquals(tc, intshould, intis, 1e-10);
    
    POLY_FREE(cpoly);
    POLY_FREE(cpoly2);
    ope_opts_free(opts);
    fwrap_destroy(fw);
    fwrap_destroy(fw2);
}

int fherm6(size_t N, const double * x, double * out, void * arg)
{
    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = pow(x[ii],2)*sin(x[ii]+0.5); 
    }
    return 0;     
}

void Test_hermite_norm_w(CuTest * tc){
    
    printf("Testing function: orth_poly_expansion_norm_w with hermite poly\n");
    
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,fherm6,NULL);

    // approximation
    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-15);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);

    double intshould = sqrt(5*cos(1)/2/exp(2) + 3.0/2.0);
    double intis = orth_poly_expansion_norm_w(cpoly);
    CuAssertDblEquals(tc, sqrt(intshould), intis, 1e-13);
    
    POLY_FREE(cpoly);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}

void Test_hermite_norm(CuTest * tc){
    
    printf("Testing function: orth_poly_expansion_norm with hermite poly\n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,fherm6,NULL);

    // approximation
    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-15);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);

    double intshould = sqrt(5*cos(1)/2/exp(2) + 3.0/2.0);
    double intis = orth_poly_expansion_norm_w(cpoly);
    CuAssertDblEquals(tc, sqrt(intshould), intis, 1e-13);
    
    POLY_FREE(cpoly);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}


int fherm7(size_t N, const double * x, double * out, void * arg)
{
    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = pow(x[ii],2);
    }
    return 0;     
}

int fherm8(size_t N, const double * x, double * out, void * arg)
{
    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = 2.0 + 3.0*pow(x[ii],5);
    }
    return 0;     
}

void Test_hermite_product(CuTest * tc){

    printf("Testing function: orth_poly_expansion_product with hermite poly \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,fherm7,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,fherm8,NULL);

    // approximation
    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-15);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);
    opoly_t cpoly2 = orth_poly_expansion_approx_adapt(opts,fw2);
    opoly_t  cpoly3 = orth_poly_expansion_prod(cpoly,cpoly2);
    
    size_t N = 100;
    double * pts = linspace(-1,1,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = POLY_EVAL(cpoly3,pts[ii]);
        double eval2 = POLY_EVAL(cpoly,pts[ii]) *
                        POLY_EVAL(cpoly2,pts[ii]);
        double diff= fabs(eval1-eval2);
        CuAssertDblEquals(tc, 0.0, diff, 1e-10);
    }

    free(pts); pts = NULL;
    POLY_FREE(cpoly);
    POLY_FREE(cpoly2);
    POLY_FREE(cpoly3);
    ope_opts_free(opts);
    fwrap_destroy(fw);
    fwrap_destroy(fw2);
}

void Test_hermite_axpy(CuTest * tc){

    printf("Testing function: orth_poly_expansion_axpy with hermite poly \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,fherm6,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,fherm7,NULL);

    // approximation
    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-15);
    opoly_t cpoly = orth_poly_expansion_approx_adapt(opts,fw);
    opoly_t cpoly2 = orth_poly_expansion_approx_adapt(opts,fw2);
    int success = orth_poly_expansion_axpy(2.0,cpoly2,cpoly);
    CuAssertIntEquals(tc,0,success);

    size_t N = 100;
    double * pts = linspace(-1,1,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = POLY_EVAL(cpoly,pts[ii]);
        double eval2a,eval2b;
        fherm7(1,pts+ii,&eval2a,NULL);
        fherm6(1,pts+ii,&eval2b,NULL);
        double eval2 = 2.0 * eval2a + eval2b;
        double diff= fabs(eval1-eval2);
        CuAssertDblEquals(tc, 0.0, diff, 1e-7);
    }
    free(pts); pts = NULL;
    
    POLY_FREE(cpoly);
    POLY_FREE(cpoly2);
    ope_opts_free(opts);
    fwrap_destroy(fw);
    fwrap_destroy(fw2);
}

void Test_hermite_linear(CuTest * tc){

    printf("Testing function: orth_poly_expansion_linear with hermite poly \n");
    
    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    double a = 2.0, offset=3.0;
    opoly_t poly = orth_poly_expansion_linear(a,offset,opts);
    
    size_t N = 100;
    double * pts = linspace(-1,1,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = POLY_EVAL(poly,pts[ii]);
        double eval2 = a*pts[ii] + offset;
        double diff= fabs(eval1-eval2);
        CuAssertDblEquals(tc, 0.0, diff, 1e-7);
    }
    free(pts); pts = NULL;
    
    POLY_FREE(poly);
    ope_opts_free(opts);
}

void Test_hermite_quadratic(CuTest * tc){

    printf("Testing function: orth_poly_expansion_quadratic with hermite poly \n");
 
    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    double a = 2.0, offset=3.0;
    opoly_t poly = orth_poly_expansion_quadratic(a,offset,opts);
 
    size_t N = 100;
    double * pts = linspace(-1,1,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = POLY_EVAL(poly,pts[ii]);
        double eval2 = a*pow(pts[ii] - offset,2);
        double diff= fabs(eval1-eval2);
        CuAssertDblEquals(tc, 0.0, diff, 1e-7);
    }
    free(pts); pts = NULL;
    
    POLY_FREE(poly);
    ope_opts_free(opts);
}

void Test_hermite_orth_poly_expansion_create_with_params_and_grad(CuTest * tc){
    
    printf("Testing functions: orth_poly_expansion_create_with_params and grad with hermite poly\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OrthPolyExpansion * ope = NULL;
    ope = orth_poly_expansion_create_with_params(opts,nparams,params);


    double grad[10];
    double xloc = 0.4;
    int res = 
        orth_poly_expansion_param_grad_eval(ope,1,&xloc,grad);
    CuAssertIntEquals(tc,0,res);


    // numerical derivative
    struct OrthPolyExpansion * ope1 = NULL;
    struct OrthPolyExpansion * ope2 = NULL;

    size_t dim = nparams;
    double * x1 = calloc_double(dim);
    double * x2 = calloc_double(dim);
    for (size_t ii = 0; ii < dim; ii++){
        x1[ii] = params[ii];
        x2[ii] = params[ii];
    }
    
    double diff = 0.0;
    double v1,v2;
    double norm = 0.0;
    double eps = 1e-8;
    for (size_t ii = 0; ii < dim; ii++){
        x1[ii] += eps;
        x2[ii] -= eps;
        ope1 = orth_poly_expansion_create_with_params(opts,nparams,x1);
        v1 = orth_poly_expansion_eval(ope1,xloc);

        ope2 = orth_poly_expansion_create_with_params(opts,nparams,x2);
        v2 = orth_poly_expansion_eval(ope2,xloc);

        double diff_iter = pow( (v1-v2)/2.0/eps - grad[ii], 2 );
        /* printf("current diff = %G\n",diff_iter); */
        /* printf("\t norm = %G\n",grad[ii]); */
        diff += diff_iter;
        norm += pow( (v1-v2)/2.0/eps,2);
        
        x1[ii] -= eps;
        x2[ii] += eps;

        orth_poly_expansion_free(ope1); ope1 = NULL;
        orth_poly_expansion_free(ope2); ope2 = NULL;
    }
    if (norm > 1){
        diff /= norm;
    }
    free(x1); x1 = NULL;
    free(x2); x2 = NULL;
    CuAssertDblEquals(tc,0.0,diff,1e-7);
    
    ope_opts_free(opts); opts = NULL;
    orth_poly_expansion_free(ope); ope = NULL;
}


CuSuite * HermGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_hermite_approx);
    SUITE_ADD_TEST(suite, Test_hermite_approx_adapt);
    SUITE_ADD_TEST(suite, Test_hermite_integrate);
    SUITE_ADD_TEST(suite, Test_hermite_inner);
    SUITE_ADD_TEST(suite, Test_hermite_norm_w);
    SUITE_ADD_TEST(suite, Test_hermite_norm);
    SUITE_ADD_TEST(suite, Test_hermite_product);
    SUITE_ADD_TEST(suite, Test_hermite_axpy);
    SUITE_ADD_TEST(suite, Test_hermite_linear);
    SUITE_ADD_TEST(suite, Test_hermite_quadratic);
    SUITE_ADD_TEST(suite, Test_hermite_orth_poly_expansion_create_with_params_and_grad);
    return suite;
}


void Test_orth_to_standard_poly(CuTest * tc){
    
    printf("Testing function: orth_to_standard_poly \n");
    struct OrthPoly * leg = init_leg_poly();
    struct OrthPoly * cheb = init_cheb_poly();
    
    struct StandardPoly * p = orth_to_standard_poly(leg,0);
    CuAssertDblEquals(tc, 1.0, p->coeff[0], 1e-13);
    standard_poly_free(p);

    p = orth_to_standard_poly(leg,1);
    CuAssertDblEquals(tc, 0.0, p->coeff[0], 1e-13);
    CuAssertDblEquals(tc, 1.0, p->coeff[1], 1e-13);
    standard_poly_free(p);

    p = orth_to_standard_poly(leg,5);
    CuAssertDblEquals(tc, 0.0, p->coeff[0], 1e-13);
    CuAssertDblEquals(tc, 15.0/8.0, p->coeff[1], 1e-13);
    CuAssertDblEquals(tc, 0.0, p->coeff[2], 1e-13);
    CuAssertDblEquals(tc, -70.0/8.0, p->coeff[3], 1e-13);
    CuAssertDblEquals(tc, 0.0, p->coeff[4], 1e-13);
    CuAssertDblEquals(tc, 63.0/8.0, p->coeff[5], 1e-13);
    standard_poly_free(p);
    
    p = orth_to_standard_poly(cheb,5);
    CuAssertDblEquals(tc, 0.0, p->coeff[0], 1e-13);
    CuAssertDblEquals(tc, 5.0, p->coeff[1], 1e-13);
    CuAssertDblEquals(tc, 0.0, p->coeff[2], 1e-13);
    CuAssertDblEquals(tc, -20.0, p->coeff[3], 1e-13);
    CuAssertDblEquals(tc, 0.0, p->coeff[4], 1e-13);
    CuAssertDblEquals(tc, 16.0, p->coeff[5], 1e-13);
    standard_poly_free(p);


    free_orth_poly(leg);
    free_orth_poly(cheb);
}

void Test_orth_poly_expansion_to_standard_poly(CuTest * tc){
    
    printf("Testing function: orth_poly_expansion_to_standard_poly \n");
    
    struct OrthPolyExpansion * pl =
            orth_poly_expansion_init(LEGENDRE,10,-1.0,1.0);
    pl->coeff[0] = 5.0;
    pl->coeff[4] = 2.0;
    pl->coeff[7] = 3.0;

    struct StandardPoly * p = orth_poly_expansion_to_standard_poly(pl);

    CuAssertDblEquals(tc, 5.0 + 2.0 *3.0/8.0, p->coeff[0], 1e-13);
    CuAssertDblEquals(tc, 3.0 * -35.0/16.0, p->coeff[1], 1e-13);
    CuAssertDblEquals(tc, 2.0 * -30.0/8.0, p->coeff[2], 1e-13);
    CuAssertDblEquals(tc, 3.0 * 315.0/16.0, p->coeff[3], 1e-13);
    CuAssertDblEquals(tc, 2.0 * 35.0/8.0, p->coeff[4], 1e-13);
    CuAssertDblEquals(tc, 3.0 * -693.0/16.0, p->coeff[5], 1e-13);
    CuAssertDblEquals(tc, 0.0, p->coeff[6], 1e-13);
    CuAssertDblEquals(tc, 3.0 * 429/16.0, p->coeff[7], 1e-13);
    CuAssertDblEquals(tc, 0.0, p->coeff[8], 1e-13);
    CuAssertDblEquals(tc, 0.0, p->coeff[9], 1e-13);

    standard_poly_free(p);
    orth_poly_expansion_free(pl);
}

int opeTosp(size_t N,const double * x, double * out, void * args)
{
    (void)(args);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = 1.0 + 2.0 * x[ii] +
                        5.0 * pow(x[ii],3) +
                        2.0 * pow(x[ii],5) +
                        1.5 * pow(x[ii],6);
    }
    return 0;
}

void Test_orth_poly_expansion_to_standard_poly_ab(CuTest * tc){
    
    printf("Testing function: orth_expansion_to_standard on (-3.0, 2.0) \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,opeTosp,NULL);

    // approximation
    double lb = -3.0, ub = 2.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-15);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    opoly_t pl = orth_poly_expansion_approx_adapt(opts,fw);

    struct StandardPoly * p = orth_poly_expansion_to_standard_poly(pl);
    
    size_t ii;
    for (ii = 0; ii < p->num_poly; ii++){
        if (ii == 0){
            CuAssertDblEquals(tc, 1.0, p->coeff[ii], 1e-10);
        }
        else if (ii == 1){
            CuAssertDblEquals(tc, 2.0, p->coeff[ii], 1e-10);
        }
        else if (ii == 3){
            CuAssertDblEquals(tc, 5.0, p->coeff[ii], 1e-10);
        }
        else if (ii == 5){
            CuAssertDblEquals(tc, 2.0, p->coeff[ii], 1e-8);
        }
        else if (ii == 6){
            CuAssertDblEquals(tc, 1.5, p->coeff[ii], 1e-8);
        }
        else{
            CuAssertDblEquals(tc, 0.0, p->coeff[ii], 1e-8);
        }
    }
    
    standard_poly_free(p);
    ope_opts_free(opts);
    POLY_FREE(pl);
    fwrap_destroy(fw);
}

CuSuite * StandardPolyGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_orth_to_standard_poly);
    SUITE_ADD_TEST(suite, Test_orth_poly_expansion_to_standard_poly);
    SUITE_ADD_TEST(suite, Test_orth_poly_expansion_to_standard_poly_ab);

    return suite;
}

void Test_orth_poly_expansion_real_roots(CuTest * tc){
    
    printf("Testing function: orth_poly_expansion_real_roots \n");

    
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,polyroots,NULL);

    // approximation
    double lb = -3.0, ub = 2.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-15);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    opoly_t pl = orth_poly_expansion_approx_adapt(opts,fw);

    size_t nroots;
    double * roots = orth_poly_expansion_real_roots(pl, &nroots);
    
   /* printf("roots are: "); */
   /* dprint(nroots, roots); */

    CuAssertIntEquals(tc, 5, nroots);
    CuAssertDblEquals(tc, -3.0, roots[0], 1e-9);
    CuAssertDblEquals(tc, 0.0, roots[1], 1e-9);
    CuAssertDblEquals(tc, 1.0, roots[2], 1e-5);
    CuAssertDblEquals(tc, 1.0, roots[3], 1e-5);
    CuAssertDblEquals(tc, 2.0, roots[4], 1e-9);

    free(roots);
    POLY_FREE(pl);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}

void Test_maxmin_poly_expansion(CuTest * tc){
    
    printf("Testing functions: absmax, max and min of orth_poly_expansion \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,maxminpoly,NULL);

    // approximation
    double lb = -1.0, ub = 2.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-15);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    opoly_t pl = orth_poly_expansion_approx_adapt(opts,fw);

    double loc;
    double max = orth_poly_expansion_max(pl, &loc);
    double min = orth_poly_expansion_min(pl, &loc);
    double absmax = orth_poly_expansion_absmax(pl, &loc,NULL);

    double diff;

    diff = fabs(1.0-max);
    //printf("diff =%G\n",diff);
    CuAssertDblEquals(tc,0.0, diff, 1e-9);
    CuAssertDblEquals(tc, -1.0, min, 1e-9);
    CuAssertDblEquals(tc, 1.0, absmax, 1e-9);

    POLY_FREE(pl);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}

CuSuite * PolyAlgorithmsGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_orth_poly_expansion_real_roots);
    SUITE_ADD_TEST(suite, Test_maxmin_poly_expansion);

    return suite;
}

void Test_serialize_orth_poly(CuTest * tc){
    
    printf("Testing functions: (de)serialize_orth_poly \n");
        
    struct OrthPoly * poly = init_leg_poly();
    
    unsigned char * text = serialize_orth_poly(poly);
    
    struct OrthPoly * pt = deserialize_orth_poly(text);
    CuAssertIntEquals(tc,0,pt->ptype);
    
    free(text);
    free_orth_poly(poly);
    free_orth_poly(pt);

    poly = init_cheb_poly();

    text = serialize_orth_poly(poly);
    pt = deserialize_orth_poly(text);

    CuAssertIntEquals(tc,1,pt->ptype);
    free_orth_poly(pt);
    free_orth_poly(poly);
    free(text);
}

void Test_serialize_orth_poly_expansion(CuTest * tc){
    
    printf("Testing functions: (de)serializing orth_poly_expansion \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,maxminpoly,NULL);

    // approximation
    double lb = -1.0, ub = 2.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-15);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    opoly_t pl = orth_poly_expansion_approx_adapt(opts,fw);

    unsigned char * text = NULL;
    size_t size_to_be;
    serialize_orth_poly_expansion(text, pl, &size_to_be);
    text = malloc(size_to_be * sizeof(char));

    serialize_orth_poly_expansion(text, pl, NULL);

    opoly_t pt = NULL;
    deserialize_orth_poly_expansion(text, &pt);
            
    double * xtest = linspace(lb,ub,1000);
    size_t ii;
    double err = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(POLY_EVAL(pl,xtest[ii]) - POLY_EVAL(pt,xtest[ii]),2);
    }
    err = sqrt(err);
    CuAssertDblEquals(tc, 0.0, err, 1e-15);
    free(xtest);
    free(text);

    POLY_FREE(pl);
    POLY_FREE(pt);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}

void Test_orth_poly_expansion_savetxt(CuTest * tc){
    
    printf("Testing functions: orth_poly_expansion_savetxt and _loadtxt \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,maxminpoly,NULL);

    // approximation
    double lb = -1.0, ub = 2.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-15);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    opoly_t pl = orth_poly_expansion_approx_adapt(opts,fw);


    FILE * fp = fopen("testorthpoly.txt","w+");
    size_t prec = 21;
    orth_poly_expansion_savetxt(pl,fp,prec);

    opoly_t pt = NULL;
    rewind(fp);
    pt = orth_poly_expansion_loadtxt(fp);

    double * xtest = linspace(lb,ub,1000);
    size_t ii;
    double err = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(POLY_EVAL(pl,xtest[ii]) - POLY_EVAL(pt,xtest[ii]),2);
    }
    err = sqrt(err);
    CuAssertDblEquals(tc, 0.0, err, 1e-15);
    free(xtest);

    fclose(fp);
    POLY_FREE(pl);
    POLY_FREE(pt);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}

void Test_serialize_generic_function(CuTest * tc){
    
    printf("Testing functions: (de)serializing generic_function \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,maxminpoly,NULL);

    // approximation
    double lb = -1.0, ub = 2.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-15);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    
    struct GenericFunction * pl =
        generic_function_approximate1d(POLYNOMIAL,opts,fw);
    
    unsigned char * text = NULL;
    size_t size_to_be;
    serialize_generic_function(text, pl, &size_to_be);
    text = malloc(size_to_be * sizeof(char));

    serialize_generic_function(text, pl, NULL);
    
    struct GenericFunction * pt = NULL;
    deserialize_generic_function(text, &pt);

    double * xtest = linspace(lb,ub,1000);
    size_t ii;
    double err = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(pl,xtest[ii]) -
                   generic_function_1d_eval(pt,xtest[ii]),2);
    }
    err = sqrt(err);
    CuAssertDblEquals(tc, 0.0, err, 1e-15);
    free(xtest);
    free(text);

    ope_opts_free(opts);
    generic_function_free(pl);
    generic_function_free(pt);
    fwrap_destroy(fw);
}

void Test_generic_function_savetxt(CuTest * tc){
    
    printf("Testing functions: generic_function_savetxt and _loadtxt \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,maxminpoly,NULL);

    // approximation
    double lb = -1.0, ub = 2.0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,10);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-15);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    
    struct GenericFunction * pl =
        generic_function_approximate1d(POLYNOMIAL,opts,fw);
    
    FILE * fp = fopen("genfunctest.txt","w+");
    size_t prec = 21;
    generic_function_savetxt(pl,fp,prec);

    struct GenericFunction * pt = NULL;
    rewind(fp);
    pt = generic_function_loadtxt(fp);

    double * xtest = linspace(lb,ub,1000);
    size_t ii;
    double err = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(pl,xtest[ii]) -
                   generic_function_1d_eval(pt,xtest[ii]),2);
    }
    err = sqrt(err);
    CuAssertDblEquals(tc, 0.0, err, 1e-15);
    free(xtest);


    fclose(fp);
    ope_opts_free(opts);
    generic_function_free(pl);
    generic_function_free(pt);
    fwrap_destroy(fw);
}



CuSuite * PolySerializationGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_serialize_orth_poly);
    SUITE_ADD_TEST(suite, Test_serialize_orth_poly_expansion);
    SUITE_ADD_TEST(suite, Test_orth_poly_expansion_savetxt);
    SUITE_ADD_TEST(suite, Test_serialize_generic_function);
    SUITE_ADD_TEST(suite, Test_generic_function_savetxt);

    return suite;
}

static void regress_func(size_t N, const double * x, double * out)
{
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = -1.0 * pow(x[ii],7) + 2.0*pow(x[ii],2) + 0.2 * 0.5*x[ii]; 
    }
}

void Test_LS_cheb_regress(CuTest * tc){
    
    printf("Testing functions: least squares regression with cheb\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,LS,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_initial_parameters(regopts,params);

    // check derivative
    c3opt_add_objective(optimizer,param_LSregress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
    
    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}

void Test_LS_leg_regress(CuTest * tc){
    
    printf("Testing functions: least squares regression with legendre\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,LS,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_LSregress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    
    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}

void Test_LS_herm_regress(CuTest * tc){
    
    printf("Testing functions: least squares regression with hermite\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(HERMITE);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,LS,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_LSregress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    
    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}

void Test_RLS2_cheb_regress(CuTest * tc){
    
    printf("Testing functions: ridge regression with cheb\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,RLS2,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_regularization_penalty(regopts,1e-2/sqrt(ndata));
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_RLS2regress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}

void Test_RLS2_leg_regress(CuTest * tc){
    
    printf("Testing functions: ridge regression with legendre\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,RLS2,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_regularization_penalty(regopts,1e2/sqrt(ndata));
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_RLS2regress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    regress_1d_opts_set_regularization_penalty(regopts,1e-2/sqrt(ndata));
    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}

void Test_RLS2_herm_regress(CuTest * tc){
    
    printf("Testing functions: ridge regression with hermite\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(HERMITE);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,RLS2,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_regularization_penalty(regopts,1e-4/sqrt(ndata));
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_RLS2regress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    
    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    printf("\t note: must have smaller regularization\n");
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}

void Test_RLSD2_cheb_regress(CuTest * tc){
    
    printf("Testing functions: second deriv reg with cheb\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,RLSD2,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_regularization_penalty(regopts,1e-2/sqrt(ndata));
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_RLSD2regress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}

void Test_RLSD2_leg_regress(CuTest * tc){
    
    printf("Testing functions: second deriv reg with legendre\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,RLSD2,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_regularization_penalty(regopts,1e-2/sqrt(ndata));
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_RLSD2regress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}

void Test_RLSD2_herm_regress(CuTest * tc){
    
    printf("Testing functions: second deriv reg with hermite\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(HERMITE);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,RLSD2,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_regularization_penalty(regopts,1e-4/sqrt(ndata));
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_RLSD2regress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    printf("\t note: must have smaller regularization\n");
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}

void Test_RLSRKHS_alg_cheb_regress(CuTest * tc){
    
    printf("Testing functions: RKHS regularization with chebyshev, algebraic decay\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,RLSRKHS,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_regularization_penalty(regopts,1e-2/sqrt(ndata));
    regress_1d_opts_set_RKHS_decay_rate(regopts,ALGEBRAIC,0.9);
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_RLSRKHSregress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    
    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}

void Test_RLSRKHS_exp_cheb_regress(CuTest * tc){
    
    printf("Testing functions: RKHS regularization with chebyshev, exponential decay\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(CHEBYSHEV);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,RLSRKHS,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_regularization_penalty(regopts,1e-2/sqrt(ndata));
    regress_1d_opts_set_RKHS_decay_rate(regopts,EXPONENTIAL,2.0);
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_RLSRKHSregress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}


void Test_RLSRKHS_alg_leg_regress(CuTest * tc){
    
    printf("Testing functions: RKHS regularization with legendre, algebraic decay\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,RLSRKHS,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_regularization_penalty(regopts,1e-2/sqrt(ndata));
    regress_1d_opts_set_RKHS_decay_rate(regopts,ALGEBRAIC,0.9);
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_RLSRKHSregress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
    
    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}

void Test_RLSRKHS_exp_leg_regress(CuTest * tc){
    
    printf("Testing functions: RKHS regularization with legendre, exponential decay\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,RLSRKHS,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_regularization_penalty(regopts,1e-2/sqrt(ndata));
    regress_1d_opts_set_RKHS_decay_rate(regopts,EXPONENTIAL,2.0);
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_RLSRKHSregress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
    
    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}


void Test_RLSRKHS_alg_herm_regress(CuTest * tc){
    
    printf("Testing functions: RKHS regularization with hermite, algebraic decay\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(HERMITE);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,RLSRKHS,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_regularization_penalty(regopts,1e-4/sqrt(ndata));
    regress_1d_opts_set_RKHS_decay_rate(regopts,ALGEBRAIC,0.9);
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_RLSRKHSregress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
    
    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    printf("\t note: must have smaller regularization\n");
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}

void Test_RLSRKHS_exp_herm_regress(CuTest * tc){
    
    printf("Testing functions: RKHS regularization with hermite, exponential decay\n");

    size_t nparams = 10;
    double params[10] = {0.12,2.04,9.0,-0.2,0.4,
                         0.6,-0.9,-.2,0.04,1.2};
    struct OpeOpts * aopts = ope_opts_alloc(HERMITE);
    ope_opts_set_nparams(aopts,nparams);
    
    // create data
    size_t ndata = 5000;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,RLSRKHS,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,POLYNOMIAL,aopts);
    regress_1d_opts_set_regularization_penalty(regopts,1e-4/sqrt(ndata));
    regress_1d_opts_set_RKHS_decay_rate(regopts,EXPONENTIAL,1.1);
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_RLSRKHSregress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    /* for (size_t ii = 0; ii < nparams; ii++){ */
    /*     printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
    /*     /\* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); *\/ */
    /* } */
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-3);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    ope_opts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}



CuSuite * PolyRegressionSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_LS_cheb_regress);
    SUITE_ADD_TEST(suite, Test_LS_leg_regress);
    SUITE_ADD_TEST(suite, Test_LS_herm_regress);
    
    SUITE_ADD_TEST(suite, Test_RLS2_cheb_regress);
    SUITE_ADD_TEST(suite, Test_RLS2_leg_regress);
    SUITE_ADD_TEST(suite, Test_RLS2_herm_regress);

    /* SUITE_ADD_TEST(suite, Test_RLSD2_cheb_regress); */
    /* SUITE_ADD_TEST(suite, Test_RLSD2_leg_regress); */
    /* SUITE_ADD_TEST(suite, Test_RLSD2_herm_regress); */

    SUITE_ADD_TEST(suite, Test_RLSRKHS_alg_cheb_regress);
    SUITE_ADD_TEST(suite, Test_RLSRKHS_exp_cheb_regress);
    SUITE_ADD_TEST(suite, Test_RLSRKHS_alg_leg_regress);
    SUITE_ADD_TEST(suite, Test_RLSRKHS_exp_leg_regress);
    SUITE_ADD_TEST(suite, Test_RLSRKHS_alg_herm_regress);
    SUITE_ADD_TEST(suite, Test_RLSRKHS_exp_herm_regress);

    return suite;
}


