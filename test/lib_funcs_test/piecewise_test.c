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

#include "lib_funcs.h"

typedef struct PiecewisePoly* opoly_t;
#define POLY_EVAL piecewise_poly_eval
#define POLY_FREE piecewise_poly_free

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

double pw_lin(double x,void * args){
    (void)(args);
    return 2.0 * x + -0.2;
}
void Test_pw_linear(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_linear \n");
    
    double lb = -2.0;
    double ub = 0.7;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    
    struct PiecewisePoly * pw = piecewise_poly_linear(2.0,-0.2,opts);

    // compute error
    double abs_err;
    double func_norm;
    compute_error(lb,ub,1000,pw,pw_lin,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    POLY_FREE(pw);
    pw_poly_opts_free(opts);    
}

double pw_quad(double x,void * args){
    (void)(args);
    /* return 1e-10 * x * x + 3.2 * x + -0.2; */
    return 1e-10 * pow(x  - 1e8,2); /* x + 3.2 * x + -0.2; */
}

void Test_pw_quad(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_quad \n");

    
    double lb = -2.0;
    double ub = 0.7;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    
    /* struct PiecewisePoly * pw = piecewise_poly_quadratic(1e-10,3.2,-0.2,opts); */
    struct PiecewisePoly * pw = piecewise_poly_quadratic(1e-10,1e8,opts);

    // compute error
    double abs_err;
    double func_norm;
    compute_error(lb,ub,1000,pw,pw_quad,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    POLY_FREE(pw);
    pw_poly_opts_free(opts);    

}

void Test_pw_approx(CuTest * tc){

    printf("Testing function: piecewise_poly_approx1 (1/1);\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    // approximation
    double lb=-1.0, ub=1.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    opoly_t pw = piecewise_poly_approx1(opts,fw);

    // compute error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,pw,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    fwrap_destroy(fw);
    POLY_FREE(pw);
    pw_poly_opts_free(opts);
}

void Test_pw_approx_nonnormal(CuTest * tc){

    printf("Testing function: piecewise_poly_approx on (a,b)\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    // approximation
    double lb=-3.0, ub=2.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    size_t N = 15;
    double * pts = linspace(lb,ub,N);
    pw_poly_opts_set_pts(opts,N,pts);
    opoly_t pw = piecewise_poly_approx1(opts,fw);

    double lb1 = piecewise_poly_get_lb(pw->branches[0]);
    double ub1 = piecewise_poly_get_ub(pw->branches[0]);
    CuAssertDblEquals(tc,pts[0],lb1,1e-14);
    CuAssertDblEquals(tc,pts[1],ub1,1e-14);

    // compute error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,pw,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    fwrap_destroy(fw);
    POLY_FREE(pw);
    pw_poly_opts_free(opts);
    free(pts);
}

void Test_pw_approx1_adapt(CuTest * tc){

    printf("Testing function:  pw_approx1_adapt\n");

   // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    // approximation
    double lb=-1.0, ub=1.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    opoly_t pw = piecewise_poly_approx1_adapt(opts,fw);

    size_t nbounds;
    double * bounds = NULL;
    piecewise_poly_boundaries(pw,&nbounds,&bounds,NULL);
    //dprint(nbounds,bounds);
    free(bounds);

    // compute error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,pw,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    fwrap_destroy(fw);
    POLY_FREE(pw);
    pw_poly_opts_free(opts);
}

void Test_pw_approx_adapt_weird(CuTest * tc){

    printf("Testing function: piecewise_poly_approx1_adapt on (a,b)\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    // approximation
    double lb=-2.0, ub = -1.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    opoly_t pw = piecewise_poly_approx1_adapt(opts,fw);

    // this is just to make sure no memory errors or segfaults
    size_t nbounds;
    double * bounds = NULL;
    piecewise_poly_boundaries(pw,&nbounds,&bounds,NULL);
    //dprint(nbounds,bounds);
    free(bounds);

    // compute error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,pw,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    fwrap_destroy(fw);
    POLY_FREE(pw);
    pw_poly_opts_free(opts);
}

static int pw_disc(size_t N, const double * x, double * out, void * args)
{
    
    (void)(args);

    double split = 0.0;
    for (size_t ii = 0; ii < N; ii++){
        if (x[ii] > split){
            out[ii] = sin(x[ii]);
        }
        else{
            out[ii] = pow(x[ii],2) + 2.0 * x[ii] + 1.0;
        }
    }
    return 0;
}

void Test_pw_approx1(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_approx1 on discontinuous function (1/2) \n");
    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,pw_disc,NULL);

    // approximation
    double lb=-5.0, ub = 1.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    pw_poly_opts_set_minsize(opts,1e-5);
    opoly_t pw = piecewise_poly_approx1_adapt(opts,fw);

    // compute error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,pw,pw_disc,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    fwrap_destroy(fw);
    POLY_FREE(pw);
    pw_poly_opts_free(opts);
}

void Test_pw_flatten(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_flatten \n");
    
    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,pw_disc,NULL);

    // approximation
    double lb=-5.0, ub = 1.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    pw_poly_opts_set_minsize(opts,1e-5);


    opoly_t pw = piecewise_poly_approx1_adapt(opts,fw);
    double * nodesa = NULL;
    size_t Na;
    piecewise_poly_boundaries(pw,&Na, &nodesa, NULL);


    // THIS TEST NEEDS TO BE FIXED BECAUSE APPROX1_ADAPT now always outputs a flattened poly --AG 5/18/2017
    size_t nregions = piecewise_poly_nregions(pw);
    /* int isflat = piecewise_poly_isflat(pw); */
    /* CuAssertIntEquals(tc,0,isflat); */
    /* piecewise_poly_flatten(pw); */
    /* CuAssertIntEquals(tc,nregions,pw->nbranches); */
    /* isflat = piecewise_poly_isflat(pw); */
    /* CuAssertIntEquals(tc,1,isflat); */

    size_t npb = piecewise_poly_nregions(pw);
    CuAssertIntEquals(tc,nregions,npb);
    double * nodesb = NULL;
    size_t Nb;
    piecewise_poly_boundaries(pw,&Nb, &nodesb, NULL);
    CuAssertIntEquals(tc,nregions+1,Nb);
    CuAssertDblEquals(tc,nodesb[0],lb,1e-15);
    CuAssertDblEquals(tc,nodesb[Nb-1],ub,1e-15);
    for (size_t ii = 0; ii < nregions; ii++){
        CuAssertDblEquals(tc,nodesa[ii+1],piecewise_poly_get_ub(pw->branches[ii]),1e-15);
        CuAssertDblEquals(tc,nodesa[ii],piecewise_poly_get_lb(pw->branches[ii]),1e-15);
    }

    free(nodesa); nodesa = NULL;
    free(nodesb); nodesb = NULL;
    
    // compute error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,pw,pw_disc,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    fwrap_destroy(fw);
    POLY_FREE(pw);
    pw_poly_opts_free(opts);
}

static int pw_disc2(size_t N, const double * x, double * out, void * args)
{
     (void)(args);

    double split = 0.2;
    for (size_t ii = 0; ii < N; ii++){
        if (x[ii] < split){
            out[ii] = sin(x[ii]);
        }
        else{
            out[ii] = pow(x[ii],2) + 2.0 * x[ii];
        }
    }
    return 0;
}

void Test_poly_match(CuTest * tc){

    printf("Testing functions: piecewise_poly_match \n");

    double lb = -2.0;
    double ub = 0.7;

    size_t Na, Nb;
    double * nodesa = NULL;
    double * nodesb = NULL;

    
    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,pw_disc,NULL);

    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    pw_poly_opts_set_minsize(opts,1e-3);

    
    struct PiecewisePoly * a = piecewise_poly_approx1_adapt(opts,fw);

    size_t npa = piecewise_poly_nregions(a);
    /* printf("nregions = %zu\n",npa); */
    piecewise_poly_boundaries(a,&Na, &nodesa, NULL);
    CuAssertIntEquals(tc, npa, Na-1);
    CuAssertDblEquals(tc,-2.0,nodesa[0],1e-15);
    CuAssertDblEquals(tc,0.7,nodesa[Na-1],1e-15);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,pw_disc2,NULL);


    struct PiecewisePoly * b = piecewise_poly_approx1_adapt(opts,fw2);

    size_t npb = piecewise_poly_nregions(b);
    /* printf("nregions b = %zu\n",npb); */
    piecewise_poly_boundaries(b,&Nb, &nodesb, NULL);
    CuAssertIntEquals(tc, npb, Nb-1);
    CuAssertDblEquals(tc,-2.0,nodesb[0],1e-15);
    CuAssertDblEquals(tc,0.7,nodesb[Nb-1],1e-15);

    
    struct PiecewisePoly * aa = NULL;
    struct PiecewisePoly * bb = NULL;
    /* printf("matching\n"); */
    piecewise_poly_match(a,&aa,b,&bb);
    /* printf("matched\n"); */

    size_t npaa = piecewise_poly_nregions(aa);
    size_t npbb = piecewise_poly_nregions(bb);
    CuAssertIntEquals(tc,npaa,npbb);

    size_t Naa, Nbb;
    double * nodesaa = NULL;
    double * nodesbb = NULL;
    
    piecewise_poly_boundaries(aa,&Naa, &nodesaa, NULL);
    CuAssertDblEquals(tc,-2.0,nodesaa[0],1e-15);
    CuAssertDblEquals(tc,0.7,nodesaa[Naa-1],1e-15);

    piecewise_poly_boundaries(bb,&Nbb, &nodesbb, NULL);
    CuAssertDblEquals(tc,-2.0,nodesbb[0],1e-15);
    CuAssertDblEquals(tc,0.7,nodesbb[Nbb-1],1e-15);
    
    CuAssertIntEquals(tc,Naa,Nbb);
    size_t ii;
    for (ii = 0; ii < Naa; ii++){
        CuAssertDblEquals(tc,nodesaa[ii],nodesbb[ii],1e-15);
    }

    fwrap_destroy(fw);
    fwrap_destroy(fw2);
    pw_poly_opts_free(opts);
    free(nodesa);
    free(nodesb);
    free(nodesaa);
    free(nodesbb);
    piecewise_poly_free(a);
    piecewise_poly_free(b);
    piecewise_poly_free(aa);
    piecewise_poly_free(bb);

}


void Test_pw_integrate(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_integrate (1/2) \n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,pw_disc2,NULL);

    // approximation
    double lb=-2.0, ub = 1.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    pw_poly_opts_set_minsize(opts,1e-5);
    opoly_t pw = piecewise_poly_approx1_adapt(opts,fw);

    // compute error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,pw,pw_disc2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    double sol;
    if ( ub > 0.2 ) {
        sol = pow(ub,3)/3.0 + pow(ub,2) -  pow(0.2,3)/3.0 - pow(0.2,2) +
                ( -cos(0.2) - (-cos(lb)));
    }
    else{
        sol = -cos(ub) - (-cos(lb));
    }

    double ints = piecewise_poly_integrate(pw);

    CuAssertDblEquals(tc, sol, ints, 1e-6);

    fwrap_destroy(fw);
    pw_poly_opts_free(opts);
    POLY_FREE(pw);
}

void Test_pw_integrate2(CuTest * tc){

    printf("Testing function: piecewise_poly_integrate (2/2)\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,powX2,NULL);

    // approximation
    double lb=-2.0, ub = 3.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    pw_poly_opts_set_minsize(opts,1e-5);
    opoly_t pw = piecewise_poly_approx1_adapt(opts,fw);

    double intshould = (pow(ub,3) - pow(lb,3))/3;
    double ints = piecewise_poly_integrate(pw);
    /* printf("%G,%G,%G",intshould,ints,fabs(intshould-ints)); */
    CuAssertDblEquals(tc, intshould, ints, 1e-6);

    fwrap_destroy(fw);
    pw_poly_opts_free(opts);
    POLY_FREE(pw);
}

void Test_pw_prod(CuTest * tc){

    printf("Testing function: piecewise_poly_product\n");

    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,TwoPowX3,NULL);

    double lb = -2.0, ub = 3.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    opoly_t cpoly = piecewise_poly_approx1_adapt(opts,fw1);
    opoly_t cpoly2 = piecewise_poly_approx1_adapt(opts,fw2);
    opoly_t cpoly3 = piecewise_poly_prod(cpoly,cpoly2);

    double * xtest = linspace(lb,ub,100);
    size_t ii;
    double abs_err = 0.0;
    double func_norm = 0.0;
    double val,val1,val2;
    for (ii = 0; ii < 100; ii++){
        powX2(1,xtest+ii,&val1,NULL);
        TwoPowX3(1,xtest+ii,&val2,NULL);
        val = val1*val2;
        abs_err += pow(POLY_EVAL(cpoly3,xtest[ii]) - val,2);
        func_norm += pow(val,2);
    }
    free(xtest); xtest = NULL;
    
    CuAssertDblEquals(tc, 0.0,abs_err/func_norm, 1e-10);
    
    POLY_FREE(cpoly);
    POLY_FREE(cpoly2);
    POLY_FREE(cpoly3);
    pw_poly_opts_free(opts);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
}

void Test_pw_inner(CuTest * tc){

    printf("Testing function: piecewise_poly_inner\n");

    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,TwoPowX3,NULL);

    double lb = -2.0, ub = 3.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    opoly_t cpoly = piecewise_poly_approx1_adapt(opts,fw1);
    opoly_t cpoly2 = piecewise_poly_approx1_adapt(opts,fw2);

    double intshould = (pow(ub,6) - pow(lb,6))/3;
    double intis = piecewise_poly_inner(cpoly,cpoly2);
    CuAssertDblEquals(tc, intshould, intis, 1e-10);
    
    POLY_FREE(cpoly);
    POLY_FREE(cpoly2);
    pw_poly_opts_free(opts);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
}

void Test_pw_norm(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_norm (1/2)\n");
    
    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,pw_disc2,NULL);

    // approximation
    double lb=-2.0, ub = 0.7;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    pw_poly_opts_set_minsize(opts,1e-7);
    opoly_t pw = piecewise_poly_approx1_adapt(opts,fw);

    double sol = sqrt(1.19185 + 0.718717);
    double ints = piecewise_poly_norm(pw);
    CuAssertDblEquals(tc, 0,fabs(sol-ints)/fabs(sol), 1e-5);

    fwrap_destroy(fw);
    pw_poly_opts_free(opts);
    POLY_FREE(pw);
}

void Test_pw_norm2(CuTest * tc){
    
    printf("Testing function: piecewise_poly_norm (2/2)\n");

    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    double lb = -2.0, ub = 3.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    opoly_t cpoly = piecewise_poly_approx1_adapt(opts,fw1);

    
    double intshould = (pow(ub,5) - pow(lb,5))/5.0;
    double intis = piecewise_poly_norm(cpoly);
    CuAssertDblEquals(tc, sqrt(intshould), intis, 1e-10);
    
    POLY_FREE(cpoly);
    pw_poly_opts_free(opts);
    fwrap_destroy(fw1);
}

void Test_pw_daxpby(CuTest * tc){

    printf("Testing functions: piecewise_poly_daxpby (1/2)\n");

    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,pw_disc2,NULL);
    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,pw_disc,NULL);

    double lb = -2.0, ub = 0.7;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);

    opoly_t a = piecewise_poly_approx1_adapt(opts,fw1);
    opoly_t b = piecewise_poly_approx1_adapt(opts,fw2);
    opoly_t c = piecewise_poly_daxpby(0.4,a,0.5,b);
    
    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double errden = 0.0;
    double err = 0.0;
    double diff;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double val1,val2;
        pw_disc2(1,xtest+ii,&val1,NULL);
        pw_disc(1,xtest+ii,&val2,NULL);
        double val = 0.4 * val1 + 0.5*val2;
        diff= piecewise_poly_eval(c,xtest[ii]) - val;
        err+= pow(diff,2.0);
        errden += pow(val,2.0);

    }
    err = sqrt(err/errden);
    CuAssertDblEquals(tc, 0.0, err, 2e-11);

    free(xtest); xtest = NULL;
    POLY_FREE(a); a = NULL;
    POLY_FREE(b); b = NULL;
    POLY_FREE(c); c = NULL;
    pw_poly_opts_free(opts);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
}

int pw_exp(size_t N, const double * x,double * out, void * args)
{

    (void)(args);

    for (size_t ii = 0; ii < N; ii++){
        if (x[ii] < -0.2){
            out[ii] = 0.0;
        }
        else{
            out[ii] = (exp(5.0 * x[ii]));
        }
    }
    return 0;
}

void Test_pw_daxpby2(CuTest * tc){

    printf("Testing functions: piecewise_poly_daxpby (2/2)\n");

    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,pw_disc2,NULL);
    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,pw_exp,NULL);

    double lb = -1.0, ub = 1.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);

    opoly_t a = piecewise_poly_approx1_adapt(opts,fw1);
    opoly_t b = piecewise_poly_approx1_adapt(opts,fw2);
    opoly_t c = piecewise_poly_daxpby(0.5,a,0.5,b);
    
    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double terr;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double val1,val2;
        pw_disc2(1,xtest+ii,&val1,NULL);
        pw_exp(1,xtest+ii,&val2,NULL);
        double val = 0.5 * val1 + 0.5*val2;
        terr = fabs(piecewise_poly_eval(c,xtest[ii]) - val);
        err+= terr;
    }
    CuAssertDblEquals(tc, 0.0, err/N, 1e-10);

    free(xtest);
    xtest = NULL;
    piecewise_poly_free(a); a = NULL;
    piecewise_poly_free(b); b = NULL;
    piecewise_poly_free(c); c = NULL;
    pw_poly_opts_free(opts);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
}

void Test_pw_derivative(CuTest * tc){

    printf("Testing function: piecewise_poly_deriv  on (a,b)\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    // approximation
    double lb = -2.0, ub = -1.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    opoly_t cpoly = piecewise_poly_approx1_adapt(opts,fw);
    opoly_t der = piecewise_poly_deriv(cpoly);
    
    // error
    double abs_err;
    double func_norm;
    compute_error(lb,ub,1000,der,funcderiv,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);
    
    POLY_FREE(cpoly);
    POLY_FREE(der);
    pw_poly_opts_free(opts);
    fwrap_destroy(fw);
}

void Test_pw_real_roots(CuTest * tc){
    
    printf("Testing function: piecewise_poly_real_roots \n");
    
    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,polyroots,NULL);

    // approximation
    double lb = -3.0, ub = 2.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    opoly_t pl = piecewise_poly_approx1_adapt(opts,fw);

    size_t nroots;
    double * roots = piecewise_poly_real_roots(pl, &nroots);
    
    /* printf("roots are: (double roots in piecewise_poly)\n"); */
    /* dprint(nroots, roots); */
    
    CuAssertIntEquals(tc, 1, 1);
    
    /* CuAssertIntEquals(tc, 5, nroots); */
    /* CuAssertDblEquals(tc, -3.0, roots[0], 1e-9); */
    /* CuAssertDblEquals(tc, 0.0, roots[1], 1e-9); */
    /* CuAssertDblEquals(tc, 1.0, roots[2], 1e-5); */
    /* CuAssertDblEquals(tc, 1.0, roots[3], 1e-5); */
    /* CuAssertDblEquals(tc, 2.0, roots[4], 1e-9); */

    free(roots);
    piecewise_poly_free(pl);
    pw_poly_opts_free(opts);
    fwrap_destroy(fw);
}

void Test_maxmin_pw(CuTest * tc){
    
    printf("Testing functions: absmax, max and min of pw \n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,maxminpoly,NULL);

    // approximation
    double lb = -1.0, ub = 2.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    opoly_t pl = piecewise_poly_approx1_adapt(opts,fw);
    
    double loc;
    double max = piecewise_poly_max(pl, &loc);
    double min = piecewise_poly_min(pl, &loc);
    double absmax = piecewise_poly_absmax(pl, &loc,NULL);

    CuAssertDblEquals(tc, 0.0, (1.0-max)/1.0, 1e-8);
    CuAssertDblEquals(tc, -1.0, min, 1e-10);
    CuAssertDblEquals(tc, 1.0, absmax, 1e-10);

    piecewise_poly_free(pl);
    fwrap_destroy(fw);
    pw_poly_opts_free(opts);
}


void Test_pw_serialize(CuTest * tc){
   
    printf("Testing functions: (de)serialize_piecewise_poly (and approx2) \n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,pw_disc2,NULL);

    // approximation
    double lb=-2.0, ub = 0.7;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    pw_poly_opts_set_minsize(opts,1e-5);
    opoly_t pw = piecewise_poly_approx1_adapt(opts,fw);

    //printf("approximated \n");
    size_t size;
    serialize_piecewise_poly(NULL,pw,&size);
    //printf("size=%zu \n",size);
    unsigned char * text = malloc(size);
    serialize_piecewise_poly(text,pw,NULL);
    
    struct PiecewisePoly * pw2 = NULL;
    deserialize_piecewise_poly(text,&pw2);

    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double terr;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        terr = fabs(piecewise_poly_eval(pw2,xtest[ii]) -
                     piecewise_poly_eval(pw,xtest[ii]));
        err+= terr;
    }

    CuAssertDblEquals(tc, 0.0, err, 1e-12);

    free(xtest);
    xtest = NULL;
    free(text); text = NULL;
    piecewise_poly_free(pw);
    piecewise_poly_free(pw2);
    pw = NULL;
    pw2 = NULL;
    fwrap_destroy(fw);
    pw_poly_opts_free(opts);

}

void Test_pw_savetxt(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_savetxt and _loadtxt \n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,pw_disc2,NULL);

    // approximation
    double lb=-2.0, ub = 0.7;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    pw_poly_opts_set_minsize(opts,1e-5);
    opoly_t pw = piecewise_poly_approx1_adapt(opts,fw);

    //printf("approximated \n");
    FILE * fp = fopen("testpw.txt","w+");
    size_t prec = 21;
    piecewise_poly_savetxt(pw,fp,prec);

    struct PiecewisePoly * pw2 = NULL;
    rewind(fp);
    pw2 = piecewise_poly_loadtxt(fp);

    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double terr;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        terr = fabs(piecewise_poly_eval(pw2,xtest[ii]) -
                     piecewise_poly_eval(pw,xtest[ii]));
        err+= terr;
    }

    CuAssertDblEquals(tc, 0.0, err, 1e-12);

    free(xtest);
    xtest = NULL;
    fclose(fp);
    piecewise_poly_free(pw);
    piecewise_poly_free(pw2);
    pw = NULL;
    pw2 = NULL;
    fwrap_destroy(fw);
    pw_poly_opts_free(opts);

}

void Test_minmod_disc_exists(CuTest * tc)
{
    printf("Testing functions: minmod_disc_exists \n");

    size_t N = 20;
    double * xtest = linspace(-4.0,1.0,N);
    double * vals = calloc_double(N);
    pw_disc(N,xtest,vals,NULL);

    size_t minm = 2;
    size_t maxm = 5;
    
    double x;
    int disc;
    double jumpval;
    for (size_t ii = 0; ii < N-1; ii++){
        x = (xtest[ii]+xtest[ii+1])/2.0;
        disc = minmod_disc_exists(x,xtest,vals,N,minm,maxm);
        jumpval = minmod_eval(x,xtest,vals,N,minm,maxm);
        printf("x,disc,jumpval = %G,%d,%G\n",x,disc,jumpval);
        if ( (xtest[ii] < 0.0) && (xtest[ii+1]) > 0.0){
            CuAssertIntEquals(tc,1,disc);
            break;
        }
        /* else{ */
        /*    CuAssertIntEquals(tc,0,disc); */
        /* } */
    }
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;
}

void Test_locate_jumps(CuTest * tc)
{
    printf("Testing functions: locate_jumps (1/2) \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,pw_disc,NULL);

    double lb = -4.0;
    double ub = 1.0;
    double tol = DBL_EPSILON/1000.0;
    size_t nsplit = 10;

    double * edges = NULL;
    size_t nEdge = 0;
    
    locate_jumps(fw,lb,ub,nsplit,tol,&edges,&nEdge);
    printf("number of edges = %zu\n",nEdge);
    printf("Edges = \n");
    size_t ii = 0;
    for (ii = 0; ii < nEdge; ii++){
        CuAssertDblEquals(tc,0.0,edges[ii],1e-12);
        printf("%G ", edges[ii]);
    }
    printf("\n");
    fwrap_destroy(fw); fw = NULL;
    CuAssertIntEquals(tc,1,1);
    free(edges); edges = NULL;
}

static int pw_multi_disc(size_t N, const double * x, double * out , void * args){
    
    assert ( args == NULL );
    double split1 = 0.0;
    double split2 = 0.5;
    for (size_t ii = 0; ii < N; ii++){
        if (x[ii] < split1){
            out[ii] = pow(x[ii],2) + 2.0 * x[ii] + 1.0;
        }
        else if (x[ii] < split2){
            out[ii] = sin(x[ii]);
        }
        else{
            out[ii] = exp(x[ii]);
        }
    }
    return 0;
}


void Test_locate_jumps2(CuTest * tc)
{
    printf("Testing functions: locate_jumps (2/2)\n");
    
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,pw_multi_disc,NULL);

    double lb = -4.0;
    double ub = 1.0;
    double tol = 1e-7;
    size_t nsplit = 10;

    double * edges = NULL;
    size_t nEdge = 0;
    
    locate_jumps(fw,lb,ub,nsplit,tol,&edges,&nEdge);
    printf("number of edges = %zu\n",nEdge);
    printf("Edges = \n");
    size_t ii = 0;
    for (ii = 0; ii < nEdge; ii++){
        
       printf("%G ", edges[ii]);
    }
    printf("\n");
    free(edges); edges = NULL;
    fwrap_destroy(fw); fw = NULL;
    CuAssertIntEquals(tc,1,1);
    free(edges);
}

int pap1(size_t N, const double * x,double * out, void * args)
{
    (void)(args);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = 5.0 * exp(5.0*x[ii]);
    }
	
    return 0;
}

/* void Test_polyannih(CuTest * tc){ */

/*     printf("Testing function: approx (1/1) \n"); */

/*     // function */
/*     struct Fwrap * fw = fwrap_create(1,"general-vec"); */
/*     fwrap_set_fvec(fw,pw_multi_disc,NULL); */

/*     // approximation */
/*     double lb = -5.0, ub = 5.0; */
/*     struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub); */
/*     pw_poly_opts_set_minsize(opts,1e-1); */
/*     pw_poly_opts_set_tol(opts,1e-5); */
/*     pw_poly_opts_set_maxorder(opts,8); */
/*     pw_poly_opts_set_nregions(opts,5); */
/*     opoly_t pw = piecewise_poly_approx2(fw,opts); */

/*     // error */
/*     double abs_err; */
/*     double func_norm; */
/*     compute_error_vec(lb,ub,1000,pw,pap1,NULL,&abs_err,&func_norm); */
/*     double err = abs_err / func_norm; */
/*     CuAssertDblEquals(tc, 0.0, err, 1e-13); */

/*     fwrap_destroy(fw); */
/*     pw_poly_opts_free(opts); */
/*     POLY_FREE(pw); */
/* } */


CuSuite * PiecewisePolyGetSuite(){
    CuSuite * suite = CuSuiteNew();
    
    SUITE_ADD_TEST(suite, Test_pw_linear);
    SUITE_ADD_TEST(suite, Test_pw_quad);
    SUITE_ADD_TEST(suite, Test_pw_approx);
    SUITE_ADD_TEST(suite, Test_pw_approx_nonnormal);
    SUITE_ADD_TEST(suite, Test_pw_approx1_adapt);
    SUITE_ADD_TEST(suite, Test_pw_approx_adapt_weird);
    SUITE_ADD_TEST(suite, Test_pw_approx1);
    SUITE_ADD_TEST(suite, Test_pw_flatten);
    SUITE_ADD_TEST(suite, Test_poly_match);
    SUITE_ADD_TEST(suite, Test_pw_integrate);
    SUITE_ADD_TEST(suite, Test_pw_integrate2);
    SUITE_ADD_TEST(suite, Test_pw_prod);
    SUITE_ADD_TEST(suite, Test_pw_inner);
    SUITE_ADD_TEST(suite, Test_pw_norm);
    SUITE_ADD_TEST(suite, Test_pw_norm2);
    SUITE_ADD_TEST(suite, Test_pw_daxpby);
    SUITE_ADD_TEST(suite, Test_pw_daxpby2);
    SUITE_ADD_TEST(suite, Test_pw_derivative);
    SUITE_ADD_TEST(suite, Test_pw_real_roots);
    SUITE_ADD_TEST(suite, Test_maxmin_pw);
    SUITE_ADD_TEST(suite, Test_pw_serialize);
    SUITE_ADD_TEST(suite, Test_pw_savetxt);


    /* SUITE_ADD_TEST(suite, Test_minmod_disc_exists); */
    /* SUITE_ADD_TEST(suite, Test_locate_jumps); */
    /* SUITE_ADD_TEST(suite, Test_locate_jumps2); */
    /* SUITE_ADD_TEST(suite, Test_polyannih); */
    
    // these below don't work yet
    //SUITE_ADD_TEST(suite, Test_pw_approx1pa);
    //SUITE_ADD_TEST(suite, Test_pw_approx12);
    //SUITE_ADD_TEST(suite, Test_pw_approx12pa);
    //SUITE_ADD_TEST(suite, Test_pw_trim);
    return suite;
}


void Test_pap1(CuTest * tc){

    printf("Testing function: approx (1/1) \n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,pap1,NULL);

    // approximation
    double lb = -5.0, ub = 5.0;
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,lb,ub);
    /* pw_poly_opts_set_minsize(opts,1e-1); */
    /* pw_poly_opts_set_tol(opts,1e-5); */
    /* pw_poly_opts_set_maxorder(opts,5); */
    opoly_t pw = piecewise_poly_approx1_adapt(opts,fw);

    // error
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,pw,pap1,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    fwrap_destroy(fw);
    pw_poly_opts_free(opts);
    POLY_FREE(pw);
}

CuSuite * PolyApproxSuite(){
    CuSuite * suite = CuSuiteNew();
    
    SUITE_ADD_TEST(suite, Test_pap1);
    return suite;
}



//old stuff
/*
void Test_pw_approx1pa(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_approx2 (1/2) \n");
    
    double lb = -5.0;
    double ub = 1.0;
    
    struct PiecewisePoly * pw = 
            piecewise_poly_approx1(pw_disc, NULL, lb, ub, NULL);

    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double terr;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        terr = fabs(pw_disc(xtest[ii],NULL) -
                            piecewise_poly_eval(pw,xtest[ii]));
        err += terr;
        //printf("x=%G, terr=%G\n",xtest[ii],terr);
    }
    CuAssertDblEquals(tc, 0.0, err, 1e-9);
    free(xtest); xtest=NULL;
    piecewise_poly_free(pw);
    pw = NULL;
}

void Test_pw_approx12(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_approx1 (2/2) \n");
    
    double lb = -1.0;
    double ub = 1.0;
    
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 6;
    aopts.minsize = 1e-3;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-10;

    struct PiecewisePoly * p2 = 
            piecewise_poly_approx1(pw_disc, NULL, lb, ub, &aopts);

    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double terr;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        terr = fabs(pw_disc(xtest[ii],NULL) -
                            piecewise_poly_eval(p2,xtest[ii]));
        err += terr;
       // printf("terr=%G\n",terr);
    }

    CuAssertDblEquals(tc, 0.0, err, 1e-12);

    free(xtest);
    xtest = NULL;
    piecewise_poly_free(p2);
    p2 = NULL;
}

void Test_pw_approx12pa(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_approx2 (2/2) \n");
    
    double lb = -1.0;
    double ub = 1.0;
    
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 10;
    aopts.minsize = 1e-5;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-10;

    struct PiecewisePoly * pw = 
            piecewise_poly_approx1(pw_disc, NULL, lb, ub, &aopts);
    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double terr;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        terr = fabs(pw_disc(xtest[ii],NULL) -
                            piecewise_poly_eval(pw,xtest[ii]));
        err += terr;
       // printf("terr=%G\n",terr);
    }

    CuAssertDblEquals(tc, 0.0, err, 1e-12);

    free(xtest);
    xtest = NULL;
    piecewise_poly_free(pw);
    pw = NULL;
}
*/

/*
void Test_pw_trim(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_trim \n");
    
    double lb = -1.0;
    double ub = 1.0;
    
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 10;
    aopts.minsize = 1e-7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-10;

    struct PiecewisePoly * pw = 
            piecewise_poly_approx1(pw_disc, NULL, lb, ub, &aopts);
    //printf("got approximation \n");
    size_t M;
    double * nodes = NULL;
    piecewise_poly_boundaries(pw,&M,&nodes,NULL);

    double new_lb = nodes[1];
    struct OrthPolyExpansion * temp = piecewise_poly_trim_left(&pw);

    double new_lb_check = piecewise_poly_get_lb(pw);
    CuAssertDblEquals(tc,new_lb,new_lb_check,1e-15);

    orth_poly_expansion_free(temp);
    temp=NULL;

    //printf("number of pieces is %zu\n",M);
    //printf("nodes are =");
    //dprint(M,nodes);
    free(nodes); nodes = NULL;

    piecewise_poly_free(pw);
    pw = NULL;

}
*/

