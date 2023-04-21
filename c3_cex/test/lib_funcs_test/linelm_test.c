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

typedef struct LinElemExp* le_t;
#define LINELEM_EVAL lin_elem_exp_eval
#define LINELEM_FREE lin_elem_exp_free

static void
compute_error(double lb,double ub, size_t N, le_t cpoly,
              double (*func)(double,void*), void* arg,
              double * abs_err, double * func_norm)
{
    double * xtest = linspace(lb,ub,N);
    size_t ii;
    *abs_err = 0.0;
    *func_norm = 0.0;
    for (ii = 0; ii < N; ii++){
        *abs_err += pow(LINELEM_EVAL(cpoly,xtest[ii]) - func(xtest[ii],arg),2);
        *func_norm += pow(func(xtest[ii],arg),2);
    }
    free(xtest); xtest = NULL;
}

static void
compute_error_vec(double lb,double ub, size_t N, le_t le,
                  int (*func)(size_t, const double *,double *,void*),
                  void* arg,
                  double * abs_err, double * func_norm)
{
    double * xtest = linspace(lb,ub,N);
    size_t ii;
    *abs_err = 0.0;
    *func_norm = 0.0;
    double val;
    double vfunc;
    for (ii = 0; ii < N; ii++){
        func(1,xtest+ii,&val,arg);
        vfunc = LINELEM_EVAL(le,xtest[ii]);
        /* printf("val = %3.5G, vfunc = %3.5G\n", vfunc, val); */
        *abs_err += pow(vfunc - val,2);
        *func_norm += pow(val,2);
    }
    free(xtest); xtest = NULL;
}

void Test_lin_elem_exp_approx(CuTest * tc){

    printf("Testing function: lin_elem_exp_init\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    size_t N = 100; double lb=-1.0,ub=1.0;
    double * x = linspace(lb,ub,N);
    double f[100];
    fwrap_eval(N,x,f,fw);
    le_t le = lin_elem_exp_init(N,x,f);

    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,le,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-5);
    
    free(x);
    LINELEM_FREE(le);
    fwrap_destroy(fw);
}

int fleadapt(size_t N, const double * x, double * out, void * count)
{
    int * c = count;
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = 1.0 / (pow(x[ii] - 0.3,2) + 0.01);
        out[ii] += 1.0/(pow(x[ii]-0.9,2) + 0.04);
        *c += 1;
    }
    return 0;
}

void Test_lin_elem_exp_adapt(CuTest * tc){

    printf("Testing function: lin_elem_adapt\n");

    // function
    int N = 0;
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,fleadapt,&N);
    
    double lr[2] = {0.0, 3.0};
    double flr[2];
    int res = fwrap_eval(2,lr,flr,fw);
    CuAssertIntEquals(tc,0,res);
    

    double delta = 5e-2;
    double hmin = 1e-3;
    struct LinElemXY * xy = NULL;
    lin_elem_adapt(fw,
                   lr[0],flr[0],
                   lr[1],flr[1],
                   delta,hmin,&xy);
    
    struct LinElemXY * temp = xy;
    size_t count = 0;
    while (temp != NULL){
        count ++;
        temp = lin_elem_xy_next(temp);
    }
    CuAssertIntEquals(tc,count,N);
    
    lin_elem_xy_free(xy); xy = NULL;
    fwrap_destroy(fw);
}

void Test_lin_elem_exp_approx_adapt(CuTest * tc){

    printf("Testing function: lin_elem_exp_approx (1) \n");
    
    // function
    int N = 0;
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,fleadapt,&N);

    double lb = -1.0, ub = 1.0;
    double delta = 1e-4, hmin = 1e-2;
    struct LinElemExpAopts * opts = NULL;
    opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,lb,ub,delta,hmin);
    
    le_t le = lin_elem_exp_approx(opts,fw);
    CuAssertIntEquals(tc,N,le->num_nodes);

    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,le,fleadapt,&N,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-5);

    lin_elem_exp_aopts_free(opts);
    LINELEM_FREE(le);
    fwrap_destroy(fw);
}

void Test_lin_elem_exp_approx_adapt2(CuTest * tc){

    printf("Testing function: lin_elem_exp_approx (2) \n");

        // function
    int N = 0;
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,&N);

    double lb = -1.0, ub = 1.0;
    double delta = 1e-4, hmin = 1e-2;
    struct LinElemExpAopts * opts = NULL;
    opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,lb,ub,delta,hmin);
    
    le_t le = lin_elem_exp_approx(opts,fw);
    CuAssertIntEquals(tc,N,le->num_nodes);

    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,le,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-5);

    lin_elem_exp_aopts_free(opts);
    LINELEM_FREE(le);
    fwrap_destroy(fw);
}

void Test_lin_elem_exp_prod(CuTest * tc){

    printf("Testing function: lin_elem_exp_prod \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,powX2,NULL);

    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,TwoPowX3,NULL);

    double lb = -3.0, ub = 2.0;
    double delta = 1e-2, hmin = 1e-2;
    struct LinElemExpAopts * opts = NULL;
    opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,
                                          lb,ub,
                                          delta,hmin);

    le_t f1 = lin_elem_exp_approx(opts,fw);
    le_t f2 = lin_elem_exp_approx(opts,fw1);
    le_t f3 = lin_elem_exp_prod(f1,f2);
    
    size_t N = 100;
    double * pts = linspace(lb,ub,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = LINELEM_EVAL(f3,pts[ii]);
        double eval2 = LINELEM_EVAL(f1,pts[ii]) *
                       LINELEM_EVAL(f2,pts[ii]);
        double diff= fabs(eval1-eval2);
        CuAssertDblEquals(tc, 0.0, diff, 1e-2);
    }
    free(pts); pts = NULL;
    
    LINELEM_FREE(f1);
    LINELEM_FREE(f2);
    LINELEM_FREE(f3);
    lin_elem_exp_aopts_free(opts);
    fwrap_destroy(fw);
    fwrap_destroy(fw1);
}

void Test_lin_elem_exp_derivative(CuTest * tc){

    printf("Testing function: lin_elem_exp_deriv  on (a,b)\n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    double lb = -2.0,ub = -1.0;
    double delta = 1e-4, hmin = 1e-3;
    struct LinElemExpAopts * opts = NULL;
    opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,lb,ub,delta,hmin);

    le_t f1 = lin_elem_exp_approx(opts,fw);
    le_t der = lin_elem_exp_deriv(f1);

    // error
    double abs_err;
    double func_norm;
    compute_error(lb,ub,1000,der,funcderiv,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-5);

    LINELEM_FREE(f1);
    LINELEM_FREE(der);
    lin_elem_exp_aopts_free(opts);
    fwrap_destroy(fw);
}

void Test_lin_elem_exp_integrate(CuTest * tc){

    printf("Testing function: lin_elem_exp_integrate\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,powX2,NULL);


    size_t N = 1000; double lb=-2.0,ub=3.0;
    double * x = linspace(lb,ub,N);
    double f[1000];
    fwrap_eval(N,x,f,fw);
    le_t le = lin_elem_exp_init(N,x,f);

    double intshould = (pow(ub,3) - pow(lb,3))/3;
    double intis = lin_elem_exp_integrate(le);
    CuAssertDblEquals(tc, intshould, intis, 1e-4);

    LINELEM_FREE(le);
    free(x);
    fwrap_destroy(fw);
}

void Test_lin_elem_exp_inner(CuTest * tc){

    printf("Testing function: lin_elem_exp_inner (1) \n");

    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,TwoPowX3,NULL);

    size_t N = 1000; double lb=-2.0,ub=3.0;
    double * x = linspace(lb,ub,N);
    double f[1000], g[1000];
    fwrap_eval(N,x,f,fw1);
    fwrap_eval(N,x,g,fw2);
    
    le_t fa = lin_elem_exp_init(N,x,f);
    le_t fb = lin_elem_exp_init(N,x,g);
        
    double intshould = (pow(ub,6) - pow(lb,6))/3;
    double intis = lin_elem_exp_inner(fa,fb);
    double diff = fabs(intshould-intis)/fabs(intshould);
    CuAssertDblEquals(tc, 0.0, diff, 1e-5);

    LINELEM_FREE(fa);
    LINELEM_FREE(fb);
    free(x);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
    
}

void Test_lin_elem_exp_inner2(CuTest * tc){

    printf("Testing function: lin_elem_exp_inner (2)\n");
    
    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,TwoPowX3,NULL);

    double lb=-2.0,ub=3.0;
    
    size_t N1 = 230;
    double * p1 = linspace(lb,0.5,N1);
    double f[1000];
    fwrap_eval(N1,p1,f,fw1);

    size_t N2 = 349;
    double * p2 = linspace(0.0,ub,N2);
    double g[1000];
    fwrap_eval(N2,p2,g,fw2);
    
    le_t fa = lin_elem_exp_init(N1,p1,f);
    le_t fb = lin_elem_exp_init(N2,p2,g);
        
    double intis = lin_elem_exp_inner(fa,fb);
    double intis2 = lin_elem_exp_inner(fb,fa);

    size_t ntest = 1000000;
    double * xtest = linspace(lb,ub,ntest);
    double integral = 0.0;
    for (size_t ii = 0; ii < ntest; ii++){
        integral += (LINELEM_EVAL(fa,xtest[ii]) *
                     LINELEM_EVAL(fb,xtest[ii]));
    }
    integral /= (double) ntest;
    integral *= (ub - lb);
    double intshould = integral;
    /* double inttrue = 0.00520833333333333; */
    /* printf("int mc %3.15G, int true %3.15G\n", intshould, inttrue); */
    double diff = fabs(intshould-intis)/fabs(intshould);
    /* double difftrue = fabs(inttrue - intis)/ fabs(inttrue); */
    /* printf("\n"); */
    /* printf("diff = %3.15G\n", diff); */
    /* printf("difftrue = %3.15G\n", difftrue); */
    
    /* printf("\n\n\n\n\n\n\n"); */
    CuAssertDblEquals(tc, 0.0, diff, 3e-4);
    CuAssertDblEquals(tc,intis,intis2,1e-15);
    free(xtest);

    LINELEM_FREE(fa);
    LINELEM_FREE(fb);
    free(p1);
    free(p2);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
    
}

void Test_lin_elem_exp_inner3(CuTest * tc){

    printf("Testing function: lin_elem_exp_inner (3) \n");

    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,TwoPowX3,NULL);

    size_t N = 20; double lb=0.0,ub=0.5;
    double * x = linspace(lb,ub,N);
    /* printf("x = "); dprint(N, x); */
    double f[20], g[20];
    fwrap_eval(N,x,f,fw1);
    fwrap_eval(N,x,g,fw2);
    
    le_t fa = lin_elem_exp_init(N,x,f);
    le_t fb = lin_elem_exp_init(N,x,g);
    le_t fc = lin_elem_exp_prod(fa, fb);

    /* printf("integrating product:\n"); */
    double intshould = lin_elem_exp_integrate(fc);

    /* printf("\n\n\n"); */
    /* printf("integrating inner:\n"); */
    /* double inttrue = 0.00520833333333333; */
    double intis = lin_elem_exp_inner(fa,fb);
    double diff = fabs(intshould-intis)/fabs(intshould);
    CuAssertDblEquals(tc, 0.0, diff, 1e-10);

    /* printf("int integrate %3.15G, int true %3.15G\n", intshould, inttrue); */
    /* printf("int is %3.15G\n", intis); */

    /* double difftrue = fabs(inttrue-intis)/fabs(inttrue); */
    /* printf("difftrue = %3.15G\n", difftrue); */

    LINELEM_FREE(fa);
    LINELEM_FREE(fb);
    LINELEM_FREE(fc);
    free(x);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
    
}

void Test_lin_elem_exp_norm(CuTest * tc){
    
    printf("Testing function: lin_elem_exp_norm\n");
    
    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    double lb=-2.0,ub=3.0;
    size_t N = 1000;
    double * x = linspace(lb,ub,N);
    double f[1000];
    fwrap_eval(N,x,f,fw1);
    
    le_t fa = lin_elem_exp_init(N,x,f);

    double intshould = (pow(ub,5) - pow(lb,5))/5;
    double intis = lin_elem_exp_norm(fa);
    double diff = fabs(sqrt(intshould) - intis)/fabs(sqrt(intshould));
    CuAssertDblEquals(tc, 0.0, diff, 3e-6);

    free(x); x = NULL;
    LINELEM_FREE(fa);
    fwrap_destroy(fw1);
}

void Test_lin_elem_exp_axpy(CuTest * tc){

    printf("Testing function: lin_elem_exp_axpy (1) \n");

    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,TwoPowX3,NULL);

    double lb=-2.0, ub=1.0;
    
    size_t N1 = 100;
    double * x1 = linspace(lb,ub,N1);
    double f1[1000];
    fwrap_eval(N1,x1,f1,fw2);

    size_t N2 = 100;
    double * x2 = linspace(lb,ub,N2);
    double f2[1000];
    fwrap_eval(N2,x2,f2,fw1);

    le_t le1 = lin_elem_exp_init(N1,x1,f1);
    le_t le2 = lin_elem_exp_init(N2,x2,f2);
    le_t le3 = lin_elem_exp_copy(le2);

    int success = lin_elem_exp_axpy(2.0,le1,le3);
    CuAssertIntEquals(tc,0,success);

    
    size_t N = 200;
    double * pts = linspace(lb-0.5,ub+0.5,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = LINELEM_EVAL(le3,pts[ii]);
        double eval2 = 2.0 * LINELEM_EVAL(le1,pts[ii]) +
            LINELEM_EVAL(le2,pts[ii]);
        double diff= fabs(eval1-eval2);
        CuAssertDblEquals(tc, 0.0, diff, 4e-15);
    }
    free(pts); pts = NULL;

    
    free(x1); x1 = NULL;
    free(x2); x2 = NULL;
    LINELEM_FREE(le1);
    LINELEM_FREE(le2);
    LINELEM_FREE(le3);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
}

void Test_lin_elem_exp_axpy2(CuTest * tc){

    printf("Testing function: lin_elem_exp_axpy (2) \n");

        // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,TwoPowX3,NULL);

    double lb=-2.0, ub=1.0;
    
    size_t N1 = 302;
    double * x1 = linspace(lb,0.2,N1);
    double f1[1000];
    fwrap_eval(N1,x1,f1,fw2);

    size_t N2 = 20;
    double * x2 = linspace(-0.15,ub,N2);
    double f2[1000];
    fwrap_eval(N2,x2,f2,fw1);

    le_t le1 = lin_elem_exp_init(N1,x1,f1);
    le_t le2 = lin_elem_exp_init(N2,x2,f2);
    le_t le3 = lin_elem_exp_copy(le2);

    int success = lin_elem_exp_axpy(2.0,le1,le3);
    CuAssertIntEquals(tc,0,success);
    
    size_t N = 200;
    double * pts = linspace(lb-0.5,ub+0.5,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = LINELEM_EVAL(le3,pts[ii]);
        double eval2 = 2.0 * LINELEM_EVAL(le1,pts[ii]) +
            LINELEM_EVAL(le2,pts[ii]);
        double diff= fabs(eval1-eval2);
        CuAssertDblEquals(tc, 0.0, diff, 4e-15);
    }
    free(pts); pts = NULL;

    
    free(x1); x1 = NULL;
    free(x2); x2 = NULL;
    LINELEM_FREE(le1);
    LINELEM_FREE(le2);
    LINELEM_FREE(le3);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);

}

void Test_lin_elem_exp_constant(CuTest * tc)
{
    printf("Testing function: lin_elem_exp_constant\n");
    double lb = -2.0;
    double ub = 0.2;
    size_t N = 2;
    double x[2] = {lb,ub};
    struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc(N,x);
    le_t f = lin_elem_exp_constant(2.0,opts);

    double * xtest = linspace(lb,ub,1000);
    for (size_t ii = 0; ii < 1000; ii++){
        double val = LINELEM_EVAL(f,xtest[ii]);
        CuAssertDblEquals(tc,2.0,val,1e-15);
    }
    free(xtest);

    LINELEM_FREE(f);
    lin_elem_exp_aopts_free(opts);
}

void Test_lin_elem_exp_flipsign(CuTest * tc)
{
    printf("Testing function: lin_elem_exp_flip_sign\n");
    double lb = -2.0;
    double ub = 0.2;
    size_t N = 2;
    double x[2] = {lb,ub};
    struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc(N,x);
    le_t f = lin_elem_exp_constant(0.3,opts);
    lin_elem_exp_flip_sign(f);
    
    double * xtest = linspace(lb,ub,1000);
    for (size_t ii = 0; ii < 1000; ii++){
        double val = LINELEM_EVAL(f,xtest[ii]);
        CuAssertDblEquals(tc,-0.3,val,1e-15);
    }
    free(xtest);
    
    LINELEM_FREE(f);
    lin_elem_exp_aopts_free(opts);
}

void Test_lin_elem_exp_scale(CuTest * tc)
{
    printf("Testing function: lin_elem_exp_scale\n");
    double lb = -2.0;
    double ub = 0.2;
    size_t N = 2;
    double x[2] = {lb,ub};
    struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc(N,x);
    le_t f = lin_elem_exp_constant(0.3,opts);
    lin_elem_exp_scale(0.3, f);
    double * xtest = linspace(lb,ub,1000);
    for (size_t ii = 0; ii < 1000; ii++){
        double val = LINELEM_EVAL(f,xtest[ii]);
        CuAssertDblEquals(tc,0.09,val,1e-15);
    }
    free(xtest);
    
    LINELEM_FREE(f);
    lin_elem_exp_aopts_free(opts);
}

void Test_lin_elem_exp_orth_basis(CuTest * tc)
{
    printf("Testing function: lin_elem_exp_orth_basis\n");
    double lb = -2.0;
    double ub = 0.2;
    size_t N = 100;
    double * x = linspace(lb,ub,N);
    struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc(N,x);
    
    /* double * coeff = calloc_double(N); */
    le_t f[100];
    for (size_t ii = 0; ii < N; ii++){
        f[ii] = NULL;// lin_elem_exp_init(N,x,coeff);
    }

    lin_elem_exp_orth_basis(N,f,opts);
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            double val = lin_elem_exp_inner(f[ii],f[jj]);
            if (ii == jj){
                CuAssertDblEquals(tc,1.0,val,1e-15);
            }
            else{
                CuAssertDblEquals(tc,0.0,val,1e-15);
            }
        }
    }

    for (size_t ii = 0; ii < N; ii++){
        LINELEM_FREE(f[ii]);
    }
    free(x); x = NULL;
    /* free(coeff); coeff = NULL; */
    lin_elem_exp_aopts_free(opts);
}


void Test_lin_elem_exp_serialize(CuTest * tc){
    
    printf("Testing functions: (de)serializing lin_elem_exp \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,TwoPowX3,NULL);
    
    double lb = -1.0;
    double ub = 2.0;
    size_t N1 = 10;
    double * x1 = linspace(lb,ub,N1);
    double f1[1000];
    fwrap_eval(N1,x1,f1,fw);

    le_t pl = lin_elem_exp_init(N1,x1,f1);
    free(x1); x1 = NULL;

      
    unsigned char * text = NULL;
    size_t size_to_be;
    serialize_lin_elem_exp(text, pl, &size_to_be);
    text = malloc(size_to_be * sizeof(unsigned char));
    serialize_lin_elem_exp(text, pl, NULL);
     

    le_t pt = NULL;
    deserialize_lin_elem_exp(text, &pt);

    double * xtest = linspace(lb,ub,1000);
    size_t ii;
    double err = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(LINELEM_EVAL(pl,xtest[ii]) -
                   LINELEM_EVAL(pt,xtest[ii]),2);
    }
    err = sqrt(err);
    CuAssertDblEquals(tc, 0.0, err, 1e-15);

    free(xtest);
    free(text);
    LINELEM_FREE(pl);
    LINELEM_FREE(pt);
    fwrap_destroy(fw);
}

void Test_lin_elem_exp_savetxt(CuTest * tc){
    
    printf("Testing functions: lin_elem_exp_savetxt and _loadtxt \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,TwoPowX3,NULL);
    
    double lb = -1.0;
    double ub = 2.0;
    size_t N1 = 10;
    double * x1 = linspace(lb,ub,N1);
    double f1[1000];
    fwrap_eval(N1,x1,f1,fw);

    le_t pl = lin_elem_exp_init(N1,x1,f1);
    free(x1); x1 = NULL;

      
    FILE * fp = fopen("testlesave.txt","w+");
    size_t prec = 21;
    lin_elem_exp_savetxt(pl,fp,prec);
    rewind(fp);

    le_t pt = NULL;
    pt = lin_elem_exp_loadtxt(fp);

    double * xtest = linspace(lb,ub,1000);
    size_t ii;
    double err = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(LINELEM_EVAL(pl,xtest[ii]) -
                   LINELEM_EVAL(pt,xtest[ii]),2);
    }
    err = sqrt(err);
    CuAssertDblEquals(tc, 0.0, err, 1e-15);

    
    free(xtest);
    fclose(fp);
    LINELEM_FREE(pl);
    LINELEM_FREE(pt);
    fwrap_destroy(fw);
}

static void regress_func(size_t N, const double * x, double * out)
{
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = -1.0 * pow(x[ii],7) + 2.0*pow(x[ii],2) + 0.2 * 0.5*x[ii]; 
    }
}

void Test_LS_regress(CuTest * tc){
    
    printf("Testing functions: least squares regression with linear elements\n");

    double lb = -1.0;
    double ub =  1.0;
    size_t nparams = 40;
    double * params = linspace(lb,ub,nparams);

    struct LinElemExpAopts * aopts =
        lin_elem_exp_aopts_alloc(nparams,params);
    
    // create data
    size_t ndata = 1000;
    double * x = linspace(lb,ub,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    c3opt_set_relftol(optimizer,1e-20);
    c3opt_set_absxtol(optimizer,1e-20);
    c3opt_set_gtol(optimizer,1e-20);
    
    struct Regress1DOpts * regopts =
        regress_1d_opts_create(PARAMETRIC,LS,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,LINELM,aopts);
    regress_1d_opts_set_initial_parameters(regopts,params);

    /* printf("check deriv\n"); */
    // check derivative
    c3opt_add_objective(optimizer,param_LSregress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    for (size_t ii = 0; ii < nparams; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        /* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); */
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    /* printf("start\n"); */
    int info;
    struct GenericFunction * gf =
        generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(lb,ub,1000);
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
    /* CuAssertDblEquals(tc, 0.0, rat, 1e-3); */
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;
    generic_function_free(gf); gf = NULL;;

    
    free(params);
    free(x); x = NULL;
    free(y); y = NULL;
    lin_elem_exp_aopts_free(aopts);  aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;

}

void Test_RLS2_regress(CuTest * tc){
    
    printf("Testing functions: ridge regression with linear elements\n");

    size_t nparams = 50;
    double lb = -1.0;
    double ub =  1.0;
    double * params = linspace(lb,ub,nparams);

    struct LinElemExpAopts * aopts =
        lin_elem_exp_aopts_alloc(nparams,params);

    // create data
    size_t ndata = 1000;
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
    regress_1d_opts_set_parametric_form(regopts,LINELM,aopts);
    /* regress_1d_opts_set_regularization_penalty(regopts,1e-2/sqrt(ndata)); */
    regress_1d_opts_set_regularization_penalty(regopts,1);
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_RLS2regress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    for (size_t ii = 0; ii < nparams; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        /* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); */
    }
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(lb,ub,1000);
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
    /* CuAssertDblEquals(tc, 0.0, rat, 1e-3); */
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(params); params = NULL;
    free(x); x = NULL;
    free(y); y = NULL;
    lin_elem_exp_aopts_free(aopts);
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}

void Test_RLSD2_regress(CuTest * tc){
    
    printf("Testing functions: ridge regression on second deriv with linear elements\n");

    size_t nparams = 50;
    double lb = -1.0;
    double ub =  1.0;
    double * params = linspace(lb,ub,nparams);

    struct LinElemExpAopts * aopts =
        lin_elem_exp_aopts_alloc(nparams,params);

    // create data
    size_t ndata = 1000;
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
    regress_1d_opts_set_parametric_form(regopts,LINELM,aopts);
    regress_1d_opts_set_regularization_penalty(regopts,1e-4/sqrt(ndata));
    /* regress_1d_opts_set_regularization_penalty(regopts,1); */
    regress_1d_opts_set_initial_parameters(regopts,params);

    c3opt_add_objective(optimizer,param_RLS2regress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    for (size_t ii = 0; ii < nparams; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        /* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); */
    }
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    CuAssertIntEquals(tc,1,info>-1);
    
    double * xtest = linspace(lb,ub,1000);
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
    /* CuAssertDblEquals(tc, 0.0, rat, 1e-3); */
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    free(params); params = NULL;
    lin_elem_exp_aopts_free(aopts); aopts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
}

void Test_lin_elem_exp_dderiv(CuTest * tc){

    printf("Testing function: lin_elem_exp_dderiv\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,gaussbump2,NULL);

    size_t N = 100;
    double lb=-7.0,ub=7.0;
    double * x = linspace(lb,ub,N);
    double f[100];
    fwrap_eval(N,x,f,fw);
    le_t le = lin_elem_exp_init(N,x,f);

    le_t ledd = lin_elem_exp_dderiv(le);
    
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,100,ledd,gaussbump2dd,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-5);
    
    free(x);
    LINELEM_FREE(le);
    LINELEM_FREE(ledd);
    fwrap_destroy(fw);
}

void Test_lin_elem_exp_dderiv_periodic(CuTest * tc){

    printf("Testing function: lin_elem_exp_dderiv_periodic (1/2)\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,gaussbump2,NULL);

    size_t N = 100;
    double lb=-7.0,ub=7.0;
    double * x = linspace(lb,ub,N);
    double f[100];
    fwrap_eval(N,x,f,fw);
    le_t le = lin_elem_exp_init(N,x,f);

    le_t ledd = lin_elem_exp_dderiv_periodic(le);
    
    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,100,ledd,gaussbump2dd,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-16);
    
    free(x);
    LINELEM_FREE(le);
    LINELEM_FREE(ledd);
    fwrap_destroy(fw);
}

void Test_lin_elem_exp_dderiv_periodic2(CuTest * tc){

    printf("DOESNT WORK DONT RUN YET Testing function: lin_elem_exp_dderiv_periodic (2/2)\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,sin_lift,NULL);

    size_t N = 20;
    double lb=0.0,ub=2.0*M_PI;
    double * x = linspace(lb,ub,N);
    double f[100];
    fwrap_eval(N,x,f,fw);
    le_t le = lin_elem_exp_init(N-1,x,f);

    le_t ledd = lin_elem_exp_dderiv_periodic(le);
    
    double abs_err;
    double func_norm;
    compute_error_vec(lb,x[N-2],N-1,ledd,sin_liftdd,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    CuAssertDblEquals(tc, 0.0, err, 1e-14);
    
    free(x);
    LINELEM_FREE(le);
    fwrap_destroy(fw);
}


CuSuite * LelmGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_approx);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_adapt);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_approx_adapt);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_approx_adapt2);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_prod);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_derivative);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_integrate);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_inner);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_inner2);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_inner3);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_norm);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_axpy);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_axpy2);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_constant);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_flipsign);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_scale);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_orth_basis);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_serialize);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_savetxt);

    SUITE_ADD_TEST(suite, Test_LS_regress);
    SUITE_ADD_TEST(suite, Test_RLS2_regress);
    SUITE_ADD_TEST(suite, Test_RLSD2_regress);

    SUITE_ADD_TEST(suite, Test_lin_elem_exp_dderiv);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_dderiv_periodic);

    // doesnt work yet
    /* SUITE_ADD_TEST(suite, Test_lin_elem_exp_dderiv_periodic2); */
    return suite;
}
