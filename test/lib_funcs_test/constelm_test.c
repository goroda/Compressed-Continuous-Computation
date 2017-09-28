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

typedef struct ConstElemExp* le_t;
#define CONSTELEM_EVAL const_elem_exp_eval
#define CONSTELEM_FREE const_elem_exp_free

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
    for (ii = 0; ii < N; ii++){
        func(1,xtest+ii,&val,arg);
        double eval = CONSTELEM_EVAL(le,xtest[ii]);
        /* printf("eval = %G, val = %G\n",eval,val); */
        *abs_err += pow(eval - val,2);
        *func_norm += pow(val,2);
    }
    free(xtest); xtest = NULL;
}

void Test_const_elem_exp_approx(CuTest * tc){

    printf("Testing function: const_elem_exp_init\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,Sin3xTx2,NULL);

    size_t N = 500; double lb=-1.0,ub=1.0;
    double * x = linspace(lb,ub,N);
    double * f = calloc_double(N);
    fwrap_eval(N,x,f,fw);
    le_t le = const_elem_exp_init(N,x,f);

    double abs_err;
    double func_norm;
    compute_error_vec(lb,ub,1000,le,Sin3xTx2,NULL,&abs_err,&func_norm);
    double err = abs_err / func_norm;
    /* printf("err = %G, norm = %G\n",abs_err,func_norm); */
    CuAssertDblEquals(tc, 0.0, err, 1e-4);
    
    free(x); x = NULL;
    free(f); f = NULL;
    CONSTELEM_FREE(le);
    fwrap_destroy(fw);
}

void Test_const_elem_exp_prod(CuTest * tc){

    printf("Testing function: const_elem_exp_prod \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,powX2,NULL);

    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,TwoPowX3,NULL);

    double lb = -3.0, ub = 2.0;
    size_t Nx = 100;
    double * x = linspace(lb,ub,Nx);
    struct ConstElemExpAopts * opts = NULL;
    opts = const_elem_exp_aopts_alloc(Nx,x);

    le_t f1 = const_elem_exp_approx(opts,fw);
    le_t f2 = const_elem_exp_approx(opts,fw1);
    le_t f3 = const_elem_exp_prod(f1,f2);
    
    size_t N = 100;
    double * pts = linspace(lb,ub,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = CONSTELEM_EVAL(f3,pts[ii]);
        double eval2 = CONSTELEM_EVAL(f1,pts[ii]) *
                       CONSTELEM_EVAL(f2,pts[ii]);
        double diff= fabs(eval1-eval2);
        CuAssertDblEquals(tc, 0.0, diff, 1e-2);
    }
    free(pts); pts = NULL;

    free(x); x = NULL;
    CONSTELEM_FREE(f1);
    CONSTELEM_FREE(f2);
    CONSTELEM_FREE(f3);
    const_elem_exp_aopts_free(opts);
    fwrap_destroy(fw);
    fwrap_destroy(fw1);
}

void Test_const_elem_exp_integrate(CuTest * tc){

    printf("Testing function: const_elem_exp_integrate\n");

    // function
    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,powX2,NULL);


    size_t N = 1000; double lb=-2.0,ub=3.0;
    double * x = linspace(lb,ub,N);
    double f[1000];
    fwrap_eval(N,x,f,fw);
    le_t le = const_elem_exp_init(N,x,f);

    double intshould = (pow(ub,3) - pow(lb,3))/3;
    double intis = const_elem_exp_integrate(le);
    CuAssertDblEquals(tc, intshould, intis, 1e-4);

    CONSTELEM_FREE(le);
    free(x);
    fwrap_destroy(fw);
}

void Test_const_elem_exp_inner(CuTest * tc){

    printf("Testing function: const_elem_exp_inner (1) \n");

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
    
    le_t fa = const_elem_exp_init(N,x,f);
    le_t fb = const_elem_exp_init(N,x,g);
        
    double intshould = (pow(ub,6) - pow(lb,6))/3;
    double intis = const_elem_exp_inner(fa,fb);
    double diff = fabs(intshould-intis)/fabs(intshould);
    CuAssertDblEquals(tc, 0.0, diff, 1e-5);

    CONSTELEM_FREE(fa);
    CONSTELEM_FREE(fb);
    free(x);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
    
}

void Test_const_elem_exp_inner2(CuTest * tc){

    printf("Testing function: const_elem_exp_inner (2)\n");
    
    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    struct Fwrap * fw2 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw2,TwoPowX3,NULL);

    double lb=-2.0,ub=3.0;
    
    size_t N1 = 200;
    double * p1 = linspace(lb,0.5,N1);
    double f[1000];
    fwrap_eval(N1,p1,f,fw1);

    size_t N2 = 800;
    double * p2 = linspace(0.0,ub,N2);
    double g[1000];
    fwrap_eval(N2,p2,g,fw2);
    
    le_t fa = const_elem_exp_init(N1,p1,f);
    le_t fb = const_elem_exp_init(N2,p2,g);
        
    double intis = const_elem_exp_inner(fa,fb);
    double intis2 = const_elem_exp_inner(fb,fa);

    size_t ntest = 10000000;
    double * xtest = linspace(lb,ub,ntest);
    double integral = 0.0;
    for (size_t ii = 0; ii < ntest; ii++){
        integral += (CONSTELEM_EVAL(fa,xtest[ii]) *
                     CONSTELEM_EVAL(fb,xtest[ii]));
    }
    integral /= (double) ntest;
    integral *= (ub - lb);
    double intshould = integral;
    double diff = fabs(intshould-intis)/fabs(intshould);
    CuAssertDblEquals(tc, 0.0, diff, 1e-3);
    CuAssertDblEquals(tc,intis,intis2,1e-15);
    free(xtest);

    CONSTELEM_FREE(fa);
    CONSTELEM_FREE(fb);
    free(p1);
    free(p2);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
    
}

void Test_const_elem_exp_norm(CuTest * tc){
    
    printf("Testing function: const_elem_exp_norm\n");
    
    // function
    struct Fwrap * fw1 = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw1,powX2,NULL);

    double lb=-2.0,ub=3.0;
    size_t N = 1000;
    double * x = linspace(lb,ub,N);
    double f[1000];
    fwrap_eval(N,x,f,fw1);
    
    le_t fa = const_elem_exp_init(N,x,f);

    double intshould = (pow(ub,5) - pow(lb,5))/5;
    double intis = const_elem_exp_norm(fa);
    double diff = fabs(sqrt(intshould) - intis)/fabs(sqrt(intshould));
    CuAssertDblEquals(tc, 0.0, diff, 1e-4);

    free(x); x = NULL;
    CONSTELEM_FREE(fa);
    fwrap_destroy(fw1);
}

void Test_const_elem_exp_axpy(CuTest * tc){

    printf("Testing function: const_elem_exp_axpy (1) \n");

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

    le_t le1 = const_elem_exp_init(N1,x1,f1);
    le_t le2 = const_elem_exp_init(N2,x2,f2);
    le_t le3 = const_elem_exp_copy(le2);

    int success = const_elem_exp_axpy(2.0,le1,le3);
    CuAssertIntEquals(tc,0,success);

    
    size_t N = 200;
    double * pts = linspace(lb-0.5,ub+0.5,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = CONSTELEM_EVAL(le3,pts[ii]);
        double eval2 = 2.0 * CONSTELEM_EVAL(le1,pts[ii]) +
            CONSTELEM_EVAL(le2,pts[ii]);
        double diff= fabs(eval1-eval2);
        CuAssertDblEquals(tc, 0.0, diff, 4e-15);
    }
    free(pts); pts = NULL;

    
    free(x1); x1 = NULL;
    free(x2); x2 = NULL;
    CONSTELEM_FREE(le1);
    CONSTELEM_FREE(le2);
    CONSTELEM_FREE(le3);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);
}

void Test_const_elem_exp_axpy2(CuTest * tc){

    printf("Testing function: const_elem_exp_axpy (2) \n");

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

    le_t le1 = const_elem_exp_init(N1,x1,f1);
    le_t le2 = const_elem_exp_init(N2,x2,f2);
    le_t le3 = const_elem_exp_copy(le2);

    int success = const_elem_exp_axpy(2.0,le1,le3);
    CuAssertIntEquals(tc,0,success);
    
    size_t N = 200;
    double * pts = linspace(lb-0.5,ub+0.5,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = CONSTELEM_EVAL(le3,pts[ii]);
        double eval2 = 2.0 * CONSTELEM_EVAL(le1,pts[ii]) +
            CONSTELEM_EVAL(le2,pts[ii]);
        double diff= fabs(eval1-eval2);
        CuAssertDblEquals(tc, 0.0, diff, 1e-3);
    }
    free(pts); pts = NULL;

    
    free(x1); x1 = NULL;
    free(x2); x2 = NULL;
    CONSTELEM_FREE(le1);
    CONSTELEM_FREE(le2);
    CONSTELEM_FREE(le3);
    fwrap_destroy(fw1);
    fwrap_destroy(fw2);

}

void Test_const_elem_exp_constant(CuTest * tc)
{
    printf("Testing function: const_elem_exp_constant\n");
    double lb = -2.0;
    double ub = 0.2;
    size_t N = 2;
    double x[2] = {lb,ub};
    struct ConstElemExpAopts * opts = const_elem_exp_aopts_alloc(N,x);
    le_t f = const_elem_exp_constant(2.0,opts);

    double * xtest = linspace(lb,ub,1000);
    for (size_t ii = 0; ii < 1000; ii++){
        double val = CONSTELEM_EVAL(f,xtest[ii]);
        CuAssertDblEquals(tc,2.0,val,1e-15);
    }
    free(xtest);

    CONSTELEM_FREE(f);
    const_elem_exp_aopts_free(opts);
}

void Test_const_elem_exp_flipsign(CuTest * tc)
{
    printf("Testing function: const_elem_exp_flip_sign\n");
    double lb = -2.0;
    double ub = 0.2;
    size_t N = 2;
    double x[2] = {lb,ub};
    struct ConstElemExpAopts * opts = const_elem_exp_aopts_alloc(N,x);
    le_t f = const_elem_exp_constant(0.3,opts);
    const_elem_exp_flip_sign(f);
    
    double * xtest = linspace(lb,ub,1000);
    for (size_t ii = 0; ii < 1000; ii++){
        double val = CONSTELEM_EVAL(f,xtest[ii]);
        CuAssertDblEquals(tc,-0.3,val,1e-15);
    }
    free(xtest);
    
    CONSTELEM_FREE(f);
    const_elem_exp_aopts_free(opts);
}

void Test_const_elem_exp_scale(CuTest * tc)
{
    printf("Testing function: const_elem_exp_scale\n");
    double lb = -2.0;
    double ub = 0.2;
    size_t N = 2;
    double x[2] = {lb,ub};
    struct ConstElemExpAopts * opts = const_elem_exp_aopts_alloc(N,x);
    le_t f = const_elem_exp_constant(0.3,opts);
    const_elem_exp_scale(0.3, f);
    double * xtest = linspace(lb,ub,1000);
    for (size_t ii = 0; ii < 1000; ii++){
        double val = CONSTELEM_EVAL(f,xtest[ii]);
        CuAssertDblEquals(tc,0.09,val,1e-15);
    }
    free(xtest);
    
    CONSTELEM_FREE(f);
    const_elem_exp_aopts_free(opts);
}

void Test_const_elem_exp_orth_basis(CuTest * tc)
{
    printf("Testing function: const_elem_exp_orth_basis\n");
    double lb = -2.0;
    double ub = 0.2;
    size_t N = 100;
    double * x = linspace(lb,ub,N);
    struct ConstElemExpAopts * opts = const_elem_exp_aopts_alloc(N,x);
    
    /* double * coeff = calloc_double(N); */
    le_t f[100];
    for (size_t ii = 0; ii < N; ii++){
        f[ii] = NULL;// const_elem_exp_init(N,x,coeff);
    }

    const_elem_exp_orth_basis(N,f,opts);
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            double val = const_elem_exp_inner(f[ii],f[jj]);
            if (ii == jj){
                CuAssertDblEquals(tc,1.0,val,1e-15);
            }
            else{
                CuAssertDblEquals(tc,0.0,val,1e-15);
            }
        }
    }

    for (size_t ii = 0; ii < N; ii++){
        CONSTELEM_FREE(f[ii]);
    }
    free(x); x = NULL;
    /* free(coeff); coeff = NULL; */
    const_elem_exp_aopts_free(opts);
}


void Test_const_elem_exp_serialize(CuTest * tc){
    
    printf("Testing functions: (de)serializing const_elem_exp \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,TwoPowX3,NULL);
    
    double lb = -1.0;
    double ub = 2.0;
    size_t N1 = 10;
    double * x1 = linspace(lb,ub,N1);
    double f1[1000];
    fwrap_eval(N1,x1,f1,fw);

    le_t pl = const_elem_exp_init(N1,x1,f1);
    free(x1); x1 = NULL;

      
    unsigned char * text = NULL;
    size_t size_to_be;
    serialize_const_elem_exp(text, pl, &size_to_be);
    text = malloc(size_to_be * sizeof(unsigned char));
    serialize_const_elem_exp(text, pl, NULL);
     

    le_t pt = NULL;
    deserialize_const_elem_exp(text, &pt);

    double * xtest = linspace(lb,ub,1000);
    size_t ii;
    double err = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(CONSTELEM_EVAL(pl,xtest[ii]) -
                   CONSTELEM_EVAL(pt,xtest[ii]),2);
    }
    err = sqrt(err);
    CuAssertDblEquals(tc, 0.0, err, 1e-15);

    free(xtest);
    free(text);
    CONSTELEM_FREE(pl);
    CONSTELEM_FREE(pt);
    fwrap_destroy(fw);
}

void Test_const_elem_exp_savetxt(CuTest * tc){
    
    printf("Testing functions: const_elem_exp_savetxt and _loadtxt \n");

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,TwoPowX3,NULL);
    
    double lb = -1.0;
    double ub = 2.0;
    size_t N1 = 10;
    double * x1 = linspace(lb,ub,N1);
    double f1[1000];
    fwrap_eval(N1,x1,f1,fw);

    le_t pl = const_elem_exp_init(N1,x1,f1);
    free(x1); x1 = NULL;

      
    FILE * fp = fopen("testlesave.txt","w+");
    size_t prec = 21;
    const_elem_exp_savetxt(pl,fp,prec);
    rewind(fp);

    le_t pt = NULL;
    pt = const_elem_exp_loadtxt(fp);

    double * xtest = linspace(lb,ub,1000);
    size_t ii;
    double err = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(CONSTELEM_EVAL(pl,xtest[ii]) -
                   CONSTELEM_EVAL(pt,xtest[ii]),2);
    }
    err = sqrt(err);
    CuAssertDblEquals(tc, 0.0, err, 1e-15);

    
    free(xtest);
    fclose(fp);
    CONSTELEM_FREE(pl);
    CONSTELEM_FREE(pt);
    fwrap_destroy(fw);
}



CuSuite * CelmGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_const_elem_exp_approx);
    SUITE_ADD_TEST(suite, Test_const_elem_exp_prod);
    SUITE_ADD_TEST(suite, Test_const_elem_exp_integrate);
    SUITE_ADD_TEST(suite, Test_const_elem_exp_inner);
    SUITE_ADD_TEST(suite, Test_const_elem_exp_inner2);
    SUITE_ADD_TEST(suite, Test_const_elem_exp_norm);
    SUITE_ADD_TEST(suite, Test_const_elem_exp_axpy);
    SUITE_ADD_TEST(suite, Test_const_elem_exp_axpy2);
    SUITE_ADD_TEST(suite, Test_const_elem_exp_constant);
    SUITE_ADD_TEST(suite, Test_const_elem_exp_flipsign);
    SUITE_ADD_TEST(suite, Test_const_elem_exp_scale);
    SUITE_ADD_TEST(suite, Test_const_elem_exp_orth_basis);
    SUITE_ADD_TEST(suite, Test_const_elem_exp_serialize);
    SUITE_ADD_TEST(suite, Test_const_elem_exp_savetxt);

    return suite;
}
