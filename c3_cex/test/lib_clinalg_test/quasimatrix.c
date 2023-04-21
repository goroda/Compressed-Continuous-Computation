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

static void
quasimatrix_funcs_equal(CuTest * tc,
                        size_t n, struct Quasimatrix * A,
                        struct Quasimatrix * B, double level)

{
    for (size_t ii = 0; ii < n; ii++){
        struct GenericFunction *f1,*f2;
        f1 = quasimatrix_get_func(A,ii);
        f2 = quasimatrix_get_func(B,ii);
        double diff = generic_function_norm2diff(f1,f2);
        /* printf("ii=%zu, diff=%G\n",ii,diff); */
        CuAssertDblEquals(tc,0.0,diff,level);
    }
}

void Test_quasimatrix_rank(CuTest * tc){

    printf("Testing function: quasimatrix_rank (1/2) \n");

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,3);
    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func,NULL);
    fwrap_set_func_array(fw,2,func,NULL);
    
    // approximation options
    enum function_class fc[3] = {POLYNOMIAL, POLYNOMIAL, POLYNOMIAL};
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.0);
    ope_opts_set_ub(opts,1.0);

    // first test
    struct Quasimatrix * A = quasimatrix_approx1d(3,fw,fc,opts);
    size_t rank = quasimatrix_rank(A,opts);
    CuAssertIntEquals(tc, 1, rank);


    fwrap_set_func_array(fw,0,func, NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);

    // second test
    struct Quasimatrix * B = quasimatrix_approx1d(3,fw,fc,opts);
    size_t rank2 = quasimatrix_rank(B,opts);
    CuAssertIntEquals(tc, 3, rank2);

    quasimatrix_free(A);
    quasimatrix_free(B);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}

void Test_quasimatrix_rank2(CuTest * tc){

    printf("Testing function: quasimatrix_rank (2/2) \n");

    // functions
    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,3);
    fwrap_set_func_array(fw,0,func, NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);

    // approximation options
    enum function_class fc[3] = {POLYNOMIAL,
                                 POLYNOMIAL,
                                 POLYNOMIAL};
    enum function_class fc2[3] = {PIECEWISE,
                                 PIECEWISE,
                                 PIECEWISE};

    
    struct PwPolyOpts * pwopts = pw_poly_opts_alloc(LEGENDRE,-1.0,1.0);
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.0);
    ope_opts_set_ub(opts,1.0);

    struct Quasimatrix * A = quasimatrix_approx1d(3,fw,fc2,pwopts);
    struct Quasimatrix * At = quasimatrix_approx1d(3,fw,fc,opts);           
    quasimatrix_funcs_equal(tc,3,A,At,1e-13);

    //print_quasimatrix(A,0,NULL);
    size_t rank = quasimatrix_rank(A,pwopts);
    CuAssertIntEquals(tc, 3, rank);
    quasimatrix_free(A);
    quasimatrix_free(At);

    fwrap_set_func_array(fw,0,func,NULL);
    fwrap_set_func_array(fw,1,func,NULL);
    fwrap_set_func_array(fw,2,func,NULL);

    struct Quasimatrix * B = quasimatrix_approx1d(3,fw,fc2,pwopts);
    struct Quasimatrix * Bt = quasimatrix_approx1d(3,fw,fc,opts);           
    quasimatrix_funcs_equal(tc,3,B,Bt,1e-13);

    struct Quasimatrix * Bdiff = quasimatrix_daxpby(1.0,B,-1.0,Bt);
    double diff = quasimatrix_norm(Bdiff);
    CuAssertDblEquals(tc,0.0,diff,1e-13);

    size_t rank2 = quasimatrix_rank(Bt,pwopts);
    CuAssertIntEquals(tc, 1, rank2);
    quasimatrix_free(B);
    quasimatrix_free(Bt);
    quasimatrix_free(Bdiff);

    fwrap_destroy(fw);
    ope_opts_free(opts);
    pw_poly_opts_free(pwopts);
}

void Test_quasimatrix_householder(CuTest * tc){

    printf("Testing function: quasimatrix_householder\n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,3);
    fwrap_set_func_array(fw,0,func, NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    
    enum function_class fc[3] = {POLYNOMIAL,
                                 POLYNOMIAL,
                                 POLYNOMIAL};
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.0);
    ope_opts_set_ub(opts,1.0);

    struct Quasimatrix * A = quasimatrix_approx1d(3,fw,fc,opts);
    struct Quasimatrix * Acopy = quasimatrix_copy(A);
    double * R = calloc_double(3*3);
    struct Quasimatrix * Q = quasimatrix_householder_simple(A,R,opts);
    struct Quasimatrix * Anew = qmm(Q,R,3);
    quasimatrix_funcs_equal(tc,3,Acopy,Anew,1e-13);
    
    quasimatrix_free(A);
    quasimatrix_free(Q);
    quasimatrix_free(Anew);
    quasimatrix_free(Acopy);
    free(R);

    fwrap_destroy(fw);
    ope_opts_free(opts);
}

void Test_quasimatrix_householder_weird_domain(CuTest * tc){

    printf("Testing function: quasimatrix_householder on (-3.0, 2.0)\n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,3);
    fwrap_set_func_array(fw,0,func, NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    
    enum function_class fc[3] = {PIECEWISE,
                                 PIECEWISE,
                                 PIECEWISE};
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,-3.0,2.0);
    struct Quasimatrix * A = quasimatrix_approx1d(3,fw,fc,opts);
    
    struct Quasimatrix * Acopy = quasimatrix_copy(A);
    double * R = calloc_double(3*3);
    
    struct Quasimatrix * Q = quasimatrix_householder_simple(A,R,opts);
    for (size_t ii = 0; ii < 3; ii++){
        struct GenericFunction * gf = quasimatrix_get_func(Q,ii);
        double norm = generic_function_norm(gf);
        CuAssertDblEquals(tc,1.0,norm,1e-13);
    }

    struct Quasimatrix * Anew = qmm(Q,R,3);
    for (size_t ii = 0; ii < 3; ii++){
        struct GenericFunction * gf = quasimatrix_get_func(Anew,ii);
        double integral = generic_function_integrate(gf);

        struct GenericFunction * g2 = quasimatrix_get_func(Acopy,ii);
        double integral2 = generic_function_integrate(g2);
        CuAssertDblEquals(tc,integral,integral2,1e-13);
    }

    quasimatrix_funcs_equal(tc,3,Acopy,Anew,1e-13);
    
    quasimatrix_free(A);
    quasimatrix_free(Q);
    quasimatrix_free(Anew);
    quasimatrix_free(Acopy);
    free(R);
    fwrap_destroy(fw);
    pw_poly_opts_free(opts);
}

void Test_quasimatrix_lu1d(CuTest * tc){

    printf("Testing function: quasimatrix_lu1d\n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,3);
    fwrap_set_func_array(fw,0,func, NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    
    enum function_class fc[3] = {POLYNOMIAL,
                                 POLYNOMIAL,
                                 POLYNOMIAL};
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.0);
    ope_opts_set_ub(opts,1.0);
    
    struct Quasimatrix * A = quasimatrix_approx1d(3,fw,fc,opts);
    
    struct Quasimatrix * L = quasimatrix_alloc(3);
    double * eye = calloc_double(9);
    for (size_t ii = 0; ii < 3; ii++)  eye[ii * 3 + ii] = 1.0;

    struct Quasimatrix * Acopy = qmm(A,eye,3);
    double * U = calloc_double(3*3);
    double * piv = calloc_double(3);
    quasimatrix_lu1d(A,L,U,piv,opts,NULL);
     
    double eval;
    struct GenericFunction * lf;
    lf = quasimatrix_get_func(L,1);
    eval = generic_function_1d_eval(lf, piv[0]);
    CuAssertDblEquals(tc, 0.0, eval, 1e-13);
    lf = quasimatrix_get_func(L,2);
    eval = generic_function_1d_eval(lf, piv[0]);
    CuAssertDblEquals(tc, 0.0, eval, 1e-13);
    lf = quasimatrix_get_func(L,2);
    eval = generic_function_1d_eval(lf, piv[1]);
    CuAssertDblEquals(tc, 0.0, eval, 1e-13);

    struct Quasimatrix * Anew = qmm(L,U,3);
    quasimatrix_funcs_equal(tc,3,Acopy,Anew,1e-13);
    
    quasimatrix_free(A);
    quasimatrix_free(L);
    quasimatrix_free(Anew);
    quasimatrix_free(Acopy);
    free(eye);
    free(U);
    free(piv);
    ope_opts_free(opts);
    fwrap_destroy(fw);
}

void Test_quasimatrix_maxvol1d(CuTest * tc){

    printf("Testing function: quasimatrix_maxvol1d\n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,3);
    fwrap_set_func_array(fw,0,func, NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    
    enum function_class fc[3] = {PIECEWISE,
                                 PIECEWISE,
                                 PIECEWISE};
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,-1.0,1.0);
    struct Quasimatrix * A = quasimatrix_approx1d(3,fw,fc,opts);

    //printf("got approximation\n");
    double * Asinv = calloc_double(3*3);
    double * piv = calloc_double(3);
    quasimatrix_maxvol1d(A,Asinv,piv,opts,NULL);
     
    //printf("pivots at = \n");
    //dprint(3,piv);

    struct Quasimatrix * B = qmm(A,Asinv,3);
    double maxval, maxloc;
    quasimatrix_absmax(B,&maxloc,&maxval,NULL);
    //printf("Less = %d", 1.0+1e-2 > maxval);
    CuAssertIntEquals(tc, 1, (1.0+1e-2) > maxval);

    quasimatrix_free(B);
    quasimatrix_free(A);
    free(Asinv);
    free(piv);
    pw_poly_opts_free(opts);
    fwrap_destroy(fw);
}

/* double func2d(double x, double y, void * args) */
/* { */
/*     double out = 0.0; */
/*     if (args == NULL) */
/*         out = 5.0*pow(x,2) + 1.5*x*y;  */
/*         //out = x+5.0*y; */
/*     return out; */
/* } */

/* void Test_cross_approx_2d(CuTest * tc){ */
/*     printf("Testing function: cross_approx_2d\n"); */
    
/*     struct BoundingBox * bounds = bounding_box_init_std(2);  */
/*     double pivy[2] = {-0.25, 0.3}; */
/*     double pivx[2] = {-0.45, 0.4}; */

/*     enum poly_type p = LEGENDRE; */
/*     struct Cross2dargs * cargs = cross2d_args_create(2,1e-4,POLYNOMIAL,&p,0); */
    
/*     struct SkeletonDecomp * skd = skeleton_decomp_init2d_from_pivots( */
/*         func2d,NULL,bounds,cargs,pivx, pivy); */
    
/*     size_t ii, jj; */
/*     double det_before = 0.0; */
/*     det_before += (func2d(pivx[0],pivy[0],NULL) * func2d(pivx[1],pivy[1],NULL)); */
/*     det_before -= (func2d(pivx[1],pivy[0],NULL) * func2d(pivx[0],pivy[1],NULL)) ; */

/*     struct SkeletonDecomp * final = cross_approx_2d(func2d,NULL,bounds, */
/*                         &skd,pivx,pivy,cargs); */
    
/*     double det_after = 0.0; */
/*     det_after += (func2d(pivx[0],pivy[0],NULL) * func2d(pivx[1],pivy[1],NULL)); */
/*     det_after -= (func2d(pivx[1],pivy[0],NULL) * func2d(pivx[0],pivy[1],NULL)) ; */
    
/*     // check that |determinant| increased */
/*     /\* */
/*     printf("det_after =%3.2f\n", det_after); */
/*     printf("det_before =%3.2f\n", det_before); */
/*     printf("pivots x,y = \n"); */
/*     dprint(2,pivx); */
/*     dprint(2,pivy); */
/*     *\/ */

/*     CuAssertIntEquals(tc, 1, fabs(det_after) > fabs(det_before)); */

/*     size_t N1 = 100; */
/*     size_t N2 = 200; */
/*     double * xtest = linspace(-1.0, 1.0, N1); */
/*     double * ytest = linspace(-1.0, 1.0, N2); */

/*     double out, den, val; */
/*     out = 0.0; */
/*     den = 0.0; */
/*     for (ii = 0; ii < N1; ii++){ */
/*         for (jj = 0; jj < N2; jj++){ */
/*             val = func2d(xtest[ii],ytest[jj],NULL); */
/*             out += pow(skeleton_decomp_eval(final, xtest[ii],ytest[jj])-val,2.0); */
/*             den += pow(val,2.0); */
/*         } */
/*     } */

/*     CuAssertDblEquals(tc, 0.0, out/den, 1e-10); */
/*     free(xtest); */
/*     free(ytest); */
    
/*     skeleton_decomp_free(skd); */
/*     skeleton_decomp_free(final); */
/*     bounding_box_free(bounds); */
/*     cross2d_args_destroy(cargs); */
/* } */


void Test_quasimatrix_serialize(CuTest * tc){

    printf("Testing function: (de)quasimatrix_serialize\n");

    struct Fwrap * fw = fwrap_create(1,"array-vec");
    fwrap_set_num_funcs(fw,3);
    fwrap_set_func_array(fw,0,func, NULL);
    fwrap_set_func_array(fw,1,func2,NULL);
    fwrap_set_func_array(fw,2,func3,NULL);
    
    enum function_class fc[3] = {POLYNOMIAL,
                                 POLYNOMIAL,
                                 POLYNOMIAL};
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.0);
    ope_opts_set_ub(opts,1.0);
    
    struct Quasimatrix * A = quasimatrix_approx1d(3,fw,fc,opts);

    unsigned char * text = NULL;
    size_t ts;
    quasimatrix_serialize(NULL,A,&ts);
    text = malloc(ts * sizeof(unsigned char));
    quasimatrix_serialize(text,A,NULL);


    struct Quasimatrix * B = NULL;
    quasimatrix_deserialize(text, &B);

    size_t bsize = quasimatrix_get_size(B);
    CuAssertIntEquals(tc,3,bsize);

    for (size_t ii = 0; ii < 3; ii++){
        /* printf("ii = %zu\n",ii); */
        struct GenericFunction * gf = quasimatrix_get_func(A,ii);
        double integral = generic_function_integrate(gf);
        /* printf("\t Integral A = %G\n ",integral); */
        struct GenericFunction * g2 = quasimatrix_get_func(B,ii);
        double integral2 = generic_function_integrate(g2);
        /* printf("\t Integral B= %G\n",integral,integral2); */
        CuAssertDblEquals(tc,integral,integral2,1e-13);
    }

    quasimatrix_funcs_equal(tc,3,A,B,1e-13);
    free(text);
    quasimatrix_free(A);
    quasimatrix_free(B);
    fwrap_destroy(fw);
    ope_opts_free(opts);
}
    
CuSuite * QuasimatrixGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_quasimatrix_rank);
    SUITE_ADD_TEST(suite, Test_quasimatrix_rank2);
    SUITE_ADD_TEST(suite, Test_quasimatrix_householder);
    SUITE_ADD_TEST(suite, Test_quasimatrix_householder_weird_domain);
    SUITE_ADD_TEST(suite, Test_quasimatrix_lu1d);
    SUITE_ADD_TEST(suite, Test_quasimatrix_maxvol1d);
    SUITE_ADD_TEST(suite, Test_quasimatrix_serialize);
    /* SUITE_ADD_TEST(suite, Test_cross_approx_2d); */
    return suite;
}
