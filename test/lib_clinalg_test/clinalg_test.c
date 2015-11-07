// Copyright (c) 2014-2015, Massachusetts Institute of Technology
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

#include "array.h"
#include "CuTest.h"

#include "lib_funcs.h"
#include "lib_clinalg.h"
#include "lib_linalg.h"

typedef struct orth_poly_expansion opoly_t;

struct counter{
    int N;
};

double func(double x, void * args){
    struct counter * c = args;
    c->N = c->N+1;
    return 1.0 + 0.0*x;
}

double func2(double x, void * args){
    struct counter * c = args;
    c->N = c->N+1;
    return x;
}

double func3(double x, void * args){
    struct counter * c = args;
    c->N = c->N+1;
    return pow(x,2.0) + sin(M_PI*x);
}

double func4(double x, void * args){
    struct counter * c = args;
    c->N = c->N+1;
    return 3.0*pow(x,4.0) - 2.0*pow(x,2.0);
}

double func5(double x, void * args){
    struct counter * c = args;
    c->N = c->N+1;
    //return 3.0*cos(M_PI*x) - 2.0*pow(x,0.5);
    return x;
}

double func6(double x, void * args){
    struct counter * c = args;
    c->N = c->N+1;
    return exp(5.0*x);
}


void Test_quasimatrix_rank(CuTest * tc){

    printf("Testing function: quasimatrix_rank (1/2) \n");
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    void * args[3] = {&c, &c2, &c3};

    enum poly_type p = LEGENDRE;
    double (*funcs [3])(double, void *) = {&func, &func2, &func3};
    struct Quasimatrix * A = quasimatrix_approx1d(
                        3, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    size_t rank = quasimatrix_rank(A);
    CuAssertIntEquals(tc, 3, rank);
    quasimatrix_free(A);

    double (*funcs2 [3])(double, void *) = {&func, &func, &func};
    struct Quasimatrix * B = quasimatrix_approx1d(
                        3, funcs2, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    size_t rank2 = quasimatrix_rank(B);
    CuAssertIntEquals(tc, 1, rank2);
    quasimatrix_free(B);

}

void Test_quasimatrix_rank2(CuTest * tc){

    printf("Testing function: quasimatrix_rank (2/2) \n");
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    void * args[3] = {&c, &c2, &c3};

    enum poly_type p = LEGENDRE;
    double (*funcs [3])(double, void *) = {&func, &func2, &func3};
    struct Quasimatrix * A = quasimatrix_approx1d(
                        3, funcs, args, PIECEWISE, &p, -1.0, 1.0, NULL);


    struct Quasimatrix * Atemp = quasimatrix_approx1d(
                        3, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    
    /*
    printf("PIECEWISEL %d \n",A->funcs[0]->fc);
    print_generic_function(A->funcs[0],2,NULL);
    printf("POLYNOMIAL %d \n",Atemp->funcs[0]->fc);
    print_generic_function(Atemp->funcs[0],2,NULL);
    */
    double diff = generic_function_norm2diff(A->funcs[0],Atemp->funcs[0]);
    CuAssertDblEquals(tc,0.0,diff,1e-13);

    diff = generic_function_norm2diff(A->funcs[1],Atemp->funcs[1]);
    CuAssertDblEquals(tc,0.0,diff,1e-13);

    diff = generic_function_norm2diff(A->funcs[2],Atemp->funcs[2]);
    CuAssertDblEquals(tc,0.0,diff,1e-11);

    //print_quasimatrix(A,0,NULL);
    size_t rank = quasimatrix_rank(A);
    CuAssertIntEquals(tc, 3, rank);
    quasimatrix_free(A);
    quasimatrix_free(Atemp);

    double (*funcs2 [3])(double, void *) = {&func, &func, &func};
    struct Quasimatrix * B = quasimatrix_approx1d(
                        3, funcs2, args, PIECEWISE, &p, -1.0, 1.0, NULL);

    struct Quasimatrix * Btemp = quasimatrix_approx1d(
                        3, funcs2, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    
    diff = generic_function_norm2diff(B->funcs[0],Btemp->funcs[0]);
    CuAssertDblEquals(tc,0.0,diff,1e-13);

    diff = generic_function_norm2diff(B->funcs[1],Btemp->funcs[1]);
    CuAssertDblEquals(tc,0.0,diff,1e-13);

    diff = generic_function_norm2diff(B->funcs[2],Btemp->funcs[2]);
    CuAssertDblEquals(tc,0.0,diff,1e-13);

    struct Quasimatrix * Bdiff = quasimatrix_daxpby(1.0,B,-1.0,Btemp);
    diff = quasimatrix_norm(Bdiff);
    CuAssertDblEquals(tc,0.0,diff,1e-13);

    size_t rank2 = quasimatrix_rank(B);
    CuAssertIntEquals(tc, 1, rank2);
    quasimatrix_free(B);
    quasimatrix_free(Btemp);
    quasimatrix_free(Bdiff);

}

void Test_quasimatrix_householder(CuTest * tc){

    printf("Testing function: quasimatrix_householder\n");

    double (*funcs [3])(double, void *) = {&func, &func2, &func3};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    void * args[3] = {&c, &c2, &c3};


    enum poly_type p = LEGENDRE;
    struct Quasimatrix * A = quasimatrix_approx1d(
                        3, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    

    struct Quasimatrix * Acopy = quasimatrix_copy(A);
    double * R = calloc_double(3*3);
    struct Quasimatrix * Q = quasimatrix_householder_simple(A,R);
    
    struct Quasimatrix * Anew = qmm(Q,R,3);
    struct GenericFunction * temp;
    double diff;

    temp = generic_function_daxpby(1.0, Acopy->funcs[0],-1.0,Anew->funcs[0]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-13);
    generic_function_free(temp);

    temp = generic_function_daxpby(1.0, Acopy->funcs[1],-1.0,Anew->funcs[1]);
    diff = generic_function_norm(temp);
    generic_function_free(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-13);

    temp = generic_function_daxpby(1.0, Acopy->funcs[2],-1.0,Anew->funcs[2]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-13);
    generic_function_free(temp);

    quasimatrix_free(A);
    quasimatrix_free(Q);
    quasimatrix_free(Anew);
    quasimatrix_free(Acopy);
    free(R);
}

void Test_quasimatrix_householder_weird_domain(CuTest * tc){

    printf("Testing function: quasimatrix_householder on (-3.0, 2.0)\n");

    double lb = -3.0;
    double ub = 2.0;
    double (*funcs [3])(double, void *) = {&func, &func2, &func3};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    void * args[3] = {&c, &c2, &c3};


    enum poly_type p = LEGENDRE;
    struct Quasimatrix * A = quasimatrix_approx1d(
                        3, funcs, args, PIECEWISE, &p, lb, ub, NULL);
    

    struct Quasimatrix * Acopy = quasimatrix_copy(A);
    double * R = calloc_double(3*3);
    struct Quasimatrix * Q = quasimatrix_householder_simple(A,R);
    
    struct Quasimatrix * Anew = qmm(Q,R,3);
    struct GenericFunction * temp;
    double diff;

    temp = generic_function_daxpby(1.0, Acopy->funcs[0],-1.0,Anew->funcs[0]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-13);
    generic_function_free(temp);

    temp = generic_function_daxpby(1.0, Acopy->funcs[1],-1.0,Anew->funcs[1]);
    diff = generic_function_norm(temp);
    generic_function_free(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-12);

    temp = generic_function_daxpby(1.0, Acopy->funcs[2],-1.0,Anew->funcs[2]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-11);
    generic_function_free(temp);

    quasimatrix_free(A);
    quasimatrix_free(Q);
    quasimatrix_free(Anew);
    quasimatrix_free(Acopy);
    free(R);
}

void Test_quasimatrix_lu1d(CuTest * tc){

    printf("Testing function: quasimatrix_lu1d\n");

    double (*funcs [3])(double, void *) = {&func, &func2, &func3};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    void * args[3] = {&c, &c2, &c3};

    enum poly_type p = LEGENDRE;
    struct Quasimatrix * A = quasimatrix_approx1d(
                        3, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    
    struct Quasimatrix * L = quasimatrix_alloc(3);

    double * eye = calloc_double(9);
    size_t ii;
    for (ii = 0; ii < 3; ii++)  eye[ii * 3 + ii] = 1.0;

    struct Quasimatrix * Acopy = qmm(A,eye,3);
    double * U = calloc_double(3*3);
    double * piv = calloc_double(3);
    quasimatrix_lu1d(A,L,U,piv);
     
    double eval;
    eval = generic_function_1d_eval(L->funcs[1], piv[0]);
    CuAssertDblEquals(tc, 0.0, eval, 1e-13);
    eval = generic_function_1d_eval(L->funcs[2], piv[0]);
    CuAssertDblEquals(tc, 0.0, eval, 1e-13);
    eval = generic_function_1d_eval(L->funcs[2], piv[1]);
    CuAssertDblEquals(tc, 0.0, eval, 1e-13);

    //printf("pivots at = \n");
    //dprint(3,piv);
    //dprint2d_col(3,3,U);
    struct Quasimatrix * Anew = qmm(L,U,3);
    struct GenericFunction * temp;
    double diff;

    temp = generic_function_daxpby(1.0, Acopy->funcs[0],-1.0,Anew->funcs[0]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-15);
    generic_function_free(temp);

    temp = generic_function_daxpby(1.0, Acopy->funcs[1],-1.0,Anew->funcs[1]);
    diff = generic_function_norm(temp);
    generic_function_free(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-13);

    temp = generic_function_daxpby(1.0, Acopy->funcs[2],-1.0,Anew->funcs[2]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-15);
    generic_function_free(temp);

    quasimatrix_free(A);
    quasimatrix_free(L);
    quasimatrix_free(Anew);
    quasimatrix_free(Acopy);
    free(eye);
    free(U);
    free(piv);
}

void Test_quasimatrix_maxvol1d(CuTest * tc){

    printf("Testing function: quasimatrix_maxvol1d\n");

    double (*funcs [3])(double, void *) = {&func, &func2, &func3};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    void * args[3] = {&c, &c2, &c3};

    enum poly_type p = LEGENDRE;
    struct Quasimatrix * A = quasimatrix_approx1d(
                        3, funcs, args, PIECEWISE, &p, -1.0, 1.0, NULL);
    
    printf("got approximation\n");
    double * Asinv = calloc_double(3*3);
    double * piv = calloc_double(3);
    quasimatrix_maxvol1d(A,Asinv,piv);
     
    printf("pivots at = \n");
    dprint(3,piv);

    struct Quasimatrix * B = qmm(A,Asinv,3);
    double maxval, maxloc;
    quasimatrix_absmax(B,&maxloc,&maxval);
    //printf("Less = %d", 1.0+1e-2 > maxval);
    CuAssertIntEquals(tc, 1, (1.0+1e-2) > maxval);

    quasimatrix_free(B);
    quasimatrix_free(A);
    free(Asinv);
    free(piv);
}

double func2d(double x, double y, void * args)
{
    double out = 0.0;
    if (args == NULL)
        out = 5.0*pow(x,2) + 1.5*x*y; 
        //out = x+5.0*y;
    return out;
}

void Test_cross_approx_2d(CuTest * tc){
    printf("Testing function: cross_approx_2d\n");
    
    struct BoundingBox * bounds = bounding_box_init_std(2); 
    double pivy[2] = {-0.25, 0.3};
    double pivx[2] = {-0.45, 0.4};
    
    struct Cross2dargs cargs;
    enum poly_type p = LEGENDRE;
    cargs.r = 2;
    cargs.delta = 1e-4;
    cargs.fclass[0] = POLYNOMIAL;
    cargs.fclass[1] = POLYNOMIAL;
    cargs.sub_type[0] = &p;
    cargs.sub_type[1] = &p;
    cargs.approx_args[0] = NULL;
    cargs.approx_args[1] = NULL;
    cargs.verbose = 0;
    
    struct SkeletonDecomp * skd = skeleton_decomp_init2d_from_pivots(
                    func2d, NULL, bounds, cargs.fclass, cargs.sub_type,
                    cargs.r, pivx, pivy, cargs.approx_args);
    
    size_t ii, jj;
    double det_before = 0.0;
    det_before += (func2d(pivx[0],pivy[0],NULL) * func2d(pivx[1],pivy[1],NULL));
    det_before -= (func2d(pivx[1],pivy[0],NULL) * func2d(pivx[0],pivy[1],NULL)) ;

    struct SkeletonDecomp * final = cross_approx_2d(func2d,NULL,bounds,
                        &skd,pivx,pivy,&cargs);
    
    double det_after = 0.0;
    det_after += (func2d(pivx[0],pivy[0],NULL) * func2d(pivx[1],pivy[1],NULL));
    det_after -= (func2d(pivx[1],pivy[0],NULL) * func2d(pivx[0],pivy[1],NULL)) ;
    
    // check that |determinant| increased
    /*
    printf("det_after =%3.2f\n", det_after);
    printf("det_before =%3.2f\n", det_before);
    printf("pivots x,y = \n");
    dprint(2,pivx);
    dprint(2,pivy);
    */

    CuAssertIntEquals(tc, 1, fabs(det_after) > fabs(det_before));

    size_t N1 = 100;
    size_t N2 = 200;
    double * xtest = linspace(-1.0, 1.0, N1);
    double * ytest = linspace(-1.0, 1.0, N2);

    double out, den, val;
    out = 0.0;
    den = 0.0;
    for (ii = 0; ii < N1; ii++){
        for (jj = 0; jj < N2; jj++){
            val = func2d(xtest[ii],ytest[jj],NULL);
            out += pow(skeleton_decomp_eval(final, xtest[ii],ytest[jj])-val,2.0);
            den += pow(val,2.0);
        }
    }

    CuAssertDblEquals(tc, 0.0, out/den, 1e-10);
    free(xtest);
    free(ytest);
    
    skeleton_decomp_free(skd);
    skeleton_decomp_free(final);
    bounding_box_free(bounds);
}


void Test_quasimatrix_serialize(CuTest * tc){

    printf("Testing function: (de)quasimatrix_serialize\n");
    double (*funcs [3])(double, void *) = {&func, &func2, &func3};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    void * args[3] = {&c, &c2, &c3};

    enum poly_type p = LEGENDRE;
    struct Quasimatrix * A = quasimatrix_approx1d(
                        3, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    
    unsigned char * text = NULL;
    size_t ts;
    quasimatrix_serialize(NULL,A,&ts);
    text = malloc(ts * sizeof(unsigned char));
    quasimatrix_serialize(text,A,NULL);

    //printf("text=\n%s\n",text);

    struct Quasimatrix * B = NULL;
    quasimatrix_deserialize(text, &B);

    CuAssertIntEquals(tc,3,B->n);

    struct GenericFunction * temp;
    double diff;

    temp = generic_function_daxpby(1.0, A->funcs[0],-1.0,B->funcs[0]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-15);
    generic_function_free(temp);

    temp = generic_function_daxpby(1.0, A->funcs[1],-1.0,B->funcs[1]);
    diff = generic_function_norm(temp);
    generic_function_free(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-13);

    temp = generic_function_daxpby(1.0, A->funcs[2],-1.0,B->funcs[2]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-13);
    generic_function_free(temp);

    free(text);
    quasimatrix_free(A);
    quasimatrix_free(B);
}
    
CuSuite * CLinalgGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_quasimatrix_rank);
    SUITE_ADD_TEST(suite, Test_quasimatrix_rank2);
    SUITE_ADD_TEST(suite, Test_quasimatrix_householder);
    SUITE_ADD_TEST(suite, Test_quasimatrix_householder_weird_domain);
    SUITE_ADD_TEST(suite, Test_quasimatrix_lu1d);
    SUITE_ADD_TEST(suite, Test_quasimatrix_maxvol1d);
    SUITE_ADD_TEST(suite, Test_cross_approx_2d);
    SUITE_ADD_TEST(suite, Test_quasimatrix_serialize);
    return suite;
}

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

void Test_qmarray_householder(CuTest * tc){

    printf("Testing function: qmarray_householder (1/4)\n");

    double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};


    enum poly_type p = LEGENDRE;
    struct Qmarray * A = qmarray_approx1d(
                        2, 2, funcs, args, PIECEWISE, &p, -1.0, 1.0, NULL);
    

    struct Qmarray * Acopy = qmarray_copy(A);

    struct Quasimatrix * a1 = qmarray_extract_column(A,0);
    struct Quasimatrix * a2 = qmarray_extract_column(A,1);

    struct Quasimatrix * a3 = qmarray_extract_column(Acopy,0);
    struct Quasimatrix * a4 = qmarray_extract_column(Acopy,1);
    
    struct Quasimatrix * temp = NULL;
    double diff;
    temp = quasimatrix_daxpby(1.0, a1,-1.0, a3);
    diff = quasimatrix_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-15);
    quasimatrix_free(temp); temp = NULL;

    temp = quasimatrix_daxpby(1.0, a2, -1.0, a4);
    diff = quasimatrix_norm(temp);
    quasimatrix_free(temp); temp = NULL;
    CuAssertDblEquals(tc, 0.0, diff, 1e-15);

    quasimatrix_free(a1);
    quasimatrix_free(a2);
    quasimatrix_free(a3);
    quasimatrix_free(a4);

    
    double * R = calloc_double(2*2);

    struct Qmarray * Q = qmarray_householder_simple("QR", A,R);

    CuAssertIntEquals(tc,2,Q->nrows);
    CuAssertIntEquals(tc,2,Q->ncols);

    // test orthogonality
    struct Quasimatrix * q1a = qmarray_extract_column(Q,0);
    struct Quasimatrix * q2a = qmarray_extract_column(Q,1);
    double test1 = quasimatrix_inner(q1a,q1a);
    CuAssertDblEquals(tc,1.0,test1,1e-13);
    double test2 = quasimatrix_inner(q2a,q2a);
    CuAssertDblEquals(tc,1.0,test2,1e-13);
    double test3 = quasimatrix_inner(q1a,q2a);
    CuAssertDblEquals(tc,0.0,test3,1e-13);

    quasimatrix_free(q1a); q1a = NULL;
    quasimatrix_free(q2a); q2a = NULL;

    //dprint2d_col(2,2,R);
    
     // testt equivalence
    struct Qmarray * Anew = qmam(Q,R,2);
    
    struct Quasimatrix * q1 = qmarray_extract_column(Anew,0);
    struct Quasimatrix * q2 = qmarray_extract_column(Anew,1);

    struct Quasimatrix * q3 = qmarray_extract_column(Acopy,0);
    struct Quasimatrix * q4 = qmarray_extract_column(Acopy,1);
    
    temp = quasimatrix_daxpby(1.0, q1,-1.0, q3);
    diff = quasimatrix_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-13);
    quasimatrix_free(temp);

    temp = quasimatrix_daxpby(1.0, q2, -1.0, q4);
    diff = quasimatrix_norm(temp);
    quasimatrix_free(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-13);

    quasimatrix_free(q1);
    quasimatrix_free(q2);
    quasimatrix_free(q3);
    quasimatrix_free(q4);
    qmarray_free(Anew);

    qmarray_free(A);
    qmarray_free(Q);
    qmarray_free(Acopy);
    free(R);
}

void Test_qmarray_householder2(CuTest * tc){

    printf("Testing function: qmarray_householder (2/4)\n");

    double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};


    enum poly_type p = LEGENDRE;
    struct Qmarray * A = qmarray_approx1d(
                        1, 4, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    double * R = calloc_double(4*4);
    struct Qmarray * Q = qmarray_householder_simple("QR", A,R);


    struct Quasimatrix * A2 = quasimatrix_approx1d(
                        4, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    double * R2 = calloc_double(4*4);
    struct Quasimatrix * Q2 = quasimatrix_householder_simple(A2,R2);
    
 
    /*
    printf("=========================\n");
    printf("R=\n");
    dprint2d_col(4,4,R);

    printf("R2=\n");
    dprint2d_col(4,4,R2);
    */

    CuAssertDblEquals(tc, 0.0, norm2diff(R,R2,16),1e-14);

    struct GenericFunction * temp = NULL;
    double diff;

    temp = generic_function_daxpby(1,Q->funcs[0],-1.0,Q2->funcs[0]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff ,1e-14);
    generic_function_free(temp);
    temp = generic_function_daxpby(1,Q->funcs[1],-1.0,Q2->funcs[1]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff ,1e-14);
    generic_function_free(temp);
    temp = generic_function_daxpby(1,Q->funcs[2],-1.0,Q2->funcs[2]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff ,1e-14);
    generic_function_free(temp);
    temp = generic_function_daxpby(1,Q->funcs[3],-1.0,Q2->funcs[3]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff ,1e-14);
    generic_function_free(temp);


    qmarray_free(A);
    qmarray_free(Q);
    quasimatrix_free(A2);
    quasimatrix_free(Q2);
    free(R);
    free(R2);
}

void Test_qmarray_householder3(CuTest * tc){

    printf("Testing function: qmarray_householder (3/4)\n");

    double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func3};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};

    enum poly_type p = LEGENDRE;
    struct Qmarray * A = qmarray_approx1d(
                        1, 4, funcs, args, PIECEWISE, &p, -1.0, 1.0, NULL);
    size_t N = 100;
    double * xtest = linspace(-1.0, 1.0, N);
    double temp1, temp2, err;
    size_t ii,jj;
    for (ii = 0; ii < 4; ii++){
        err = 0.0;
        for (jj = 0; jj < N; jj++){
            temp1 = funcs[ii](xtest[jj],&c);
            temp2 = generic_function_1d_eval(A->funcs[ii],xtest[jj]);
            err += fabs(temp1-temp2);
        }
        printf("err= %3.15G\n",err);
        CuAssertDblEquals(tc,0.0,err,1e-6);
    }

    double * R = calloc_double(4*4);
    struct Qmarray * Q = qmarray_householder_simple("QR", A,R);

    struct Quasimatrix * A2 = quasimatrix_approx1d(
                        4, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    for (ii = 0; ii < 4; ii++){
        err = 0.0;
        for (jj = 0; jj < N; jj++){
            temp1 = funcs[ii](xtest[jj],&c);
            temp2 = generic_function_1d_eval(A2->funcs[ii],xtest[jj]);
            err += fabs(temp1-temp2);
        }
        //printf("err= %3.15G\n",err);
        CuAssertDblEquals(tc,0.0,err,1e-11);
    }
    free(xtest); xtest = NULL;

    double * R2 = calloc_double(4*4);
    struct Quasimatrix * Q2 = quasimatrix_householder_simple(A2,R2);


    double inner;
    inner = generic_function_inner(Q->funcs[0],Q->funcs[0]);
    CuAssertDblEquals(tc,1.0,inner,1e-13);
    inner = generic_function_inner(Q->funcs[0],Q->funcs[1]);
    CuAssertDblEquals(tc,0.0,inner,1e-13);
    inner = generic_function_inner(Q->funcs[0],Q->funcs[2]);
    CuAssertDblEquals(tc,0.0,inner,1e-13);
    inner = generic_function_inner(Q->funcs[0],Q->funcs[3]);
    CuAssertDblEquals(tc,0.0,inner,1e-12);

    inner = generic_function_inner(Q->funcs[1],Q->funcs[1]);
    CuAssertDblEquals(tc,1.0,inner,1e-13);
    inner = generic_function_inner(Q->funcs[1],Q->funcs[2]);
    CuAssertDblEquals(tc,0.0,inner,1e-13);
    inner = generic_function_inner(Q->funcs[1],Q->funcs[3]);
    CuAssertDblEquals(tc,0.0,inner,1e-13);

    inner = generic_function_inner(Q->funcs[2],Q->funcs[2]);
    CuAssertDblEquals(tc,1.0,inner,1e-13);
    inner = generic_function_inner(Q->funcs[2],Q->funcs[3]);
    CuAssertDblEquals(tc,0.0,inner,1e-13);

    inner = generic_function_inner(Q->funcs[3],Q->funcs[3]);
    CuAssertDblEquals(tc,1.0,inner,1e-11);

    inner = generic_function_inner(Q2->funcs[0],Q2->funcs[0]);
    CuAssertDblEquals(tc,1.0,inner,1e-13);
    inner = generic_function_inner(Q2->funcs[0],Q2->funcs[1]);
    CuAssertDblEquals(tc,0.0,inner,1e-13);
    inner = generic_function_inner(Q2->funcs[0],Q2->funcs[2]);
    CuAssertDblEquals(tc,0.0,inner,1e-13);
    inner = generic_function_inner(Q2->funcs[0],Q2->funcs[3]);
    CuAssertDblEquals(tc,0.0,inner,1e-12);

    inner = generic_function_inner(Q2->funcs[1],Q2->funcs[1]);
    CuAssertDblEquals(tc,1.0,inner,1e-13);
    inner = generic_function_inner(Q2->funcs[1],Q2->funcs[2]);
    CuAssertDblEquals(tc,0.0,inner,1e-13);
    inner = generic_function_inner(Q2->funcs[1],Q2->funcs[3]);
    CuAssertDblEquals(tc,0.0,inner,1e-13);

    inner = generic_function_inner(Q2->funcs[2],Q2->funcs[2]);
    CuAssertDblEquals(tc,1.0,inner,1e-13);
    inner = generic_function_inner(Q2->funcs[2],Q2->funcs[3]);
    CuAssertDblEquals(tc,0.0,inner,1e-13);

    inner = generic_function_inner(Q2->funcs[3],Q2->funcs[3]);
    CuAssertDblEquals(tc,1.0,inner,1e-13);

    /*
    printf("=========================\n");
    printf("R=\n");
    dprint2d_col(4,4,R);

    printf("R2=\n");
    dprint2d_col(4,4,R2);
    */

    CuAssertDblEquals(tc, 0.0, norm2diff(R,R2,16),1e-13);

    struct GenericFunction * temp = NULL;
    double diff;

    temp = generic_function_daxpby(1,Q->funcs[0],-1.0,Q2->funcs[0]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff ,1e-13);
    generic_function_free(temp);
    temp = generic_function_daxpby(1,Q->funcs[1],-1.0,Q2->funcs[1]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff ,1e-13);
    generic_function_free(temp);
    temp = generic_function_daxpby(1,Q->funcs[2],-1.0,Q2->funcs[2]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff ,1e-13);
    generic_function_free(temp);

    //temp = generic_function_daxpby(1,Q->funcs[3],-1.0,Q2->funcs[3]);
   // diff = generic_function_norm(temp);
    //CuAssertDblEquals(tc, 0.0, diff ,1e-14);
   // generic_function_free(temp);


    qmarray_free(A);
    qmarray_free(Q);
    quasimatrix_free(A2);
    quasimatrix_free(Q2);
    free(R);
    free(R2);
}

void Test_qmarray_householder4(CuTest * tc){

    printf("Testing function: qmarray_householder (4/4)\n");

    double (*funcs [4])(double, void *) = {&func, &func3, &func3, &func3};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};


    enum poly_type p = LEGENDRE;
    struct Qmarray * A = qmarray_approx1d(
                        1, 4, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    double * R = calloc_double(4*4);
    struct Qmarray * Q = qmarray_householder_simple("QR", A,R);


    struct Quasimatrix * A2 = quasimatrix_approx1d(
                        4, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    double * R2 = calloc_double(4*4);
    struct Quasimatrix * Q2 = quasimatrix_householder_simple(A2,R2);
    
 
    /*
    printf("=========================\n");
    printf("R=\n");
    dprint2d_col(4,4,R);

    printf("R2=\n");
    dprint2d_col(4,4,R2);
    */

    CuAssertDblEquals(tc, 0.0, norm2diff(R,R2,16),1e-14);

    struct GenericFunction * temp = NULL;
    double diff;

    temp = generic_function_daxpby(1,Q->funcs[0],-1.0,Q2->funcs[0]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff ,1e-14);
    generic_function_free(temp);
    temp = generic_function_daxpby(1,Q->funcs[1],-1.0,Q2->funcs[1]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff ,1e-14);
    generic_function_free(temp);
    temp = generic_function_daxpby(1,Q->funcs[2],-1.0,Q2->funcs[2]);
    diff = generic_function_norm(temp);
    //print_generic_function(Q->funcs[2],0,NULL);
    //print_generic_function(Q2->funcs[2],0,NULL);
    CuAssertDblEquals(tc, 0.0, diff ,1e-14);
    generic_function_free(temp);
    temp = generic_function_daxpby(1,Q->funcs[3],-1.0,Q2->funcs[3]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff ,1e-14);
    generic_function_free(temp);


    qmarray_free(A);
    qmarray_free(Q);
    quasimatrix_free(A2);
    quasimatrix_free(Q2);
    free(R);
    free(R2);
}


void Test_qmarray_householder_rows(CuTest * tc){

    printf("Testing function: qmarray_householder_rows \n");

    double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};


    enum poly_type p = LEGENDRE;
    struct Qmarray * A = qmarray_approx1d(
                        2, 2, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    

    struct Qmarray * Acopy = qmarray_copy(A);

    struct Quasimatrix * a1 = qmarray_extract_row(A,0);
    struct Quasimatrix * a2 = qmarray_extract_row(A,1);

    struct Quasimatrix * a3 = qmarray_extract_row(Acopy,0);
    struct Quasimatrix * a4 = qmarray_extract_row(Acopy,1);
    
    struct Quasimatrix * temp = NULL;
    double diff;
    temp = quasimatrix_daxpby(1.0, a1,-1.0, a3);
    diff = quasimatrix_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-15);
    quasimatrix_free(temp); temp = NULL;

    temp = quasimatrix_daxpby(1.0, a2, -1.0, a4);
    diff = quasimatrix_norm(temp);
    quasimatrix_free(temp); temp = NULL;
    CuAssertDblEquals(tc, 0.0, diff, 1e-15);

    quasimatrix_free(a1);
    quasimatrix_free(a2);
    quasimatrix_free(a3);
    quasimatrix_free(a4);

    
    double * R = calloc_double(2*2);

    struct Qmarray * Q = qmarray_householder_simple("LQ", A,R);

    CuAssertIntEquals(tc,2,Q->nrows);
    CuAssertIntEquals(tc,2,Q->ncols);

    // test orthogonality
    struct Quasimatrix * q1a = qmarray_extract_row(Q,0);
    struct Quasimatrix * q2a = qmarray_extract_row(Q,1);
    double test1 = quasimatrix_inner(q1a,q1a);
    CuAssertDblEquals(tc,1.0,test1,1e-14);
    double test2 = quasimatrix_inner(q2a,q2a);
    CuAssertDblEquals(tc,1.0,test2,1e-14);
    double test3 = quasimatrix_inner(q1a,q2a);
    CuAssertDblEquals(tc,0.0,test3,1e-14);

    quasimatrix_free(q1a); q1a = NULL;
    quasimatrix_free(q2a); q2a = NULL;

    //dprint2d_col(2,2,R);
    
     // testt equivalence
    struct Qmarray * Anew = mqma(R,Q,2);
    
    struct Quasimatrix * q1 = qmarray_extract_row(Anew,0);
    struct Quasimatrix * q2 = qmarray_extract_row(Anew,1);

    struct Quasimatrix * q3 = qmarray_extract_row(Acopy,0);
    struct Quasimatrix * q4 = qmarray_extract_row(Acopy,1);
    
    temp = quasimatrix_daxpby(1.0, q1,-1.0, q3);
    diff = quasimatrix_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-12);
    quasimatrix_free(temp);

    temp = quasimatrix_daxpby(1.0, q2, -1.0, q4);
    diff = quasimatrix_norm(temp);
    quasimatrix_free(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-12);

    quasimatrix_free(q1);
    quasimatrix_free(q2);
    quasimatrix_free(q3);
    quasimatrix_free(q4);
    qmarray_free(Anew);

    qmarray_free(A);
    qmarray_free(Q);
    qmarray_free(Acopy);
    free(R);
}

void Test_qmarray_lu1d(CuTest * tc){

    printf("Testing function: qmarray_lu1d (1/2)\n");
    double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};


    enum poly_type p = LEGENDRE;
    struct Qmarray * A = qmarray_approx1d(
                        2, 2, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    

    struct Qmarray * L = qmarray_alloc(2,2);

    struct Qmarray * Acopy = qmarray_copy(A);

    double * U = calloc_double(2*2);
    size_t * pivi = calloc_size_t(2);
    double * pivx = calloc_double(2);
    qmarray_lu1d(A,L,U,pivi,pivx);
    
    double eval;
    
    //print_qmarray(A,0,NULL);
    // check pivots
    //printf("U = \n");
    //dprint2d_col(2,2,U);
    eval = generic_function_1d_eval(L->funcs[2+ pivi[0]], pivx[0]);
    //printf("eval = %G\n",eval);
    CuAssertDblEquals(tc, 0.0, eval, 1e-13);
    
    struct Qmarray * Comb = qmam(L,U,2);
    double difff = qmarray_norm2diff(Comb,Acopy);
    //printf("difff = %G\n",difff);
    CuAssertDblEquals(tc,difff,0,1e-14);
    
    //exit(1);
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
    //this is column ordered when convertest to Qmarray
    double (*funcs [6])(double, void *) = {&func,  &func4, &func, 
                                           &func4, &func5, &func6};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    struct counter c5; c5.N = 0;
    struct counter c6; c6.N = 0;
    void * args[6] = {&c, &c2, &c3, &c4, &c5, &c6};


    enum poly_type p = LEGENDRE;
    struct Qmarray * A = qmarray_approx1d(
                        2, 3, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    printf("A = (%zu,%zu)\n",A->nrows,A->ncols);

    struct Qmarray * L = qmarray_alloc(2,3);

    struct Qmarray * Acopy = qmarray_copy(A);

    double * U = calloc_double(3*3);
    size_t * pivi = calloc_size_t(3);
    double * pivx = calloc_double(3);
    qmarray_lu1d(A,L,U,pivi,pivx);
    
    double eval;
    
    //print_qmarray(A,0,NULL);
    // check pivots
    //printf("U = \n");
    //dprint2d_col(2,2,U);
    size_t ii,jj;
    for (ii = 0; ii < 3; ii++){
        //printf("Checking column %zu \n",ii);
        //printf("---------------\n");
        for (jj = 0; jj < ii; jj++){
            //printf("Should have zero at (%zu,%G)\n",pivi[jj],pivx[jj]);
            eval = generic_function_1d_eval(L->funcs[2*ii+pivi[jj]], pivx[jj]);
            CuAssertDblEquals(tc,0.0,eval,1e-14);
            //printf("eval = %G\n",eval);
        }
        //printf("Should have one at (%zu,%G)\n",pivi[ii],pivx[ii]);
        eval = generic_function_1d_eval(L->funcs[2*ii+pivi[ii]], pivx[ii]);
        CuAssertDblEquals(tc,1.0,eval,1e-14);
        //printf("eval = %G\n",eval);
    }
    /*
    eval = generic_function_1d_eval(L->funcs[2+ pivi[0]], pivx[0]);
    printf("eval = %G\n",eval);
    eval = generic_function_1d_eval(L->funcs[4+ pivi[1]], pivx[1]);
    printf("eval = %G\n",eval);
    eval = generic_function_1d_eval(L->funcs[4+ pivi[0]], pivx[0]);
    printf("eval = %G\n",eval);
    */

    //CuAssertDblEquals(tc, 0.0, eval, 1e-13);
    
    struct Qmarray * Comb = qmam(L,U,3);
    double difff = qmarray_norm2diff(Comb,Acopy);
    //printf("difff = %G\n",difff);
    CuAssertDblEquals(tc,difff,0,1e-13);
    
    //exit(1);
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

    double (*funcs [6])(double, void *) = {&func, &func2, &func3, &func4,
                                            &func5, &func6};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    struct counter c5; c5.N = 0;
    struct counter c6; c6.N = 0;
    void * args[6] = {&c, &c2, &c3, &c4, &c5, &c6};


    enum poly_type p = LEGENDRE;
    struct Qmarray * A = qmarray_approx1d(
                        3, 2, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    

    double * Asinv = calloc_double(2*2);
    size_t * pivi = calloc_size_t(2);
    double * pivx= calloc_double(2);

    qmarray_maxvol1d(A,Asinv,pivi,pivx);
     
    /*
    printf("pivots at = \n");
    iprint_sz(3,pivi); 
    dprint(3,pivx);
    */

    struct Qmarray * B = qmam(A,Asinv,2);
    double maxval, maxloc;
    size_t maxrow, maxcol;
    qmarray_absmax1d(B,&maxloc,&maxrow, &maxcol, &maxval);
    //printf("Less = %d", 1.0+1e-2 > maxval);
    CuAssertIntEquals(tc, 1, (1.0+1e-2) > maxval);
    qmarray_free(B);

    qmarray_free(A);
    free(Asinv);
    free(pivx);
    free(pivi);
}

void Test_qmarray_maxvol1d2(CuTest * tc){

    printf("Testing function: qmarray_maxvol1d (2/2) \n");

    double (*funcs [6])(double, void *) = {&func, &func2, &func3, &func4,
                                            &func4, &func4};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    struct counter c5; c5.N = 0;
    struct counter c6; c6.N = 0;
    void * args[6] = {&c, &c2, &c3, &c4, &c5, &c6};


    enum poly_type p = LEGENDRE;
    struct Qmarray * A = qmarray_approx1d(
                        1, 6, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    

    double * Asinv = calloc_double(6*6);
    size_t * pivi = calloc_size_t(6);
    double * pivx= calloc_double(6);

    qmarray_maxvol1d(A,Asinv,pivi,pivx);
     
    //*
    printf("pivots at = \n");
    iprint_sz(6,pivi); 
    dprint(6,pivx);
    //*/

    struct Qmarray * B = qmam(A,Asinv,2);
    double maxval, maxloc;
    size_t maxrow, maxcol;
    qmarray_absmax1d(B,&maxloc,&maxrow, &maxcol, &maxval);
    //printf("Less = %d", 1.0+1e-2 > maxval);
    CuAssertIntEquals(tc, 1, (1.0+1e-2) > maxval);
    qmarray_free(B);

    qmarray_free(A);
    free(Asinv);
    free(pivx);
    free(pivi);
}


void Test_qmarray_svd(CuTest * tc){

    printf("Testing function: qmarray_svd \n");

    double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};


    enum poly_type p = LEGENDRE;
    struct Qmarray * A = qmarray_approx1d(
                        2, 2, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    
    struct Qmarray * Acopy = qmarray_copy(A);

    double * vt = calloc_double(2*2);
    double * s = calloc_double(2);
    struct Qmarray * Q = NULL;

    qmarray_svd(A,&Q,s,vt);

    CuAssertIntEquals(tc,2,Q->nrows);
    CuAssertIntEquals(tc,2,Q->ncols);

    // test orthogonality

    struct Quasimatrix * q1a = qmarray_extract_column(Q,0);
    struct Quasimatrix * q2a = qmarray_extract_column(Q,1);
    double test1 = quasimatrix_inner(q1a,q1a);
    CuAssertDblEquals(tc,1.0,test1,1e-14);
    double test2 = quasimatrix_inner(q2a,q2a);
    CuAssertDblEquals(tc,1.0,test2,1e-14);
    double test3 = quasimatrix_inner(q1a,q2a);
    CuAssertDblEquals(tc,0.0,test3,1e-14);

    quasimatrix_free(q1a); q1a = NULL;
    quasimatrix_free(q2a); q2a = NULL;

    //dprint2d_col(2,2,R);
    
     // testt equivalence
    struct Quasimatrix * temp = NULL;
    double * comb = calloc_double(2*2);

    double * sdiag = diag(2, s);
    
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, 2, 2, 2, 1.0, 
                    sdiag, 2, vt, 2, 0.0, comb, 2);
    //comb = dgemm
    free(s);
    free(sdiag);
    free(vt);

    double diff;
    struct Qmarray * Anew = qmam(Q,comb,2);
    free(comb);
    
    struct Quasimatrix * q1 = qmarray_extract_column(Anew,0);
    struct Quasimatrix * q2 = qmarray_extract_column(Anew,1);

    struct Quasimatrix * q3 = qmarray_extract_column(Acopy,0);
    struct Quasimatrix * q4 = qmarray_extract_column(Acopy,1);
    
    temp = quasimatrix_daxpby(1.0, q1,-1.0, q3);
    diff = quasimatrix_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-12);
    quasimatrix_free(temp);

    temp = quasimatrix_daxpby(1.0, q2, -1.0, q4);
    diff = quasimatrix_norm(temp);
    quasimatrix_free(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-12);
    
    quasimatrix_free(q1);
    quasimatrix_free(q2);
    quasimatrix_free(q3);
    quasimatrix_free(q4);
    qmarray_free(Anew);

    qmarray_free(A);
    qmarray_free(Q);
    qmarray_free(Acopy);
}

void Test_qmarray_serialize(CuTest * tc){

    printf("Testing function: (de)qmarray_serialize\n");

    double (*funcs [6])(double, void *) = {&func, &func2, &func3, &func4,
                                            &func5, &func6};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    struct counter c5; c5.N = 0;
    struct counter c6; c6.N = 0;
    void * args[6] = {&c, &c2, &c3, &c4, &c5, &c6};


    enum poly_type p = LEGENDRE;
    struct Qmarray * A = qmarray_approx1d(
                        3, 2, funcs, args, POLYNOMIAL, &p, -1.0, 1.0, NULL);
    
    unsigned char * text = NULL;
    size_t size;
    qmarray_serialize(NULL,A,&size);
    text = malloc(size * sizeof(unsigned char));
    qmarray_serialize(text,A,NULL);
    

    struct Qmarray * B = NULL;
    qmarray_deserialize(text,&B);
    free(text); text = NULL;

    CuAssertIntEquals(tc,3,B->nrows);
    CuAssertIntEquals(tc,2,B->ncols);

    struct GenericFunction * temp;
    double diff;

    temp = generic_function_daxpby(1.0, A->funcs[0],-1.0,B->funcs[0]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-15);
    generic_function_free(temp);

    temp = generic_function_daxpby(1.0, A->funcs[1],-1.0,B->funcs[1]);
    diff = generic_function_norm(temp);
    generic_function_free(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-13);

    temp = generic_function_daxpby(1.0, A->funcs[2],-1.0,B->funcs[2]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-13);
    generic_function_free(temp);

    temp = generic_function_daxpby(1.0, A->funcs[3],-1.0,B->funcs[3]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-15);
    generic_function_free(temp);

    temp = generic_function_daxpby(1.0, A->funcs[4],-1.0,B->funcs[4]);
    diff = generic_function_norm(temp);
    generic_function_free(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-13);

    temp = generic_function_daxpby(1.0, A->funcs[5],-1.0,B->funcs[5]);
    diff = generic_function_norm(temp);
    CuAssertDblEquals(tc, 0.0, diff, 1e-13);
    generic_function_free(temp);

    qmarray_free(A);
    qmarray_free(B);
}


CuSuite * CLinalgQmarrayGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_qmarray_orth1d_columns);
    SUITE_ADD_TEST(suite, Test_qmarray_householder);
    
    SUITE_ADD_TEST(suite, Test_qmarray_householder2);
    SUITE_ADD_TEST(suite, Test_qmarray_householder3);
    SUITE_ADD_TEST(suite, Test_qmarray_householder4);
    SUITE_ADD_TEST(suite, Test_qmarray_householder_rows);
    //SUITE_ADD_TEST(suite, Test_qmarray_lu1d);
    SUITE_ADD_TEST(suite, Test_qmarray_lu1d2);
    SUITE_ADD_TEST(suite, Test_qmarray_maxvol1d);
    SUITE_ADD_TEST(suite, Test_qmarray_maxvol1d2);
    SUITE_ADD_TEST(suite, Test_qmarray_svd);
    SUITE_ADD_TEST(suite, Test_qmarray_serialize);
    
    return suite;
}

void Test_function_train_initsum(CuTest * tc){

    printf("Testing function: function_train_initsum \n");

    double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};
    
    enum poly_type ptype = LEGENDRE;
    struct FtApproxArgs * ftargs = 
            ft_approx_args_createpoly(4, &ptype, NULL);
    struct BoundingBox * bounds = bounding_box_init_std(4);
    
    struct FunctionTrain * ft = function_train_initsum(4, funcs, args, bounds, 
                                    ftargs);
        
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
                    tval = funcs[0](pt[0],args[0]) + 
                            funcs[1](pt[1],args[1]) + 
                            funcs[2](pt[2],args[2]) + 
                            funcs[3](pt[3],args[3]);
                    val = function_train_eval(ft,pt);
                    den += pow(tval,2.0);
                    err += pow(tval-val,2.0);
                }
            }
        }
    }
    err = err/den;

    CuAssertDblEquals(tc,0.0,err,1e-15);
    //printf("err = %G\n",err);
    
    free(xtest);
    function_train_free(ft);
    bounding_box_free(bounds);
    ft_approx_args_free(ftargs);
}   

void Test_function_train_linear(CuTest * tc)
{
    printf("Testing Function: function_train_linear \n");
    
    struct BoundingBox * bounds = bounding_box_init_std(3);

    double coeffs[3] = {1.0, 2.0, 3.0};
    struct FunctionTrain * f =function_train_linear(3, bounds, coeffs,NULL);
    
    double pt[3] = { -0.1, 0.4, 0.2 };
    double eval = function_train_eval(f,pt);
    CuAssertDblEquals(tc, 1.3, eval, 1e-14);
    
    pt[0] = 0.8; pt[1] = -0.2; pt[2] = 0.3;
    eval = function_train_eval(f,pt);
    CuAssertDblEquals(tc, 1.3, eval, 1e-14);

    pt[0] = -0.8; pt[1] = 1.0; pt[2] = -0.01;
    eval = function_train_eval(f,pt);
    CuAssertDblEquals(tc, 1.17, eval, 1e-14);
    
    bounding_box_free(bounds);
    function_train_free(f);
}

void Test_function_train_quadratic(CuTest * tc)
{
    printf("Testing Function: function_train_quadratic (1/2)\n");
    size_t dim = 3;
    double lb = -3.12;
    double ub = 2.21;
    struct BoundingBox * bounds = bounding_box_init(dim,lb,ub);
    double * quad = calloc_double(dim * dim);
    double * coeff = calloc_double(dim);
    size_t ii,jj,kk;
    for (ii = 0; ii < dim; ii++){
        coeff[ii] = randu();
        for (jj = 0; jj < dim; jj++){
            quad[ii*dim+jj] = randu();
        }
    }
    struct FunctionTrain * f = function_train_quadratic(dim, bounds, quad,
                                                        coeff,NULL);
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
    bounding_box_free(bounds);
    function_train_free(f);
}

void Test_function_train_quadratic2(CuTest * tc)
{
    printf("Testing Function: function_train_quadratic (2/2)\n");
    size_t dim = 4;
    double lb = -1.32;
    double ub = 6.0;
    struct BoundingBox * bounds = bounding_box_init(dim,lb,ub);
    double * quad = calloc_double(dim * dim);
    double * coeff = calloc_double(dim);
    size_t ii,jj,kk,zz;
    for (ii = 0; ii < dim; ii++){
        coeff[ii] = randu();
        for (jj = 0; jj < dim; jj++){
            quad[ii*dim+jj] = randu();
        }
    }
    struct FunctionTrain * f = function_train_quadratic(dim, bounds, quad,coeff,NULL);
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
    bounding_box_free(bounds);
    function_train_free(f);
}


void Test_function_train_sum_function_train_round(CuTest * tc)
{
    printf("Testing Function: function_train_sum and ft_round \n");
    
    struct BoundingBox * bounds = bounding_box_init_std(3);

    double coeffs[3] = {1.0, 2.0, 3.0};
    struct FunctionTrain * a =function_train_linear(3, bounds, coeffs,NULL);

    double coeffsb[3] = {1.5, -0.2, 3.310};
    struct FunctionTrain * b = function_train_linear(3,bounds,coeffsb,NULL);
    
    struct FunctionTrain * c = function_train_sum(a,b);
    CuAssertIntEquals(tc,1,c->ranks[0]);
    CuAssertIntEquals(tc,4,c->ranks[1]);
    CuAssertIntEquals(tc,4,c->ranks[2]);
    CuAssertIntEquals(tc,1,c->ranks[3]);


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
    
    struct FunctionTrain * d = function_train_round(c, 1e-10);
    CuAssertIntEquals(tc,1,d->ranks[0]);
    CuAssertIntEquals(tc,2,d->ranks[1]);
    CuAssertIntEquals(tc,2,d->ranks[2]);
    CuAssertIntEquals(tc,1,d->ranks[3]);

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
    

    bounding_box_free(bounds);
    function_train_free(a);
    function_train_free(b);
    function_train_free(c);
    function_train_free(d);
}

void Test_function_train_scale(CuTest * tc)
{
    printf("Testing Function: function_train_scale \n");
    double (*funcs [4])(double, void *) = {func, func2, func3, func4};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};
    
    struct BoundingBox * bounds = bounding_box_init_std(4);
    struct FunctionTrain * ft = function_train_initsum(4, funcs, args, bounds,NULL); 

    double pt[4];
    double val, tval;
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
                    tval = func(pt[0],&c) + func2(pt[1],&c2) + 
                            func3(pt[2],&c3) + func4(pt[3],&c4);
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
    bounding_box_free(bounds);
    function_train_free(ft);
}

void Test_function_train_product(CuTest * tc)
{
    printf("Testing Function: function_train_product \n");
    double (*funcs [4])(double, void *) = {func, func2, func3, func4};
    double (*funcs2 [4])(double, void *) = {func2, func5, func4, func6};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};
    struct counter c5; c5.N = 0;
    struct counter c6; c6.N = 0;
    struct counter c7; c7.N = 0;
    struct counter c8; c8.N = 0;
    void * args2[4] = {&c5, &c6, &c7, &c8};
    
    struct BoundingBox * bounds = bounding_box_init_std(4);
    struct FunctionTrain * ft = function_train_initsum(4, funcs, args, bounds,NULL); 
    struct FunctionTrain * gt = function_train_initsum(4, funcs2, args2, bounds,NULL); 
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
    bounding_box_free(bounds);
    function_train_free(ft);
    function_train_free(gt);
    function_train_free(ft2);
}

void Test_function_train_integrate(CuTest * tc)
{
    printf("Testing Function: function_train_integrate \n");
    double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};
    
    struct BoundingBox * bounds = bounding_box_init_std(4);
    bounds->lb[0] = 0;
    bounds->lb[1] = -1.0;
    bounds->lb[2] = -5.0;
    bounds->lb[3] = -5.0;
    struct FunctionTrain * ft = function_train_initsum(4, funcs, args, bounds,NULL); 
    double out =  function_train_integrate(ft);
    
    double shouldbe = 110376.0/5.0;
    double rel_error = pow(out-shouldbe,2)/fabs(shouldbe);
    CuAssertDblEquals(tc, 0.0 ,rel_error,1e-15);

    bounding_box_free(bounds);
    function_train_free(ft);
}

void Test_function_train_inner(CuTest * tc)
{
    printf("Testing Function: function_train_inner \n");
    double (*funcs [4])(double, void *) = {func, func2, func3, func4};
    double (*funcs2 [4])(double, void *) = {func6, func5, func4, func3};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};
    struct counter c5; c5.N = 0;
    struct counter c6; c6.N = 0;
    struct counter c7; c7.N = 0;
    struct counter c8; c8.N = 0;
    void * args2[4] = {&c5, &c6, &c7, &c8};
    
    struct BoundingBox * bounds = bounding_box_init_std(4);
    struct FunctionTrain * ft = function_train_initsum(4,funcs,args,bounds,NULL); 
    struct FunctionTrain * gt = function_train_initsum(4,funcs2,args2,bounds,NULL); 
    struct FunctionTrain * ft2 =  function_train_product(gt,ft);
    
    double int1 = function_train_integrate(ft2);
    double int2 = function_train_inner(gt,ft);
    
    double relerr = pow(int1-int2,2)/pow(int1,2);
    CuAssertDblEquals(tc,0.0,relerr,1e-13);
    
    bounding_box_free(bounds);
    function_train_free(ft);
    function_train_free(ft2);
    function_train_free(gt);
}

// 2 dimensional function
double funcnda(double x, void * args){
    assert ( args == NULL);
    return x;
}
double funcndb(double x, void * args){
    assert ( args == NULL);
    return pow(x,2);
}
double funcndc(double x, void * args){
    assert ( args == NULL);
    return exp(x);
}
double funcndd(double x, void * args){
    assert ( args == NULL);
    return sin(x);
}

// two dimensions
double funcnd1(double * x, void * args){
    
    assert (args == NULL);
    double out = funcnda(x[0],NULL) + funcndb(x[1],NULL);
    return out;
}

double funcnd2(double * x, void * args){

    assert (args == NULL);
    
    double out;
    out = x[0] + x[1] + x[2] + x[3];
    return out;
}

void Test_ftapprox_cross(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross  (1/4)\n");
    double (*funcs [2])(double, void *) = {&funcnda, &funcndb};
    void * args[2] = {NULL, NULL};
    
    struct BoundingBox * bounds = bounding_box_init_std(2);
    struct FunctionTrain * ftref = function_train_initsum(2,funcs,args,bounds,NULL); 
       
    size_t dim = 2;
    size_t rank[3] = {1, 3, 1};
    double yr[2] = {-1.0, 0.0};

    struct BoundingBox * bds = bounding_box_init_std(dim);
    
    enum poly_type ptype = LEGENDRE;
    struct FtApproxArgs * fapp = ft_approx_args_createpoly(dim,&ptype,NULL);
    
    struct FtCrossArgs fca;
    fca.dim = 2;
    fca.ranks = rank;
    fca.epsilon = 1e-5;
    fca.maxiter = 10;
    fca.verbose = 0;

    struct IndexSet ** isr = index_set_array_rnested(dim, rank, yr);
    struct IndexSet ** isl = index_set_array_lnested(dim, rank, yr);
    
    //print_index_set_array(2,isr);
    //print_index_set_array(2,isl);

    struct FunctionTrain * ft = ftapprox_cross(funcnd1,NULL,bds,ftref,
                                    isl, isr, &fca,fapp);
    
    //print_index_set_array(2,isr);
    //print_index_set_array(2,isl);

    size_t N = 20;
    double * xtest = linspace(-1,1,N);
    size_t ii,jj;
    double err = 0.0;
    double den = 0.0;
    double pt[2];
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            pt[0] = xtest[ii];
            pt[1] = xtest[jj];
            den += pow(funcnd1(pt,NULL),2);
            err += pow(funcnd1(pt,NULL) - function_train_eval(ft,pt),2);
        }
    }
    
    err /= den;
    CuAssertDblEquals(tc,0.0,err,1e-15);

    index_set_array_free(dim,isr);
    index_set_array_free(dim,isl);
    //
    bounding_box_free(bds);
    bounding_box_free(bounds);
    free(fapp);
    function_train_free(ft);
    function_train_free(ftref);
    free(xtest);
}

void Test_ftapprox_cross2(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross  (2/4)\n");
     
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-1.0,1.0);
    size_t ii,jj,kk,ll;

    struct FunctionMonitor * fm = function_monitor_initnd(funcnd2,NULL,dim,1000);

    //struct FunctionTrain * ft = 
    //    function_train_cross(funcnd2,NULL,dim,lb,ub,NULL);
    struct FunctionTrain * ft = 
        function_train_cross(function_monitor_eval,fm,bds,NULL,NULL,NULL);
    function_monitor_free(fm);

    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;
    double pt[4];
    
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    den += pow(funcnd2(pt,NULL),2.0);
                    err += pow(funcnd2(pt,NULL)-function_train_eval(ft,pt),2.0);
                    //printf("err=%G\n",err);
                }
            }
        }
    }
    err = err/den;
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc,0.0,err,1e-10);
    //CuAssertDblEquals(tc,0.0,0.0,1e-15);

    bounding_box_free(bds);
    function_train_free(ft);
    free(xtest);
}

double disc2d(double * xy, void * args)
{
    assert (args == NULL);
     
    double x = xy[0];
    double y = xy[1];
    double out = 0.0;
    if ((x > 0.5) || (y > 0.4)){
        out = 0.0;    
    }
    else{
        out = exp(5.0 * x + 5.0 * y);
        //out = x+y;
    }
    return out;
}

void Test_ftapprox_cross3(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross  (3/4)\n");
    size_t dim = 2;
    
    struct BoundingBox * bds = bounding_box_init(2,0.0,1.0); 
    
    double coeffs[2] = {0.5, 0.5};
    size_t ranks[3] = {1, 2, 1};

    struct FunctionTrain * ftref = 
            function_train_linear(dim, bds, coeffs,NULL);
            
    struct FunctionMonitor * fm = 
            function_monitor_initnd(disc2d,NULL,dim,1000*dim);
            
    double * yr  = calloc_double(dim);
    struct IndexSet ** isr = index_set_array_rnested(dim, ranks, yr);
    struct IndexSet ** isl = index_set_array_lnested(dim, ranks, yr);

    struct FtCrossArgs fca;
    fca.epsilon = 1e-4;
    fca.maxiter = 5;
    fca.verbose = 2;
    fca.dim = dim;
    fca.ranks = ranks;

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-5;
    aopts.minsize = 1e-8;
    aopts.nregions = 5;
    aopts.pts = NULL;

    enum poly_type ptype = LEGENDRE;
    struct FtApproxArgs * fapp = 
       ft_approx_args_createpwpoly(dim,&ptype,&aopts);

    struct FunctionTrain * ft = ftapprox_cross(function_monitor_eval, fm,
                                    bds, ftref, isl, isr, &fca,fapp);


    free(yr);
    ft_approx_args_free(fapp);
    index_set_array_free(dim,isr);
    index_set_array_free(dim,isl);
            
    double v1, v2;
    size_t ii,jj;
    size_t N1 = 40;
    size_t N2 = 40;
    double * xtest = linspace(0.0,1.0,N1);
    double * ytest = linspace(0.0,1.0,N2);

    double out1=0.0;
    double den=0.0;
    double pt[2];
    for (ii = 0; ii < N1; ii++){
        for (jj = 0; jj < N2; jj++){
            pt[0] = xtest[ii]; pt[1] = ytest[jj];
            v1 = disc2d(pt, NULL);
            v2 = function_train_eval(ft,pt);
            den += pow(v1,2.0);
            out1 += pow(v1-v2,2.0);
            //printf("f(%G,%G) = %G, pred = %G\n",pt[0],pt[1],v1,v2);
        }
    }

    double err = sqrt(out1/den);
    printf("err=%G\n",err);
    CuAssertDblEquals(tc,0.0,err,1e-10);
    //CuAssertDblEquals(tc,0.0,0.0,1e-15);
    //
    bounding_box_free(bds);
    function_train_free(ftref);
    function_train_free(ft);
    function_monitor_free(fm);
    free(xtest);
    free(ytest);
}

double funcH4(double * x, void * args){
    assert (args == NULL);
    //double out = 2.0*x[0] + x[1]*pow(x[2],4) + x[3]*pow(x[0],2);
    double out = 2.0*x[0] + x[1]*pow(x[2],4) +  x[3]*pow(x[0],2);
    return out;
}

void Test_ftapprox_cross4(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross  (4/4)\n");
     
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-1.0,1.0);
    size_t ii,jj,kk,ll;

    struct FunctionMonitor * fm = function_monitor_initnd(funcH4,NULL,dim,1000);

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-3;
    aopts.minsize = 1e-2;
    aopts.nregions = 4;
    aopts.pts = NULL;

    enum poly_type ptype = LEGENDRE;
    struct FtApproxArgs * fapp = 
       ft_approx_args_createpwpoly(dim,&ptype,&aopts);


    //struct FunctionTrain * ft = 
    //    function_train_cross(funcnd2,NULL,dim,lb,ub,NULL);
    struct FunctionTrain * ft = 
        function_train_cross(function_monitor_eval,fm,bds,NULL,NULL,fapp);
    function_monitor_free(fm);
    ft_approx_args_free(fapp);
    

    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;
    double pt[4];
    
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    den += pow(funcH4(pt,NULL),2.0);
                    err += pow(funcH4(pt,NULL)-function_train_eval(ft,pt),2.0);
                    //printf("err=%G\n",err);
                }
            }
        }
    }
    err = err/den;
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc,0.0,err,1e-10);
    //CuAssertDblEquals(tc,0.0,0.0,1e-15);

    bounding_box_free(bds);
    function_train_free(ft);
    free(xtest);
}



double sin10d(double * x, void * args){
    
    assert ( args == NULL );

    size_t ii;
    double out = 0.0;
    for (ii = 0; ii < 10; ii++){
        out += x[ii];
    }
    out = sin(out);
    
    return out;
}

void Test_sin10dint(CuTest * tc)
{
    printf("Testing Function: integration of sin10d AND (de)serialization\n");
       
    size_t dim = 10;
    size_t rank[11] = {1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1};
    double yr[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    struct BoundingBox * bds = bounding_box_init(dim,0.0,1.0);
    
    enum poly_type ptype = LEGENDRE;
    struct FtApproxArgs * fapp = ft_approx_args_createpoly(dim,&ptype,NULL);
    
    struct FtCrossArgs fca;
    fca.dim = dim;
    fca.ranks = rank;
    fca.epsilon = 1e-5;
    fca.maxiter = 10;
    fca.verbose = 0;

    struct IndexSet ** isr = index_set_array_rnested(dim, rank, yr);
    struct IndexSet ** isl = index_set_array_lnested(dim, rank, yr);
    
    //print_index_set_array(dim,isr);
    //print_index_set_array(dim,isl);

    double coeffs[10] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 1.0};
    struct FunctionTrain * ftref =function_train_linear(dim, bds, coeffs,NULL);


    struct FunctionTrain * ft = ftapprox_cross(sin10d,NULL,bds,ftref,
                                    isl, isr, &fca,fapp);
    
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
    printf("Relative error of integrating 10 dimensional sin = %G\n",relerr);
    CuAssertDblEquals(tc,0.0,relerr,1e-12);


    index_set_array_free(dim,isr);
    index_set_array_free(dim,isl);
    //
    bounding_box_free(bds);
    free(fapp);
    free(text);
    function_train_free(ft);
    function_train_free(ftd);
    function_train_free(ftref);
}

double sin100d(double * x, void * args){
    
    assert ( args == NULL );

    size_t ii;
    double out = 0.0;
    for (ii = 0; ii < 100; ii++){
        out += x[ii];
    }
    out = sin(out);
    
    return out;
}

void Test_sin100dint(CuTest * tc)
{
    printf("Testing Function: integration of sin100d\n");
       
    size_t dim = 100;
    struct BoundingBox * bds = bounding_box_init(dim,0.0,1.0);

    struct FunctionMonitor * fm = 
            function_monitor_initnd(sin100d,NULL,dim,10000);
    struct FunctionTrain * ft = 
        function_train_cross(function_monitor_eval,fm,bds,NULL,NULL,NULL);
    //struct FunctionTrain * ft = function_train_cross(sin100d,NULL,dim,lb,ub,NULL);
    function_monitor_free(fm);

    double intval = function_train_integrate(ft);
    
    double should = -0.00392679526107635150777939525615131307695379649361;

    double relerr = fabs(intval-should)/fabs(should);
    printf("Relative error of integrating 100 dimensional sin = %G\n",relerr);
    CuAssertDblEquals(tc,0.0,relerr,1e-10);

    function_train_free(ft);
    bounding_box_free(bds);
}

double sin1000d(double * x, void * args){
    
    assert ( args == NULL );

    size_t ii;
    double out = 0.0;
    for (ii = 0; ii < 1000; ii++){
        out += x[ii];
    }
    out = sin(out);
    
    return out;
}

void Test_sin1000dint(CuTest * tc)
{
    printf("Testing Function: integration of sin1000d\n");
       
    size_t dim = 1000;
    size_t rank[1001];
    double yr[1000];
    double coeffs[1000];
    
    struct BoundingBox * bds = bounding_box_init(dim,0.0,1.0);
    size_t ii;
    for (ii = 0; ii < dim; ii++){
        rank[ii] = 2;
        yr[ii] = 0.0;
        coeffs[ii] = 1.0/ (double) dim;
    }
    rank[0] = 1;
    rank[dim] = 1;
    
    enum poly_type ptype = LEGENDRE;
    struct FtApproxArgs * fapp = ft_approx_args_createpoly(dim,&ptype,NULL);
    
    struct FtCrossArgs fca;
    fca.dim = dim;
    fca.ranks = rank;
    fca.epsilon = 1e-5;
    fca.maxiter = 10;
    fca.verbose = 1;

    struct IndexSet ** isr = index_set_array_rnested(dim, rank, yr);
    struct IndexSet ** isl = index_set_array_lnested(dim, rank, yr);
    
    //print_index_set_array(dim,isr);
    //print_index_set_array(dim,isl);

    struct FunctionTrain * ftref =function_train_linear(dim, bds, coeffs,NULL);
    struct FunctionTrain * ft = ftapprox_cross(sin1000d,NULL,bds,ftref,
                                    isl, isr, &fca,fapp);

    double intval = function_train_integrate(ft);
    double should = -2.6375125156875276773939642726964969819689605535e-19;

    double relerr = fabs(intval-should)/fabs(should);
    printf("Relative error of integrating 1000 dimensional sin = %G\n",relerr);
    CuAssertDblEquals(tc,0.0,relerr,1e-10);


    index_set_array_free(dim,isr);
    index_set_array_free(dim,isl);
    //
    bounding_box_free(bds);
    free(fapp);
    function_train_free(ft);
    function_train_free(ftref);
}

CuSuite * CLinalgFuncTrainGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_function_train_initsum);
    SUITE_ADD_TEST(suite, Test_function_train_linear);
    SUITE_ADD_TEST(suite, Test_function_train_quadratic);
    SUITE_ADD_TEST(suite, Test_function_train_quadratic2);
    SUITE_ADD_TEST(suite, Test_function_train_sum_function_train_round);
    SUITE_ADD_TEST(suite, Test_function_train_scale);
    SUITE_ADD_TEST(suite, Test_function_train_product);
    SUITE_ADD_TEST(suite, Test_function_train_integrate);
    SUITE_ADD_TEST(suite, Test_function_train_inner);
    
    SUITE_ADD_TEST(suite, Test_ftapprox_cross);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross2);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross3);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross4);
    SUITE_ADD_TEST(suite, Test_sin10dint);

    //SUITE_ADD_TEST(suite, Test_sin100dint);
    //SUITE_ADD_TEST(suite, Test_sin1000dint);
    return suite;
}

double funcGrad(double * x, void * args){
    assert (args == NULL);
    double out = x[0] * x[1] + x[2] * x[3];
    return out;
}

void Test_ftapprox_grad(CuTest * tc)
{
    printf("Testing Function: function_train_gradient\n");
     
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-10.0,10.0);

    struct FunctionTrain * ft = 
        function_train_cross(funcGrad,NULL,bds,NULL,NULL,NULL);
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
    bounding_box_free(bds); bds = NULL;
}

double funcHess(double * x, void * args){
    assert (args == NULL);
    //double out = 2.0*x[0] + x[1]*pow(x[2],4) + x[3]*pow(x[0],2);
    double out =  x[0] + pow(x[0],2)*x[2] +  x[1] * pow(x[2],4) ;// + x[3]*pow(x[0],2);
    return out;
}

void Test_ftapprox_hess(CuTest * tc)
{
    printf("Testing Function: function_train_hessian\n");
     
    size_t dim = 3;
    struct BoundingBox * bds = bounding_box_init(dim,-2.0,2.0);
    size_t ii,jj,kk;//,ll;
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-4;
    aopts.minsize = 1e-2;
    aopts.nregions = 5;
    aopts.pts = NULL;

    enum poly_type ptype = LEGENDRE;
    struct FtApproxArgs * fapp = 
       ft_approx_args_createpwpoly(dim,&ptype,&aopts);


    struct FunctionTrain * ft = 
        function_train_cross(funcHess,NULL,bds,NULL,NULL,fapp);
    ft_approx_args_free(fapp);
    printf("ranks are\n");
    iprint_sz(dim+1,ft->ranks);
    size_t N = 10;
    double * xtest = linspace(bds->lb[0],bds->ub[0],N);
    double err = 0.0;
    double den = 0.0;
    double ptt[3];
    
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                //for (ll = 0; ll < N; ll++){
                    ptt[0] = xtest[ii]; ptt[1] = xtest[jj]; 
                    ptt[2] = xtest[kk]; //ptt[3] = xtest[ll];
                    den += pow(funcHess(ptt,NULL),2.0);
                    err += pow(funcHess(ptt,NULL) - 
                                    function_train_eval(ft,ptt),2.0);
                    //printf("err=%G\n",err);
               // }
            }
        }
    }

    //printf("ft ranks = \n");
    //iprint_sz(dim+1,ft->ranks);
    err = sqrt(err/den);
    printf("err=%G\n",err);
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
    bounding_box_free(bds);bds = NULL;
}

CuSuite * CLinalgFuncTrainArrayGetSuite()
{
    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_ftapprox_grad);
    SUITE_ADD_TEST(suite, Test_ftapprox_hess);
    return suite;
}

void Test_rightorth(CuTest * tc)
{
    printf("Testing Function: function_train_orthor\n");
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-10.0,10.0);

    struct FunctionTrain * ft = 
        function_train_cross(funcGrad,NULL,bds,NULL,NULL,NULL);
    //struct FunctionTrain * ft = 
    //    function_train_constant(dim,1.0,bds,NULL);
    
    struct FunctionTrain * fcopy = function_train_copy(ft);
    struct FunctionTrain * ao = function_train_orthor(ft);
    size_t ii,jj,kk;
    for (ii = 1; ii < dim; ii++){
        struct Qmarray * temp = qmaqmat(ao->cores[ii],ao->cores[ii]);
        double * intmat = qmarray_integrate(temp);
        //dprint2d_col(temp->nrows, temp->ncols,intmat);
        //qmarray_free(temp); temp = NULL;
        for (jj = 0; jj < temp->ncols; jj++){
            for (kk = 0; kk < temp->nrows; kk++){
                if (jj == kk){
                    CuAssertDblEquals(tc,1.0,intmat[jj*temp->nrows+kk],1e-14);
                }
                else{
                    CuAssertDblEquals(tc,0.0,intmat[jj*temp->nrows+kk],1e-14);
                }
            }
        }
        free(intmat); intmat = NULL;
        qmarray_free(temp); temp = NULL;
    }
    
    size_t ll;
    size_t N = 20;
    double * x = linspace(-10.0,10.0,N);
    double pt[4];
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = x[ii]; pt[1] = x[jj];
                    pt[2] = x[kk]; pt[3] = x[ll];
                    double eval1 = function_train_eval(ao,pt);
                    double eval2 = function_train_eval(fcopy,pt);
                    CuAssertDblEquals(tc,eval1,eval2,1e-12);
                }
            }
        }
    }
    free(x); x = NULL;

    //double norm = function_train_norm2(fcopy);
    //double diff = function_train_norm2diff(ao,fcopy)/norm;
    //printf("\n\nfinal diff = %G\n",diff);
    //CuAssertDblEquals(tc,0.0,diff,1e-14);

    function_train_free(ft); ft = NULL;
    function_train_free(fcopy); fcopy = NULL;
    function_train_free(ao); ao = NULL;
    bounding_box_free(bds); bds = NULL;
}

void Test_dmrglr(CuTest * tc)
{

    printf("Testing Function: dmrg_sweep_lr and dmrg_sweep_rl\n");
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-10.0,10.0);

    struct FunctionTrain * ft = 
        function_train_cross(funcGrad,NULL,bds,NULL,NULL,NULL);
    double normsize = function_train_norm2(ft);
    struct FunctionTrain * fcopy = function_train_copy(ft);
    
    struct FunctionTrain * ao = function_train_orthor(ft);
    double ** phi = malloc((dim-1)*sizeof(double));
    double ** psi = malloc((dim-1)*sizeof(double));
    size_t ii;
    for (ii = 0; ii < dim-1; ii++){
        phi[ii] = NULL;
        psi[ii] = NULL;
    }
    //dmrg_update_all_right(fcopy,ao,psi);
    dmrg_update_all_right(ao,fcopy,psi);
    
    struct FunctionTrain * out = dmrg_sweep_lr(ao,fcopy,phi,psi,0);
    double diff = function_train_norm2diff(out,fcopy)/normsize;
    printf("diff = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-14);

    struct FunctionTrain * out2 = dmrg_sweep_rl(out,fcopy,phi,psi,0);
    diff = function_train_norm2diff(out2,fcopy)/normsize;
    CuAssertDblEquals(tc,0.0,diff,1e-14);

    for (ii = 0; ii < dim-1; ii++){
        free(phi[ii]);
        free(psi[ii]);
    }
    free(phi); 
    free(psi);
    function_train_free(ft); ft = NULL;
    function_train_free(ao); ao = NULL;
    function_train_free(fcopy); fcopy = NULL;
    function_train_free(out); out = NULL;
    function_train_free(out2); out2 = NULL;
    bounding_box_free(bds); bds = NULL;
}

void Test_dmrg_approx(CuTest * tc)
{

    printf("Testing Function: dmrg_approx\n");
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-1.0,1.0);
    struct FunctionTrain * ft = 
        function_train_cross(funcGrad,NULL,bds,NULL,NULL,NULL);
    printf("lets go \n");
    //struct FunctionTrain * start = function_train_copy(ft);
    double coeffs[4] = {0.5,0.5,0.5,0.5};
    struct FunctionTrain * start = function_train_constant(dim,2.0,bds,NULL);
    //struct FunctionTrain * start = function_train_linear(dim,bds,coeffs,NULL);

    dmrg_approx(&start,ft,1e-5,10,1,1e-10);
    double diff = function_train_norm2diff(start,ft);
    printf("start ranks = ");
    iprint_sz(dim+1,start->ranks);

    CuAssertDblEquals(tc,0.0,diff,1e-10);

    bounding_box_free(bds); bds = NULL;
    function_train_free(ft); ft = NULL;
    function_train_free(start); start = NULL;
    
}
 
CuSuite * CLinalgDMRGGetSuite()
{
    CuSuite * suite = CuSuiteNew();
    //SUITE_ADD_TEST(suite, Test_rightorth);
    //SUITE_ADD_TEST(suite, Test_dmrglr);
    SUITE_ADD_TEST(suite, Test_dmrg_approx);
    return suite;
}


void RunAllTests(void) {
    
    printf("Running Test Suite: lib_clinalg\n");

    CuString * output = CuStringNew();
    CuSuite * suite = CuSuiteNew();
    
    CuSuite * clin = CLinalgGetSuite();
    CuSuite * qma = CLinalgQmarrayGetSuite();
    CuSuite * ftr = CLinalgFuncTrainGetSuite();
    CuSuite * fta = CLinalgFuncTrainArrayGetSuite();
    CuSuite * dmrg = CLinalgDMRGGetSuite();
    //CuSuiteAddSuite(suite, clin);
    //CuSuiteAddSuite(suite, qma);
    //CuSuiteAddSuite(suite, ftr);
    //CuSuiteAddSuite(suite, fta);
    CuSuiteAddSuite(suite, dmrg);
    CuSuiteRun(suite);
    CuSuiteSummary(suite, output);
    CuSuiteDetails(suite, output);
    printf("%s \n", output->buffer);
    
    CuSuiteDelete(clin);
    CuSuiteDelete(qma);
    CuSuiteDelete(ftr);
    CuSuiteDelete(fta);
    CuSuiteDelete(dmrg);
    CuStringDelete(output);
    free(suite);
}

int main(void) {
    RunAllTests();
}
