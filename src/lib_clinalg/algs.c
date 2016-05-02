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

/** \file algs.c
 * Provides routines for performing continuous linear algebra and 
 * working with function trains
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>

#include "qmarray_qr.h"
#include "lib_funcs.h"
#include "indmanage.h"
#include "algs.h"
#include "array.h"
#include "linalg.h"

#define ZEROTHRESH 1e2*DBL_EPSILON
//#define ZEROTHRESH 0.0
//#define ZEROTHRESH  1e-14
//#define ZEROTHRESH 1e-20S

#ifndef VPREPCORE
    #define VPREPCORE 0
#endif

#ifndef VFTCROSS
    #define VFTCROSS 0
#endif


////////////////////////////////////////////////////////////////////////////
// function_train

/***********************************************************//**
    Evaluate a function train

    \param[in] ft - function train
    \param[in] x  - location at which to evaluate

    \return val - value of the function train
***************************************************************/
double function_train_eval(struct FunctionTrain * ft, double * x)
{
    
    size_t dim = ft->dim;
    assert(ft->ranks[0] == 1);
    assert(ft->ranks[dim] == 1);

    size_t ii = 0;
    size_t maxrank = function_train_maxrank(ft);
    if (ft->evalspace1 == NULL){
        ft->evalspace1 = calloc_double(maxrank*maxrank);
    }
    if (ft->evalspace2 == NULL){
        ft->evalspace2 = calloc_double(maxrank*maxrank);
    }
    if (ft->evalspace3 == NULL){
        ft->evalspace3 = calloc_double(maxrank*maxrank);
    }

    double * t1 = ft->evalspace1;
    generic_function_1darray_eval2(
        ft->cores[ii]->nrows * ft->cores[ii]->ncols, 
        ft->cores[ii]->funcs, x[ii],t1);

    double * t2 = ft->evalspace2;
    double * t3 = ft->evalspace3;
    int onsol = 1;
    for (ii = 1; ii < dim; ii++){
        generic_function_1darray_eval2(
            ft->cores[ii]->nrows * ft->cores[ii]->ncols, 
            ft->cores[ii]->funcs, x[ii],t2);
            
        if (ii%2 == 1){
            // previous times new core
            cblas_dgemv(CblasColMajor,CblasTrans,
                        ft->ranks[ii], ft->ranks[ii+1], 1.0, 
                        t2, ft->ranks[ii],
                        t1, 1, 0.0, t3, 1);
            onsol = 2;

        }
        else {
//            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, 1,
//                ft->ranks[ii+1], ft->ranks[ii], 1.0, t3, 1, t2,
//                ft->ranks[ii], 0.0, t1, 1)
            cblas_dgemv(CblasColMajor,CblasTrans,
                        ft->ranks[ii], ft->ranks[ii+1], 1.0, 
                        t2, ft->ranks[ii],
                        t3,1,0.0,t1,1);
            onsol = 1;

        }
    }
    
    double out;// = 0.123456789;
    if (onsol == 2){
        out = t3[0];
//        free(t3); t3 = NULL;
    }
    else if ( onsol == 1){
        out = t1[0];
//        free(t1); t1 = NULL;
    }
    else{
        fprintf(stderr,"Weird error in function_train_val\n");
        exit(1);
    }
    
    return out;
}

/***********************************************************//**
    Evaluate a function train with perturbations
    in every coordinate

    \param[in]     ft   - function train
    \param[in]     x    - location at which to evaluate
    \param[in]     pert - perturbed vals  (order of storage is )
                          (dim 1) - +
                          (dim 2) - +
                          (dim 3) - +
                          (dim 4) - +
                          ... for a total of 2d values
    \param[in,out] vals - values at perturbation points 

    \return val - value of the function train at x
***************************************************************/
double function_train_eval_co_perturb(struct FunctionTrain * ft, const double * x, const double * pert, double * vals)
{
    
    size_t dim = ft->dim;
    assert(ft->ranks[0] == 1);
    assert(ft->ranks[dim] == 1);

    size_t maxrank = function_train_maxrank(ft);
    if (ft->evalspace1 == NULL){
        ft->evalspace1 = calloc_double(maxrank*maxrank);
    }
    if (ft->evalspace2 == NULL){
        ft->evalspace2 = calloc_double(maxrank*maxrank);
    }
    if (ft->evalspace3 == NULL){
        ft->evalspace3 = calloc_double(maxrank*maxrank);
    }
    if (ft->evaldd1 == NULL){
        ft->evaldd1 = malloc_dd(dim);
        for (size_t ii = 0; ii < dim; ii++){
            ft->evaldd1[ii] = calloc_double(maxrank * maxrank);
        }
    }
    if (ft->evaldd3 == NULL){
        ft->evaldd3 = malloc_dd(dim);
        for (size_t ii = 0; ii < dim; ii++){
            ft->evaldd3[ii] = calloc_double(maxrank);
        }
    }
    if (ft->evaldd4 == NULL){
        ft->evaldd4 = malloc_dd(dim);
        for (size_t ii = 0; ii < dim; ii++){
            ft->evaldd4[ii] = calloc_double(maxrank);
        }
    }
    
    // center cores
//    double ** cores_center = malloc_dd(dim);

    /* double ** cores_neigh = malloc_dd(2*dim); // cores for neighbors */
    /* double ** bprod = malloc_dd(dim); */
    /* double ** fprod = malloc_dd(dim); */

    double ** cores_center = ft->evaldd1;
    double ** bprod = ft->evaldd3;
    double ** fprod = ft->evaldd4;
    
    // evaluate the cores for the center
    for (size_t ii = 0; ii < dim; ii++){
        /* fprod[ii] = calloc_double(ft->ranks[ii+1]); */
        /* cores_center[ii] = calloc_double(ft->ranks[ii] * ft->ranks[ii+1]); */
        if (ii == 0){
            generic_function_1darray_eval2(
                ft->cores[ii]->nrows * ft->cores[ii]->ncols, 
                ft->cores[ii]->funcs, x[ii], fprod[ii]);
            memmove(cores_center[ii],fprod[ii],ft->ranks[1]*sizeof(double));
        }
        else{
            generic_function_1darray_eval2(
                ft->cores[ii]->nrows * ft->cores[ii]->ncols, 
                ft->cores[ii]->funcs, x[ii], cores_center[ii]);
            cblas_dgemv(CblasColMajor,CblasTrans,
                        ft->ranks[ii], ft->ranks[ii+1], 1.0, 
                        cores_center[ii], ft->ranks[ii],
                        fprod[ii-1], 1, 0.0, fprod[ii], 1);
        }

        /* cores_neigh[2*ii] = calloc_double(ft->ranks[ii] * ft->ranks[ii+1]); */
        /* cores_neigh[2*ii+1] = calloc_double(ft->ranks[ii] * ft->ranks[ii+1]); */

        /* generic_function_1darray_eval2( */
        /*     ft->cores[ii]->nrows * ft->cores[ii]->ncols, */
        /*     ft->cores[ii]->funcs, pert[2*ii], cores_neigh[2*ii]); */
        /* generic_function_1darray_eval2( */
        /*     ft->cores[ii]->nrows * ft->cores[ii]->ncols, */
        /*     ft->cores[ii]->funcs, pert[2*ii+1], cores_neigh[2*ii+1]); */
    }

    for (int ii = dim-1; ii >= 0; ii--){
        /* bprod[ii] = calloc_double(ft->ranks[ii]); */
        if (ii == (int)dim-1){
            generic_function_1darray_eval2(
                ft->cores[ii]->nrows * ft->cores[ii]->ncols, 
                ft->cores[ii]->funcs, x[ii], bprod[ii]);
        }
        else{
            cblas_dgemv(CblasColMajor,CblasNoTrans,
                        ft->ranks[ii], ft->ranks[ii+1], 1.0, 
                        cores_center[ii], ft->ranks[ii],
                        bprod[ii+1], 1, 0.0, bprod[ii], 1);
        }
    }

    for (size_t ii = 0; ii < dim; ii++){

        generic_function_1darray_eval2(
            ft->cores[ii]->nrows * ft->cores[ii]->ncols,
            ft->cores[ii]->funcs, pert[2*ii], ft->evalspace1);
        generic_function_1darray_eval2(
            ft->cores[ii]->nrows * ft->cores[ii]->ncols,
            ft->cores[ii]->funcs, pert[2*ii+1], ft->evalspace2);
        if (ii == 0){
            vals[ii] = cblas_ddot(ft->ranks[1],ft->evalspace1,1,bprod[1],1);
            vals[ii+1] = cblas_ddot(ft->ranks[1],ft->evalspace2,1,bprod[1],1);
        }
        else if (ii == dim-1){
            vals[2*ii] = cblas_ddot(ft->ranks[ii],ft->evalspace1,1,
                                    fprod[dim-2],1);
            vals[2*ii+1] = cblas_ddot(ft->ranks[ii],ft->evalspace2,
                                      1,fprod[dim-2],1);
        }
        else{
            /* double * temp = calloc_double(ft->ranks[ii] * ft->ranks[ii+1]); */
            cblas_dgemv(CblasColMajor,CblasNoTrans,
                        ft->ranks[ii], ft->ranks[ii+1], 1.0, 
                        ft->evalspace1, ft->ranks[ii],
                        bprod[ii+1], 1, 0.0, ft->evalspace3, 1);
            vals[2*ii] = cblas_ddot(ft->ranks[ii],ft->evalspace3,1,
                                    fprod[ii-1],1);

            cblas_dgemv(CblasColMajor,CblasNoTrans,
                        ft->ranks[ii], ft->ranks[ii+1], 1.0, 
                        ft->evalspace2, ft->ranks[ii],
                        bprod[ii+1], 1, 0.0, ft->evalspace3, 1);

            vals[2*ii+1] = cblas_ddot(ft->ranks[ii],ft->evalspace3,1,
                                    fprod[ii-1],1);
            /* free(temp); temp = NULL; */
        }

    }

    double out = fprod[dim-1][0];
    /* printf("out = %G\n",out); */
    /* printf("should equal = %G\n",bprod[0][0]); */
    /* free_dd(dim,cores_center); */
    /* free_dd(2*dim,cores_neigh); */
    /* free_dd(dim,bprod); bprod = NULL; */
    /* free_dd(dim,fprod); fprod = NULL; */

    return out;
}

/********************************************************//**
    Right orthogonalize the cores (except the first one) of the function train

    \param[in,out] a - FT (overwritten)
    
    \return ftrl - new ft with orthogonalized cores
***********************************************************/
struct FunctionTrain * function_train_orthor(struct FunctionTrain * a)
{
    //printf("orthor\n");
    //right left sweep
    struct FunctionTrain * ftrl = function_train_alloc(a->dim);
    double * L = NULL; 
    struct Qmarray * temp = NULL;
    size_t ii = 1;
    size_t core = a->dim-ii;  
    L = calloc_double(a->cores[core]->nrows * a->cores[core]->nrows);
    memmove(ftrl->ranks,a->ranks,(a->dim+1)*sizeof(size_t));


    // update last core
//    printf("dim = %zu\n",a->dim);
    ftrl->cores[core] = qmarray_householder_simple("LQ",a->cores[core],L);
    for (ii = 2; ii < a->dim; ii++){
        core = a->dim-ii;
        //printf("on core %zu\n",core);
        temp = qmam(a->cores[core],L,ftrl->ranks[core+1]);
        free(L); L = NULL;
        L = calloc_double(ftrl->ranks[core]*ftrl->ranks[core]);
        ftrl->cores[core] = qmarray_householder_simple("LQ",temp,L);
        qmarray_free(temp); temp = NULL;
        
    }
    ftrl->cores[0] = qmam(a->cores[0],L,ftrl->ranks[1]);
    free(L); L = NULL;
    return ftrl;
}


/********************************************************//**
    Rounding of a function train

    \param[in] ain - FT 
    \param[in] epsilon - threshold

    \return ft - rounded function train
***********************************************************/
struct FunctionTrain * 
function_train_round(struct FunctionTrain * ain, double epsilon)
{
    struct FunctionTrain * a = function_train_copy(ain);
//    size_t ii;
    size_t core;
    struct Qmarray * temp = NULL;

    double delta = function_train_norm2(a);
    delta = delta * epsilon / sqrt(a->dim-1);
    //double delta = epsilon;

//    printf("begin orho\n");
    struct FunctionTrain * ftrl = function_train_orthor(a);
//    printf("ortho gonalized\n");
    struct FunctionTrain * ft = function_train_alloc(a->dim);
    //struct FunctionTrain * ft = function_train_copy(ftrl);
    double * vt = NULL;
    double * s = NULL;
    double * sdiag = NULL;
    double * svt = NULL;
    size_t rank;
    size_t sizer;
    core = 0;
    ft->ranks[core] = 1;
    sizer = ftrl->cores[core]->ncols;
    
    //*
    //printf("rank before = %zu\n",ftrl->ranks[core+1]);
    rank = qmarray_truncated_svd(ftrl->cores[core], &(ft->cores[core]),
                                        &s, &vt, delta);
    //printf("rankdone\n");
    //printf("rank after = %zu\n",rank);
    ft->ranks[core+1] = rank;
    sdiag = diag(rank,s);
    svt = calloc_double(rank * sizer);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, rank, 
                sizer, rank, 1.0, sdiag, rank, vt, rank, 0.0, 
                svt, rank);
    //printf("rank\n");
    temp = mqma(svt,ftrl->cores[core+1],rank);
    //*/

    free(vt); vt = NULL;
    free(s); s = NULL;
    free(sdiag); sdiag = NULL;
    free(svt); svt = NULL;
    for (core = 1; core < a->dim-1; core++){
        rank = qmarray_truncated_svd(temp, &(ft->cores[core]), &s, &vt, delta);
        qmarray_free(temp); temp = NULL;

        ft->ranks[core+1] = rank;
        sdiag = diag(rank,s);
        svt = calloc_double(rank * a->ranks[core+1]);
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, rank, 
                    a->ranks[core+1], rank, 1.0, sdiag, rank, vt, rank, 0.0, 
                    svt, rank);
        temp = mqma(svt,ftrl->cores[core+1],rank);
        free(vt); vt = NULL;
        free(s); s = NULL;
        free(sdiag); sdiag = NULL;
        free(svt); svt = NULL;
    }
    core = a->dim-1;
    ft->cores[core] = temp;
    ft->ranks[a->dim] = 1;
    
    function_train_free(ftrl);
    /*
    for (ii = 0; ii < ft->dim; ii++){
        qmarray_roundt(&ft->cores[ii], epsilon);
    }
    */

    function_train_free(a); a = NULL;
    return ft;
}

/********************************************************//**
    Addition of two functions in FT format

    \param[in] a - FT 1
    \param[in] b - FT 2

    \return ft - function representing a+b
***********************************************************/
struct FunctionTrain * function_train_sum(struct FunctionTrain * a,
                                          struct FunctionTrain * b)
{
    struct FunctionTrain * ft = function_train_alloc(a->dim);
    
    // first core
    ft->cores[0] = qmarray_stackh(a->cores[0], b->cores[0]);
    ft->ranks[0] = a->ranks[0];
    // middle cores
    size_t ii;
    for (ii = 1; ii < ft->dim-1; ii++){
        ft->cores[ii] = qmarray_blockdiag(a->cores[ii], b->cores[ii]);
        ft->ranks[ii] = a->ranks[ii] + b->ranks[ii];
    }
    ft->ranks[ft->dim-1] = a->ranks[a->dim-1] + b->ranks[b->dim-1];

    // last core
    ft->cores[ft->dim-1] = qmarray_stackv(a->cores[a->dim-1],
                                b->cores[b->dim-1]);
    ft->ranks[ft->dim] = a->ranks[ft->dim];
    
    return ft;
}

/********************************************************//**
    af(x) + b

    \param[in] a        - scaling factor
    \param[in] b        - offset
    \param[in] f        - object to scale
    \param[in] epsilon  - rounding tolerance

    \return ft - function representing a+b
***********************************************************/
struct FunctionTrain * function_train_afpb(double a, double b,
                        struct FunctionTrain * f, double epsilon)
{
    struct BoundingBox * bds = function_train_bds(f);
    struct FunctionTrain * off = NULL;
    if (f->cores[0]->funcs[0]->fc != LINELM){
        enum poly_type ptype = LEGENDRE;
        off  = function_train_constant(POLYNOMIAL,&ptype,
                                       f->dim,b,bds,NULL);
    }
    else{
        off = function_train_constant(LINELM,NULL,
                                       f->dim,b,bds,NULL);
    }
    struct FunctionTrain * af = function_train_copy(f);
    function_train_scale(af,a);

    struct FunctionTrain * temp = function_train_sum(af,off);
    
    
    //printf("round it \n");
    //printf("temp ranks \n");
    //iprint_sz(temp->dim+1,temp->ranks);

    struct FunctionTrain * ft = function_train_round(temp,epsilon);
    //printf("got it \n");
        
    function_train_free(off); off = NULL;
    function_train_free(af); af = NULL;
    function_train_free(temp); temp = NULL;
    bounding_box_free(bds); bds = NULL;
    
    return ft;
}


/********************************************************//**
    Scale a function train representation

    \param x [inout] - Function train
    \param a [in] - scaling factor
***********************************************************/
void function_train_scale(struct FunctionTrain * x, double a)
{
    struct GenericFunction ** temp = generic_function_array_daxpby(
        x->cores[0]->nrows*x->cores[0]->ncols,a,1,x->cores[0]->funcs,0.0,1,NULL);
    generic_function_array_free(x->cores[0]->funcs,
                x->cores[0]->nrows * x->cores[0]->ncols);
    x->cores[0]->funcs = temp;
}

/********************************************************//**
    Product of two functions in function train form

    \param[in] a  - Function train 1
    \param[in] b  - Function train 2

    \return Product \f$ f(x) = a(x)b(x) \f$
***********************************************************/
struct FunctionTrain * 
function_train_product(struct FunctionTrain * a, struct FunctionTrain * b)
{
    struct FunctionTrain * ft = function_train_alloc(a->dim);

    size_t ii;
    for (ii = 0; ii < ft->dim; ii++){
        ft->ranks[ii] = a->ranks[ii] * b->ranks[ii];
        ft->cores[ii] = qmarray_kron(a->cores[ii], b->cores[ii]);
    }

    ft->ranks[ft->dim] = a->ranks[ft->dim] * b->ranks[ft->dim];
    return ft;
}

/********************************************************//**
    Integrate a function in function train format

    \param[in] ft - Function train 1

    \return Integral \f$ \int f(x) dx \f$
***********************************************************/
double 
function_train_integrate(struct FunctionTrain * ft)
{
    size_t dim = ft->dim;
    size_t ii = 0;
    double * t1 = qmarray_integrate(ft->cores[ii]);

    double * t2 = NULL;
    double * t3 = NULL;
    for (ii = 1; ii < dim; ii++){
        t2 = qmarray_integrate(ft->cores[ii]);
        if (ii%2 == 1){
            // previous times new core
            t3 = calloc_double(ft->ranks[ii+1]);
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, 1,
                ft->ranks[ii+1], ft->ranks[ii], 1.0, t1, 1, t2,
                ft->ranks[ii], 0.0, t3, 1);
            free(t2); t2 = NULL;
            free(t1); t1 = NULL;
        }
        else {
            t1 = calloc_double(ft->ranks[ii+1]);
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, 1,
                ft->ranks[ii+1], ft->ranks[ii], 1.0, t3, 1, t2,
                ft->ranks[ii], 0.0, t1, 1);
            free(t2); t2 = NULL;
            free(t3); t3 = NULL;
        }
    }
    
    double out = 0.123456789;
    if (t1 == NULL){
        out = t3[0];
        free(t3); t3 = NULL;
    }
    else if ( t3 == NULL){
        out = t1[0];
        free(t1); t1 = NULL;
    }
    
    return out;
}

/********************************************************//**
    Inner product between two functions in FT form

    \param[in] a - Function train 1
    \param[in] b - Function train 2

    \return Inner product \f$ \int a(x)b(x) dx \f$

***********************************************************/
double function_train_inner(struct FunctionTrain * a, struct FunctionTrain * b)
{
    double out = 0.123456789;
    size_t ii;
    double * temp = qmarray_kron_integrate(b->cores[0],a->cores[0]);
    double * temp2 = NULL;

    //size_t ii;
    for (ii = 1; ii < a->dim; ii++){
        temp2 = qmarray_vec_kron_integrate(temp, a->cores[ii],b->cores[ii]);
        size_t stemp = a->cores[ii]->ncols * b->cores[ii]->ncols;
        free(temp);temp=NULL;
        temp = calloc_double(stemp);
        memmove(temp, temp2,stemp*sizeof(double));
        
        free(temp2); temp2 = NULL;
    }
    
    out = temp[0];
    free(temp); temp=NULL;

    return out;
}

/********************************************************//**
    Compute the L2 norm of a function in FT format

    \param[in] a - Function train 

    \return L2 Norm \f$ \sqrt{int a^2(x) dx} \f$
***********************************************************/
double function_train_norm2(struct FunctionTrain * a)
{
    //printf("in norm2\n");
    double out = function_train_inner(a,a);
    if (out < -ZEROTHRESH){
        if (out * out >  ZEROTHRESH){
            fprintf(stderr, "inner product of FT with itself should not be neg %G \n",out);
            exit(1);
        }
    }
    return sqrt(fabs(out));
}

/********************************************************//**
    Compute the L2 norm of the difference between two functions

    \param[in] a - function train 
    \param[in] b - function train 2

    \return L2 difference \f$ \sqrt{ \int (a(x)-b(x))^2 dx } \f$
***********************************************************/
double function_train_norm2diff(struct FunctionTrain * a, struct FunctionTrain * b)
{   
    
    struct FunctionTrain * c = function_train_copy(b);
    function_train_scale(c,-1.0);
    struct FunctionTrain * d = function_train_sum(a,c);
    //printf("in function_train_norm2diff\n");
    double val = function_train_norm2(d);
    function_train_free(c);
    function_train_free(d);
    return val;
}

/********************************************************//**
    Compute the L2 norm of the difference between two functions

    \param[in] a - function train 
    \param[in] b - function train 2

    \return Relative L2 difference 
    \f$ \sqrt{ \int (a(x)-b(x))^2 dx } / \lVert b(x) \rVert \f$
***********************************************************/
double function_train_relnorm2diff(struct FunctionTrain * a, 
                                   struct FunctionTrain * b)
{   
    
    struct FunctionTrain * c = function_train_copy(b);
    function_train_scale(c,-1.0);

    double den = function_train_inner(c,c);
    
    //printf("compute sum \n");
    struct FunctionTrain * d = function_train_sum(a,c);
    //printf("computed \n");
    double num = function_train_inner(d,d);
    
    double val = num;
    if (fabs(den) > ZEROTHRESH){
        val /= den;
    }
    if (val < -ZEROTHRESH){
//        fprintf(stderr, "relative error between two FT should not be neg %G<-%G \n",val,-ZEROTHRESH);
//        exit(1);
    }
    val = sqrt(fabs(val));

    function_train_free(c); c = NULL;
    function_train_free(d); d = NULL;
    return val;
}


/********************************************************//**
    Compute the gradient of a function train 

    \param[in] ft - Function train 

    \return gradient
***********************************************************/
struct FT1DArray * function_train_gradient(struct FunctionTrain * ft)
{
    struct FT1DArray * ftg = ft1d_array_alloc(ft->dim);
    size_t ii;
    for (ii = 0; ii < ft->dim; ii++){
        //printf("**********\n\n\n**********\n");
        //printf("ii = %zu\n",ii);
        ftg->ft[ii] = function_train_copy(ft);
        qmarray_free(ftg->ft[ii]->cores[ii]);
        ftg->ft[ii]->cores[ii] = NULL;
        //printf("get deriv ii = %zu\n",ii);
        //print_qmarray(ft->cores[ii],1,NULL);
        ftg->ft[ii]->cores[ii] = qmarray_deriv(ft->cores[ii]);
        //printf("got deriv ii = %zu\n",ii);
        //print_qmarray(ftg->ft[ii]->cores[ii],1,NULL);
    }

    return ftg;
}

/********************************************************//**
    Compute the Jacobian of a Function Train 1darray

    \param[in] fta - Function train array

    \return jacobian
***********************************************************/
struct FT1DArray * ft1d_array_jacobian(struct FT1DArray * fta)
{
    struct FT1DArray * jac = ft1d_array_alloc(fta->size * fta->ft[0]->dim);
    size_t ii,jj;
    for (ii = 0; ii < fta->ft[0]->dim; ii++){
        for (jj = 0; jj < fta->size; jj++){
            jac->ft[ii*fta->size+jj] = function_train_copy(fta->ft[jj]);
            qmarray_free(jac->ft[ii*fta->size+jj]->cores[ii]);
            jac->ft[ii*fta->size+jj]->cores[ii] = NULL;
            jac->ft[ii*fta->size+jj]->cores[ii] = 
                qmarray_deriv(fta->ft[jj]->cores[ii]);
            
        }
    }
    return jac;
}

/********************************************************//**
    Compute the hessian of a function train 

    \param[in] fta - Function train 

    \return hessian of a function train
***********************************************************/
struct FT1DArray * function_train_hessian(struct FunctionTrain * fta)
{
    struct FT1DArray * ftg = function_train_gradient(fta);

    struct FT1DArray * fth = ft1d_array_jacobian(ftg);
        
    ft1d_array_free(ftg); ftg = NULL;
    return fth;
}

/********************************************************//**
    Scale a function train array

    \param[in,out] fta   - function train array
    \param[in]     n     - number of elements in the array to scale
    \param[in]     inc   - increment between elements of array
    \param[in]     scale - value by which to scale
***********************************************************/
void ft1d_array_scale(struct FT1DArray * fta, size_t n, size_t inc, double scale)
{
    size_t ii;
    for (ii = 0; ii < n; ii++){
        function_train_scale(fta->ft[ii*inc],scale);
    }
}

/********************************************************//**
    Evaluate a function train 1darray

    \param[in] fta - Function train array to evaluate
    \param[in] x   - location at which to obtain evaluations

    \return evaluation
***********************************************************/
double * ft1d_array_eval(struct FT1DArray * fta, double * x)
{
    double * out = calloc_double(fta->size);
    size_t ii; 
    for (ii = 0; ii < fta->size; ii++){
        out[ii] = function_train_eval(fta->ft[ii], x);
    }
    return out;
}

/********************************************************//**
    Evaluate a function train 1darray 

    \param[in]     fta - Function train array to evaluate
    \param[in]     x   - location at which to obtain evaluations
    \param[in,out] out - evaluation

***********************************************************/
void ft1d_array_eval2(struct FT1DArray * fta, double * x, double * out)
{
    size_t ii; 
    for (ii = 0; ii < fta->size; ii++){
        out[ii] = function_train_eval(fta->ft[ii], x);
    }
}

/********************************************************//**
    Interpolate a function-train onto a particular grid forming 
    another function_train with a nodela basis

    \param[in] fta - Function train array to evaluate
    \param[in] N   - number of nodes in each dimension
    \param[in] x   - nodes in each dimension

    \return new function train
***********************************************************/
struct FunctionTrain *
function_train_create_nodal(struct FunctionTrain * fta, size_t * N, double ** x)
{
    struct FunctionTrain * newft = function_train_alloc(fta->dim);
    memmove(newft->ranks,fta->ranks, (fta->dim+1)*sizeof(size_t));
    for (size_t ii = 0; ii < newft->dim; ii ++){
        newft->cores[ii] = qmarray_create_nodal(fta->cores[ii],N[ii],x[ii]);
    }
    return newft;
}

/********************************************************//**
    Multiply together and sum the elements of two function train arrays
    \f[ 
        out(x) = \sum_{i=1}^{N} coeff[i] f_i(x)  g_i(x) 
    \f]
    
    \param[in] N       - number of function trains in each array
    \param[in] coeff   - coefficients to multiply each element
    \param[in] f       - first array
    \param[in] g       - second array
    \param[in] epsilon - rounding accuracy

    \return function train
***********************************************************/
struct FunctionTrain * 
ft1d_array_sum_prod(size_t N, double * coeff, 
               struct FT1DArray * f, struct FT1DArray * g, 
               double epsilon)
{
    
    struct FunctionTrain * ft1 = NULL;
    struct FunctionTrain * ft2 = NULL;
    struct FunctionTrain * out = NULL;
    struct FunctionTrain * temp = NULL;

//    printf("compute product\n");
    temp = function_train_product(f->ft[0],g->ft[0]);
    function_train_scale(temp,coeff[0]);
//    printf("scale N = %zu\n",N);
    out = function_train_round(temp,epsilon);
//    printf("rounded\n");
    function_train_free(temp); temp = NULL;
    size_t ii;
    for (ii = 1; ii < N; ii++){
        temp =  function_train_product(f->ft[ii],g->ft[ii]);
        function_train_scale(temp,coeff[ii]);

        ft2 = function_train_round(temp,epsilon);
        ft1 = function_train_sum(out, ft2);
    
        function_train_free(temp); temp = NULL;
        function_train_free(ft2); ft2 = NULL;
        function_train_free(out); out = NULL;

        out = function_train_round(ft1,epsilon);

        function_train_free(ft1); ft1 = NULL;
    }

    return out;
}

// utility function for function_train_cross (not in header file)
struct Qmarray *
prepCore(size_t ii, size_t nrows, 
         double(*f)(double *, void *), void * args,
         struct BoundingBox * bd,
         struct CrossIndex ** left_ind,struct CrossIndex ** right_ind, 
         struct FtCrossArgs * cargs, struct FtApproxArgs * fta, int t)
{

    enum function_class fc;
    void * sub_type = NULL;
    void * approx_args = NULL;
    struct FiberCut ** fcut = NULL;  
    struct Qmarray * temp = NULL;
    double ** vals = NULL;
    size_t ncols, ncuts;
    size_t dim = bd->dim;
    
    fc = ft_approx_args_getfc(fta,ii);
    sub_type = ft_approx_args_getst(fta,ii);
    approx_args = ft_approx_args_getaopts(fta, ii);
    //printf("sub_type prep_core= %d\n",*(int *)ft_approx_args_getst(fta,ii));
    //printf("t=%d\n",t);
    if (t == 1){
        ncuts = nrows;
        ncols = 1;
        /* printf("left index of merge = \n"); */
        /* print_cross_index(left_ind[ii]); */
        /* printf("right index of merge = \n"); */
        /* print_cross_index(right_ind[ii]); */
//        vals = cross_index_merge(left_ind[ii],right_ind[ii-1]);
        vals = cross_index_merge_wspace(left_ind[ii],NULL);
        fcut = fiber_cut_ndarray(f,args, dim, ii, ncuts, vals);
    }
    else if (t == -1){
        ncuts = cargs->ranks[ii+1];
        ncols = ncuts;
//        vals = cross_index_merge(left_ind[ii+1],right_ind[ii]);
        vals = cross_index_merge_wspace(NULL,right_ind[ii]);
        fcut = fiber_cut_ndarray(f,args, dim, ii, ncuts, vals);
    }
    else{
        if (left_ind[ii] == NULL){
            ncuts = right_ind[ii]->n;
        }
        else if (right_ind[ii] == NULL){
            ncuts = left_ind[ii]->n;
        }
        else{
            ncuts = left_ind[ii]->n * right_ind[ii]->n;
        }
        vals = cross_index_merge_wspace(left_ind[ii],right_ind[ii]);
        fcut = fiber_cut_ndarray(f,args, dim, ii, ncuts, vals);
        ncols = ncuts / nrows;
    }
    if (VPREPCORE){
        printf("compute from fibercuts t = %d\n",t);
        for (size_t kk = 0; kk < nrows*ncols; kk++){
            dprint(dim,vals[kk]);
        }
    }
    temp = qmarray_from_fiber_cuts(nrows, ncols,
                    fiber_cut_eval, fcut, fc, sub_type,
                    bd->lb[ii],bd->ub[ii], approx_args);

    //print_qmarray(temp,0,NULL);
    if (VPREPCORE){
        printf("computed!\n");
    }

    free_dd(ncuts,vals); vals = NULL;
    fiber_cut_array_free(ncuts, fcut); fcut = NULL;
    
    
    return temp;
}

/***********************************************************//**
    Cross approximation of a of a dim-dimensional function
    (with adaptation)

    \param[in]     f         - function
    \param[in]     args      - function arguments
    \param[in]     bd        -  bounds on input space
    \param[in,out] ftref     - initial ftrain, changed in func
    \param[in,out] left_ind  - left indices(first element should be NULL)
    \param[in,out] right_ind - right indices(last element should be NULL)
    \param[in]     cargs     - algorithm parameters
    \param[in]     apargs    - approximation arguments

    \return function train decomposition of \f$ f \f$

    \note
    both left and right indices are nested
***************************************************************/
struct FunctionTrain *
ftapprox_cross(double (*f)(double *, void *), void * args, 
               struct BoundingBox * bd,
               struct FunctionTrain * ftref, 
               struct CrossIndex ** left_ind, 
               struct CrossIndex ** right_ind, 
               struct FtCrossArgs * cargs,
               struct FtApproxArgs * apargs)
{
    size_t dim = bd->dim;
    int info;
    size_t nrows, ii, oncore;
    struct Qmarray * temp = NULL;
    struct Qmarray * Q = NULL;
    struct Qmarray * Qt = NULL;
    double * R = NULL;
    size_t * pivind = NULL;
    double * pivx = NULL;
    double diff, diff2, den;
    
    struct FunctionTrain * ft = function_train_alloc(dim);
    memmove(ft->ranks, cargs->ranks, (dim+1)*sizeof(size_t));

    struct FunctionTrain * fti = function_train_copy(ftref);
    struct FunctionTrain * fti2 = NULL;

    int done = 0;
    size_t iter = 0;


    while (done == 0){
        if (cargs->verbose > 0)
            printf("cross iter=%zu \n",iter);
      
        // left right sweep;
        nrows = 1; 
        for (ii = 0; ii < dim-1; ii++){
            //  printf("sub_type ftcross= %d\n",
            //         *(int *)ft_approx_args_getst(apargs,ii));        
            if (cargs->verbose > 1){
                printf(" ............. on left-right sweep (%zu/%zu)\n",ii,dim-1);
            }
            //printf("ii=%zu\n",ii);
            pivind = calloc_size_t(ft->ranks[ii+1]);
            pivx = calloc_double(ft->ranks[ii+1]);
            
            if (VFTCROSS){
                printf( "prepCore \n");
                printf( "left index set = \n");
                print_cross_index(left_ind[ii]);
                printf( "right index set = \n");
                print_cross_index(right_ind[ii]);
            }
            temp = prepCore(ii,nrows,f,args,bd,left_ind,right_ind,
                            cargs,apargs,0);

            if (VFTCROSS == 2){
                printf ("got it \n");
                //print_qmarray(temp,0,NULL);
                struct Qmarray * tempp = qmarray_copy(temp);
                printf("core is \n");
                //print_qmarray(tempp,0,NULL);
                R = calloc_double(temp->ncols * temp->ncols);
                Q = qmarray_householder_simple("QR", temp,R);
                printf("R=\n");
                dprint2d_col(temp->ncols, temp->ncols, R);
//                print_qmarray(Q,0,NULL);

                struct Qmarray * mult = qmam(Q,R,temp->ncols);
                //print_qmarray(Q,3,NULL);
                double difftemp = qmarray_norm2diff(mult,tempp);
                printf("difftemp = %3.15G\n",difftemp);
                qmarray_free(tempp);
                qmarray_free(mult);
            }
            else{
                R = calloc_double(temp->ncols * temp->ncols);
                Q = qmarray_householder_simple("QR", temp,R);
            }

            if (cargs->optargs != NULL){
                /* printf("before\n"); */
                /* struct c3Vector * vec = cargs->optargs->opts[ii]; */
                /* printf("n = %zu",vec->size); */
                info = qmarray_maxvol1d(Q,R,pivind,pivx,cargs->optargs->opts[ii]);
                //printf("after\n");
            }
            else{
                /* printf("SHOULD ABSOLUTELY NOT BE HERE!\n"); */
                info = qmarray_maxvol1d(Q,R,pivind,pivx,NULL);
            }

            if (VFTCROSS){
                printf( " got info=%d\n",info);
                printf("indices and pivots\n");
                iprint_sz(ft->ranks[ii+1],pivind);
                dprint(ft->ranks[ii+1],pivx);
            }

            if (info < 0){
                fprintf(stderr, "no invertible submatrix in maxvol in cross\n");
            }
            if (info > 0){
                fprintf(stderr, " error in qmarray_maxvol1d \n");
                exit(1);
            }

            cross_index_free(left_ind[ii+1]);
            if (ii > 0){
                left_ind[ii+1] =
                    cross_index_create_nested_ind(0,ft->ranks[ii+1],pivind,
                                                  pivx,left_ind[ii]);
            }
            else{
                left_ind[ii+1] = cross_index_alloc(1);
                for (size_t zz = 0; zz < ft->ranks[1]; zz++){
                    cross_index_add_index(left_ind[1],1,&(pivx[zz]));
                }
            }
            
            qmarray_free(ft->cores[ii]); ft->cores[ii]=NULL;
            ft->cores[ii] = qmam(Q,R, temp->ncols);
            nrows = left_ind[ii+1]->n;

            qmarray_free(temp); temp = NULL;
            qmarray_free(Q); Q = NULL;
            free(pivind); pivind =NULL;
            free(pivx); pivx = NULL;
            free(R); R=NULL;

        }
        ii = dim-1;
        if (cargs->verbose > 1){
            printf(" ............. on left-right sweep (%zu/%zu)\n",ii,dim-1);
        }
        qmarray_free(ft->cores[ii]); ft->cores[ii] = NULL;
        ft->cores[ii] = prepCore(ii,cargs->ranks[ii],f,args,bd,
                                 left_ind,right_ind,cargs,apargs,1);

        if (VFTCROSS == 2){
            printf ("got it \n");
            //print_qmarray(ft->cores[ii],0,NULL);
            printf("integral = %G\n",function_train_integrate(ft));
            struct FunctionTrain * tprod = function_train_product(ft,ft);
            printf("prod integral = %G\n",function_train_integrate(tprod));
            printf("norm2 = %G\n",function_train_norm2(ft));
            print_qmarray(tprod->cores[0],0,NULL);
            //print_qmarray(tprod->cores[1],0,NULL);
            function_train_free(tprod);
        }

        if (VFTCROSS){
            printf("\n\n\n Index sets after Left-Right cross\n");
            for (ii = 0; ii < dim; ii++){
                printf("ii = %zu\n",ii);
                printf( "left index set = \n");
                print_cross_index(left_ind[ii]);
                printf( "right index set = \n");
                print_cross_index(right_ind[ii]);
            }
            printf("\n\n\n");
        }
        
        //printf("compute difference \n");
        //printf("norm fti = %G\n",function_train_norm2(fti));
        //printf("norm ft = %G\n",function_train_norm2(ft));
        diff = function_train_relnorm2diff(ft,fti);
        //printf("diff = %G\n",diff);
        //den = function_train_norm2(ft);
        //diff = function_train_norm2diff(ft,fti);
        //if (den > ZEROTHRESH){
        //    diff /= den;
       // }

        if (cargs->verbose > 0){
            den = function_train_norm2(ft);
            printf("...... New FT norm L/R Sweep = %E\n",den);
            printf("...... Error L/R Sweep = %E\n",diff);
        }
        
        if (diff < cargs->epsilon){
            done = 1;
            break;
        }
        
        /* function_train_free(fti); fti=NULL; */
        /* fti = function_train_copy(ft); */
        
        //printf("copied \n");
        //printf("copy diff= %G\n", function_train_norm2diff(ft,fti));

        ///////////////////////////////////////////////////////
        // right-left sweep
        for (oncore = 1; oncore < dim; oncore++){
            
            ii = dim-oncore;

            if (cargs->verbose > 1){
                printf(" ............. on right_left sweep (%zu/%zu)\n",ii,dim-1);
            }

            nrows = ft->ranks[ii]; 

            if (VFTCROSS){
                printf("do prep\n");
                printf( "left index set = \n");
                print_cross_index(left_ind[ii]);
                printf( "right index set = \n");
                print_cross_index(right_ind[ii]);
            }
            //printf("prep core\n");
            temp = prepCore(ii,nrows,f,args,bd,left_ind,right_ind,cargs,apargs,0);
            //printf("prepped core\n");

            R = calloc_double(temp->nrows * temp->nrows);
            Q = qmarray_householder_simple("LQ", temp,R);

            Qt = qmarray_transpose(Q);

            pivind = calloc_size_t(ft->ranks[ii]);
            pivx = calloc_double(ft->ranks[ii]);

            if (cargs->optargs != NULL){
                info = qmarray_maxvol1d(Qt,R,pivind,pivx,cargs->optargs->opts[ii]);
            }
            else{
                info = qmarray_maxvol1d(Qt,R,pivind,pivx,NULL);
            }
            
            if (VFTCROSS){
                printf("got info=%d\n",info);
                printf("indices and pivots\n");
                iprint_sz(ft->ranks[ii],pivind);
                dprint(ft->ranks[ii],pivx);
            }

            //printf("got maxvol\n");
            if (info < 0){
                fprintf(stderr, "noinvertible submatrix in maxvol in rl cross\n");
            }

            if (info > 0){
                fprintf(stderr, " error in qmarray_maxvol1d \n");
                exit(1);
            }
            qmarray_free(Q); Q = NULL;

            //printf("pivx \n");
            //dprint(ft->ranks[ii], pivx);

            Q = qmam(Qt,R, temp->nrows);

            qmarray_free(ft->cores[ii]); ft->cores[ii] = NULL;
            ft->cores[ii] = qmarray_transpose(Q);

            cross_index_free(right_ind[ii-1]);
            if (ii < dim-1){
                //printf("are we really here? oncore=%zu,ii=%zu\n",oncore,ii);
                right_ind[ii-1] =
                    cross_index_create_nested_ind(1,ft->ranks[ii],pivind,
                                                  pivx,right_ind[ii]);
            }
            else{
                //printf("lets update the cross index ii=%zu\n",ii);
                right_ind[ii-1] = cross_index_alloc(1);
                for (size_t zz = 0; zz < ft->ranks[ii]; zz++){
                    cross_index_add_index(right_ind[ii-1],1,&(pivx[zz]));
                }
                //printf("updated\n");
            }

            qmarray_free(temp); temp = NULL;
            qmarray_free(Q); Q = NULL;
            qmarray_free(Qt); Qt = NULL;
            free(pivind);
            free(pivx);
            free(R); R=NULL;

        }

        ii = 0;
        qmarray_free(ft->cores[ii]); ft->cores[ii] = NULL;

        if (cargs->verbose > 1)
            printf(" ............. on right_left sweep (%zu/%zu)\n",ii,dim-1);
        ft->cores[ii] = prepCore(ii,1,f,args,bd,left_ind,right_ind,cargs,apargs,-1);
        if (cargs->verbose > 1)
            printf(" ............. done with right left sweep\n");
 

        if (VFTCROSS){
            printf("\n\n\n Index sets after Right-left cross\n");
            for (ii = 0; ii < dim; ii++){
                printf("ii = %zu\n",ii);
                printf( "left index set = \n");
                print_cross_index(left_ind[ii]);
                printf( "right index set = \n");
                print_cross_index(right_ind[ii]);
            }
            printf("\n\n\n");
        }

   

        diff = function_train_relnorm2diff(ft,fti);
        if (fti2 != NULL){
            diff2 = function_train_relnorm2diff(ft,fti2);
        }
        else{
            diff2 = diff;
        }


        //den = function_train_norm2(ft);
        //diff = function_train_norm2diff(ft,fti);
        //if (den > ZEROTHRESH){
        //    diff /= den;
       // }

        if (cargs->verbose > 0){
            den = function_train_norm2(ft);
            printf("...... New FT norm R/L Sweep = %3.9E\n",den);
            printf("...... Error R/L Sweep = %E,%E\n",diff,diff2);
        }

        if ( (diff2 < cargs->epsilon) || (diff < cargs->epsilon)){
            done = 1;
            break;
        }

        function_train_free(fti2); fti2 = NULL;
        fti2 = function_train_copy(fti);
        function_train_free(fti); fti=NULL;
        fti = function_train_copy(ft);

        iter++;
        if (iter  == cargs->maxiter){
            done = 1;
            break;
        }
    }

    function_train_free(fti); fti=NULL;
    function_train_free(fti2); fti2=NULL;
    return ft;
}

/***********************************************************//**
    Cross approximation of a of a dim-dimensional function
    (with a fixed grid)

    \param[in]     f         - function
    \param[in]     args      - function arguments
    \param[in]     bd        -  bounds on input space
    \param[in,out] ftref     - initial ftrain, changed in func
    \param[in,out] left_ind  - left indices(first element should be NULL)
    \param[in,out] right_ind - right indices(last element should be NULL)
    \param[in]     cargs     - algorithm parameters
    \param[in]     apargs    - approximation arguments

    \return function train decomposition of \f$ f \f$

    \note
    both left and right indices are nested
***************************************************************/
struct FunctionTrain *
ftapprox_cross_grid(double (*f)(double *, void *), void * args,
                    struct BoundingBox * bd,
                    struct FunctionTrain * ftref,
                    struct CrossIndex ** left_ind,
                    struct CrossIndex ** right_ind,
                    struct FtCrossArgs * cargs,
                    struct FtApproxArgs * apargs,
                    struct c3Vector ** grid
    )
{
    size_t dim = bd->dim;
    int info;
    size_t nrows, ii, oncore;
    struct Qmarray * temp = NULL;
    struct Qmarray * Q = NULL;
    struct Qmarray * Qt = NULL;
    double * R = NULL;
    size_t * pivind = NULL;
    double * pivx = NULL;
    double diff, diff2, den;
    
    struct FunctionTrain * ft = function_train_alloc(dim);
    memmove(ft->ranks, cargs->ranks, (dim+1)*sizeof(size_t));

    struct FunctionTrain * fti = function_train_copy(ftref);
    struct FunctionTrain * fti2 = NULL;

    int done = 0;
    size_t iter = 0;

    while (done == 0){
        if (cargs->verbose > 0)
            printf("cross iter=%zu \n",iter);
      
        // left right sweep;
        nrows = 1;
        for (ii = 0; ii < dim-1; ii++){
            if (cargs->verbose > 1){
                printf(" ............. on left-right sweep (%zu/%zu)\n",ii,dim-1);
            }
            //printf("ii=%zu\n",ii);
            pivind = calloc_size_t(ft->ranks[ii+1]);
            pivx = calloc_double(ft->ranks[ii+1]);
            
            if (VFTCROSS){
                printf( "prepCore \n");
                printf( "left index set = \n");
                print_cross_index(left_ind[ii]);
                printf( "right index set = \n");
                print_cross_index(right_ind[ii]);
            }
            temp = prepCore(ii,nrows,f,args,bd,left_ind,right_ind,
                            cargs,apargs,0);

            if (VFTCROSS == 2){
                printf ("got it \n");
                //print_qmarray(temp,0,NULL);
                struct Qmarray * tempp = qmarray_copy(temp);
                printf("core is \n");
                //print_qmarray(tempp,0,NULL);
                R = calloc_double(temp->ncols * temp->ncols);
                Q = qmarray_householder_simple("QR", temp,R);
                printf("R=\n");
                dprint2d_col(temp->ncols, temp->ncols, R);
//                print_qmarray(Q,0,NULL);

                struct Qmarray * mult = qmam(Q,R,temp->ncols);
                //print_qmarray(Q,3,NULL);
                double difftemp = qmarray_norm2diff(mult,tempp);
                printf("difftemp = %3.15G\n",difftemp);
                qmarray_free(tempp);
                qmarray_free(mult);
            }
            else{
                R = calloc_double(temp->ncols * temp->ncols);
                /* Q = qmarray_householder_simple("QR", temp,R); */
                Q = qmarray_householder_simple_grid("QR",temp,R,grid[ii]);
            }

            info = qmarray_maxvol1d(Q,R,pivind,pivx,grid[ii]);

            if (VFTCROSS){
                printf( " got info=%d\n",info);
                printf("indices and pivots\n");
                iprint_sz(ft->ranks[ii+1],pivind);
                dprint(ft->ranks[ii+1],pivx);
            }

            if (info < 0){
                fprintf(stderr, "no invertible submatrix in maxvol in cross\n");
            }
            if (info > 0){
                fprintf(stderr, " error in qmarray_maxvol1d \n");
                exit(1);
            }

            cross_index_free(left_ind[ii+1]);
            if (ii > 0){
                left_ind[ii+1] =
                    cross_index_create_nested_ind(0,ft->ranks[ii+1],pivind,
                                                  pivx,left_ind[ii]);
            }
            else{
                left_ind[ii+1] = cross_index_alloc(1);
                for (size_t zz = 0; zz < ft->ranks[1]; zz++){
                    cross_index_add_index(left_ind[1],1,&(pivx[zz]));
                }
            }
            
            qmarray_free(ft->cores[ii]); ft->cores[ii]=NULL;
            ft->cores[ii] = qmam(Q,R, temp->ncols);
            nrows = left_ind[ii+1]->n;

            qmarray_free(temp); temp = NULL;
            qmarray_free(Q); Q = NULL;
            free(pivind); pivind =NULL;
            free(pivx); pivx = NULL;
            free(R); R=NULL;

        }
        ii = dim-1;
        if (cargs->verbose > 1){
            printf(" ............. on left-right sweep (%zu/%zu)\n",ii,dim-1);
        }
        qmarray_free(ft->cores[ii]); ft->cores[ii] = NULL;
        ft->cores[ii] = prepCore(ii,cargs->ranks[ii],f,args,bd,
                                 left_ind,right_ind,cargs,apargs,1);

        if (VFTCROSS == 2){
            printf ("got it \n");
            //print_qmarray(ft->cores[ii],0,NULL);
            printf("integral = %G\n",function_train_integrate(ft));
            struct FunctionTrain * tprod = function_train_product(ft,ft);
            printf("prod integral = %G\n",function_train_integrate(tprod));
            printf("norm2 = %G\n",function_train_norm2(ft));
            print_qmarray(tprod->cores[0],0,NULL);
            //print_qmarray(tprod->cores[1],0,NULL);
            function_train_free(tprod);
        }

        if (VFTCROSS){
            printf("\n\n\n Index sets after Left-Right cross\n");
            for (ii = 0; ii < dim; ii++){
                printf("ii = %zu\n",ii);
                printf( "left index set = \n");
                print_cross_index(left_ind[ii]);
                printf( "right index set = \n");
                print_cross_index(right_ind[ii]);
            }
            printf("\n\n\n");
        }
        
        //printf("compute difference \n");
        //printf("norm fti = %G\n",function_train_norm2(fti));
        //printf("norm ft = %G\n",function_train_norm2(ft));
        diff = function_train_relnorm2diff(ft,fti);
        //printf("diff = %G\n",diff);
        //den = function_train_norm2(ft);
        //diff = function_train_norm2diff(ft,fti);
        //if (den > ZEROTHRESH){
        //    diff /= den;
       // }

        if (cargs->verbose > 0){
            den = function_train_norm2(ft);
            printf("...... New FT norm L/R Sweep = %E\n",den);
            printf("...... Error L/R Sweep = %E\n",diff);
        }
        
        if (diff < cargs->epsilon){
            done = 1;
            break;
        }
        
        /* function_train_free(fti); fti=NULL; */
        /* fti = function_train_copy(ft); */
        
        //printf("copied \n");
        //printf("copy diff= %G\n", function_train_norm2diff(ft,fti));

        ///////////////////////////////////////////////////////
        // right-left sweep
        for (oncore = 1; oncore < dim; oncore++){
            
            ii = dim-oncore;

            if (cargs->verbose > 1){
                printf(" ............. on right_left sweep (%zu/%zu)\n",ii,dim-1);
            }

            nrows = ft->ranks[ii];

            if (VFTCROSS){
                printf("do prep\n");
                printf( "left index set = \n");
                print_cross_index(left_ind[ii]);
                printf( "right index set = \n");
                print_cross_index(right_ind[ii]);
            }
            //printf("prep core\n");
            temp = prepCore(ii,nrows,f,args,bd,left_ind,right_ind,cargs,apargs,0);
            //printf("prepped core\n");

            R = calloc_double(temp->nrows * temp->nrows);
            Q = qmarray_householder_simple_grid("LQ", temp,R,grid[ii]);
            Qt = qmarray_transpose(Q);
            pivind = calloc_size_t(ft->ranks[ii]);
            pivx = calloc_double(ft->ranks[ii]);

            info = qmarray_maxvol1d(Qt,R,pivind,pivx,grid[ii]);
            
            if (VFTCROSS){
                printf("got info=%d\n",info);
                printf("indices and pivots\n");
                iprint_sz(ft->ranks[ii],pivind);
                dprint(ft->ranks[ii],pivx);
            }

            //printf("got maxvol\n");
            if (info < 0){
                fprintf(stderr, "noinvertible submatrix in maxvol in rl cross\n");
            }

            if (info > 0){
                fprintf(stderr, " error in qmarray_maxvol1d \n");
                exit(1);
            }
            qmarray_free(Q); Q = NULL;

            //printf("pivx \n");
            //dprint(ft->ranks[ii], pivx);

            Q = qmam(Qt,R, temp->nrows);

            qmarray_free(ft->cores[ii]); ft->cores[ii] = NULL;
            ft->cores[ii] = qmarray_transpose(Q);

            cross_index_free(right_ind[ii-1]);
            if (ii < dim-1){
                //printf("are we really here? oncore=%zu,ii=%zu\n",oncore,ii);
                right_ind[ii-1] =
                    cross_index_create_nested_ind(1,ft->ranks[ii],pivind,
                                                  pivx,right_ind[ii]);
            }
            else{
                //printf("lets update the cross index ii=%zu\n",ii);
                right_ind[ii-1] = cross_index_alloc(1);
                for (size_t zz = 0; zz < ft->ranks[ii]; zz++){
                    cross_index_add_index(right_ind[ii-1],1,&(pivx[zz]));
                }
                //printf("updated\n");
            }

            qmarray_free(temp); temp = NULL;
            qmarray_free(Q); Q = NULL;
            qmarray_free(Qt); Qt = NULL;
            free(pivind);
            free(pivx);
            free(R); R=NULL;

        }

        ii = 0;
        qmarray_free(ft->cores[ii]); ft->cores[ii] = NULL;

        if (cargs->verbose > 1)
            printf(" ............. on right_left sweep (%zu/%zu)\n",ii,dim-1);
        ft->cores[ii] = prepCore(ii,1,f,args,bd,left_ind,right_ind,
                                 cargs,apargs,-1);
        if (cargs->verbose > 1)
            printf(" ............. done with right left sweep\n");
 

        if (VFTCROSS){
            printf("\n\n\n Index sets after Right-left cross\n");
            for (ii = 0; ii < dim; ii++){
                printf("ii = %zu\n",ii);
                printf( "left index set = \n");
                print_cross_index(left_ind[ii]);
                printf( "right index set = \n");
                print_cross_index(right_ind[ii]);
            }
            printf("\n\n\n");
        }

        diff = function_train_relnorm2diff(ft,fti);
        if (fti2 != NULL){
            diff2 = function_train_relnorm2diff(ft,fti2);
        }
        else{
            diff2 = diff;
        }

        //den = function_train_norm2(ft);
        //diff = function_train_norm2diff(ft,fti);
        //if (den > ZEROTHRESH){
        //    diff /= den;
       // }

        if (cargs->verbose > 0){
            den = function_train_norm2(ft);
            printf("...... New FT norm R/L Sweep = %3.9E\n",den);
            printf("...... Error R/L Sweep = %E,%E\n",diff,diff2);
        }

        if ( (diff2 < cargs->epsilon) || (diff < cargs->epsilon)){
            done = 1;
            break;
        }

        function_train_free(fti2); fti2 = NULL;
        fti2 = function_train_copy(fti);
        function_train_free(fti); fti=NULL;
        fti = function_train_copy(ft);

        iter++;
        if (iter  == cargs->maxiter){
            done = 1;
            break;
        }
    }

    function_train_free(fti); fti=NULL;
    function_train_free(fti2); fti2=NULL;
    return ft;
}

/***********************************************************//**
   Allocate space for cross approximation arguments
   Set elements to a default value
***************************************************************/
struct FtCrossArgs * ft_cross_args_alloc(size_t dim, size_t start_rank)
{

    struct FtCrossArgs * ftc = malloc(sizeof(struct FtCrossArgs));
    if (ftc == NULL){
        fprintf(stderr,"Failure allocating FtCrossArgs\n");
        exit(1);
    }
    
    ftc->dim = dim;
    ftc->ranks = calloc_size_t(dim+1);
    ftc->ranks[0] = 1;
    for (size_t ii = 1; ii < dim; ii++){
        ftc->ranks[ii] = start_rank;
    }
    ftc->ranks[dim] = 1;
    ftc->epsilon = 1e-10;
    ftc->maxiter = 5;

    ftc->adapt = 1;    
    ftc->epsround = 1e-10;
    ftc->kickrank = 5;
//    ftc->maxiteradapt = 5;
    ftc->maxranks = calloc_size_t(dim-1);
    for (size_t ii = 0; ii < dim-1;ii++){
        // 4 adaptation steps
        ftc->maxranks[ii] = ftc->ranks[ii+1] + ftc->kickrank*4; 
    }

    ftc->verbose = 0;

    ftc->optargs = NULL;

    return ftc;
}

/***********************************************************//**
    Set a reference to optimization arguments
***************************************************************/
void ft_cross_args_set_optargs(struct FtCrossArgs * fca, void * fopt)
{
    fca->optargs = fopt;
}

/***********************************************************//**
    Set the rounding tolerance
***************************************************************/
void ft_cross_args_set_round_tol(struct FtCrossArgs * fca, double epsround)
{
    fca->epsround = epsround;
}
/***********************************************************//**
    Set the kickrank
***************************************************************/
void ft_cross_args_set_kickrank(struct FtCrossArgs * fca, size_t kickrank)
{
    fca->kickrank = kickrank;
}

/***********************************************************//**
    Set the maxiter
***************************************************************/
void ft_cross_args_set_maxiter(struct FtCrossArgs * fca, size_t maxiter)
{
    fca->maxiter = maxiter;
}

/***********************************************************//**
    Turn off adaptation
***************************************************************/
void ft_cross_args_set_no_adaptation(struct FtCrossArgs * fca)
{
    fca->adapt = 0;
}

/***********************************************************//**
    Turn onn adaptation
***************************************************************/
void ft_cross_args_set_adaptation(struct FtCrossArgs * fca)
{
    fca->adapt = 1;
}

/***********************************************************//**
    Set maximum ranks for adaptation
***************************************************************/
void ft_cross_args_set_maxrank_all(struct FtCrossArgs * fca, size_t maxrank)
{
//    fca->maxiteradapt = maxiteradapt;
    for (size_t ii = 0; ii < fca->dim-1; ii++){
        fca->maxranks[ii] = maxrank;
    }
}

/***********************************************************//**
    Set maximum ranks for adaptation per dimension
***************************************************************/
void 
ft_cross_args_set_maxrank_ind(struct FtCrossArgs * fca, size_t maxrank, size_t ind)
{
//    fca->maxiteradapt = maxiteradapt;
    assert (ind < (fca->dim-1));
    fca->maxranks[ind] = maxrank;
}

/***********************************************************//**
    Set the cross approximation tolerance
***************************************************************/
void ft_cross_args_set_cross_tol(struct FtCrossArgs * fca, double tol)
{
    fca->epsilon = tol;
}

/***********************************************************//**
    Set the verbosity level
***************************************************************/
void ft_cross_args_set_verbose(struct FtCrossArgs * fca, int verbose)
{
    fca->verbose = verbose;
}

/***********************************************************//**
    Initialize a baseline cross-approximation args

    \param[in,out] fca - cross approximation arguments 
***************************************************************/
void ft_cross_args_init(struct FtCrossArgs * fca)
{
    fca->ranks = NULL;    
    fca->optargs = NULL;
}

/***********************************************************//**
    Copy cross approximation arguments
***************************************************************/
struct FtCrossArgs * ft_cross_args_copy(struct FtCrossArgs * fca)
{
    if (fca == NULL){
        return NULL;
    }
    else{
        struct FtCrossArgs * f2 = malloc(sizeof(struct FtCrossArgs));
        assert (f2 != NULL);
        f2->dim = fca->dim;
        f2->ranks = calloc_size_t(fca->dim+1);
        memmove(f2->ranks,fca->ranks,(fca->dim+1) * sizeof(size_t));
        f2->epsilon = fca->epsilon;
        f2->maxiter = fca->maxiter;
        f2->adapt = fca->adapt;
        f2->epsround = fca->epsround;
        f2->kickrank = fca->kickrank;
        f2->maxranks = calloc_size_t(fca->dim-1);
        memmove(f2->maxranks,fca->maxranks, (fca->dim-1)*sizeof(size_t));
        f2->verbose = fca->verbose;

        f2->optargs = fca->optargs;
        return f2;
    }
}

/***********************************************************//**
    free cross approximation arguments
***************************************************************/
void ft_cross_args_free(struct FtCrossArgs * fca)
{
    if (fca != NULL){
        free(fca->ranks); fca->ranks = NULL;
        free(fca->maxranks); fca->maxranks = NULL;
        free(fca); fca = NULL;
    }
}

/***********************************************************//**
    An interface for cross approximation of a function

    \param[in] f      - function
    \param[in] args   - function arguments
    \param[in] bds    - bounding box 
    \param[in] xstart - location for first fibers 
                        (if null then middle of domain)
    \param[in] fca    - cross approximation args, 
                        if NULL then default exists
    \param[in] apargs - function approximation arguments 
                        (if null then defaults)

    \return function train decomposition of f

    \note
    Nested indices both left and right
***************************************************************/
struct FunctionTrain *
function_train_cross(double (*f)(double *, void *), void * args, 
                     struct BoundingBox * bds,
                     double ** xstart,
                     struct FtCrossArgs * fca,
                     struct FtApproxArgs * apargs)
{   
    size_t dim = bds->dim;
    
    double ** init_x = NULL;
    struct FtCrossArgs * fcause = NULL;
    struct FtApproxArgs * fapp = NULL;
    
    double * init_coeff = darray_val(dim,1.0/ (double) dim);
    
    size_t ii;

    if (fca != NULL){
        fcause = ft_cross_args_copy(fca);
    }
    else{
        size_t init_rank = 5;
        fcause = ft_cross_args_alloc(dim,init_rank);
    }

    if ( xstart == NULL) {
        init_x = malloc_dd(dim);
        // not used in cross because left->right first
        //init_x[0] = calloc_double(fcause->ranks[1]); 
        
        init_x[0] = linspace(bds->lb[0],bds->ub[0],
                                fcause->ranks[1]);
        for (ii = 0; ii < dim-1; ii++){
            init_x[ii+1] = linspace(bds->lb[ii+1],bds->ub[ii+1],
                                    fcause->ranks[ii+1]);
//            dprint(fcause->ranks[ii+1],init_x[ii+1]);
        }

    }
    else{
        init_x = xstart;
    }
    
    /* dprint(fcause->ranks[1],init_x[0]); */
    /* for (size_t zz = 0; zz < dim-1; zz++){ */
    /*     dprint(fcause->ranks[zz+1],init_x[zz+1]); */
    /* } */
    
    enum poly_type ptype;
    if (apargs != NULL){
        fapp = apargs;
    }
    else{
        ptype = LEGENDRE;
        fapp = ft_approx_args_createpoly(dim,&ptype,NULL);
    }
    
    assert (dim <= 1000);
    struct CrossIndex * isl[1000];
    struct CrossIndex * isr[1000];
    /* cross_index_array_initialize(dim,isl,1,0,NULL,NULL); */
    /* cross_index_array_initialize(dim,isr,0,1,fcause->ranks,init_x); */
    /* struct FunctionTrain * ftref = function_train_constant_d(fapp,1.0,bds); */

    cross_index_array_initialize(dim,isr,0,1,fcause->ranks,init_x);
    cross_index_array_initialize(dim,isl,0,0,fcause->ranks+1,init_x);
    /* exit(1); */
    struct FunctionTrain * ftref = function_train_alloc(dim);
/*     printf("ranks are !!!\n"); */
/*     iprint_sz(dim+1,fcause->ranks); */
    /* printf("create ftref!\n"); */
    for (ii = 0; ii < dim; ii++){

        /* printf("-------------------------\n"); */
        /* printf("ii = %zu\n",ii); */
        /* printf( "left index set = \n"); */
        /* print_cross_index(isl[ii]); */
        /* printf( "right index set = \n"); */
        /* print_cross_index(isr[ii]); */
        /* printf("-------------------------------\n"); */
        size_t nrows = fcause->ranks[ii];
        size_t ncols = fcause->ranks[ii+1];
        ftref->cores[ii] = prepCore(ii,nrows,f,args,bds,isl,isr,fcause,fapp,0);
        ftref->ranks[ii] = nrows;
        ftref->ranks[ii+1] = ncols;
     
    }

    /* exit(1); */
    struct FunctionTrain * ft  = NULL;
    /* printf("start cross!\n"); */
    ft = ftapprox_cross_rankadapt(f,args,bds,ftref,isl,
                                  isr,fcause,fapp);
    
    
    if (xstart == NULL){
        free_dd(dim, init_x); //init_x = NULL;
    }
    if (apargs == NULL){
        ft_approx_args_free(fapp); fapp = NULL;
    }
    ft_cross_args_free(fcause);

    function_train_free(ftref); ftref = NULL;
    for (ii = 0; ii < dim;ii++){
        cross_index_free(isl[ii]);
        cross_index_free(isr[ii]);
    }
    free(init_coeff); init_coeff = NULL;
    return ft;
}


/***********************************************************//**
    An interface for cross approximation of a function
    on an unbounded domain

    \param[in] f        - function
    \param[in] args     - function arguments
    \param[in] xstart   - location for first fibers 
                          (if null then middle of domain)
    \param[in] fcain    - cross approximation args, 
                          if NULL then default exists
    \param[in] apargsin - function approximation arguments 
                          (if null then defaults)
    \param[in] foptin   - optimization arguments                          

    \return function train decomposition of f

    \note
    Nested indices both left and right
***************************************************************/
struct FunctionTrain *
function_train_cross_ub(double (*f)(double *, void *), void * args,
                        size_t dim,
                        double ** xstart,
                        struct FtCrossArgs * fcain,
                        struct FtApproxArgs * apargsin,
                        struct FiberOptArgs * foptin)
{
    enum poly_type ptype = HERMITE;
    struct FtCrossArgs  * fca   = NULL;
    struct FtApproxArgs * aparg = NULL;
    struct FiberOptArgs * fopt  = NULL;
    struct OpeAdaptOpts * aopts = NULL;
    double ** startnodes = NULL;;
    struct c3Vector * optnodes = NULL;

    if (fcain != NULL){
        fca = ft_cross_args_copy(fcain);
    }
    else{
        size_t init_rank = 5;
        fca = ft_cross_args_alloc(dim,init_rank);
    }

    if (foptin != NULL){
        fopt = foptin;
    }
    else{
        // default pivot / fiber locations
        size_t N = 100;
        double * x = linspace(-10.0,10.0,N);
        optnodes = c3vector_alloc(N,x);
        fopt = fiber_opt_args_bf_same(dim,optnodes);
        free(x); x = NULL;
        ft_cross_args_set_maxrank_all(fca,N);
    }
    ft_cross_args_set_optargs(fca,fopt);

    if (apargsin != NULL){
        aparg =apargsin;
    }
    else{
        aopts = ope_adapt_opts_alloc();
        ope_adapt_opts_set_start(aopts,5);
        ope_adapt_opts_set_maxnum(aopts,30);
        ope_adapt_opts_set_coeffs_check(aopts,2);
        ope_adapt_opts_set_tol(aopts,1e-5);
        aparg = ft_approx_args_createpoly(dim,&ptype,aopts);
        
    }

    if (xstart != NULL){
        startnodes = xstart;
    }
    else{
        startnodes = malloc_dd(dim);
        double lbs = -2.0;
        double ubs = 2.0;
        startnodes[0] = calloc_double(fca->ranks[1]); 
        for (size_t ii = 0; ii < dim-1; ii++){
            startnodes[ii+1] = linspace(lbs,ubs,fca->ranks[ii+1]);
        }
    }

    struct BoundingBox * bds = bounding_box_init(dim,-DBL_MAX,DBL_MAX);
    
    struct FunctionTrain * ft = NULL;
    ft = function_train_cross(f,args,bds,startnodes,fca,aparg);
    bounding_box_free(bds); bds = NULL;

    if (foptin == NULL){
        c3vector_free(optnodes); optnodes = NULL;
        fiber_opt_args_free(fopt); fopt = NULL;
    }
    if (apargsin == NULL){
        ope_adapt_opts_free(aopts); aopts = NULL;
        ft_approx_args_free(aparg); aparg = NULL;
    }
    ft_cross_args_free(fca); fca = NULL;
    if (xstart == NULL){
        free_dd(dim,startnodes);
        startnodes = NULL;
    }
    return ft;
}

/***********************************************************//**
    Cross approximation of a of a dim-dimensional function with rank adaptation

    \param[in]     f      - function
    \param[in]     args   - function arguments
    \param[in]     bds    - bounds on input space
    \param[in,out] ftref  - initial ftrain decomposition, 
                            changed in func
    \param[in,out] isl    - left indices (first element should be NULL)
    \param[in,out] isr    - right indices (last element should be NULL)
    \param[in]     fca    - algorithm parameters, 
                            if NULL then default paramaters used
    \param[in]     apargs - function approximation args 

    \return function train decomposition of f

    \note
    both left and right indices are nested
***************************************************************/
struct FunctionTrain *
ftapprox_cross_rankadapt( double (*f)(double *, void *),
                          void * args,
                          struct BoundingBox * bds, 
                          struct FunctionTrain * ftref, 
                          struct CrossIndex ** isl, 
                          struct CrossIndex ** isr,
                          struct FtCrossArgs * fca,
                          struct FtApproxArgs * apargs)
                
{
    size_t dim = bds->dim;
    double eps = fca->epsround;
    size_t kickrank = fca->kickrank;

    struct FunctionTrain * ft = NULL;


    ft = ftapprox_cross(f,args,bds,ftref,isl,isr,fca,apargs);

    if (fca->adapt == 0){
        return ft;
    }
    // printf("found left index\n");
    // print_cross_index(isl[1]);
    // printf("found right index\n");
    //print_cross_index(isr[0]);
    
    //return ft;
    if (fca->verbose > 0){
        printf("done with first cross... rounding\n");
    }
    size_t * ranks_found = calloc_size_t(dim+1);
    memmove(ranks_found,fca->ranks,(dim+1)*sizeof(size_t));
    
    struct FunctionTrain * ftc = function_train_copy(ft);
    struct FunctionTrain * ftr = function_train_round(ft, eps);
    /* printf("rounded ranks = "); iprint_sz(dim+1,ftr->ranks); */
    //struct FunctionTrain * ftr = function_train_copy(ft);
    //return ftr; 
    //printf("DOOONNTT FORGET MEEE HEERREEEE \n");
    int adapt = 0;
    size_t ii;
    for (ii = 1; ii < dim; ii++){
        /* printf("%zu == %zu\n",ranks_found[ii],ftr->ranks[ii]); */
        if (ranks_found[ii] == ftr->ranks[ii]){
            if (fca->ranks[ii] < fca->maxranks[ii-1]){
                adapt = 1;
                size_t kicksize; 
                if ( (ranks_found[ii]+kickrank) <= fca->maxranks[ii-1]){
                    kicksize = kickrank;
                }
                else{
                    kicksize = fca->maxranks[ii-1] -ranks_found[ii];
                }
                fca->ranks[ii] = ranks_found[ii] + kicksize;
                
                // simply repeat the last nodes
                // this is not efficient but I don't have a better
                // idea. Could do it using random locations but
                // I don't want to.
                cross_index_copylast(isr[ii-1],kicksize);
            
                ranks_found[ii] = fca->ranks[ii];
            }
        }
    }

    //printf("adapt here! adapt=%zu\n",adapt);

    size_t iter = 0;
//    double * xhelp = NULL;
    while ( adapt == 1 )
    {
        adapt = 0;
        if (fca->verbose > 0){
            printf("adapting \n");
            printf("Increasing rank\n");
            iprint_sz(ft->dim+1,fca->ranks);
        }
        
        function_train_free(ft); ft = NULL;
        ft = ftapprox_cross(f, args, bds, ftc, isl, isr, fca,apargs);
         
        function_train_free(ftc); ftc = NULL;
        function_train_free(ftr); ftr = NULL;

        //printf("copying\n");
        function_train_free(ftc); ftc = NULL;
        ftc = function_train_copy(ft);
        //printf("rounding\n");
        function_train_free(ftr); ftr = NULL;
        //ftr = function_train_copy(ft);//, eps);
        ftr = function_train_round(ft, eps);
        //printf("done rounding\n");
        for (ii = 1; ii < dim; ii++){
            if (ranks_found[ii] == ftr->ranks[ii]){
                if (fca->ranks[ii] < fca->maxranks[ii-1]){
                    adapt = 1;
                    size_t kicksize; 
                    if ( (ranks_found[ii]+kickrank) <= fca->maxranks[ii-1]){
                        kicksize = kickrank;
                    }
                    else{
                        kicksize = fca->maxranks[ii-1] -ranks_found[ii];
                    }
                    fca->ranks[ii] = ranks_found[ii] + kicksize;
                
            
                    // simply repeat the last nodes
                    // this is not efficient but I don't have a better
                    // idea. Could do it using random locations but
                    // I don't want to.
                    cross_index_copylast(isr[ii-1],kicksize);
            
                    ranks_found[ii] = fca->ranks[ii];
                }
            }
        }

        iter++;
        /* if (iter == fca->maxiteradapt) { */
        /*     adapt = 0; */
        /* } */
        //adapt = 0;

    }
    function_train_free(ft); ft = NULL;
    function_train_free(ftc); ftc = NULL;
    free(ranks_found);
    return ftr;
}

/***********************************************************//**
    Computes the maximum rank of a FT
    
    \param[in] ft - function train

    \return maxrank
***************************************************************/
size_t function_train_maxrank(struct FunctionTrain * ft)
{
    
    size_t maxrank = 1;
    size_t ii;
    for (ii = 0; ii < ft->dim+1; ii++){
        if (ft->ranks[ii] > maxrank){
            maxrank = ft->ranks[ii];
        }
    }
    
    return maxrank;
}

/***********************************************************//**
    Computes the average rank of a FT. Doesn't cound first and last ranks
    
    \param[in] ft - function train

    \return avgrank
***************************************************************/
double function_train_avgrank(struct FunctionTrain * ft)
{
    
    double avg = 0;
    if (ft->dim == 1){
        return 1.0;
    }
    else if (ft->dim == 2){
        return (double) ft->ranks[1];
    }
    else{
        size_t ii;
        for (ii = 1; ii < ft->dim; ii++){
            avg += (double)ft->ranks[ii];
        }
    }
    return (avg / (ft->dim - 1.0));
}

struct vfdata
{   
    double (*f)(double *,size_t,void *);
    size_t ind;
    void * args;
};

double vfunc_eval(double * x, void * args)
{
    struct vfdata * data = args;
    double out = data->f(x,data->ind,data->args);
    return out;
}

/***********************************************************//**
    An interface for cross approximation of a vector-valued function

    \param[in] f      - function
    \param[in] args   - function arguments
    \param[in] nfuncs - function arguments
    \param[in] bds    - bounding box 
    \param[in] xstart - location for first fibers 
                        (if null then middle of domain)
    \param[in] fcain  - cross approximation args, if NULL then 
                        default exists
    \param[in] apargs - function approximation arguments 
                        (if null then defaults)

    \return function train decomposition of f

    \note
    Nested indices both left and right
***************************************************************/
struct FT1DArray *
ft1d_array_cross(double (*f)(double *, size_t, void *), void * args, 
                size_t nfuncs,
                struct BoundingBox * bds,
                double ** xstart,
                struct FtCrossArgs * fcain,
                struct FtApproxArgs * apargs)
{   
    struct vfdata data;
    data.f = f;
    data.args = args;

    struct FT1DArray * fta = ft1d_array_alloc(nfuncs);
    for (size_t ii = 0; ii < nfuncs; ii++){
        data.ind = ii;
        struct FtCrossArgs * fca = ft_cross_args_copy(fcain);
        fta->ft[ii] = function_train_cross(vfunc_eval,&data,bds,xstart,fca,apargs);
        ft_cross_args_free(fca); fca = NULL;
    }
    return fta;
}



//////////////////////////////////////////////////////////////////////
// Blas type interface 1
//

/***********************************************************//**
    Computes 
    \f[
        y \leftarrow \texttt{round}(a x + y, epsilon)
    \f]

    \param a [in] - scaling factor
    \param x [in] - first function train
    \param y [inout] - second function train
    \param epsilon - rounding accuracy (0 for exact)
***************************************************************/
void c3axpy(double a, struct FunctionTrain * x, struct FunctionTrain ** y, 
            double epsilon)
{
    
    struct FunctionTrain * temp = function_train_copy(x);
    function_train_scale(temp,a);
    struct FunctionTrain * z = function_train_sum(temp,*y);
    function_train_free(*y); *y = NULL;
    if (epsilon > 0){
        *y = function_train_round(z,epsilon);
        function_train_free(z); z = NULL;
    }
    else{
        *y = z;
    }
    function_train_free(temp); temp = NULL;
}

/***********************************************************//**
    Computes 
    \f$
        \langle x,y \rangle
    \f$

    \param x [in] - first function train
    \param y [in] - second function train

    \return out - inner product between two function trains
***************************************************************/
double c3dot(struct FunctionTrain * x, struct FunctionTrain * y)
{
    double out = function_train_inner(x,y);
    return out;
}

//////////////////////////////////////////////////////////////////////
// Blas type interface 2
/***********************************************************//**
    Computes 
    \f[
        y \leftarrow alpha \sum_{i=1}^n \texttt{round}(\texttt{product}(A[i*inca],x),epsilon) +
         beta y 
    \f]
    \f[
        y \leftarrow \texttt{round}(y,epsilon)
    \f]
    
    \note
    Also rounds after every summation
***************************************************************/
void c3gemv(double alpha, size_t n, struct FT1DArray * A, size_t inca,
        struct FunctionTrain * x,double beta, struct FunctionTrain * y,
        double epsilon)
{

    size_t ii;
    assert (y != NULL);
    function_train_scale(y,beta);
    

    if (epsilon > 0){
        struct FunctionTrain * runinit = function_train_product(A->ft[0],x);
        struct FunctionTrain * run = function_train_round(runinit,epsilon);
        for (ii = 1; ii < n; ii++){
            struct FunctionTrain * temp = 
                function_train_product(A->ft[ii*inca],x);
            struct FunctionTrain * tempround = function_train_round(temp,epsilon);
            c3axpy(1.0,tempround,&run,epsilon);
            function_train_free(temp); temp = NULL;
            function_train_free(tempround); tempround = NULL;
        }
        c3axpy(alpha,run,&y,epsilon);
        function_train_free(run); run = NULL;
        function_train_free(runinit); runinit = NULL;
    }
    else{
        struct FunctionTrain * run = function_train_product(A->ft[0],x);
        for (ii = 1; ii < n; ii++){
            struct FunctionTrain * temp = 
                function_train_product(A->ft[ii*inca],x);
            c3axpy(1.0,temp,&run,epsilon);
            function_train_free(temp); temp = NULL;
        }
        c3axpy(alpha,run,&y,epsilon);
        function_train_free(run); run = NULL;
    }
       
}

//////////////////////////////////////////////////////////////////////
// Blas type interface 1 (for ft arrays

/***********************************************************//**
    Computes for \f$ i=1 \ldots n\f$
    \f[
        y[i*incy] \leftarrow \texttt{round}(a * x[i*incx] + y[i*incy],epsilon)
    \f]

***************************************************************/
void c3vaxpy(size_t n, double a, struct FT1DArray * x, size_t incx, 
            struct FT1DArray ** y, size_t incy, double epsilon)
{
    size_t ii;
    for (ii = 0; ii < n; ii++){
        c3axpy(a,x->ft[ii*incx],&((*y)->ft[ii*incy]),epsilon);
    }
}

/***********************************************************//**
    Computes for \f$ i=1 \ldots n\f$
    \f[
        z \leftarrow alpha\sum_{i=1}^n y[incy*ii]*x[ii*incx] + beta * z
    \f]

***************************************************************/
void c3vaxpy_arr(size_t n, double alpha, struct FT1DArray * x, 
                size_t incx, double * y, size_t incy, double beta,
                struct FunctionTrain ** z, double epsilon)
{
    assert (*z != NULL);
    function_train_scale(*z,beta);

    size_t ii;
    for (ii = 0; ii < n; ii++){
        c3axpy(alpha*y[incy*ii],x->ft[ii*incx], z,epsilon);
    }
}

/***********************************************************//**
    Computes 
    \f[
        z \leftarrow \texttt{round}(a\sum_{i=1}^n \texttt{round}(x[i*incx]*y[i*incy],epsilon) + beta * z,epsilon)
        
    \f]
    
    \note
    Also rounds after every summation.

***************************************************************/
void c3vprodsum(size_t n, double a, struct FT1DArray * x, size_t incx,
                struct FT1DArray * y, size_t incy, double beta,
                struct FunctionTrain ** z, double epsilon)
{
    assert (*z != NULL);
    function_train_scale(*z,beta);
        
    size_t ii;

    if (epsilon > 0){
        struct FunctionTrain * runinit = 
            function_train_product(x->ft[0],y->ft[0]);
        struct FunctionTrain * run = function_train_round(runinit,epsilon);
        for (ii = 1; ii < n; ii++){
            struct FunctionTrain * tempinit = 
                function_train_product(x->ft[ii*incx],y->ft[ii*incy]);
            struct FunctionTrain * temp = function_train_round(tempinit,epsilon);
            c3axpy(1.0,temp,&run,epsilon);
            function_train_free(temp); temp = NULL;
            function_train_free(tempinit); tempinit = NULL;
        }
        c3axpy(a,run,z,epsilon);
        function_train_free(run); run = NULL;
        function_train_free(runinit); runinit = NULL;
    }
    else{
        struct FunctionTrain * run = 
            function_train_product(x->ft[0],y->ft[0]);
        for (ii = 1; ii < n; ii++){
            struct FunctionTrain * temp = 
                function_train_product(x->ft[ii*incx],y->ft[ii*incy]);
            c3axpy(1.0,temp,&run,epsilon);
            function_train_free(temp); temp = NULL;
        }
        c3axpy(a,run,z,epsilon);
        function_train_free(run); run = NULL;
    }
}

/***********************************************************//**
    Computes for \f$ i = 1 \ldots m \f$
    \f[
        y[i*incy] \leftarrow alpha*\sum_{j=1}^n \texttt{product}(A[i,j*lda],x[j*incx]) + beta * y[i*incy]
    \f]
    
    \note
    Rounds with tolerance epsilon after summation and multiplication. Not shown to avoid clutter.

***************************************************************/
void c3vgemv(size_t m, size_t n, double alpha, struct FT1DArray * A, size_t lda,
        struct FT1DArray * x, size_t incx, double beta, struct FT1DArray ** y,
        size_t incy, double epsilon)
{

    size_t ii;
    assert (*y != NULL);
    ft1d_array_scale(*y,m,incy,beta);
    
    struct FT1DArray * run = ft1d_array_alloc(m); 
    for (ii = 0; ii < m; ii++){
        struct FT1DArray ftatemp;
        ftatemp.ft = A->ft+ii;
        c3vprodsum(n,1.0,&ftatemp,lda,x,incx,0.0,&(run->ft[ii*incy]),epsilon);
    }
    c3vaxpy(m,alpha,run,1,y,incy,epsilon);

    ft1d_array_free(run); run = NULL;
}

/***********************************************************//**
    Computes for \f$ i = 1 \ldots m \f$
    \f[
        y[i*incy] \leftarrow alpha*\sum_{j=1}^n \texttt{product}(A[i,j],B[j*incb]) + beta * y[i*incy]
    \f]
    
    \note
    Rounds with tolerance epsilon after summation and multiplication. Not shown to avoid clutter.
    trans = 0 means not to transpose A
    trans = 1 means to transpose A
***************************************************************/
void c3vgemv_arr(int trans, size_t m, size_t n, double alpha, double * A, 
        size_t lda, struct FT1DArray * B, size_t incb, double beta,
        struct FT1DArray ** y, size_t incy, double epsilon)
{
    size_t ii;
    assert (*y != NULL);
    ft1d_array_scale(*y,m,incy,beta);
        
    if (trans == 0){
        for (ii = 0; ii < m; ii++){
            c3vaxpy_arr(n,alpha,B,incb,A+ii,lda,beta,&((*y)->ft[ii*incy]),epsilon);
        }
    }
    else if (trans == 1){
        for (ii = 0; ii < m; ii++){
            c3vaxpy_arr(n,alpha,B,incb,A+ii*lda,1,beta,&((*y)->ft[ii*incy]),epsilon);
        }
    }
}



