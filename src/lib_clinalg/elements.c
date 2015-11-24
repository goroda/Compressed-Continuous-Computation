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

/** \file elements.c
 * Provides routines for initializing / using elements of continuous linear algebra
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "stringmanip.h"
#include "array.h"
#include "linalg.h"
#include "elements.h"
#include "algs.h"

// helper functions for function_train_initsum2
struct wrap_spec_func
{
    double (*f)(double, size_t, void *);
    size_t which;
    void * args;
};
double eval_spec_func(double x, void * args)
{
    struct wrap_spec_func * spf = args;
    return spf->f(x,spf->which,spf->args);
}


/***********************************************************//**
    Allocate space for a quasimatrix

    \param n [in] - number of columns of quasimatrix

    \return qm -  quasimatrix
***************************************************************/
struct Quasimatrix * 
quasimatrix_alloc(size_t n){
    
    struct Quasimatrix * qm;
    if ( NULL == (qm = malloc(sizeof(struct Quasimatrix)))){
        fprintf(stderr, "failed to allocate memory for quasimatrix.\n");
        exit(1);
    }
    qm->n = n;
    if ( NULL == (qm->funcs = malloc(n * sizeof(struct GenericFunction *)))){
        fprintf(stderr, "failed to allocate memory for quasimatrix.\n");
        exit(1);
    }
    size_t ii;
    for (ii = 0; ii < n; ii++){
        qm->funcs[ii] = NULL;
    }
    return qm;
}

/***********************************************************//**
    Create a quasimatrix by approximating 1d functions

    \param n [in] - number of columns of quasimatrix
    \param funcs [in] - functions
    \param args [in] - extra arguments to each function
    \param fc [in] - function class of each column
    \param st [in] - sub_type of each column
    \param lb [in] - lower bound of inputs to functions
    \param ub [in] - upper bound of inputs to functions
    \param aopts [in] - approximation options

    \return qm - quasimatrix
***************************************************************/
struct Quasimatrix * 
quasimatrix_approx1d(size_t n, double (**funcs)(double, void *),
                    void ** args, enum function_class fc, void * st, double lb,
                    double ub, void * aopts)
{
    struct Quasimatrix * qm = quasimatrix_alloc(n);
    size_t ii;
    for (ii = 0; ii < n; ii++){
        qm->funcs[ii] = generic_function_approximate1d(funcs[ii], args[ii], 
                                    fc, st, lb, ub, aopts);
    }
    return qm;
}

/***********************************************************//**
    Create a quasimatrix from a fiber_cuts array

    \param n [in] - number of columns of quasimatrix
    \param f [in] - function
    \param fcut [in] - array of fiber cuts
    \param fc [in] - function class of each column
    \param sub_type [in] - sub_type of each column
    \param lb [in] - lower bound of inputs to functions
    \param ub [in] - upper bound of inputs to functions
    \param aopts [in] - approximation options

    \return qm - quasimatrix
***************************************************************/
struct Quasimatrix * 
quasimatrix_approx_from_fiber_cuts(size_t n, 
                    double (*f)(double, void *), struct FiberCut ** fcut, 
                    enum function_class fc, void * sub_type, double lb,
                    double ub, void * aopts)
{
    struct Quasimatrix * qm = quasimatrix_alloc(n);
    size_t ii;
    for (ii = 0; ii < n; ii++){
        qm->funcs[ii] = generic_function_approximate1d(f, fcut[ii], 
                                    fc, sub_type, lb, ub, aopts);
    }
    return qm;
}

/***********************************************************//**
    Allocate space for and initialize a quasimatrix

    \param fdim [in] - dimension of each function in the quasimatrix
    \param n [in] - number of columns of quasimatrix
    \param fc [in] - function class of each column
    \param sub_type [in] - sub_type of each column
    \param f [in] - functions which make up each column
    \param fargs [in] - function arguments

    \return qm - quasimatrix
***************************************************************/
struct Quasimatrix * 
quasimatrix_init(size_t fdim, size_t n, enum function_class * fc,
            void ** sub_type, void ** f, void ** fargs)
{
    struct Quasimatrix * qm = quasimatrix_alloc(n);
    size_t ii;
    for (ii = 0; ii < n; ii++){
        qm->funcs[ii] = generic_function_alloc(fdim, fc[ii],sub_type[ii]);
        qm->funcs[ii]->f = f[ii];
        if (fargs == NULL){
            qm->funcs[ii]->fargs = NULL;
        }
        else{
            qm->funcs[ii]->fargs = fargs[ii];
        }
    }

    return qm;
}

/***********************************************************//**
    Free memory allocated to quasimatrix

    \param qm [in] - quasimatrix
***************************************************************/
void quasimatrix_free(struct Quasimatrix * qm){
    
    if (qm != NULL){
        size_t ii = 0;
        for (ii = 0; ii < qm->n; ii++){
            if (qm->funcs[ii] != NULL)
                generic_function_free(qm->funcs[ii]);
        }
        free(qm->funcs);
        free(qm);qm=NULL;
    }
}

/***********************************************************//**
    copy a quasimatrix

    \param qm [in] - quasimatrix to copy

    \return qmo  - copied quasimatrix
***************************************************************/
struct Quasimatrix * quasimatrix_copy(struct Quasimatrix * qm)
{
    struct Quasimatrix * qmo = quasimatrix_alloc(qm->n);
    size_t ii;
    for (ii = 0; ii < qm->n; ii++){
        qmo->funcs[ii] = generic_function_copy(qm->funcs[ii]);
    }

    return qmo;
}


/***********************************************************//**
    Extract a copy of a quasimatrix from a column of a qmarray

    \param qma [in] - qmarray
    \param col [in] - column to copy

    \return qm [out] - quasimatrix
***************************************************************/
struct Quasimatrix *
qmarray_extract_column(struct Qmarray * qma, size_t col)
{

    struct Quasimatrix * qm = quasimatrix_alloc(qma->nrows);
    size_t ii;
    for (ii = 0; ii < qma->nrows; ii++){
        qm->funcs[ii] = generic_function_copy(qma->funcs[col*qma->nrows+ii]);
    }
    return qm;
}

/***********************************************************//**
    Extract a copy of the first nkeep columns of a qmarray

    \param a [in] - qmarray from which to extract columns
    \param nkeep [in] : number of columns to extract

    \return qm - qmarray
***************************************************************/
struct Qmarray * qmarray_extract_ncols(struct Qmarray * a, size_t nkeep)
{
    
    struct Qmarray * qm = qmarray_alloc(a->nrows,nkeep);
    size_t ii,jj;
    for (ii = 0; ii < nkeep; ii++){
        for (jj = 0; jj < a->nrows; jj++){
            qm->funcs[ii*a->nrows+jj] = generic_function_copy(a->funcs[ii*a->nrows+jj]);
        }
    }
    return qm;
}


/***********************************************************//**
    Extract a copy of a quasimatrix from a row of a qmarray

    \param qma [in] - qmarray
    \param row [in] - row to copy

    \return qm - quasimatrix
***************************************************************/
struct Quasimatrix *
qmarray_extract_row(struct Qmarray * qma, size_t row)
{

    struct Quasimatrix * qm = quasimatrix_alloc(qma->ncols);
    size_t ii;
    for (ii = 0; ii < qma->ncols; ii++){
        qm->funcs[ii] = generic_function_copy(qma->funcs[ii*qma->nrows+row]);
    }
    return qm;
}


/***********************************************************//**
    Set the column of a quasimatrix array to a given quasimatrix by copying

    \param qma (IN/OUT) : qmarray
    \param col [in] - column to set
    \param qm (IN)  : quasimatrix to copy
***************************************************************/
void
qmarray_set_column(struct Qmarray * qma, size_t col, 
                        struct Quasimatrix * qm)
{
    size_t ii;
    for (ii = 0; ii < qma->nrows; ii++){
        generic_function_free(qma->funcs[col*qma->nrows+ii]);
        qma->funcs[col*qma->nrows+ii] = 
            generic_function_copy(qm->funcs[ii]);
    }
}

/***********************************************************//**
    Function qmarray_set_column_gf

    Purpose: Set the column of a quasimatrix array 
             to a given array of generic functions

    \param qma [inout] - qmarray
    \param col [in] - column to set
    \param gf [in] - array of generic functions

***************************************************************/
void
qmarray_set_column_gf(struct Qmarray * qma, size_t col, 
                        struct GenericFunction ** gf)
{
    size_t ii;
    for (ii = 0; ii < qma->nrows; ii++){
        generic_function_free(qma->funcs[col*qma->nrows+ii]);
        qma->funcs[col*qma->nrows+ii] = generic_function_copy(gf[ii]);
    }
}


/***********************************************************//**
    Set the row of a quasimatrix array to a given quasimatrix by copying

    \param qma [inout] - qmarray
    \param row [in] - row to set
    \param qm [in] - quasimatrix to copy
***************************************************************/
void
qmarray_set_row(struct Qmarray * qma, size_t row, struct Quasimatrix * qm)
{
    size_t ii;
    for (ii = 0; ii < qma->ncols; ii++){
        generic_function_free(qma->funcs[ii*qma->nrows+row]);
        qma->funcs[ii*qma->nrows+row] = generic_function_copy(qm->funcs[ii]);
    }
}


/***********************************************************//**
    Serialize a quasimatrix

    \param ser [inout] - stream to serialize to
    \param qm [in] - quasimatrix
    \param totSizeIn [inout] if NULL then serialize, if not NULL then return size;

    \return ptr - ser shifted by number of bytes
***************************************************************/
unsigned char * 
quasimatrix_serialize(unsigned char * ser, struct Quasimatrix * qm, 
                    size_t *totSizeIn)
{

    // n -> func -> func-> ... -> func
    size_t ii;
    size_t totSize = sizeof(size_t);
    size_t size_temp;
    for (ii = 0; ii < qm->n; ii++){
        serialize_generic_function(NULL,qm->funcs[ii],&size_temp);
        totSize += size_temp;
    }
    if (totSizeIn != NULL){
        *totSizeIn = totSize;
        return ser;
    }
    
    unsigned char * ptr = ser;
    ptr = serialize_size_t(ptr, qm->n);
    for (ii = 0; ii < qm->n; ii++){
        ptr = serialize_generic_function(ptr, qm->funcs[ii],NULL);
    }
    return ptr;
}

/***********************************************************//**
    Deserialize a quasimatrix

    \param ser [in] - serialized quasimatrix
    \param qm [inout] - quasimatrix

    \return ptr - shifted ser after deserialization
***************************************************************/
unsigned char *
quasimatrix_deserialize(unsigned char * ser, struct Quasimatrix ** qm)
{
    unsigned char * ptr = ser;

    size_t n;
    ptr = deserialize_size_t(ptr, &n);
    *qm = quasimatrix_alloc(n);

    size_t ii;
    for (ii = 0; ii < n; ii++){
        ptr = deserialize_generic_function(ptr, &((*qm)->funcs[ii]));
    }
    
    return ptr;
}


/***********************************************************//**
    Generate a quasimatrix with orthonormal columns

    \param fc [in] - function class
    \param st [in] - function class sub_type
    \param n [in] -  number of columns
    \param lb [in] - lower bound on functions
    \param ub [in] - upper bound on functions

    \return qm - quasimatrix with orthonormal columns
***************************************************************/
struct Quasimatrix *
quasimatrix_orth1d(enum function_class fc, void * st, size_t n, 
                            double lb, double ub)
{
    struct Interval ob;
    ob.lb = lb;
    ob.ub = ub;

    struct Quasimatrix * qm = quasimatrix_alloc(n);
    generic_function_array_orth(n, fc, st, qm->funcs, (void *)(&ob));
    return qm;
}

/***********************************************************//**
    Find the absolute maximum element of a quasimatrix of 1d functions

    \param qm [in] - quasimatrix
    \param absloc [inout] : location (row) of maximum
    \param absval [inout] : value of maximum

    \return col - column of maximum element
***************************************************************/
size_t 
quasimatrix_absmax(struct Quasimatrix * qm, 
                        double * absloc, double * absval)
{   
    size_t col = 0;
    size_t ii;
    double temp_loc;
    double temp_max;
    *absval = generic_function_absmax(qm->funcs[0], absloc) ;
    for (ii = 1; ii < qm->n; ii++){
        temp_max = generic_function_absmax(qm->funcs[ii], &temp_loc);
        if (temp_max > *absval){
            col = ii;
            *absval = temp_max;
            *absloc = temp_loc;
        }
    }
    return col;
}

/***********************************************************//**
    Allocate memory for a skeleton decomposition

    \param r [in] - rank

    \return skd - skeleton decomposition
***************************************************************/
struct SkeletonDecomp * skeleton_decomp_alloc(size_t r)
{

    struct SkeletonDecomp * skd;
    if ( NULL == (skd = malloc(sizeof(struct SkeletonDecomp)))){
        fprintf(stderr, 
                "failed to allocate memory for skeleton decomposition.\n");
        exit(1);
    }
    skd->r = r;
    skd->xqm = quasimatrix_alloc(r);
    skd->yqm = quasimatrix_alloc(r);
    skd->skeleton = calloc_double(r*r);
    return skd;
}

/***********************************************************//**
    Copy a skeleton decomposition

    \param skd [in] - skeleton decomposition
    \return snew  - copied skeleton decomposition
***************************************************************/
struct SkeletonDecomp * skeleton_decomp_copy(struct SkeletonDecomp * skd)
{
    struct SkeletonDecomp * snew;
    if ( NULL == (snew = malloc(sizeof(struct SkeletonDecomp)))){
        fprintf(stderr, 
                "failed to allocate memory for skeleton decomposition.\n");
        exit(1);
    }
    double * eye = calloc_double(skd->r * skd->r);
    size_t ii;
    for (ii = 0; ii < skd->r; ii++) eye[ii*skd->r +ii] = 1.0;
    snew->r = skd->r;
    snew->xqm = qmm(skd->xqm,eye, skd->r);
    snew->yqm = qmm(skd->yqm,eye, skd->r);
    snew->skeleton = calloc_double(skd->r * skd->r);

    /*
    printf("skeleton = \n");
    dprint2d_col(skd->r,skd->r, skd->skeleton);
    */
    memmove(snew->skeleton,skd->skeleton,skd->r *skd->r* sizeof(double));
    
    free(eye);eye=NULL;
    return snew;
}

/***********************************************************//**
    Free memory allocated to skeleton decomposition

    \param skd [inout] - skeleton decomposition
***************************************************************/
void skeleton_decomp_free(struct SkeletonDecomp * skd)
{
    if (skd != NULL){
        quasimatrix_free(skd->xqm);
        quasimatrix_free(skd->yqm);
        free(skd->skeleton);
        free(skd);skd=NULL;
    }
}

/***********************************************************//**
    Allocate and initialize skeleton decomposition 
    with a set of pivots and a given approximation

    \param f [in] - function to approximate
    \param args [in] - function arguments
    \param bounds [in] - bounds on function
    \param fc [in] - function classes of approximation (2)
    \param sub_type [in] - sub_type of approximation (2)
    \param r [in] - rank
    \param pivx [in] - x pivots
    \param pivy [in] - y pivots
    \param approx_args [in] - approximation arguments (2);

    \return skd - skeleton decomposition
***************************************************************/
struct SkeletonDecomp * 
skeleton_decomp_init2d_from_pivots(double (*f)(double,double,void *),
                void * args, struct BoundingBox * bounds,
                enum function_class * fc, void ** sub_type,
                size_t r, double * pivx, double * pivy, void ** approx_args)
{
    
    struct SkeletonDecomp * skd = skeleton_decomp_alloc(r);

    struct FiberCut ** fx;  
    struct FiberCut ** fy;

    fx = fiber_cut_2darray(f,args,0, r, pivy);
    quasimatrix_free(skd->xqm);
    skd->xqm = quasimatrix_approx_from_fiber_cuts(
            r, fiber_cut_eval2d, fx, fc[0], sub_type[0],
            bounds->lb[0],bounds->ub[0], approx_args[0]);
    fiber_cut_array_free(r, fx);

    fy = fiber_cut_2darray(f,args,1, r, pivx);
    quasimatrix_free(skd->yqm);
    skd->yqm = quasimatrix_approx_from_fiber_cuts(
            r, fiber_cut_eval2d, fy, fc[1], sub_type[1],
            bounds->lb[1],bounds->ub[1], approx_args[1]);
    fiber_cut_array_free(r, fy);

    
    size_t ii,jj;
    double * cmat = calloc_double(r*r);
    for (ii = 0; ii < r; ii++){
        for (jj = 0; jj < r; jj++){
            cmat[ii * r + jj] = f(pivx[jj],pivy[ii], args);
        }
    }

    /*
    printf("cmat = ");
    dprint2d_col(r,r,cmat);
    */
    pinv(r,r,r,cmat,skd->skeleton,1e-15);

    free(cmat);cmat=NULL;
    return skd;
}

/***********************************************************//**
    Evaluate a skeleton decomposition

    \param skd [in] - skeleton decomposition
    \param x [in] - x-location to evaluate
    \param y [in] - y-location to evaluate

    \return out - evaluation
***************************************************************/
double skeleton_decomp_eval(struct SkeletonDecomp * skd, double x, double y)
{
    double out = 0.0;
    struct Quasimatrix * t1 = qmm(skd->xqm, skd->skeleton, skd->r);
    
    /*
    printf("T11111111111111\n");
    printf("skeleton = \n");
    dprint2d_col(skd->r,skd->r, skd->skeleton);
    print_quasimatrix(t1,0,NULL);
    print_quasimatrix(skd->xqm,0,NULL);
    printf("COOOOL\n");
    */
    size_t ii;
    for (ii = 0; ii < t1->n; ii++){
        out += generic_function_1d_eval(t1->funcs[ii],x) * 
                generic_function_1d_eval(skd->yqm->funcs[ii],y);
    }
    quasimatrix_free(t1);
    return out;
}


/////////////////////////////////////////////////////////
// qm_array

/***********************************************************//**
    Allocate space for a qmarray

    \param nrows [in] - number of rows of quasimatrix
    \param ncols [in] - number of cols of quasimatrix

    \return qm - qmarray
***************************************************************/
struct Qmarray * qmarray_alloc(size_t nrows, size_t ncols){
    
    struct Qmarray * qm = NULL;
    if ( NULL == (qm = malloc(sizeof(struct Qmarray)))){
        fprintf(stderr, "failed to allocate memory for qmarray.\n");
        exit(1);
    }
    qm->nrows = nrows;
    qm->ncols = ncols;
    if ( NULL == (qm->funcs=malloc(nrows*ncols*sizeof(struct GenericFunction *)))){
        fprintf(stderr, "failed to allocate memory for qmarray.\n");
        exit(1);
    }
    size_t ii;
    for (ii = 0; ii < nrows*ncols; ii++){
        qm->funcs[ii] = NULL;
    }
    return qm;
}


/***********************************************************//**
    Create a qmarray by approximating 1d functions

    \param nrows [in] : number of rows of quasimatrix
    \param ncols [in] : number of cols of quasimatrix
    \param funcs [in] : functions (nrows*ncols)
    \param args [in] : extra arguments to each function
    \param fc [in] : function class of each column
    \param st [in] : sub_type of each column
    \param lb [in] : lower bound of inputs to functions
    \param ub [in] : upper bound of inputs to functions
    \param aopts [in] : approximation options

    \return qm - qmarray
***************************************************************/
struct Qmarray * 
qmarray_approx1d(size_t nrows, size_t ncols,
                    double (**funcs)(double, void *),
                    void ** args, enum function_class fc, void * st, double lb,
                    double ub, void * aopts)
{
    struct Qmarray * qm = qmarray_alloc(nrows,ncols);
    size_t ii;
    for (ii = 0; ii < nrows*ncols; ii++){
        qm->funcs[ii] = generic_function_approximate1d(funcs[ii], args[ii], 
                                    fc, st, lb, ub, aopts);
    }
    return qm;
}

/********************************************************//**
*    Create a qmarray consisting of pseudo-random orth poly expansion
*   
*   \param nrows [in] - number of rows
*   \param ncols [in] - number of columns
*   \param maxorder [in] - maximum order of the polynomial
*   \param lower [in] - lower bound of input
*   \param upper [in] - upper bound of input
*
*   \return qm - qmarray
************************************************************/
struct Qmarray *
qmarray_poly_randu(size_t nrows, size_t ncols, 
    size_t maxorder, double lower, double upper)
{
    struct Qmarray * qm = qmarray_alloc(nrows,ncols);
    size_t ii;
    for (ii = 0; ii < nrows*ncols; ii++){
        qm->funcs[ii] = generic_function_poly_randu(maxorder,lower,upper);
    }
    return qm;
}


/***********************************************************//**
    Create a qmarray from a fiber_cuts array

    \param nrows [in] - number of rows of qmarray
    \param ncols [in] - number of columns of qmarray
    \param f [in] - functions
    \param fcut [in] - array of fiber cuts
    \param fc [in] - function class of each column
    \param sub_type [in] - sub_type of each column
    \param lb [in] - lower bound of inputs to functions
    \param ub [in] - upper bound of inputs to functions
    \param aopts [in] - approximation options

    \return qm - qmarray
***************************************************************/
struct Qmarray * 
qmarray_from_fiber_cuts(size_t nrows, size_t ncols, 
                    double (*f)(double, void *),struct FiberCut ** fcut, 
                    enum function_class fc, void * sub_type, double lb,
                    double ub, void * aopts)
{
    struct Qmarray * qm = qmarray_alloc(nrows,ncols);
    size_t ii;
    for (ii = 0; ii < nrows*ncols; ii++){
        qm->funcs[ii] = generic_function_approximate1d(f, fcut[ii], 
                                    fc, sub_type, lb, ub, aopts);
    }
    return qm;
}

/***********************************************************//**
    Generate a qmarray with orthonormal columns consisting of
    one dimensional functions

    \param fc [in] - function class
    \param st [in] - function class sub_type
    \param nrows [in] -  number of rows
    \param ncols [in] -  number of columns
    \param lb [in] - lower bound on 1d functions
    \param ub [in] - upper bound on 1d functions

    \return qm - qmarray with orthonormal columns

    \note
        - Not super efficient because of copies
***************************************************************/
struct Qmarray *
qmarray_orth1d_columns(enum function_class fc, void * st, size_t nrows,
                            size_t ncols, double lb, double ub)
{
    struct Interval ob;
    ob.lb = lb;
    ob.ub = ub;

    struct Qmarray * qm = qmarray_alloc(nrows,ncols);
    
    struct GenericFunction ** funcs = NULL;
    if ( NULL == (funcs = malloc(ncols*sizeof(struct GenericFunction *)))){
        fprintf(stderr, "failed to allocate memory for quasimatrix.\n");
        exit(1);
    }
    size_t ii, jj,kk;
    for (ii = 0; ii < ncols; ii++){
        funcs[ii] = NULL;
    }
    generic_function_array_orth(ncols, fc, st, funcs, &ob);
    
    struct GenericFunction * zero = generic_function_constant(0.0,fc,st,lb,ub,NULL);
    
    size_t onnon = 0;
    size_t onorder = 0;
    for (jj = 0; jj < ncols; jj++){
        qm->funcs[jj*nrows+onnon] = generic_function_copy(funcs[onorder]);
        for (kk = 0; kk < onnon; kk++){
            qm->funcs[jj*nrows+kk] = generic_function_copy(zero);
        }
        for (kk = onnon+1; kk < nrows; kk++){
            qm->funcs[jj*nrows+kk] = generic_function_copy(zero);
        }
        onnon = onnon+1;
        if (onnon == nrows){
            generic_function_free(funcs[onorder]);
            funcs[onorder] = NULL;
            onorder = onorder+1;
            onnon = 0;
        }
    }
    //printf("max order cols = %zu\n",onorder);
    
    generic_function_free(zero); zero = NULL;
    free(funcs); funcs=NULL;

    return qm;
}

/***********************************************************//**
    Generate a qmarray with orthonormal rows

    \param fc [in] - function class
    \param st [in] - function class sub_type
    \param nrows [in] -  number of rows
    \param ncols [in] -  number of columns
    \param lb [in] - lower bound on 1d functions
    \param ub [in] - upper bound on 1d functions

    \return qm (OUT) - qmarray with orthonormal columns

    \note
        Not super efficient because of copies
***************************************************************/
struct Qmarray *
qmarray_orth1d_rows(enum function_class fc, void * st, size_t nrows,
                            size_t ncols, double lb, double ub)
{
    struct Interval ob;
    ob.lb = lb;
    ob.ub = ub;

    struct Qmarray * qm = qmarray_alloc(nrows,ncols);
    
    struct GenericFunction ** funcs = NULL;
    if ( NULL == (funcs = malloc(nrows*sizeof(struct GenericFunction *)))){
        fprintf(stderr, "failed to allocate memory for quasimatrix.\n");
        exit(1);
    }
    size_t ii, jj,kk;
    for (ii = 0; ii < nrows; ii++){
        funcs[ii] = NULL;
    }
    //printf("wwwhat\n");
    generic_function_array_orth(nrows, fc, st, funcs, &ob);
    //printf("wwwhere\n");
    
     struct GenericFunction * zero = generic_function_constant(0.0,fc,st,lb,ub,NULL);
    
    size_t onnon = 0;
    size_t onorder = 0;
    for (jj = 0; jj < nrows; jj++){
        qm->funcs[onnon*nrows+jj] = generic_function_copy(funcs[onorder]);
        for (kk = 0; kk < onnon; kk++){
            qm->funcs[kk*nrows+jj] = generic_function_copy(zero);
        }
        for (kk = onnon+1; kk < ncols; kk++){
            qm->funcs[kk*nrows+jj] = generic_function_copy(zero);
        }
        onnon = onnon+1;
        if (onnon == ncols){
            generic_function_free(funcs[onorder]);
            funcs[onorder] = NULL;
            onorder = onorder+1;
            onnon = 0;
        }
    }
    //printf("max order rows = %zu\n",onorder);
    
    generic_function_free(zero); zero = NULL;
    free(funcs); funcs=NULL;

    return qm;
}


/***********************************************************//**
    copy qmarray

    \param qm [in] - qmarray

    \return qmo [out] 
***************************************************************/
struct Qmarray * qmarray_copy(struct Qmarray * qm)
{
    struct Qmarray * qmo = qmarray_alloc(qm->nrows,qm->ncols);
    size_t ii;
    for (ii = 0; ii < qm->nrows*qm->ncols; ii++){
        qmo->funcs[ii] = generic_function_copy(qm->funcs[ii]);
    }

    return qmo;
}


/***********************************************************//**
    Free memory allocated to qmarray 

    \param qm (IN) - qmarray
***************************************************************/
void qmarray_free(struct Qmarray * qm){
    
    if (qm != NULL){
        size_t ii = 0;
        for (ii = 0; ii < qm->nrows*qm->ncols; ii++){
            generic_function_free(qm->funcs[ii]);
        }
        free(qm->funcs);
        free(qm);qm=NULL;
    }
}

/***********************************************************//**
    Serialize a qmarray

    \param ser [inout] - stream to serialize to
    \param qma [in] - quasimatrix array
    \param totSizeIn [inout] if NULL then serialize, if not NULL
                             then return size;

    \return ptr [out] - ser shifted by number of ytes
***************************************************************/
unsigned char * 
qmarray_serialize(unsigned char * ser, struct Qmarray * qma, 
                    size_t *totSizeIn)
{

    // nrows -> ncols -> func -> func-> ... -> func
    size_t ii;
    size_t totSize = 2*sizeof(size_t);
    size_t size_temp;
    for (ii = 0; ii < qma->nrows * qma->ncols; ii++){
        serialize_generic_function(NULL,qma->funcs[ii],&size_temp);
        totSize += size_temp;
    }
    if (totSizeIn != NULL){
        *totSizeIn = totSize;
        return ser;
    }
    
    //printf("in serializing qmarray\n");
    unsigned char * ptr = ser;
    ptr = serialize_size_t(ptr, qma->nrows);
    ptr = serialize_size_t(ptr, qma->ncols);
    for (ii = 0; ii < qma->nrows * qma->ncols; ii++){
       //printf("on function number (%zu/%zu)\n",ii,qma->nrows * qma->nrows);
       // print_generic_function(qma->funcs[ii],3,NULL);
        ptr = serialize_generic_function(ptr, qma->funcs[ii],NULL);
    }
    return ptr;
}

/***********************************************************//**
    Deserialize a qmarray

    \param ser [in] - serialized qmarray
    \param qma [inout] - qmarray

    \return ptr - shifted ser after deserialization
***************************************************************/
unsigned char *
qmarray_deserialize(unsigned char * ser, struct Qmarray ** qma)
{
    unsigned char * ptr = ser;

    size_t nrows, ncols;
    ptr = deserialize_size_t(ptr, &nrows);
    ptr = deserialize_size_t(ptr, &ncols);
    *qma = qmarray_alloc(nrows, ncols);

    size_t ii;
    for (ii = 0; ii < nrows * ncols; ii++){
        ptr = deserialize_generic_function(ptr, &((*qma)->funcs[ii]));
    }
    
    return ptr;
}

////////////////////////////////////////////////////////////////////
// function_train 
//
/***********************************************************//**
    Create the arguments to give to use for approximation
    in the function train. Specifically, legendre polynomials
    for all dimensions

    \param dim [in] - dimension of function train
    \param ptype [in] - type of polynomial
    \param aopts [in] - arguments for creating the approximation (could be NULL)

    \return fargs -- approximation arguments
***************************************************************/
struct FtApproxArgs * ft_approx_args_createpoly(size_t dim, 
            enum poly_type * ptype,
            struct OpeAdaptOpts * aopts)
{
    struct FtApproxArgs * fargs;
    if ( NULL == (fargs = malloc(sizeof(struct FtApproxArgs)))){
        fprintf(stderr, "failed to allocate memory for ft approx args.\n");
        exit(1);
    }
    fargs->dim = dim;
    fargs->targs = 0;
    
    enum function_class fc = POLYNOMIAL;
    fargs->fc.fc0 = fc;    
    //printf("create poly type %d \n",*ptype);
    fargs->sub_type.st0 = ptype;
    fargs->approx_opts.ao0 = aopts;
    
    return fargs;
}

/***********************************************************//**
    Create the arguments to give to use for approximation
    in the function train. Specifically, piecewise legendre polynomials

    \param dim [in] - dimension of function train
    \param ptype [in] - type of polynomial
    \param aopts [in] - arguments for creating the approximation (could be NULL)

    \return fargs - approximation arguments
***************************************************************/
struct FtApproxArgs * ft_approx_args_createpwpoly(size_t dim, 
            enum poly_type * ptype,
            struct PwPolyAdaptOpts * aopts)
{
    struct FtApproxArgs * fargs;
    if ( NULL == (fargs = malloc(sizeof(struct FtApproxArgs)))){
        fprintf(stderr, "failed to allocate memory for ft approx args.\n");
        exit(1);
    }
    fargs->dim = dim;
    fargs->targs = 0;
    
    enum function_class fc = PIECEWISE;
    fargs->fc.fc0 = fc;    
    fargs->sub_type.st0 = ptype;
    fargs->approx_opts.ao0 = aopts;
    
    return fargs;
}

/***********************************************************//**
    Free memory allocated to FtApproxArgs

    \param fargs [inout] - function train approximation arguments
***************************************************************/
void ft_approx_args_free(struct FtApproxArgs * fargs)
{
    if (fargs != NULL){
        if (fargs->targs == 0){
            free(fargs);
            fargs = NULL;
        }
        else{
            fprintf(stderr, 
                    "Freeing ftapprox args of this type not implemnted");
            exit(1);
        }
    }
}


/***********************************************************//**
    Extract the function class to use for the approximation of the
    *dim*-th dimensional functions 

    \param fargs [in] - function train approximation arguments
    \param dim [in] - dimension to extract

    \return function_class of the approximation
***************************************************************/
enum function_class 
ft_approx_args_getfc(struct FtApproxArgs * fargs, size_t dim)
{
    
    if (fargs->targs == 0){
        return fargs->fc.fc0;
    }
    else if (fargs->targs == 1){
        return fargs->fc.fc1[dim];
    }
    else{
        fprintf(stderr, "ftapprox args of this type not implemnted");
        exit(1);
    }
}

/***********************************************************//**
    Extract the sub type to use for the approximation of the
    *dim*-th dimensional functions 

    \param fargs [in] - function train approximation arguments
    \param dim [in] - dimension to extract

    \return sub type of the approximation
***************************************************************/
void * ft_approx_args_getst(struct FtApproxArgs * fargs, size_t dim)
{
    
    if (fargs->targs == 0){

        return fargs->sub_type.st0;
    }
    else if (fargs->targs == 1){
        return fargs->sub_type.st1[dim];
    }
    else{
        fprintf(stderr, "ftapprox args of this type not implemnted");
        exit(1);
    }
}

/***********************************************************//**
    Extract the approximation arguments to use for the approximation of the
    *dim*-th dimensional functions 

    \param fargs [in] - function train approximation arguments
    \param dim [in] - dimension to extract

    \return approximation arguments
***************************************************************/
void * ft_approx_args_getaopts(struct FtApproxArgs * fargs, size_t dim)
{
    
    if (fargs->targs == 0){
        return fargs->approx_opts.ao0;
    }
    else if (fargs->targs == 1){
        return fargs->approx_opts.ao1[dim];
    }
    else{
        fprintf(stderr, "ftapprox args of this type not implemnted");
        exit(1);
    }
}

struct BoundingBox * function_train_bds(struct FunctionTrain * ft)
{
    
    struct BoundingBox * bds = bounding_box_init_std(ft->dim);
    size_t ii;
    for (ii = 0; ii < ft->dim; ii++){
        bds->lb[ii] = generic_function_get_lower_bound(ft->cores[ii]->funcs[0]);
        bds->ub[ii] = generic_function_get_upper_bound(ft->cores[ii]->funcs[0]);
    }

    return bds;
}   


/***********************************************************//**
    Allocate space for a function_train

    \param dim [in] - dimension of function train

    \return ft - function train
***************************************************************/
struct FunctionTrain * function_train_alloc(size_t dim)
{
    struct FunctionTrain * ft = NULL;
    if ( NULL == (ft = malloc(sizeof(struct FunctionTrain)))){
        fprintf(stderr, "failed to allocate memory for function train.\n");
        exit(1);
    }
    ft->dim = dim;
    ft->ranks = calloc_size_t(dim+1);
    if ( NULL == (ft->cores = malloc(dim * sizeof(struct Qmarray *)))){
        fprintf(stderr, "failed to allocate memory for function train.\n");
        exit(1);
    }
    size_t ii;
    for (ii = 0; ii < dim; ii++){
        ft->cores[ii] = NULL;
    }
    return ft;
}

/***********************************************************//**
    Copy a function train

    \param a [in] - function train to copy

    \return b - function train
***************************************************************/
struct FunctionTrain * function_train_copy(struct FunctionTrain * a){

    struct FunctionTrain * b = function_train_alloc(a->dim);
    
    size_t ii;
    for (ii = 0; ii < a->dim; ii++){
        b->ranks[ii] = a->ranks[ii];
        b->cores[ii] = qmarray_copy(a->cores[ii]);
    }
    b->ranks[a->dim] = a->ranks[a->dim];
    return b;
}

/***********************************************************//**
    Free memory allocated to a function train

    \param ft [inout] - function train to free
***************************************************************/
void function_train_free(struct FunctionTrain * ft)
{
    if (ft != NULL){
        free(ft->ranks);
        ft->ranks = NULL;
        size_t ii;
        for (ii = 0; ii < ft->dim; ii++){
            qmarray_free(ft->cores[ii]);ft->cores[ii] = NULL;
        }
        free(ft->cores); ft->cores=NULL;
        free(ft); ft = NULL;
    }
}

/********************************************************//**
*    Create a functiontrain consisting of pseudo-random orth poly expansion
*   
*   \param bds [in] - boundaries
*   \param ranks [in] - (dim+1,1) array of ranks
*   \param maxorder [in] - maximum order of the polynomial
*
*   \return ft - runction train
************************************************************/
struct FunctionTrain *
function_train_poly_randu(struct BoundingBox * bds, size_t * ranks, size_t maxorder)
{
    size_t dim = bds->dim;
    struct FunctionTrain * ft = function_train_alloc(dim);
    memmove(ft->ranks,ranks, (dim+1)*sizeof(size_t));

    size_t ii;
    for (ii = 0; ii < dim; ii++){
        ft->cores[ii] = qmarray_poly_randu(ranks[ii],ranks[ii+1],maxorder,
                            bds->lb[ii],bds->ub[ii]);
    }
    return ft;
}

/***********************************************************//**
    Compute a function train representation of \f$ f1(x1)*f2(x2)*...* fd(xd)\f$

    \param dim [in] - dimension of function train
    \param f [in] -  one dimensional functions
    \param args [in] - arguments to the function
    \param bds [in] - boundarys of each dimension
    \param ftargs [in] - parameters for computation

    \return ft - function train
***************************************************************/
struct FunctionTrain *
function_train_rankone(size_t dim,  double (*f)(double, size_t, void *), 
        void * args, struct BoundingBox * bds, struct FtApproxArgs * ftargs)
{
    struct FtApproxArgs * ftargs_use;
    if (ftargs == NULL){
        enum poly_type ptype = LEGENDRE;
       // printf("the ptype is %d \n", ptype);
        ftargs_use = ft_approx_args_createpoly(dim,&ptype,NULL);
    
        //printf("check \n");
        //enum poly_type * pptype = ftargs_use->sub_type.st0;
        //printf("before anything type %d \n",*pptype);
    }
    else{
        ftargs_use = ftargs;
    }

    //printf("check2 \n");
    //enum poly_type * ppptype = ftargs_use->sub_type.st0;
    //printf("before anything type check2 %d \n",*ppptype);
    enum function_class fc;
    void * sub_type = NULL;
    void * approx_opts = NULL;

    //printf("check3 \n");
   // enum poly_type * pppptype = ftargs_use->sub_type.st0;
    //printf("before anything type check3 %d \n",*pppptype);

    struct wrap_spec_func spf;
    spf.f = f;
    spf.args = args;

    //printf("check4 dim = %zu\n",dim);
    //enum poly_type * ppppptype = ftargs_use->sub_type.st0;
    //printf("before anything type check4 %d \n",*ppppptype);

    struct FunctionTrain * ft = function_train_alloc(dim);
    
    size_t onDim;
    for (onDim = 0; onDim < dim; onDim++){
        //printf("ii = %zu\n",onDim);

        fc = ft_approx_args_getfc(ftargs_use, onDim);
        sub_type = ft_approx_args_getst(ftargs_use, onDim);
        //enum poly_type * ptype = sub_type;
        //printf("approximate with in ft type %d \n",*ptype);
        approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);
        spf.which = onDim;
        ft->ranks[onDim] = 1;
        ft->cores[onDim] = qmarray_alloc(1,1);
        
        //printf("lets go lb=%G, ub = %G\n",bds->lb[onDim], bds->ub[onDim]);
        ft->cores[onDim]->funcs[0] = 
            generic_function_approximate1d(eval_spec_func, &spf,
            fc, sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
        //printf("got it \n");
    }

    ft->ranks[dim] = 1;
    if (ftargs == NULL){
        ft_approx_args_free(ftargs_use);
    }
    return ft;
}

/***********************************************************//**
    Compute a function train representation of \f$ f1 + f2 + ... + fd \f$

    \param dim [in] - dimension of function train
    \param f [in] - array of one dimensional functions
    \param args [in] - array of arguments to the functions
    \param bds [in] - boundarys of each dimension
    \param ftargs [in] - parameters for computation

    \return ft - function train
***************************************************************/
struct FunctionTrain * 
function_train_initsum(size_t dim,  double (**f)(double, void *), 
        void ** args, struct BoundingBox * bds, struct FtApproxArgs * ftargs)
{
    struct FtApproxArgs * ftargs_use;
    if (ftargs == NULL){
        enum poly_type ptype = LEGENDRE;
        ftargs_use = ft_approx_args_createpoly(dim,&ptype,NULL);
    }
    else{
        ftargs_use = ftargs;
    }
    enum function_class fc;
    void * sub_type = NULL;
    void * approx_opts = NULL;

    struct FunctionTrain * ft = function_train_alloc(dim);

    size_t onDim = 0;
    fc = ft_approx_args_getfc(ftargs_use, onDim);
    sub_type = ft_approx_args_getst(ftargs_use, onDim);
    approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);
    ft->ranks[onDim] = 1;
    if (dim == 1){
        ft->cores[onDim] = qmarray_alloc(1,1);
    }
    else{
        ft->cores[onDim] = qmarray_alloc(1,2);
    }

    ft->cores[onDim]->funcs[0] = generic_function_approximate1d(f[onDim], 
                                    args[onDim], fc, sub_type, bds->lb[onDim], 
                                    bds->ub[onDim], approx_opts);
    ft->cores[onDim]->funcs[1] = generic_function_constant(1.0, fc, sub_type, 
                                 bds->lb[onDim], bds->ub[onDim], approx_opts);

    if (dim > 1){
        for (onDim = 1; onDim < dim-1; onDim++){
            fc = ft_approx_args_getfc(ftargs_use, onDim);
            sub_type = ft_approx_args_getst(ftargs_use, onDim);
            approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

            ft->ranks[onDim] = 2;
            ft->cores[onDim] = qmarray_alloc(2,2);

            ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);

            ft->cores[onDim]->funcs[1] =
                generic_function_approximate1d(f[onDim], args[onDim], fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);

            ft->cores[onDim]->funcs[2] = generic_function_constant(0.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
            ft->cores[onDim]->funcs[3] = generic_function_constant(1.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
        }

        onDim = dim-1;

        fc = ft_approx_args_getfc(ftargs_use, onDim);
        sub_type = ft_approx_args_getst(ftargs_use, onDim);
        approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

        ft->ranks[onDim] = 2;
        ft->ranks[onDim+1] = 1;
        ft->cores[onDim] = qmarray_alloc(2,1);

        ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                    sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
        ft->cores[onDim]->funcs[1] = generic_function_approximate1d(f[onDim], 
                        args[onDim], fc, sub_type, bds->lb[onDim],
                        bds->ub[onDim], approx_opts);
    }

    if (ftargs == NULL){
        ft_approx_args_free(ftargs_use);
    }
    return ft;
}

/***********************************************************//**
    Compute a tensor train representation of  \f$ f1 + f2 + ...+ fd \f$

    \param dim [in] - dimension of function train
    \param f [in] - one dimensional function with args x and i where  
               should denote which function to evaluate
    \param args [in] - array of arguments to the functions
    \param bds [in] - boundarys of each dimension
    \param ftargs [in] - parameters for computation

    \return ft - function train
***************************************************************/
struct FunctionTrain * 
function_train_initsum2(size_t dim,  double (*f)(double, size_t, void *), 
        void * args, struct BoundingBox * bds, struct FtApproxArgs * ftargs)
{
    struct FtApproxArgs * ftargs_use;
    if (ftargs == NULL){
        enum poly_type ptype = LEGENDRE;
        ftargs_use = ft_approx_args_createpoly(dim,&ptype,NULL);
    }
    else{
        ftargs_use = ftargs;
    }
    enum function_class fc;
    void * sub_type = NULL;
    void * approx_opts = NULL;

    struct wrap_spec_func spf;
    spf.f = f;
    spf.args = args;

    struct FunctionTrain * ft = function_train_alloc(dim);

    size_t onDim = 0;
    //printf("onDim=(%zu/%zu)\n",onDim,dim);
    fc = ft_approx_args_getfc(ftargs_use, onDim);
    sub_type = ft_approx_args_getst(ftargs_use, onDim);
    approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

    spf.which = onDim;
    ft->ranks[onDim] = 1;
    if (dim == 1){
        ft->cores[onDim] = qmarray_alloc(1,1);
    }
    else{
        ft->cores[onDim] = qmarray_alloc(1,2);
    }

    ft->cores[onDim]->funcs[0] = 
            generic_function_approximate1d(eval_spec_func, &spf,
            fc, sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);

    ft->cores[onDim]->funcs[1] = 
                generic_function_constant(1.0, fc, sub_type, 
                bds->lb[onDim], bds->ub[onDim], NULL);

    if (dim > 1){
        for (onDim = 1; onDim < dim-1; onDim++){
            //printf("onDim=(%zu/%zu)\n",onDim,dim);
            fc = ft_approx_args_getfc(ftargs_use, onDim);
            sub_type = ft_approx_args_getst(ftargs_use, onDim);
            approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);
            spf.which = onDim;

            ft->ranks[onDim] = 2;
            ft->cores[onDim] = qmarray_alloc(2,2);

            ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], NULL);

            ft->cores[onDim]->funcs[1] =
                generic_function_approximate1d(eval_spec_func, &spf,
                fc, sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);

            ft->cores[onDim]->funcs[2] = generic_function_constant(0.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], NULL);
            ft->cores[onDim]->funcs[3] = generic_function_constant(1.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], NULL);
        }

        onDim = dim-1;
        //printf("onDim=(%zu/%zu)\n",onDim,dim);

        fc = ft_approx_args_getfc(ftargs_use, onDim);
        sub_type = ft_approx_args_getst(ftargs_use, onDim);
        approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);
        spf.which = onDim;

        ft->ranks[onDim] = 2;
        ft->ranks[onDim+1] = 1;
        ft->cores[onDim] = qmarray_alloc(2,1);

        ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                    sub_type, bds->lb[onDim], bds->ub[onDim], NULL);
        ft->cores[onDim]->funcs[1] = 
                generic_function_approximate1d(eval_spec_func, &spf,
                fc, sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
    }

    if (ftargs == NULL){
        ft_approx_args_free(ftargs_use);
    }
    return ft;
}

/***********************************************************//**
    Compute a tensor train representation of 
    \f$ (x_1c_1+a_1) + (x_2c_2+a_2)  + .... + (x_dc_d+a_d) \f$

    \param dim [in] - dimension of function train
    \param bds [in] - boundarys of each dimension
    \param c [in] -  slope of the function in each dimension (ldc x dim)
    \param ldc [in] - stride of coefficients in array of c
    \param a [in] - offset of the function in each dimension (ldc x dim)
    \param lda [in] - stride of coefficients in array a
    \param ftargs [in] - parameters for computation

    \return ft - function train
***************************************************************/
struct FunctionTrain * 
function_train_linear2(size_t dim, struct BoundingBox * bds, 
        double * c, size_t ldc, double * a, size_t lda,
        struct FtApproxArgs * ftargs)
{
    struct FtApproxArgs * ftargs_use;
    if (ftargs == NULL){
        enum poly_type ptype = LEGENDRE;
        ftargs_use = ft_approx_args_createpoly(dim,&ptype,NULL);
    }
    else{
        ftargs_use = ftargs;
    }
    enum function_class fc;
    void * sub_type = NULL;
    void * approx_opts = NULL;
    
    struct FunctionTrain * ft = function_train_alloc(dim);

    size_t onDim = 0;
    fc = ft_approx_args_getfc(ftargs_use, onDim);
    sub_type = ft_approx_args_getst(ftargs_use, onDim);
    approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);
    ft->ranks[onDim] = 1;

    if (dim == 1){
        ft->cores[onDim] = qmarray_alloc(1,1);
    }
    else{
        ft->cores[onDim] = qmarray_alloc(1,2);
    }
    
    ft->cores[onDim]->funcs[0] = 
        generic_function_linear(c[onDim*ldc], a[onDim*lda], 
                            fc, sub_type, bds->lb[onDim], bds->ub[onDim],
                            approx_opts);
    
    if (dim > 1){

        ft->cores[onDim]->funcs[1] = 
            generic_function_constant(1.0, fc, sub_type, 
                bds->lb[onDim], bds->ub[onDim], approx_opts);
    
        for (onDim = 1; onDim < dim-1; onDim++){
            fc = ft_approx_args_getfc(ftargs_use, onDim);
            sub_type = ft_approx_args_getst(ftargs_use, onDim);
            approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

            ft->ranks[onDim] = 2;
            ft->cores[onDim] = qmarray_alloc(2,2);

            ft->cores[onDim]->funcs[0] = 
                generic_function_constant(1.0, fc, 
                    sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);


            ft->cores[onDim]->funcs[1] = 
                generic_function_linear(c[onDim*ldc], a[onDim*lda], 
                            fc, sub_type, bds->lb[onDim], bds->ub[onDim],
                            approx_opts);

            ft->cores[onDim]->funcs[2] = 
                generic_function_constant(0.0, fc, 
                    sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
            ft->cores[onDim]->funcs[3] = 
                generic_function_constant(1.0, fc, 
                    sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
        }

        onDim = dim-1;

        fc = ft_approx_args_getfc(ftargs_use, onDim);
        sub_type = ft_approx_args_getst(ftargs_use, onDim);
        approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

        ft->ranks[onDim] = 2;
        ft->ranks[onDim+1] = 1;
        ft->cores[onDim] = qmarray_alloc(2,1);

        ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                    sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);

        ft->cores[onDim]->funcs[1] = 
            generic_function_linear(c[onDim*ldc], a[onDim*lda], 
                            fc, sub_type, bds->lb[onDim], bds->ub[onDim],
                            approx_opts);
    }

    if (ftargs == NULL){
        ft_approx_args_free(ftargs_use);
    }
    return ft;
}
 
/***********************************************************//**
    Compute a function train representation of \f$ a \f$

    \param dim [in] - dimension of function train
    \param a [in] - value of tensor train
    \param bds [in] - boundarys of each dimension
    \param ftargs [in] - parameters for computation

    \return ft - function train

    \note 
        Puts the constant into the first core
***************************************************************/
struct FunctionTrain * 
function_train_constant(size_t dim, double a, struct BoundingBox * bds,  
                        struct FtApproxArgs * ftargs)
{
    struct FtApproxArgs * ftargs_use;
    if (ftargs == NULL){
        enum poly_type ptype = LEGENDRE;
        ftargs_use = ft_approx_args_createpoly(dim,&ptype,NULL);
    }
    else{
        ftargs_use = ftargs;
    }
    enum function_class fc;
    void * sub_type = NULL;
    //void * approx_opts = NULL;
    
    struct FunctionTrain * ft = function_train_alloc(dim);
    
    //double val = pow(a,1.0/(double) dim);
    //printf("a = %G\n",a);
    //printf("val = %G\n",val);

    size_t onDim = 0;
    fc = ft_approx_args_getfc(ftargs_use, onDim);
    sub_type = ft_approx_args_getst(ftargs_use, onDim);
    //approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

    ft->ranks[onDim] = 1;
    ft->cores[onDim] = qmarray_alloc(1,1);
    ft->cores[onDim]->funcs[0] = generic_function_constant(a,
               fc, sub_type, bds->lb[onDim], bds->ub[onDim], NULL);

    for (onDim = 1; onDim < dim; onDim++){
        fc = ft_approx_args_getfc(ftargs_use, onDim);
        sub_type = ft_approx_args_getst(ftargs_use, onDim);
        //approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

        ft->ranks[onDim] = 1;
        ft->cores[onDim] = qmarray_alloc(1,1);
        ft->cores[onDim]->funcs[0] = generic_function_constant(1,
                   fc, sub_type, bds->lb[onDim], bds->ub[onDim], NULL);
    }
    ft->ranks[dim] = 1;

    if (ftargs == NULL){
        ft_approx_args_free(ftargs_use);
    }

    return ft;
}

/***********************************************************//**
    Compute a tensor train representation of \f$ x_1c_1 + x_2c_2 + .... + x_dc_d \f$

    \param dim [in] - dimension of function train
    \param bds [in] - boundarys of each dimension
    \param coeffs [in] -  slope of the function in each dimension
    \param ftargs [in] - parameters for computation

    \return ft - function train
***************************************************************/
struct FunctionTrain * 
function_train_linear(size_t dim, struct BoundingBox * bds, double * coeffs, 
                        struct FtApproxArgs * ftargs)
{
    
    struct FtApproxArgs * ftargs_use;
    if (ftargs == NULL){
        enum poly_type ptype = LEGENDRE;
        ftargs_use = ft_approx_args_createpoly(dim,&ptype,NULL);
    }
    else{
        ftargs_use = ftargs;
    }
    enum function_class fc;
    void * sub_type = NULL;
    void * approx_opts = NULL;
    
    struct FunctionTrain * ft = function_train_alloc(dim);

    size_t onDim = 0;
    fc = ft_approx_args_getfc(ftargs_use, onDim);
    sub_type = ft_approx_args_getst(ftargs_use, onDim);
    approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);
    ft->ranks[onDim] = 1;

    if (dim == 1){
        ft->cores[onDim] = qmarray_alloc(1,1);
    }
    else{
        ft->cores[onDim] = qmarray_alloc(1,2);
    }
    
    if (dim > 1){
        ft->cores[onDim]->funcs[0] = generic_function_linear(coeffs[onDim], 0.0, 
                                        fc, sub_type, bds->lb[onDim], 
                                        bds->ub[onDim], approx_opts);
        ft->cores[onDim]->funcs[1] = generic_function_constant(1.0, fc, sub_type, 
                                     bds->lb[onDim], bds->ub[onDim], approx_opts);
        
        for (onDim = 1; onDim < dim-1; onDim++){
            fc = ft_approx_args_getfc(ftargs_use, onDim);
            sub_type = ft_approx_args_getst(ftargs_use, onDim);
            approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

            ft->ranks[onDim] = 2;
            ft->cores[onDim] = qmarray_alloc(2,2);

            ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);

            ft->cores[onDim]->funcs[1] = generic_function_linear(coeffs[onDim],0.0,
                                        fc, sub_type, bds->lb[onDim], 
                                        bds->ub[onDim], approx_opts);

            ft->cores[onDim]->funcs[2] = generic_function_constant(0.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
            ft->cores[onDim]->funcs[3] = generic_function_constant(1.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
        }

        onDim = dim-1;

        fc = ft_approx_args_getfc(ftargs_use, onDim);
        sub_type = ft_approx_args_getst(ftargs_use, onDim);
        approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

        ft->ranks[onDim] = 2;
        ft->ranks[onDim+1] = 1;
        ft->cores[onDim] = qmarray_alloc(2,1);

        ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                    sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
        ft->cores[onDim]->funcs[1] = generic_function_linear(coeffs[onDim], 0.0,
                                    fc, sub_type, bds->lb[onDim], 
                                    bds->ub[onDim], approx_opts);

    }
    else{
        ft->cores[onDim]->funcs[0] = generic_function_linear(coeffs[onDim], 0.0, 
                                        fc, sub_type, bds->lb[onDim], 
                                        bds->ub[onDim], approx_opts);
    }

    if (ftargs == NULL){
        ft_approx_args_free(ftargs_use);
    }
    return ft;
}

/***********************************************************//**
    Compute a function train representation of \f$ (x-m)^T Q (x-m) \f$

    \param dim [in] - dimension of function train
    \param bds [in] - boundarys of each dimension
    \param coeffs (IN): Q matrix
    \param m [in] - m (dim,)
    \param ftargs [in] - parameters for computation

    \returns ft - function train

    \note
        Could be more efficient with a better distribution of ranks
***************************************************************/
struct FunctionTrain * 
function_train_quadratic(size_t dim, struct BoundingBox * bds, double * coeffs, 
                        double * m, struct FtApproxArgs * ftargs)
{
    assert (dim > 1); //
    struct FtApproxArgs * ftargs_use;
    if (ftargs == NULL){
        enum poly_type ptype = LEGENDRE;
        ftargs_use = ft_approx_args_createpoly(dim,&ptype,NULL);
    }
    else{
        ftargs_use = ftargs;
    }
    enum function_class fc;
    void * sub_type = NULL;
    void * approx_opts = NULL;
    double temp;
    
    struct FunctionTrain * ft = function_train_alloc(dim);
    
    size_t kk,ll;

    size_t onDim = 0;
    fc = ft_approx_args_getfc(ftargs_use, onDim);
    sub_type = ft_approx_args_getst(ftargs_use, onDim);
    approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);


    ft->ranks[onDim] = 1;
    ft->cores[onDim] = qmarray_alloc(1,dim+1);
    
    // should be quadratic
    ft->cores[onDim]->funcs[0] = generic_function_quadratic(
                coeffs[onDim*dim+onDim], m[onDim],
                fc, sub_type, bds->lb[onDim], 
                bds->ub[onDim], approx_opts);

    for (kk = 1; kk < dim; kk++)
    {
        //printf("on dimension 1 (%zu/%zu)\n",kk,dim);
        temp = coeffs[onDim*dim+kk] + coeffs[kk*dim + onDim];
        ft->cores[onDim]->funcs[kk] = 
                generic_function_linear(temp,-temp*m[onDim], fc, 
                                    sub_type, bds->lb[onDim], 
                                    bds->ub[onDim], approx_opts);
    }

    ft->cores[onDim]->funcs[dim] = generic_function_constant(1.0, fc, sub_type, 
                                 bds->lb[onDim], bds->ub[onDim], approx_opts);
    
    for (onDim = 1; onDim < dim-1; onDim++){
        printf("on dimension (%zu/%zu)\n",onDim+1,dim);
        fc = ft_approx_args_getfc(ftargs_use, onDim);
        sub_type = ft_approx_args_getst(ftargs_use, onDim);
        approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

        ft->ranks[onDim] = dim-onDim+2;
        ft->cores[onDim] = qmarray_alloc(dim-onDim+2,dim-onDim+1);
        
        for (kk = 0; kk < (dim-onDim+1); kk++){ // loop over columns
            for (ll = 0; ll < (dim - onDim + 2); ll++){ //rows;
                if ( (ll == 0) && (kk == 0)){ // upper left corner
                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] = 
                            generic_function_constant(1.0, fc, 
                            sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);

                }
                else if ( (kk == 0) && (ll == 1) ){ // first element of lower diagonal
                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] = 
                            generic_function_linear(1.0,-m[onDim], fc, 
                                                sub_type, bds->lb[onDim], 
                                                bds->ub[onDim], approx_opts);
                }
                else if (ll == (kk+1)){ // lower diagonal
                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] = 
                            generic_function_constant(1.0, fc, 
                            sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
                }
                else if ( (ll == (dim-onDim+1)) && (kk == 0)){ //lower left corner
                    //quadratic
                     ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] = 
                                generic_function_quadratic(
                                coeffs[onDim*dim+onDim], m[onDim],
                                fc, sub_type, bds->lb[onDim], 
                                bds->ub[onDim], approx_opts);
                }
                else if ( ll == (dim-onDim+1) ){ // rest of bottom row
                    temp = coeffs[onDim*dim+onDim+kk] + 
                                coeffs[(onDim+kk)*dim + onDim];

                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] = 
                        generic_function_linear(temp,-temp*m[onDim], fc, 
                                    sub_type, bds->lb[onDim], 
                                    bds->ub[onDim], approx_opts);
                }
                else{ // zeros
                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] = 
                            generic_function_constant(0.0, fc, 
                            sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
                }
                //printf("generic function for core %zu num (%zu,%zu) \n",onDim,ll,kk);
                //print_generic_function(ft->cores[onDim]->funcs[kk*dim+ll],0,NULL);
            }
        }
    }

    onDim = dim-1;

    fc = ft_approx_args_getfc(ftargs_use, onDim);
    sub_type = ft_approx_args_getst(ftargs_use, onDim);
    approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

    ft->ranks[onDim] = dim-onDim+2;
    ft->cores[onDim] = qmarray_alloc(dim-onDim+2,1);

    ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);

    ft->cores[onDim]->funcs[1] = generic_function_linear(1.0, -m[onDim],
                                    fc, sub_type, bds->lb[onDim], 
                                    bds->ub[onDim], approx_opts);

    ft->cores[onDim]->funcs[2] = generic_function_quadratic(
                                coeffs[onDim*dim+onDim], m[onDim],
                                fc, sub_type, bds->lb[onDim], 
                                bds->ub[onDim], approx_opts);
    ft->ranks[dim] = 1;
    if (ftargs == NULL){
        ft_approx_args_free(ftargs_use);
    }
    return ft;
}

/***********************************************************//**
    Compute a tensor train representation of \f$ (x_1-m_1)^2c_1 + (x_2-m_2)^2c_2 + .... + (x_d-m_d)^2c_d \f$

    \param bds [in] - boundarys of each dimension
    \param coeffs [in] - coefficients for each dimension
    \param m [in] - offset in each dimension
    \param ftargs [in] - parameters for computation

    \return ft - function train
***************************************************************/
struct FunctionTrain * 
function_train_quadratic_aligned(struct BoundingBox * bds, 
            double * coeffs, double * m,
            struct FtApproxArgs * ftargs)
{
    size_t dim = bds->dim;

    struct FtApproxArgs * ftargs_use;
    if (ftargs == NULL){
        enum poly_type ptype = LEGENDRE;
        ftargs_use = ft_approx_args_createpoly(dim,&ptype,NULL);
    }
    else{
        ftargs_use = ftargs;
    }
    enum function_class fc;
    void * sub_type = NULL;
    void * approx_opts = NULL;
    
    struct FunctionTrain * ft = function_train_alloc(dim);

    size_t onDim = 0;
    fc = ft_approx_args_getfc(ftargs_use, onDim);
    sub_type = ft_approx_args_getst(ftargs_use, onDim);
    approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);
    ft->ranks[onDim] = 1;

    if (dim == 1){
        ft->cores[onDim] = qmarray_alloc(1,1);
    }
    else{
        ft->cores[onDim] = qmarray_alloc(1,2);
    }
    
    
    if (dim > 1){
        ft->cores[onDim]->funcs[0] = generic_function_quadratic(
                                    coeffs[onDim], m[onDim],
                                    fc, sub_type, bds->lb[onDim], 
                                    bds->ub[onDim], approx_opts);
        ft->cores[onDim]->funcs[1] = generic_function_constant(1.0, fc, sub_type, 
                                     bds->lb[onDim], bds->ub[onDim], approx_opts);
        for (onDim = 1; onDim < dim-1; onDim++){
            fc = ft_approx_args_getfc(ftargs_use, onDim);
            sub_type = ft_approx_args_getst(ftargs_use, onDim);
            approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

            ft->ranks[onDim] = 2;
            ft->cores[onDim] = qmarray_alloc(2,2);

            ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);

            ft->cores[onDim]->funcs[1] = generic_function_quadratic(
                                coeffs[onDim], m[onDim],
                                fc, sub_type, bds->lb[onDim], 
                                bds->ub[onDim], approx_opts);

            ft->cores[onDim]->funcs[2] = generic_function_constant(0.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
            ft->cores[onDim]->funcs[3] = generic_function_constant(1.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
        }

        onDim = dim-1;

        fc = ft_approx_args_getfc(ftargs_use, onDim);
        sub_type = ft_approx_args_getst(ftargs_use, onDim);
        approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

        ft->ranks[onDim] = 2;
        ft->ranks[onDim+1] = 1;
        ft->cores[onDim] = qmarray_alloc(2,1);

        ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                    sub_type, bds->lb[onDim], bds->ub[onDim], approx_opts);
        ft->cores[onDim]->funcs[1] = generic_function_quadratic(
                            coeffs[onDim], m[onDim],
                            fc, sub_type, bds->lb[onDim], 
                            bds->ub[onDim], approx_opts);

    }
    else{
        ft->cores[onDim]->funcs[0] = generic_function_quadratic(
                                    coeffs[onDim], m[onDim],
                                    fc, sub_type, bds->lb[onDim], 
                                    bds->ub[onDim], approx_opts);
    }

    if (ftargs == NULL){
        ft_approx_args_free(ftargs_use);
    }
    return ft;
}


/***********************************************************//**
    Serialize a function_train

    \param ser [inout] - stream to serialize to
    \param ft [in] - function train
    \param totSizeIn [inout] - if NULL then serialize, if not NULL then return size

    \return ptr - ser shifted by number of ytes
***************************************************************/
unsigned char * 
function_train_serialize(unsigned char * ser, struct FunctionTrain * ft,
                size_t *totSizeIn)
{

    // dim -> nranks -> core1 -> core2 -> ... -> cored

    size_t ii;
    //dim + ranks
    size_t totSize = sizeof(size_t) + (ft->dim+1)*sizeof(size_t);
    size_t size_temp;
    for (ii = 0; ii < ft->dim; ii++){
        qmarray_serialize(NULL,ft->cores[ii],&size_temp);
        totSize += size_temp;
    }
    if (totSizeIn != NULL){
        *totSizeIn = totSize;
        return ser;
    }
    
    unsigned char * ptr = ser;
    ptr = serialize_size_t(ptr, ft->dim);
    //printf("serializing ranks\n");
    for (ii = 0; ii < ft->dim+1; ii++){
        ptr = serialize_size_t(ptr,ft->ranks[ii]);
    }
    //printf("done serializing ranks dim\n");
    for (ii = 0; ii < ft->dim; ii++){
        //printf("Serializing core (%zu/%zu)\n",ii,ft->dim);
        ptr = qmarray_serialize(ptr, ft->cores[ii],NULL);
    }
    return ptr;
}

/***********************************************************//**
    Deserialize a function train

    \param ser [inout] - serialized function train
    \param ft [inout] - function_train

    \return ptr - shifted ser after deserialization
***************************************************************/
unsigned char *
function_train_deserialize(unsigned char * ser, struct FunctionTrain ** ft)
{
    unsigned char * ptr = ser;

    size_t dim;
    ptr = deserialize_size_t(ptr, &dim);
    //printf("deserialized dim=%zu\n",dim);
    *ft = function_train_alloc(dim);
    
    size_t ii;
    for (ii = 0; ii < dim+1; ii++){
        ptr = deserialize_size_t(ptr, &((*ft)->ranks[ii]));
    }
    for (ii = 0; ii < dim; ii++){
        ptr = qmarray_deserialize(ptr, &((*ft)->cores[ii]));
    }
    
    return ptr;
}

/***********************************************************//**
    Save a function train to file
    
    \param ft [in] - function train to save
    \param filename [in] - name of file to save to

    \return success (1) or failure (0) of opening the file
***************************************************************/
int function_train_save(struct FunctionTrain * ft, char * filename)
{

    FILE *fp;
    fp = fopen(filename, "w");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open %s\n", filename);
        return 0;
    }
    
    size_t totsize;
    function_train_serialize(NULL,ft, &totsize);

    unsigned char * data = malloc(totsize+sizeof(size_t));
    if (data == NULL){
        fprintf(stderr, "can't allocate space for saving density\n");
        return 0;
    }
    
    // serialize size first!
    unsigned char * ptr = serialize_size_t(data,totsize);
    ptr = function_train_serialize(ptr,ft,NULL);

    fwrite(data,sizeof(unsigned char),totsize+sizeof(size_t),fp);

    free(data); data = NULL;
    fclose(fp);
    return 1;
}

/***********************************************************//**
    Load a function train from a file
    
    \param filename [in] - filename of file to load

    \return ft if successfull NULL otherwise
***************************************************************/
struct FunctionTrain * function_train_load(char * filename)
{
    FILE *fp;
    fp =  fopen(filename, "r");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open %s\n", filename);
        return NULL;
    }

    size_t totsize;
    size_t k = fread(&totsize,sizeof(size_t),1,fp);
    if ( k != 1){
        printf("error reading file %s\n",filename);
        return NULL;
    }

    unsigned char * data = malloc(totsize);
    if (data == NULL){
        fprintf(stderr, "can't allocate space for loading density\n");
        return NULL;
    }

    k = fread(data,sizeof(unsigned char),totsize,fp);

    struct FunctionTrain * ft = NULL;
    function_train_deserialize(data,&ft);
    
    free(data); data = NULL;
    fclose(fp);
    return ft;
}


/***********************************************************//**
    Allocate a 1d array of function trains

    \param dimout [in] - number of function trains

    \return fta - function train array

    \note 
        Each ft of the array is set to NULL;
***************************************************************/
struct FT1DArray * ft1d_array_alloc(size_t dimout)
{
    struct FT1DArray * fta = malloc(sizeof(struct FT1DArray));
    if (fta == NULL){
        fprintf(stderr, "Error allocating 1d function train array");
        exit(1);
    }
    
    fta->size = dimout;

    fta->ft = malloc(dimout * sizeof(struct FunctionTrain *));
    if (fta == NULL){
        fprintf(stderr, "Error allocating 1d function train array");
        exit(1);
    }
        
    size_t ii;
    for (ii = 0; ii < dimout; ii ++){
        fta->ft[ii] = NULL;
    }
    
    return fta;
}
/***********************************************************//**
    Copy an array of function trains

    \param fta [in] - array to coppy

    \return ftb - copied array
***************************************************************/
struct FT1DArray * ft1d_array_copy(struct FT1DArray * fta)
{
    struct FT1DArray * ftb = ft1d_array_alloc(fta->size);
    size_t ii;
    for (ii = 0; ii < fta->size; ii++){
        ftb->ft[ii] = function_train_copy(fta->ft[ii]);
    }
    return ftb;
}

/***********************************************************//**
    Free a 1d array of function trains

    \param fta [inout] - function train array to free
***************************************************************/
void ft1d_array_free(struct FT1DArray * fta)
{
    if (fta != NULL){
        size_t ii = 0;
        for (ii = 0; ii < fta->size; ii++){
            function_train_free(fta->ft[ii]);
            fta->ft[ii] = NULL;
        }
        free(fta->ft);
        fta->ft = NULL;
        free(fta);
        fta = NULL;
    }
}

/////////////////////////////////////////////////////////////////////
// Indices

/***********************************************************//**
    Allocate an index set for cross approxmiation

    \param type [in] - 0 for left index set, 1 for right index set
    \param totdim [in] - total dimension of approximation
    \param dim [in] - current dimension
    \param rank [in] - rank of current core

    \return is  - index set
***************************************************************/
struct IndexSet * 
index_set_alloc(int type, size_t totdim, size_t dim, size_t rank)
{
    struct IndexSet * is = NULL;
    if ( NULL == (is = malloc(sizeof(struct IndexSet)))){
        fprintf(stderr, "failed to allocate memory for index_set.\n");
        exit(1);
    }
    is->type = type;
    is->totdim = totdim;
    is->dim = dim;
    is->rank = rank;
    //printf("rank =%zu\n", rank);
    if ( NULL == (is->inds = malloc( rank *sizeof(double *)))){
        fprintf(stderr, "failed to allocate memory for index_set_indices.\n");
        exit(1);
    }
    size_t ii;
    // left indices
    if (type == 0){
        assert (dim > 0);
        for (ii = 0; ii < rank; ii++){
            is->inds[ii] = calloc_double(dim);    
        }
    }
    else if (type == 1){
        assert (dim < (totdim-1));
        for (ii = 0; ii < rank; ii++){
            is->inds[ii] = calloc_double(totdim-dim);    
        }
    }
    return is;
}

/***********************************************************//**
    Free memory allocated for index set and set it to NULL

    \param is [inout] - index set to free
***************************************************************/
void index_set_free(struct IndexSet * is)
{
    if (is != NULL){
        size_t ii;
        size_t rank = is->rank;
        for (ii = 0; ii < rank; ii++){
            free(is->inds[ii]);
            is->inds[ii] = NULL;
        }
        free(is->inds);
        is->inds = NULL;
        free(is);
        is = NULL;
    }
}

/***********************************************************//**
    Free memory allocated to index set array
    
    \param n [in] - number of index sets
    \param is  [inout] - index set array to free
***************************************************************/
void index_set_array_free(size_t n, struct IndexSet ** is)
{
    if (is != NULL) {

        size_t ii;
        for (ii = 0; ii < n; ii++){
            index_set_free(is[ii]);
            is[ii] = NULL;
        }
        free(is); is = NULL;
    }
}

/***********************************************************//**
    Initialize a right nested index set
    
    \param dim [in] - dimension
    \param ranks [in] - ranks of the function
    \param opts [in] - a set of locations

    \return isr - right index set

    \note
        This function is mainly used for the initialization of the
        right index set in the function train approximation
        algorithm
***************************************************************/
struct IndexSet ** 
index_set_array_rnested(size_t dim, size_t * ranks, double * opts)
{
    struct IndexSet ** isr = NULL;
    if ( NULL == (isr = malloc(dim * sizeof(struct IndexSet * )))){
        fprintf(stderr, "failed to allocate memory for index_set array.\n");
        exit(1);
    }
    
    size_t jj,kk;
    size_t ii = 0;
    for (ii = 0; ii < dim-1; ii++){
        isr[ii] = index_set_alloc(1, dim, ii, ranks[ii+1]);
        for (jj = 0; jj < ranks[ii+1]; jj++){
            for (kk = ii+1; kk < dim; kk++){
                isr[ii]->inds[jj][kk-ii-1] = opts[kk];
                //isr[ii]->inds[jj][kk-ii-1] = randu();
            }
        }
    }
    isr[dim-1] = NULL;
    
    return isr;
}

/***********************************************************//**
    Initialize a left-nested index set
    
    \param dim [in] - dimension
    \param ranks [in] - ranks of the function
    \param opts [in] - a set of locations

    \return isl - left index set

    \note
        This function is mainly used for the initialization of the
        left index set in the function train approximation
        algorithm
***************************************************************/
struct IndexSet ** 
index_set_array_lnested(size_t dim, size_t * ranks, double * opts)
{
    
    struct IndexSet ** isl = NULL;
    if ( NULL == (isl = malloc(dim * sizeof(struct IndexSet *)))){
        fprintf(stderr, "failed to allocate memory for index_set array.\n");
        exit(1);
    }
    
    size_t jj,kk;
    size_t ii = 0;
    isl[ii] = NULL;
    for (ii = 1; ii < dim; ii++){
        isl[ii] = index_set_alloc(0, dim, ii, ranks[ii]);
        for (jj = 0; jj < ranks[ii]; jj++){
            for (kk = 0; kk < ii; kk++){
                //isl[ii]->inds[jj][kk] = randu();
                isl[ii]->inds[jj][kk] = opts[kk];
            }
        }
    }
    return isl;
}

double ** 
index_set_merge(struct IndexSet * isl, struct IndexSet * isr, size_t * nvals)
{
    double ** out = NULL;   
    size_t rankl, rankr;
    size_t endleft, dim;
    if ((isl == NULL) && (isr != NULL)){
        rankl = 0;
        rankr = isr->rank;
        endleft = 0;
        dim = isr->totdim;
        out = malloc(rankr * sizeof(double *));
        *nvals = rankr;
    }
    else if ( (isl != NULL) && (isr == NULL)){
        rankl = isl->rank;
        rankr = 0;
        endleft = isl->totdim;
        dim = isl->totdim;
        out = malloc(rankl * sizeof(double *));
        *nvals = rankl;
    }
    else if ( (isl != NULL) && (isr != NULL)){
        rankl = isl->rank;
        rankr = isr->rank;
        endleft = isl->dim;
        dim = isl->totdim;
        out = malloc(rankl * rankr * sizeof(double *));
        *nvals = rankl*rankr;
    }
    else{
        fprintf(stderr, "cannot merge these indices.\n");
        exit(1);
    }

    if ( out == NULL){
        fprintf(stderr, "failed to allocate memory for index_set.\n");
        exit(1);
    }

    size_t ii, jj, kk;
    if (rankl == 0){
        for (ii = 0; ii < rankr; ii++){
            out[ii] = calloc_double(dim);
            out[ii][0] = 0.0; // arbitray
            for (jj = 1; jj < dim; jj++){
                out[ii][jj] = isr->inds[ii][jj-1]; 
            }
        }
    }
    else if (rankr == 0){
        for (ii = 0; ii < rankl; ii++){
            out[ii] = calloc_double(dim);
            for (jj = 0; jj < dim-1; jj++){
                out[ii][jj] = isl->inds[ii][jj]; 
            }
            out[ii][dim-1] = 0.0; // arbitray
        }
    }
    else {
        for (ii = 0; ii < rankr; ii++){
            for (jj = 0; jj < rankl; jj++){
                out[ii*rankl+jj] = calloc_double(dim);
                for (kk = 0; kk < endleft; kk++){
                    out[ii*rankl+jj][kk] = isl->inds[jj][kk];
                }
                out[ii*rankl+jj][endleft] = 0.0; // arbitrary;
                for (kk = endleft+1; kk < dim; kk++){
                    out[ii*rankl+jj][kk] = isr->inds[ii][kk-endleft-1];
                }
            }
        }
    }
    
    return out;
}

double ** 
index_set_merge_fill_end(struct IndexSet * isl, double ** isr)
                            
{
    double ** out = NULL;   
    size_t rankl, dim, ii, jj;
    assert (isl != NULL);

    rankl = isl->rank;

    dim = isl->totdim;
    out = malloc(rankl * sizeof(double *));

    if ( out == NULL){
        fprintf(stderr, "failed to allocate memory for index_set.\n");
        exit(1);
    }

    for (ii = 0; ii < rankl; ii++){
        out[ii] = calloc_double(dim);
        for (jj = 0; jj < dim-1; jj++){
            out[ii][jj] = isl->inds[ii][jj]; 
        }
        out[ii][dim-1] = isr[ii][0];
    }
    return out;
}

double ** 
index_set_merge_fill_beg(double ** isl, struct IndexSet * isr)
                            
{
    double ** out = NULL;   
    size_t rankr, dim, ii, jj;
    assert (isr != NULL);

    rankr = isr->rank;

    dim = isr->totdim;
    out = malloc(rankr * sizeof(double *));

    if ( out == NULL){
        fprintf(stderr, "failed to allocate memory for index_set.\n");
        exit(1);
    }

    for (ii = 0; ii < rankr; ii++){
        out[ii] = calloc_double(dim);
        out[ii][0] = isl[ii][0];
        for (jj = 1; jj < dim; jj++){
            out[ii][jj] = isr->inds[ii][jj-1]; 
        }
    }
    return out;
}


/////////////////////////////////////////////////////////
// Utilities
//
void print_quasimatrix(struct Quasimatrix * qm, size_t prec, void * args)
{

    printf("Quasimatrix consists of %zu columns\n",qm->n);
    printf("=========================================\n");
    size_t ii;
    for (ii = 0; ii < qm->n; ii++){
        print_generic_function(qm->funcs[ii],prec,args);
        printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    }
}

void print_qmarray(struct Qmarray * qm, size_t prec, void * args)
{

    printf("Quasimatrix Array (%zu,%zu)\n",qm->nrows, qm->ncols);
    printf("=========================================\n");
    size_t ii,jj;
    for (ii = 0; ii < qm->nrows; ii++){
        for (jj = 0; jj < qm->ncols; jj++){
            printf("(%zu, %zu)\n",ii,jj);
            print_generic_function(qm->funcs[jj*qm->nrows+ ii],prec,args);
            printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
        }
    }
}

void print_index_set_array(size_t N, struct IndexSet ** ar){
    
    size_t ii, jj;
    for (ii = 0; ii < N; ii++){
        printf("Index Set (%zu/%zu) .... \n",ii,N);
        if (ar[ii] == NULL){
            printf("NULL\n");
        }
        else{
            if (ar[ii]->type == 1){
                for (jj = 0; jj < ar[ii]->rank; jj++){
                    dprint(N-ii-1, ar[ii]->inds[jj]);
                }
            }
            else if (ar[ii]->type == 0){
                for (jj = 0; jj < ar[ii]->rank; jj++){
                    dprint(ii, ar[ii]->inds[jj]);
                }
            }
            else{
                fprintf(stderr, "type of index set not specified\n");
                exit(1);
            }
        }
    }

}
