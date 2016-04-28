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

/* /\**********************************************************\//\** */
/*     Extract a copy of the first nkeep columns of a qmarray */

/*     \param a [in] - qmarray from which to extract columns */
/*     \param nkeep [in] : number of columns to extract */

/*     \return qm - qmarray */
/* **************************************************************\/ */
/* struct Qmarray * qmarray_extract_ncols(struct Qmarray * a, size_t nkeep) */
/* { */
    
/*     struct Qmarray * qm = qmarray_alloc(a->nrows,nkeep); */
/*     size_t ii,jj; */
/*     for (ii = 0; ii < nkeep; ii++){ */
/*         for (jj = 0; jj < a->nrows; jj++){ */
/*             qm->funcs[ii*a->nrows+jj] = generic_function_copy(a->funcs[ii*a->nrows+jj]); */
/*         } */
/*     } */
/*     return qm; */
/* } */

/////////////////////////////////////////////////////////
// qm_array

/***********************************************************//**
    Create a qmarray by approximating 1d functions

    \param[in] nrows - number of rows of quasimatrix
    \param[in] ncols - number of cols of quasimatrix
    \param[in] funcs - functions (nrows*ncols)
    \param[in] args  - extra arguments to each function
    \param[in] fc    - function class of each column
    \param[in] st    - sub_type of each column
    \param[in] lb    - lower bound of inputs to functions
    \param[in] ub    - upper bound of inputs to functions
    \param[in] aopts - approximation options

    \return qmarray
***************************************************************/
struct Qmarray * 
qmarray_approx1d(size_t nrows, size_t ncols,
                 double (**funcs)(double, void *),
                 void ** args, 
                 enum function_class fc, void * st, double lb,
                 double ub, void * aopts)
{
    struct Qmarray * qm = qmarray_alloc(nrows,ncols);
    size_t ii;
    for (ii = 0; ii < nrows*ncols; ii++){
        qm->funcs[ii] = 
            generic_function_approximate1d(funcs[ii], 
                                           args[ii], 
                                           fc, st, lb, ub, aopts);
    }
    return qm;
}

/***********************************************************//**
    Create a qmarray from a fiber_cuts array

    \param[in] nrows    - number of rows of qmarray
    \param[in] ncols    - number of columns of qmarray
    \param[in] f        - functions
    \param[in] fcut     - array of fiber cuts
    \param[in] fc       - function class of each column
    \param[in] sub_type - sub_type of each column
    \param[in] lb       - lower bound of inputs to functions
    \param[in] ub       - upper bound of inputs to functions
    \param[in] aopts    - approximation options

    \return qmarray
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

    \param[in] fc    - function class
    \param[in] st    - function class sub_type
    \param[in] nrows - number of rows
    \param[in] ncols - number of columns
    \param[in] lb    - lower bound on 1d functions
    \param[in] ub    - upper bound on 1d functions

    \return qmarray with orthonormal columns

    \note
        - Not super efficient because of copies
***************************************************************/
struct Qmarray *
qmarray_orth1d_columns(enum function_class fc, void * st, size_t nrows,
                       size_t ncols, double lb, double ub)
{

    struct Qmarray * qm = qmarray_alloc(nrows,ncols);
    struct Qmarray * qmtemp = qmarray_alloc(ncols,1);
    generic_function_array_orth1d_columns(qm->funcs,qmtemp->funcs,
                                          fc,st,nrows,ncols,lb,ub);
    
    qmarray_free(qmtemp); qmtemp = NULL;
    return qm;
}

/***********************************************************//**
    Generate a qmarray with orthonormal columns consisting of
    one dimensional functions of linear elements on a certain grid

    \param[in] nrows - number of rows
    \param[in] ncols - number of columns
    \param[in] grid  - grid size

    \return qmarray with orthonormal columns

    \note
        - Not super efficient because of copies
***************************************************************/
struct Qmarray *
qmarray_orth1d_linelm_grid(size_t nrows,size_t ncols, struct c3Vector * grid)
{
    struct Qmarray * qm = qmarray_alloc(nrows,ncols);
    generic_function_array_orth1d_linelm_columns(qm->funcs,nrows,ncols,grid);
    return qm;
}

/***********************************************************//**
    Generate a qmarray with orthonormal rows

    \param[in] fc    - function class
    \param[in] st    - function class sub_type
    \param[in] nrows -  number of rows
    \param[in] ncols -  number of columns
    \param[in] lb    - lower bound on 1d functions
    \param[in] ub    - upper bound on 1d functions

    \return qmarray with orthonormal rows

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
            onorder = onorder+1;
            onnon = 0;
        }
    }
    //printf("max order rows = %zu\n",onorder);
    
    for (ii = 0; ii < nrows;ii++){
        generic_function_free(funcs[ii]);
        funcs[ii] = NULL;
    }
    free(funcs); funcs=NULL;
    generic_function_free(zero); zero = NULL;

    return qm;
}



/***********************************************************//**
    Serialize a qmarray

    \param[in,out] ser       - stream to serialize to
    \param[in]     qma       - quasimatrix array
    \param[in,out] totSizeIn - if NULL then serialize, if not NULL
                               then return size;

    \return ser shifted by number of bytes
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
struct FtOneApprox * 
ft_one_approx_alloc(enum function_class fc, void * sub_type, 
                    void * aopts)
{

    struct FtOneApprox * app = malloc(sizeof(struct FtOneApprox));
    if (app == NULL){
        fprintf(stderr,"Cannot allocate FtOneApprox\n");
        exit(1);
    }
    app->fc = fc;
    app->sub_type = sub_type;
    app->aopts = aopts;
    return app;
}

void ft_one_approx_free(struct FtOneApprox * oa)
{
    if (oa != NULL){
        free(oa); oa = NULL;
    }
}

struct FtApproxArgs * ft_approx_args_alloc(size_t dim)
{
    struct FtApproxArgs * fargs;
    if ( NULL == (fargs = malloc(sizeof(struct FtApproxArgs)))){
        fprintf(stderr, "Cannot allocate space for FtApproxArgs.\n");
        exit(1);
    }
    fargs->aopts = malloc(dim * sizeof(struct FtOneApprox *));
    if (fargs->aopts == NULL){
        fprintf(stderr, "Cannot allocat FtApproxArgs\n");
        exit(1);
    }
    fargs->dim = dim;
    for (size_t ii = 0; ii < dim; ii++){
        fargs->aopts[ii] = NULL;
    }

    return fargs;
}

/***********************************************************//**
    Free memory allocated to FtApproxArgs

    \param[in,out] fargs - function train approximation arguments
***************************************************************/
void ft_approx_args_free(struct FtApproxArgs * fargs)
{
    if (fargs != NULL){
        for (size_t ii = 0; ii < fargs->dim;ii++){
            ft_one_approx_free(fargs->aopts[ii]);
            fargs->aopts[ii] = NULL;
        }
        free(fargs->aopts); fargs->aopts = NULL;
        free(fargs); fargs = NULL;
    }
}

/***********************************************************//**
    Create the arguments to give to use for approximation
    in the function train. Specifically, legendre polynomials
    for all dimensions

    \param[in] dim   - dimension of function train
    \param[in] ptype - type of polynomial
    \param[in] aopts - arguments for creating the approximation 
                       (could be NULL)

    \return approximation arguments
***************************************************************/
struct FtApproxArgs * 
ft_approx_args_createpoly(size_t dim, 
                          enum poly_type * ptype,
                          struct OpeAdaptOpts * aopts)
{
    struct FtApproxArgs * fargs = ft_approx_args_alloc(dim);
    fargs->dim = dim;
    for (size_t ii = 0; ii < dim; ii++){
        fargs->aopts[ii] = ft_one_approx_alloc(POLYNOMIAL,ptype,aopts);
    }

    return fargs;
}

/***********************************************************//**
    Create the arguments to give to use for approximation
    in the function train. Specifically, piecewise legendre polynomials

    \param[in] dim   - dimension of function train
    \param[in] ptype - type of polynomial
    \param[in] aopts - arguments for creating the approximation 
                       (could be NULL)

    \return fargs - approximation arguments
***************************************************************/
struct FtApproxArgs * 
ft_approx_args_createpwpoly(size_t dim, 
                            enum poly_type * ptype,
                            struct PwPolyAdaptOpts * aopts)
{
    struct FtApproxArgs * fargs = ft_approx_args_alloc(dim);
    fargs->dim = dim;
    for (size_t ii = 0; ii < dim; ii++){
        fargs->aopts[ii] = ft_one_approx_alloc(PIECEWISE,ptype,aopts);
    }
    return fargs;
}

/***********************************************************//**
    Create the linear element approximation args to sent to
    function cross

    \param[in] dim   - dimension of function train
    \param[in] aopts - arguments for creating the approximation 
                       (could be NULL)

    \return fargs - approximation arguments
***************************************************************/
struct FtApproxArgs * 
ft_approx_args_create_le(size_t dim, 
                         struct LinElemExpAopts * aopts)
{
    struct FtApproxArgs * fargs = ft_approx_args_alloc(dim);
    fargs->dim = dim;
    int zero = 0;
    for (size_t ii = 0; ii < dim; ii++){
        fargs->aopts[ii] = ft_one_approx_alloc(LINELM,&zero,aopts);
    }
    return fargs;
}

/***********************************************************//**
    Create the linear element approximation args to sent to
    function cross

    \param[in] dim   - dimension of function train
    \param[in] aopts - arguments for creating the approximation 
                       (could be NULL)

    \return fargs - approximation arguments
***************************************************************/
struct FtApproxArgs * 
ft_approx_args_create_le2(size_t dim, 
                         struct LinElemExpAopts ** aopts)
{
    struct FtApproxArgs * fargs = ft_approx_args_alloc(dim);
    fargs->dim = dim;
    int zero = 0;
    for (size_t ii = 0; ii < dim; ii++){
        fargs->aopts[ii] = ft_one_approx_alloc(LINELM,&zero,aopts[ii]);
    }
    return fargs;
}

/***********************************************************//**
    Extract the function class to use for the approximation of the
    *dim*-th dimensional functions 

    \param[in] fargs - function train approximation arguments
    \param[in] dim   - dimension to extract

    \return function_class of the approximation
***************************************************************/
enum function_class 
ft_approx_args_getfc(struct FtApproxArgs * fargs, size_t dim)
{
    return fargs->aopts[dim]->fc;
}

/***********************************************************//**
    Extract the sub type to use for the approximation of the
    *dim*-th dimensional functions 

    \param[in] fargs - function train approximation arguments
    \param[in] dim   - dimension to extract

    \return sub type of the approximation
***************************************************************/
void * ft_approx_args_getst(struct FtApproxArgs * fargs, size_t dim)
{
    return fargs->aopts[dim]->sub_type;
}

/***********************************************************//**
    Extract the approximation arguments to use for the approximation of the
    *dim*-th dimensional functions 

    \param[in] fargs - function train approximation arguments
    \param[in] dim   - dimension to extract

    \return approximation arguments
***************************************************************/
void * ft_approx_args_getaopts(struct FtApproxArgs * fargs, size_t dim)
{
    return fargs->aopts[dim]->aopts;
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

    \param[in] dim - dimension of function train

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
    ft->evalspace1 = NULL;
    ft->evalspace2 = NULL;
    ft->evalspace3 = NULL;
    ft->evaldd1 = NULL;
    ft->evaldd2 = NULL;
    ft->evaldd3 = NULL;
    ft->evaldd4 = NULL;
    return ft;
}

/**********************************************************//**
    Copy a function train

    \param[in] a - function train to copy

    \return b - function train
**************************************************************/
struct FunctionTrain * function_train_copy(struct FunctionTrain * a){

    if (a == NULL){
        return NULL;
    }
    struct FunctionTrain * b = function_train_alloc(a->dim);
    
    size_t ii;
    for (ii = 0; ii < a->dim; ii++){
        b->ranks[ii] = a->ranks[ii];
        b->cores[ii] = qmarray_copy(a->cores[ii]);
    }
    b->ranks[a->dim] = a->ranks[a->dim];
    return b;
}

/**********************************************************//**
    Free memory allocated to a function train

    \param[in,out] ft - function train to free
**************************************************************/
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
        free(ft->evalspace1); ft->evalspace1 = NULL;
        free(ft->evalspace2); ft->evalspace2 = NULL;
        free(ft->evalspace3); ft->evalspace3 = NULL;
        free_dd(ft->dim,ft->evaldd1); ft->evaldd1 = NULL;
        /* free_dd(2*ft->dim,ft->evaldd2);   ft->evaldd2 = NULL; */
        free_dd(ft->dim,ft->evaldd3);     ft->evaldd3 = NULL;
        free_dd(ft->dim,ft->evaldd4);     ft->evaldd4 = NULL;
        free(ft); ft = NULL;
    }
}

/********************************************************//**
*    Create a functiontrain consisting of pseudo-random orth poly expansion
*   
*   \param[in] ptype    - polynomial type
*   \param[in] bds      - boundaries
*   \param[in] ranks    - (dim+1,1) array of ranks
*   \param[in] maxorder - maximum order of the polynomial
*
*   \return function train
************************************************************/
struct FunctionTrain *
function_train_poly_randu(enum poly_type ptype,struct BoundingBox * bds,
                          size_t * ranks, size_t maxorder)
{
    size_t dim = bds->dim;
    struct FunctionTrain * ft = function_train_alloc(dim);
    memmove(ft->ranks,ranks, (dim+1)*sizeof(size_t));

    size_t ii;
    for (ii = 0; ii < dim; ii++){
        ft->cores[ii] = 
            qmarray_poly_randu(ptype,ranks[ii],ranks[ii+1],maxorder,
                               bds->lb[ii],bds->ub[ii]);
    }
    return ft;
}

/***********************************************************//**
    Compute a function train representation of \f$ f1(x1)*f2(x2)*...* fd(xd)\f$

    \param[in] dim    - dimension of function train
    \param[in] f      -  one dimensional functions
    \param[in] args   - arguments to the function
    \param[in] bds    - boundarys of each dimension
    \param[in] ftargs - parameters for computation

    \return function train
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

    \param[in] dim    - dimension of function train
    \param[in] f      - array of one dimensional functions
    \param[in] args   - array of arguments to the functions
    \param[in] bds    - boundarys of each dimension
    \param[in] ftargs - parameters for computation

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

    \param[in] dim    - dimension of function train
    \param[in] f      - one dimensional function with args x and i where  
                        should denote which function to evaluate
    \param[in] args   - array of arguments to the functions
    \param[in] bds    - boundarys of each dimension
    \param[in] ftargs - parameters for computation

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
    Compute a function train representation of \f$ a \f$

    \param[in] fc       - function class
    \param[in] sub_type - sub type of function class
    \param[in] dim      - dimension of function train
    \param[in] a        - value of tensor train
    \param[in] bds      - boundarys of each dimension
    \param[in] aopts    - oapproximation options in each dimension

    \return function train

    \note 
    Puts the constant into the first core
***************************************************************/
struct FunctionTrain * 
function_train_constant(enum function_class fc,
                        void * sub_type, size_t dim, double a, 
                        struct BoundingBox * bds, void * aopts)
{
    struct FunctionTrain * ft = function_train_alloc(dim);
    
    size_t onDim = 0;
        //approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

    ft->ranks[onDim] = 1;
    ft->cores[onDim] = qmarray_alloc(1,1);
    ft->cores[onDim]->funcs[0] =
        generic_function_constant(a,fc,sub_type,bds->lb[onDim],bds->ub[onDim],
                                  aopts);

    for (onDim = 1; onDim < dim; onDim++){
        //approx_opts = ft_approx_args_getaopts(ftargs_use, onDim);

        ft->ranks[onDim] = 1;
        ft->cores[onDim] = qmarray_alloc(1,1);
        ft->cores[onDim]->funcs[0] = 
            generic_function_constant(1,fc,sub_type,
                                      bds->lb[onDim],bds->ub[onDim],aopts);
    }
    ft->ranks[dim] = 1;

    return ft;
}

 
/***********************************************************//**
    Compute a function train representation of \f$ a \f$ 

    \param[in] fta - approximation arguments for each dimension
    \param[in] a   - value of tensor train
    \param[in] bds - boundarys of each dimension

    \return function train

    \note 
    Puts the constant into the first core
***************************************************************/
struct FunctionTrain * 
function_train_constant_d(struct FtApproxArgs * fta, double a,
                          struct BoundingBox * bds)
{
    assert (fta != NULL);
    size_t dim = fta->dim;
    struct FunctionTrain * ft = function_train_alloc(dim);
    
    size_t onDim = 0;

    enum function_class fc = ft_approx_args_getfc(fta, onDim);
    void * sub_type = ft_approx_args_getst(fta, onDim);
    void * aopts = ft_approx_args_getaopts(fta,onDim);

    ft->ranks[onDim] = 1;
    ft->cores[onDim] = qmarray_alloc(1,1);
    ft->cores[onDim]->funcs[0] =
        generic_function_constant(a,fc,sub_type,bds->lb[onDim],bds->ub[onDim],
                                  aopts);

    for (onDim = 1; onDim < dim; onDim++){

        fc = ft_approx_args_getfc(fta, onDim);
        sub_type = ft_approx_args_getst(fta, onDim);
        aopts = ft_approx_args_getaopts(fta,onDim);

        ft->ranks[onDim] = 1;
        ft->cores[onDim] = qmarray_alloc(1,1);
        ft->cores[onDim]->funcs[0] = generic_function_constant(1,fc,sub_type,
                                     bds->lb[onDim],bds->ub[onDim],aopts);
    }
    ft->ranks[dim] = 1;

    return ft;
}

/***********************************************************//**
    Compute a tensor train representation of \f$ x_1c_1 + x_2c_2 + .... + x_dc_d \f$

    \param[in] fc       - function class
    \param[in] sub_type - sub type of function class
    \param[in] dim      - dimension of function train
    \param[in] bds      - boundarys of each dimension
    \param[in] coeffs   - slope of the function in each dimension
    \param[in] aopts    - approximation options

    \return function train
    
    \note same parameterization for each dimension
***************************************************************/
struct FunctionTrain * 
function_train_linear(enum function_class fc, void * sub_type,
                      size_t dim, struct BoundingBox * bds, double * coeffs, 
                      void * aopts)
{
    
    struct FunctionTrain * ft = function_train_alloc(dim);

    size_t onDim = 0;
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
                                        bds->ub[onDim], aopts);
        ft->cores[onDim]->funcs[1] = generic_function_constant(1.0, fc, sub_type, 
                                     bds->lb[onDim], bds->ub[onDim], aopts);
        
        for (onDim = 1; onDim < dim-1; onDim++){
            ft->ranks[onDim] = 2;
            ft->cores[onDim] = qmarray_alloc(2,2);

            ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], aopts);

            ft->cores[onDim]->funcs[1] = generic_function_linear(coeffs[onDim],0.0,
                                        fc, sub_type, bds->lb[onDim], 
                                        bds->ub[onDim], aopts);

            ft->cores[onDim]->funcs[2] = 
                generic_function_constant(0.0, fc, 
                                          sub_type,bds->lb[onDim],bds->ub[onDim],
                                          aopts);
            ft->cores[onDim]->funcs[3] = generic_function_constant(1.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], aopts);
        }

        onDim = dim-1;

        ft->ranks[onDim] = 2;
        ft->ranks[onDim+1] = 1;
        ft->cores[onDim] = qmarray_alloc(2,1);

        ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                    sub_type, bds->lb[onDim], bds->ub[onDim], aopts);
        ft->cores[onDim]->funcs[1] = generic_function_linear(coeffs[onDim], 0.0,
                                    fc, sub_type, bds->lb[onDim], 
                                    bds->ub[onDim], aopts);

    }
    else{
        ft->cores[onDim]->funcs[0] = generic_function_linear(coeffs[onDim], 0.0, 
                                        fc, sub_type, bds->lb[onDim], 
                                        bds->ub[onDim], aopts);
    }

    return ft;
}

/***********************************************************//**
    Compute a tensor train representation of 
    \f$ (x_1c_1+a_1) + (x_2c_2+a_2)  + .... + (x_dc_d+a_d) \f$

    \param[in] fc       - function class
    \param[in] sub_type - sub type of function class
    \param[in] dim      - dimension of function train
    \param[in] bds      - boundarys of each dimension
    \param[in] c        - slope of the function in each dimension (ldc x dim)
    \param[in] ldc      - increment between coefficients in array of c
    \param[in] a        - offset of the function in each dimension (ldc x dim)
    \param[in] lda      - increment between coefficients in array a
    \param[in] aopts    - approximation arguments

    \return function train
***************************************************************/
struct FunctionTrain * 
function_train_linear2(enum function_class fc, const void * sub_type, size_t dim,
                       const struct BoundingBox * bds, 
                       const double * c, size_t ldc, const double * a, size_t lda,
                       void * aopts)
{
    struct FunctionTrain * ft = function_train_alloc(dim);

    size_t onDim = 0;
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
                            aopts);
    
    if (dim > 1){

        ft->cores[onDim]->funcs[1] = 
            generic_function_constant(1.0, fc, sub_type, 
                bds->lb[onDim], bds->ub[onDim], aopts);
    
        for (onDim = 1; onDim < dim-1; onDim++){
            ft->ranks[onDim] = 2;
            ft->cores[onDim] = qmarray_alloc(2,2);

            ft->cores[onDim]->funcs[0] = 
                generic_function_constant(1.0, fc, 
                    sub_type, bds->lb[onDim], bds->ub[onDim], aopts);


            ft->cores[onDim]->funcs[1] = 
                generic_function_linear(c[onDim*ldc], a[onDim*lda], 
                            fc, sub_type, bds->lb[onDim], bds->ub[onDim],
                            aopts);

            ft->cores[onDim]->funcs[2] = 
                generic_function_constant(0.0, fc, 
                    sub_type, bds->lb[onDim], bds->ub[onDim], aopts);
            ft->cores[onDim]->funcs[3] = 
                generic_function_constant(1.0, fc, 
                    sub_type, bds->lb[onDim], bds->ub[onDim], aopts);
        }

        onDim = dim-1;

        ft->ranks[onDim] = 2;
        ft->ranks[onDim+1] = 1;
        ft->cores[onDim] = qmarray_alloc(2,1);

        ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                    sub_type, bds->lb[onDim], bds->ub[onDim], aopts);

        ft->cores[onDim]->funcs[1] = 
            generic_function_linear(c[onDim*ldc], a[onDim*lda], 
                            fc, sub_type, bds->lb[onDim], bds->ub[onDim],
                            aopts);
    }

    return ft;
}


/***********************************************************//**
    Compute a function train representation of \f$ (x-m)^T Q (x-m) \f$

    \param[in] fc       - function class
    \param[in] sub_type - sub type of function class
    \param[in] dim      - dimension of function train
    \param[in] bds      - boundarys of each dimension
    \param[in] coeffs   - Q matrix
    \param[in] m        - m (dim,)
    \param[in] aopts    - approximation options

    \returns function train

    \note
    Could be more efficient with a better distribution of ranks
***************************************************************/
struct FunctionTrain * 
function_train_quadratic(enum function_class fc, void * sub_type, size_t dim, 
                         struct BoundingBox * bds, 
                         double * coeffs, 
                         double * m, void * aopts)
{
    assert (dim > 1); //
    double temp;
    struct FunctionTrain * ft = function_train_alloc(dim);
    size_t kk,ll;
    size_t onDim = 0;

    ft->ranks[onDim] = 1;
    ft->cores[onDim] = qmarray_alloc(1,dim+1);
    
    // should be quadratic
    ft->cores[onDim]->funcs[0] = generic_function_quadratic(
                coeffs[onDim*dim+onDim], m[onDim],
                fc, sub_type, bds->lb[onDim], 
                bds->ub[onDim], aopts);

    for (kk = 1; kk < dim; kk++)
    {
        //printf("on dimension 1 (%zu/%zu)\n",kk,dim);
        temp = coeffs[onDim*dim+kk] + coeffs[kk*dim + onDim];
        ft->cores[onDim]->funcs[kk] = 
                generic_function_linear(temp,-temp*m[onDim], fc, 
                                    sub_type, bds->lb[onDim], 
                                    bds->ub[onDim], aopts);
    }

    ft->cores[onDim]->funcs[dim] = generic_function_constant(1.0, fc, sub_type, 
                                 bds->lb[onDim], bds->ub[onDim], aopts);
    
    for (onDim = 1; onDim < dim-1; onDim++){
        //printf("on dimension (%zu/%zu)\n",onDim+1,dim);

        ft->ranks[onDim] = dim-onDim+2;
        ft->cores[onDim] = qmarray_alloc(dim-onDim+2,dim-onDim+1);
        
        for (kk = 0; kk < (dim-onDim+1); kk++){ // loop over columns
            for (ll = 0; ll < (dim - onDim + 2); ll++){ //rows;
                if ( (ll == 0) && (kk == 0)){ // upper left corner
                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] = 
                            generic_function_constant(1.0, fc, 
                            sub_type, bds->lb[onDim], bds->ub[onDim], aopts);
                }
                else if ( (kk == 0) && (ll == 1) ){ // first element of lower diagonal
                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] = 
                            generic_function_linear(1.0,-m[onDim], fc, 
                                                sub_type, bds->lb[onDim], 
                                                bds->ub[onDim], aopts);
                }
                else if (ll == (kk+1)){ // lower diagonal
                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] = 
                            generic_function_constant(1.0, fc, 
                            sub_type, bds->lb[onDim], bds->ub[onDim],aopts);
                }
                else if ( (ll == (dim-onDim+1)) && (kk == 0)){ //lower left corner
                    //quadratic
                     ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] = 
                                generic_function_quadratic(
                                coeffs[onDim*dim+onDim], m[onDim],
                                fc, sub_type, bds->lb[onDim], 
                                bds->ub[onDim], aopts);
                }
                else if ( ll == (dim-onDim+1) ){ // rest of bottom row
                    temp = coeffs[onDim*dim+onDim+kk] + 
                                coeffs[(onDim+kk)*dim + onDim];

                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] = 
                        generic_function_linear(temp,-temp*m[onDim], fc, 
                                    sub_type, bds->lb[onDim], 
                                    bds->ub[onDim], aopts);
                }
                else{ // zeros
                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] = 
                            generic_function_constant(0.0, fc, 
                            sub_type, bds->lb[onDim], bds->ub[onDim], aopts);
                }
                //printf("generic function for core %zu num (%zu,%zu) \n",onDim,ll,kk);
                //print_generic_function(ft->cores[onDim]->funcs[kk*dim+ll],0,NULL);
            }
        }
    }

    onDim = dim-1;

    ft->ranks[onDim] = dim-onDim+2;
    ft->cores[onDim] = qmarray_alloc(dim-onDim+2,1);

    ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                sub_type, bds->lb[onDim], bds->ub[onDim], aopts);

    ft->cores[onDim]->funcs[1] = generic_function_linear(1.0, -m[onDim],
                                    fc, sub_type, bds->lb[onDim], 
                                    bds->ub[onDim], aopts);

    ft->cores[onDim]->funcs[2] = generic_function_quadratic(
                                coeffs[onDim*dim+onDim], m[onDim],
                                fc, sub_type, bds->lb[onDim], 
                                bds->ub[onDim], aopts);
    ft->ranks[dim] = 1;
    return ft;
}

/***********************************************************//**
    Compute a tensor train representation of \f[ (x_1-m_1)^2c_1 + (x_2-m_2)^2c_2 + .... + (x_d-m_d)^2c_d \f]

    \param[in] fc       - function class
    \param[in] sub_type - sub type of function class
    \param[in] bds      - boundarys of each dimension
    \param[in] coeffs   - coefficients for each dimension
    \param[in] m        - offset in each dimension
    \param[in] aopts    - approximation arguments

    \return a function train
***************************************************************/
struct FunctionTrain * 
function_train_quadratic_aligned(enum function_class fc, 
                                 void * sub_type,struct BoundingBox * bds, 
                                 double * coeffs, double * m,
                                 void * aopts)
{
    size_t dim = bds->dim;
    
    struct FunctionTrain * ft = function_train_alloc(dim);

    size_t onDim = 0;
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
                                    bds->ub[onDim], aopts);
        ft->cores[onDim]->funcs[1] = generic_function_constant(1.0, fc, sub_type, 
                                     bds->lb[onDim], bds->ub[onDim], aopts);
        for (onDim = 1; onDim < dim-1; onDim++){
            ft->ranks[onDim] = 2;
            ft->cores[onDim] = qmarray_alloc(2,2);

            ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], aopts);

            ft->cores[onDim]->funcs[1] = generic_function_quadratic(
                                coeffs[onDim], m[onDim],
                                fc, sub_type, bds->lb[onDim], 
                                bds->ub[onDim], aopts);

            ft->cores[onDim]->funcs[2] = generic_function_constant(0.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], aopts);
            ft->cores[onDim]->funcs[3] = generic_function_constant(1.0, fc, 
                        sub_type, bds->lb[onDim], bds->ub[onDim], aopts);
        }

        onDim = dim-1;

        ft->ranks[onDim] = 2;
        ft->ranks[onDim+1] = 1;
        ft->cores[onDim] = qmarray_alloc(2,1);

        ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, fc, 
                    sub_type, bds->lb[onDim], bds->ub[onDim], aopts);
        ft->cores[onDim]->funcs[1] = generic_function_quadratic(
                            coeffs[onDim], m[onDim],
                            fc, sub_type, bds->lb[onDim], 
                            bds->ub[onDim], aopts);

    }
    else{
        ft->cores[onDim]->funcs[0] = generic_function_quadratic(
                                    coeffs[onDim], m[onDim],
                                    fc, sub_type, bds->lb[onDim], 
                                    bds->ub[onDim], aopts);
    }

    return ft;
}


/***********************************************************//**
    Serialize a function_train

    \param[in,out] ser       - stream to serialize to
    \param[in]     ft        - function train
    \param[in,out] totSizeIn - if NULL then serialize, if not NULL then return size

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

    \param[in,out] ser - serialized function train
    \param[in,out] ft  - function_train

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
    
    \param[in] ft       - function train to save
    \param[in] filename - name of file to save to

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
    
    \param[in] filename - filename of file to load

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

    \param[in] dimout - number of function trains

    \return function train array

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
    Serialize a function train array

    \param[in,out] ser        - stream to serialize to
    \param[in]     ft         - function train array
    \param[in,out] totSizeIn  - if NULL then serialize, if not NULL then return size

    \return ptr - ser shifted by number of bytes
***************************************************************/
unsigned char * 
ft1d_array_serialize(unsigned char * ser, struct FT1DArray * ft,
                     size_t *totSizeIn)
{

    // size -> ft1 -> ft2 -> ... ->ft[size]

    // size
    size_t totSize = sizeof(size_t);
    size_t size_temp;
    for (size_t ii = 0; ii < ft->size; ii++){
        function_train_serialize(NULL,ft->ft[ii],&size_temp);
        totSize += size_temp;
    }
    if (totSizeIn != NULL){
        *totSizeIn = totSize;
        return ser;
    }

    // serialize the size
    unsigned char * ptr = ser;
    ptr = serialize_size_t(ptr, ft->size);
    
    // serialize each function train
    for (size_t ii = 0; ii < ft->size; ii++){
        ptr = function_train_serialize(ptr,ft->ft[ii],NULL);
    }
    
    return ptr;
}

/***********************************************************//**
    Deserialize a function train array

    \param[in,out] ser - serialized function train array
    \param[in,out] ft  - function_train array

    \return ptr - shifted ser after deserialization
***************************************************************/
unsigned char *
ft1d_array_deserialize(unsigned char * ser, struct FT1DArray ** ft)
{
    unsigned char * ptr = ser;

    // deserialize the number of fts in the array
    size_t size;
    ptr = deserialize_size_t(ptr, &size);
    *ft = ft1d_array_alloc(size);

    // deserialize each function train
    for (size_t ii = 0; ii < size; ii++){
        ptr = function_train_deserialize(ptr, &((*ft)->ft[ii]));
    }
    return ptr;
}

/***********************************************************//**
    Save a function train array to file
    
    \param[in] ft       - function train array to save
    \param[in] filename - name of file to save to

    \return success (1) or failure (0) of opening the file
***************************************************************/
int ft1d_array_save(struct FT1DArray * ft, char * filename)
{

    FILE *fp;
    fp = fopen(filename, "w");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open %s\n", filename);
        return 0;
    }
    
    size_t totsize;
    ft1d_array_serialize(NULL,ft, &totsize);

    unsigned char * data = malloc(totsize+sizeof(size_t));
    if (data == NULL){
        fprintf(stderr, "can't allocate space for saving density\n");
        return 0;
    }
    
    // serialize size first!
    unsigned char * ptr = serialize_size_t(data,totsize);
    ptr = ft1d_array_serialize(ptr,ft,NULL);

    fwrite(data,sizeof(unsigned char),totsize+sizeof(size_t),fp);

    free(data); data = NULL;
    fclose(fp);
    return 1;
}

/***********************************************************//**
    Load a function train array from a file
    
    \param[in] filename - filename of file to load

    \return ft if successfull NULL otherwise
***************************************************************/
struct FT1DArray * ft1d_array_load(char * filename)
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

    struct FT1DArray * ft = NULL;
    ft1d_array_deserialize(data,&ft);
    
    free(data); data = NULL;
    fclose(fp);
    return ft;
}

/***********************************************************//**
    Copy an array of function trains

    \param[in] fta - array to coppy

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

    \param[in,out] fta - function train array to free
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

/***********************************************************//**
  Allocate fiber optimization options
***************************************************************/
struct FiberOptArgs * fiber_opt_args_alloc()
{
    struct FiberOptArgs * fopt = NULL;
    fopt = malloc(sizeof(struct FiberOptArgs));
    if (fopt == NULL){
        fprintf(stderr,"Memory failure allocating FiberOptArgs\n");
        exit(1);
    }
    return fopt;
}

/***********************************************************//**
    Initialize a baseline optimization arguments class

    \param[in] dim - dimension of problem

    \return fiber optimzation arguments that are NULL in each dimension
***************************************************************/
struct FiberOptArgs * fiber_opt_args_init(size_t dim)
{
    struct FiberOptArgs * fopt = fiber_opt_args_alloc();
    fopt->dim = dim;
    
    fopt->opts = malloc(dim * sizeof(void *));
    if (fopt->opts == NULL){
        fprintf(stderr,"Memory failure initializing fiber opt args\n");
        exit(1);
    }
    for (size_t ii = 0; ii < dim; ii++){
        fopt->opts[ii] = NULL;
    }
    return fopt;
}

/***********************************************************//**
    Initialize a bruteforce optimization with the same nodes 
    in each dimension

    \param[in] dim   - dimension of problem
    \param[in] nodes - nodes over which to optimize 
                       (same ones used for each dimension)

    \return fiber opt args
***************************************************************/
struct FiberOptArgs * 
fiber_opt_args_bf_same(size_t dim, struct c3Vector * nodes)
{
    struct FiberOptArgs * fopt = fiber_opt_args_alloc();
    fopt->dim = dim;
    
    fopt->opts = malloc(dim * sizeof(void *));
    if (fopt->opts == NULL){
        fprintf(stderr,"Memory failure initializing fiber opt args\n");
        exit(1);
    }
    for (size_t ii = 0; ii < dim; ii++){
        fopt->opts[ii] = nodes;
    }
    return fopt;
}

/***********************************************************//**
    Initialize a bruteforce optimization with different nodes
    in each dimension

    \param[in] dim   - dimension of problem
    \param[in] nodes - nodes over which to optimize 
                       (same ones used for each dimension)

    \return fiber opt args
***************************************************************/
struct FiberOptArgs * 
fiber_opt_args_bf(size_t dim, struct c3Vector ** nodes)
{
    struct FiberOptArgs * fopt = fiber_opt_args_alloc();
    fopt->dim = dim;
    
    fopt->opts = malloc(dim * sizeof(void *));
    if (fopt->opts == NULL){
        fprintf(stderr,"Memory failure initializing fiber opt args\n");
        exit(1);
    }
    for (size_t ii = 0; ii < dim; ii++){
        fopt->opts[ii] = nodes[ii];
    }
    return fopt;
}

/***********************************************************//**
    Free memory allocate to fiber optimization arguments

    \param[in,out] fopt - fiber optimization structure
***************************************************************/
void fiber_opt_args_free(struct FiberOptArgs * fopt)
{
    if (fopt != NULL){
        free(fopt->opts); fopt->opts = NULL;
        free(fopt); fopt = NULL;
    }
}


/////////////////////////////////////////////////////////
// Utilities
//


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

