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

/** \file ft.c
 * Provides algorithms for function train manipulation
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "stringmanip.h"
#include "array.h"
#include "linalg.h"
#include "lib_funcs.h"

#include "ft.h"

#ifndef ZERODEF
#define ZEROTHRESH 1e2*DBL_EPSILON
#endif

/** \struct FunctionTrain
 * \brief Functrain train
 * \var FunctionTrain::dim
 * dimension of function
 * \var FunctionTrain::ranks
 * function train ranks
 * \var FunctionTrain::cores
 * function train cores
 */
struct FunctionTrain {
    size_t dim;
    size_t * ranks;
    struct Qmarray ** cores;
    
    double * evalspace1;
    double * evalspace2;
    double * evalspace3;

    double ** evaldd1;
    double ** evaldd2;
    double ** evaldd3;
    double ** evaldd4;
};

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
struct FunctionTrain * function_train_copy(const struct FunctionTrain * a)
{

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
    Get ranks                                                            
***************************************************************/
size_t * function_train_get_ranks(const struct FunctionTrain * ft)
{
    assert (ft != NULL);
    assert (ft->ranks != NULL);
    return ft->ranks;
}

/***********************************************************//**
    Computes the maximum rank of a FT
    
    \param[in] ft - function train

    \return maxrank
***************************************************************/
size_t function_train_get_maxrank(const struct FunctionTrain * ft)
{
    assert (ft != NULL);
    assert (ft->ranks != NULL);

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
double function_train_get_avgrank(const struct FunctionTrain * ft)
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


/***********************************************************//**
    Evaluate a function train

    \param[in] ft - function train
    \param[in] x  - location at which to evaluate

    \return val - value of the function train
***************************************************************/
double function_train_eval(struct FunctionTrain * ft, const double * x)
{
    
    size_t dim = ft->dim;
    assert(ft->ranks[0] == 1);
    assert(ft->ranks[dim] == 1);

    size_t ii = 0;
    size_t maxrank = function_train_get_maxrank(ft);
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
double function_train_eval_co_perturb(struct FunctionTrain * ft, 
                                      const double * x, const double * pert, 
                                      double * vals)
{
    
    size_t dim = ft->dim;
    assert(ft->ranks[0] == 1);
    assert(ft->ranks[dim] == 1);

    size_t maxrank = function_train_get_maxrank(ft);
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
*    Create a function train consisting of pseudo-random orth poly expansion
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
    size_t dim = bounding_box_get_dim(bds);
    double * lb = bounding_box_get_lb(bds);
    double * ub = bounding_box_get_ub(bds);
    struct FunctionTrain * ft = function_train_alloc(dim);
    memmove(ft->ranks,ranks, (dim+1)*sizeof(size_t));

    size_t ii;
    for (ii = 0; ii < dim; ii++){
        ft->cores[ii] = 
            qmarray_poly_randu(ptype,ranks[ii],ranks[ii+1],maxorder,
                               lb[ii],ub[ii]);
    }
    return ft;
}


/***********************************************************//**
    Compute a function train representation of \f$ f1(x1)*f2(x2)*...* fd(xd)\f$

    \param[in] ftargs - approximation options
    \param[in] fw     - wrapped functions

    \return function train
***************************************************************/
struct FunctionTrain *
function_train_rankone(struct MultiApproxOpts * ftargs, struct Fwrap * fw)
{
    assert (ftargs != NULL);
    assert (fw != NULL);
    size_t dim = multi_approx_opts_get_dim(ftargs);
    struct OneApproxOpts * aopt;

    struct FunctionTrain * ft = function_train_alloc(dim);
    size_t onDim;
    for (onDim = 0; onDim < dim; onDim++){

        aopt = multi_approx_opts_get_aopts(ftargs,onDim);
        
        fwrap_set_which_eval(fw,onDim);

        ft->ranks[onDim] = 1;
        ft->cores[onDim] = qmarray_alloc(1,1);
        ft->cores[onDim]->funcs[0] = 
            generic_function_approximate1d(aopt->fc,fw,aopt->aopts);
    }
    ft->ranks[dim] = 1;
    return ft;
}

/***********************************************************//**
    Compute a function train representation of \f$ f1 + f2 + ... + fd \f$

    \param[in] ftargs - approximation options
    \param[in] fw     - wrapped functions

    \return ft - function train
***************************************************************/
struct FunctionTrain *
function_train_initsum(struct MultiApproxOpts * ftargs, struct Fwrap * fw)
{
    assert (ftargs != NULL);
    assert (fw != NULL);
    size_t dim = multi_approx_opts_get_dim(ftargs);
    struct OneApproxOpts * aopt;


    struct FunctionTrain * ft = function_train_alloc(dim);

    size_t onDim = 0;
    fwrap_set_which_eval(fw,onDim);
    aopt = multi_approx_opts_get_aopts(ftargs, onDim);
    ft->ranks[onDim] = 1;
    if (dim == 1){
        ft->cores[onDim] = qmarray_alloc(1,1);
    }
    else{
        ft->cores[onDim] = qmarray_alloc(1,2);
    }

    ft->cores[onDim]->funcs[0] = 
        generic_function_approximate1d(aopt->fc,fw,aopt->aopts);
    ft->cores[onDim]->funcs[1] = generic_function_constant(1.0, aopt->fc, 
                                                           aopt->aopts);

    if (dim > 1){
        for (onDim = 1; onDim < dim-1; onDim++){

            fwrap_set_which_eval(fw,onDim);
            aopt = multi_approx_opts_get_aopts(ftargs, onDim);
            ft->ranks[onDim] = 2;
            ft->cores[onDim] = qmarray_alloc(2,2);

            ft->cores[onDim]->funcs[0] = 
                generic_function_constant(1.0,aopt->fc,aopt->aopts); 

            ft->cores[onDim]->funcs[1] =
                generic_function_approximate1d(aopt->fc,fw,aopt->aopts);

            ft->cores[onDim]->funcs[2] = 
                generic_function_constant(0.0, aopt->fc, aopt->aopts);

            ft->cores[onDim]->funcs[3] = 
                generic_function_constant(1.0, aopt->fc, aopt->aopts);
        }

        onDim = dim-1;
        fwrap_set_which_eval(fw,onDim);
        aopt = multi_approx_opts_get_aopts(ftargs, onDim);

        ft->ranks[onDim] = 2;
        ft->ranks[onDim+1] = 1;
        ft->cores[onDim] = qmarray_alloc(2,1);

        ft->cores[onDim]->funcs[0] = 
            generic_function_constant(1.0, aopt->fc, aopt->aopts);

        ft->cores[onDim]->funcs[1] = 
            generic_function_approximate1d(aopt->fc,fw,aopt->aopts);
    }

    return ft;
}
 
/***********************************************************//**
    Compute a function train representation of \f$ a \f$

    \param[in] a    - value
    \param[in] opts - approximation options

    \return function train

    \note
    Puts the constant into the first core
***************************************************************/
struct FunctionTrain *
function_train_constant(double a, struct MultiApproxOpts * aopts)
{
    assert (aopts != NULL);
    size_t dim = multi_approx_opts_get_dim(aopts);
    struct FunctionTrain * ft = function_train_alloc(dim);
    
    size_t onDim = 0;

    struct OneApproxOpts * oneopt = multi_approx_opts_get_aopts(aopts,onDim);

    ft->ranks[onDim] = 1;
    ft->cores[onDim] = qmarray_alloc(1,1);
    ft->cores[onDim]->funcs[0] =
        generic_function_constant(a,oneopt->fc,oneopt->aopts);

    for (onDim = 1; onDim < dim; onDim++){
        oneopt = multi_approx_opts_get_aopts(aopts,onDim);
        ft->ranks[onDim] = 1;
        ft->cores[onDim] = qmarray_alloc(1,1);
        ft->cores[onDim]->funcs[0] =
            generic_function_constant(1,oneopt->fc,oneopt->aopts);
    }
    ft->ranks[dim] = 1;

    return ft;
}

/***********************************************************//**
    Compute a tensor train representation of

    \f$ (x_1c_1+a_1) + (x_2c_2+a_2)  + .... + (x_dc_d+a_d) \f$

    \param[in] c    - slope of the function in each dimension (ldc x dim)
    \param[in] ldc  - stride of c
    \param[in] a    - value
    \param[in] lda  - stride of a
    \param[in] opts - approximation options

    \return function train
***************************************************************/
struct FunctionTrain *
function_train_linear(const double * c, size_t ldc, 
                      const double * a, size_t lda,
                      struct MultiApproxOpts * aopts)
{

    assert (aopts != NULL);
    size_t dim = multi_approx_opts_get_dim(aopts);
    struct FunctionTrain * ft = function_train_alloc(dim);

    size_t onDim = 0;
    ft->ranks[onDim] = 1;
    if (dim == 1){
        ft->cores[onDim] = qmarray_alloc(1,1);
    }
    else{
        ft->cores[onDim] = qmarray_alloc(1,2);
    }
    
    struct OneApproxOpts * o = NULL;
    o = multi_approx_opts_get_aopts(aopts,onDim);

    ft->cores[onDim]->funcs[0] =
        generic_function_linear(c[onDim*ldc], a[onDim*lda],o->fc,o->aopts);
    
    if (dim > 1){

        ft->cores[onDim]->funcs[1] =
            generic_function_constant(1.0, o->fc,o->aopts);
    
        for (onDim = 1; onDim < dim-1; onDim++){
            o = multi_approx_opts_get_aopts(aopts,onDim);
            ft->ranks[onDim] = 2;
            ft->cores[onDim] = qmarray_alloc(2,2);

            ft->cores[onDim]->funcs[0] =
                generic_function_constant(1.0, o->fc,o->aopts);

            ft->cores[onDim]->funcs[1] =
                generic_function_linear(c[onDim*ldc], a[onDim*lda],
                                        o->fc,o->aopts);

            ft->cores[onDim]->funcs[2] =
                generic_function_constant(0.0, o->fc, o->aopts);

            ft->cores[onDim]->funcs[3] =
                generic_function_constant(1.0, o->fc, o->aopts);
        }

        onDim = dim-1;
        o = multi_approx_opts_get_aopts(aopts,onDim);
        ft->ranks[onDim] = 2;
        ft->ranks[onDim+1] = 1;
        ft->cores[onDim] = qmarray_alloc(2,1);

        ft->cores[onDim]->funcs[0] = 
            generic_function_constant(1.0, o->fc, o->aopts);

        ft->cores[onDim]->funcs[1] =
            generic_function_linear(c[onDim*ldc], a[onDim*lda],
                                    o->fc, o->aopts);
    }

    return ft;
}


/***********************************************************//**
    Compute a function train representation of \f$ (x-m)^T Q (x-m) \f$

    \param[in] coeffs   - Q matrix
    \param[in] m        - m (dim,)
    \param[in] aopts    - approximation options

    \returns function train

    \note
    Could be more efficient with a better distribution of ranks
***************************************************************/
struct FunctionTrain *
function_train_quadratic(const double * coeffs,
                         const double * m, struct MultiApproxOpts * aopts)
{
    assert (aopts != NULL);
    size_t dim = multi_approx_opts_get_dim(aopts);
    assert (dim > 1); //

    struct FunctionTrain * ft = function_train_alloc(dim);
    struct OneApproxOpts * o = NULL;

    double temp;
    size_t kk,ll;
    size_t onDim = 0;

    ft->ranks[onDim] = 1;
    ft->cores[onDim] = qmarray_alloc(1,dim+1);

    o = multi_approx_opts_get_aopts(aopts,onDim);

    // should be quadratic
    ft->cores[onDim]->funcs[0] = 
        generic_function_quadratic(coeffs[onDim*dim+onDim], m[onDim],
                                   o->fc,o->aopts);

    for (kk = 1; kk < dim; kk++)
    {
        //printf("on dimension 1 (%zu/%zu)\n",kk,dim);
        temp = coeffs[onDim*dim+kk] + coeffs[kk*dim + onDim];
        ft->cores[onDim]->funcs[kk] =
            generic_function_linear(temp,-temp*m[onDim], o->fc,o->aopts);
    }

    ft->cores[onDim]->funcs[dim] = generic_function_constant(1.0,o->fc,o->aopts);

    for (onDim = 1; onDim < dim-1; onDim++){
        //printf("on dimension (%zu/%zu)\n",onDim+1,dim);

        o = multi_approx_opts_get_aopts(aopts,onDim);
        ft->ranks[onDim] = dim-onDim+2;
        ft->cores[onDim] = qmarray_alloc(dim-onDim+2,dim-onDim+1);
        
        for (kk = 0; kk < (dim-onDim+1); kk++){ // loop over columns
            for (ll = 0; ll < (dim - onDim + 2); ll++){ //rows;
                if ( (ll == 0) && (kk == 0)){ // upper left corner
                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] =
                        generic_function_constant(1.0, o->fc,o->aopts);
                }
                else if ( (kk == 0) && (ll == 1) ){ // first element of lower diagonal
                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] =
                            generic_function_linear(1.0,-m[onDim], o->fc,
                                                    o->aopts);
                }
                else if (ll == (kk+1)){ // lower diagonal
                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] =
                        generic_function_constant(1.0, o->fc,o->aopts);
                }
                else if ( (ll == (dim-onDim+1)) && (kk == 0)){ //lower left corner
                    //quadratic
                     ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] =
                                generic_function_quadratic(
                                coeffs[onDim*dim+onDim], m[onDim],
                                o->fc, o->aopts);
                }
                else if ( ll == (dim-onDim+1) ){ // rest of bottom row
                    temp = coeffs[onDim*dim+onDim+kk] +
                                coeffs[(onDim+kk)*dim + onDim];

                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] =
                        generic_function_linear(temp,-temp*m[onDim],o->fc,o->aopts);
                }
                else{ // zeros
                    ft->cores[onDim]->funcs[kk*ft->cores[onDim]->nrows+ll] =
                        generic_function_constant(0.0,o->fc,o->aopts);
                }
            }
        }
    }

    onDim = dim-1;
    o = multi_approx_opts_get_aopts(aopts,onDim);
    ft->ranks[onDim] = dim-onDim+2;
    ft->cores[onDim] = qmarray_alloc(dim-onDim+2,1);

    ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, o->fc,o->aopts);
    ft->cores[onDim]->funcs[1] = generic_function_linear(1.0, -m[onDim],
                                                         o->fc,o->aopts);
    ft->cores[onDim]->funcs[2] = generic_function_quadratic(
                                coeffs[onDim*dim+onDim], m[onDim],
                                o->fc, o->aopts);
    ft->ranks[dim] = 1;
    return ft;
}

/***********************************************************//**
    Compute a tensor train representation of \f[ (x_1-m_1)^2c_1 + (x_2-m_2)^2c_2 + .... + (x_d-m_d)^2c_d \f]

    \param[in] coeffs   - coefficients for each dimension
    \param[in] m        - offset in each dimension
    \param[in] aopts    - approximation arguments

    \return a function train
***************************************************************/
struct FunctionTrain *
function_train_quadratic_aligned(const double * coeffs, const double * m,
                                 struct MultiApproxOpts * aopts)
{  

    assert (aopts != NULL);
    size_t dim = multi_approx_opts_get_dim(aopts);
    

    struct FunctionTrain * ft = function_train_alloc(dim);
    struct OneApproxOpts * o = NULL;

    size_t onDim = 0;
    ft->ranks[onDim] = 1;
    if (dim == 1){
        ft->cores[onDim] = qmarray_alloc(1,1);
    }
    else{
        ft->cores[onDim] = qmarray_alloc(1,2);
    }
    
    if (dim > 1){
        o = multi_approx_opts_get_aopts(aopts,onDim);

        ft->cores[onDim]->funcs[0] = 
            generic_function_quadratic(coeffs[onDim], m[onDim],o->fc,o->aopts);

        ft->cores[onDim]->funcs[1] = 
            generic_function_constant(1.0, o->fc, o->aopts);

        for (onDim = 1; onDim < dim-1; onDim++){
            o = multi_approx_opts_get_aopts(aopts,onDim);
            ft->ranks[onDim] = 2;
            ft->cores[onDim] = qmarray_alloc(2,2);

            ft->cores[onDim]->funcs[0] = generic_function_constant(1.0,
                                                                   o->fc,o->aopts);

            ft->cores[onDim]->funcs[1] = 
                generic_function_quadratic(coeffs[onDim], m[onDim],
                                           o->fc, o->aopts);


            ft->cores[onDim]->funcs[2] = generic_function_constant(0.0, o->fc,
                                                                   o->aopts);

            ft->cores[onDim]->funcs[3] = generic_function_constant(1.0, o->fc,
                                                                   o->aopts);
        }

        onDim = dim-1;

        ft->ranks[onDim] = 2;
        ft->ranks[onDim+1] = 1;
        ft->cores[onDim] = qmarray_alloc(2,1);

        ft->cores[onDim]->funcs[0] = generic_function_constant(1.0, o->fc,
                                                               o->aopts);

        ft->cores[onDim]->funcs[1] = 
            generic_function_quadratic(coeffs[onDim], m[onDim],o->fc,o->aopts);
    }
    else{
        o = multi_approx_opts_get_aopts(aopts,onDim);
        ft->cores[onDim]->funcs[0] = 
            generic_function_quadratic(coeffs[onDim], m[onDim],o->fc,o->aopts);
    }

    return ft;
}

/********************************************************//**
    Integrate a function in function train format

    \param[in] ft - Function train 1

    \return Integral \f$ \int f(x) dx \f$
***********************************************************/
double
function_train_integrate(const struct FunctionTrain * ft)
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
double function_train_inner(const struct FunctionTrain * a, 
                            const struct FunctionTrain * b)
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
double function_train_norm2(const struct FunctionTrain * a)
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
    Right orthogonalize the cores (except the first one) of the function train

    \param[in,out] a     - FT (overwritten)
    \param[in]     aopts - approximation options to QR
    
    \return ftrl - new ft with orthogonalized cores
***********************************************************/
struct FunctionTrain * 
function_train_orthor(struct FunctionTrain * a, 
                      struct MultiApproxOpts * aopts)
{
    struct OneApproxOpts * o = NULL;
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
    o = multi_approx_opts_get_aopts(aopts,core);
    ftrl->cores[core] = qmarray_householder_simple("LQ",a->cores[core],L,o);
    for (ii = 2; ii < a->dim; ii++){
        core = a->dim-ii;
        //printf("on core %zu\n",core);
        temp = qmam(a->cores[core],L,ftrl->ranks[core+1]);
        free(L); L = NULL;
        L = calloc_double(ftrl->ranks[core]*ftrl->ranks[core]);
        o = multi_approx_opts_get_aopts(aopts,core);
        ftrl->cores[core] = qmarray_householder_simple("LQ",temp,L,o);
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
    \param[in] aopts - approximation options to QR

    \return ft - rounded function train
***********************************************************/
struct FunctionTrain *
function_train_round(struct FunctionTrain * ain, double epsilon,
                     struct MultiApproxOpts * aopts)
{
    struct OneApproxOpts * o = NULL;
    struct FunctionTrain * a = function_train_copy(ain);
//    size_t ii;
    size_t core;
    struct Qmarray * temp = NULL;

    double delta = function_train_norm2(a);
    delta = delta * epsilon / sqrt(a->dim-1);
    //double delta = epsilon;

//    printf("begin orho\n");
    struct FunctionTrain * ftrl = function_train_orthor(a,aopts);
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
    o = multi_approx_opts_get_aopts(aopts,core);
    rank = qmarray_truncated_svd(ftrl->cores[core], &(ft->cores[core]),
                                 &s, &vt, delta,o);
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
        o = multi_approx_opts_get_aopts(aopts,core);
        rank = qmarray_truncated_svd(temp, &(ft->cores[core]), &s, &vt, delta,o);
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
struct FunctionTrain * 
function_train_sum(const struct FunctionTrain * a,
                   const struct FunctionTrain * b)
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
    Scale a function train representation

    \param[in,out] x - Function train
    \param[in]     a - scaling factor
***********************************************************/
void function_train_scale(struct FunctionTrain * x, double a)
{
    struct GenericFunction ** temp = generic_function_array_daxpby(
        x->cores[0]->nrows*x->cores[0]->ncols,a,1,x->cores[0]->funcs,0.0,1,NULL);
    generic_function_array_free(x->cores[0]->funcs,
                x->cores[0]->nrows * x->cores[0]->ncols);
    x->cores[0]->funcs = temp;
}

/* /\********************************************************\//\** */
/*     \f[ af(x) + b \f] */

/*     \param[in] a        - scaling factor */
/*     \param[in] b        - offset */
/*     \param[in] f        - object to scale */
/*     \param[in] epsilon  - rounding tolerance */

/*     \return ft - function representing a+b */
/* ***********************************************************\/ */
/* struct FunctionTrain *  */
/* function_train_afpb(double a, double b, */
/*                     const struct FunctionTrain * f, double epsilon) */
/* { */
/*     struct BoundingBox * bds = function_train_bds(f); */
/*     struct FunctionTrain * off = NULL; */
/*     if (f->cores[0]->funcs[0]->fc != LINELM){ */
/*         enum poly_type ptype = LEGENDRE; */
/*         off  = function_train_constant(POLYNOMIAL,&ptype, */
/*                                        f->dim,b,bds,NULL); */
/*     } */
/*     else{ */
/*         off = function_train_constant(LINELM,NULL, */
/*                                        f->dim,b,bds,NULL); */
/*     } */
/*     struct FunctionTrain * af = function_train_copy(f); */
/*     function_train_scale(af,a); */

/*     struct FunctionTrain * temp = function_train_sum(af,off); */
    
    
/*     //printf("round it \n"); */
/*     //printf("temp ranks \n"); */
/*     //iprint_sz(temp->dim+1,temp->ranks); */

/*     struct FunctionTrain * ft = function_train_round(temp,epsilon); */
/*     //printf("got it \n"); */
        
/*     function_train_free(off); off = NULL; */
/*     function_train_free(af); af = NULL; */
/*     function_train_free(temp); temp = NULL; */
/*     bounding_box_free(bds); bds = NULL; */
    
/*     return ft; */
/* } */

/********************************************************//**
    Product of two functions in function train form

    \param[in] a  - Function train 1
    \param[in] b  - Function train 2

    \return Product \f$ f(x) = a(x)b(x) \f$
***********************************************************/
struct FunctionTrain *
function_train_product(const struct FunctionTrain * a, 
                       const struct FunctionTrain * b)
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
    Compute the L2 norm of the difference between two functions

    \param[in] a - function train
    \param[in] b - function train 2

    \return L2 difference \f$ \sqrt{ \int (a(x)-b(x))^2 dx } \f$
***********************************************************/
double function_train_norm2diff(const struct FunctionTrain * a, 
                                const struct FunctionTrain * b)
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
    Compute the relative L2 norm of the difference between two functions

    \param[in] a - function train
    \param[in] b - function train 2

    \return Relative L2 difference
    \f$ \sqrt{ \int (a(x)-b(x))^2 dx } / \lVert b(x) \rVert \f$
***********************************************************/
double function_train_relnorm2diff(const struct FunctionTrain * a,
                                   const struct FunctionTrain * b)
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

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/////////////////////                          ///////////////////////////
/////////////////////    Starting FT1D Array   ///////////////////////////
/////////////////////                          ///////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


/** \struct FT1DArray
 * \brief One dimensional array of function trains
 * \var FT1DArray::size
 * size of array
 * \var FT1DArray::ft
 * array of function trains
 */
struct FT1DArray{
    size_t size;
    struct FunctionTrain ** ft;
};

/* /\********************************************************\//\** */
/*     Compute the gradient of a function train  */

/*     \param[in] ft - Function train  */

/*     \return gradient */
/* ***********************************************************\/ */
/* struct FT1DArray * function_train_gradient(struct FunctionTrain * ft) */
/* { */
/*     struct FT1DArray * ftg = ft1d_array_alloc(ft->dim); */
/*     size_t ii; */
/*     for (ii = 0; ii < ft->dim; ii++){ */
/*         //printf("**********\n\n\n**********\n"); */
/*         //printf("ii = %zu\n",ii); */
/*         ftg->ft[ii] = function_train_copy(ft); */
/*         qmarray_free(ftg->ft[ii]->cores[ii]); */
/*         ftg->ft[ii]->cores[ii] = NULL; */
/*         //printf("get deriv ii = %zu\n",ii); */
/*         //print_qmarray(ft->cores[ii],1,NULL); */
/*         ftg->ft[ii]->cores[ii] = qmarray_deriv(ft->cores[ii]); */
/*         //printf("got deriv ii = %zu\n",ii); */
/*         //print_qmarray(ftg->ft[ii]->cores[ii],1,NULL); */
/*     } */

/*     return ftg; */
/* } */

/* /\***********************************************************\//\** */
/*     Allocate a 1d array of function trains */

/*     \param[in] dimout - number of function trains */

/*     \return function train array */

/*     \note  */
/*         Each ft of the array is set to NULL; */
/* ***************************************************************\/ */
/* struct FT1DArray * ft1d_array_alloc(size_t dimout) */
/* { */
/*     struct FT1DArray * fta = malloc(sizeof(struct FT1DArray)); */
/*     if (fta == NULL){ */
/*         fprintf(stderr, "Error allocating 1d function train array"); */
/*         exit(1); */
/*     } */
    
/*     fta->size = dimout; */

/*     fta->ft = malloc(dimout * sizeof(struct FunctionTrain *)); */
/*     if (fta == NULL){ */
/*         fprintf(stderr, "Error allocating 1d function train array"); */
/*         exit(1); */
/*     } */
        
/*     size_t ii; */
/*     for (ii = 0; ii < dimout; ii ++){ */
/*         fta->ft[ii] = NULL; */
/*     } */
    
/*     return fta; */
/* } */

/* /\***********************************************************\//\** */
/*     Serialize a function train array */

/*     \param[in,out] ser        - stream to serialize to */
/*     \param[in]     ft         - function train array */
/*     \param[in,out] totSizeIn  - if NULL then serialize, if not NULL then return size */

/*     \return ptr - ser shifted by number of bytes */
/* ***************************************************************\/ */
/* unsigned char *  */
/* ft1d_array_serialize(unsigned char * ser, struct FT1DArray * ft, */
/*                      size_t *totSizeIn) */
/* { */

/*     // size -> ft1 -> ft2 -> ... ->ft[size] */

/*     // size */
/*     size_t totSize = sizeof(size_t); */
/*     size_t size_temp; */
/*     for (size_t ii = 0; ii < ft->size; ii++){ */
/*         function_train_serialize(NULL,ft->ft[ii],&size_temp); */
/*         totSize += size_temp; */
/*     } */
/*     if (totSizeIn != NULL){ */
/*         *totSizeIn = totSize; */
/*         return ser; */
/*     } */

/*     // serialize the size */
/*     unsigned char * ptr = ser; */
/*     ptr = serialize_size_t(ptr, ft->size); */
    
/*     // serialize each function train */
/*     for (size_t ii = 0; ii < ft->size; ii++){ */
/*         ptr = function_train_serialize(ptr,ft->ft[ii],NULL); */
/*     } */
    
/*     return ptr; */
/* } */

/* /\***********************************************************\//\** */
/*     Deserialize a function train array */

/*     \param[in,out] ser - serialized function train array */
/*     \param[in,out] ft  - function_train array */

/*     \return ptr - shifted ser after deserialization */
/* ***************************************************************\/ */
/* unsigned char * */
/* ft1d_array_deserialize(unsigned char * ser, struct FT1DArray ** ft) */
/* { */
/*     unsigned char * ptr = ser; */

/*     // deserialize the number of fts in the array */
/*     size_t size; */
/*     ptr = deserialize_size_t(ptr, &size); */
/*     *ft = ft1d_array_alloc(size); */

/*     // deserialize each function train */
/*     for (size_t ii = 0; ii < size; ii++){ */
/*         ptr = function_train_deserialize(ptr, &((*ft)->ft[ii])); */
/*     } */
/*     return ptr; */
/* } */

/* /\***********************************************************\//\** */
/*     Save a function train array to file */
    
/*     \param[in] ft       - function train array to save */
/*     \param[in] filename - name of file to save to */

/*     \return success (1) or failure (0) of opening the file */
/* ***************************************************************\/ */
/* int ft1d_array_save(struct FT1DArray * ft, char * filename) */
/* { */

/*     FILE *fp; */
/*     fp = fopen(filename, "w"); */
/*     if (fp == NULL){ */
/*         fprintf(stderr, "cat: can't open %s\n", filename); */
/*         return 0; */
/*     } */
    
/*     size_t totsize; */
/*     ft1d_array_serialize(NULL,ft, &totsize); */

/*     unsigned char * data = malloc(totsize+sizeof(size_t)); */
/*     if (data == NULL){ */
/*         fprintf(stderr, "can't allocate space for saving density\n"); */
/*         return 0; */
/*     } */
    
/*     // serialize size first! */
/*     unsigned char * ptr = serialize_size_t(data,totsize); */
/*     ptr = ft1d_array_serialize(ptr,ft,NULL); */

/*     fwrite(data,sizeof(unsigned char),totsize+sizeof(size_t),fp); */

/*     free(data); data = NULL; */
/*     fclose(fp); */
/*     return 1; */
/* } */

/* /\***********************************************************\//\** */
/*     Load a function train array from a file */
    
/*     \param[in] filename - filename of file to load */

/*     \return ft if successfull NULL otherwise */
/* ***************************************************************\/ */
/* struct FT1DArray * ft1d_array_load(char * filename) */
/* { */
/*     FILE *fp; */
/*     fp =  fopen(filename, "r"); */
/*     if (fp == NULL){ */
/*         fprintf(stderr, "cat: can't open %s\n", filename); */
/*         return NULL; */
/*     } */

/*     size_t totsize; */
/*     size_t k = fread(&totsize,sizeof(size_t),1,fp); */
/*     if ( k != 1){ */
/*         printf("error reading file %s\n",filename); */
/*         return NULL; */
/*     } */

/*     unsigned char * data = malloc(totsize); */
/*     if (data == NULL){ */
/*         fprintf(stderr, "can't allocate space for loading density\n"); */
/*         return NULL; */
/*     } */

/*     k = fread(data,sizeof(unsigned char),totsize,fp); */

/*     struct FT1DArray * ft = NULL; */
/*     ft1d_array_deserialize(data,&ft); */
    
/*     free(data); data = NULL; */
/*     fclose(fp); */
/*     return ft; */
/* } */

/* /\***********************************************************\//\** */
/*     Copy an array of function trains */

/*     \param[in] fta - array to coppy */

/*     \return ftb - copied array */
/* ***************************************************************\/ */
/* struct FT1DArray * ft1d_array_copy(struct FT1DArray * fta) */
/* { */
/*     struct FT1DArray * ftb = ft1d_array_alloc(fta->size); */
/*     size_t ii; */
/*     for (ii = 0; ii < fta->size; ii++){ */
/*         ftb->ft[ii] = function_train_copy(fta->ft[ii]); */
/*     } */
/*     return ftb; */
/* } */

/* /\***********************************************************\//\** */
/*     Free a 1d array of function trains */

/*     \param[in,out] fta - function train array to free */
/* ***************************************************************\/ */
/* void ft1d_array_free(struct FT1DArray * fta) */
/* { */
/*     if (fta != NULL){ */
/*         size_t ii = 0; */
/*         for (ii = 0; ii < fta->size; ii++){ */
/*             function_train_free(fta->ft[ii]); */
/*             fta->ft[ii] = NULL; */
/*         } */
/*         free(fta->ft); */
/*         fta->ft = NULL; */
/*         free(fta); */
/*         fta = NULL; */
/*     } */
/* } */

/* /\********************************************************\//\** */
/*     Compute the Jacobian of a Function Train 1darray */

/*     \param[in] fta - Function train array */

/*     \return jacobian */
/* ***********************************************************\/ */
/* struct FT1DArray * ft1d_array_jacobian(struct FT1DArray * fta) */
/* { */
/*     struct FT1DArray * jac = ft1d_array_alloc(fta->size * fta->ft[0]->dim); */
/*     size_t ii,jj; */
/*     for (ii = 0; ii < fta->ft[0]->dim; ii++){ */
/*         for (jj = 0; jj < fta->size; jj++){ */
/*             jac->ft[ii*fta->size+jj] = function_train_copy(fta->ft[jj]); */
/*             qmarray_free(jac->ft[ii*fta->size+jj]->cores[ii]); */
/*             jac->ft[ii*fta->size+jj]->cores[ii] = NULL; */
/*             jac->ft[ii*fta->size+jj]->cores[ii] =  */
/*                 qmarray_deriv(fta->ft[jj]->cores[ii]); */
            
/*         } */
/*     } */
/*     return jac; */
/* } */

/* /\********************************************************\//\** */
/*     Compute the hessian of a function train  */

/*     \param[in] fta - Function train  */

/*     \return hessian of a function train */
/* ***********************************************************\/ */
/* struct FT1DArray * function_train_hessian(struct FunctionTrain * fta) */
/* { */
/*     struct FT1DArray * ftg = function_train_gradient(fta); */

/*     struct FT1DArray * fth = ft1d_array_jacobian(ftg); */
        
/*     ft1d_array_free(ftg); ftg = NULL; */
/*     return fth; */
/* } */

/* /\********************************************************\//\** */
/*     Scale a function train array */

/*     \param[in,out] fta   - function train array */
/*     \param[in]     n     - number of elements in the array to scale */
/*     \param[in]     inc   - increment between elements of array */
/*     \param[in]     scale - value by which to scale */
/* ***********************************************************\/ */
/* void ft1d_array_scale(struct FT1DArray * fta, size_t n, size_t inc, double scale) */
/* { */
/*     size_t ii; */
/*     for (ii = 0; ii < n; ii++){ */
/*         function_train_scale(fta->ft[ii*inc],scale); */
/*     } */
/* } */

/* /\********************************************************\//\** */
/*     Evaluate a function train 1darray */

/*     \param[in] fta - Function train array to evaluate */
/*     \param[in] x   - location at which to obtain evaluations */

/*     \return evaluation */
/* ***********************************************************\/ */
/* double * ft1d_array_eval(struct FT1DArray * fta, double * x) */
/* { */
/*     double * out = calloc_double(fta->size); */
/*     size_t ii;  */
/*     for (ii = 0; ii < fta->size; ii++){ */
/*         out[ii] = function_train_eval(fta->ft[ii], x); */
/*     } */
/*     return out; */
/* } */

/* /\********************************************************\//\** */
/*     Evaluate a function train 1darray  */

/*     \param[in]     fta - Function train array to evaluate */
/*     \param[in]     x   - location at which to obtain evaluations */
/*     \param[in,out] out - evaluation */

/* ***********************************************************\/ */
/* void ft1d_array_eval2(struct FT1DArray * fta, double * x, double * out) */
/* { */
/*     size_t ii;  */
/*     for (ii = 0; ii < fta->size; ii++){ */
/*         out[ii] = function_train_eval(fta->ft[ii], x); */
/*     } */
/* } */

/* /\********************************************************\//\** */
/*     Interpolate a function-train onto a particular grid forming  */
/*     another function_train with a nodela basis */

/*     \param[in] fta - Function train array to evaluate */
/*     \param[in] N   - number of nodes in each dimension */
/*     \param[in] x   - nodes in each dimension */

/*     \return new function train */
/* ***********************************************************\/ */
/* struct FunctionTrain * */
/* function_train_create_nodal(struct FunctionTrain * fta, size_t * N, double ** x) */
/* { */
/*     struct FunctionTrain * newft = function_train_alloc(fta->dim); */
/*     memmove(newft->ranks,fta->ranks, (fta->dim+1)*sizeof(size_t)); */
/*     for (size_t ii = 0; ii < newft->dim; ii ++){ */
/*         newft->cores[ii] = qmarray_create_nodal(fta->cores[ii],N[ii],x[ii]); */
/*     } */
/*     return newft; */
/* } */

/* /\********************************************************\//\** */
/*     Multiply together and sum the elements of two function train arrays */
/*     \f[  */
/*         out(x) = \sum_{i=1}^{N} coeff[i] f_i(x)  g_i(x)  */
/*     \f] */
    
/*     \param[in] N       - number of function trains in each array */
/*     \param[in] coeff   - coefficients to multiply each element */
/*     \param[in] f       - first array */
/*     \param[in] g       - second array */
/*     \param[in] epsilon - rounding accuracy */

/*     \return function train */
/* ***********************************************************\/ */
/* struct FunctionTrain *  */
/* ft1d_array_sum_prod(size_t N, double * coeff,  */
/*                struct FT1DArray * f, struct FT1DArray * g,  */
/*                double epsilon) */
/* { */
    
/*     struct FunctionTrain * ft1 = NULL; */
/*     struct FunctionTrain * ft2 = NULL; */
/*     struct FunctionTrain * out = NULL; */
/*     struct FunctionTrain * temp = NULL; */

/* //    printf("compute product\n"); */
/*     temp = function_train_product(f->ft[0],g->ft[0]); */
/*     function_train_scale(temp,coeff[0]); */
/* //    printf("scale N = %zu\n",N); */
/*     out = function_train_round(temp,epsilon); */
/* //    printf("rounded\n"); */
/*     function_train_free(temp); temp = NULL; */
/*     size_t ii; */
/*     for (ii = 1; ii < N; ii++){ */
/*         temp =  function_train_product(f->ft[ii],g->ft[ii]); */
/*         function_train_scale(temp,coeff[ii]); */

/*         ft2 = function_train_round(temp,epsilon); */
/*         ft1 = function_train_sum(out, ft2); */
    
/*         function_train_free(temp); temp = NULL; */
/*         function_train_free(ft2); ft2 = NULL; */
/*         function_train_free(out); out = NULL; */

/*         out = function_train_round(ft1,epsilon); */

/*         function_train_free(ft1); ft1 = NULL; */
/*     } */

/*     return out; */
/* } */


/* /\***********************************************************\//\** */
/*     Computes  */
/*     \f[ */
/*         y \leftarrow \texttt{round}(a x + y, epsilon) */
/*     \f] */

/*     \param a [in] - scaling factor */
/*     \param x [in] - first function train */
/*     \param y [inout] - second function train */
/*     \param epsilon - rounding accuracy (0 for exact) */
/* ***************************************************************\/ */
/* void c3axpy(double a, struct FunctionTrain * x, struct FunctionTrain ** y,  */
/*             double epsilon) */
/* { */
    
/*     struct FunctionTrain * temp = function_train_copy(x); */
/*     function_train_scale(temp,a); */
/*     struct FunctionTrain * z = function_train_sum(temp,*y); */
/*     function_train_free(*y); *y = NULL; */
/*     if (epsilon > 0){ */
/*         *y = function_train_round(z,epsilon); */
/*         function_train_free(z); z = NULL; */
/*     } */
/*     else{ */
/*         *y = z; */
/*     } */
/*     function_train_free(temp); temp = NULL; */
/* } */

/* /\***********************************************************\//\** */
/*     Computes  */
/*     \f$ */
/*         \langle x,y \rangle */
/*     \f$ */

/*     \param x [in] - first function train */
/*     \param y [in] - second function train */

/*     \return out - inner product between two function trains */
/* ***************************************************************\/ */
/* double c3dot(struct FunctionTrain * x, struct FunctionTrain * y) */
/* { */
/*     double out = function_train_inner(x,y); */
/*     return out; */
/* } */

/* ////////////////////////////////////////////////////////////////////// */
/* // Blas type interface 2 */
/* /\***********************************************************\//\** */
/*     Computes  */
/*     \f[ */
/*         y \leftarrow alpha \sum_{i=1}^n \texttt{round}(\texttt{product}(A[i*inca],x),epsilon) + */
/*          beta y  */
/*     \f] */
/*     \f[ */
/*         y \leftarrow \texttt{round}(y,epsilon) */
/*     \f] */
    
/*     \note */
/*     Also rounds after every summation */
/* ***************************************************************\/ */
/* void c3gemv(double alpha, size_t n, struct FT1DArray * A, size_t inca, */
/*         struct FunctionTrain * x,double beta, struct FunctionTrain * y, */
/*         double epsilon) */
/* { */

/*     size_t ii; */
/*     assert (y != NULL); */
/*     function_train_scale(y,beta); */
    

/*     if (epsilon > 0){ */
/*         struct FunctionTrain * runinit = function_train_product(A->ft[0],x); */
/*         struct FunctionTrain * run = function_train_round(runinit,epsilon); */
/*         for (ii = 1; ii < n; ii++){ */
/*             struct FunctionTrain * temp =  */
/*                 function_train_product(A->ft[ii*inca],x); */
/*             struct FunctionTrain * tempround = function_train_round(temp,epsilon); */
/*             c3axpy(1.0,tempround,&run,epsilon); */
/*             function_train_free(temp); temp = NULL; */
/*             function_train_free(tempround); tempround = NULL; */
/*         } */
/*         c3axpy(alpha,run,&y,epsilon); */
/*         function_train_free(run); run = NULL; */
/*         function_train_free(runinit); runinit = NULL; */
/*     } */
/*     else{ */
/*         struct FunctionTrain * run = function_train_product(A->ft[0],x); */
/*         for (ii = 1; ii < n; ii++){ */
/*             struct FunctionTrain * temp =  */
/*                 function_train_product(A->ft[ii*inca],x); */
/*             c3axpy(1.0,temp,&run,epsilon); */
/*             function_train_free(temp); temp = NULL; */
/*         } */
/*         c3axpy(alpha,run,&y,epsilon); */
/*         function_train_free(run); run = NULL; */
/*     } */
       
/* } */

/* ////////////////////////////////////////////////////////////////////// */
/* // Blas type interface 1 (for ft arrays */

/* /\***********************************************************\//\** */
/*     Computes for \f$ i=1 \ldots n\f$ */
/*     \f[ */
/*         y[i*incy] \leftarrow \texttt{round}(a * x[i*incx] + y[i*incy],epsilon) */
/*     \f] */

/* ***************************************************************\/ */
/* void c3vaxpy(size_t n, double a, struct FT1DArray * x, size_t incx,  */
/*             struct FT1DArray ** y, size_t incy, double epsilon) */
/* { */
/*     size_t ii; */
/*     for (ii = 0; ii < n; ii++){ */
/*         c3axpy(a,x->ft[ii*incx],&((*y)->ft[ii*incy]),epsilon); */
/*     } */
/* } */

/* /\***********************************************************\//\** */
/*     Computes for \f$ i=1 \ldots n\f$ */
/*     \f[ */
/*         z \leftarrow alpha\sum_{i=1}^n y[incy*ii]*x[ii*incx] + beta * z */
/*     \f] */

/* ***************************************************************\/ */
/* void c3vaxpy_arr(size_t n, double alpha, struct FT1DArray * x,  */
/*                 size_t incx, double * y, size_t incy, double beta, */
/*                 struct FunctionTrain ** z, double epsilon) */
/* { */
/*     assert (*z != NULL); */
/*     function_train_scale(*z,beta); */

/*     size_t ii; */
/*     for (ii = 0; ii < n; ii++){ */
/*         c3axpy(alpha*y[incy*ii],x->ft[ii*incx], z,epsilon); */
/*     } */
/* } */

/* /\***********************************************************\//\** */
/*     Computes  */
/*     \f[ */
/*         z \leftarrow \texttt{round}(a\sum_{i=1}^n \texttt{round}(x[i*incx]*y[i*incy],epsilon) + beta * z,epsilon) */
        
/*     \f] */
    
/*     \note */
/*     Also rounds after every summation. */

/* ***************************************************************\/ */
/* void c3vprodsum(size_t n, double a, struct FT1DArray * x, size_t incx, */
/*                 struct FT1DArray * y, size_t incy, double beta, */
/*                 struct FunctionTrain ** z, double epsilon) */
/* { */
/*     assert (*z != NULL); */
/*     function_train_scale(*z,beta); */
        
/*     size_t ii; */

/*     if (epsilon > 0){ */
/*         struct FunctionTrain * runinit =  */
/*             function_train_product(x->ft[0],y->ft[0]); */
/*         struct FunctionTrain * run = function_train_round(runinit,epsilon); */
/*         for (ii = 1; ii < n; ii++){ */
/*             struct FunctionTrain * tempinit =  */
/*                 function_train_product(x->ft[ii*incx],y->ft[ii*incy]); */
/*             struct FunctionTrain * temp = function_train_round(tempinit,epsilon); */
/*             c3axpy(1.0,temp,&run,epsilon); */
/*             function_train_free(temp); temp = NULL; */
/*             function_train_free(tempinit); tempinit = NULL; */
/*         } */
/*         c3axpy(a,run,z,epsilon); */
/*         function_train_free(run); run = NULL; */
/*         function_train_free(runinit); runinit = NULL; */
/*     } */
/*     else{ */
/*         struct FunctionTrain * run =  */
/*             function_train_product(x->ft[0],y->ft[0]); */
/*         for (ii = 1; ii < n; ii++){ */
/*             struct FunctionTrain * temp =  */
/*                 function_train_product(x->ft[ii*incx],y->ft[ii*incy]); */
/*             c3axpy(1.0,temp,&run,epsilon); */
/*             function_train_free(temp); temp = NULL; */
/*         } */
/*         c3axpy(a,run,z,epsilon); */
/*         function_train_free(run); run = NULL; */
/*     } */
/* } */

/* /\***********************************************************\//\** */
/*     Computes for \f$ i = 1 \ldots m \f$ */
/*     \f[ */
/*         y[i*incy] \leftarrow alpha*\sum_{j=1}^n \texttt{product}(A[i,j*lda],x[j*incx]) + beta * y[i*incy] */
/*     \f] */
    
/*     \note */
/*     Rounds with tolerance epsilon after summation and multiplication. Not shown to avoid clutter. */

/* ***************************************************************\/ */
/* void c3vgemv(size_t m, size_t n, double alpha, struct FT1DArray * A, size_t lda, */
/*         struct FT1DArray * x, size_t incx, double beta, struct FT1DArray ** y, */
/*         size_t incy, double epsilon) */
/* { */

/*     size_t ii; */
/*     assert (*y != NULL); */
/*     ft1d_array_scale(*y,m,incy,beta); */
    
/*     struct FT1DArray * run = ft1d_array_alloc(m);  */
/*     for (ii = 0; ii < m; ii++){ */
/*         struct FT1DArray ftatemp; */
/*         ftatemp.ft = A->ft+ii; */
/*         c3vprodsum(n,1.0,&ftatemp,lda,x,incx,0.0,&(run->ft[ii*incy]),epsilon); */
/*     } */
/*     c3vaxpy(m,alpha,run,1,y,incy,epsilon); */

/*     ft1d_array_free(run); run = NULL; */
/* } */

/* /\***********************************************************\//\** */
/*     Computes for \f$ i = 1 \ldots m \f$ */
/*     \f[ */
/*         y[i*incy] \leftarrow alpha*\sum_{j=1}^n \texttt{product}(A[i,j],B[j*incb]) + beta * y[i*incy] */
/*     \f] */
    
/*     \note */
/*     Rounds with tolerance epsilon after summation and multiplication. Not shown to avoid clutter. */
/*     trans = 0 means not to transpose A */
/*     trans = 1 means to transpose A */
/* ***************************************************************\/ */
/* void c3vgemv_arr(int trans, size_t m, size_t n, double alpha, double * A,  */
/*         size_t lda, struct FT1DArray * B, size_t incb, double beta, */
/*         struct FT1DArray ** y, size_t incy, double epsilon) */
/* { */
/*     size_t ii; */
/*     assert (*y != NULL); */
/*     ft1d_array_scale(*y,m,incy,beta); */
        
/*     if (trans == 0){ */
/*         for (ii = 0; ii < m; ii++){ */
/*             c3vaxpy_arr(n,alpha,B,incb,A+ii,lda,beta,&((*y)->ft[ii*incy]),epsilon); */
/*         } */
/*     } */
/*     else if (trans == 1){ */
/*         for (ii = 0; ii < m; ii++){ */
/*             c3vaxpy_arr(n,alpha,B,incb,A+ii*lda,1,beta,&((*y)->ft[ii*incy]),epsilon); */
/*         } */
/*     } */
/* } */
