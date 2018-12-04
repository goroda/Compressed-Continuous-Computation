// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016-2017 Sandia Corporation
// Copyright (c) 2017 NTESS, LLC.

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

#ifndef ZEROTHRESH
/// @private
    #define ZEROTHRESH 1e0* DBL_EPSILON
    /* #define ZEROTHRESH 1e0*DBL_EPSILON */
    /* #define ZEROTHRESH 1e-200 */
#endif

/* #define ZEROTHRESH 1e-3*DBL_EPSILON */

#ifndef VPREPCORE
/// @private
    #define VPREPCORE 0
#endif

#ifndef VFTCROSS
/// @private
    #define VFTCROSS 0
#endif


/// @private
struct FTMemSpace
{
    size_t num_spaces;
    size_t space_size; // increment 1
    double * vals; // num_spaces x space_size
};

/// @private
struct FTMemSpace * ft_mem_space_alloc(size_t num_spaces, size_t space_size)
{
    struct FTMemSpace * ftmem = malloc(sizeof(struct FTMemSpace));
    if (ftmem == NULL){
        fprintf(stderr, "Failure to allocate memory in FT\n");
        return NULL;
    }

    ftmem->num_spaces = num_spaces;
    ftmem->space_size = space_size;
    ftmem->vals = calloc_double(num_spaces * space_size);

    return ftmem;
}

/// @private
struct FTMemSpace ** ft_mem_space_arr_alloc(size_t dim, size_t num_spaces, size_t space_size)
{
    struct FTMemSpace ** ftmem = malloc(dim * sizeof(struct FTMemSpace * ));
    if (ftmem == NULL){
        fprintf(stderr, "Failure to allocate memory in FT\n");
        return NULL;
    }
    for (size_t ii = 0; ii < dim; ii++){
        ftmem[ii] = ft_mem_space_alloc(num_spaces,space_size);
    }

    return ftmem;
}

/// @private
void ft_mem_space_free(struct FTMemSpace * ftmem)
{
    if (ftmem != NULL){
        free(ftmem->vals); ftmem->vals = NULL;
        free(ftmem); ftmem = NULL;
    }
}

/// @private
void ft_mem_space_arr_free(size_t dim, struct FTMemSpace ** ftmem)
{
    if (ftmem != NULL){
        for (size_t ii = 0; ii < dim; ii++){
            ft_mem_space_free(ftmem[ii]);
            ftmem[ii] = NULL;
        }
        free(ftmem); ftmem = NULL;
    }
}

/// @private
int ft_mem_space_check(struct FTMemSpace * ftmem,
                       size_t num_spaces,
                       size_t space_size)
{
    if (ftmem == NULL){
        return 1;
    }
    if ( (ftmem->num_spaces < num_spaces) ||
         (ftmem->space_size != space_size))
    {
        ftmem->num_spaces = num_spaces;
        ftmem->space_size = space_size;
        free(ftmem->vals); ftmem->vals = NULL;
        ftmem->vals = calloc_double(num_spaces * space_size);
    }
    
    return 0;
    
}

/// @private
size_t ft_mem_space_get_incr(struct FTMemSpace * ftmem)
{
    return ftmem->space_size;
}

/***********************************************************//**
    Allocate space for a function_train

    \param[in] dim - dimension of function train

    \return ft - function train
***************************************************************/
struct FunctionTrain * function_train_alloc(size_t dim)
{
    /* printf("Allocating Function Train\n"); */
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
    ft->evalspace4 = NULL;
    ft->evaldd1 = NULL;
    ft->evaldd2 = NULL;
    ft->evaldd3 = NULL;
    ft->evaldd4 = NULL;

    /* printf("Done Allocating Function Train\n"); */
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
    /* printf("\t Freeing Function Train\n"); */
    if (ft != NULL){
        /* printf("\t\t FT is not null\n"); */
        free(ft->ranks);
        ft->ranks = NULL;
        size_t ii;
        for (ii = 0; ii < ft->dim; ii++){
            qmarray_free(ft->cores[ii]);ft->cores[ii] = NULL;
        }
        free(ft->cores); ft->cores=NULL;

        ft_mem_space_free(ft->evalspace1); ft->evalspace1 = NULL;
        ft_mem_space_free(ft->evalspace2); ft->evalspace2 = NULL;
        ft_mem_space_free(ft->evalspace3); ft->evalspace3 = NULL;

        ft_mem_space_arr_free(ft->dim, ft->evalspace4); ft->evalspace4 = NULL;

        free_dd(ft->dim,ft->evaldd1); ft->evaldd1 = NULL;
        /* free_dd(2*ft->dim,ft->evaldd2);   ft->evaldd2 = NULL; */
        free_dd(ft->dim,ft->evaldd3);     ft->evaldd3 = NULL;
        free_dd(ft->dim,ft->evaldd4);     ft->evaldd4 = NULL;
        free(ft); ft = NULL;
    }
    /* printf("\t\tDone Freeing Function Train\n"); */
}

struct FunctionTrain *
function_train_operate(const struct FunctionTrain * ft,
                       struct Operator ** op)
{

    struct FunctionTrain * ft_out = function_train_alloc(ft->dim);
    ft_out->ranks[1] = ft->ranks[1];
    for (size_t ii = 0; ii < ft_out->dim; ii++){
        ft_out->cores[ii] = qmarray_operate_elements(ft->cores[ii],
                                                     op[ii]);
        ft_out->ranks[ii+1] = ft->ranks[ii+1];
    }

    return ft_out;
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
        /* printf("Serializing core (%zu/%zu)\n",ii+1,ft->dim); */
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
    /* printf("deserialized dim=%zu\n",dim); */
    *ft = function_train_alloc(dim);
    
    size_t ii;
    for (ii = 0; ii < dim+1; ii++){
        ptr = deserialize_size_t(ptr, &((*ft)->ranks[ii]));
    }
    /* printf("deserialized ranks = \n"); iprint_sz(dim+1,(*ft)->ranks); */
    for (ii = 0; ii < dim; ii++){
        ptr = qmarray_deserialize(ptr, &((*ft)->cores[ii]));
    }
    
    return ptr;
}

/***********************************************************//**
    Save a function train to text file for portability

    \param[in]     ft     - function train
    \param[in,out] stream - stream to write to
    \param[in]     prec   - precision with which to save

***************************************************************/
void function_train_savetxt(struct FunctionTrain * ft, FILE * stream,
                            size_t prec)
{

    // dim -> nranks -> core1 -> core2 -> ... -> cored

    size_t ii;
    //dim + ranks
    
    fprintf(stream,"%zu ",ft->dim);
    for (ii = 0; ii < ft->dim+1; ii++){
        fprintf(stream,"%zu ",ft->ranks[ii]);
    }

    for (ii = 0; ii < ft->dim; ii++){
        qmarray_savetxt(ft->cores[ii],stream,prec);
    }
}

/***********************************************************//**
    Load a function train stored as a text file

    \param[in,out] fp - stream from which to load

    \return function train
***************************************************************/
struct FunctionTrain * function_train_loadtxt(FILE * fp)
{
    size_t dim;
    int num = fscanf(fp,"%zu ",&dim);
    assert (num == 1);

    struct FunctionTrain * ft = function_train_alloc(dim);
    
    size_t ii;
    for (ii = 0; ii < dim+1; ii++){
        num = fscanf(fp,"%zu ",ft->ranks + ii);
        assert (num == 1);
    }
    for (ii = 0; ii < dim; ii++){
        ft->cores[ii] = qmarray_loadtxt(fp);
    }
    
    return ft;
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
    Get dim
***************************************************************/
size_t function_train_get_dim(const struct FunctionTrain * ft)
{
    assert(ft != NULL);
    return ft->dim;
}

/***********************************************************//**
    Get core
***************************************************************/
struct Qmarray* function_train_get_core(const struct FunctionTrain * ft, size_t ind)
{
    assert(ft != NULL);
    assert (ind < ft->dim);
    return ft->cores[ind];
}

/***********************************************************//**
    Get rank
***************************************************************/
size_t function_train_get_rank(const struct FunctionTrain * ft, size_t index)
{
    assert (ft != NULL);
    assert (ft->ranks != NULL);
    assert (index < ft->dim+1);
    return ft->ranks[index];
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
    Evaluate a new core and multiply against the previous evaluations (Left-Right)

    \param[in]     ft             - function train
    \param[in]     core           - which core to evaluate
    \param[in]     N              - number of evaluations
    \param[in]     x              - evaluation location
    \param[in]     incx           - increment of x
    \param[in]     prev_eval      - result of evaluating previous cores
    \param[in]     inc_pr         - increment for previous evaluation
    \param[in,out] new_eval_space - space for the evaluation of a core
    \param[in]     inc_new_space  - increment for new evaluation
    \param[in,out] new_eval       - output location for prev_eval * new_eval_space
    \param[in]     inc_new        - increment for final multiplication

    \note Increments are between storage locations for each evaluation
***************************************************************/
void function_train_eval_next_core(struct FunctionTrain * ft, size_t core, size_t N,
                                   const double * x,        size_t incx,
                                   double * prev_eval,      size_t inc_pr,
                                   double * new_eval_space, size_t inc_new_space,
                                   double * new_eval,       size_t inc_new)
{
    assert (core > 0);

    size_t r1 = ft->cores[core]->nrows;
    size_t r2 = ft->cores[core]->ncols;
    generic_function_1darray_eval2N(
        r1*r2, ft->cores[core]->funcs, N, x, incx, new_eval_space, inc_new_space);

    for (size_t ii = 0; ii < N; ii++){
        cblas_dgemv(CblasColMajor,CblasTrans,
                    r1,r2, 1.0,
                    new_eval_space + ii * inc_new_space, r1,
                    prev_eval + ii*inc_pr, 1, 0.0, new_eval + ii*inc_new, 1);
    }
    
}

/***********************************************************//**
    Evaluate a new core and multiply against the previous evaluations (Right-Left)

    \param[in]     ft             - function train
    \param[in]     core           - which core to evaluate
    \param[in]     N              - number of evaluations
    \param[in]     x              - evaluation location
    \param[in]     incx           - increment of x
    \param[in]     prev_eval      - result of evaluating previous cores
    \param[in]     inc_pr         - increment for previous evaluation
    \param[in,out] new_eval_space - space for the evaluation of a core
    \param[in]     inc_new_space  - increment for new evaluation
    \param[in,out] new_eval       - output location for prev_eval * new_eval_space
    \param[in]     inc_new        - increment for final multiplication

    \note Increments are between storage locations for each evaluation
***************************************************************/
void function_train_eval_prev_core(struct FunctionTrain * ft, size_t core, size_t N,
                                   const double * x,        size_t incx,
                                   double * prev_eval,      size_t inc_pr,
                                   double * new_eval_space, size_t inc_new_space,
                                   double * new_eval,       size_t inc_new)
{
    assert (core > 0);

    size_t r1 = ft->cores[core]->nrows;
    size_t r2 = ft->cores[core]->ncols;
    generic_function_1darray_eval2N(
        r1*r2, ft->cores[core]->funcs, N, x, incx, new_eval_space, inc_new_space);

    for (size_t ii = 0; ii < N; ii++){
        cblas_dgemv(CblasColMajor,CblasNoTrans,
                    r1, r2, 1.0,
                    new_eval_space + ii * inc_new_space, r1,
                    prev_eval + ii * inc_pr, 1, 0.0, new_eval + ii*inc_new, 1);
    }
    
}

/***********************************************************//**
    Some overhead for function train evaluation functions
***************************************************************/
void function_train_pre_eval(struct FunctionTrain * ft, size_t * maxrank,
                             double ** t1,double ** t2,double ** t3, size_t N)
{

    *maxrank = function_train_get_maxrank(ft);
    size_t mr2 = *maxrank * *maxrank;
    int ok = ft_mem_space_check(ft->evalspace1, N, mr2);
    if (ok == 1){
        ft->evalspace1 = ft_mem_space_alloc(N, mr2);
    }
    ok = ft_mem_space_check(ft->evalspace2, N, mr2);
    if (ok == 1){
        ft->evalspace2 = ft_mem_space_alloc(N, mr2);
    }
    ok = ft_mem_space_check(ft->evalspace3, N, mr2);
    if (ok == 1){
        ft->evalspace3 = ft_mem_space_alloc(N, mr2);
    }
    
    *t1 = ft->evalspace1->vals;
    *t2 = ft->evalspace2->vals;
    *t3 = ft->evalspace3->vals;
}

/***********************************************************//**
    Evaluate a function train up to, and including core k

    \param[in]     ft      - function train
    \param[in]     core    - last core to evaluate
    \param[in]     x       - location at which to evaluate
    \param[in,out] out     - evaluation, with enough space
    \param[in,out] out_dim - size of out

    \note NOT TESTED
***************************************************************/
void function_train_eval_up_to_core(struct FunctionTrain * ft, size_t core, const double * x, double * out, size_t *out_dim)
{
    size_t dim = ft->dim;
    assert(ft->ranks[0] == 1);
    assert(ft->ranks[dim] == 1);

    size_t maxrank,ii = 0;
    double *t1=NULL, *t2=NULL, *t3=NULL;
    function_train_pre_eval(ft,&maxrank,&t1,&t2,&t3,1);

    generic_function_1darray_eval2(
        ft->cores[ii]->nrows * ft->cores[ii]->ncols,
        ft->cores[ii]->funcs, x[ii],t1);
    
    int onsol = 1;
    for (ii = 1; ii < core; ii++){
        if (ii%2 == 1){
            function_train_eval_next_core(ft,ii,1,x+ii,0,t1,0,t2,0,t3,0); // zeros because only 1 data point
            onsol = 2;
        }
        else{
            function_train_eval_next_core(ft,ii,1,x+ii,0,t3,0,t2,0,t1,0);
            onsol=1;
        }
    }

    *out_dim = ft->cores[core]->ncols;
    if (onsol == 2){
        memmove(out,t3,*out_dim*sizeof(double));
    }
    else if ( onsol == 1){
        memmove(out,t1,*out_dim*sizeof(double));
    }
    else{
        fprintf(stderr,"Weird error in function_train_val\n");
        exit(1);
    }
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

    size_t maxrank;
    double *t1=NULL, *t2=NULL, *t3=NULL;
    function_train_pre_eval(ft,&maxrank,&t1,&t2,&t3,1);

    size_t ii = 0;
    generic_function_1darray_eval2(
        ft->cores[ii]->nrows * ft->cores[ii]->ncols,
        ft->cores[ii]->funcs, x[ii],t1);
    int onsol = 1;
    /* printf("Start running eval: "); */
    /* dprint(ft->ranks[ii+1],t1); */
    for (ii = 1; ii < dim; ii++){
        if (ii%2 == 1){
            function_train_eval_next_core(ft,ii,1,x+ii,0,t1,0,t2,0,t3,0); // zero because 1 data pint
            onsol = 2;
            /* printf("new eval = \n"); */
            /* dprint2d_col(ft->ranks[ii],ft->ranks[ii+1],t2); */
            /* printf("\t running eval: "); */
            /* dprint(ft->ranks[ii+1],t3); */
        }
        else{
            function_train_eval_next_core(ft,ii,1,x+ii,0,t3,0,t2,0,t1,0);
            onsol=1;
            /* printf("new eval = \n"); */
            /* dprint2d_col(ft->ranks[ii],ft->ranks[ii+1],t2); */
            /* printf("\t running eval: "); */
            /* dprint(ft->ranks[ii+1],t1); */
        }

        /* if (ii == dim-1){ */
        /*     if (onsol == 2){ */
        /*         printf("pre_eval = "); dprint(2,t1); */
        /*     } */
        /*     else{ */
        /*         printf("pre_eval = "); dprint(2,t3); */
        /*     } */
        /*     printf("last_core_eval = "); dprint(2,t2); */
        /* } */
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
   Get number of parameters of a particular univariate function

   \param[in] ft   - functoin train
   \param[in] core - which core parameters to count
   \param[in] ii   - row
   \param[in] jj   - column

   \returns number of parameters in the core
***************************************************************/
size_t function_train_func_get_nparams(const struct FunctionTrain * ft,
                                       size_t core, size_t ii, size_t jj)
{
    size_t nparam =
        qmarray_func_get_nparams(ft->cores[core],ii,jj);
    return nparam;
}

/***********************************************************//**
   Get number of parameters describing a core

   \param[in]     ft           - functoin train
   \param[in]     core         - which core parameters to count
   \param[in,out] maxparamfunc - number of parameters in the function
                                 with most parameters within the core

   \returns number of parameters in the core
***************************************************************/
size_t function_train_core_get_nparams(const struct FunctionTrain * ft,
                                       size_t core,
                                       size_t * maxparamfunc)
{
    size_t ncoreparams = qmarray_get_nparams(ft->cores[core],maxparamfunc);
    return ncoreparams;
}

/***********************************************************//**
   Get number of parameters describing a ft

   \param[in] ft - function train

   \returns number of parameters in the FT
***************************************************************/
size_t function_train_get_nparams(const struct FunctionTrain * ft)
{

    size_t ncore;
    size_t maxp;
    size_t tot = 0;
    for (size_t ii = 0; ii < ft->dim; ii++){
        ncore = qmarray_get_nparams(ft->cores[ii],&maxp);
        tot += ncore;
    }

    return tot;
}

/***********************************************************//**
   Get core parameters
***************************************************************/
size_t function_train_core_get_params(const struct FunctionTrain * ft,
                                      size_t core,
                                      double * param)
{
    size_t nparam = qmarray_get_params(ft->cores[core],param);
    return nparam;
}

/***********************************************************//**
   Get function paramaters
***************************************************************/
void * function_train_get_uni(const struct FunctionTrain * ft,
                              size_t core, size_t row, size_t col)
{
    return qmarray_get_func_base(ft->cores[core], row, col);
}

/***********************************************************//**
   Get function paramaters
***************************************************************/
struct GenericFunction *
function_train_get_gfuni(const struct FunctionTrain * ft,
                         size_t core, size_t row, size_t col)
{
    return qmarray_get_func(ft->cores[core], row, col);
}

/***********************************************************//**
   Get parameters
***************************************************************/
size_t function_train_get_params(const struct FunctionTrain * ft,
                                 double * param)
{
    size_t running = 0;
    for (size_t ii = 0; ii < ft->dim; ii++){
        size_t nparam = qmarray_get_params(ft->cores[ii],param+running);
        running += nparam;
    }
    return running;
}


/***********************************************************//**
   Update the parameters of a single univariate function
***************************************************************/
void function_train_uni_update_params(struct FunctionTrain * ft,
                                       size_t core,
                                       size_t row,
                                       size_t col,
                                       size_t nparam,
                                       const double * param)
{
    qmarray_uni_update_params(ft->cores[core], row, col, nparam, param);
}

/***********************************************************//**
   Update the parameters defining a function train core
***************************************************************/
void function_train_core_update_params(struct FunctionTrain * ft,
                                       size_t core,
                                       size_t nparam,
                                       const double * param)
{
    qmarray_update_params(ft->cores[core],nparam,param);
}

/***********************************************************//**
   Update the parameters of a function train
***************************************************************/
size_t function_train_update_params(struct FunctionTrain * ft, const double * param)
{
    size_t running = 0;
    size_t incore = 0;
    for (size_t ii = 0; ii < ft->dim; ii++){
        incore = qmarray_get_nparams(ft->cores[ii],NULL);
        qmarray_update_params(ft->cores[ii],incore,param+running);
        running += incore;
    }
    return running;
}

/***********************************************************//**
    Sum the L2 norms of each function of each core

    \param[in]     ft      - quasimatrix array
    \param[in]     weights - weights for the sum of each column
    \param[in,out] grad    - compute gradient if not null (store gradient here)

    \returns sum of squared L2 norms of each univariate function

    \note 
    If gradient is not null then it adds scaled values of the new gradients to the
    existing gradient. Gradient is stored function first, row second, column third, core fourth
***************************************************************/
double function_train_param_grad_sqnorm(struct FunctionTrain * ft, double * weights, double * grad)
{

    size_t running = 0;
    size_t incr;
    size_t nfmax; // not used
    double out = 0.0;
    for (size_t ii = 0; ii < ft->dim; ii++){
        incr = qmarray_get_nparams(ft->cores[ii], &nfmax);
        if (grad != NULL){
            out += qmarray_param_grad_sqnorm(ft->cores[ii],weights[ii],grad+running);
        }
        else{
            out += qmarray_param_grad_sqnorm(ft->cores[ii],weights[ii],NULL);
        }
        running+=incr;
        /* printf("out = %3.15G\n",out); */
    }

    return out;
}

    
/***********************************************************//**
    Evaluate a function train at particular indices which
    should be valid for underlying univariate functions 
    that are nodal basis functions

    \param[in] ft  - function train
    \param[in] ind - location at which to evaluate

    \return val - value of the function train
***************************************************************/
double function_train_eval_ind(struct FunctionTrain * ft, const size_t * ind)
{
    
    size_t dim = ft->dim;
    assert(ft->ranks[0] == 1);
    assert(ft->ranks[dim] == 1);

    size_t maxrank;
    double *t1=NULL, *t2=NULL, *t3=NULL;
    function_train_pre_eval(ft,&maxrank,&t1,&t2,&t3,1);

    size_t ii = 0;
    generic_function_1darray_eval2_ind(
        ft->cores[ii]->nrows * ft->cores[ii]->ncols,
        ft->cores[ii]->funcs, ind[ii],t1);

    int onsol = 1;
    for (ii = 1; ii < dim; ii++){
        generic_function_1darray_eval2_ind(
            ft->cores[ii]->nrows * ft->cores[ii]->ncols,
            ft->cores[ii]->funcs, ind[ii],t2);
            
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
    Evaluate a fiber of the function-train at particular indices
    when each fiber is represented in a (fixed) nodal basis

    \param[in]     ft        - function train
    \param[in]     fixed_ind - indices of fixed_values
    \param[in]     N         - number of evaluations
    \param[in]     ind       - locations of evaluations
    \param[in]     dim_vary  - dimension that is varying
    \param[in,out] out       - output values

***************************************************************/
void function_train_eval_fiber_ind(struct FunctionTrain * ft, 
                                   const size_t * fixed_ind,
                                   size_t N, const size_t * ind,
                                   size_t dim_vary,
                                   double * out)
{
    
    size_t dim = ft->dim;
    assert(ft->ranks[0] == 1);
    assert(ft->ranks[dim] == 1);

    size_t maxrank;
    double *t1=NULL, *t2=NULL, *t3=NULL;
    function_train_pre_eval(ft,&maxrank,&t1,&t2,&t3,1);

    double * temp_tensor = NULL;
    double * temp_mat = NULL;

    size_t nrows, ncols;

    struct GenericFunction ** arr = ft->cores[dim_vary]->funcs;
    nrows = ft->cores[dim_vary]->nrows;
    ncols = ft->cores[dim_vary]->ncols;
    temp_tensor = calloc_double(nrows * ncols * N);
    temp_mat    = calloc_double(ncols * N);
    for (size_t jj = 0; jj < N; jj++){
        generic_function_1darray_eval2_ind(nrows * ncols,arr,ind[jj],
                                           temp_tensor + jj * nrows * ncols);
    }

    if (dim_vary == 0){
        assert (nrows == 1);
        memmove(temp_mat, temp_tensor, ncols * N * sizeof(double));
    }
    else{ // (dim_vary != 0){
        arr = ft->cores[0]->funcs;
        nrows = ft->cores[0]->nrows;
        ncols = ft->cores[0]->ncols;
        generic_function_1darray_eval2_ind(nrows*ncols,arr,fixed_ind[0],t1);
        
        /* printf("first core is \n \t"); */
        /* dprint(ncols,t1); */
        int onsol = 1;    
        for (size_t ii = 1; ii < dim_vary; ii++){
            nrows = ft->cores[ii]->nrows;
            ncols = ft->cores[ii]->ncols;
            arr = ft->cores[ii]->funcs;

            generic_function_1darray_eval2_ind(nrows * ncols, arr,fixed_ind[ii],t2);
            /* printf("\t\t t2 = \n"); */
            /* dprint2d_col(nrows,ncols,t2); */
            if (onsol == 1){
                // previous times new core
                /* cblas_dgemv(CblasColMajor,CblasTrans, */
                /*             nrows, ncols, 1.0, t2, nrows, */
                /*             t1, 1, 0.0, t3, 1); */
                cblas_dgemv(CblasColMajor,CblasTrans,
                            ft->ranks[ii], ft->ranks[ii+1], 1.0,
                            t2, ft->ranks[ii],
                            t1, 1, 0.0, t3, 1);

                onsol = 2;
                /* printf("\t next is t3 \n \t"); */
                /* dprint(ncols,t3); */
            }
            else {
                cblas_dgemv(CblasColMajor,CblasTrans, 
                            nrows, ncols, 1.0,
                            t2, nrows,
                            t3,1,0.0,t1,1);
                onsol = 1;
                /* printf("\t next is t1  \n \t"); */
                /* dprint(ncols,t1); */
            }
        }
        
        nrows = ft->cores[dim_vary]->nrows;
        ncols = ft->cores[dim_vary]->ncols;
        if (onsol == 1){
            /* printf("t1 before multiplying by matrix is \n \t"); */
            /* dprint(nrows,t1); */
            /* dprint2d_col(nrows,ncols,temp_tensor); */
            for (size_t jj = 0; jj < N; jj++){
                cblas_dgemv(CblasColMajor,CblasTrans,
                            nrows, ncols, 1.0, temp_tensor + jj*nrows*ncols, 
                            nrows, t1, 1, 0.0, temp_mat + jj*ncols, 1);
            }
        }
        else{
            for (size_t jj = 0; jj < N; jj++){
                cblas_dgemv(CblasColMajor,CblasTrans,
                            nrows, ncols, 1.0, temp_tensor + jj*nrows*ncols, 
                            nrows, t3, 1, 0.0, temp_mat + jj*ncols, 1);
            }
        }
    }

    // got everything before and including and stored in temp_mat
    if (dim_vary == dim-1){ // done b/c ncols = 1
        /* printf("we are here\n"); */
        memmove(out,temp_mat, N * sizeof(double));
        /* printf("out = "); dprint(N,out); */
    }
    else{ // come from the back

        nrows = ft->cores[dim-1]->nrows;
        ncols = ft->cores[dim-1]->ncols;
        arr = ft->cores[dim-1]->funcs;
        generic_function_1darray_eval2_ind(nrows * ncols,arr,fixed_ind[dim-1],t1);
        int onsol = 1;
        for (size_t ii = dim-2; ii > dim_vary; ii--){
            nrows = ft->cores[ii]->nrows;
            ncols = ft->cores[ii]->ncols;
            arr = ft->cores[ii]->funcs;
            generic_function_1darray_eval2_ind(nrows*ncols,arr,fixed_ind[ii],t2);
            if (onsol == 1){
                // previous times new core
                cblas_dgemv(CblasColMajor,CblasNoTrans,
                            nrows, ncols, 1.0, t2, nrows,
                            t1, 1, 0.0, t3, 1);
                onsol = 2;
            }
            else {
                cblas_dgemv(CblasColMajor,CblasNoTrans, 
                            nrows, ncols, 1.0,
                            t2, nrows,
                            t3,1,0.0,t1,1);
                onsol = 1;
            }
        }

        // now combine
        ncols = ft->cores[dim_vary]->ncols;
        if (onsol == 1){
            cblas_dgemv(CblasColMajor,CblasTrans,
                        ncols, N, 1.0, temp_mat,
                        ncols, t1, 1, 0.0, out, 1);
        }
        else{
            cblas_dgemv(CblasColMajor,CblasTrans,
                        ncols, N, 1.0, temp_mat,
                        ncols, t3, 1, 0.0, out, 1);
        }

    }

    free(temp_mat); temp_mat = NULL;
    free(temp_tensor); temp_tensor = NULL;
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


    size_t maxrank;
    double *t1=NULL, *t2=NULL, *t3=NULL;
    function_train_pre_eval(ft,&maxrank,&t1,&t2,&t3,1);

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
            ft->cores[ii]->funcs, pert[2*ii],t1);
        generic_function_1darray_eval2(
            ft->cores[ii]->nrows * ft->cores[ii]->ncols,
            ft->cores[ii]->funcs, pert[2*ii+1], t2);
        if (ii == 0){
            vals[ii] = cblas_ddot(ft->ranks[1],t1,1,bprod[1],1);
            vals[ii+1] = cblas_ddot(ft->ranks[1],t2,1,bprod[1],1);
        }
        else if (ii == dim-1){
            vals[2*ii] = cblas_ddot(ft->ranks[ii],t1,1,
                                    fprod[dim-2],1);
            vals[2*ii+1] = cblas_ddot(ft->ranks[ii],t2,
                                      1,fprod[dim-2],1);
        }
        else{
            /* double * temp = calloc_double(ft->ranks[ii] * ft->ranks[ii+1]); */
            cblas_dgemv(CblasColMajor,CblasNoTrans,
                        ft->ranks[ii], ft->ranks[ii+1], 1.0,
                        t1, ft->ranks[ii],
                        bprod[ii+1], 1, 0.0, t3, 1);
            vals[2*ii] = cblas_ddot(ft->ranks[ii],t3,1,
                                    fprod[ii-1],1);

            cblas_dgemv(CblasColMajor,CblasNoTrans,
                        ft->ranks[ii], ft->ranks[ii+1], 1.0,
                        t2, ft->ranks[ii],
                        bprod[ii+1], 1, 0.0, t3, 1);

            vals[2*ii+1] = cblas_ddot(ft->ranks[ii],t3,1,
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

/***********************************************************//**
    Evaluate a function train with perturbations
    in every coordinate for nodal basis function

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
double function_train_eval_co_perturb_ind(struct FunctionTrain * ft, 
                                          const size_t * x, const size_t * pert, 
                                          double * vals)
{
    
    size_t dim = ft->dim;
    assert(ft->ranks[0] == 1);
    assert(ft->ranks[dim] == 1);

    size_t maxrank;
    double *t1=NULL, *t2=NULL, *t3=NULL;
    function_train_pre_eval(ft,&maxrank,&t1,&t2,&t3,1);   
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
            generic_function_1darray_eval2_ind(
                ft->cores[ii]->nrows * ft->cores[ii]->ncols,
                ft->cores[ii]->funcs, x[ii], fprod[ii]);
            memmove(cores_center[ii],fprod[ii],ft->ranks[1]*sizeof(double));
        }
        else{
            generic_function_1darray_eval2_ind(
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
            generic_function_1darray_eval2_ind(
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

        generic_function_1darray_eval2_ind(
            ft->cores[ii]->nrows * ft->cores[ii]->ncols,
            ft->cores[ii]->funcs, pert[2*ii], t1);
        generic_function_1darray_eval2_ind(
            ft->cores[ii]->nrows * ft->cores[ii]->ncols,
            ft->cores[ii]->funcs, pert[2*ii+1], t2);
        if (ii == 0){
            vals[ii] = cblas_ddot(ft->ranks[1],t1,1,bprod[1],1);
            vals[ii+1] = cblas_ddot(ft->ranks[1],t2,1,bprod[1],1);
        }
        else if (ii == dim-1){
            vals[2*ii] = cblas_ddot(ft->ranks[ii],t1,1,
                                    fprod[dim-2],1);
            vals[2*ii+1] = cblas_ddot(ft->ranks[ii],t2,
                                      1,fprod[dim-2],1);
        }
        else{
            /* double * temp = calloc_double(ft->ranks[ii] * ft->ranks[ii+1]); */
            cblas_dgemv(CblasColMajor,CblasNoTrans,
                        ft->ranks[ii], ft->ranks[ii+1], 1.0,
                        t1, ft->ranks[ii],
                        bprod[ii+1], 1, 0.0, t3, 1);
            vals[2*ii] = cblas_ddot(ft->ranks[ii],t3,1,
                                    fprod[ii-1],1);

            cblas_dgemv(CblasColMajor,CblasNoTrans,
                        ft->ranks[ii], ft->ranks[ii+1], 1.0,
                        t2, ft->ranks[ii],
                        bprod[ii+1], 1, 0.0, t3, 1);

            vals[2*ii+1] = cblas_ddot(ft->ranks[ii],t3,1,
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
            generic_function_approximate1d(aopt->fc,aopt->aopts,fw);
    }
    ft->ranks[dim] = 1;
    return ft;
}

/***********************************************************//**
    Compute a function train representation of \f$ f1(x_1) + f2(x_2) + ... + fd(x_d) \f$

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
        generic_function_approximate1d(aopt->fc,aopt->aopts,fw);
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
                generic_function_approximate1d(aopt->fc,aopt->aopts,fw);

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
            generic_function_approximate1d(aopt->fc,aopt->aopts,fw);
    }

    return ft;
}

/***********************************************************//**
    Create a zeros function train

    \param[in] aopts - approximation options
    \param[in] ranks - ranks

    \return function train

    \note
    Puts the constant into the first core
***************************************************************/
struct FunctionTrain *
function_train_zeros(struct MultiApproxOpts * aopts, const size_t * ranks)
{
    assert (aopts != NULL);
    
    size_t dim = multi_approx_opts_get_dim(aopts);
    struct FunctionTrain * ft = function_train_alloc(dim);
    memmove(ft->ranks,ranks,(dim+1) * sizeof(size_t));

    struct OneApproxOpts * oneopt = NULL;
    for (size_t onDim = 0; onDim < dim; onDim++){
        oneopt = multi_approx_opts_get_aopts(aopts,onDim);
        ft->cores[onDim] = qmarray_alloc(ranks[onDim],ranks[onDim+1]);
        for (size_t jj = 0; jj < ranks[onDim] * ranks[onDim+1]; jj++){
            ft->cores[onDim]->funcs[jj] =
                generic_function_zero(oneopt->fc,oneopt->aopts,1);
        }
    }
    return ft;
}

/***********************************************************//**
    Compute a function train representation of \f$ a \f$

    \param[in] a     - value
    \param[in] aopts - approximation options

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


    double sign = 1;
    if (a < 0){
        sign = -1;
    }
    double apow = pow(fabs(a),1.0/dim);
    /* printf("a = %3.15G apow = %3.15G\n",a,apow); */
    struct OneApproxOpts * oneopt = multi_approx_opts_get_aopts(aopts,onDim);
    ft->ranks[onDim] = 1;
    ft->cores[onDim] = qmarray_alloc(1,1);
    ft->cores[onDim]->funcs[0] =
        generic_function_constant(sign*apow,oneopt->fc,oneopt->aopts);

    for (onDim = 1; onDim < dim; onDim++){
        oneopt = multi_approx_opts_get_aopts(aopts,onDim);
        ft->ranks[onDim] = 1;
        ft->cores[onDim] = qmarray_alloc(1,1);
        ft->cores[onDim]->funcs[0] =
            generic_function_constant(apow,oneopt->fc,oneopt->aopts);
    }
    ft->ranks[dim] = 1;

    return ft;
}

/***********************************************************//**
    Compute a tensor train representation of

    \f$ (x_1c_1+a_1) + (x_2c_2+a_2)  + .... + (x_dc_d+a_d) \f$

    \param[in] c     - slope of the function in each dimension (ldc x dim)
    \param[in] ldc   - stride of c
    \param[in] a     - value
    \param[in] lda   - stride of a
    \param[in] aopts - approximation options

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
    Update a function-train representation of

    \f$ (x_1c_1+a_1) + (x_2c_2+a_2)  + .... + (x_dc_d+a_d) \f$

    \param[in] ft    - existing linear ft, formed by function_train_linear
    \param[in] c     - slope of the function in each dimension (ldc x dim)
    \param[in] ldc   - stride of c
    \param[in] a     - value
    \param[in] lda   - stride of a

    \return 0 if successfull, 1 otherwise

    \note This assumes that we are updating an FT created (and not modified)
    with function_train_linear. 
***************************************************************/
int
function_train_linear_update(
    struct FunctionTrain * ft,
    const double * c, size_t ldc, 
    const double * a, size_t lda)
{

    size_t dim = ft->dim;
    size_t onDim = 0;
    generic_function_linear_update(ft->cores[onDim]->funcs[0],
                                   c[onDim*ldc], a[onDim*lda]);
                                   
    
    if (dim > 1){
        for (onDim = 1; onDim < dim-1; onDim++){
            generic_function_linear_update(ft->cores[onDim]->funcs[1],
                                           c[onDim*ldc],
                                           a[onDim*lda]);
        }

        onDim = dim-1;
        generic_function_linear_update(ft->cores[onDim]->funcs[1],
                                       c[onDim*ldc],
                                       a[onDim*lda]);
    }

    return 0;
}

/***********************************************************//**
    Compute a function train representation of \f$ \prod_{i=1}^d(a_ix_i) \f$

    \param[in] coeffs   - coefficients
    \param[in] aopts    - approximation options

    \returns function train

***************************************************************/
struct FunctionTrain *
function_train_rankone_prod(const double * coeffs,
                            struct MultiApproxOpts * aopts)
{
    assert (aopts != NULL);
    size_t dim = multi_approx_opts_get_dim(aopts);
    assert (dim > 1); //

    struct FunctionTrain * ft = function_train_alloc(dim);
    struct OneApproxOpts * o = NULL;

    size_t onDim = 0;
    ft->ranks[onDim] = 1;
    ft->cores[onDim] = qmarray_alloc(1,1);

    o = multi_approx_opts_get_aopts(aopts,onDim);

    // should be quadratic
    ft->cores[onDim]->funcs[0] = 
        generic_function_linear(coeffs[onDim],0.0,o->fc,o->aopts);

    for (onDim = 1; onDim < dim-1; onDim++){
        //printf("on dimension (%zu/%zu)\n",onDim+1,dim);

        o = multi_approx_opts_get_aopts(aopts,onDim);
        ft->ranks[onDim] = 1;
        ft->cores[onDim] = qmarray_alloc(1,1);
        ft->cores[onDim]->funcs[0] = 
            generic_function_linear(coeffs[onDim],0.0,o->fc,o->aopts);
    }

    onDim = dim-1;
    o = multi_approx_opts_get_aopts(aopts,onDim);
    ft->ranks[onDim] = 1;
    ft->cores[onDim] = qmarray_alloc(1,1);

    ft->cores[onDim]->funcs[0] =
        generic_function_linear(coeffs[onDim],0.0,o->fc,o->aopts);

    ft->ranks[dim] = 1;
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
    Interpolate a function-train onto a particular grid forming
    another function_train with a nodela basis

    \param[in] fta - Function train array to evaluate
    \param[in] N   - number of nodes in each dimension
    \param[in] x   - nodes in each dimension

    \return new function train
***********************************************************/
struct FunctionTrain *
function_train_create_nodal(const struct FunctionTrain * fta, size_t * N, double ** x)
{
    struct FunctionTrain * newft = function_train_alloc(fta->dim);
    memmove(newft->ranks,fta->ranks, (fta->dim+1)*sizeof(size_t));
    for (size_t ii = 0; ii < newft->dim; ii ++){
        newft->cores[ii] = qmarray_create_nodal(fta->cores[ii],N[ii],x[ii]);
    }
    return newft;
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
    Integrate a function in function train format with respect
    to an appropriately weight

    \param[in] ft - Function train 1

    \return Integral \f$ \int f(x) w(x) dx \f$
    
    \note w(x) depends on underlying parameterization
    for example, it is 1/2 for legendre (and default for others),
    gauss for hermite,etc
***********************************************************/
double
function_train_integrate_weighted(const struct FunctionTrain * ft)
{
    size_t dim = ft->dim;
    size_t ii = 0;
    double * t1 = qmarray_integrate_weighted(ft->cores[ii]);

    double * t2 = NULL;
    double * t3 = NULL;
    for (ii = 1; ii < dim; ii++){
        t2 = qmarray_integrate_weighted(ft->cores[ii]);
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
    Contract the function along certain dimensions with respect
    to an appropriate reight

    \param[in] ft            - Function train 1
    \param[in] ndim_contract - number of dimensions to contract
    \param[in] dims_contract - dimensions over which to contract (increasing order)
    
    \return Evaluates \f$ g(x_{\neq c}) = \int f(x_c) w(x_c) dx_c \f$
    
    \note w(x) depends on underlying parameterization
    for example, it is 1/2 for legendre (and default for others),
    gauss for hermite,etc
***********************************************************/
struct FunctionTrain *
function_train_integrate_weighted_subset(
    const struct FunctionTrain * ft,
    size_t ndim_contract,
    size_t * dims_contract)
{

    
    size_t newdim = ft->dim - ndim_contract;
    assert (newdim > 0);
    /* printf("newdim = %zu\n",newdim); */
    struct FunctionTrain * newft = function_train_alloc(newdim);
    newft->ranks[0] = 1;
    newft->ranks[newdim] = 1;

    double * left_mult = NULL;
    size_t ncols = 0;

    size_t on_contract = 0;
    size_t ii = 0;
    
    while ((dims_contract[on_contract] == ii ) && (on_contract < ndim_contract)){
        double * mat = qmarray_integrate_weighted(ft->cores[ii]);

        if (left_mult == NULL){
            ncols = ft->cores[ii]->ncols;
            left_mult = calloc_double(ncols);
            memmove(left_mult,mat,ncols * sizeof(double));
        }
        else{
            double * new_left_mult = calloc_double(ft->cores[ii]->ncols);
            c3linalg_multiple_vec_mat(1,ft->cores[ii]->nrows,ft->cores[ii]->ncols,
                                      left_mult,ncols,
                                      mat,ft->cores[ii]->nrows*ft->cores[ii]->ncols,
                                      new_left_mult,ft->cores[ii]->ncols);
            free(left_mult); left_mult = NULL;

            ncols = ft->cores[ii]->ncols;
            left_mult = calloc_double(ncols);
            memmove(left_mult,new_left_mult,ncols*sizeof(double));
            free(new_left_mult); new_left_mult = NULL;
        }

        free(mat); mat = NULL;
        ii++;
        on_contract++;

        if (on_contract == ndim_contract){
            break;
        }
    }

    size_t on_new_core = 0;
    int done = 0;
    if (on_contract == ndim_contract){
        on_contract--;
        done = 1;
    }
    for (; ii < ft->dim; ii++){
        if ((dims_contract[on_contract] == ii) && (done == 0)){
            double * mat = qmarray_integrate_weighted(ft->cores[ii]);

            struct Qmarray * newcore = qmam(newft->cores[on_new_core-1],mat,ft->cores[ii]->ncols);
            qmarray_free(newft->cores[on_new_core-1]);
            newft->cores[on_new_core-1] = qmarray_copy(newcore);

            on_contract++;
            if (on_contract == ndim_contract){
                done = 1;
                on_contract--;
            }
            free(mat); mat = NULL;
            qmarray_free(newcore); newcore = NULL;
        }
        else{
            if (left_mult != NULL){
                newft->cores[on_new_core] = mqma(left_mult,ft->cores[ii],1);
                free(left_mult); left_mult = NULL;
            }
            else{
                newft->cores[on_new_core] = qmarray_copy(ft->cores[ii]);
            }
            on_new_core++;
        }
    }

    
    for (ii = 0; ii < newdim; ii++){
        newft->ranks[ii]   = newft->cores[ii]->nrows;
        newft->ranks[ii+1] = newft->cores[ii]->ncols;
    }
    
    return newft;
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

    double * temp = qmarray_kron_integrate(b->cores[0],a->cores[0]);
    double * temp2 = NULL;

    for (size_t ii = 1; ii < a->dim; ii++){
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
    Inner product between two functions in FT form with weighted
    space w(x)

    \param[in] a - Function train 1
    \param[in] b - Function train 2

    \return Inner product \f$ \int a(x)b(x) w(x) dx \f$

    \note 
    In order for this function to make sense a(x) and b(x)
    shoud have the same underlying basis
***********************************************************/
double function_train_inner_weighted(const struct FunctionTrain * a, 
                                     const struct FunctionTrain * b)
{
    double out = 0.123456789;
    size_t ii;
    double * temp = qmarray_kron_integrate_weighted(b->cores[0],a->cores[0]);
    double * temp2 = NULL;

    //size_t ii;
    for (ii = 1; ii < a->dim; ii++){
        temp2 = qmarray_vec_kron_integrate_weighted(temp, a->cores[ii],b->cores[ii]);
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
            /* fprintf(stderr, "inner product of FT with itself should not be neg %G \n",out); */
            /* exit(1); */
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
    /* printf("go orthor, core norm = %3.15G\n", qmarray_norm2(a->cores[core])); */
    ftrl->cores[core] = qmarray_householder_simple("LQ",a->cores[core],L,o);
    /* printf("go orthor, core norm = %3.15G\n", qmarray_norm2(ftrl->cores[core])); */
    /* ftrl->cores[core] = mqma(L, ftrl->cores[core], ftrl->cores[core]->nrows); */
    /* printf("done with LQ\n"); */
    for (ii = 2; ii < a->dim; ii++){
        core = a->dim-ii;
        //printf("on core %zu\n",core);
        temp = qmam(a->cores[core],L,ftrl->ranks[core+1]);
        free(L); L = NULL;
        L = calloc_double(ftrl->ranks[core]*ftrl->ranks[core]);
        o = multi_approx_opts_get_aopts(aopts,core);
        ftrl->cores[core] = qmarray_householder_simple("LQ",temp,L,o);
        /* printf("go orthor, core norm = %3.15G\n", qmarray_norm2(ftrl->cores[core])); */
        qmarray_free(temp); temp = NULL;
        
    }
    ftrl->cores[0] = qmam(a->cores[0],L,ftrl->ranks[1]);
    /* ftrl->cores[0] = qmarray_copy(a->cores[0]); */
    /* printf("go orthor, core norm = %3.15G\n", qmarray_norm2(ftrl->cores[core]));     */
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
    double normain = function_train_norm2(ain);
    /* printf("normain = %3.15G\n", normain); */
    if (normain < ZEROTHRESH){
        return function_train_constant(0.0, aopts);
    }
    struct OneApproxOpts * o = NULL;
    struct FunctionTrain * a = function_train_copy(ain);
//    size_t ii;
    size_t core;
    struct Qmarray * temp = NULL;

    double delta = function_train_norm2(a);
    delta = delta * epsilon / sqrt(a->dim-1);
    //double delta = epsilon;

    /* printf("Rounding Starting norm = %G\n",function_train_norm2(a)); */
//    printf("begin orho\n");
    struct FunctionTrain * ftrl = function_train_orthor(a,aopts);

    /* printf("Rounding Ortho norm %G\n",function_train_norm2(ftrl)); */
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
    Truncate functions in FT

    \param[in,out] ft - FT
    \param[in] epsilon - threshold
***********************************************************/
void function_train_truncate(struct FunctionTrain * ft, double epsilon)
{
    for (size_t ii = 0; ii < ft->dim; ii++){
        qmarray_roundt(&ft->cores[ii], epsilon);
    }    
}

/********************************************************//**
    Reapproximate a function train with new approximation options
    in a nonadaptive way. MultiApproxOpts should have all info needed

    \param[in,out] ft    - function train to reapproximate
    \param[in]     aopts - approximation options

***********************************************************/
/* void function_train_reapprox_nonadapt(struct FunctionTrain * ft, struct MultiApproxOpts * aopts) */
/* { */
/*     assert (ft != NULL); */
/*     assert (aopts != NULL); */
/*     struct OneApproxOpts * o = NULL; */
/*     for (size_t ii = 0; ii < ft->dim; ii++){ */
/*         o = multi_approx_opts_get_aopts(aopts,ii); */
/*     } */
/* } */

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
/////////////////////    Cross Approximation   ///////////////////////////
/////////////////////                          ///////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

/** \struct FtCrossArgs
 *  \brief Arguments for function-train cross approximation
 *  \var FtCrossArgs::dim
 *  dimension of function
 *  \var FtCrossArgs::ranks
 *  (dim+1,) array of ranks
 *  \var FtCrossArgs::epsilon
 *  cross-approximation convergence criteria
 *  \var FtCrossArgs::maxiter
 *  maximum number of iteration for cross approximation
 *  \var FtCrossArgs::epsround
 *  rounding tolerance for adaptive rank cross approximation
 *  \var FtCrossArgs::kickrank
 *  size of rank increase for adaptation
 *  \var FtCrossArgs::maxranks
 *  maximum rank to go to during adaptation (dim-1,1)
 *  \var FtCrossArgs::verbose
 *  verbosity level (0,1,2)
 * */
struct FtCrossArgs
{
    size_t dim;
    size_t * ranks;
    double epsilon;
    size_t maxiter;
    
    // adaptation parameters
    int adapt;
    double epsround;
    size_t kickrank;
    size_t * maxranks; //maxiteradapt;

    int verbose;
};

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

    return ftc;
}

/***********************************************************//**
    Set the rank for a particular index;
***************************************************************/
void ft_cross_args_set_rank(struct FtCrossArgs * fca, size_t ind, size_t rank)
{
    fca->ranks[ind] = rank;
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
    Get the ranks
***************************************************************/
size_t * ft_cross_args_get_ranks(const struct FtCrossArgs * fca)
{
    assert (fca != NULL);
    return fca->ranks;
}



/***********************************************************//**
    Copy cross approximation arguments
***************************************************************/
struct FtCrossArgs * ft_cross_args_copy(const struct FtCrossArgs * fca)
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

struct Qmarray *
prepCore(size_t ii, size_t nrows, size_t ncols,
         struct Fwrap * fw, struct OneApproxOpts * o,
         struct CrossIndex ** left_ind,struct CrossIndex ** right_ind)
{
    assert (fw != NULL);
    assert (o != NULL);
    assert (left_ind != NULL);
    assert (right_ind != NULL);

    if (VPREPCORE) printf("in prepCore \n");
    size_t ncuts = nrows * ncols;
    fwrap_initialize_fiber_approx(fw,ii,ncuts);
    if (VPREPCORE) printf("initialized fiber approx ncuts=%zu \n",ncuts);

    double * left = NULL, *right = NULL;
    size_t nl, nr;
    for (size_t jj = 0; jj < ncols; jj++){
        nr = 0;
        right = cross_index_get_node_value(right_ind[ii],jj,&nr);
        for (size_t kk = 0; kk < nrows; kk++){
            nl = 0;
            left = cross_index_get_node_value(left_ind[ii],kk,&nl);
            if (VPREPCORE){
                printf("left = "); dprint(nl,left);
                printf("right = "); dprint(nr,right);
            }
            
            fwrap_add_fiber(fw,jj*nrows+kk,nl,left,nr,right);
        }
    }

    if (VPREPCORE) printf("compute from fibercuts \n");
   
    struct Qmarray * temp = qmarray_alloc(nrows,ncols);
    for (size_t jj = 0; jj < ncols; jj++){
        for (size_t kk = 0; kk < nrows; kk++){
            fwrap_set_which_fiber(fw,jj*nrows+kk);
            temp->funcs[jj*nrows+kk] = 
                generic_function_approximate1d(o->fc,o->aopts,fw);
        }
    }

    if (VPREPCORE) printf("computed!\n");
    

    fwrap_clean_fiber_approx(fw);
    return temp;
}



/***********************************************************//**
    Initialize an FT from left and right indices
    *UNTESTED* USE AT YOUR OWN RISK
    *UNTESTED* USE AT YOUR OWN RISK
    \param[in] fw        - wrapped function
    \param[in] ranks     - ranks to use
    \param[in] left_ind  - left indices
    \param[in] right_ind - right indices
    \param[in] apargs    - approximation arguments
    *UNTESTED* USE AT YOUR OWN RISK
    \return function train decomposition of \f$ f \f$
    *UNTESTED* USE AT YOUR OWN RISK
***************************************************************/
struct FunctionTrain *
function_train_init_from_fibers(struct Fwrap * fw,
                                size_t * ranks,
                                struct CrossIndex ** left_ind,
                                struct CrossIndex ** right_ind,
                                struct MultiApproxOpts * apargs)
{
    assert (fw != NULL);
    assert (left_ind != NULL);
    assert (right_ind != NULL);
    assert (apargs != NULL);

    size_t d = multi_approx_opts_get_dim(apargs);
    struct FunctionTrain * ft = function_train_alloc(d);
    memmove(ft->ranks, ranks, (d+1)*sizeof(size_t));


    struct OneApproxOpts * o = NULL;
    for (size_t ii = 0; ii < d; ii++){
        o = multi_approx_opts_get_aopts(apargs,ii);
        ft->cores[ii] = prepCore(ii,ranks[ii],ranks[ii+1],fw,o,
                                 left_ind,right_ind);

    }
    struct FunctionTrain * ft2 = function_train_round(ft,ZEROTHRESH,apargs);
    function_train_free(ft); ft = NULL;

    return ft2;
}




/***********************************************************//**
    Cross approximation of a of a dim-dimensional function
    (with adaptation)
    
    \param[in]     fw        - wrapped function
    \param[in]     cargs     - cross approximation arguments
    \param[in,out] left_ind  - left indices
    \param[in,out] right_ind - right indices
    \param[in]     apargs    - approximation arguments
    \param[in]     optargs   - fiber optimizationa arguments
    \param[in]     ftref     - reference ft for first error approximation

    \return function train decomposition of \f$ f \f$

    \note
    both left and right indices are nested
***************************************************************/
struct FunctionTrain *
ftapprox_cross(struct Fwrap * fw,
               struct FtCrossArgs * cargs,
               struct CrossIndex ** left_ind,
               struct CrossIndex ** right_ind,
               struct MultiApproxOpts * apargs,
               struct FiberOptArgs * optargs,
               struct FunctionTrain * ftref)
{
    assert (fw != NULL);
    assert (cargs != NULL);
    assert (left_ind != NULL);
    assert (right_ind != NULL);
    assert (apargs != NULL);
    assert (optargs != NULL);
    
    size_t dim = multi_approx_opts_get_dim(apargs);
    struct OneApproxOpts * o = NULL;
    void * opt = NULL;
    int info;
    /* size_t nrows, ii, oncore; */
    size_t ii,oncore;
    struct Qmarray * temp = NULL;
    struct Qmarray * Q = NULL;
    struct Qmarray * Qt = NULL;
    double * R = NULL;
    size_t * pivind = NULL;
    double * pivx = NULL;
    double diff, diff2, den;
    
    struct FunctionTrain * ft = function_train_alloc(dim);
    memmove(ft->ranks, cargs->ranks, (dim+1)*sizeof(size_t));
    size_t * ranks = function_train_get_ranks(ft);

    struct FunctionTrain * fti = function_train_copy(ftref);
    struct FunctionTrain * fti2 = NULL;

    int done = 0;
    size_t iter = 0;

    while (done == 0){
        if (cargs->verbose > 0)
            printf("cross iter=%zu \n",iter);
      
        // left right sweep;
//        nrows = 1;
        for (ii = 0; ii < dim-1; ii++){
            o = multi_approx_opts_get_aopts(apargs,ii);
            opt = fiber_opt_args_get_opts(optargs,ii);

            //  printf("sub_type ftcross= %d\n",
            //         *(int *)ft_approx_args_getst(apargs,ii));
            if (cargs->verbose > 1){
                printf(" ............. on left-right sweep (%zu/%zu)\n",ii,dim-1);
            }
            //printf("ii=%zu\n",ii);
            pivind = calloc_size_t(ft->ranks[ii+1]);
            pivx = calloc_double(ft->ranks[ii+1]);
            
            if (VFTCROSS){
                printf("=======================================\n\n\n\n");
                printf( "prepCore \n");
                printf( "left index set = \n");
                print_cross_index(left_ind[ii]);
                printf( "right index set = \n");
                print_cross_index(right_ind[ii]);
            }

            /* printf("prepCore\n"); */
            temp = prepCore(ii,ranks[ii],ranks[ii+1],fw,o,
                            left_ind,right_ind);
            /* printf("prepped\n"); */

            if (VFTCROSS == 2){
                printf ("got it \n");
                //print_qmarray(temp,0,NULL);
                struct Qmarray * tempp = qmarray_copy(temp);
                printf("core is \n");
                //print_qmarray(tempp,0,NULL);
                R = calloc_double(temp->ncols * temp->ncols);
                Q = qmarray_householder_simple("QR",temp,R,o);
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
                Q = qmarray_householder_simple("QR", temp,R,o);
            }

            
            info = qmarray_maxvol1d(Q,R,pivind,pivx,o,opt);

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
                    cross_index_add_index(left_ind[1],1,&(pivx[zz]),sizeof(double));
                }
            }
            
            qmarray_free(ft->cores[ii]); ft->cores[ii]=NULL;
            ft->cores[ii] = qmam(Q,R, temp->ncols);
//            nrows = left_ind[ii+1]->n;

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
        /* ft->cores[ii] = prepCore(ii,cargs->ranks[ii],f,args,bd, */
        /*                          left_ind,right_ind,cargs,apargs,1); */
        o = multi_approx_opts_get_aopts(apargs,ii);
        ft->cores[ii] = prepCore(ii,ranks[ii],ranks[ii+1],fw,o,
                                 left_ind,right_ind);

        if (VFTCROSS == 2){
            printf ("got it \n");
            //print_qmarray(ft->cores[ii],0,NULL);
            printf("integral = %G\n",function_train_integrate(ft));
            struct FunctionTrain * tprod = function_train_product(ft,ft);
            printf("prod integral = %G\n",function_train_integrate(tprod));
            printf("norm2 = %G\n",function_train_norm2(ft));
            print_qmarray(tprod->cores[0],0,NULL,stdout);
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

        //printf("diff = %G\n",diff);

       /*  diff = function_train_norm2diff(ft,fti); */
       /*  if (den > ZEROTHRESH){ */
       /*     diff /= den; */
       /* } */

        /* printf("compute norm\n"); */

        diff = function_train_norm2diff(ft,fti);
        den = function_train_norm2(ft);

        if (den > 1e0){
            diff = diff / den;
        }

        // uncomment below to have error checking after a L/R sweep
        /* diff = function_train_relnorm2diff(ft,fti); */
        /* den = function_train_norm2(ft); */
        if (cargs->verbose > 0){
            den = function_train_norm2(ft);
            printf("...... New FT norm L/R Sweep = %3.15E\n",den);
            printf("...... Error L/R Sweep = %E\n",diff);
        }
        

        /* if (diff < cargs->epsilon){ */
        /*     done = 1; */
        /*     break; */
        /* } */


        // Can't do this because rank-adaptation doesn't care about
        // left_ind so need to end on a right-left sweep
        /* if (diff < cargs->epsilon){ */
        /*     done = 1; */
        /*     break; */
        /* } */

        
        /* function_train_free(fti); fti=NULL; */
        /* fti = function_train_copy(ft); */

        //Up to here
        ////

        
        //printf("copied \n");
        //printf("copy diff= %G\n", function_train_norm2diff(ft,fti));

        ///////////////////////////////////////////////////////
        // right-left sweep
        for (oncore = 1; oncore < dim; oncore++){
            
            ii = dim-oncore;
            o = multi_approx_opts_get_aopts(apargs,ii);
            opt = fiber_opt_args_get_opts(optargs,ii);


            if (cargs->verbose > 1){
                printf(" ............. on right_left sweep (%zu/%zu)\n",ii,dim-1);
            }

//            nrows = ft->ranks[ii];

            if (VFTCROSS){
                printf("do prep\n");
                printf( "left index set = \n");
                print_cross_index(left_ind[ii]);
                printf( "right index set = \n");
                print_cross_index(right_ind[ii]);
            }
            //printf("prep core\n");
            /* temp = prepCore(ii,nrows,f,args,bd,left_ind,right_ind,cargs,apargs,0); */
            temp = prepCore(ii,ranks[ii],ranks[ii+1],fw,o,left_ind,right_ind);

            /* printf("right after prepping core\n"); */
            /* print_qmarray(temp,0,NULL); */
            //printf("prepped core\n");

            R = calloc_double(temp->nrows * temp->nrows);
            Q = qmarray_householder_simple("LQ", temp,R,o);

            /* printf("right after taking QR of prepped core\n"); */
            /* print_qmarray(Q,0,NULL); */
            
            Qt = qmarray_transpose(Q);
            pivind = calloc_size_t(ft->ranks[ii]);
            pivx = calloc_double(ft->ranks[ii]);
            info = qmarray_maxvol1d(Qt,R,pivind,pivx,o,opt);
                        
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
                    cross_index_add_index(right_ind[ii-1],1,&(pivx[zz]),sizeof(double));
                }
                //printf("updated\n");
            }

            qmarray_free(temp); temp = NULL;
            qmarray_free(Q); Q = NULL;
            qmarray_free(Qt); Qt = NULL;
            free(pivind);
            free(pivx);
            free(R); R=NULL;

            /* break; //AG REMOVE THIS 5/18 */

        }

        ii = 0;
        qmarray_free(ft->cores[ii]); ft->cores[ii] = NULL;

        if (cargs->verbose > 1)
            printf(" ............. on right_left sweep (%zu/%zu)\n",ii,dim-1);
        /* ft->cores[ii] = prepCore(ii,1,f,args,bd,left_ind,right_ind,cargs,apargs,-1); */
        o = multi_approx_opts_get_aopts(apargs,ii);
        ft->cores[ii] = prepCore(ii,ranks[ii],ranks[ii+1],fw,o,left_ind,right_ind);
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
            printf("...... New FT norm R/L Sweep = %3.15E\n",den);
            printf("...... Error R/L Sweep = %E,%E\n",diff,diff2);
        }

        if ( (diff2 < cargs->epsilon) && (diff < cargs->epsilon)){
        /* if ( (diff2 < cargs->epsilon) || (diff < cargs->epsilon)){ */
        /* if ( diff < cargs->epsilon){ */
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
    Cross approximation of a of a dim-dimensional function with rank adaptation

    \param[in]     fw      - wrapped function
    \param[in]     cargs   - cross approximation arguments
    \param[in,out] isl     - left indices
    \param[in,out] isr     - right indices
    \param[in]     apargs  - approximation arguments
    \param[in]     optargs - fiber optimizationa arguments
    \param[in]     ftref   - reference ft for first error approximation

    \return function train decomposition of f

    \note
    both left and right indices are nested
***************************************************************/
struct FunctionTrain *
ftapprox_cross_rankadapt(struct Fwrap * fw,
                         struct FtCrossArgs * cargs,
                         struct CrossIndex ** isl, 
                         struct CrossIndex ** isr,
                         struct MultiApproxOpts * apargs,
                         struct FiberOptArgs * optargs,
                         struct FunctionTrain * ftref)
{


    assert (fw != NULL);
    assert (cargs != NULL);
    assert (isl != NULL);
    assert (isr != NULL);
    assert (apargs != NULL);
    assert (optargs != NULL);

    size_t dim = multi_approx_opts_get_dim(apargs);
    double eps = cargs->epsround;
    size_t kickrank = cargs->kickrank;

    struct FunctionTrain * ft = NULL;

    ft = ftapprox_cross(fw,cargs,isl,isr,apargs,optargs,ftref);
    if (cargs->adapt == 0){
        return ft;
    }
    // printf("found left index\n");
    // print_cross_index(isl[1]);
    // printf("found right index\n");
    //print_cross_index(isr[0]);
    
    //return ft;
    if (cargs->verbose > 0){
        printf("done with first cross... rounding\n");
    }
    size_t * ranks_found = calloc_size_t(dim+1);
    memmove(ranks_found,cargs->ranks,(dim+1)*sizeof(size_t));
    
    struct FunctionTrain * ftc = function_train_copy(ft);
    struct FunctionTrain * ftr = function_train_round(ft,eps,apargs);
    if (cargs->verbose > 0){
        printf("rounded ranks = "); iprint_sz(dim+1,ftr->ranks);
    }
    //struct FunctionTrain * ftr = function_train_copy(ft);
    //return ftr; 
    //printf("DOOONNTT FORGET MEEE HEERREEEE \n");
    int adapt = 0;
    size_t ii;
    for (ii = 1; ii < dim; ii++){
        /* printf("%zu == %zu\n",ranks_found[ii],ftr->ranks[ii]); */
        if (ranks_found[ii] == ftr->ranks[ii]){
            if (cargs->ranks[ii] < cargs->maxranks[ii-1]){
                adapt = 1;
                size_t kicksize; 
                if ( (ranks_found[ii]+kickrank) <= cargs->maxranks[ii-1]){
                    kicksize = kickrank;
                }
                else{
                    kicksize = cargs->maxranks[ii-1] -ranks_found[ii];
                }
                cargs->ranks[ii] = ranks_found[ii] + kicksize;
                
                // simply repeat the last nodes
                // this is not efficient but I don't have a better
                // idea. Could do it using random locations but
                // I don't want to.
                // I don't need to modify isl because it is rewritten on the 
                // first left-right sweep anyways
                cross_index_copylast(isr[ii-1],kicksize);

            
                ranks_found[ii] = cargs->ranks[ii];
            }
        }
    }

    //printf("adapt here! adapt=%zu\n",adapt);

    size_t iter = 0;
//    double * xhelp = NULL;
    while ( adapt == 1 )
    {
        adapt = 0;
        if (cargs->verbose > 0){
            printf("adapting \n");            
            printf("Increasing rank to \n");
            iprint_sz(ft->dim+1,cargs->ranks);
        }
        
        function_train_free(ft); ft = NULL;
        
        ft = ftapprox_cross(fw,cargs,isl,isr,apargs,optargs,ftref);
         
        function_train_free(ftc); ftc = NULL;
        function_train_free(ftr); ftr = NULL;

        //printf("copying\n");
        function_train_free(ftc); ftc = NULL;
        ftc = function_train_copy(ft);
        //printf("rounding\n");
        function_train_free(ftr); ftr = NULL;
        //ftr = function_train_copy(ft);//, eps);
        ftr = function_train_round(ft,eps,apargs);
        if (cargs->verbose > 0){
            printf("rounded ranks = "); iprint_sz(dim+1,ftr->ranks);
        }
        //printf("done rounding\n");
        for (ii = 1; ii < dim; ii++){
            if (ranks_found[ii] == ftr->ranks[ii]){
                if (cargs->ranks[ii] < cargs->maxranks[ii-1]){
                    adapt = 1;
                    size_t kicksize; 
                    if ( (ranks_found[ii]+kickrank) <= cargs->maxranks[ii-1]){
                        kicksize = kickrank;
                    }
                    else{
                        kicksize = cargs->maxranks[ii-1] -ranks_found[ii];
                    }
                    cargs->ranks[ii] = ranks_found[ii] + kicksize;
                
            
                    // simply repeat the last nodes
                    // this is not efficient but I don't have a better
                    // idea. Could do it using random locations but
                    // I don't want to.
                    cross_index_copylast(isr[ii-1],kicksize);
            
                    ranks_found[ii] = cargs->ranks[ii];
                }
            }
        }

        iter++;
        /* if (iter == fca->maxiteradapt) { */
        /*     adapt = 0; */
        /* } */
        //adapt = 0;

    }

    if (cargs->verbose > 0){
        printf("Final norm = %G\n",function_train_norm2(ftr));
        printf("Final ranks: ");
        iprint_sz(ftr->dim+1,ftr->ranks);
    }
    
    function_train_free(ft); ft = NULL;
    function_train_free(ftc); ftc = NULL;
    free(ranks_found);
    return ftr;
}


/***********************************************************//**
    Determine whether kristoffel weighting is active

    \param[in] ft - function_train

    \return 1 if active, 0 otherwise

    \note It is assumed that if kristoffel is active on 1
    core it is active on all cores
***************************************************************/
int function_train_is_kristoffel_active(const struct FunctionTrain * ft)
{
    int active = qmarray_is_kristoffel_active(ft->cores[0]);
    for (size_t ii = 1; ii < ft->dim; ii++){

        if (qmarray_is_kristoffel_active(ft->cores[ii]) != active){
            fprintf(stderr, "Kristoffel weighting cannot be active on some cores\n");
            fprintf(stderr, "and inactive on other cores\n");
            exit(1);
        }
    }

    return active;
}

void function_train_activate_kristoffel(struct FunctionTrain *ft)
{
    for (size_t ii = 0; ii < ft->dim; ii++){
        qmarray_activate_kristoffel(ft->cores[ii]);
    }
}

void function_train_deactivate_kristoffel(struct FunctionTrain *ft)
{
    for (size_t ii = 0; ii < ft->dim; ii++){
        qmarray_deactivate_kristoffel(ft->cores[ii]);
    }
}

/***********************************************************//**
    Get the kristoffel normalization factor                                                            

    \param[in] ft - function train
    \param[in] x  - location at which to obtain normalization

    \return normalization factor
***************************************************************/
double function_train_get_kristoffel_weight(const struct FunctionTrain * ft,
                                            const double * x)
{
    double weight = 1.0;
    for (size_t ii = 0; ii < ft->dim; ii++){
        weight *= qmarray_get_kristoffel_weight(ft->cores[ii],x[ii]);
    }

    return weight;
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
struct FT1DArray * ft1d_array_copy(const struct FT1DArray * fta)
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

/********************************************************//**
    Compute the gradient of a function train

    \param[in] ft - Function train

    \return gradient
***********************************************************/
struct FT1DArray * function_train_gradient(const struct FunctionTrain * ft)
{
    struct FT1DArray * ftg = ft1d_array_alloc(ft->dim);
    size_t ii;
    for (ii = 0; ii < ft->dim; ii++){
        ftg->ft[ii] = function_train_copy(ft);
        qmarray_free(ftg->ft[ii]->cores[ii]);
        ftg->ft[ii]->cores[ii] = NULL;
        ftg->ft[ii]->cores[ii] = qmarray_deriv(ft->cores[ii]);
    }

    return ftg;
}



/********************************************************//**
    Evaluate the gradient of a function train

    \param[in]     ft  - Function train
    \param[in]     x   - location at which to evaluate the gradient
    \param[in,out] out - allocate space for the gradient
***********************************************************/
void function_train_gradient_eval(const struct FunctionTrain * ft,
                                  const double * x,
                                  double * out)
{
    /* (void)ft; */
    /* (void)x; */
    /* (void)out; */
    /* assert (1 == 0); */
    
    size_t dim  = ft->dim;
    size_t maxrank = function_train_get_maxrank(ft);
    size_t * ranks = ft->ranks;
    
    size_t mr2 = maxrank*maxrank;
    double * evals_lr = calloc_double(maxrank * dim + maxrank);
    double * evals_rl = calloc_double(maxrank * dim + maxrank);
    double * grads_lr = calloc_double(maxrank * dim + maxrank);
    double * evals = calloc_double(mr2 * dim);
    double * grads  = calloc_double(mr2 * dim);
    size_t onuni = 0;
    size_t indmem = 0;

    size_t kk = 0;
    for (size_t col = 0; col < ranks[kk+1]; col++){
        evals_lr[col] = generic_function_1d_eval(ft->cores[kk]->funcs[col],x[kk]);
        grads_lr[col] = generic_function_deriv_eval(ft->cores[kk]->funcs[col],x[kk]);
        onuni++;
    }
    
    for (kk = 1; kk < dim; kk++){
        for (size_t col = 0; col < ranks[kk+1]; col++){
            evals_lr[indmem+ranks[kk]+col] = 0.0;
            grads_lr[indmem+ranks[kk]+col] = 0.0;
            for (size_t row = 0; row < ranks[kk]; row++){
                evals[onuni] = generic_function_1d_eval(ft->cores[kk]->funcs[row + col * ranks[kk]],x[kk]);
                grads[onuni] = generic_function_deriv_eval(ft->cores[kk]->funcs[row + col * ranks[kk]],x[kk]);
                evals_lr[indmem+ranks[kk]+col] += evals_lr[indmem+row] * evals[onuni];
                grads_lr[indmem+ranks[kk]+col] += evals_lr[indmem+row] * grads[onuni];
                onuni++;
            }
        }
        indmem += ranks[kk];
    }

    out[dim-1] = grads_lr[indmem];

    size_t ind_eval = onuni-ranks[dim-1];
    size_t ind_rl = 0,r1,r2,backind;
    memmove(evals_rl, evals + ind_eval, ranks[dim-1] * sizeof(double));

    for (size_t zz = 1; zz < dim-1; zz++)
    {
        backind = dim-1-zz;
        r1 = ranks[backind];
        r2 = ranks[backind+1];

        indmem -= r2;

        out[backind] = cblas_ddot(r2,grads_lr + indmem,1,evals_rl + ind_rl, 1);


        ind_eval -= ranks[backind]*ranks[backind+1];

        cblas_dgemv(CblasColMajor, CblasNoTrans,
                    r1,r2,1.0,
                    evals + ind_eval, r1,
                    evals_rl + ind_rl,1,
                    0.0,evals_rl + ind_rl + r2, 1);
        ind_rl += r2;
    }

    backind = 0;
    out[backind] = cblas_ddot(ranks[1],grads_lr,1,evals_rl + ind_rl, 1);

    free(evals_lr); evals_lr = NULL;
    free(evals_rl); evals_rl = NULL;
    free(grads_lr); grads_lr = NULL;
    free(evals);    evals = NULL;
    free(grads);    grads = NULL;
}


/********************************************************//**
    Compute the Jacobian of a Function Train 1darray

    \param[in] fta - Function train array

    \return jacobian
***********************************************************/
struct FT1DArray * ft1d_array_jacobian(const struct FT1DArray * fta)
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
struct FT1DArray * function_train_hessian(const struct FunctionTrain * fta)
{
    struct FT1DArray * ftg = function_train_gradient(fta);

    struct FT1DArray * fth = ft1d_array_jacobian(ftg);
        
    ft1d_array_free(ftg); ftg = NULL;
    return fth;
}

/********************************************************//**
    Compute the diagpnal of the hessian a function train

    \param[in] ft - Function train

    \return gradient
***********************************************************/
struct FT1DArray * function_train_hessian_diag(const struct FunctionTrain * ft)
{
    struct FT1DArray * ftg = ft1d_array_alloc(ft->dim);
    size_t ii;
    for (ii = 0; ii < ft->dim; ii++){
        ftg->ft[ii] = function_train_copy(ft);
        qmarray_free(ftg->ft[ii]->cores[ii]);
        ftg->ft[ii]->cores[ii] = NULL;
        ftg->ft[ii]->cores[ii] = qmarray_dderiv(ft->cores[ii]);
    }

    return ftg;
}

/********************************************************//**
    Compute the diagpnal of the hessian a function train
    enforce periodic boundary conditions

    \param[in] ft - Function train

    \return gradient
***********************************************************/
struct FT1DArray * function_train_hessian_diag_periodic(const struct FunctionTrain * ft)
{
    struct FT1DArray * ftg = ft1d_array_alloc(ft->dim);
    size_t ii;
    for (ii = 0; ii < ft->dim; ii++){
        ftg->ft[ii] = function_train_copy(ft);
        qmarray_free(ftg->ft[ii]->cores[ii]);
        ftg->ft[ii]->cores[ii] = NULL;
        ftg->ft[ii]->cores[ii] = qmarray_dderiv_periodic(ft->cores[ii]);
    }

    return ftg;
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
double * ft1d_array_eval(const struct FT1DArray * fta, const double * x)
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
void ft1d_array_eval2(const struct FT1DArray * fta, const double * x, double * out)
{
    size_t ii;
    for (ii = 0; ii < fta->size; ii++){
        out[ii] = function_train_eval(fta->ft[ii], x);
    }
}


/********************************************************
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

    \note
    this needs the approximation options because it does rounding
    this is a bit unfortunate -- commenting it out for now
***********************************************************/
/* struct FunctionTrain * */
/* ft1d_array_sum_prod(size_t N, double * coeff, */
/*                     const struct FT1DArray * f, const struct FT1DArray * g, */
/*                     double epsilon) */
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


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/////////////////////                          ///////////////////////////
/////////////////////    BLAS TYPE INTERFACE   ///////////////////////////
/////////////////////                          ///////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


/***********************************************************//**
    Computes
    \f[
        y \leftarrow \texttt{round}(a x + y, epsilon)
    \f]

    \param[in]     a       - scaling factor
    \param[in]     x       - first function train
    \param[in,out] y       - second function train
    \param[in]     epsilon - rounding accuracy (0 for exact)

    \note
    putting NULL it to round will cause an error
***************************************************************/
void c3axpy(double a, struct FunctionTrain * x, struct FunctionTrain ** y,
            double epsilon)
{
    
    struct FunctionTrain * temp = function_train_copy(x);
    function_train_scale(temp,a);
    struct FunctionTrain * z = function_train_sum(temp,*y);
    function_train_free(*y); *y = NULL;
    if (epsilon > 0){
        *y = function_train_round(z,epsilon,NULL);
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

    \param[in] x - first function train
    \param[in] y - second function train

    \return out - inner product between two function trains
***************************************************************/
double c3dot(struct FunctionTrain * x, struct FunctionTrain * y)
{
    double out = function_train_inner(x,y);
    return out;
}



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
    Also rounds after every summation. Will cause an error because putting a NULL into it
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
        struct FunctionTrain * run = function_train_round(runinit,epsilon,NULL);
        for (ii = 1; ii < n; ii++){
            struct FunctionTrain * temp =
                function_train_product(A->ft[ii*inca],x);
            struct FunctionTrain * tempround = function_train_round(temp,epsilon,NULL);
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
        struct FunctionTrain * run = function_train_round(runinit,epsilon,NULL);
        for (ii = 1; ii < n; ii++){
            struct FunctionTrain * tempinit =
                function_train_product(x->ft[ii*incx],y->ft[ii*incy]);
            struct FunctionTrain * temp = function_train_round(tempinit,epsilon,NULL);
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
