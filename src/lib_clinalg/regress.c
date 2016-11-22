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

/** \file regress.c
 * Provides routines for FT-regression
 */

#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "array.h"
#include "ft.h"
/* #include "regress.h" */
#include "lib_linalg.h"

struct RegMemSpace
{

    size_t ndata;
    size_t one_data_size;
    double * vals;
};

struct RegMemSpace * reg_mem_space_alloc(size_t ndata, size_t one_data_size)
{
    struct RegMemSpace * mem = malloc(sizeof(struct RegMemSpace));
    if (mem == NULL){
        fprintf(stderr, "Failure to allocate memmory of regression\n");
        return NULL;
    }

    mem->ndata = ndata;
    mem->one_data_size = one_data_size;
    mem->vals = calloc_double(ndata * one_data_size);
    return mem;
}

void reg_mem_space_free(struct RegMemSpace * rmem)
{
    if (rmem != NULL){
        free(rmem->vals); rmem->vals = NULL;
        free(rmem); rmem = NULL;
    }
}

size_t reg_mem_space_get_data_inc(struct RegMemSpace * rmem)
{
    assert (rmem != NULL);
    return rmem->one_data_size;
}

/** \struct RegressALS
 *  \brief Alternating least squares regression
 *  \var RegressALS::dim
 *  dimension of approximation
 *  \var RegressALS::nparams
 *  number of params in each core
 *  \var RegressALS::N
 *  number of data points
 *  \var RegressALS::y
 *  evaluations
 *  \var RegressALS::x
 *  input locations
 *  \var RegressALS::core
 *  core over which to optimize
 *  \var RegressALS::pre_eval
 *  evaluations prior to dimension *core*
 * \var RegressALS::post_eval
 *  evaluations post dimension *core*
 * \var RegressALS:grad_space
 *  spare space for gradient evaluations
 * \var RegressALS::grad_core_space
 *  space for evaluation of the gradient of the core with respect to every param
 * \var RegressALS::fparam_space
 *  space for evaluation of gradient of params for a given function in a core
 * \var RegressALS::evals
 *  space for evaluation of cost function
 * \var RegressALS::ft
 *  function_train
 * \var RegressALS::ft_param
 *  flattened array of function train parameters
 */
struct RegressALS
{
    size_t dim;
    size_t * nparams;
    
    size_t N; 
    const double * y;
    const double * x;
    
    size_t core;
    struct RegMemSpace * prev_eval;
    struct RegMemSpace * post_eval;
    struct RegMemSpace * curr_eval;
    
    struct RegMemSpace * grad_space;
    struct RegMemSpace * grad_core_space;
    struct RegMemSpace * fparam_space;

    struct RegMemSpace * evals;

    struct FunctionTrain * ft;
    double * ft_param;
};

/***********************************************************//**
    Allocate ALS regression
***************************************************************/
struct RegressALS * regress_als_alloc(size_t dim)
{
    struct RegressALS * als = malloc(sizeof(struct RegressALS));
    if (als == NULL){
        fprintf(stderr, "Failure to allocate RegressALS\n");
        return NULL;
    }
    
    als->dim = dim;
    als->nparams = calloc_size_t(dim);

    als->N = 0;
    als->y = NULL;
    als->x = NULL;

    als->prev_eval = NULL;
    als->post_eval = NULL;
    als->curr_eval = NULL;

    als->grad_space      = NULL;
    als->grad_core_space = NULL;
    als->fparam_space    = NULL;

    als->evals = NULL;
    
    
    als->ft = NULL;
    als->ft_param = NULL;
    
    return als;
}

/***********************************************************//**
    Free ALS regression options
***************************************************************/
void regress_als_free(struct RegressALS * als)
{
    if (als != NULL){

        free(als->nparams);  als->nparams    = NULL;

        
        reg_mem_space_free(als->prev_eval); als->prev_eval = NULL;
        reg_mem_space_free(als->post_eval); als->post_eval = NULL;
        reg_mem_space_free(als->curr_eval); als->curr_eval = NULL;

        reg_mem_space_free(als->grad_space);      als->grad_space      = NULL;
        reg_mem_space_free(als->grad_core_space); als->grad_core_space = NULL;
        reg_mem_space_free(als->fparam_space);    als->fparam_space    = NULL;

        reg_mem_space_free(als->evals); als->evals = NULL;

        function_train_free(als->ft); als->ft = NULL;
        free(als->ft_param);          als->ft_param = NULL;
        free(als); als = NULL;
    }
}

/***********************************************************//**
    Add data to ALS
***************************************************************/
void regress_als_add_data(struct RegressALS * als, size_t N, const double * x, const double * y)
{
    assert (als != NULL);
    als->N = N;
    als->x = x;
    als->y = y;
}

/***********************************************************//**
    Prepare memmory
***************************************************************/
void regress_als_prep_memory(struct RegressALS * als, struct FunctionTrain * ft, int how)
{
    assert (als != NULL);
    assert (ft != NULL);
    if (als->dim != ft->dim){
        fprintf(stderr, "ALS Regression dimension is not the same as FT dimension\n");
        assert (als->dim == ft->dim);
    }
    
    size_t maxrank = function_train_get_maxrank(ft);

    size_t max_param_within_func = 0;
    size_t num_param_within_func = 0;
    size_t max_num_param_within_core = 0;
    size_t num_totparams = 0;
    for (size_t ii = 0; ii < ft->dim; ii++){
        num_param_within_func = 0;
        als->nparams[ii] = function_train_core_get_nparams(ft,ii,&num_param_within_func);
        if (num_param_within_func > max_param_within_func){
            max_param_within_func = num_param_within_func;
        }
        if (als->nparams[ii] > max_num_param_within_core){
            max_num_param_within_core = als->nparams[ii];
        }
        num_totparams += als->nparams[ii];
    }

    if (how == 0){
        als->prev_eval = reg_mem_space_alloc(1,maxrank);
        als->post_eval = reg_mem_space_alloc(1,maxrank);
        als->curr_eval = reg_mem_space_alloc(1,maxrank * maxrank);
        
        als->grad_space      = reg_mem_space_alloc(1,max_num_param_within_core);
        als->grad_core_space = reg_mem_space_alloc(1,max_num_param_within_core * maxrank * maxrank);

        als->evals = reg_mem_space_alloc(1,1);
    }
    else if (how == 1){
        if (als->N == 0){
            fprintf(stderr, "Must add data before prepping memory with option 1\n");
            exit(1);
        }
        als->prev_eval = reg_mem_space_alloc(als->N,maxrank);
        als->post_eval = reg_mem_space_alloc(als->N,maxrank);
        als->curr_eval = reg_mem_space_alloc(als->N,maxrank * maxrank);

        als->grad_space      = reg_mem_space_alloc(als->N,max_num_param_within_core);
        als->grad_core_space = reg_mem_space_alloc(als->N, max_num_param_within_core * maxrank * maxrank);

        als->evals = reg_mem_space_alloc(als->N,1);
    }
    else{
        fprintf(stderr, "Memory prepping with option %d is undefined\n",how);
        exit(1);
    }
    als->fparam_space = reg_mem_space_alloc(1,max_param_within_func);
    
    als->ft       = function_train_copy(ft);
    als->ft_param = calloc_double(num_totparams);

    size_t running = 0, incr;
    for (size_t ii = 0; ii < ft->dim; ii++){
        incr = function_train_core_get_params(ft,ii,als->ft_param + running);
        running += incr;
    }
}

/********************************************************//**
    Set which core we are regressing on
************************************************************/
void regress_als_set_core(struct RegressALS * als, size_t core)
{
    assert (als != NULL);
    assert (core < als->dim);
    als->core = core;
}

/********************************************************//**
    LS regression objective function
************************************************************/
double regress_core_LS(size_t nparam, const double * param, double * grad, void * arg)
{

    struct RegressALS * als = arg;
    size_t d      = als->dim;
    size_t core   = als->core;
    assert( nparam == als->nparams[core]);

    /* printf("N=%zu, d= %zu\n",als->N,d); */
    /* printf("data = \n"); */
    /* dprint2d_col(d,als->N,als->x); */
    /* printf("update core params\n"); */
    function_train_core_update_params(als->ft,core,nparam,param);
    /* printf("updated core params\n"); */

    if (grad != NULL){
        for (size_t ii = 0; ii < nparam; ii++){
            grad[ii] = 0.0;
        }
    }
    double out=0.0, resid;
    if (grad != NULL){
        function_train_core_param_grad_eval(
            als->ft,
            als->N, als->x, als->core, nparam,
            als->grad_core_space->vals, reg_mem_space_get_data_inc(als->grad_core_space),
            als->fparam_space->vals,
            als->grad_space->vals,
            als->prev_eval->vals, reg_mem_space_get_data_inc(als->prev_eval),
            als->curr_eval->vals, reg_mem_space_get_data_inc(als->curr_eval),
            als->post_eval->vals, reg_mem_space_get_data_inc(als->post_eval),
            als->evals->vals);
        
        for (size_t ii = 0; ii < als->N; ii++){
            /* eval = function_train_core_param_grad_eval(als->ft, als->x+ii*d, */
            /*                                            core, nparam, */
            /*                                            als->grad_core_space, */
            /*                                            als->fparam_space, */
            /*                                            als->grad_space, */
            /*                                            als->prev_eval, */
            /*                                            als->curr_eval, */
            /*                                            als->post_eval); */

            resid = als->y[ii]-als->evals->vals[ii];
            out += 0.5*resid*resid;
            cblas_daxpy(nparam,-resid,als->grad_space->vals + ii * nparam,1,grad,1);
        }
    }
    else{
        for (size_t ii = 0; ii < als->N; ii++){
            /* dprint(d,als->x+ii*d); */
            als->evals->vals[ii] = function_train_eval(als->ft, als->x+ii*d);
            resid = als->y[ii]-als->evals->vals[ii];
            out += 0.5*resid*resid;
        }
    }

    return out;
}


/********************************************************//**
    Optimize over a particular core
************************************************************/
int regress_als_run_core(struct RegressALS * als, struct c3Opt * optimizer, double *val)
{
    assert (als != NULL);
    assert (als->ft != NULL);

    size_t core = als->core;
    size_t nparambefore = 0;
    for (size_t ii = 0; ii < core; ii++){
        nparambefore += als->nparams[ii];
    }
    int info = c3opt_minimize(optimizer,als->ft_param + nparambefore,val);
    if (info < 0){
        fprintf(stderr, "Minimizing core %zu resulted in nonconvergence with output %d\n",core,info);
    }
    
    function_train_core_update_params(als->ft,core,als->nparams[core],als->ft_param + nparambefore);
    return info;
}


/********************************************************//**
    Advance to the right
************************************************************/
void regress_als_step_right(struct RegressALS * als)
{

    assert (als != NULL);
    assert (als->ft != NULL);
    if (als->core == als->dim-1){
        fprintf(stderr, "Cannot step right in ALS regression, already on last dimension!!\n");
        exit(1);
    }
    als->core++;
}

/********************************************************//**
    Left-right sweep
************************************************************/
double regress_als_sweep_lr(struct RegressALS * als, struct c3Opt ** optimizers, int skip_first)
{
    
    assert (als != NULL);
    assert (als->ft != NULL);
    size_t start = 0;
    if (skip_first == 0){
        als->core = 0;
    }
    else{
        start = 1;
        als->core = 1;
    }
    /* int res; */
    double val;
    for (size_t ii = start; ii < als->dim; ii++){
        regress_als_run_core(als,optimizers[als->core],&val);
        printf("Core:%zu, Objective=%G\n",als->core,val);
        als->core++;
    }
    return val;
}

/********************************************************//**
    Left-right sweep
************************************************************/
double regress_als_sweep_rl(struct RegressALS * als, struct c3Opt ** optimizers,int skip_first)
{
    
    assert (als != NULL);
    assert (als->ft != NULL);

    size_t start = 0;
    if (skip_first == 0){
        als->core = als->dim-1;
    }
    else{
        start = 1;
        als->core = als->dim-2;
    }
    /* int res; */
    double val;
    for (size_t ii = start; ii < als->dim; ii++){
        regress_als_run_core(als,optimizers[als->core],&val);
        printf("\t Core:%zu, Objective=%G\n",als->core,val);
        als->core--;
    }
    return val;
    
}



