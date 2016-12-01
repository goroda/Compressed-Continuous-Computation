// Copyright (c) 2016, Sandia Corporation. Under the terms of Contract
// DE-AC04-94AL85000, there is a non-exclusive license for use of this
// work by or on behalf of the U.S. Government. Export of this program
// may require a license from the United States Government


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
#include <string.h>
#include <assert.h>
#include <math.h>

/* #include "array.h" */
/* #include "ft.h" */
#include "lib_linalg.h"
#include "regress.h"

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
 * \var RegressALS::reset
 *  internal parameter for specfying when I need to reset some parameters
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

    // Even more internal
    int reset;
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


    // extra stuff
    als->reset = 1;
    
    return als;
}

/***********************************************************//**
    Free ALS regression opntions
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
    Get the current function train from the regression struct.
***************************************************************/
struct FunctionTrain * regress_als_get_ft(const struct RegressALS * als)
{
    assert (als != NULL);
    return als->ft;
}

/***********************************************************//**
    Get the current ftparam stored in the struct
***************************************************************/
double * regress_als_get_ftparam(const struct RegressALS * als)
{
    assert (als != NULL);
    assert (als->ft_param != NULL);
    return als->ft_param;
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
    Get number of parameters
***************************************************************/
size_t regress_als_get_num_params(const struct RegressALS * als)
{
    assert (als != NULL);
    assert (als->nparams != NULL);
    size_t totnum = 0;
    for (size_t ii = 0; ii < als->dim; ii++){
        totnum += als->nparams[ii];
    }
    return totnum;
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
        als->grad_core_space = reg_mem_space_alloc(als->N,max_num_param_within_core * maxrank * maxrank);

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
    Do things to reset memory and algorithms
************************************************************/
void regress_als_reset(struct RegressALS * als)
{
    assert (als != NULL);
    als->reset = 1;
}

/********************************************************//**
    LS regression objective function
************************************************************/
double regress_core_LS(size_t nparam, const double * param, double * grad, void * arg)
{

    struct RegressALS * als = arg;
    /* size_t d      = als->dim; */
    size_t core   = als->core;
    assert( nparam == als->nparams[core]);

    function_train_core_update_params(als->ft,core,nparam,param);

    if (grad != NULL){
        for (size_t ii = 0; ii < nparam; ii++){
            grad[ii] = 0.0;
        }
    }
    double out=0.0, resid;
    if (grad != NULL){
        /* printf("evaluate\n"); */
        if (als->reset == 1){
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
            // turn off the reset since this is in the middle of optimization
            als->reset = 0;
        }
        else{
            function_train_core_param_grad_eval_single(
                als->ft,als->N,als->x,als->core,
                als->prev_eval->vals, reg_mem_space_get_data_inc(als->prev_eval),
                als->curr_eval->vals, reg_mem_space_get_data_inc(als->curr_eval),
                als->post_eval->vals, reg_mem_space_get_data_inc(als->post_eval),
                als->grad_core_space->vals, reg_mem_space_get_data_inc(als->grad_core_space),
                als->fparam_space->vals,
                als->evals->vals,als->grad_space->vals,nparam);                
        }
            
        /* printf("done with evaluations\n"); */
        for (size_t ii = 0; ii < als->N; ii++){
            resid = als->y[ii]-als->evals->vals[ii];
            out += 0.5*resid*resid;
            cblas_daxpy(nparam,-resid,als->grad_space->vals + ii * nparam,1,grad,1);
        }
    }
    else{
        if (als->reset == 1){
            for (size_t ii = 0; ii < als->N; ii++){
                /* dprint(d,als->x+ii*d); */
                als->evals->vals[ii] = function_train_eval(als->ft, als->x+ii*als->dim);
                resid = als->y[ii]-als->evals->vals[ii];
                out += 0.5*resid*resid;
            }
        }
        else{
            function_train_core_param_grad_eval_single(
                als->ft,als->N,als->x,als->core,
                als->prev_eval->vals, reg_mem_space_get_data_inc(als->prev_eval),
                als->curr_eval->vals, reg_mem_space_get_data_inc(als->curr_eval),
                als->post_eval->vals, reg_mem_space_get_data_inc(als->post_eval),
                NULL,0,NULL,als->evals->vals,NULL,0);
            for (size_t ii = 0; ii < als->N; ii++){
                /* dprint(d,als->x+ii*d); */
                resid = als->y[ii]-als->evals->vals[ii];
                out += 0.5*resid*resid;
            }
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

    // reset core before running it
    regress_als_reset(als);
    
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



///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
//////////////////                       //////////////////////
////////////////// All in one regression //////////////////////
//////////////////                       //////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

/** \struct RegressAIO
 *  \brief Regress all in one
 *  \var RegressAIO::dim
 *  dimension of approximation
 *  \var RegressAIO::nparams
 *  number of params in each core
 *  \var RegressAIO::totparams
 *  total number of parameters
 *  \var RegressAIO::N
 *  number of data points
 *  \var RegressAIO::y
 *  evaluations
 *  \var RegressAIO::x
 *  input locations
 * \var RegressAIO:grad_space
 *  spare space for gradient evaluations
 * \var RegressAIO::evals
 *  space for evaluation of cost function
 * \var RegressAIO::ft
 *  function_train
 * \var RegressAIO::ft_param
 *  flattened array of function train parameters
 * \var RegressAIO::reset
 *  internal parameter for specfying when I need to reset some parameters
 */
struct RegressAIO
{
    size_t dim;
    size_t * nparams;
    size_t totparams;

    size_t N; 
    const double * y;
    const double * x;

    struct RunningCoreTotal * running_evals_lr;
    struct RunningCoreTotal * running_evals_rl;
    struct RunningCoreTotal ** running_grad;
    struct RegMemSpace * evals;
    struct RegMemSpace * grad;
    struct RegMemSpace * grad_space;
    struct RegMemSpace * fparam_space;
    
    struct FunctionTrain * ft;
    double * ft_param;

    // Even more internal
    int reset;
};


/***********************************************************//**
    Allocate AIO regression
***************************************************************/
struct RegressAIO * regress_aio_alloc(size_t dim)
{
    struct RegressAIO * aio = malloc(sizeof(struct RegressAIO));
    if (aio == NULL){
        fprintf(stderr, "Failure to allocate RegressAIO\n");
        return NULL;
    }
    
    aio->dim = dim;
    aio->nparams = calloc_size_t(dim);

    aio->N = 0;
    aio->y = NULL;
    aio->x = NULL;

    aio->running_evals_lr = NULL;
    aio->running_evals_rl = NULL;
    aio->running_grad  = NULL;

    aio->evals      = NULL;
    aio->grad          = NULL;
    aio->grad_space    = NULL;
    aio->fparam_space  = NULL;

    aio->ft = NULL;
    aio->ft_param = NULL;

    return aio;
}

/***********************************************************//**
    Free AIO regression opntions
***************************************************************/
void regress_aio_free(struct RegressAIO * aio)
{
    if (aio != NULL){

        free(aio->nparams);  aio->nparams    = NULL;

        reg_mem_space_free(aio->grad_space);      aio->grad_space      = NULL;
        reg_mem_space_free(aio->fparam_space);    aio->fparam_space    = NULL;
        reg_mem_space_free(aio->evals); aio->evals = NULL;
        reg_mem_space_free(aio->grad);  aio->grad = NULL;

        running_core_total_free(aio->running_evals_lr);
        running_core_total_free(aio->running_evals_rl);
        running_core_total_arr_free(aio->dim,aio->running_grad);
        
        function_train_free(aio->ft); aio->ft = NULL;
        free(aio->ft_param);          aio->ft_param = NULL;
        free(aio); aio = NULL;
    }
}

/***********************************************************//**
    Get the current function train from the regression struct.
***************************************************************/
struct FunctionTrain * regress_aio_get_ft(const struct RegressAIO * aio)
{
    assert (aio != NULL);
    return aio->ft;
}

/***********************************************************//**
    Get the current ftparam stored in the struct
***************************************************************/
double * regress_aio_get_ftparam(const struct RegressAIO * aio)
{
    assert (aio != NULL);
    assert (aio->ft_param != NULL);
    return aio->ft_param;
}

/***********************************************************//**
    Add data to AIO
***************************************************************/
void regress_aio_add_data(struct RegressAIO * aio, size_t N, const double * x, const double * y)
{
    assert (aio != NULL);
    aio->N = N;
    aio->x = x;
    aio->y = y;
}

/***********************************************************//**
    Get number of parameters
***************************************************************/
size_t regress_aio_get_num_params(const struct RegressAIO * aio)
{
    assert (aio != NULL);
    assert (aio->nparams != NULL);
    size_t totnum = 0;
    for (size_t ii = 0; ii < aio->dim; ii++){
        totnum += aio->nparams[ii];
    }
    return totnum;
}

/***********************************************************//**
    Prepare memmory
***************************************************************/
void regress_aio_prep_memory(struct RegressAIO * aio, struct FunctionTrain * ft, int how)
{
    assert (aio != NULL);
    assert (ft != NULL);
    if (aio->dim != ft->dim){
        fprintf(stderr, "AIO Regression dimension is not the same as FT dimension\n");
        assert (aio->dim == ft->dim);
    }
    
    size_t maxrank = function_train_get_maxrank(ft);
    size_t max_param_within_func = 0;
    size_t num_param_within_func = 0;
    size_t max_num_param_within_core = 0;
    size_t num_tot_params = 0;
    for (size_t ii = 0; ii < ft->dim; ii++){
        num_param_within_func = 0;
        aio->nparams[ii] = function_train_core_get_nparams(ft,ii,&num_param_within_func);
        if (num_param_within_func > max_param_within_func){
            max_param_within_func = num_param_within_func;
        }
        if (aio->nparams[ii] > max_num_param_within_core){
            max_num_param_within_core = aio->nparams[ii];
        }
        num_tot_params += aio->nparams[ii];
    }

    if (how == 0){
        aio->grad_space = reg_mem_space_alloc(1,max_num_param_within_core * maxrank * maxrank);
        aio->grad       = reg_mem_space_alloc(1,num_tot_params);
        aio->evals = reg_mem_space_alloc(1,1);
    }
    else if (how == 1){
        if (aio->N == 0){
            fprintf(stderr, "Must add data before prepping memory with option 1\n");
            exit(1);
        }
        aio->grad_space = reg_mem_space_alloc(aio->N,
                                              max_num_param_within_core * maxrank * maxrank);
        
        aio->grad       = reg_mem_space_alloc(aio->N,num_tot_params);
        aio->evals      = reg_mem_space_alloc(aio->N,1);
    }
    else{
        fprintf(stderr, "Memory prepping with option %d is undefined\n",how);
        exit(1);
    }
    aio->fparam_space = reg_mem_space_alloc(1,max_param_within_func);

    aio->running_evals_lr = ftutil_running_tot_space(ft);
    aio->running_evals_rl = ftutil_running_tot_space(ft);
    aio->running_grad  = ftutil_running_tot_space_eachdim(ft);
    
    aio->ft       = function_train_copy(ft);
    aio->ft_param = calloc_double(num_tot_params);

    size_t running = 0, incr;
    for (size_t ii = 0; ii < ft->dim; ii++){
        incr = function_train_core_get_params(ft,ii,aio->ft_param + running);
        running += incr;
    }
}


/********************************************************//**
    LS regression objective function
************************************************************/
double regress_aio_LS(size_t nparam, const double * param, double * grad, void * arg)
{

    struct RegressAIO * aio = arg;

    size_t running = 0;
    for (size_t ii = 0; ii < aio->dim; ii++){
        function_train_core_update_params(aio->ft,ii,aio->nparams[ii],param + running);
        running += aio->nparams[ii];
    }
    assert (running == nparam);

    running_core_total_restart(aio->running_evals_lr);
    running_core_total_restart(aio->running_evals_rl);
    running_core_total_arr_restart(aio->dim, aio->running_grad);
    
    double out=0.0, resid;
    if (grad != NULL){
        function_train_param_grad_eval(
            aio->ft, aio->N,
            aio->x, aio->running_evals_lr, aio->running_evals_rl, aio->running_grad,
            aio->nparams, aio->evals->vals, aio->grad->vals,
            aio->grad_space->vals, reg_mem_space_get_data_inc(aio->grad_space),
            aio->fparam_space->vals
            );


        for (size_t ii = 0; ii < nparam; ii++){
            grad[ii] = 0.0;
        }
        for (size_t ii = 0; ii < aio->N; ii++){
            /* printf("grad = "); dprint(nparam,aio->grad->vals + ii * nparam); */
            resid = aio->y[ii] - aio->evals->vals[ii];
            /* printf("resid = %G\n",resid); */
            out += 0.5 * resid * resid;
            cblas_daxpy(nparam,-resid,aio->grad->vals + ii * nparam,1,grad,1);
        }        
    }
    else{
        function_train_param_grad_eval(
            aio->ft, aio->N,
            aio->x, aio->running_evals_lr,NULL,NULL,
            aio->nparams, aio->evals->vals, NULL,NULL,0,NULL);
        for (size_t ii = 0; ii < aio->N; ii++){
            resid = aio->y[ii] - aio->evals->vals[ii];
            out += 0.5 * resid * resid;
        }
    }

    return out;
}


////////////////////////////////////
// Common Interface
////////////////////////////////////

struct FTRegress
{
    enum REGTYPE type;
    size_t dim;
    size_t ntotparam;
    union {
        struct RegressALS * als;
        struct RegressAIO * aio;
    } reg;

    struct c3Opt * optimizer;
    size_t * start_ranks;
};

/***********************************************************//**
    Allocate function train regression structure
***************************************************************/
struct FTRegress * ft_regress_alloc(size_t dim)
{
    struct FTRegress * ftr = malloc(sizeof(struct FTRegress));
    if (ftr == NULL){
        fprintf(stderr, "Cannot allocate FTRegress structure\n");
        exit(1);
    }
    ftr->type = REGNONE;
    ftr->dim = dim;
    ftr->ntotparam = 0;
    ftr->optimizer = NULL;
    ftr->start_ranks = NULL;
    return ftr;
}
    
/***********************************************************//**
    Free FT regress structure
***************************************************************/
void ft_regress_free(struct FTRegress * ftr)
{
    if (ftr != NULL){
        if (ftr->type == ALS){
            regress_als_free(ftr->reg.als); ftr->reg.als = NULL;
        }
        else if (ftr->type == AIO){
            regress_aio_free(ftr->reg.aio); ftr->reg.aio = NULL;
        }

        c3opt_free(ftr->optimizer); ftr->optimizer = NULL;
        free(ftr->start_ranks); ftr->start_ranks = NULL;
        
        free(ftr); ftr = NULL;
    }
}

/***********************************************************//**
    Set regression type
***************************************************************/
void ft_regress_set_type(struct FTRegress * ftr, enum REGTYPE type)
{
    assert (ftr != NULL);
    if (ftr->type != REGNONE){
        fprintf(stdout,"Warning: respecfiying type of regression may lead to memory problems");
        fprintf(stdout,"Use function ft_regress_reset_type instead\n");
    }
    ftr->type = type;
    if (type == ALS){
        ftr->reg.als = regress_als_alloc(ftr->dim);
    }
    else if (type == AIO){
        ftr->reg.aio = regress_aio_alloc(ftr->dim);
    }
    else{
        fprintf(stderr,"Error: Regularization type %d not available. Options include\n",type);
        fprintf(stderr,"       ALS = 0\n");
        fprintf(stderr,"       AIO = 1\n");
    }
}


/***********************************************************//**
    Set regression type
***************************************************************/
void ft_regress_set_start_ranks(struct FTRegress * ftr, const size_t * ranks)
{
    assert (ftr != NULL);
    ftr->start_ranks = calloc_size_t(ftr->dim+1);
    memmove(ftr->start_ranks,ranks,(ftr->dim+1)*sizeof(size_t));
}

/***********************************************************//**
    Add data
***************************************************************/
void ft_regress_set_data(struct FTRegress * ftr, size_t N,
                         const double * x, size_t incx,
                         const double * y, size_t incy)
{

    assert (ftr != NULL);
    assert (incx == 1);
    assert (incy == 1);

    enum REGTYPE type = ftr->type;
    if (type == ALS){
        regress_als_add_data(ftr->reg.als,N,x,y);
    }
    else if (type == AIO){
        regress_aio_add_data(ftr->reg.aio,N,x,y);
    }
    else{
        fprintf(stderr,"Must set regression type before setting data\n");
    }

}

/***********************************************************//**
    Set regression type
***************************************************************/
void ft_regress_prep_memory(struct FTRegress * ftr, struct FunctionTrain * ft, int how)
{

    assert (ftr != NULL);
    enum REGTYPE type = ftr->type;
    if (type == ALS){
        regress_als_prep_memory(ftr->reg.als,ft,how);
        ftr->ntotparam = regress_als_get_num_params(ftr->reg.als);
    }
    else if (type == AIO){
        regress_aio_prep_memory(ftr->reg.aio,ft,how);
        ftr->ntotparam = regress_aio_get_num_params(ftr->reg.aio);
    }
    else{
        fprintf(stderr,"Must set regression type before calling prep_memory\n");
        exit(1);
    }

    /* printf("Number of total parameters = %zu\n",ftr->ntotparam); */
    ftr->optimizer = c3opt_alloc(BFGS,ftr->ntotparam);
    c3opt_set_maxiter(ftr->optimizer,10000);
}

/***********************************************************//**
    Run the regression
***************************************************************/
struct FunctionTrain * ft_regress_run(struct FTRegress * ftr, enum REGOBJ obj)
{
    assert (ftr != NULL);
    assert (ftr->ntotparam > 0);
    assert (ftr->optimizer != NULL);
    enum REGTYPE type = ftr->type;

    /* double * guess = NULL; */
    if (obj == FTLS){
        if (type == ALS){
            fprintf(stderr, "Cannot yet run ALS, not complete!\n");
            exit(1);
            /* c3opt_add_objective(ftr->optimizer,regress_als_LS,ftr->reg.als); */
            /* guess = regress_als_get_ftparam(ftr->reg.als); */
        }
        else if (type == AIO){
            c3opt_add_objective(ftr->optimizer,regress_aio_LS,ftr->reg.aio);
            /* guess = regress_aio_get_ftparam(ftr->reg.aio); */
        }
    }
    else{
        fprintf(stderr,"Must call prep_memory before regress_run \n");
        exit(1);
    }

    /* printf("initial guess\n"); */
    /* dprint(ftr->ntotparam,guess); */
    /* exit(1); */
    double * guess = calloc_double(ftr->ntotparam);
    for (size_t ii = 0; ii < ftr->ntotparam; ii++){
        guess[ii] = randn();
    }
    assert (guess != NULL);
    double val;
    c3opt_set_verbose(ftr->optimizer,0);
    int res = c3opt_minimize(ftr->optimizer,guess,&val);
    if (res < 0){
        fprintf(stderr,"Warning: optimizer exited with code %d\n",res);
    }

    struct FunctionTrain * ft_final = NULL;
    if (type == ALS){
        ft_final = regress_als_get_ft(ftr->reg.als);
    }
    else if (type == AIO){
        ft_final = regress_aio_get_ft(ftr->reg.aio);
    }
    free(guess); guess = NULL;
    
    return ft_final;
}
