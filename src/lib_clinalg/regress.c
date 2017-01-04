// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016, Sandia Corporation

// This file is part of the Compressed Continuous Computation (C3) Library
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

struct RegMemSpace ** reg_mem_space_arr_alloc(size_t dim, size_t ndata, size_t one_data_size)
{
    struct RegMemSpace ** mem = malloc(dim * sizeof(struct RegMemSpace * ));
    if (mem == NULL){
        fprintf(stderr, "Failure to allocate memmory of regression\n");
        return NULL;
    }
    for (size_t ii = 0; ii < dim; ii++){
        mem[ii] = reg_mem_space_alloc(ndata,one_data_size);
    }

    return mem;
}

void reg_mem_space_free(struct RegMemSpace * rmem)
{
    if (rmem != NULL){
        free(rmem->vals); rmem->vals = NULL;
        free(rmem); rmem = NULL;
    }
}

void reg_mem_space_arr_free(size_t dim, struct RegMemSpace ** rmem)
{
    if (rmem != NULL){
        for (size_t ii = 0; ii < dim; ii++){
            reg_mem_space_free(rmem[ii]); rmem[ii] = NULL;
        }
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
    (void)(nparam);
    (void)(param);
    (void)(grad);
    (void)(arg);
    
    /* struct RegressALS * als = arg; */
    /* /\* size_t d      = als->dim; *\/ */
    /* size_t core = als->core; */
    /* assert(nparam == als->nparams[core]); */

    /* function_train_core_update_params(als->ft,core,nparam,param); */

    /* if (grad != NULL){ */
    /*     for (size_t ii = 0; ii < nparam; ii++){ */
    /*         grad[ii] = 0.0; */
    /*     } */
    /* } */
    /* double out=0.0, resid; */
    /* if (grad != NULL){ */
    /*     /\* printf("evaluate\n"); *\/ */
    /*     if (als->reset == 1){ */
    /*         function_train_core_param_grad_eval( */
    /*             als->ft, */
    /*             als->N, als->x, als->core, nparam, */
    /*             als->grad_core_space->vals, reg_mem_space_get_data_inc(als->grad_core_space), */
    /*             als->fparam_space->vals, */
    /*             als->grad_space->vals, */
    /*             als->prev_eval->vals, reg_mem_space_get_data_inc(als->prev_eval), */
    /*             als->curr_eval->vals, reg_mem_space_get_data_inc(als->curr_eval), */
    /*             als->post_eval->vals, reg_mem_space_get_data_inc(als->post_eval), */
    /*             als->evals->vals); */
    /*         // turn off the reset since this is in the middle of optimization */
    /*         als->reset = 0; */
    /*     } */
    /*     else{ */
    /*         function_train_core_param_grad_eval_single( */
    /*             als->ft,als->N,als->x,als->core, */
    /*             als->prev_eval->vals, reg_mem_space_get_data_inc(als->prev_eval), */
    /*             als->curr_eval->vals, reg_mem_space_get_data_inc(als->curr_eval), */
    /*             als->post_eval->vals, reg_mem_space_get_data_inc(als->post_eval), */
    /*             als->grad_core_space->vals, reg_mem_space_get_data_inc(als->grad_core_space), */
    /*             als->fparam_space->vals, */
    /*             als->evals->vals,als->grad_space->vals,nparam);                 */
    /*     } */
            
    /*     /\* printf("done with evaluations\n"); *\/ */
    /*     for (size_t ii = 0; ii < als->N; ii++){ */
    /*         resid = als->y[ii]-als->evals->vals[ii]; */
    /*         out += 0.5*resid*resid; */
    /*         cblas_daxpy(nparam,-resid,als->grad_space->vals + ii * nparam,1,grad,1); */
    /*     } */
    /* } */
    /* else{ */
    /*     if (als->reset == 1){ */
    /*         for (size_t ii = 0; ii < als->N; ii++){ */
    /*             /\* dprint(d,als->x+ii*d); *\/ */
    /*             als->evals->vals[ii] = function_train_eval(als->ft, als->x+ii*als->dim); */
    /*             resid = als->y[ii]-als->evals->vals[ii]; */
    /*             out += 0.5*resid*resid; */
    /*         } */
    /*     } */
    /*     else{ */
    /*         function_train_core_param_grad_eval_single( */
    /*             als->ft,als->N,als->x,als->core, */
    /*             als->prev_eval->vals, reg_mem_space_get_data_inc(als->prev_eval), */
    /*             als->curr_eval->vals, reg_mem_space_get_data_inc(als->curr_eval), */
    /*             als->post_eval->vals, reg_mem_space_get_data_inc(als->post_eval), */
    /*             NULL,0,NULL,als->evals->vals,NULL,0); */
    /*         for (size_t ii = 0; ii < als->N; ii++){ */
    /*             /\* dprint(d,als->x+ii*d); *\/ */
    /*             resid = als->y[ii]-als->evals->vals[ii]; */
    /*             out += 0.5*resid*resid; */
    /*         } */
    /*     } */
    /* } */

    return 0.0;
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Clean Attempt
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Parameterized FT
struct FTparam
{
    struct FunctionTrain * ft;
    size_t dim;

    size_t * nparams_per_uni; // number of parmaters per univariate funciton
    size_t * nparams_per_core;
    size_t nparams;
    double * params;
    
    struct MultiApproxOpts * approx_opts;
};

/***********************************************************//**
    Free FT regress structure
***************************************************************/
void ft_param_free(struct FTparam * ftr)
{
    if (ftr != NULL){
        function_train_free(ftr->ft); ftr->ft = NULL;
        free(ftr->nparams_per_uni); ftr->nparams_per_uni = NULL;
        free(ftr->nparams_per_core); ftr->nparams_per_core = NULL;
        free(ftr->params); ftr->params = NULL;
        free(ftr); ftr = NULL;
    }
}


/***********************************************************//**
    Allocate parameterized function train
***************************************************************/
struct FTparam *
ft_param_alloc(size_t dim,
               struct MultiApproxOpts * aopts,
               double * params, size_t * ranks)
{
    struct FTparam * ftr = malloc(sizeof(struct FTparam));
    if (ftr == NULL){
        fprintf(stderr, "Cannot allocate FTparam structure\n");
        exit(1);
    }
    ftr->approx_opts = aopts;

    ftr->ft = function_train_zeros(aopts,ranks);
    ftr->dim = dim;
    
    ftr->nparams_per_core = calloc_size_t(ftr->dim);
    ftr->nparams = 0;
    size_t nuni = 0; // number of univariate functions
    for (size_t jj = 0; jj < ftr->dim; jj++)
    {
        nuni += ranks[jj]*ranks[jj+1];
        ftr->nparams_per_core[jj] = ranks[jj]*ranks[jj+1] * multi_approx_opts_get_dim_nparams(aopts,jj);
        function_train_core_update_params(ftr->ft,jj,ftr->nparams_per_core[jj],params+ftr->nparams);
        ftr->nparams += ftr->nparams_per_core[jj];
    }
    
    ftr->nparams_per_uni = calloc_size_t(nuni);
    size_t onind = 0;
    for (size_t jj = 0; jj < ftr->dim; jj++){

        for (size_t ii = 0; ii < ranks[jj]*ranks[jj+1]; ii++){
            ftr->nparams_per_uni[onind] = multi_approx_opts_get_dim_nparams(aopts,jj);
            onind++;
        }
    }
    
    ftr->params = calloc_double(ftr->nparams);
    memmove(ftr->params,params,ftr->nparams * sizeof(double) );
    
    return ftr;
}
    
void ft_param_update_params(struct FTparam * ftp, const double * params)
{
    size_t runparam = 0;
    for (size_t ii = 0; ii < ftp->dim; ii ++){
        function_train_core_update_params(ftp->ft,ii,
                                          ftp->nparams_per_core[ii],
                                          params+runparam);
        runparam += ftp->nparams_per_core[ii];
    }

    assert (runparam == ftp->nparams);

    memmove(ftp->params,params,ftp->nparams * sizeof(double) );
}


void ft_param_update_core_params(struct FTparam * ftp, size_t core, const double * params)
{
    size_t runparam = 0;
    for (size_t ii = 0; ii < core; ii ++){
        runparam += ftp->nparams_per_core[ii];
    }
    function_train_core_update_params(ftp->ft,core,
                                      ftp->nparams_per_core[core],
                                      params);

    memmove(ftp->params + runparam,params,ftp->nparams_per_core[core] * sizeof(double) );
}

void ft_param_update_structure(struct FTparam ** ftp,
                               struct MultiApproxOpts * opts,
                               size_t * new_ranks, double * new_vals)
{
    // just overwrites
    size_t dim = (*ftp)->dim;
    ft_param_free(*ftp); *ftp = NULL;
    *ftp = ft_param_alloc(dim,opts,new_vals,new_ranks);
}

size_t * ft_param_get_num_params_per_core(const struct FTparam * ftp)
{
    assert (ftp != NULL);
    return ftp->nparams_per_core;
}

struct FunctionTrain * ft_param_get_ft(const struct FTparam * ftp)
{
    assert(ftp != NULL);
    return ftp->ft;
}
        
struct RegressionMemManager
{

    size_t dim;
    size_t N;
    struct RunningCoreTotal *  running_evals_lr;
    struct RunningCoreTotal *  running_evals_rl;
    struct RunningCoreTotal ** running_grad;
    struct RegMemSpace *       evals;
    struct RegMemSpace *       grad;
    struct RegMemSpace *       grad_space;
    struct RegMemSpace **      all_grads;
    struct RegMemSpace *       fparam_space;

    enum FTPARAM_ST structure;
    int once_eval_structure;

    double ** lin_structure_vals;
    size_t * lin_structure_inc;
};

void regression_mem_manager_free(struct RegressionMemManager * mem)
{
    if (mem != NULL){
    
        reg_mem_space_free(mem->grad_space);
        mem->grad_space = NULL;
        
        reg_mem_space_free(mem->fparam_space);
        mem->fparam_space = NULL;
        
        reg_mem_space_free(mem->evals);
        mem->evals = NULL;
        
        reg_mem_space_free(mem->grad);
        mem->grad = NULL;

        reg_mem_space_arr_free(mem->dim,mem->all_grads);
        mem->all_grads = NULL;

        running_core_total_free(mem->running_evals_lr);
        running_core_total_free(mem->running_evals_rl);
        running_core_total_arr_free(mem->dim,mem->running_grad);

        free(mem->lin_structure_vals); mem->lin_structure_vals = NULL;
        free(mem->lin_structure_inc);  mem->lin_structure_inc  = NULL;
        
        free(mem); mem = NULL;
    }

}


struct RegressionMemManager *
regress_mem_manager_alloc(size_t d, size_t n,
                          size_t * num_params_per_core,
                          size_t * ranks,
                          size_t max_param_within_uni,
                          enum FTPARAM_ST structure)
{

    struct RegressionMemManager * mem = malloc(sizeof(struct RegressionMemManager));
    if (mem == NULL){
        fprintf(stderr, "Cannot allocate struct RegressionMemManager\n");
        exit(1);
    }

    mem->dim = d;
    mem->N = n;

    mem->fparam_space = reg_mem_space_alloc(1,max_param_within_uni);

    mem->all_grads = malloc(d * sizeof(struct RegMemSpace * ));
    assert (mem->all_grads != NULL);
    size_t mr2_max = 0;
    size_t mr2;
    size_t num_tot_params = 0;
    size_t max_param_within_core = 0;
    for (size_t ii = 0; ii < d; ii++){
        mr2 = ranks[ii]*ranks[ii+1];
        mem->all_grads[ii] = reg_mem_space_alloc(n,num_params_per_core[ii] * mr2 );
        num_tot_params += num_params_per_core[ii];
        if (mr2 > mr2_max){
            mr2_max = mr2;
        }
        if (num_params_per_core[ii] > max_param_within_core){
            max_param_within_core = num_params_per_core[ii];
        }
    }
    mem->running_evals_lr = running_core_total_alloc(mr2_max);
    mem->running_evals_rl = running_core_total_alloc(mr2_max);
    mem->running_grad     = running_core_total_arr_alloc(d,mr2_max);
    
    mem->grad       = reg_mem_space_alloc(n,num_tot_params);
    mem->evals      = reg_mem_space_alloc(n,1);


    mem->grad_space = reg_mem_space_alloc(n, max_param_within_core * mr2_max);
    if ( structure != LINEAR_ST){
        if (structure != NONE_ST){
            fprintf(stderr,"Error: Parameter structure type %d is unknown\n",structure);
            exit(1);
        }
    }
    mem->structure = structure;
    mem->once_eval_structure = 0;
    mem->lin_structure_vals = malloc(mem->dim * sizeof(double * ));
    assert (mem->lin_structure_vals != NULL);
    mem->lin_structure_inc = calloc_size_t(mem->dim * sizeof(size_t));
    assert (mem->lin_structure_inc != NULL);
    
    return mem;
    
}

void regress_mem_manager_reset_running(struct RegressionMemManager * mem)
{
    running_core_total_restart(mem->running_evals_lr);
    running_core_total_restart(mem->running_evals_rl);
    running_core_total_arr_restart(mem->dim, mem->running_grad);
}

// might be a mismatch here need to compare N and the actual amount of memory allocated
void regress_mem_manager_check_structure(struct RegressionMemManager * mem, struct FTparam * ftp,
                                         const double * x)
{

    if ((mem->structure == LINEAR_ST) && (mem->once_eval_structure == 0)){
        for (size_t ii = 0; ii < ftp->dim; ii++){
            if (ii == 0){
                qmarray_param_grad_eval(
                    ftp->ft->cores[ii],mem->N,
                    x + ii,ftp->dim,
                    NULL, 0,
                    mem->all_grads[ii]->vals,
                    reg_mem_space_get_data_inc(mem->all_grads[ii]),
                    mem->fparam_space->vals);
            }
            else{
                qmarray_param_grad_eval_sparse_mult(
                    ftp->ft->cores[ii],
                    mem->N,x + ii,
                    ftp->dim,
                    NULL, 0,
                    mem->all_grads[ii]->vals,
                    reg_mem_space_get_data_inc(mem->all_grads[ii]),
                    NULL,NULL,0);
            }

            mem->lin_structure_vals[ii] = mem->all_grads[ii]->vals;
            mem->lin_structure_inc[ii]  = reg_mem_space_get_data_inc(mem->all_grads[ii]);
        }
        /* printf("OKOKOK\n"); */
        mem->once_eval_structure = 1;
    }
}

struct RegressOpts
{

    enum REGTYPE type;
    enum REGOBJ obj;

    size_t nmem_alloc; // number of evaluations there is space for
    struct RegressionMemManager * mem;


    size_t N;
    size_t d;
    const double * x;
    const double * y;

    size_t nbatches; // should be N/nmem_alloc

    struct c3Opt * optimizer;


    int verbose;
    
    // relative function tolerance
    double convtol;
    
    // for ALS
    size_t active_core;
    size_t maxsweeps;
    
};


void regress_opts_free(struct RegressOpts * opts)
{
    if (opts != NULL){
        regression_mem_manager_free(opts->mem); opts->mem = NULL;
        c3opt_free(opts->optimizer); opts->optimizer = NULL;
        free(opts); opts = NULL;
    }
}
    
struct RegressOpts * regress_opts_alloc()
{
    struct RegressOpts * ropts = malloc(sizeof(struct RegressOpts));
    if (ropts == NULL){
        fprintf(stderr,"Error: Failure to allocate\n\t struct RegressOpts\n");
        exit(1);
    }
    ropts->x = NULL;
    ropts->y = NULL;
    ropts->mem = NULL;
    ropts->optimizer = NULL;
    ropts->verbose = 0;
    ropts->convtol = 1e-10;
    ropts->active_core = 0;
    ropts->maxsweeps = 10;
    return ropts;
}

struct RegressOpts *
regress_opts_create(enum REGTYPE type, enum REGOBJ obj, size_t N,
                    size_t d, const double * x, const double * y)
{

    struct RegressOpts * opts = regress_opts_alloc();

    if (type == AIO){
        /* printf("type == AIO\n"); */
        opts->type = AIO;
    }
    else if (type == ALS){
        opts->type = ALS;
        opts->active_core = 0;
    }
    else{
        fprintf(stderr,"Error: REGTYPE %d unavailable\n",type);
        exit(1);
    }

    if (obj == FTLS){
        opts->obj = FTLS;
    }
    else{
        fprintf(stderr,"Error: REGOBJ %d unavailable\n",obj);
        exit(1);
    }

    opts->N = N;
    opts->d = d;
    opts->x = x;
    opts->y = y;
    return opts;
}

void regress_opts_set_als_maxsweep(struct RegressOpts * opts, size_t maxsweeps)
{
    assert (opts != NULL);
    opts->maxsweeps = maxsweeps;
}

void regress_opts_set_convtol(struct RegressOpts * opts, double convtol)
{
    assert (opts != NULL);
    opts->convtol = convtol;
}


void regress_opts_set_verbose(struct RegressOpts * opts, int verbose)
{
    assert (opts != NULL);
    opts->verbose = verbose;
}

void regress_opts_initialize_memory(struct RegressOpts * opts, size_t * num_params_per_core,
                                    size_t * ranks, size_t max_param_within_uni,
                                    enum FTPARAM_ST structure)
{

    opts->nmem_alloc = opts->N;
    opts->nbatches = 1;
    if (opts->mem != NULL){
        regression_mem_manager_free(opts->mem); opts->mem = NULL;
    }
    opts->mem = regress_mem_manager_alloc(opts->d,opts->N,
                                          num_params_per_core,
                                          ranks,
                                          max_param_within_uni,
                                          structure);

    if (opts->type == AIO){
        size_t num_tot_params = 0;
        for (size_t ii = 0; ii < opts->d; ii++){
            num_tot_params += num_params_per_core[ii];
        }
        if (opts->optimizer != NULL){
            c3opt_free(opts->optimizer); opts->optimizer = NULL;
        }
        opts->optimizer = c3opt_alloc(BFGS,num_tot_params);
    }
}

void regress_opts_prepare_als_core(struct RegressOpts * opts, size_t core, struct FTparam * ftp)
{
    opts->active_core = core;
    function_train_core_pre_post_run(ftp->ft,core,opts->N,opts->x,
                                     opts->mem->running_evals_lr,
                                     opts->mem->running_evals_rl);
}

double ft_param_eval_objective_aio_ls(struct FTparam * ftp, struct RegressOpts * regopts,
                                      double * grad)
{
    double out = 0.0;
    double resid;
    if (grad != NULL){
        if (regopts->mem->structure == LINEAR_ST){
            function_train_linparam_grad_eval(
                ftp->ft, regopts->N,
                regopts->x, regopts->mem->running_evals_lr, regopts->mem->running_evals_rl,
                regopts->mem->running_grad,
                ftp->nparams_per_core, regopts->mem->evals->vals, regopts->mem->grad->vals,
                regopts->mem->lin_structure_vals,regopts->mem->lin_structure_inc);
        }
        else{
            function_train_param_grad_eval(
                ftp->ft, regopts->N,
                regopts->x, regopts->mem->running_evals_lr, regopts->mem->running_evals_rl,
                regopts->mem->running_grad,
                ftp->nparams_per_core, regopts->mem->evals->vals, regopts->mem->grad->vals,
                regopts->mem->grad_space->vals,
                reg_mem_space_get_data_inc(regopts->mem->grad_space),
                regopts->mem->fparam_space->vals
                );
        }
        
        for (size_t ii = 0; ii < regopts->N; ii++){
            /* printf("grad = "); dprint(ftp->nparams,regopts->mem->grad->vals + ii * ftp->nparams); */
            resid = regopts->y[ii] - regopts->mem->evals->vals[ii];
            out += 0.5 * resid * resid;
            cblas_daxpy(ftp->nparams,-resid,regopts->mem->grad->vals + ii * ftp->nparams,
                        1,grad,1);
        }        
    }
    else{
        function_train_param_grad_eval(
            ftp->ft, regopts->N,
            regopts->x, regopts->mem->running_evals_lr,NULL,NULL,
            ftp->nparams_per_core, regopts->mem->evals->vals, NULL,NULL,0,NULL);
        for (size_t ii = 0; ii < regopts->N; ii++){
            /* aio->evals->vals[ii] = function_train_eval(aio->ft,aio->x+ii*aio->dim); */
            resid = regopts->y[ii] - regopts->mem->evals->vals[ii];
            out += 0.5 * resid * resid;
        }   
    }
    return out;
}
    
double ft_param_eval_objective_aio(struct FTparam * ftp,
                                   struct RegressOpts * regopts,
                                   double * grad)
{
    double out = 0.0;
    if (regopts->obj == FTLS){
        out = ft_param_eval_objective_aio_ls(ftp,regopts,grad);
    }
    else{
        assert (1 == 0);
    }

    return out;
}

double ft_param_eval_objective_als_ls(struct FTparam * ftp, struct RegressOpts * regopts,
                                      double * grad)
{
    double out = 0.0;
    double resid;
    if (grad != NULL){
        if (regopts->mem->structure == LINEAR_ST){
            assert ( 1 == 0);
            /* function_train_linparam_grad_eval( */
            /*     ftp->ft, regopts->N, */
            /*     regopts->x, regopts->mem->running_evals_lr, regopts->mem->running_evals_rl, */
            /*     regopts->mem->running_grad, */
            /*     ftp->nparams_per_core, regopts->mem->evals->vals, regopts->mem->grad->vals, */
            /*     regopts->mem->lin_structure_vals,regopts->mem->lin_structure_inc); */
        }
        else{
            function_train_core_param_grad_eval(
                ftp->ft, regopts->active_core, regopts->N,
                regopts->x, regopts->mem->running_evals_lr, regopts->mem->running_evals_rl,
                regopts->mem->running_grad[regopts->active_core],
                ftp->nparams_per_core[regopts->active_core],
                regopts->mem->evals->vals, regopts->mem->grad->vals,
                regopts->mem->grad_space->vals,
                reg_mem_space_get_data_inc(regopts->mem->grad_space),
                regopts->mem->fparam_space->vals
                );
        }
        
        for (size_t ii = 0; ii < regopts->N; ii++){
            /* printf("grad = "); dprint(ftp->nparams,regopts->mem->grad->vals + ii * ftp->nparams); */
            resid = regopts->y[ii] - regopts->mem->evals->vals[ii];
            out += 0.5 * resid * resid;
            cblas_daxpy(ftp->nparams_per_core[regopts->active_core],
                        -resid,
                        regopts->mem->grad->vals + ii * ftp->nparams_per_core[regopts->active_core],
                        1,grad,1);
        }        
    }
    else{
        function_train_core_param_grad_eval(
            ftp->ft, regopts->active_core,regopts->N,
            regopts->x,
            regopts->mem->running_evals_lr,
            regopts->mem->running_evals_rl,NULL,
            ftp->nparams_per_core[regopts->active_core],
            regopts->mem->evals->vals, NULL,NULL,0,NULL);
        for (size_t ii = 0; ii < regopts->N; ii++){
            /* aio->evals->vals[ii] = function_train_eval(aio->ft,aio->x+ii*aio->dim); */
            resid = regopts->y[ii] - regopts->mem->evals->vals[ii];
            out += 0.5 * resid * resid;
        }   
    }
    return out;
}

double ft_param_eval_objective_als(struct FTparam * ftp,
                                   struct RegressOpts * regopts,
                                   double * grad)
{
    double out = 0.0;
    if (regopts->obj == FTLS){
        out = ft_param_eval_objective_als_ls(ftp,regopts,grad);
    }
    else{
        assert (1 == 0);
    }

    return out;
}


struct PP
{
    struct FTparam * ftp;
    struct RegressOpts * opts;
};

double regress_opts_minimize_aio(size_t nparam, const double * param,
                                 double * grad, void * args)
{
    (void)(nparam);

    struct PP * pp = args;
    assert (pp->opts->nmem_alloc == pp->opts->N);

    // check if special structure exists / initialized
    regress_mem_manager_check_structure(pp->opts->mem, pp->ftp,pp->opts->x);
    regress_mem_manager_reset_running(pp->opts->mem);

    ft_param_update_params(pp->ftp,param);
    if (grad != NULL){
        for (size_t ii = 0; ii < nparam; ii++){
            grad[ii] = 0.0;
        }
    }
    return ft_param_eval_objective_aio(pp->ftp,pp->opts,grad);
}


struct FunctionTrain *
c3_regression_run_aio(struct FTparam * ftp, struct RegressOpts * ropts)
{

    struct PP pp;
    pp.ftp = ftp;
    pp.opts = ropts;

    /* enum FTPARAM_ST structure = NONE_ST; */
    enum FTPARAM_ST structure = LINEAR_ST;
    size_t max_param_within_uni = 1000;
    size_t * ranks = function_train_get_ranks(ftp->ft);
    regress_opts_initialize_memory(ropts, ftp->nparams_per_core,
                                   ranks, max_param_within_uni, structure);

    assert (ropts->optimizer != NULL);
    c3opt_add_objective(ropts->optimizer,regress_opts_minimize_aio,&pp);

    if (ropts->verbose == 3){
        c3opt_set_verbose(ropts->optimizer,1);
    }
    c3opt_set_relftol(ropts->optimizer,ropts->convtol);
    c3opt_set_absxtol(ropts->optimizer,1e-20);
    /* c3opt_ls_set_beta(ftr->optimizer,0.99); */
    c3opt_set_gtol(ropts->optimizer,1e-20);

    double * guess = calloc_double(ftp->nparams);
    memmove(guess,ftp->params,ftp->nparams * sizeof(double));

    double val;
    int res = c3opt_minimize(ropts->optimizer,guess,&val);
    if (res < -1){
        fprintf(stderr,"Warning: optimizer exited with code %d\n",res);
    }

    if (ropts->verbose == 1){
        printf("Objective value = %3.5G\n",val);
    }
    ft_param_update_params(ftp,guess);
    struct FunctionTrain * ft_final = function_train_copy(ftp->ft);

    free(guess); guess = NULL;

    return ft_final;
}




double regress_opts_minimize_als(size_t nparam, const double * params,
                                 double * grad, void * args)
{
    struct PP * pp = args;
    assert (pp->opts->nmem_alloc == pp->opts->N);

    // check if special structure exists / initialized
    regress_mem_manager_check_structure(pp->opts->mem,pp->ftp,pp->opts->x);

    // just need to reset the gradient of the active core
    running_core_total_restart(pp->opts->mem->running_grad[pp->opts->active_core]);


    ft_param_update_core_params(pp->ftp,pp->opts->active_core,params);
    if (grad != NULL){
        for (size_t ii = 0; ii < nparam; ii++){
            grad[ii] = 0.0;
        }
    }
    return ft_param_eval_objective_als(pp->ftp,pp->opts,grad);
}

struct FunctionTrain *
c3_regression_run_als(struct FTparam * ftp, struct RegressOpts * ropts)
{

    enum FTPARAM_ST structure = NONE_ST;
    /* enum FTPARAM_ST structure = LINEAR_ST; */
    size_t max_param_within_uni = 1000;
    size_t * ranks = function_train_get_ranks(ftp->ft);
    regress_opts_initialize_memory(ropts, ftp->nparams_per_core,
                                   ranks, max_param_within_uni, structure);

    for (size_t sweep = 1; sweep <= ropts->maxsweeps; sweep++){
        if (ropts->verbose == 1){
            printf("Sweep %zu\n",sweep);
        }
        struct FunctionTrain * start = function_train_copy(ftp->ft);
        for (size_t ii = 0; ii < ftp->dim; ii++){
            if (ropts->verbose == 1){
                printf("\tDim %zu: ",ii);
            }
            regress_mem_manager_reset_running(ropts->mem);
            regress_opts_prepare_als_core(ropts,ii,ftp);


            c3opt_free(ropts->optimizer); ropts->optimizer = NULL;
            ropts->optimizer = c3opt_alloc(BFGS,ftp->nparams_per_core[ii]);

            struct PP pp;
            pp.ftp = ftp;
            pp.opts = ropts;
        
            c3opt_add_objective(ropts->optimizer,regress_opts_minimize_als,&pp);
            
            c3opt_set_verbose(ropts->optimizer,0);
            if (ropts->verbose == 3){
                c3opt_set_verbose(ropts->optimizer,1);
            }
            c3opt_set_relftol(ropts->optimizer,ropts->convtol);
            c3opt_set_absxtol(ropts->optimizer,1e-20);
            /* c3opt_ls_set_beta(ftr->optimizer,0.99); */
            c3opt_set_gtol(ropts->optimizer,1e-20);

            double * guess = calloc_double(ftp->nparams_per_core[ii]);
            function_train_core_get_params(ftp->ft,ii,guess);
            double val;
            int res = c3opt_minimize(ropts->optimizer,guess,&val);
            /* printf("active core = %zu\n",pp.opts->active_core); */
            if (res < -1){
                fprintf(stderr,"Warning: optimizer exited with code %d\n",res);
            }

            if (ropts->verbose == 1){
                printf("\t\tObjVal = %3.5G\n",val);
            }

            ft_param_update_core_params(ftp,ii,guess);
            free(guess); guess = NULL;
        }

        for (size_t jj = 1; jj < ftp->dim-1; jj++){
            size_t ii = ftp->dim-1-jj;
            if (ropts->verbose == 1){
                printf("\tDim %zu: ",ii);
            }
            regress_mem_manager_reset_running(ropts->mem);
            regress_opts_prepare_als_core(ropts,ii,ftp);


            c3opt_free(ropts->optimizer); ropts->optimizer = NULL;
            ropts->optimizer = c3opt_alloc(BFGS,ftp->nparams_per_core[ii]);

            struct PP pp;
            pp.ftp = ftp;
            pp.opts = ropts;
        
            c3opt_add_objective(ropts->optimizer,regress_opts_minimize_als,&pp);

            c3opt_set_verbose(ropts->optimizer,0);
            c3opt_set_relftol(ropts->optimizer,1e-10);
            c3opt_set_absxtol(ropts->optimizer,1e-20);
            /* c3opt_ls_set_beta(ftr->optimizer,0.99); */
            c3opt_set_gtol(ropts->optimizer,1e-20);

            double * guess = calloc_double(ftp->nparams_per_core[ii]);
            function_train_core_get_params(ftp->ft,ii,guess);
            double val;
            int res = c3opt_minimize(ropts->optimizer,guess,&val);
            /* printf("active core = %zu\n",pp.opts->active_core); */
            if (res < -1){
                fprintf(stderr,"Warning: optimizer exited with code %d\n",res);
            }

            if (ropts->verbose == 1){
                printf("\t\tObjVal = %3.5G\n",val);
            }

            ft_param_update_core_params(ftp,ii,guess);
            free(guess); guess = NULL;
        }
        
        double diff = function_train_relnorm2diff(ftp->ft,start);
        if (ropts->verbose == 1){
            printf("\n\tSweep Difference = %G\n",diff);
        }
        function_train_free(start); start = NULL;
    }


    struct FunctionTrain * ft_final = function_train_copy(ftp->ft);

    return ft_final;
}


struct FunctionTrain *
c3_regression_run(struct FTparam * ftp, struct RegressOpts * regopts)
{
    // check if special structure exists / initialized
    
    struct FunctionTrain * ft = NULL;
    if (regopts->type == AIO){
        ft = c3_regression_run_aio(ftp,regopts);
    }
    else if (regopts->type == ALS){
        ft = c3_regression_run_als(ftp,regopts);
    }
    else{
        printf("Regress opts of type %d is unknown\n",regopts->type);
        printf("                     AIO is %d \n",AIO);
        printf("                     ALS is %d \n",AIO);
        exit(1);
    }

    return ft;
}



////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
// Top-level regression parameters
// like regularization, rank,
// poly order
////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
////////////////////////////////////

enum REGPARAMTYPE {DISCRETE,CONTINUOUS};
struct RegParameter
{
    char name[30];
    enum REGPARAMTYPE type;

    size_t nops;
    size_t * discrete_ops;
    size_t discval;
    
    double lbc;
    double ubc;
    double cval;
};

/***********************************************************//**
    Allocate parameter
***************************************************************/
struct RegParameter * reg_param_alloc(char * name, enum REGPARAMTYPE type)
{
    struct RegParameter * rp = malloc(sizeof(struct RegParameter));
    if (rp == NULL){
        fprintf(stderr, "Cannot allocate RegParameter structure\n");
        exit(1);
    }
    strcpy(rp->name,name);
    rp->type = type;

    rp->nops = 0;
    rp->discrete_ops = NULL;

    rp->lbc = -DBL_MAX;
    rp->ubc = DBL_MAX;
    
    return rp;
}


/***********************************************************//**
    Allocate parameter
***************************************************************/
struct RegParameter ** reg_param_array_alloc(size_t N)
{
    struct RegParameter ** rp = malloc(N * sizeof(struct RegParameter *));
    if (rp == NULL){
        fprintf(stderr, "Cannot allocate array of RegParameter structure\n");
        exit(1);
    }
    for (size_t ii = 0; ii < N; ii++){
        rp[ii] = NULL;
    }
    return rp;
}

/***********************************************************//**
    Free parameter
***************************************************************/
void reg_param_free(struct RegParameter * rp)
{
    if (rp != NULL){
        free(rp->discrete_ops); rp->discrete_ops = NULL;
        free(rp); rp = NULL;
    }
}

/***********************************************************//**
   Add Discrete parameter options                                                        
***************************************************************/
void reg_param_add_discrete_options(struct RegParameter * rp,
                                    size_t nops, size_t * opts)
{
    assert (rp != NULL);
    assert (rp->type == DISCRETE);
    rp->nops = nops;

    free(rp->discrete_ops);
    rp->discrete_ops = calloc_size_t(nops);
    memmove(rp->discrete_ops,opts, nops * sizeof(size_t));
}

/***********************************************************//**
   Add Discrete parameter value
***************************************************************/
void reg_param_add_discrete_val(struct RegParameter * rp, size_t val)
{
    assert (rp != NULL);
    assert (rp->type == DISCRETE);
    rp->discval = val;
}


/***********************************************************//**
   Get discrete value
***************************************************************/
size_t reg_param_get_discrete_val(const struct RegParameter * rp)
{
    assert (rp != NULL);
    assert (rp->type == DISCRETE);
    return rp->discval;
}

////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
// Common Interface
////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
////////////////////////////////////

#define NRPARAM 3
static char reg_param_opts[NRPARAM][30] = {"rank","num_param","opt maxiter"};

struct FTRegress
{
    size_t type;
    size_t obj;
    size_t dim;
    size_t ntotparam;

    struct MultiApproxOpts * approx_opts;
    struct FTparam * ftp;
    struct RegressOpts * regopts;
    
    // high level parameters
    size_t nhlparam;
    struct RegParameter * hlparam[NRPARAM];

    size_t optmaxiter;
    size_t * start_ranks;

    size_t N;
    const double * x;
    const double * y;
    
    // internal
    size_t processed_param;
};

/***********************************************************//**
    Allocate function train regression structure
***************************************************************/
struct FTRegress *
ft_regress_alloc(size_t dim, struct MultiApproxOpts * aopts)
{
    struct FTRegress * ftr = malloc(sizeof(struct FTRegress));
    if (ftr == NULL){
        fprintf(stderr, "Cannot allocate FTRegress structure\n");
        exit(1);
    }
    ftr->type = REGNONE;
    ftr->obj = REGOBJNONE;
    ftr->dim = dim;
    ftr->ntotparam = 0;

    ftr->approx_opts = aopts;
    ftr->ftp = NULL;
    ftr->regopts = NULL;
    
    ftr->nhlparam = 0;
    for (size_t ii = 0; ii < NRPARAM; ii++){
        ftr->hlparam[ii] = NULL;
    }

    ftr->optmaxiter = 1000;
    ftr->start_ranks = NULL;

    ftr->N = 0;
    ftr->x = NULL;
    ftr->y = NULL;
    
    ftr->processed_param = 0;
    return ftr;
}
    
/***********************************************************//**
    Free FT regress structure
***************************************************************/
void ft_regress_free(struct FTRegress * ftr)
{
    if (ftr != NULL){
        ft_param_free(ftr->ftp); ftr->ftp = NULL;
        regress_opts_free(ftr->regopts); ftr->regopts = NULL;

        for (size_t ii = 0; ii < ftr->nhlparam; ii++){
            reg_param_free(ftr->hlparam[ii]); ftr->hlparam[ii] = NULL;
        }

        free(ftr->start_ranks); ftr->start_ranks = NULL;
        free(ftr); ftr = NULL;
    }
}

/***********************************************************//**
    Clear memory, keep only dim
***************************************************************/
void ft_regress_reset(struct FTRegress ** ftr)
{

    size_t dim = (*ftr)->dim;
    struct MultiApproxOpts * aopts = (*ftr)->approx_opts;
    ft_regress_free(*ftr); *ftr = NULL;
    *ftr = ft_regress_alloc(dim,aopts);
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
    
    if ((type != ALS) && (type != AIO)){
        fprintf(stderr,"Error: Regularization type %d not available. Options include\n",type);
        fprintf(stderr,"       ALS = 0\n");
        fprintf(stderr,"       AIO = 1\n");
        exit(1);
    }
}

/***********************************************************//**
    Set Regression Objective Function
***************************************************************/
void ft_regress_set_obj(struct FTRegress * ftr, enum REGOBJ obj)
{
    assert (ftr != NULL);
    if (ftr->obj != REGOBJNONE){
        fprintf(stdout,"Warning: respecifying regression objective function may lead to memory problems");
        fprintf(stdout,"Use function ft_regress_reset_obj instead\n");
    }
    ftr->obj = obj;
    
    if (obj != FTLS){
        fprintf(stderr,"Error: Regularization type %d not available. Options include\n",obj);
        fprintf(stderr,"       FTLS = 0\n");
        exit(1);
    }
}

/***********************************************************//**
    Set regression type
***************************************************************/
void ft_regress_set_discrete_parameter(struct FTRegress * ftr, char * name, size_t val)
{

    assert (ftr != NULL);

    // check if allowed parameter
    int match = 0;
    for (size_t ii = 0; ii < NRPARAM; ii++){
        if (strcmp(reg_param_opts[ii],name) == 0){
            match = 1;
            break;
        }
    }
    if (match == 0){
        fprintf(stderr,"Unknown regression parameter type %s\n",name);
        exit(1);
    }

    // check if parameter exists
    int exist = 0;
    for (size_t ii = 0; ii < ftr->nhlparam; ii++){
        if (strcmp(ftr->hlparam[ii]->name,name) == 0){
            reg_param_add_discrete_val(ftr->hlparam[ii],val);
            exist = 1;
            break;
        }
    }

    /* printf("num param = %zu\n",ftr->nhlparam); */
    if (exist == 0){
        ftr->hlparam[ftr->nhlparam] = reg_param_alloc(name,DISCRETE);
        reg_param_add_discrete_val(ftr->hlparam[ftr->nhlparam],val);
        ftr->nhlparam++;
        assert (ftr->nhlparam <= NRPARAM);
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

    ftr->N = N;
    ftr->x = x;
    ftr->y = y;
}

/***********************************************************//**
   Update parameters
***************************************************************/
void ft_regress_update_params(struct FTRegress * ftr, const double * param)
{
    assert (ftr != NULL);
    assert (ftr->ftp != NULL);

    ft_param_update_params(ftr->ftp,param);
}


/***********************************************************//**
    Process all parameters (not those optimizing over, but problem structure)
***************************************************************/
void ft_regress_process_parameters(struct FTRegress * ftr)
{

    if (ftr->type == REGNONE){
        fprintf(stderr, "Must call ft_regress_set_type before processing parameters\n");
        exit(1);
    }
    else if (ftr->obj == REGOBJNONE){
        fprintf(stderr, "Must call ft_regress_set_obj before processing parameters\n");
        exit(1);
    }
    else if (ftr->N == 0){
        fprintf(stderr, "Must add data before processing arguments\n");
        exit(1);
    }
    
    if (ftr->ftp != NULL){
        ft_param_free(ftr->ftp); ftr->ftp = NULL;
    }
    if (ftr->regopts != NULL){
        regress_opts_free(ftr->regopts); ftr->regopts = NULL;
    }
    
    for (size_t ii = 0; ii < ftr->nhlparam; ii++){
        if (strcmp(ftr->hlparam[ii]->name,"rank") == 0){
            /* printf("processed rank\n"); */
            free(ftr->start_ranks); ftr->start_ranks = NULL;
            ftr->start_ranks = calloc_size_t(ftr->dim+1);
            for (size_t kk = 1; kk < ftr->dim; kk++){
                ftr->start_ranks[kk] = reg_param_get_discrete_val(ftr->hlparam[ii]);
            }
            ftr->start_ranks[0] = 1;
            ftr->start_ranks[ftr->dim] = 1;
        }
        else if (strcmp(ftr->hlparam[ii]->name,"num_param") == 0){
            /* printf("processed num_param\n"); */
            size_t val = reg_param_get_discrete_val(ftr->hlparam[ii]);
            for (size_t kk = 0; kk < ftr->dim; kk++){
                multi_approx_opts_set_dim_nparams(ftr->approx_opts,kk,val);
            }
            /* fprintf(stderr,"Cannot process nparam_func parameter yet\n"); */
            /* exit(1); */
        }
        else if (strcmp(ftr->hlparam[ii]->name,"opt maxiter") == 0){
            /* printf("processed maxiter\n"); */
            ftr->optmaxiter = reg_param_get_discrete_val(ftr->hlparam[ii]);
        }
    }



    double avgy = 0.0;
    for (size_t ii = 0; ii < ftr->N; ii++){
        avgy += ftr->y[ii];
    }
    avgy /= (double)ftr->N;

    assert (ftr->approx_opts != NULL);
    size_t running=0;
    size_t maxparams_core = 0;
    for (size_t ii = 0; ii < ftr->dim; ii++){

        size_t nparams_core = multi_approx_opts_get_dim_nparams(ftr->approx_opts,ii);


        if (nparams_core > maxparams_core){
            maxparams_core = nparams_core;
        }
        running += nparams_core * ftr->start_ranks[ii] * ftr->start_ranks[ii+1];
    }
    
    /* struct FunctionTrain * a = function_train_constant(avgy,ftr->approx_opts); */
    /* double * a_params = calloc_double(running); */
    
    double * init_params = calloc_double(running);
    running = 0;
    for (size_t ii = 0; ii < ftr->dim; ii++){
        size_t nparams_uni = multi_approx_opts_get_dim_nparams(ftr->approx_opts,ii);
        size_t nparams_core = nparams_uni * ftr->start_ranks[ii] * ftr->start_ranks[ii+1];
        /* size_t na = function_train_core_get_params(a,ii,a_params); */
        /* for (size_t kk = 0; kk < na; kk++){ // since a is a constant it is rank one so just copy as many params */
        /*     init_params[running+kk] = a_params[kk]; */
        /*     if (kk >= nparams_uni){  */
        /*         break; */
        /*     } */
        /* } */
        for (size_t kk = 0; kk < nparams_core; kk++){
            init_params[running+kk] = randu()*2.0-1.0; /* * 1e-2 / (double)ftr->dim; */
        }
        running += nparams_core;
     }
    
    ftr->ftp = ft_param_alloc(ftr->dim,ftr->approx_opts,init_params,ftr->start_ranks);
    free(init_params); init_params = NULL;
    /* free(a_params); a_params = NULL; */
    /* function_train_free(a); a = NULL; */
    
    ftr->regopts = regress_opts_create(ftr->type,ftr->obj,ftr->N,ftr->dim,ftr->x,ftr->y);
    regress_opts_initialize_memory(
        ftr->regopts,ft_param_get_num_params_per_core(ftr->ftp),
        ftr->start_ranks,maxparams_core,LINEAR_ST);

    
    ftr->processed_param = 1;
}


/***********************************************************//**
    Run the regression
***************************************************************/
struct FunctionTrain *
ft_regress_run(struct FTRegress * ftr)
{
    assert (ftr != NULL);
    assert (ftr->ftp != NULL);
    assert (ftr->regopts != NULL);

    struct FunctionTrain * ft = c3_regression_run(ftr->ftp,ftr->regopts);
    return ft;
}

////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
// Cross validation
////////////////////////////////////
////////////////////////////////////
////////////////////////////////////
////////////////////////////////////

struct CrossValidate
{
    size_t N;
    size_t dim;
    const double * x;
    const double * y;

    size_t kfold;

    size_t nparam;
    struct RegParameter * params[NRPARAM];
};


/***********************************************************//**
   Allocate cross validation struct
***************************************************************/
struct CrossValidate * cross_validate_alloc()
{
    struct CrossValidate * cv = malloc(sizeof(struct CrossValidate));
    if (cv == NULL){
        fprintf(stderr, "Cannot allocate CrossValidate structure\n");
        exit(1);
    }
    cv->N = 0;
    cv->dim = 0;
    cv->x = NULL;
    cv->y = NULL;

    cv->kfold = 0;
    
    cv->nparam = 0;
    for (size_t ii = 0; ii < NRPARAM; ii++){
        cv->params[ii] = NULL;
    }
    return cv;
}


/***********************************************************//**
   Free cross validation struct
***************************************************************/
void cross_validate_free(struct CrossValidate * cv)
{
    if (cv != NULL){
        for (size_t ii = 0; ii < cv->nparam; ii++){
            reg_param_free(cv->params[ii]); cv->params[ii] = NULL;
        }
        free(cv); cv = NULL;
    }
}

/***********************************************************//**
   Initialize cross validation
***************************************************************/
struct CrossValidate * cross_validate_init(size_t N, size_t dim,
                                           const double * x,
                                           const double * y,
                                           size_t kfold)
{
    struct CrossValidate * cv = cross_validate_alloc();
    cv->N = N;
    cv->dim = dim;
    cv->x = x;
    cv->y = y;

    cv->kfold = kfold;
    
    return cv;
}

/***********************************************************//**
   Add a discrete cross validation parameter
***************************************************************/
void cross_validate_add_discrete_param(struct CrossValidate * cv,
                                       char * name, size_t nopts,
                                       size_t * opts)
{
    assert (cv!= NULL);

    cv->params[cv->nparam] = reg_param_alloc(name,DISCRETE);
    reg_param_add_discrete_options(cv->params[cv->nparam],nopts,opts);
    cv->nparam++;
}

/***********************************************************//**
   Cross validation run
***************************************************************/
void extract_data(struct CrossValidate * cv, size_t start,
                  size_t num_extract,
                  double ** xtest, double ** ytest,
                  double ** xtrain, double ** ytrain)
{

    memmove(*xtest,cv->x + start * cv->dim,
            cv->dim * num_extract * sizeof(double));
    memmove(*ytest,cv->y + start, num_extract * sizeof(double));
        
    memmove(*xtrain,cv->x, cv->dim * start * sizeof(double));
    memmove(*ytrain,cv->y, start * sizeof(double));

    memmove(*xtrain + start * cv->dim,
            cv->x + (start+num_extract)*cv->dim,
            cv->dim * (cv->N-start-num_extract) * sizeof(double));
    memmove(*ytrain + start, cv->y + (start+num_extract),
            (cv->N-start-num_extract) * sizeof(double));
}



/***********************************************************//**
   Cross validation run
***************************************************************/
double cross_validate_run(struct CrossValidate * cv,
                          struct FTRegress * reg)
{

    size_t batch = cv->N / cv->kfold;

    /* printf("batch size = %zu\n",batch); */

    double err = 0.0;
    double start_num = 0;
    double * xtrain = calloc_double(cv->N * cv->dim);
    double * ytrain = calloc_double(cv->N);
    double * xtest = calloc_double(cv->N * cv->dim);
    double * ytest = calloc_double(cv->N);

    double erri = 0.0;
    double norm = 0.0;
    for (size_t ii = 0; ii < cv->kfold; ii++){

        if (start_num + batch > cv->N){
            batch = cv->N-start_num;
        }

        /* printf("batch[%zu] = %zu\n",ii,batch); */
        
        extract_data(cv, start_num, batch, &xtest, &ytest,
                     &xtrain, &ytrain);

        // train with all but *ii*th batch
        /* printf("ytrain = ");dprint(cv->N-batch,ytrain); */


        // update data and reprep memmory
        ft_regress_set_data(reg,cv->N-batch,xtrain,1,ytrain,1);
        ft_regress_process_parameters(reg);
        struct FunctionTrain * ft = ft_regress_run(reg);


        double newerr = 0.0;
        double newnorm = 0.0;
        for (size_t jj = 0; jj < batch; jj++){
            double eval = function_train_eval(ft,xtest+jj*cv->dim);
            /* printf("ytest,eval = %G,%G\n",ytest[jj],eval); */
            /* if (fabs(ytest[jj] - eval)/fabs(ytest[jj]) > 0.5){ */
            /*     dprint(cv->dim, xtest+jj*cv->dim); */
            /*     /\* exit(1); *\/ */
            /* } */
            
            newerr += (ytest[jj]-eval) * (ytest[jj]-eval);
            newnorm += (ytest[jj]*ytest[jj]);
        }
        /* erri /= (double)(batch); */

        /* printf("Relative error on batch = %G\n",newerr/newnorm); */
        /* if (newerr / newnorm > 100){ */
            /* printf("Ranks are "); */
            /* iprint_sz(cv->dim, ft->ranks); */

            /* printf("training points are\n"); */
            /* dprint2d_col(cv->dim,cv->N-batch, xtrain); */

            /* printf("training values are\n"); */
            /* dprint(cv->N-batch,ytrain); */
            
            /* printf("Error is too large\n"); */
            /* exit(1); */
        /* } */

        erri += newerr;
        norm += newnorm;
        start_num += batch;
    
    }
    err = erri/norm;
    // do last back

    free(xtrain); xtrain = NULL;
    free(ytrain); ytrain = NULL;
    free(xtest); xtest = NULL;
    free(ytest); ytest = NULL;

    ft_regress_set_data(reg,cv->N,cv->x,1,cv->y,1);

    return err;
}


/***********************************************************//**
   Cross validation optimize         
   ONLY WORKS FOR DISCRETE OPTIONS
***************************************************************/
void cross_validate_opt(struct CrossValidate * cv,
                        struct FTRegress * ftr, int verbose)
{
    if (cv->nparam == 1){
        size_t nopts = cv->params[0]->nops;
        char * name = cv->params[0]->name;
        double minerr = 0;
        size_t minval = 0;
        for (size_t ii = 0; ii < nopts; ii++){
            size_t val = cv->params[0]->discrete_ops[ii];
            ft_regress_set_discrete_parameter(ftr,name,val);
            ft_regress_process_parameters(ftr);
            double err = cross_validate_run(cv,ftr);
            if (verbose > 1){
                printf("\"%s\", val=%zu, cv_err=%G\n",name,val,err);
            }

            if (ii == 0){
                minerr = err;
                minval = val;
            }
            else{
                if (err < minerr){
                    minerr = err;
                    minval = val;
                }
            }
        }

        if (verbose > 0){
            printf("Optimal CV params: ");
            printf("\t \"%s\", val=%zu, cv_err=%G\n",name,minval,minerr);
        }
        ft_regress_set_discrete_parameter(ftr,name,minval);
        ft_regress_process_parameters(ftr);
    }
    else if (cv->nparam == 2){
        size_t nopts = cv->params[0]->nops;
        char * name = cv->params[0]->name;
        size_t nopts2 = cv->params[1]->nops;
        char * name2 = cv->params[1]->name;

        double minerr = 0.0;;
        size_t minval[2] = {0,0};
        for (size_t ii = 0; ii < nopts; ii++){
            size_t val = cv->params[0]->discrete_ops[ii];
            ft_regress_set_discrete_parameter(ftr,name,val);

            for (size_t jj = 0; jj < nopts2; jj++){

                size_t val2 = cv->params[1]->discrete_ops[jj];
                ft_regress_set_discrete_parameter(ftr,name2,val2);
                ft_regress_process_parameters(ftr);
            
                double err = cross_validate_run(cv,ftr);
                if (verbose > 1){
                    printf("\"%s\":%zu, \"%s\":%zu, ",name,val,name2,val2);
                    printf("cv_err=%G\n",err);
                }
                if ((ii==0) && (jj==0)){
                    minerr = err;
                    minval[0] = val;
                    minval[1] = val2;
                }
                else{
                    if (err < minerr){
                        minerr = err;
                        minval[0] = val;
                        minval[1] = val2;
                    }
                }
            }
        }
        if (verbose > 0){
            printf("Optimal CV params: ");
            printf("\"%s\":%zu, \"%s\":%zu, ",name,minval[0],name2,minval[1]);
            printf("cv_err=%G\n",minerr);
        }
        ft_regress_set_discrete_parameter(ftr,name,minval[0]);
        ft_regress_set_discrete_parameter(ftr,name2,minval[1]);
        ft_regress_process_parameters(ftr);
    }
    else if (cv->nparam == 3){
        size_t nopts = cv->params[0]->nops;
        char * name = cv->params[0]->name;
        size_t nopts2 = cv->params[1]->nops;
        char * name2 = cv->params[1]->name;
        size_t nopts3 = cv->params[2]->nops;
        char * name3 = cv->params[2]->name;

        double minerr = 0.0;
        size_t minval[3] = {0,0,0};
        for (size_t ii = 0; ii < nopts; ii++){
            size_t val = cv->params[0]->discrete_ops[ii];
            ft_regress_set_discrete_parameter(ftr,name,val);

            for (size_t jj = 0; jj < nopts2; jj++){
                size_t val2 = cv->params[1]->discrete_ops[jj];
                ft_regress_set_discrete_parameter(ftr,name2,val2);
                
                for (size_t kk = 0; kk < nopts3; kk++){
                    size_t val3 = cv->params[2]->discrete_ops[kk];
                    ft_regress_set_discrete_parameter(ftr,name3,val3);
                    ft_regress_process_parameters(ftr);
            
                    double err = cross_validate_run(cv,ftr);
                    if (verbose > 1){
                        printf("\"%s\":%zu, \"%s\":%zu,",name,val,name2,val2);
                        printf("\"%s\":%zu, cv_err=%G\n",name3,val3,err);
                    }

                    if ((ii==0) && (jj==0) && (kk == 0)){
                        minerr = err;
                        minval[0] = val;
                        minval[1] = val2;
                        minval[2] = val3;
                    }
                    else{
                        if (err < minerr){
                            minerr = err;
                            minval[0] = val;
                            minval[1] = val2;
                            minval[2] = val3;
                        }
                    }
                }
            }
        }

        if (verbose > 0){
            printf("Optimal CV params: \n");
            printf("\"%s\":%zu, \"%s\":%zu,",name,minval[0],name2,minval[1]);
            printf("\"%s\":%zu, cv_err=%G\n",name3,minval[2],minerr);
        }
        ft_regress_set_discrete_parameter(ftr,name,minval[0]);
        ft_regress_set_discrete_parameter(ftr,name2,minval[1]);
        ft_regress_set_discrete_parameter(ftr,name3,minval[2]);
        ft_regress_process_parameters(ftr);
    }
    else {
        fprintf(stderr,"Cannot cross validate over arbitrary number of \n");
        fprintf(stderr,"parameters yet.\n");
    }

}

