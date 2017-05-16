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

/** \file regress.c
 * Provides routines for FT-regression
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "lib_linalg.h"
#include "regress.h"


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
// Regression code
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



double c3_objective_function_least_squares(
    size_t nparam, const double * param,
    size_t N, size_t dim, const double * x, const double * y,
    int (*mapping)(size_t nparam, const double * param, size_t N,
                   const double * x, const double * y, double ** evals, double ** grads, void * args),
    void * args, double * grad)
{


    double * evals = NULL;
    double * grads = NULL; 

    double out = 0.0;
    double dy;
    if (grad == NULL){
        int res = mapping(nparam,param,N,dim,x,y,&evals,NULL,args);
        for (size_t ii = 0; ii < N; ii++){
            dy = y[ii] - evals[ii];
            out += dy * dy;
        }
    }
    else{
        int res = mapping(nparam,param,N,dim,x,y,&evals,&grads,args);
        assert (res == 0);
        for (size_t ii = 0; ii < N; ii++){
            dy = y[ii] - evals[ii];
            out += dy * dy;
            for (size_t jj = 0; jj < nparam; jj++){
                grads[jj] += 2.0 * dy * grads[ii*nparam+jj];
            }
        }
        for (size_t jj = 0; jj < nparam; jj++){
            grads[jj] /= (double)N;
        }
    }

    out /= (double)N;
    
    return out;
}



/** \struct RegressOpts
 * \brief Options for regression
 * \var RegressOpts::type
 * Regression type (ALS,AIO)
 * \var RegressOpts::obj
 * Regression objective (FTLS, FTLS_SPARSEL2)
 * \var RegressOpts::dim
 * size of feature space
 * \var RegressOpts::verbose
 * verbosity options
 * \var RegressOpts::regularization_weight
 * regularization weight for regularization objectives
 * \var RegressOpts::max_als_sweeps
 * maximum number of sweeps for ALS
 * \var RegressOpts::als_active_core
 * flag for active core within ALS
 * \var RegressOpts::als_conv_tol
 * convergence tolerance for als
 * \var RegressOpts::restrict_rank_opt
 * Restrict optimization of ranks to those >= values here
 */
struct RegressOpts
{
    enum REGTYPE type;
    enum REGOBJ obj;

    size_t dim;
    int verbose;
    double regularization_weight;
    size_t max_als_sweeps;
    size_t als_active_core;
    double als_conv_tol;

    size_t * restrict_rank_opt;


    int stoch_obj; // 1 for stochastic obj, 0 for not
};


/***********************************************************//**
    Free regression options
    
    \param[in,out] opts - options to free
***************************************************************/
void regress_opts_free(struct RegressOpts * opts)
{
    if (opts != NULL){
        free(opts->restrict_rank_opt); opts->restrict_rank_opt = NULL;
        free(opts); opts = NULL;
    }
}

/***********************************************************//**
    Allocate default regression options
    
    \param[in] dim - number of features

    \returns regression options
***************************************************************/
struct RegressOpts * regress_opts_alloc(size_t dim)
{
    struct RegressOpts * ropts = malloc(sizeof(struct RegressOpts));
    if (ropts == NULL){
        fprintf(stderr,"Error: Failure to allocate\n\t struct RegressOpts\n");
        exit(1);
    }
    ropts->regularization_weight = 1e-10;
    ropts->verbose = 0;
    ropts->als_active_core = 0;
    ropts->max_als_sweeps = 10;
    ropts->dim = dim;
    ropts->als_conv_tol = 1e-5;
    ropts->restrict_rank_opt = calloc_size_t(dim);

    ropts->stoch_obj = 0;
    return ropts;
}

/***********************************************************//**
    Allocate default regression options for a problem type
    
    \param[in] dim  - number of features
    \param[in] type - type of optimization
                      AIO: all-at-once
                      ALS: alternating least squares
    \param[in] obj  - objective type
                      FTLS: least squares
                      FTLS_SPARSEL2: regularize on sparsity of cores 
                                     measured by L2 norm of univariate
                                     functions

    \returns regression options
***************************************************************/
struct RegressOpts *
regress_opts_create(size_t dim, enum REGTYPE type, enum REGOBJ obj)
{

    struct RegressOpts * opts = regress_opts_alloc(dim);

    if (type == AIO){
        /* printf("type == AIO\n"); */
        opts->type = AIO;
    }
    else if (type == ALS){
        opts->type = ALS;
        opts->als_active_core = 0;
    }
    else{
        fprintf(stderr,"Error: REGTYPE %d unavailable\n",type);
        exit(1);
    }

    if (obj == FTLS){
        opts->obj = FTLS;
    }
    else if (obj == FTLS_SPARSEL2){
        opts->obj = FTLS_SPARSEL2;
    }
    else{
        fprintf(stderr,"Error: REGOBJ %d unavailable\n",obj);
        exit(1);
    }

    return opts;
}


/***********************************************************//**
    Set option specifying will use a stochastic optimizer
    
    \param[in,out] ropts - regression structure
    \param[in]     on    - 1 for yes, 0 for no
***************************************************************/
void regress_opts_set_stoch_obj(struct RegressOpts * ropts, int on)
{
    assert (ropts != NULL);
    ropts->stoch_obj = on;
}

/***********************************************************//**
    Set maximum number of als sweeps
    
    \param[in,out] opts      - regression options
    \param[in]     maxsweeps - number of sweeps     
***************************************************************/
void regress_opts_set_max_als_sweep(struct RegressOpts * opts, size_t maxsweeps)
{
    assert (opts != NULL);
    opts->max_als_sweeps = maxsweeps;
}

/***********************************************************//**
    Set the ALS convergence tolerance
    
    \param[in,out] opts - regression options
    \param[in]     tol  - number of sweeps     
***************************************************************/
void regress_opts_set_als_conv_tol(struct RegressOpts * opts, double tol)
{
    assert (opts != NULL);
    opts->als_conv_tol = tol;
}

/***********************************************************//**
    Set regularization weight
    
    \param[in,out] opts   - regression options
    \param[in]     weight - regularization weight   
***************************************************************/
void regress_opts_set_regularization_weight(struct RegressOpts * opts, double weight)
{
    assert (opts != NULL);
    opts->regularization_weight = weight;
}

/***********************************************************//**
    Get regularization weight
    
    \param[in] opts - regression options
    
    \returns regularization weight   
***************************************************************/
double regress_opts_get_regularization_weight(const struct RegressOpts * opts)
{
    assert (opts != NULL);
    return opts->regularization_weight;
}


/***********************************************************//**
    Set verbosity level
    
    \param[in,out] opts    - regression options
    \param[in]     verbose - verbosity level
***************************************************************/
void regress_opts_set_verbose(struct RegressOpts * opts, int verbose)
{
    assert (opts != NULL);
    opts->verbose = verbose;
}

/***********************************************************//**
    Set a rank to restrict
    
    \param[in,out] opts - regression options
    \param[in]     ind  - index of rank to restrict
    \param[in]     rank - threshold of restriction
***************************************************************/
void regress_opts_set_restrict_rank(struct RegressOpts * opts, size_t ind, size_t rank)
{
    assert (opts != NULL);
    opts->restrict_rank_opt[ind] = rank;
}

///////////////////////////////////////////////////////////////////////////
// Parameterized FT
///////////////////////////////////////////////////////////////////////////

/***********************************************************//**
    Prepare a core for ALS by evaluating the previous and 
    next cores

    \param[in]     ftp  - parameterized FT
    \param[in]     core - which core to prepare
    \param[in,out] mem  - memory manager that will store the evaluations
    \param[in]     N    - Number of evaluations
    \param[in]     x    - evaluation locations
***************************************************************/
static void ft_param_prepare_als_core(struct FTparam * ftp,size_t core,
                                      struct RegressionMemManager * mem, size_t N,
                                      const double * x)
{
    function_train_core_pre_post_run(ftp->ft,core,N,x,
                                     mem->running_evals_lr,
                                     mem->running_evals_rl);
}


static void ft_param_eval_grad_preprocess(struct FTparam * ftp, struct RegressOpts * ropts,
                                          size_t nparam, const double * param)
{
    int restrict_needed = restrict_ranksp(ropts);
    if (restrict_needed == 0){
        ft_param_update_params(ftp,param);
    }
    else{
        ft_param_update_restricted_ranks(ftp,param,ropts->restrict_rank_opt);
    }
}

static void ft_param_eval_grad_postprocess(struct FTparam * ftp, struct RegressOpts * ropts,
                                           size_t nparam, const double * param)
{

    // NEED TO DEAL WITH RESTRICTED RANKS
    int restrict_needed = restrict_ranksp(ropts);
    if (restrict_needed != 0){
        ft_param_update_restricted_ranks(ftp,param,ropts->restrict_rank_opt);
        extract_restricted_vals(opts,ftp,grad_use,grad);
        free(grad_use); grad_use = NULL;
    }
    
}
    

    struct PP * mem_opts = arg;
    struct RegressionMemManager * mem   = mem_opts->mem;
    struct RegressOpts *          ropts = mem_opts->opts;
    struct FTParam *              ftp   = mem_opts->ftp;
    ft_param_eval_grad_preprocess(ftp,ropts,nparam,param);

    ft_param_eval_grad_postprocess(ftop,ropts,nparam,param);

int ft_param_eval_grad(size_t N, const double * x, const double * y, double ** evals,
                       double ** grads, void * arg)
{

    struct PP * mem_opts = arg;
    struct RegressionMemManager * mem   = mem_opts->mem;
    struct RegressOpts *          ropts = mem_opts->opts;
    struct FTParam *              ftp   = mem_opts->ftp;
    if (ropts->alg == AIO){
        if (grads != NULL){
            if (mem->structure == LINEAR_ST){
                function_train_linparam_grad_eval(
                    ftp->ft,
                    N,x,
                    mem->running_evals_lr, mem->running_evals_rl, mem->running_grad,
                    ftp->nparams_per_core,
                    mem->evals->vals, mem->grad->vals,
                    mem->lin_structure_vals, mem->lin_structure_inc);
            }
            else{
                /* printf("in here\n"); */
                function_train_param_grad_eval(
                    ftp->ft,
                    N, x,
                    mem->running_evals_lr, mem->running_evals_rl, mem->running_grad,
                    ftp->nparams_per_core, mem->evals->vals, mem->grad->vals,
                    mem->grad_space->vals,
                    reg_mem_space_get_data_inc(mem->grad_space),
                    mem->fparam_space->vals
                    );
            }

            // NEED TO DEAL WITH RESTRICTED RANKS
            *evals = mem->evals->vals;
            *grads = mem->grad->vals;
        }
        else{
            function_train_param_grad_eval(
                ftp->ft,
                N, x,
                mem->running_evals_lr,NULL,NULL,
                ftp->nparams_per_core,
                mem->evals->vals, NULL,NULL,0,NULL);

            *evals = mem->evals->vals;
        }
    }
    else if (ropts->alg == ALS){
        size_t active_core = ropts->als_active_core;
        if (grads != NULL){
            if (mem->structure == LINEAR_ST){
                function_train_core_linparam_grad_eval(
                    ftp->ft, active_core,
                    N, x, mem->running_evals_lr, mem->running_evals_rl,
                    mem->running_grad[active_core],
                    ftp->nparams_per_core[active_core],
                    mem->evals->vals, mem->grad->vals,
                    mem->lin_structure_vals[active_core],
                    mem->lin_structure_inc[active_core]);
            }
            else{
                function_train_core_param_grad_eval(
                    ftp->ft, active_core, N,
                    x, mem->running_evals_lr, mem->running_evals_rl,
                    mem->running_grad[active_core],
                    ftp->nparams_per_core[active_core],
                    mem->evals->vals, mem->grad->vals,
                    mem->grad_space->vals,
                    reg_mem_space_get_data_inc(mem->grad_space),
                    mem->fparam_space->vals
                    );
            }
            *evals = mem->evals->vals;
            *grads = mem->grad->vals;
        }
        else{
            function_train_core_param_grad_eval(
                ftp->ft, active_core,N,
                x,
                mem->running_evals_lr,
                mem->running_evals_rl,NULL,
                ftp->nparams_per_core[active_core],
                mem->evals->vals, NULL,NULL,0,NULL);

            *evals = mem->evals->vals;
        }
    }
    return 0;
}

double ft_param_grad_sq_norm(void * args, double * grad)
{
    struct PP *                   mem_opts = args;
    struct RegressOpts *          ropts    = mem_opts->opts;
    struct FTParam *              ftp      = mem_opts->ftp;
    double regval;
    if (ropts->alg == AIO){
        double * weights = calloc_double(ftp->dim);
        for (size_t ii = 0; ii < ftp->dim; ii++){
            weights[ii] = 0.5*regopts->regularization_weight;
        }
        regval = function_train_param_grad_sqnorm(ftp->ft,weights,grad);
        free(weights); weights = NULL;
    }
    else if (ropts->alg == ALS){
        double weight = 0.5*regopts->regularization_weight;
        regval = qmarray_param_grad_sqnorm(ftp->ft->cores[ropts->als_active_core],weight,grad);
    }
    else{
        fprintf(stderr, "Unrecognized optimization framework. Needs to be either AIO or ALS\n");
        exit(1);
    }
    return regval;
    
}

    

/***********************************************************//**
    Check if ranks need to be restricted

    \param[in] opts - regression options

    \returns 1 if yes 0 if no
***************************************************************/
int restrict_ranksp(const struct RegressOpts * opts)
{
    // check if ranks need to be restricted
    int restrict_needed = 0;
    for (size_t ii = 0; ii < opts->dim; ii++){
        if (opts->restrict_rank_opt[ii] > 0){
            restrict_needed = 1;
            break;
        }
    }
    return restrict_needed;
}


/***********************************************************//**
    Get parameter values from restricted set of ranks

    \param[in]    regopts        - regression options
    \param[in]     ftp           - parameterized FT
    \param[in]     full_vals     - all parameters 
    \param[in,out] restrict_vals - subset of *full_vals* obtained from univariate functions corresponding
                                   to restricted ranks sepcified in regopts

***************************************************************/
void extract_restricted_vals(const struct RegressOpts * regopts, const struct FTparam * ftp,
                             const double * full_vals, double * restrict_vals)
{
    size_t ind = 0;
    size_t onparam = 0;
    size_t onparam_new = 0;
    size_t * rank_start = regopts->restrict_rank_opt;

    /* printf("restrict ranks = "); */
    /* iprint_sz(regopts->dim,rank_start); */
    for (size_t kk = 0; kk < regopts->dim; kk++){
        for (size_t jj = 0; jj < ftp->ft->ranks[kk+1]; jj++){
            for (size_t ii = 0; ii < ftp->ft->ranks[kk]; ii++){
                size_t np = ftp->nparams_per_uni[ind];
                for (size_t ll = 0; ll < np; ll++){
                    if (kk > 0){
                        if ( (ii >= rank_start[kk-1]) || (jj >= rank_start[kk]) ){
                            restrict_vals[onparam_new] = full_vals[onparam];
                            onparam_new++;
                        }
                    }
                    else{ // kk == 0
                        if (jj >= rank_start[kk]){
                            /* printf("onparam = %zu\n",onparam); */
                            /* printf("onparam_new = %zu\n",onparam_new); */
                            restrict_vals[onparam_new] = full_vals[onparam];
                            onparam_new++;
                        }
                    }
                    onparam+=1;
                }
                ind++;
            }
        }
    }
}



/***********************************************************//**
    Evaluate an objective function, if needed gradient is with respect
    to all parameters.

    \param[in]     ftp     - parameterized FT
    \param[in]     regopts - regression options
    \param[in,out] mem     - memory manager that will store the evaluations
    \param[in]     N       - number of evaluations
    \param[in]     x       - evaluation locations
    \param[in]     y       - evaluation labels
    \param[in,out] grad    - gradient (doesn't evaluate if NULL)

    \returns evaluation
***************************************************************/
double ft_param_eval_objective_aio(struct FTparam * ftp,
                                   struct RegressOpts * regopts,
                                   struct RegressionMemManager * mem,
                                   size_t N,
                                   const double * x,
                                   const double * y,
                                   double * grad)
{

    int alloc_mem=0;
    struct RegressionMemManager * mem_here = mem;
    if (mem == NULL){
        alloc_mem=1;

        enum FTPARAM_ST structure = ft_param_extract_structure(ftp);
        if (regopts->stoch_obj == 1){
            structure = NONE_ST;
        }
        size_t * ranks = function_train_get_ranks(ftp->ft);
        mem_here = regress_mem_manager_alloc(ftp->dim,N,
                                             ftp->nparams_per_core,ranks,
                                             ftp->max_param_uni,structure);
        /* printf("SHOULD NOT BE HERE\n"); */
    }
    int check_mem = regress_mem_manager_enough(mem_here,N);
    if (check_mem == 0){
        assert (1 == 0);
    }

    double out = 0.0;
    /* printf("before calling aio_ls r2 in first core is %zu\n",mem_here->running_grad[0]->r2); */
    out = ft_param_eval_objective_aio_ls(ftp,mem_here,N,x,y,grad);
    /* if (isnan(out)){ */
    /*     fprintf(stderr,"Result from aio_ls is NaN\n"); */
    /*     exit(1); */
    /* } */
    /* if (isinf(out)){ */
    /*     fprintf(stderr,"Result from aio_ls is inf\n"); */
    /*     exit(1); */
    /* } */
    if (regopts->obj == FTLS_SPARSEL2){        
        /* out = ft_param_eval_objective_aio_ls(ftp,mem_here,N,x,y,grad); */

        double * weights = calloc_double(ftp->dim);
        for (size_t ii = 0; ii < ftp->dim; ii++){
            weights[ii] = 0.5*regopts->regularization_weight;
        }
        double regval = function_train_param_grad_sqnorm(ftp->ft,weights,grad);
        out += regval;
        free(weights); weights = NULL;
    }
    

    if (alloc_mem == 1){
        regression_mem_manager_free(mem_here); mem_here = NULL;
    }

    if (isnan(out)){
        fprintf(stderr,"Regress aio eval objective is NaN\n");
        exit(1);
    }
    else if (isinf(out)){
        fprintf(stderr,"Regress aio eval objective is Inf\n");
        exit(1);
    }
    return out;
}


/***********************************************************//**
    Evaluate an objective function, if needed gradient is with respect
    to parameters of the active core, specified in regopts

    \param[in]     ftp     - parameterized FT
    \param[in]     regopts - regression options
    \param[in,out] mem     - memory manager that will store the evaluations
    \param[in]     N       - number of evaluations
    \param[in]     x       - evaluation locations
    \param[in]     y       - evaluation labels
    \param[in,out] grad    - gradient (doesn't evaluate if NULL)

    \returns evaluation
***************************************************************/
double ft_param_eval_objective_als(struct FTparam * ftp,
                                   struct RegressOpts * regopts,
                                   struct RegressionMemManager * mem,
                                   size_t N,
                                   const double * x,
                                   const double * y,
                                   double * grad)
{
    int check_mem = regress_mem_manager_enough(mem,N);
    if (check_mem == 0){
        assert (1 == 0);
    }
    
    double out = 0.0;
    if (regopts->obj == FTLS){
        out = ft_param_eval_objective_als_ls(ftp,regopts->als_active_core,mem,N,x,y,grad);
    }
    else if (regopts->obj == FTLS_SPARSEL2){
        out = ft_param_eval_objective_als_ls(ftp,regopts->als_active_core,mem,N,x,y,grad);
        double weight = 0.5*regopts->regularization_weight;
        double regval = qmarray_param_grad_sqnorm(ftp->ft->cores[regopts->als_active_core],weight,grad);
        out += regval;
    }
    else{
        assert (1 == 0);
    }

    return out;
}


/** \struct PP
 * \brief Store items that will be passed as void to optimizer
 */
struct PP
{
    struct FTparam * ftp;
    struct RegressOpts * opts;
    struct RegressionMemManager * mem;
    size_t N;
    const double * x;
    const double * y;

};



/***********************************************************//**
    General all-at-once regression objective for c3opt optimizer
    ** THIS IS AN INTERFACE FUNCTION FOR C3OPT **

    \param[in]     nparam - number of parameters
    \param[in]     param  - parameter values
    \param[in,out] grad   - gradient (doesn't evaluate if NULL)
    \param[in,out] args   - additional arguments to optimizer

    \returns evaluation
***************************************************************/
double regress_opts_minimize_aio(size_t nparam, const double * param,
                                 double * grad, void * args)
{


    for (size_t ii = 0; ii < nparam; ii++){
        if (isnan(param[ii])){
            fprintf(stderr,"Optimizer requesting params that are NaN\n");
            fprintf(stderr,"param[%zu] = nan\n",ii);
            exit(1);
        }
        else if (isinf(param[ii])){
            fprintf(stderr,"Optimizer requesting params that are inf\n");
            exit(1);
        }
    }

    struct PP * pp = args;

    // check if special structure exists / initialized
    regress_mem_manager_check_structure(pp->mem, pp->ftp,pp->x);
    regress_mem_manager_reset_running(pp->mem);

    double eval = run_ft_param_eval_objective_aio(pp->ftp, pp->opts,pp->mem,nparam,param,pp->N,
                                                  pp->x,pp->y,grad);


    if (isnan(eval)){
        fprintf(stderr,"Regress aio objective is NaN\n");
        exit(1);
    }
    else if (isinf(eval)){
        fprintf(stderr,"Regress aio objective is Inf\n");
        exit(1);
    }



    return eval;
}

/***********************************************************//**
    General all-at-once regression objective for c3opt optimizer
    when stochastic optimization is used 
    ** THIS IS AN INTERFACE FUNCTION FOR C3OPT **

    \param[in]     nparam - number of parameters
    \param[in]     ind    - index of data point
    \param[in]     param  - parameter values
    \param[in,out] grad   - gradient (doesn't evaluate if NULL)
    \param[in,out] args   - additional arguments to optimizer

    \returns evaluation
***************************************************************/
double regress_opts_stoch_minimize_aio(size_t nparam, size_t ind,
                                       const double * param,
                                       double * grad, void * args)
{
    (void)(nparam);

    /* printf("ind = %zu\n",ind); */
    struct PP * pp = args;
    
    if (ind >= pp->N){
        fprintf(stderr, "Stochastic optimization index is greater than the number of data points\n");
        exit(1);
    }

    size_t dim = pp->ftp->dim;

    // check if special structure exists / initialized
    regress_mem_manager_check_structure(pp->mem, pp->ftp,pp->x);
    regress_mem_manager_reset_running(pp->mem);

    double eval = run_ft_param_eval_objective_aio(pp->ftp, pp->opts,pp->mem,nparam,param,1,
                                                  pp->x+ind*dim,pp->y+ind,grad);   

    return eval;
}


/***********************************************************//**
    Run all-at-once regression and return the result

    \param[in,out] ftp       - parameterized function train
    \param[in,out] ropts     - regression options
    \param[in,out] optimizer - optimization arguments
    \param[in]     N         - number of data points
    \param[in]     x         - training samples
    \param[in]     y         - training labels

    \returns function train
***************************************************************/
struct FunctionTrain *
c3_regression_run_aio(struct FTparam * ftp, struct RegressOpts * ropts,
                      struct c3Opt * optimizer,
                      size_t N, const double * x, const double * y)
{

    size_t * ranks = function_train_get_ranks(ftp->ft);

    enum FTPARAM_ST structure = ft_param_extract_structure(ftp);

    if (ropts->stoch_obj == 1){
        structure = NONE_ST; // don't take advante of structure in this case
    }
    struct RegressionMemManager * mem =
        regress_mem_manager_alloc(ftp->dim,N,
                                  ftp->nparams_per_core,ranks,
                                  ftp->max_param_uni,structure);


    // package enough
    struct PP pp;
    pp.ftp = ftp;
    pp.opts = ropts;
    pp.mem = mem;
    pp.N = N;
    pp.x = x;
    pp.y = y;

    size_t nparams;
    double * guess = NULL;
    int restrict_needed = restrict_ranksp(ropts);
    if (restrict_needed == 0){
        nparams = function_train_get_nparams(ftp->ft);
        guess = calloc_double(ftp->nparams);
        memmove(guess,ftp->params,ftp->nparams * sizeof(double));
    }
    else{
        nparams = ft_param_get_nparams_restrict(ftp,ropts->restrict_rank_opt);
        guess = calloc_double(nparams);
        extract_restricted_vals(ropts,ftp,ftp->params,guess);
    }

    for (size_t ii = 0; ii < nparams; ii++){
        if (isnan(guess[ii])){
            fprintf(stderr,"Initial guess for AIO regression is NaN\n");
            fprintf(stderr,"param[%zu] = nan\n",ii);
            exit(1);
        }
        else if (isinf(guess[ii])){
            fprintf(stderr,"Initial guess for AIO regression is inf\n");
            exit(1);
        }
    }


    double val;
    c3opt_set_nvars(optimizer,nparams);

    if (ropts->stoch_obj == 0){
        c3opt_add_objective(optimizer,regress_opts_minimize_aio,&pp);
    }
    else{
        /* printf("adding stochastic minimizer!\n"); */
        c3opt_add_objective_stoch(optimizer,regress_opts_stoch_minimize_aio,&pp);
    }
    
    int res = c3opt_minimize(optimizer,guess,&val);
    if (res < -1){
        fprintf(stderr,"Warning: optimizer exited with code %d\n",res);
    }
    if (ropts->verbose == 1){
        printf("Objective value = %3.5G\n",val);
    }

    if (restrict_needed == 0){
        ft_param_update_params(ftp,guess);        
    }
    else{
        ft_param_update_restricted_ranks(ftp,guess,ropts->restrict_rank_opt);
    }

    struct FunctionTrain * ft_final = function_train_copy(ftp->ft);

    free(guess); guess = NULL;
    regression_mem_manager_free(mem);

    return ft_final;
}


/***********************************************************//**
    General ALS regression objective for c3opt optimizer

    \param[in]     nparam - number of parameters
    \param[in]     params - parameter values
    \param[in,out] grad   - gradient (doesn't evaluate if NULL)
    \param[in,out] args   - additional arguments to optimizer

    \returns evaluation
***************************************************************/
double regress_opts_minimize_als(size_t nparam, const double * params,
                                 double * grad, void * args)
{
    struct PP * pp = args;

    // check if special structure exists / initialized
    regress_mem_manager_check_structure(pp->mem,pp->ftp,pp->x);

    // just need to reset the gradient of the active core
    running_core_total_restart(pp->mem->running_grad[pp->opts->als_active_core]);

    ft_param_update_core_params(pp->ftp,pp->opts->als_active_core,params);
    if (grad != NULL){
        for (size_t ii = 0; ii < nparam; ii++){
            grad[ii] = 0.0;
        }
    }

    return ft_param_eval_objective_als(pp->ftp,pp->opts,pp->mem,pp->N,pp->x,pp->y,grad);
}



/***********************************************************//**
    Run als regression and return the result

    \param[in,out] ftp       - parameterized function train
    \param[in,out] ropts     - regression options
    \param[in,out] optimizer - optimization arguments
    \param[in]     N         - number of data points
    \param[in]     x         - training samples
    \param[in]     y         - training labels

    \returns function train
***************************************************************/
struct FunctionTrain *
c3_regression_run_als(struct FTparam * ftp, struct RegressOpts * ropts, struct c3Opt * optimizer,
                      size_t N, const double * x, const double * y)
{

    enum FTPARAM_ST structure = ft_param_extract_structure(ftp);
    size_t * ranks = function_train_get_ranks(ftp->ft);
    struct RegressionMemManager * mem =
        regress_mem_manager_alloc(ftp->dim,N,
                                  ftp->nparams_per_core,ranks,ftp->max_param_uni,structure);

    for (size_t sweep = 1; sweep <= ropts->max_als_sweeps; sweep++){
        if (ropts->verbose > 0){
            printf("Sweep %zu\n",sweep);
        }
        struct FunctionTrain * start = function_train_copy(ftp->ft);
        for (size_t ii = 0; ii < ftp->dim; ii++){
            if (ropts->verbose > 1){
                printf("\tDim %zu: ",ii);
            }
            regress_mem_manager_reset_running(mem);
            ropts->als_active_core = ii;
            ft_param_prepare_als_core(ftp,ii,mem,N,x);

            /* c3opt_free(ropts->optimizer); ropts->optimizer = NULL; */



            struct PP pp;
            pp.ftp = ftp;
            pp.opts = ropts;
            pp.mem = mem;
            pp.N = N;
            pp.x = x;
            pp.y = y;

            c3opt_set_nvars(optimizer,ftp->nparams_per_core[ii]);

            assert (ropts->stoch_obj == 0);
            c3opt_add_objective(optimizer,regress_opts_minimize_als,&pp);
            
            double * guess = calloc_double(ftp->nparams_per_core[ii]);
            function_train_core_get_params(ftp->ft,ii,guess);
            double val;
            int res = c3opt_minimize(optimizer,guess,&val);
            /* printf("active core = %zu\n",pp.opts->active_core); */
            if (res < -1){
                fprintf(stderr,"Warning: optimizer exited with code %d\n",res);
            }

            if (ropts->verbose > 1){
                printf("\t\tObjVal = %3.5G\n",val);
            }

            ft_param_update_core_params(ftp,ii,guess);
            free(guess); guess = NULL;
        }

        for (size_t jj = 1; jj < ftp->dim-1; jj++){
            size_t ii = ftp->dim-1-jj;
            if (ropts->verbose > 1){
                printf("\tDim %zu: ",ii);
            }
            
            regress_mem_manager_reset_running(mem);
            ropts->als_active_core = ii;
            ft_param_prepare_als_core(ftp,ii,mem,N,x);


            struct PP pp;
            pp.ftp = ftp;
            pp.opts = ropts;
            pp.mem = mem;
            pp.N = N;
            pp.x = x;
            pp.y = y;


            c3opt_set_nvars(optimizer,ftp->nparams_per_core[ii]);
            c3opt_add_objective(optimizer,regress_opts_minimize_als,&pp);

            double * guess = calloc_double(ftp->nparams_per_core[ii]);
            function_train_core_get_params(ftp->ft,ii,guess);
            double val;
            int res = c3opt_minimize(optimizer,guess,&val);
            /* printf("active core = %zu\n",pp.opts->active_core); */
            if (res < -1){
                fprintf(stderr,"Warning: optimizer exited with code %d\n",res);
            }

            if (ropts->verbose > 1){
                printf("\t\tObjVal = %3.5G\n",val);
            }

            ft_param_update_core_params(ftp,ii,guess);
            free(guess); guess = NULL;
        }
        
        double diff = function_train_norm2diff(ftp->ft,start);
        double no = function_train_norm2(ftp->ft);
        if (ropts->verbose > 0){
            printf("\n\t ||f||=%G, ||f-f_p||=%G ||f-f_p||/||f||=%G\n",no,diff,diff/no);
        }

        function_train_free(start); start = NULL;
        if ( (diff / no) < ropts->als_conv_tol){
            break;
        }
    }


    struct FunctionTrain * ft_final = function_train_copy(ftp->ft);
    regression_mem_manager_free(mem); mem = NULL;
    return ft_final;
}


/***********************************************************//**
    Run regression and return the result

    \param[in,out] ftp       - parameterized function train
    \param[in,out] regopts   - regression options
    \param[in,out] optimizer - optimization arguments
    \param[in]     N         - number of data points
    \param[in]     x         - training samples
    \param[in]     y         - training labels

    \returns function train
***************************************************************/
struct FunctionTrain *
c3_regression_run(struct FTparam * ftp, struct RegressOpts * regopts, struct c3Opt * optimizer,
                  size_t N, const double * x, const double * y)
{
    // check if special structure exists / initialized
    
    struct FunctionTrain * ft = NULL;
    if (regopts->type == AIO){
        ft = c3_regression_run_aio(ftp,regopts,optimizer,N,x,y);
    }
    else if (regopts->type == ALS){
        ft = c3_regression_run_als(ftp,regopts,optimizer,N,x,y);
    }
    else{
        printf("Regress opts of type %d is unknown\n",regopts->type);
        printf("                     AIO is %d \n",AIO);
        printf("                     ALS is %d \n",AIO);
        exit(1);
    }

    return ft;
}


/* //////////////////////////////////// */
/* //////////////////////////////////// */
/* //////////////////////////////////// */
/* //////////////////////////////////// */
/* // Common Interface */
/* //////////////////////////////////// */
/* //////////////////////////////////// */
/* //////////////////////////////////// */
/* //////////////////////////////////// */


/** \struct FTRegress
 * \brief Interface to regression
 * \var FTRegress::type
 * algorithm for regression (ALS,AIO)
 * \var FTRegress::obj
 * objective function (FTLS, FTLS_SPARSEL2)
 * \var FTRegress::dim
 * dimension of the feature space
 * \var FTRegress::approx_opts
 * approximation options
 * \var FTRegress::ftp
 * parameterized function train
 * \var FTRegress::regopts
 * regression options
 */ 
struct FTRegress
{
    size_t type;
    size_t obj;
    size_t dim;

    struct MultiApproxOpts* approx_opts;
    struct FTparam* ftp;
    struct RegressOpts* regopts;    


    int adapt;
    size_t kickrank;
    size_t maxrank;
    double roundtol;
    size_t kfold;
    int finalize;
    int opt_restricted;

};

/***********************************************************//**
    Allocate function train regression structure

    \param[in] dim   - dimension of feature space
    \param[in] aopts - approximation options
    \param[in] ranks - rank of approxiamtion

    \returns regression structure
***************************************************************/
struct FTRegress *
ft_regress_alloc(size_t dim, struct MultiApproxOpts * aopts, size_t * ranks)
{
    struct FTRegress * ftr = malloc(sizeof(struct FTRegress));
    if (ftr == NULL){
        fprintf(stderr, "Cannot allocate FTRegress structure\n");
        exit(1);
    }
    ftr->type = REGNONE;
    ftr->obj = REGOBJNONE;
    ftr->dim = dim;

    ftr->approx_opts = aopts;
    ftr->ftp = ft_param_alloc(dim,aopts,NULL,ranks);
    ftr->regopts = NULL;

    ftr->adapt = 0;
    ftr->kickrank = 1;
    ftr->maxrank = 10;
    ftr->roundtol = 1e-8;
    ftr->kfold = 3;
    ftr->finalize = 1;
    ftr->opt_restricted = 0;

    return ftr;
}

/***********************************************************//**
    Get dimension
    
    \param[in] ftr - regression structure
    \returns dimension
***************************************************************/
size_t ft_regress_get_dim(const struct FTRegress * ftr)
{
    assert (ftr != NULL);
    return ftr->dim;
}


/***********************************************************//**
    Turn on/off adaptation
    
    \param[in,out] ftr - regression structure
    \param[in]     on  - 1 for yes, 0 for no
***************************************************************/
void ft_regress_set_adapt(struct FTRegress * ftr, int on)
{
    assert (ftr != NULL);
    ftr->adapt = on;
}

/***********************************************************//**
    Specify the maximum rank allowable within adaptation
    
    \param[in,out] ftr      - regression structure
    \param[in]     maxrank  - maxrank
***************************************************************/
void ft_regress_set_maxrank(struct FTRegress * ftr, size_t maxrank)
{
    assert (ftr != NULL);
    ftr->maxrank = maxrank;
}


/***********************************************************//**
    Specify the rank increase parameter
    
    \param[in,out] ftr      - regression structure
    \param[in]     kickrank - rank increase parameter
***************************************************************/
void ft_regress_set_kickrank(struct FTRegress * ftr, size_t kickrank)
{
    assert (ftr != NULL);
    ftr->kickrank = kickrank;
}

/***********************************************************//**
    Specify the rounding tolerance
    
    \param[in,out] ftr - regression structure
    \param[in]     tol - rounding tolerance
***************************************************************/
void ft_regress_set_roundtol(struct FTRegress * ftr, double tol)
{
    assert (ftr != NULL);
    ftr->roundtol = tol;
}

/***********************************************************//**
    Specify the cross validation number within rank adaptation
    
    \param[in,out] ftr   - regression structure
    \param[in]     kfold - number of cv sets
***************************************************************/
void ft_regress_set_kfold(struct FTRegress * ftr, size_t kfold)
{
    assert (ftr != NULL);
    ftr->kfold = kfold;
}

/***********************************************************//**
    Specify whether or not to finalize an approximation 
    after rank adaptation after converging by reapproximating
    
    \param[in,out] ftr - regression structure
    \param[in]     fin - 1 finalize 0 dont
***************************************************************/
void ft_regress_set_finalize(struct FTRegress * ftr, int fin)
{
    assert (ftr != NULL);
    ftr->finalize = fin;
}

/***********************************************************//**
    Specify whether or not to only optimize on a restricted set of ranks
    after increaseing ranks in adaptation
    
    \param[in,out] ftr - regression structure
    \param[in]     res - 0 dont, 1 do
***************************************************************/
void ft_regress_set_opt_restrict(struct FTRegress * ftr, int res)
{
    assert (ftr != NULL);
    ftr->opt_restricted = res;
}

/***********************************************************//**
    Free memory allocated to FT regress structure
    
    \param[in,out] ftr - regression structure
.***************************************************************/
void ft_regress_free(struct FTRegress * ftr)
{
    if (ftr != NULL){
        ft_param_free(ftr->ftp); ftr->ftp = NULL;
        regress_opts_free(ftr->regopts); ftr->regopts = NULL;
        free(ftr); ftr = NULL;
    }
}

/***********************************************************//**
    Clear memory, keep only dim

    \param[in,out] ftr   - regression structuture
    \param[in]     aopts - approximation options
    \param[in]     ranks - FT ranks
***************************************************************/
void ft_regress_reset_param(struct FTRegress* ftr, struct MultiApproxOpts * aopts, size_t * ranks)
{

    ftr->approx_opts = aopts;
    ft_param_free(ftr->ftp); ftr->ftp = NULL;
    ftr->ftp = ft_param_alloc(ftr->dim,aopts,NULL,ranks);;
}


/***********************************************************//**
    Set regression algorithm type

    \param[in,out] ftr  - regression structuture
    \param[in]     type - ALS or AIO
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

    \param[in,out] ftr - regression structuture
    \param[in]     obj - FTLS or FTLS_SPARSEL2
***************************************************************/
void ft_regress_set_obj(struct FTRegress * ftr, enum REGOBJ obj)
{
    assert (ftr != NULL);
    if (ftr->obj != REGOBJNONE){
        fprintf(stdout,"Warning: respecifying regression objective function may lead to memory problems");
        fprintf(stdout,"Use function ft_regress_reset_obj instead\n");
    }
    ftr->obj = obj;
    
    if ((obj != FTLS) && (obj != FTLS_SPARSEL2)){
        fprintf(stderr,"Error: Regularization type %d not available. Options include\n",obj);
        fprintf(stderr,"       FTLS          = 0\n");
        fprintf(stderr,"       FTLS_SPARSEL2 = 1\n");
        exit(1);
    }
}

/***********************************************************//**
    Set regression algorithm and objective

    \param[in,out] ftr  - regression structuture
    \param[in]     type - ALS or AIO
    \param[in]     obj  - FTLS or FTLS_SPARSEL2
***************************************************************/
void ft_regress_set_alg_and_obj(struct FTRegress * ftr, enum REGTYPE type, enum REGOBJ obj)
{
    ft_regress_set_type(ftr,type);
    ft_regress_set_obj(ftr,obj);
    if (ftr->regopts != NULL){
        regress_opts_free(ftr->regopts);
    }
    ftr->regopts = regress_opts_create(ftr->dim,type,obj);   
}

/***********************************************************//**
    Set option specifying will use a stochastic optimizer
    
    \param[in,out] ftr - regression structure
    \param[in]     on  - 1 for yes, 0 for no
***************************************************************/
void ft_regress_set_stoch_obj(struct FTRegress * ftr, int on)
{
    assert (ftr != NULL);
    assert (ftr->regopts != NULL);
    regress_opts_set_stoch_obj(ftr->regopts,on);
}


/***********************************************************//**
    Get the parameters of the underlying FT

    \param[in]     ftr    - regression structuture
    \param[in,out] nparam - reference to parameters (could be NULL)

    \returns parameters
***************************************************************/
double * ft_regress_get_params(struct FTRegress * ftr, size_t * nparam)
{
    *nparam = ftr->ftp->nparams;
    double * param = calloc_double(*nparam);
    function_train_get_params(ftr->ftp->ft,param);
    return param;
}

/***********************************************************//**
    Update parameters

    \param[in,out] ftr   - regression structuture
    \param[in]     param - parameters
***************************************************************/
void ft_regress_update_params(struct FTRegress * ftr, const double * param)
{
    assert (ftr != NULL);
    assert (ftr->ftp != NULL);

    ft_param_update_params(ftr->ftp,param);
}


/***********************************************************//**
    Set ALS convergence tolerance

    \param[in,out] opts - regression structuture
    \param[in]     tol  - convergence tolerance
***************************************************************/
void ft_regress_set_als_conv_tol(struct FTRegress * opts, double tol)
{
    assert (opts != NULL);
    assert (opts->regopts != NULL);
    regress_opts_set_als_conv_tol(opts->regopts,tol);
}

/***********************************************************//**
    Set maximum number of als sweeps

    \param[in,out] opts      - regression structuture
    \param[in]     maxsweeps - maximum number of sweeps
***************************************************************/
void ft_regress_set_max_als_sweep(struct FTRegress * opts, size_t maxsweeps)
{
    assert (opts != NULL);
    assert (opts->regopts != NULL);
    regress_opts_set_max_als_sweep(opts->regopts,maxsweeps);
}

/***********************************************************//**
    Set verbosity level

    \param[in,out] opts    - regression structuture
    \param[in]     verbose - maximum number of sweeps
***************************************************************/
void ft_regress_set_verbose(struct FTRegress * opts, int verbose)
{
    assert (opts != NULL);
    assert (opts->regopts != NULL);
    regress_opts_set_verbose(opts->regopts,verbose);
}

/***********************************************************//**
    Set regularization weight

    \param[in,out] opts   - regression structuture
    \param[in]     weight - weight
***************************************************************/
void ft_regress_set_regularization_weight(struct FTRegress * opts, double weight)
{
    assert (opts != NULL);
    assert (opts->regopts != NULL);    
    regress_opts_set_regularization_weight(opts->regopts,weight);
}

/***********************************************************//**
    Get regularization weight

    \param[in] opts - regression structuture

    \returns weight
***************************************************************/
double ft_regress_get_regularization_weight(const struct FTRegress * opts)
{
    assert (opts != NULL);
    assert (opts->regopts != NULL);    
    return regress_opts_get_regularization_weight(opts->regopts);
}

/***********************************************************//**
    Run regression and return the result

    \param[in,out] ftr       - regression parameters
    \param[in,out] optimizer - optimization arguments
    \param[in]     N         - number of data points
    \param[in]     x         - training samples
    \param[in]     y         - training labels

    \returns function train
***************************************************************/
struct FunctionTrain *
ft_regress_run(struct FTRegress * ftr, struct c3Opt * optimizer,
               size_t N, const double * x, const double * y)
{
    assert (ftr != NULL);
    assert (ftr->ftp != NULL);
    assert (ftr->regopts != NULL);
    double param_norm = cblas_ddot(ftr->ftp->nparams,ftr->ftp->params,1,ftr->ftp->params,1);
    if (fabs(param_norm) <= 1e-15){
        /* double yavg = 0.0; for (size_t ii = 0; ii < N; ii++){ yavg += y[ii];} */
        /* yavg /= (double) N; */

        ft_param_create_from_lin_ls(ftr->ftp,N,x,y,1e-3);

        
        /* ft_param_create_from_constant */


        /* fprintf(stderr, "Cannot run ft_regression with zeroed parameters\n"); */
        /* exit(1); */
    }
    struct FunctionTrain * ft = NULL;
    if (ftr->adapt == 1){
        ft = ft_regress_run_rankadapt(ftr,ftr->roundtol,ftr->maxrank,ftr->kickrank,
                                      optimizer,ftr->opt_restricted,N,x,y,ftr->finalize);
    }
    else{

        ft = c3_regression_run(ftr->ftp,ftr->regopts,optimizer,N,x,y);
    }
    return ft;
}


/***********************************************************//**
    Run regression with rank adaptation and return the result

    \param[in,out] ftr                 - regression parameters
    \param[in]     tol                 - rounding tolerance
    \param[in]     maxrank             - upper bound on rank
    \param[in]     kickrank            - size of rank increase
    \param[in,out] optimizer           - optimization arguments
    \param[in]     opt_only_restricted - optimize on a restricted set of ranks
    \param[in]     N                   - number of data points
    \param[in]     x                   - training samples
    \param[in]     y                   - training labels
    \param[in]     finalize            - specify whether or not to reapproximate at the end

    \returns function train

    \note when optimizing over restricted set of ranks, only newly added
    univariate functions from the kicking procedure are optimized over
***************************************************************/
struct FunctionTrain *
ft_regress_run_rankadapt(struct FTRegress * ftr,
                         double tol, size_t maxrank, size_t kickrank,
                         struct c3Opt * optimizer, int opt_only_restricted,
                         size_t N, const double * x, const double * y,
                         int finalize)
{
    assert (ftr != NULL);
    assert (ftr->ftp != NULL);
    assert (ftr->regopts != NULL);

    ftr->adapt = 0; // turn off adaptation since in here!
    size_t * ranks = calloc_size_t(ftr->dim+1);

    size_t kfold = 5;
    int cvverbose = 0;
    struct CrossValidate * cv = cross_validate_init(N,ftr->dim,x,y,kfold,cvverbose);


    if (ftr->regopts->verbose > 0){
        printf("run initial  cv\n");
    }

    double err = cross_validate_run(cv,ftr,optimizer);

    if (ftr->regopts->verbose > 0){
        printf("Initial CV Error: %G\n",err);
    }

    struct FunctionTrain * ft = ft_regress_run(ftr,optimizer,N,x,y);
    size_t * ftranks = function_train_get_ranks(ft);
    memmove(ranks,ftranks,(ftr->dim+1)*sizeof(size_t));
    if (ftr->regopts->verbose > 0){
        printf("Initial ranks: "); iprint_sz(ftr->dim+1,ftranks);
    }

   
    int kicked = 1;
    /* c3opt_set_gtol(optimizer,1e-7); */
    /* c3opt_set_relftol(optimizer,1e-7); */
    size_t maxiter = 10;
    struct FunctionTrain * ftround = NULL;
    double * init_new = NULL;
    double * rounded_params = NULL;
    /* int opt_only_restricted = 0; */
    for (size_t jj = 0; jj < maxiter; jj++){
        if (ftr->regopts->verbose > 0){
            printf("\n");
        }
        ftround = function_train_round(ft,tol,ftr->ftp->approx_opts);
        size_t * ftr_ranks = function_train_get_ranks(ftround);

        if (ftr->regopts->verbose > 0){
            printf("Rounded ranks: "); iprint_sz(ftr->dim+1,ftr_ranks);
        }
        
        kicked = 0;

        for (size_t ii = 1; ii < ft->dim; ii++){
            if (ftr_ranks[ii] == ranks[ii]){
                ftr->regopts->restrict_rank_opt[ii-1] = ranks[ii];
                ranks[ii] += kickrank;
                if (ranks[ii] > maxrank){
                    ranks[ii] = maxrank;
                }
                kicked = 1;
            }
        }

        int allmax = 1;
        for (size_t ii = 1; ii < ft->dim;ii++){
            if (ranks[ii] < maxrank){
                allmax = 0;                
                break;
            }
        }

        /* printf("kicked == %zu\n",kicked); */
        if ((kicked == 0) || (allmax == 1)){
            function_train_free(ftround); ftround = NULL;
            free(ranks); ranks = NULL;
            break;
        }
        size_t nparams_rounded = function_train_get_nparams(ftround);
        if (ftr->regopts->verbose > 0){
            printf("Kicked ranks: "); iprint_sz(ftr->dim+1,ranks);
            if (opt_only_restricted == 1){
                printf("restrict optimization to >=: "); iprint_sz(ft->dim-1,ftr->regopts->restrict_rank_opt);
            }
            printf("Nrounded params: %zu\n",nparams_rounded);
        }


        rounded_params = calloc_double(nparams_rounded);
        function_train_get_params(ftround,rounded_params);

        ft_regress_reset_param(ftr,ftr->ftp->approx_opts,ranks);

        if (opt_only_restricted == 1){
            ft_param_update_inside_restricted_ranks(ftr->ftp,rounded_params,
                                                    ftr->regopts->restrict_rank_opt);
            size_t nnew_solve = ft_param_get_nparams_restrict(ftr->ftp,ftr->regopts->restrict_rank_opt);
            /* printf("Number of new unknowns = %zu\n",nnew_solve); */
            init_new = calloc_double(nnew_solve);
            for (size_t ii = 0; ii < nnew_solve; ii++){
                init_new[ii] = 1e-4;
            }
            ft_param_update_restricted_ranks(ftr->ftp,init_new,ftr->regopts->restrict_rank_opt);
        }
        else{
            for (size_t ii = 0; ii < ftr->dim; ii++){
                ftr->regopts->restrict_rank_opt[ii] = 0.0;
            }
        }

        function_train_free(ft); ft = NULL;
        double new_err = cross_validate_run(cv,ftr,optimizer);
        if (ftr->regopts->verbose > 0){
            printf("CV Error: %G\n",new_err);
        }
        if (new_err > err){
            if (ftr->regopts->verbose > 0){
                printf("Cross validation larger than previous rank, so not keeping this run\n");
                iprint_sz(ftr->dim+1,ftr_ranks);
            }
            ft_regress_reset_param(ftr,ftr->ftp->approx_opts,ftr_ranks);
            ft_param_update_params(ftr->ftp,rounded_params);
            ft = function_train_copy(ftr->ftp->ft);
            function_train_free(ftround); ftround = NULL;
            break;
        }
        else{
            ft = ft_regress_run(ftr,optimizer,N,x,y);
            err = new_err;
        }

        function_train_free(ftround); ftround = NULL;
        free(rounded_params); rounded_params = NULL;
        free(init_new); init_new = NULL;
    }
    function_train_free(ftround); ftround = NULL;
    free(rounded_params); rounded_params = NULL;
    free(init_new); init_new = NULL;

    if (finalize == 1){
        for (size_t ii = 0; ii < ftr->dim; ii++){
            ftr->regopts->restrict_rank_opt[ii] = 0;
        }
        function_train_free(ft); ft = NULL;
        if (ftr->regopts->verbose > 0){
            printf("Final run\n");
        }
        /* c3opt_set_gtol(optimizer,1e-10); */
        /* c3opt_set_relftol(optimizer,1e-10); */
        ft = ft_regress_run(ftr,optimizer,N,x,y);
    }

    cross_validate_free(cv); cv = NULL;
    free(ranks); ranks = NULL;
    ftr->adapt = 1; // turn adaptation back on
    return ft;
}

/* //////////////////////////////////// */
/* //////////////////////////////////// */
/* //////////////////////////////////// */
/* //////////////////////////////////// */
/* // Cross validation */
/* //////////////////////////////////// */
/* //////////////////////////////////// */
/* //////////////////////////////////// */
/* //////////////////////////////////// */

/** \struct CrossValidate
 * \brief Used to perform cross validation
 * \var CrossValidate::N
 * Number of training samples
 * \var CrossValidate::x
 * Features
 * \var CrossValidate::y
 * labels
 * \var CrossValidate::kfold
 * number of folds for cross validation
 * \var CrossValidate::verbose
 * verbosity level
 */ 
struct CrossValidate
{
    size_t N;
    size_t dim;
    const double * x;
    const double * y;
    size_t kfold;
    int verbose;
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
    cv->verbose = 0;
    
    return cv;
}


/***********************************************************//**
   Free cross validation struct
***************************************************************/
void cross_validate_free(struct CrossValidate * cv)
{
    if (cv != NULL){        
        free(cv); cv = NULL;
    }
}

/***********************************************************//**
   Initialize cross validation

   \param[in] N       - number of training samples
   \param[in] dim     - dimension of feature space
   \param[in] x       - features as a flattened vector
   \param[in] y       - labels
   \param[in] kfold   - number of folds in CV
   \param[in] verbose - verbosity level

   \return Cross validation structure
***************************************************************/
struct CrossValidate * cross_validate_init(size_t N, size_t dim,
                                           const double * x,
                                           const double * y,
                                           size_t kfold, int verbose)
{
    struct CrossValidate * cv = cross_validate_alloc();
    cv->N = N;
    cv->dim = dim;
    cv->x = x;
    cv->y = y;
    cv->kfold = kfold;
    cv->verbose = verbose;
    
    return cv;
}

/***********************************************************//**
   Cross validation separate data

   \param[in]     cv          - CV structure
   \param[in]     start       - index specifying the separation
   \param[in]     num_extract - number of data to extract after *start* index
   \param[in,out] xtest       - pointer to an allocated array for storing x[:start,:], y[:start]
   \param[in,out] ytest       - point to an array for storing lables up to *start* index
   \param[in,out] xtrain      - pointer to an array for storing features after start upto start+num_extract
   \param[in,out] ytrain      - pointer to an array for storing labels after start upto start+num_extract
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

   \param[in] cv        - cross validation structure
   \param[in] reg       - regression options
   \param[in] optimizer - optimizer

   \note
   FTregress parameters do not change as a result of running
   cross validation. At the end, they are set to what they are 
   upon input
***************************************************************/
double cross_validate_run(struct CrossValidate * cv,
                          struct FTRegress * reg,
                          struct c3Opt * optimizer)
{

    size_t batch = cv->N / cv->kfold;

    if (cv->verbose > 0){
        printf("\n Running %zu-fold Cross Validation:\n\n",cv->kfold);
    }

    double err = 0.0;
    size_t start_num = 0;
    double * xtrain = calloc_double(cv->N * cv->dim);
    double * ytrain = calloc_double(cv->N);
    double * xtest = calloc_double(cv->N * cv->dim);
    double * ytest = calloc_double(cv->N);

    double erri = 0.0;
    double norm = 0.0;

    // keep the same starting parameters for each cv;
    double * params = calloc_double(reg->ftp->nparams);
    memmove(params,reg->ftp->params, reg->ftp->nparams * sizeof(double));
    for (size_t ii = 0; ii < cv->kfold; ii++){

        if (ii == cv->kfold-1){
            batch = cv->N - start_num;
        }

        if (cv->verbose > 2){
            printf("\t On fold %zu, batch size = %zu\n",ii,batch);
        }
        
        extract_data(cv, start_num, batch, &xtest, &ytest,
                     &xtrain, &ytrain);

        // train with all but *ii*th batch
        /* printf("ytrain = ");dprint(cv->N-batch,ytrain); */



        // update data and reprep memmory
        size_t ntrain = cv->N-batch;
        /* printf("ntrain = %zu\n",ntrain); */
        /* dprint2d(ntrain,cv->dim,xtrain); */

        struct FunctionTrain * ft = ft_regress_run(reg,optimizer,ntrain,xtrain,ytrain);
        
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
        function_train_free(ft); ft = NULL;
        if (cv->verbose > 2){
            printf("\t Error on batch = %G, norm of batch = %G\n",newerr/(double)batch, newnorm/(double)batch);
        }

        erri += newerr;
        norm += newnorm;
        start_num += batch;

        // reset parameters
        ft_regress_update_params(reg,params);
    }
    if (cv->verbose > 1){
        printf("\t CV Err = %G, norm  = %G, relative_err = %G\n",erri/cv->N,norm/cv->N,erri/norm);
    }
    err = erri/norm;
    // do last back

    free(xtrain); xtrain = NULL;
    free(ytrain); ytrain = NULL;
    free(xtest); xtest = NULL;
    free(ytest); ytest = NULL;
    free(params); params = NULL;

    return err;
}


// Set up optimization of cross validation error

enum RDTYPE {RPUINT, RPDBL, RPINT};

#define NRPARAM 4
static char reg_param_names[NRPARAM][30] = {"rank","num_param","opt_maxiter","reg_weight"};
static enum RDTYPE reg_param_types[NRPARAM]={RPUINT,RPUINT,RPUINT,RPDBL};

int get_reg_ind(char * name)
{
    int out = -1;
    for (int ii = 0; ii < NRPARAM; ii++){
        if (strcmp(name,reg_param_names[ii]) == 0){
            out = ii;
            break;
        }
    }
    return out;
}


struct RegParameter
{
    char name[30];
    enum RDTYPE type;
    int ival;
    size_t uval;
    double dval;
};

void reg_parameter_print(struct RegParameter * p,char * str)
{
    /* printf("ii=%zu, name=%s\n",ii,cv->name[ii]); */
    if (strcmp(p->name,"reg_weight")==0){
        sprintf(str,"%s=%3.5G ",p->name,p->dval);
    }
    else{
        sprintf(str,"%s=%zu ",p->name,p->uval);
    }
}

/***********************************************************//**
    Allocate parameter
***************************************************************/
struct RegParameter * reg_parameter_alloc(char * name, void * val)
{
    struct RegParameter * rp = malloc(sizeof(struct RegParameter));
    if (rp == NULL){
        fprintf(stderr, "Cannot allocate RegParameter structure\n");
        exit(1);
    }
    strcpy(rp->name,name);

    int reg = get_reg_ind(name);
    if (reg == -1){
        fprintf(stderr,"Regression parameter %s unknown\n",name);
        fprintf(stderr,"Options are: \n");
        for (size_t ii = 0; ii < NRPARAM; ii++){
            fprintf(stderr,"\t %s\n",reg_param_names[ii]);
        }
        exit(1);
    }
    
    rp->type = reg_param_types[reg];
    if (rp->type == RPUINT){
        rp->uval = *(size_t *)val;
    }
    else if (rp->type == RPDBL){
        rp->dval = *(double *)val;
    }
    else if (rp->type == RPINT){
        rp->ival = *(int *)val;
    }
    else{
        fprintf(stderr,"Error processing parameter %s\n",name);
        exit(1);
    }
    return rp;
}


/***********************************************************//**
    Allocate parameter
***************************************************************/
struct RegParameter ** reg_parameter_array_alloc(size_t N)
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
void reg_parameter_free(struct RegParameter * rp)
{
    if (rp != NULL){
        free(rp); rp = NULL;
    }
}
    
struct CVCase
{
    size_t nparam;
    struct RegParameter ** params;

    size_t onparam;
};

struct CVCase * cv_case_alloc(size_t nparam)
{
    struct CVCase * cvc = malloc(sizeof(struct CVCase));
    if (cvc == NULL){
        fprintf(stderr,"Cannot allocate CV case\n");
        exit(1);
    }

    cvc->nparam = nparam;
    cvc->params = reg_parameter_array_alloc(nparam);
    assert (cvc->params != NULL);
    cvc->onparam = 0;

    return cvc;
}


void cv_case_process(struct CVCase * cv, struct FTRegress * ftr, struct c3Opt * optimizer)
{
    int up_reg_weight = 0;
    double nrw = 0;

    int up_rank = 0;
    size_t rank = 0;

    int up_maxiter = 0;
    size_t maxiter = 0;

    int up_num_param = 0;
    size_t num_param = 0;

    struct FTparam * ftp = ftr->ftp;
        
    for (size_t ii = 0; ii < cv->onparam; ii++){
        if (strcmp(cv->params[ii]->name,"rank") == 0){
            up_rank = 1;
            rank = cv->params[ii]->uval;
        }
        else if(strcmp(cv->params[ii]->name,"num_param") == 0){
            up_num_param = 1;
            num_param = cv->params[ii]->uval;
        }
        else if (strcmp(cv->params[ii]->name,"reg_weight") == 0){
            up_reg_weight = 1;
            nrw  = cv->params[ii]->dval;
        }
        else if (strcmp(cv->params[ii]->name,"opt_maxiter") == 0){
            up_maxiter = 1;
            maxiter = cv->params[ii]->uval;
        }
    }

    size_t * old_ranks = function_train_get_ranks(ftp->ft);
    size_t * ranks = calloc_size_t(ftp->dim+1);
    memmove(ranks,old_ranks, (ftp->dim+1)*sizeof(size_t));

    if (up_rank == 1){
        ranks[0] = 1;
        ranks[ftp->dim] = 1;
        for (size_t ii = 1; ii < ftp->dim; ii++){
            ranks[ii] = rank;
        }
    }
    if (up_num_param == 1){
        for (size_t ii = 0; ii < ftp->dim; ii++){
            multi_approx_opts_set_dim_nparams(ftp->approx_opts,ii,num_param);
        }
    }
    
    // always reset!!
    ft_regress_reset_param(ftr,ftp->approx_opts,ranks);
    free(ranks); ranks = NULL;
    
    if (up_reg_weight == 1){
        ft_regress_set_regularization_weight(ftr,nrw);
    }

    if (up_maxiter == 1){
        c3opt_set_maxiter(optimizer,maxiter);
    }
}


void cv_case_add_param(struct CVCase * cv, char * name, void * val)
{
    if (cv->onparam == cv->nparam){
        fprintf(stderr,"Adding too many parameters to cv_case\n");
        exit(1);
    }
    cv->params[cv->onparam] = reg_parameter_alloc(name,val);
    cv->onparam++;
}

void cv_case_string(struct CVCase * cv, char * str, int sbytes)
{
    int start = 0;
    int used = 0;
    for (size_t ii = 0; ii < cv->nparam; ii++){
        char strr[36];
        reg_parameter_print(cv->params[ii],strr);

        used = snprintf(str+start,sbytes-start,"%s ",strr);

        start += used;
    }
}

void cv_case_free(struct CVCase * cv)
{
    if (cv != NULL){
        for (size_t ii = 0; ii < cv->nparam;ii++){
            reg_parameter_free(cv->params[ii]); cv->params[ii] = NULL;
        }
        free(cv->params); cv->params = NULL;
        free(cv); cv = NULL;
    }
}

struct CVCList
{
    struct CVCase * cv;
    struct CVCList * next;
};

struct CVCList * cvc_list_alloc()
{
    struct CVCList * cvl = malloc(sizeof(struct CVCList));
    if (cvl == NULL){
        fprintf(stderr,"Cannot allocate CVCase List\n");
        exit(1);
    }
    cvl->next = NULL;

    return cvl;
}

void cvc_list_push(struct CVCList ** list, size_t nparam)
{
    struct CVCList * newList = cvc_list_alloc();
    newList->cv = cv_case_alloc(nparam);
    newList->next = *list;
    *list = newList;
}

void cvc_list_update_case(struct CVCList * list, char * name, void * val)
{
    cv_case_add_param(list->cv,name,val);
}

void cvc_list_print(struct CVCList * list, FILE * fp)
{
    struct CVCList * temp = list;
    while (temp != NULL){
        char * ss = malloc(256 * sizeof(char));
        cv_case_string(temp->cv,ss,256);
        fprintf(fp,"%s\n",ss);
        /* fprintf(fp,"%zu\n",temp->cv->nparam); */
        temp = temp->next;
        free(ss); ss = NULL;
    }
}

void cvc_list_free(struct CVCList * cv)
{
    if (cv != NULL){
        cv_case_free(cv->cv);
        cvc_list_free(cv->next);
        free(cv); cv = NULL;
    }
}


struct CVOptGrid
{
    size_t ncvparam;

    char names[NRPARAM][32];
    enum RDTYPE * types;
    size_t * nvals;
    void **params;
    size_t onparam;

    int verbose;
};

struct CVOptGrid * cv_opt_grid_init(size_t ncvparam)
{
    struct CVOptGrid * cvg = malloc(sizeof(struct CVOptGrid));
    if (cvg == NULL){
        fprintf(stderr, "Failure to allocate CVOptGrid\n");
        exit(1);
    }

    cvg->ncvparam = ncvparam;
    cvg->types = malloc(ncvparam * sizeof(enum RDTYPE));
    if (cvg->types == NULL){
        fprintf(stderr, "Failure to allocate CVOptGrid\n");
        exit(1);
    }
    cvg->nvals = calloc_size_t(ncvparam);
    cvg->params = malloc(ncvparam * sizeof(void *));
    if (cvg->params == NULL){
        fprintf(stderr, "Failure to allocate CVOptGrid\n");
        exit(1);
    }
    for (size_t ii = 0; ii < ncvparam; ii++){
        cvg->params[ii] = NULL;
    }
    cvg->onparam = 0;

    cvg->verbose = 0;
    return cvg;
}

void cv_opt_grid_free(struct CVOptGrid * cv)
{
    if (cv != NULL){
        for (size_t ii = 0; ii < cv->ncvparam; ii++){
            free(cv->params[ii]); cv->params[ii] = NULL;
        }
        free(cv->params); cv->params = NULL;
        free(cv->types); cv->types = NULL;
        free(cv->nvals); cv->nvals = NULL;
        free(cv); cv= NULL;
    }
}

void cv_opt_grid_set_verbose(struct CVOptGrid * cv, int verbose)
{
    assert (cv != NULL);
    cv->verbose = verbose;
}


int cv_name_exists(struct CVOptGrid * cvg, char * name)
{
    int exists = -1;
    for (size_t ii = 0; ii < cvg->onparam; ii++){
        if (strcmp(cvg->names[ii],name) == 0){
            exists = ii;
            break;
        }
    }
    return exists;
}

void cv_opt_grid_add_param(struct CVOptGrid * cv, char * name, size_t nvals, void * vals)
{
    int ind = get_reg_ind(name);
    enum RDTYPE type = reg_param_types[ind];

    if (cv->onparam == cv->ncvparam){
        fprintf(stderr,"Cannot add another parameter to cross validation\n");
        exit(1);
    }
    strcpy(cv->names[cv->onparam],name);
    cv->types[cv->onparam] = type;
    cv->nvals[cv->onparam] = nvals;
    if (type == RPUINT)
    {
        cv->params[cv->onparam] = calloc_size_t(nvals);
        memmove(cv->params[cv->onparam],(size_t *)vals, nvals * sizeof(size_t));
    }
    else if (type == RPDBL)
    {
        cv->params[cv->onparam] = calloc_double(nvals);
        memmove(cv->params[cv->onparam],(double *)vals, nvals * sizeof(double));   
    }
    else if (type == RPINT)
    {
        cv->params[cv->onparam] = calloc_int(nvals);
        memmove(cv->params[cv->onparam],(int *)vals, nvals * sizeof(double));   
    }
    else{
        fprintf(stderr, "Unrecognized type to add to cv grid parameters\n");
    }
    
    cv->onparam++;
}


static void update_based_type(struct CVCList * cvlist, enum RDTYPE type, char * name, void * param_list, int ind)
{
    if (type == RPUINT){
        cvc_list_update_case(cvlist,name,((size_t *)param_list)+ind);
    }
    else if (type == RPDBL){
        cvc_list_update_case(cvlist,name,((double *)param_list)+ind);
    }
    else if (type == RPINT){
        cvc_list_update_case(cvlist,name,((int *)param_list)+ind);
    }
}

/***********************************************************//**
   Setup cross validation cases in a list
   ONLY WORKS FOR DISCRETE OPTIONS
***************************************************************/
struct CVCList * cv_opt_grid_setup_cases(struct CVOptGrid * cv)
{
    struct CVCList * cvlist = NULL;
    if (cv->onparam == 1){
        for (size_t ii = 0; ii < cv->nvals[0]; ii++){
            cvc_list_push(&cvlist,1);
            update_based_type(cvlist,cv->types[0],cv->names[0],cv->params[0],ii);
        }
    }
    else if (cv->onparam == 2){
        for (size_t ii = 0; ii < cv->nvals[0]; ii++){
            for (size_t jj = 0; jj < cv->nvals[1]; jj++){
                cvc_list_push(&cvlist,2);
                update_based_type(cvlist,cv->types[0],cv->names[0],cv->params[0],ii);
                update_based_type(cvlist,cv->types[1],cv->names[1],cv->params[1],jj);
            }
        }
    }
    else if (cv->onparam == 3){
        for (size_t ii = 0; ii < cv->nvals[0]; ii++){
            for (size_t jj = 0; jj < cv->nvals[1]; jj++){
                for (size_t kk = 0; kk < cv->nvals[2]; kk++){
                    cvc_list_push(&cvlist,3);
                    update_based_type(cvlist,cv->types[0],cv->names[0],cv->params[0],ii);
                    update_based_type(cvlist,cv->types[1],cv->names[1],cv->params[1],jj);
                    update_based_type(cvlist,cv->types[2],cv->names[2],cv->params[2],kk);
                }
            }
        }
    }
    else {
        fprintf(stderr,"Cannot cross validate over arbitrary number of \n");
        fprintf(stderr,"parameters yet.\n");
    }

    return cvlist;
}

/***********************************************************//**
   Cross validation optimize
   ONLY WORKS FOR DISCRETE OPTIONS
***************************************************************/
void cross_validate_grid_opt(struct CrossValidate * cv,
                             struct CVOptGrid * cvgrid,
                             struct FTRegress * ftr,
                             struct c3Opt * optimizer)
{

    int verbose = cvgrid->verbose;
    
    /* printf("setup cases\n"); */
    struct CVCList * cvlist = cv_opt_grid_setup_cases(cvgrid);
    /* printf("print cases\n"); */

    /* cvc_list_print(cvlist,stdout); */
    
    // Run over all the CV cases to get the best one
    struct CVCList * temp = cvlist;
    size_t iter = 0;
    size_t bestiter = 0;
    double besterr = DBL_MAX;
    while (temp != NULL){

        cv_case_process(temp->cv,ftr,optimizer);
        double err = cross_validate_run(cv,ftr,optimizer);
        if (verbose > 1){
            char str[256];
            cv_case_string(temp->cv,str,256);
            printf("%s : cv_err=%3.15G\n",str,err);
        }
        if (err < besterr*0.999999999){
            besterr = err;
            bestiter = iter;
        }
        temp = temp->next;
        iter++;
    }

    // go back to best parameter set
    temp = cvlist;
    for (size_t ii = 0; ii < bestiter; ii++){
        temp = temp->next;
    }
    if (verbose > 0){
        char str[256];
        cv_case_string(temp->cv,str,256);
        printf("\tBest Parameters are\n\t\t%s : cv_err=%G\n",str,besterr);
    }

    // set best parameters
    /* for (size_t ii = 0; ii < temp->cv->nparam; ii++){ */
    cv_case_process(temp->cv,ftr,optimizer);
    /* }     */
    cvc_list_free(cvlist); cvlist = NULL;
}

