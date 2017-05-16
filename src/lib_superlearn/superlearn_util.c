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

/** \file superlearn_util.c
 * Provides utilities for supervised learning
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

/** \struct RegMemSpace
 * \brief Memory manager for certain regression objects
 * \var RegMemSpace::ndata
 * number of objects being stored
 * \var RegMemSpace::one_data_size
 * size of memory for storing some values corresponding to a single object
 * \var RegMemSpace::vals
 * values stored
 */
struct RegMemSpace
{
    size_t ndata;
    size_t one_data_size;
    double * vals;
};


/***********************************************************//**
    Allocate regression memory structure

    \param[in] ndata         - number of objects being stored
    \param[in] one_data_size - number of elements of one object

    \returns Regression memory structure
***************************************************************/
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


/***********************************************************//**
    Allocate an array of regression memory structures
    
    \param[in] dim           - number of structures to allocate
    \param[in] ndata         - number of objects being stored
    \param[in] one_data_size - number of elements of one object

    \returns Array of regression memory structures

    \note
    Each element of the array is the same size
***************************************************************/
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


/***********************************************************//**
    Free a regression memory structure
    
    \param[in,out] rmem - structure to free
***************************************************************/
void reg_mem_space_free(struct RegMemSpace * rmem)
{
    if (rmem != NULL){
        free(rmem->vals); rmem->vals = NULL;
        free(rmem); rmem = NULL;
    }
}

/***********************************************************//**
    Free an array of regression memory structures

    \param[in]     dim  - size of array
    \param[in,out] rmem - array of structure to free
***************************************************************/
void reg_mem_space_arr_free(size_t dim, struct RegMemSpace ** rmem)
{
    if (rmem != NULL){
        for (size_t ii = 0; ii < dim; ii++){
            reg_mem_space_free(rmem[ii]); rmem[ii] = NULL;
        }
        free(rmem); rmem = NULL;
    }
}


/***********************************************************//**
    Return increment between objects 

    \param[in] rmem - regression memomory structure
***************************************************************/
size_t reg_mem_space_get_data_inc(const struct RegMemSpace * rmem)
{
    assert (rmem != NULL);
    return rmem->one_data_size;
}


/** \struct Regression Memory Manager
 * \brief Manages all memory for supervised learning
 * \var RegressionMemManager::dim
 * size of feature space
 * \var RegressionMemManager::N
 * number of data points for which to store objects
 * \var RegressionMemManager::running_evals_lr
 * space for storing the evaluations of cores from left to right
 * \var RegressionMemManager::running_evals_rl
 * space for storing the evaluations of cores from right to left
 * \var RegressionMemManager::running_grad
 * Running evaluations of gradient from left to right
 * \var RegressionMemManager::evals
 * space for storing the evaluations of an FT at all the data apoints
 * \var RegressionMemManager::grad
 * space for storing the gradient of the FT at all the data points
 * \var RegressionMemManager::grad_space
 * space for gradient computations
 * \var RegressionMemManager::all_grads
 * space for storing the gradients of univariate functions in particular cores
 * \var RegressionMemManager::fparam_space
 * space for storing the gradients of any single univariate function
 * \var RegressionMemManager::structure
 * flag for whether or not the parameters are linearly mapped to outputs
 * \var RegressionMemManager::once_eval_structure
 * flag for whether or structure has been precomputed
 * \var RegressionMemManager::lin_structure_vals
 * precomputed elements for linearly dependent parameters
 * \var RegressionMemManager::lin_structure_inc
 * precomputed incremement for linearly dependent parameters between data points
 */ 
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



/***********************************************************//**
    Free memory allocated 
    
    \param[in,out] mem - memory to free
***************************************************************/
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

/***********************************************************//**
    Allocate memory for regression
    
    \param[in] d                    - size of feature space
    \param[in] n                    - number of data points to make room for
    \param[in] num_params_per_core  - number of parameters in each core
    \param[in] max_param_within_uni - upper bound on number of parameters in a univariate funciton
    \param[in] structure            - either (LINEAR_ST or NONE)

    \returns Regression memomry manager
***************************************************************/
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


/***********************************************************//**
    Check whether enough memory has been allocated
    
    \param[in] mem - memory structure
    \param[in] N   - number of data points 

    \returns 1 if yes, 0  if no
***************************************************************/
int regress_mem_manager_enough(struct RegressionMemManager * mem, size_t N)
{
    if (mem->N < N){
        return 0;
    }
    return 1;
}

/***********************************************************//**
    Reset left,right, and gradient running evaluations
    
    \param[in,out] mem - memory structure
***************************************************************/
void regress_mem_manager_reset_running(struct RegressionMemManager * mem)
{
    running_core_total_restart(mem->running_evals_lr);
    running_core_total_restart(mem->running_evals_rl);
    running_core_total_arr_restart(mem->dim, mem->running_grad);

    /* printf("r2 in first core is %zu\n",mem->running_grad[0]->r2); */
}

/***********************************************************//**
    Check if special structure exists and if so, precompute
    
    \param[in,out] mem - memory structure
    \param[in]     ftp - parameterized ftp
    \param[in]     x   - all training points for which to precompute

    \note
    This is an aggressive function, there might be a mismatch between the size
    of x, plus recall that x can change for non-batch gradient
***************************************************************/
void regress_mem_manager_check_structure(struct RegressionMemManager * mem,
                                         const struct FTparam * ftp,
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
