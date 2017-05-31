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

#include "superlearn_util.h"
#include "ft.h"

/***********************************************************//**
    Allocate a memory structure for storing an array of an array of doubles

    \param[in] ndata         - number of arrays
    \param[in] one_data_size - size of each array (same size)

    \returns Regression memory structure
***************************************************************/
struct DblMemArray * dbl_mem_array_alloc(size_t ndata, size_t one_data_size)
{
    struct DblMemArray * mem = malloc(sizeof(struct DblMemArray));
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
struct DblMemArray ** dbl_mem_array_arr_alloc(size_t dim, size_t ndata, size_t one_data_size)
{
    struct DblMemArray ** mem = malloc(dim * sizeof(struct DblMemArray * ));
    if (mem == NULL){
        fprintf(stderr, "Failure to allocate memmory of regression\n");
        return NULL;
    }
    for (size_t ii = 0; ii < dim; ii++){
        mem[ii] = dbl_mem_array_alloc(ndata,one_data_size);
    }

    return mem;
}


/***********************************************************//**
    Free a memory structure
    
    \param[in,out] rmem - structure to free
***************************************************************/
void dbl_mem_array_free(struct DblMemArray * rmem)
{
    if (rmem != NULL){
        free(rmem->vals); rmem->vals = NULL;
        free(rmem); rmem = NULL;
    }
}

/***********************************************************//**
    Free an array of memory 

    \param[in]     dim  - size of array
    \param[in,out] rmem - array of structure to free
***************************************************************/
void dbl_mem_array_arr_free(size_t dim, struct DblMemArray ** rmem)
{
    if (rmem != NULL){
        for (size_t ii = 0; ii < dim; ii++){
            dbl_mem_array_free(rmem[ii]); rmem[ii] = NULL;
        }
        free(rmem); rmem = NULL;
    }
}


/***********************************************************//**
    Return increment between objects 

    \param[in] rmem - memory structure
***************************************************************/
size_t dbl_mem_array_get_data_inc(const struct DblMemArray * rmem)
{
    assert (rmem != NULL);
    return rmem->one_data_size;
}

/***********************************************************//**
    Return a pointer to element of the array

    \param[in] rmem - memory structure
    \param[in] ind  - index
***************************************************************/
double * dbl_mem_array_get_element(const struct DblMemArray * rmem, size_t index)
{
    assert (rmem != NULL);
    assert (index < rmem->ndata);
    return rmem->vals + index * rmem->one_data_size;
}



/***********************************************************//**
    Free memory allocated 
    
    \param[in,out] mem - memory to free
***************************************************************/
void sl_mem_manager_free(struct SLMemManager * mem)
{
    if (mem != NULL){
    
        dbl_mem_array_free(mem->grad_space);
        mem->grad_space = NULL;
        
        dbl_mem_array_free(mem->fparam_space);
        mem->fparam_space = NULL;
        
        dbl_mem_array_free(mem->evals);
        mem->evals = NULL;
        
        dbl_mem_array_free(mem->grad);
        mem->grad = NULL;

        dbl_mem_array_arr_free(mem->dim,mem->all_grads);
        mem->all_grads = NULL;

        running_core_total_free(mem->running_evals_lr);
        running_core_total_free(mem->running_evals_rl);
        running_core_total_arr_free(mem->dim,mem->running_grad);


        free(mem->lin_structure_vals); mem->lin_structure_vals = NULL;
        free(mem->lin_structure_inc);  mem->lin_structure_inc  = NULL;

        for (size_t ii = 0; ii < mem->dim; ii++){
            free(mem->lin_structure_vals_subset[ii]); mem->lin_structure_vals_subset[ii] = NULL;
        }
        free(mem->lin_structure_vals_subset); mem->lin_structure_vals_subset = NULL;
        free(mem); mem = NULL;
    }

}

/***********************************************************//**
    Allocate memory for supervised learning
    
    \param[in] d                    - size of feature space
    \param[in] n                    - number of data points to make room for
    \param[in] num_params_per_core  - number of parameters in each core
    \param[in] max_param_within_uni - upper bound on number of parameters in a univariate funciton
    \param[in] structure            - either (LINEAR_ST or NONE)

    \returns Supervised learning memory manager
***************************************************************/
struct SLMemManager *
sl_mem_manager_alloc(size_t d, size_t n,
                          size_t * num_params_per_core,
                          size_t * ranks,
                          size_t max_param_within_uni,
                          enum FTPARAM_ST structure)
{

    struct SLMemManager * mem = malloc(sizeof(struct SLMemManager));
    if (mem == NULL){
        fprintf(stderr, "Cannot allocate struct SLMemManager\n");
        exit(1);
    }

    mem->dim = d;
    mem->N = n;

    mem->fparam_space = dbl_mem_array_alloc(1,max_param_within_uni);

    mem->all_grads = malloc(d * sizeof(struct DblMemArray * ));
    assert (mem->all_grads != NULL);
    size_t mr2_max = 0;
    size_t mr2;
    size_t num_tot_params = 0;
    size_t max_param_within_core = 0;
    for (size_t ii = 0; ii < d; ii++){
        mr2 = ranks[ii]*ranks[ii+1];
        mem->all_grads[ii] = dbl_mem_array_alloc(n,num_params_per_core[ii] * mr2 );
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
    
    mem->grad       = dbl_mem_array_alloc(n,num_tot_params);
    mem->evals      = dbl_mem_array_alloc(n,1);


    mem->grad_space = dbl_mem_array_alloc(n, max_param_within_core * mr2_max);
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


    mem->lin_structure_vals_subset = malloc(mem->dim * sizeof(double * ));
    assert (mem->lin_structure_vals_subset != NULL);
    for (size_t ii = 0; ii < mem->dim; ii++){
        mem->lin_structure_vals_subset[ii] = NULL;
    }
    return mem;
    
}


/***********************************************************//**
    Check whether enough memory has been allocated
    
    \param[in] mem - memory structure
    \param[in] N   - number of data points 

    \returns 1 if yes, 0  if no
***************************************************************/
int sl_mem_manager_enough(struct SLMemManager * mem, size_t N)
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
void sl_mem_manager_reset_running(struct SLMemManager * mem)
{
    running_core_total_restart(mem->running_evals_lr);
    running_core_total_restart(mem->running_evals_rl);
    running_core_total_arr_restart(mem->dim, mem->running_grad);

    /* printf("r2 in first core is %zu\n",mem->running_grad[0]->r2); */
}

/***********************************************************//**
    Reset the running gradient of a particular core
    
    \param[in,out] mem - memory structure
***************************************************************/
void sl_mem_manager_reset_core_grad(struct SLMemManager * mem, size_t ii)
{
    running_core_total_restart(mem->running_grad[ii]);
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
void sl_mem_manager_check_structure(struct SLMemManager * mem,
                                    const struct FTparam * ftp,
                                    const double * x)
{


    
    if ((mem->structure == LINEAR_ST) && (mem->once_eval_structure == 0)){
        for (size_t ii = 0; ii < ftp->dim; ii++){
            if (ii == 0){
                qmarray_param_grad_eval(
                    ftp->ft->cores[ii],
                    mem->N,
                    x + ii,
                    ftp->dim,
                    NULL,
                    0,
                    mem->all_grads[ii]->vals,
                    dbl_mem_array_get_data_inc(mem->all_grads[ii]),
                    mem->fparam_space->vals);
            }
            else{
                qmarray_param_grad_eval_sparse_mult(
                    ftp->ft->cores[ii],
                    mem->N,
                    x + ii,
                    ftp->dim,
                    NULL,
                    0,
                    mem->all_grads[ii]->vals,
                    dbl_mem_array_get_data_inc(mem->all_grads[ii]),
                    NULL,NULL,0);
            }

            /* mem->lin_structure_vals[ii] = mem->all_grads[ii]->vals; */
            /* mem->lin_structure_inc[ii]  = dbl_mem_array_get_data_inc(mem->all_grads[ii]); */
        }
        /* printf("OKOKOK\n"); */
        mem->once_eval_structure = 1;
    }
}

void sl_mem_manager_init_gradient_subset(struct SLMemManager * mem, size_t N, const size_t * ind)
{
    if (ind == NULL){
        for (size_t ii = 0; ii < mem->dim; ii++){
            mem->lin_structure_vals[ii] = mem->all_grads[ii]->vals;
            mem->lin_structure_inc[ii] = dbl_mem_array_get_data_inc(mem->all_grads[ii]);
        }
    }
    else{
        for (size_t ii = 0; ii < mem->dim; ii++){
            mem->lin_structure_inc[ii] = dbl_mem_array_get_data_inc(mem->all_grads[ii]);

            free(mem->lin_structure_vals_subset[ii]); mem->lin_structure_vals_subset[ii] = NULL;
            mem->lin_structure_vals_subset[ii] = calloc_double(mem->lin_structure_inc[ii]*N);
            for (size_t jj = 0; jj < N; jj++){
                memmove(mem->lin_structure_vals_subset[ii] + jj * mem->lin_structure_inc[ii],
                        dbl_mem_array_get_element(mem->all_grads[ii],ind[jj]),
                        mem->lin_structure_inc[ii]*sizeof(double));
            }
            mem->lin_structure_vals[ii] = mem->lin_structure_vals_subset[ii];
        }
    }

}

///////////////////////////////////////////////////////////////////////////
// Data management
////////////////////////////////////////////////////////////////////////////

struct Data
{
    size_t N;
    size_t dim;
    const double * x;
    const double * y;

    double * xpointer;
};

struct Data * data_alloc(size_t N, size_t dim)
{
    struct Data * data = malloc(sizeof(struct Data));
    if (data == NULL){
        fprintf(stderr,"Failure to allocate memory for data\n");
        exit(1);
    }

    data->N = N;
    data->dim = dim;
    data->xpointer = calloc_double(N*dim);

    return data;
}

void data_free(struct Data * data)
{
    if (data != NULL){
        free(data->xpointer);
        data->xpointer = NULL;
        free(data); data = NULL;
    }
}

size_t data_get_N(const struct Data * data)
{
    return data->N;
}

void data_set_xy(struct Data * data, const double * x, const double * y)
{
    data->x = x;
    data->y = y;
}

const double * data_get_subset_ref(struct Data * data, size_t Nsub, size_t * indsub)
{
    assert (data != NULL);
    assert (Nsub <= data->N);

    if (indsub == NULL){
        return data->x;
    }
    
    for (size_t ii = 0; ii < Nsub; ii++){
        for (size_t jj = 0; jj < data->dim; jj++){
            data->xpointer[jj + ii*data->dim] = data->x[jj + indsub[ii]*data->dim];
        }
    }
    return data->xpointer;    
}

double data_subtract_from_y(const struct Data * data, size_t ind, double val)
{
    return data->y[ind]-val;
}
