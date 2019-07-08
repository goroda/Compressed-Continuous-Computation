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
    Return whether or not the gradient is precomputed
***************************************************************/
int sl_mem_manager_gradient_precomputedp(const struct SLMemManager * mem){
    return ((mem->structure == LINEAR_ST) && (mem->once_eval_structure == 1));
}

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

    \param[in] rmem  - memory structure
    \param[in] index - index
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
    
        dbl_mem_array_free(mem->evals);
        mem->evals = NULL;
        
        dbl_mem_array_free(mem->grad);
        mem->grad = NULL;

        for (size_t ii = 0; ii < mem->dim; ii++){
            for (size_t jj = 0; jj < mem->N; jj++){
                free(mem->running_lr[ii][jj]); mem->running_lr[ii][jj] = NULL;
                free(mem->running_rl[ii][jj]); mem->running_rl[ii][jj] = NULL;
            }
            free(mem->running_lr[ii]); mem->running_lr[ii] = NULL;
            free(mem->running_rl[ii]); mem->running_rl[ii] = NULL;
        }
        free(mem->running_lr); mem->running_lr = NULL;
        free(mem->running_rl); mem->running_rl = NULL;
        
        free(mem->running_eval); mem->running_eval = NULL;
        free(mem->running_grad); mem->running_grad = NULL;
        
        free(mem->lin_structure_vals); mem->lin_structure_vals = NULL;
        free(mem); mem = NULL;
    }

}

/***********************************************************//**
    Allocate memory for supervised learning
    
    \param[in] d         - size of feature space
    \param[in] n         - number of data points to make room for
    \param[in] nparam    - number of total parameters
    \param[in] structure - either (LINEAR_ST or NONE)

    \returns Supervised learning memory manager
***************************************************************/
struct SLMemManager *
sl_mem_manager_alloc(size_t d, size_t n,
                     size_t nparam,
                     enum FTPARAM_ST structure)
{

    struct SLMemManager * mem = malloc(sizeof(struct SLMemManager));
    if (mem == NULL){
        fprintf(stderr, "Cannot allocate struct SLMemManager\n");
        exit(1);
    }

    mem->dim = d;
    mem->N = n;

    mem->running_eval = calloc_double(2*nparam);
    mem->running_grad = calloc_double(2*nparam);

    mem->running_lr = malloc(d * sizeof(double **));
    mem->running_rl = malloc(d * sizeof(double **));
    if ((mem->running_lr == NULL) || (mem->running_rl == NULL)){
        fprintf(stderr, "Failure to allocate memory for supervised learning memory manager\n");
        exit(1);
    }
    for (size_t ii = 0; ii < d; ii++){
        mem->running_lr[ii] = malloc(n * sizeof(double *));
        mem->running_rl[ii] = malloc(n * sizeof(double *));
        if ((mem->running_lr[ii] == NULL) || (mem->running_rl[ii] == NULL)){
            fprintf(stderr, "Failure to allocate memory for supervised learning memory manager\n");
            exit(1);
        }
        for (size_t jj = 0; jj < n; jj++){
            mem->running_lr[ii][jj] = NULL;
        }
        for (size_t jj = 0; jj < n; jj++){
            mem->running_rl[ii][jj] = NULL;
        }
    }

    mem->grad  = dbl_mem_array_alloc(n,nparam);
    mem->evals = dbl_mem_array_alloc(n,1);

    if ((structure != LINEAR_ST) && (structure != NONE_ST)){
        fprintf(stderr,"Error: Parameter structure type %d is unknown\n",structure);
        exit(1);
    }

    mem->structure = structure;
    mem->once_eval_structure = 0;
    mem->lin_structure_vals = malloc(mem->N * nparam * sizeof(double));

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

    struct FunctionTrain * ft = ft_param_get_ft(ftp);
    if ((mem->structure == LINEAR_ST) && (mem->once_eval_structure == 0)){

        for (size_t ii = 0; ii < mem->N; ii++){
            size_t onparam = 0;
            size_t onuni = 0;
            for (size_t kk = 0; kk < ftp->dim; kk++){
                for (size_t jj = 0; jj < ft->ranks[kk]*ft->ranks[kk+1]; jj++){
                    generic_function_param_grad_eval(
                        ft->cores[kk]->funcs[jj],1,x+ii*ftp->dim+kk,
                        mem->lin_structure_vals + ii * ftp->nparams + onparam);
                    onparam += ftp->nparams_per_uni[onuni];
                    onuni++;
                }
            }
        }
        mem->once_eval_structure = 1;
    }

    for (size_t ii = 0; ii < ftp->dim; ii++){
        for (size_t jj = 0; jj < mem->N; jj++){
            mem->running_lr[ii][jj] = calloc_double(ftp->ft->ranks[ii+1]);
            mem->running_rl[ii][jj] = calloc_double(ftp->ft->ranks[ii]);
        }
    }
    
}



///////////////////////////////////////////////////////////////////////////
// Data management
////////////////////////////////////////////////////////////////////////////

/** \struct Data
 * \brief Stores data
 * \var Data::N
 * number of data points for which to store objects
 * \var Data::dim
 * size of feature space
 * \var Data::x
 * training samples
 * \var Data::y
 * training labels
 * \var Data::xpointer
 * auxilary variable 
 */ 
struct Data
{
    size_t N;
    size_t dim;
    const double * x;
    const double * y;

    double * xpointer;
};


/***********************************************************//**
    Allocate a structure for data

    \param[in] N   - number of data points
    \param[in] dim - number of features

    \returns Space for storing a reference to the data
***************************************************************/
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

/***********************************************************//**
    Free data

    \param[in,out] data - data structure to free
***************************************************************/
void data_free(struct Data * data)
{
    if (data != NULL){
        free(data->xpointer);
        data->xpointer = NULL;
        free(data); data = NULL;
    }
}

/***********************************************************//**
    Get the number of data points
***************************************************************/
size_t data_get_N(const struct Data * data)
{
    return data->N;
}

/***********************************************************//**
    Set the data

    \param[in,out] data - data structure
    \param[in]     x    - training samples (N * dim), flattened with dim changing fastests
    \param[in]     y    - training labels (N)
***************************************************************/
void data_set_xy(struct Data * data, const double * x, const double * y)
{
    data->x = x;
    data->y = y;
}

/***********************************************************//**
    Get a reference to a subset of the data specified by indsub

    \param[in] data   - data structure
    \param[in] Nsub   - number of data points
    \param[in] indsub - index of subsets (Nsub,)

    \return pointer to training samples
***************************************************************/
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

/***********************************************************//**
    Subtract a particular data label from a reference label

    \param[in] data - data structure
    \param[in] ind  - index to subtract
    \param[in] val  - value to subtract

    \return difference
***************************************************************/
double data_subtract_from_y(const struct Data * data, size_t ind, double val)
{
    return data->y[ind]-val;
}
