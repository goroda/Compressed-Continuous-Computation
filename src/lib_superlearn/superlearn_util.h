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


/** \file superlearn_util.h
 * Provides prototypes for superlearn_util.c
 */


#ifndef SUPERLEARN_UTIL
#define SUPERLEARN_UTIL

#include "parameterization.h"


/** \struct DblMemArray
 * \brief Memory manager for certain regression objects
 * \var DblMemArray::ndata
 * number of objects being stored
 * \var DblMemArray::one_data_size
 * size of memory for storing some values corresponding to a single object
 * \var DblMemArray::vals
 * values stored
 */
struct DblMemArray
{
    size_t ndata;
    size_t one_data_size;
    double * vals;
};

size_t dbl_mem_array_get_data_inc(const struct DblMemArray *);

/** \struct SLMemManager
 * \brief Manages all memory for supervised learning
 * \var SLMemManager::dim
 * size of feature space
 * \var SLMemManager::N
 * number of data points for which to store objects
 * \var SLMemManager::running_eval
 * space for storing the evaluations of cores from left to right
 * \var SLMemManager::running_grad
 * space for storing the evaluations of gradients
 * \var SLMemManager::running_lr
 * space for storing the evaluations of cores from left to right
 * \var SLMemManager::running_rl
 * space for storing the evaluations of cores from right to left
 * \var SLMemManager::evals
 * space for storing the evaluations of an FT at all the data apoints
 * \var SLMemManager::grad
 * space for storing the gradient of the FT at all the data points
 * \var SLMemManager::structure
 * flag for whether or not the parameters are linearly mapped to outputs
 * \var SLMemManager::once_eval_structure
 * flag for whether or structure has been precomputed
 * \var SLMemManager::lin_structure_vals
 * precomputed elements for linearly dependent parameters
 */ 
struct SLMemManager
{

    size_t dim;
    size_t N;
    double * running_eval;
    double * running_grad;
    
    double *** running_lr;
    double *** running_rl;
    
    struct DblMemArray * evals;
    struct DblMemArray * grad;

    enum FTPARAM_ST structure;
    int once_eval_structure;
    double * lin_structure_vals;
};

void sl_mem_manager_check_structure(struct SLMemManager *, const struct FTparam *,const double *);


/***********************************************************//**
    Return whether or not the gradient is precomputed
***************************************************************/
int sl_mem_manager_gradient_precomputedp(const struct SLMemManager * mem);
/* inline int sl_mem_manager_gradient_precomputedp(const struct SLMemManager * mem) */
/* { */
/*     return ((mem->structure == LINEAR_ST) && (mem->once_eval_structure == 1)); */
/* } */

struct SLMemManager * sl_mem_manager_alloc(size_t, size_t, size_t,enum FTPARAM_ST);
void sl_mem_manager_free(struct SLMemManager *);


struct Data;
struct Data * data_alloc(size_t, size_t);
size_t data_get_N(const struct Data *);
void data_set_xy(struct Data *, const double *, const double *);
void data_free(struct Data *);
const double * data_get_subset_ref(struct Data *, size_t, size_t *);
double data_subtract_from_y(const struct Data *, size_t, double);
#endif
