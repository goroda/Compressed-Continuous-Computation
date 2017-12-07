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


/** \file learning_options.c
 * Options handling for supervised learning
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "array.h"
#include "learning_options.h"

/***********************************************************//**
    Free regression options
    
    \param[in,out] opts - options to free
***************************************************************/
void regress_opts_free(struct RegressOpts * opts)
{
    if (opts != NULL){
        free(opts->restrict_rank_opt); opts->restrict_rank_opt = NULL;
        if (opts->stored_fvals != NULL){
            free(opts->stored_fvals); opts->stored_fvals = NULL;
        }
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

    ropts->kristoffel_active = 0;

    ropts->nepochs = 0;
    ropts->stored_fvals = NULL;


    ropts->sample_weights = NULL;
    
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
struct RegressOpts * regress_opts_create(size_t dim, enum REGTYPE type, enum REGOBJ obj)
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
    Set weights on data by reference
    
    \param[in,out] opts    - regression options
    \param[in]     weights - weights
***************************************************************/
void regress_opts_set_sample_weights(struct RegressOpts * opts, const double * weights)
{
    assert (opts != NULL);
    opts->sample_weights = weights;
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
    Set whether or not to use kristoffel weighting
    
    \param[in,out] opts   - regression options
    \param[in]     active - 0 to not use, 1 to use
***************************************************************/
void regress_opts_set_kristoffel(struct RegressOpts * opts, int active)
{
    assert (opts != NULL);
    opts->kristoffel_active = active;
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


/***********************************************************//**
    Add function values per epoch of optimization
    
    \param[in,out] opts    - regression options
    \param[in]     nepochs - number of epochs
    \param[in]     fvals   - objective value per epoch
***************************************************************/
void regress_opts_add_stored_vals(struct RegressOpts * opts, size_t nepochs, double * fvals)
{
    assert (opts != NULL);
    opts->nepochs = nepochs;
    opts->stored_fvals = fvals;
}

/***********************************************************//**
    Get number of epochs
    
    \param[in] opts    - regression options
    
    \return nepochs
***************************************************************/
size_t regress_opts_get_nepochs(const struct RegressOpts * opts)
{
    assert (opts != NULL);
    return opts->nepochs;
}

/***********************************************************//**
    Get objective function per epoch
    
    \param[in] opts - regression options
    
    \return objective value per epoch
***************************************************************/
double * regress_opts_get_stored_fvals(const struct RegressOpts * opts)
{
    assert (opts != NULL);
    return opts->stored_fvals;
}

