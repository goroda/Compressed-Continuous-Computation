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





/** \file learning_options.h
 * Provides prototypes for learning_options.c
 */

#ifndef C3_LEARN_OPTIONS
#define C3_LEARN_OPTIONS

#include <stddef.h>

enum REGTYPE {ALS,AIO,REGNONE};
enum REGOBJ  {FTLS,FTLS_SPARSEL2,REGOBJNONE};

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

    int kristoffel_active;

    size_t nepochs;
    double * stored_fvals;

    const double * sample_weights;
};

struct RegressOpts * regress_opts_alloc(size_t);
void regress_opts_free(struct RegressOpts *);
struct RegressOpts * regress_opts_create(size_t, enum REGTYPE, enum REGOBJ);
void regress_opts_set_sample_weights(struct RegressOpts *, const double *);
void regress_opts_set_max_als_sweep(struct RegressOpts *, size_t);
void regress_opts_set_kristoffel(struct RegressOpts *, int);
void regress_opts_set_als_conv_tol(struct RegressOpts *, double);
void regress_opts_set_verbose(struct RegressOpts *, int);
void regress_opts_set_restrict_rank(struct RegressOpts *, size_t, size_t);
void regress_opts_set_regularization_weight(struct RegressOpts *, double);
double regress_opts_get_regularization_weight(const struct RegressOpts *);
void regress_opts_add_stored_vals(struct RegressOpts *, size_t, double *);
size_t regress_opts_get_nepochs(const struct RegressOpts *);
double * regress_opts_get_stored_fvals(const struct RegressOpts *);

#endif
