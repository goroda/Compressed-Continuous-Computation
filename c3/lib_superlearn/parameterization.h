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


/** \file parameterization.h
 * Provides prototypes for parameterization.c
 */

#ifndef PARAMETERIZATION_H
#define PARAMETERIZATION_H

#include <stddef.h>
#include "lib_funcs.h"


enum FTPARAM_ST {LINEAR_ST, NONE_ST};

/** \struct FTparam
 * \brief Stores an FT and parameterization information
 * \var FTparam::ft
 * function-train 
 * \var FTparam::dim
 * size of input space
 * \var FTparam::nparams_per_uni
 * number of parameters in each univariate function of each core
 * \var FTparam::nparams_per_core
 * number of parameters in each core
 * \var FTparam::max_param_uni
 * Upper bound on the maximum number of parameters within any given function
 * \var FTparam::nparams
 * Number of total params describing the function train
 * \var FTparam::params
 * Array of the parameters describing the FT
 * \var FTparam::approx_opts
 * Approximation options
 */
struct FTparam
{
    struct FunctionTrain * ft;
    size_t dim;

    size_t * nparams_per_uni; // number of parmaters per univariate funciton
    size_t * nparams_per_core;
    size_t max_param_uni;
    size_t nparams;
    double * params;
    
    struct MultiApproxOpts * approx_opts;
};


struct FTparam *
ft_param_alloc(size_t,struct MultiApproxOpts *,
               double *, size_t *);
void ft_param_free(struct FTparam *);
double ft_param_get_param(const struct FTparam *, size_t);
size_t ft_param_get_nparams(const struct FTparam *);
size_t ft_param_get_nparams_restrict(const struct FTparam *, const size_t *);
size_t * ft_param_get_nparams_per_core(const struct FTparam *);
struct FunctionTrain * ft_param_get_ft(const struct FTparam *);
void ft_param_create_constant(struct FTparam *, double, double);
void ft_param_create_from_lin_ls(struct FTparam *, size_t,
                                 const double *, const double *,
                                 double);

void ft_param_update_params(struct FTparam *, const double *);
void ft_param_update_restricted_ranks(struct FTparam *, const double *,const size_t *);
void ft_param_update_inside_restricted_ranks(struct FTparam *,const double *, const size_t *);
                                             
void ft_param_update_core_params(struct FTparam *, size_t, const double *);
enum FTPARAM_ST ft_param_extract_structure(const struct FTparam *);

// AIO HELPERS
double ft_param_gradeval(struct FTparam *, const double *,double*, double *, double*,double*);
double ft_param_eval_lin(struct FTparam *, const double *,double*);
double ft_param_gradeval_lin(struct FTparam *, const double *,double *,double *,double *);


// ALS HELPERS
void process_sweep_left_right(struct FTparam * ftp, size_t current_core,double x, double * evals,
                              double * running_lr, double * running_lr_up);
void process_sweep_right_left(struct FTparam * ftp, size_t current_core,double x, double * evals,
                              double * running_rl, double * running_rl_up);

void process_sweep_left_right_lin(struct FTparam * ftp, size_t current_core,double * evals,
                                  double * running_lr, double * running_lr_up);
void process_sweep_right_left_lin(struct FTparam * ftp, size_t current_core,double * evals,
                                  double * running_rl, double * running_rl_up);
double ft_param_core_gradeval(struct FTparam * ftp, size_t core, double x,
                              double *,  double *,double * running_rl,
                              double *);
double ft_param_core_eval_lin(struct FTparam *,size_t,double *,double *,const double *);
double ft_param_core_gradeval_lin(struct FTparam * ftp, size_t core,
                                  double * grad,  double * running_lr,double * running_rl,
                                  double * grad_evals);

// HESSIAN
double ft_param_hessvec(struct FTparam * ftp, const double * x,
                        const double * vec,
                        double * hess_vec);

#endif
