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


/** \file regress.h
 * Provides header files for FT regression
 */

#ifndef C3_FT_REGRESSION
#define C3_FT_REGRESSION

#include "ft.h"

enum REGTYPE {ALS,AIO,REGNONE};
enum REGOBJ  {FTLS,FTLS_SPARSEL2,REGOBJNONE};
struct FTRegress;
struct FTRegress * ft_regress_alloc(size_t, struct MultiApproxOpts *);
void ft_regress_free(struct FTRegress *);
void ft_regress_set_type(struct FTRegress *, enum REGTYPE);
void ft_regress_set_obj(struct FTRegress *, enum REGOBJ);
void ft_regress_set_start_ranks(struct FTRegress *, const size_t * );
void ft_regress_set_parameter(struct FTRegress *, char *, void *);
void ft_regress_set_data(struct FTRegress *, size_t, const double *, size_t,
                         const double *, size_t);
void ft_regress_process_parameters(struct FTRegress *);
void ft_regress_set_als_maxsweep(struct FTRegress *, size_t);
void ft_regress_set_convtol(struct FTRegress *, double);
void ft_regress_set_verbose(struct FTRegress *, int);
void ft_regress_set_regweight(struct FTRegress *, double);
double * ft_regress_get_params(struct FTRegress *, size_t *);
void ft_regress_update_params(struct FTRegress *, const double *);

/* void ft_regress_prep_memory(struct FTRegress *, int); */
/* void ft_regress_prep_memory(struct FTRegress *, struct FunctionTrain *, int); */
struct FunctionTrain * ft_regress_run(struct FTRegress *);



struct CrossValidate;
void cross_validate_free(struct CrossValidate *);
struct CrossValidate * cross_validate_init(size_t, size_t,
                                           const double *,
                                           const double *,
                                           size_t);
void cross_validate_add_discrete_param(struct CrossValidate *,
                                       char *, size_t,void *);
double cross_validate_run(struct CrossValidate *,
                          struct FTRegress *);
void cross_validate_opt(struct CrossValidate *,
                        struct FTRegress *,int);

enum FTPARAM_ST {LINEAR_ST, NONE_ST};

struct FTparam;
struct FTparam *
ft_param_alloc(size_t,struct MultiApproxOpts *,
               double *, size_t *);
void ft_param_free(struct FTparam *);
size_t * ft_param_get_num_params_per_core(const struct FTparam *);
struct FunctionTrain * ft_param_get_ft(const struct FTparam *);
void ft_param_update_params(struct FTparam *, const double *);
void ft_param_update_core_params(struct FTparam *, size_t, const double *);

struct RegressOpts;
struct RegressOpts *
regress_opts_create(enum REGTYPE, enum REGOBJ, size_t,
                    size_t, const double *, const double *);
void regress_opts_free(struct RegressOpts *);

void regress_opts_set_convtol(struct RegressOpts *, double);
void regress_opts_set_als_maxsweep(struct RegressOpts *, size_t);
void regress_opts_set_verbose(struct RegressOpts *, int);
void regress_opts_set_regweight(struct RegressOpts *, double);
void regress_opts_initialize_memory(struct RegressOpts *, size_t *,
                                    size_t *, size_t,
                                    enum FTPARAM_ST);

double ft_param_eval_objective_aio(struct FTparam *,
                                   struct RegressOpts *,
                                   double *);
struct FunctionTrain *
c3_regression_run(struct FTparam *, struct RegressOpts *);

#endif


