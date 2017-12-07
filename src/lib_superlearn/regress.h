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





/** \file regress.h
 * Provides header files for FT regression
 */

#ifndef C3_FT_REGRESSION
#define C3_FT_REGRESSION

#include "ft.h"
#include "parameterization.h"
#include "superlearn.h"
#include "superlearn_util.h"
#include "learning_options.h"

/* int ft_param_eval_grad(size_t,const double *, double **, */
/*                        double **, struct SLMemManager *, void *); */
/* int ft_param_learning_interface(size_t nparam, const double * param, */
/*                                 size_t N, size_t * ind, */
/*                                 struct SLMemManager * mem, */
/*                                 struct Data * data, */
/*                                 double ** evals, double ** grads, void * args); */

struct FTRegress;
struct FTRegress * ft_regress_alloc(size_t dim, struct MultiApproxOpts *,size_t * ranks);
void ft_regress_set_sample_weights(const struct FTRegress *, const double *);
size_t ft_regress_get_dim(const struct FTRegress *);
void ft_regress_set_adapt(struct FTRegress *, int);
void ft_regress_set_maxrank(struct FTRegress *, size_t);
void ft_regress_set_kickrank(struct FTRegress *, size_t);
void ft_regress_set_roundtol(struct FTRegress *, double);
void ft_regress_set_kfold(struct FTRegress *, size_t);
void ft_regress_set_finalize(struct FTRegress *, int);
void ft_regress_set_opt_restrict(struct FTRegress *, int);
void ft_regress_set_kristoffel(struct FTRegress *, int);

void ft_regress_free(struct FTRegress *);
void ft_regress_reset_param(struct FTRegress*, struct MultiApproxOpts *, size_t *);
void ft_regress_set_type(struct FTRegress *, enum REGTYPE);
void ft_regress_set_obj(struct FTRegress *, enum REGOBJ);
void ft_regress_set_alg_and_obj(struct FTRegress *, enum REGTYPE, enum REGOBJ);

/* void ft_regress_set_parameter(struct FTRegress *, char *, void *); */
/* void ft_regress_process_parameters(struct FTRegress *); */
void ft_regress_set_als_conv_tol(struct FTRegress *, double);
void ft_regress_set_max_als_sweep(struct FTRegress *, size_t);
void ft_regress_set_verbose(struct FTRegress *, int);
void ft_regress_set_regularization_weight(struct FTRegress *, double);
double ft_regress_get_regularization_weight(const struct FTRegress *);

size_t ft_regress_get_nepochs(const struct FTRegress *);
double ft_regress_get_stored_fvals(const struct FTRegress *, size_t);
    
double * ft_regress_get_params(struct FTRegress *, size_t *);
void ft_regress_update_params(struct FTRegress *, const double *);



struct FunctionTrain *
ft_regress_run(struct FTRegress *,struct c3Opt *,size_t,const double* xdata, const double * ydata);
struct FunctionTrain *
ft_regress_run_rankadapt(struct FTRegress *, double, size_t, size_t,
                         struct c3Opt *,int,
                         size_t, const double *, const double *,int);



struct CrossValidate;
void cross_validate_free(struct CrossValidate *);
struct CrossValidate * cross_validate_init(size_t, size_t,
                                           const double *,
                                           const double *,
                                           size_t, int);
double cross_validate_run(struct CrossValidate *,
                          struct FTRegress *,
                          struct c3Opt *);

struct CVOptGrid;
struct CVOptGrid * cv_opt_grid_init(size_t);
void cv_opt_grid_free(struct CVOptGrid *);
void cv_opt_grid_set_verbose(struct CVOptGrid *, int);
void cv_opt_grid_add_param(struct CVOptGrid *, char *, size_t, void * paramlist);

void cross_validate_grid_opt(struct CrossValidate *,
                             struct CVOptGrid *,
                             struct FTRegress *,
                             struct c3Opt *);


struct FunctionTrain *
c3_regression_run(struct FTparam *, struct RegressOpts *, struct c3Opt *,
                  size_t N, const double * x, const double * y);

#endif


