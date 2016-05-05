// Copyright (c) 2014-2016, Massachusetts Institute of Technology
//
// This file is part of the Compressed Continuous Computation (C3) toolbox
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

/** \file ft.h 
 Provides header files for ft.c 
*/

#ifndef FT_H
#define FT_H

#include "qmarray.h"
#include "indmanage.h"

struct FunctionTrain;
struct FunctionTrain * function_train_alloc(size_t);
struct FunctionTrain * function_train_copy(const struct FunctionTrain *);
void function_train_free(struct FunctionTrain *);
unsigned char *
function_train_serialize(unsigned char *, struct FunctionTrain *,
                             size_t *);
unsigned char *
function_train_deserialize(unsigned char *, struct FunctionTrain **);
int function_train_save(struct FunctionTrain *, char *);
struct FunctionTrain * function_train_load(char *);

// getters and setters
size_t * function_train_get_ranks(const struct FunctionTrain *);
size_t function_train_get_maxrank(const struct FunctionTrain *);
double function_train_get_avgrank(const struct FunctionTrain *);

// evaluators
double function_train_eval(struct FunctionTrain *, const double *);
double function_train_eval_co_perturb(struct FunctionTrain *, 
                                      const double *, const double *, 
                                          double *);

// generators
struct FunctionTrain *
function_train_rankone(struct MultiApproxOpts *, struct Fwrap *);
struct FunctionTrain *
function_train_initsum(struct MultiApproxOpts *, struct Fwrap *);
struct FunctionTrain *
function_train_constant(double a, struct MultiApproxOpts *);
struct FunctionTrain *
function_train_linear(const double *, size_t, const double *, size_t,
                      struct MultiApproxOpts *);
struct FunctionTrain *
function_train_quadratic(const double *,const double *,
                             struct MultiApproxOpts *);


// computations
double
function_train_integrate(const struct FunctionTrain *);
double function_train_inner(const struct FunctionTrain *, 
                            const struct FunctionTrain * );
double function_train_norm2(const struct FunctionTrain *);
struct FunctionTrain * 
function_train_orthor(struct FunctionTrain *, 
                      struct MultiApproxOpts *);
struct FunctionTrain *
function_train_round(struct FunctionTrain *, double,
                         struct MultiApproxOpts *);
struct FunctionTrain * 
function_train_sum(const struct FunctionTrain *,
                       const struct FunctionTrain *);
void function_train_scale(struct FunctionTrain *, double);
struct FunctionTrain *
function_train_product(const struct FunctionTrain *, 
                           const struct FunctionTrain *);
double function_train_norm2diff(const struct FunctionTrain *, 
                                const struct FunctionTrain *);
double function_train_relnorm2diff(const struct FunctionTrain *,
                                       const struct FunctionTrain *);



//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/////////////////////                          ///////////////////////////
/////////////////////    Cross Approximation   ///////////////////////////
/////////////////////                          ///////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

struct FtCrossArgs;
struct FtCrossArgs * ft_cross_args_alloc(size_t, size_t);
void ft_cross_args_set_round_tol(struct FtCrossArgs *, double);
void ft_cross_args_set_kickrank(struct FtCrossArgs *, size_t);
void ft_cross_args_set_maxiter(struct FtCrossArgs *, size_t);
void ft_cross_args_set_no_adaptation(struct FtCrossArgs *);
void ft_cross_args_set_adaptation(struct FtCrossArgs *);
void ft_cross_args_set_maxrank_all(struct FtCrossArgs *, size_t);
void ft_cross_args_set_maxrank_ind(struct FtCrossArgs *, size_t, size_t);
void ft_cross_args_set_cross_tol(struct FtCrossArgs *, double);
void ft_cross_args_set_verbose(struct FtCrossArgs *, int);
;size_t * ft_cross_args_get_ranks(const struct FtCrossArgs *);
struct FtCrossArgs * ft_cross_args_copy(const struct FtCrossArgs *);
void ft_cross_args_free(struct FtCrossArgs *);

struct FunctionTrain *
ftapprox_cross(struct Fwrap *,
               struct FtCrossArgs *,
               struct CrossIndex **,
               struct CrossIndex **,
               struct MultiApproxOpts *,
               struct FiberOptArgs *,
               struct FunctionTrain *);
#endif
