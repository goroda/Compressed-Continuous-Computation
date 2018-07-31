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





/** \file fapprox.h
 * Provides header files and structure definitions for functions in fapprox.c 
 */

#ifndef FAPPROX_H
#define FAPPROX_H

#include "functions.h"

struct OneApproxOpts
{
    enum function_class fc;
    void * aopts;
};

struct OneApproxOpts;
struct OneApproxOpts * 
one_approx_opts_alloc(enum function_class, void *);
/* struct OneApproxOpts *  */
/* one_approx_opts_ref(enum function_class, void **); */
void one_approx_opts_free(struct OneApproxOpts *);
void one_approx_opts_free_deep(struct OneApproxOpts **);
size_t one_approx_opts_get_nparams(const struct OneApproxOpts *);
void   one_approx_opts_set_nparams(struct OneApproxOpts *, size_t);
double one_approx_opts_get_lb(const struct OneApproxOpts *);
double one_approx_opts_get_ub(const struct OneApproxOpts *);
int one_approx_opts_linear_p(const struct OneApproxOpts *);

struct MultiApproxOpts;
struct MultiApproxOpts * multi_approx_opts_alloc(size_t);
void multi_approx_opts_free(struct MultiApproxOpts *);
void multi_approx_opts_free_deep(struct MultiApproxOpts **);
void multi_approx_opts_set_dim(struct MultiApproxOpts *,
                               size_t ,
                               struct OneApproxOpts *);
void multi_approx_opts_set_dim_ref(struct MultiApproxOpts *,size_t,
                                   struct OneApproxOpts **);
void
multi_approx_opts_set_all_same(struct MultiApproxOpts *,
                               struct OneApproxOpts *);
enum function_class 
multi_approx_opts_get_fc(const struct MultiApproxOpts *, size_t);
int multi_approx_opts_linear_p(const struct MultiApproxOpts *, size_t);
void * multi_approx_opts_get_aopts(const struct MultiApproxOpts *, size_t);
size_t multi_approx_opts_get_dim(const struct MultiApproxOpts *);
size_t multi_approx_opts_get_dim_nparams(const struct MultiApproxOpts *, size_t);
void   multi_approx_opts_set_dim_nparams(struct MultiApproxOpts *, size_t, size_t);
double multi_approx_opts_get_dim_ub(const struct MultiApproxOpts *, size_t);
double multi_approx_opts_get_dim_lb(const struct MultiApproxOpts *, size_t);

struct FiberOptArgs;
struct FiberOptArgs * fiber_opt_args_alloc(void);
struct FiberOptArgs * fiber_opt_args_init(size_t);
void fiber_opt_args_set_dim(struct FiberOptArgs *, size_t, void *);
struct FiberOptArgs * fiber_opt_args_bf(size_t,struct c3Vector **);
struct FiberOptArgs * fiber_opt_args_bf_same(size_t,struct c3Vector *);
void * fiber_opt_args_get_opts(const struct FiberOptArgs *, size_t);
void fiber_opt_args_free(struct FiberOptArgs *);

struct Operator
{
    struct GenericFunction * (*f)(const struct GenericFunction *,
                                  void * opts);
    void * opts;
};


#endif
