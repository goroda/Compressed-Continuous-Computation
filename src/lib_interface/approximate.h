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





/** \file approximate.h
 * Provides header files and structure definitions for functions in in approximate.c
 */

#ifndef C3_APPROX_H
#define C3_APPROX_H

#include "lib_clinalg.h"
#include "regress.h"

enum C3ATYPE { CROSS, REGRESS, C3UNSPEC };

struct C3Approx;
struct C3Approx * c3approx_create(enum C3ATYPE, size_t);
void c3approx_destroy(struct C3Approx *);

void c3approx_set_approx_opts_dim(struct C3Approx *,size_t,
                                  struct OneApproxOpts *);
void c3approx_set_opt_opts_dim(struct C3Approx *, size_t,void *);

void c3approx_set_round_tol(struct C3Approx *, double);
void c3approx_set_verbose(struct C3Approx *, int);

///////////////////////////////////////
// Cross approximation
///////////////////////////////////////
void c3approx_init_cross(struct C3Approx * c3a, size_t, int,
                        double **);
void c3approx_set_cross_tol(struct C3Approx *, double);
void c3approx_set_adapt_kickrank(struct C3Approx *, size_t);
void c3approx_set_adapt_maxrank_all(struct C3Approx *, size_t);
//void c3approx_set_adapt_maxiter(struct C3Approx *, size_t);
void c3approx_set_cross_maxiter(struct C3Approx *, size_t);
// perform cross approximation
struct FunctionTrain * c3approx_do_cross(struct C3Approx *,struct Fwrap *,int);

///////////////////////////////////////
// Regression
///////////////////////////////////////
/* void c3approx_set_regress_type(struct C3Approx *, enum REGTYPE); */
/* void c3approx_set_regress_start_ranks(struct C3Approx *, const size_t *); */
/* void c3approx_set_regress_start_ranks(struct C3Approx *, size_t); */
/* void c3approx_set_regress_num_param_per_func(struct C3Approx *, size_t); */
/* void c3approx_init_regress(struct C3Approx *); */
/* struct FunctionTrain * */
/* c3approx_do_regress(struct C3Approx *, size_t, */
/*                     const double *, size_t, */
/*                     const double *, size_t, enum REGOBJ); */


////////////////////
// Getting
////////////////////
struct MultiApproxOpts * c3approx_get_approx_args(const struct C3Approx *);
size_t c3approx_get_dim(const struct C3Approx *);




#endif
