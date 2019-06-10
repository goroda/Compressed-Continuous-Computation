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






/** \file constelm.h
 * Provides header files and structure definitions for functions in constelm.c
 */

#ifndef CONSTELM_H
#define CONSTELM_H

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

#include <stddef.h>
#include <stdio.h>

#include "fwrap.h"

struct ConstElemExpAopts;
struct ConstElemExpAopts * const_elem_exp_aopts_alloc(size_t, double * onedx);
struct ConstElemExpAopts *
const_elem_exp_aopts_alloc_adapt(size_t,double *,double,double,double, double);
void const_elem_exp_aopts_free(struct ConstElemExpAopts *);
void const_elem_exp_aopts_free_deep(struct ConstElemExpAopts **);

double const_elem_exp_aopts_get_lb(const struct ConstElemExpAopts *);
double const_elem_exp_aopts_get_ub(const struct ConstElemExpAopts *);
void const_elem_exp_aopts_set_nodes(struct ConstElemExpAopts *,
                                  size_t, double *);

void const_elem_exp_aopts_set_adapt(struct ConstElemExpAopts *,double, double);
void const_elem_exp_aopts_set_delta(struct ConstElemExpAopts *, double);
void const_elem_exp_aopts_set_hmin(struct ConstElemExpAopts *, double);

size_t const_elem_exp_aopts_get_num_nodes(const struct ConstElemExpAopts *);
size_t const_elem_exp_aopts_get_nparams(const struct ConstElemExpAopts *);
void   const_elem_exp_aopts_set_nparams(struct ConstElemExpAopts*, size_t);

/////////////////////////////////////////////////////////////////////

/** \struct ConstElemExp
 * \brief structure to represent an expansion of piecewise constant elements
 * \var ConstElemExp::num_nodes
 * number of basis functions or nodes
 * \var ConstElemExp::nodes
 * nodes of basis functions
 * \var ConstElemExp::coeff
 * coefficients of basis functions
 * \var ConstElemExp::diff
 * difference between nodes
 */
struct ConstElemExp{

    size_t num_nodes;
    double * nodes;
    double * coeff;
    double * diff;
};

size_t const_elem_exp_get_num_nodes(const struct ConstElemExp *);
size_t const_elem_exp_get_num_params(const struct ConstElemExp *);
struct ConstElemExp * const_elem_exp_alloc(void);
struct ConstElemExp * const_elem_exp_copy(const struct ConstElemExp *);
void const_elem_exp_free(struct ConstElemExp *);
struct ConstElemExp * const_elem_exp_init(size_t, double *, double *);
struct ConstElemExp *
const_elem_exp_create_with_params(struct ConstElemExpAopts *,
                                size_t, const double *);
int
const_elem_exp_update_params(struct ConstElemExp *,
                             size_t, const double *);

size_t const_elem_exp_get_params(const struct ConstElemExp *, double *);
double * const_elem_exp_get_params_ref(const struct ConstElemExp *, size_t *);

unsigned char *
serialize_const_elem_exp(unsigned char *, struct ConstElemExp *,size_t *);
unsigned char * deserialize_const_elem_exp(unsigned char *, 
                                         struct ConstElemExp **);
double const_elem_exp_eval(const struct ConstElemExp *, double);
double const_elem_exp_deriv_eval(const struct ConstElemExp * f, double x);
/* inline double const_elem_exp_deriv_eval(const struct ConstElemExp * f, double x); */
/* { */
/*     (void)(x); */
/*     (void)(f); */
/*     return 0.0; */
/* } */

void const_elem_exp_evalN(const struct ConstElemExp *, size_t,
                        const double *, size_t, double *, size_t);
double const_elem_exp_get_nodal_val(const struct ConstElemExp *, size_t);
struct ConstElemExp * const_elem_exp_deriv(const struct ConstElemExp *);
struct ConstElemExp * const_elem_exp_dderiv(const struct ConstElemExp *);
struct ConstElemExp * const_elem_exp_dderiv_periodic(const struct ConstElemExp *);
int const_elem_exp_param_grad_eval(
    struct ConstElemExp *, size_t, const double *, double *);
double const_elem_exp_param_grad_eval2(struct ConstElemExp *, double, double *);

int
const_elem_exp_squared_norm_param_grad(const struct ConstElemExp *,
                                     double, double *);
double const_elem_exp_integrate(const struct ConstElemExp *);
double const_elem_exp_integrate_weighted(const struct ConstElemExp *);
double const_elem_exp_inner(const struct ConstElemExp *,const struct ConstElemExp *);
int const_elem_exp_axpy(double, const struct ConstElemExp *,struct ConstElemExp *);
struct ConstElemExp *
const_elem_exp_prod(const struct ConstElemExp *,const struct ConstElemExp *);
double const_elem_exp_norm(const struct ConstElemExp *);
double const_elem_exp_max(const struct ConstElemExp *, double *);
double const_elem_exp_min(const struct ConstElemExp *, double *);
double const_elem_exp_absmax(const struct ConstElemExp *,void *,size_t,void *);

struct ConstElemExp *
const_elem_exp_approx(struct ConstElemExpAopts *, struct Fwrap *);

void const_elem_exp_scale(double, struct ConstElemExp *);
void const_elem_exp_flip_sign(struct ConstElemExp *);
void const_elem_exp_orth_basis(size_t,struct ConstElemExp **, struct ConstElemExpAopts *);
struct ConstElemExp * 
const_elem_exp_zero(const struct ConstElemExpAopts *, int);
struct ConstElemExp *
const_elem_exp_constant(double,
                      const struct ConstElemExpAopts *);
struct ConstElemExp * 
const_elem_exp_linear(double, double,
                      const struct ConstElemExpAopts *);
int const_elem_exp_linear_update(struct ConstElemExp *, double, double);
struct ConstElemExp * 
const_elem_exp_quadratic(double, double, const struct ConstElemExpAopts *);

double const_elem_exp_get_lb(struct ConstElemExp *);
double const_elem_exp_get_ub(struct ConstElemExp *);
struct ConstElemExp *
const_elem_exp_onezero(size_t, double *,
                     struct ConstElemExpAopts *);

void print_const_elem_exp(const struct ConstElemExp *, size_t, void *, FILE *);

void const_elem_exp_savetxt(const struct ConstElemExp *,FILE *, size_t);
struct ConstElemExp * const_elem_exp_loadtxt(FILE *);//, size_t);

#endif
