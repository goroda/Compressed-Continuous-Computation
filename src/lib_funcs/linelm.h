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






/** \file linelm.h
 * Provides header files and structure definitions for functions in linelm.c
 */

#ifndef LINELM_H
#define LINELM_H

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

#include <stddef.h>
#include <stdio.h>

#include "fwrap.h"

struct LinElemExpAopts;
struct LinElemExpAopts * lin_elem_exp_aopts_alloc(size_t, double * ydata);
struct LinElemExpAopts *
lin_elem_exp_aopts_alloc_adapt(size_t,double *,double,double,double, double);
void lin_elem_exp_aopts_free(struct LinElemExpAopts *);
void lin_elem_exp_aopts_free_deep(struct LinElemExpAopts **);
double lin_elem_exp_aopts_get_lb(const struct LinElemExpAopts *);
double lin_elem_exp_aopts_get_ub(const struct LinElemExpAopts *);
void lin_elem_exp_aopts_set_nodes(struct LinElemExpAopts *,
                                  size_t, double *);
void lin_elem_exp_aopts_set_nodes_copy(struct LinElemExpAopts *,
				       size_t, const double * ydata);

void lin_elem_exp_aopts_set_adapt(struct LinElemExpAopts *,double, double);
void lin_elem_exp_aopts_set_delta(struct LinElemExpAopts *, double);
void lin_elem_exp_aopts_set_hmin(struct LinElemExpAopts *, double);

size_t lin_elem_exp_aopts_get_num_nodes(const struct LinElemExpAopts *);
size_t lin_elem_exp_aopts_get_nparams(const struct LinElemExpAopts *);
void   lin_elem_exp_aopts_set_nparams(struct LinElemExpAopts*, size_t);

/////////////////////////////////////////////////////////////////////

/** \struct LinElemExp
 * \brief structure to represent an expansion of linear elements
 * \var LinElemExp::num_nodes
 * number of basis functions or nodes
 * \var LinElemExp::nodes
 * nodes of basis functions
 * \var LinElemExp::coeff
 * coefficients of basis functions
 * \var LinElemExp::diff
 * difference between nodes
 * \var LinElemExp::inner
 * inner products of basis functions
 */
struct LinElemExp{

    size_t num_nodes;
    double * nodes;
    double * coeff;
    double * diff;
    double * inner;

    // FFT-based periodic boundary condition information
    double * Lp;
};

size_t lin_elem_exp_get_num_nodes(const struct LinElemExp *);
size_t lin_elem_exp_get_num_params(const struct LinElemExp *);
struct LinElemExp * lin_elem_exp_alloc(void);
struct LinElemExp * lin_elem_exp_copy(struct LinElemExp *);
void lin_elem_exp_free(struct LinElemExp *);
struct LinElemExp * lin_elem_exp_init(size_t, double *, double *);
struct LinElemExp *
lin_elem_exp_create_with_params(struct LinElemExpAopts *,
                                size_t, const double *);
int
lin_elem_exp_update_params(struct LinElemExp *,
                           size_t, const double *);
size_t lin_elem_exp_get_params(const struct LinElemExp *, double *);
double * lin_elem_exp_get_params_ref(const struct LinElemExp *, size_t *);

unsigned char *
serialize_lin_elem_exp(unsigned char *, struct LinElemExp *,size_t *);
unsigned char * deserialize_lin_elem_exp(unsigned char *, 
                                         struct LinElemExp **);
double lin_elem_exp_eval(const struct LinElemExp *, double);
double lin_elem_exp_deriv_eval(const struct LinElemExp *, double);
void lin_elem_exp_evalN(const struct LinElemExp *, size_t,
                        const double *, size_t, double *, size_t);
double lin_elem_exp_get_nodal_val(const struct LinElemExp *, size_t);
struct LinElemExp * lin_elem_exp_deriv(const struct LinElemExp *);
struct LinElemExp * lin_elem_exp_dderiv(const struct LinElemExp *);
struct LinElemExp * lin_elem_exp_dderiv_periodic(struct LinElemExp *);
int lin_elem_exp_param_grad_eval(
    struct LinElemExp *, size_t, const double *, double *);
double lin_elem_exp_param_grad_eval2(struct LinElemExp *, double, double *);

int
lin_elem_exp_squared_norm_param_grad(const struct LinElemExp *,
                                     double, double *);
double lin_elem_exp_integrate(const struct LinElemExp *);
double lin_elem_exp_integrate_weighted(const struct LinElemExp *);
double lin_elem_exp_inner(const struct LinElemExp *,const struct LinElemExp *);
int lin_elem_exp_axpy(double, const struct LinElemExp *,struct LinElemExp *);
struct LinElemExp *
lin_elem_exp_prod(const struct LinElemExp *, const struct LinElemExp *);
double lin_elem_exp_norm(const struct LinElemExp *);
double lin_elem_exp_max(const struct LinElemExp *, double *);
double lin_elem_exp_min(const struct LinElemExp *, double *);
double lin_elem_exp_absmax(const struct LinElemExp *,void *,size_t,void *);

double lin_elem_exp_err_est(struct LinElemExp *, double *, short,short);


struct LinElemXY;
void lin_elem_adapt(struct Fwrap *,
                    double, double,double, double,
                    double, double,struct LinElemXY ** x);
double lin_elem_xy_get_x(struct LinElemXY *);
double lin_elem_xy_get_y(struct LinElemXY *);
struct LinElemXY * lin_elem_xy_next(struct LinElemXY *);
void lin_elem_xy_free(struct LinElemXY *);

struct LinElemExp *
lin_elem_exp_approx(struct LinElemExpAopts *, struct Fwrap *);

void lin_elem_exp_scale(double, struct LinElemExp *);
void lin_elem_exp_flip_sign(struct LinElemExp *);
void lin_elem_exp_orth_basis(size_t,struct LinElemExp **, struct LinElemExpAopts *);
struct LinElemExp * 
lin_elem_exp_zero(const struct LinElemExpAopts *, int);
struct LinElemExp *
lin_elem_exp_constant(double,
                      const struct LinElemExpAopts *);
struct LinElemExp * 
lin_elem_exp_linear(double, double,
                    const struct LinElemExpAopts *);
int
lin_elem_exp_linear_update(struct LinElemExp *, double, double);
struct LinElemExp * 
lin_elem_exp_quadratic(double, double,
                       const struct LinElemExpAopts *);
double lin_elem_exp_get_lb(struct LinElemExp *);
double lin_elem_exp_get_ub(struct LinElemExp *);
struct LinElemExp *
lin_elem_exp_onezero(size_t, double *,
                     struct LinElemExpAopts *);

void print_lin_elem_exp(const struct LinElemExp *, size_t, void *, FILE *);

void lin_elem_exp_savetxt(const struct LinElemExp *,FILE *, size_t);
struct LinElemExp * lin_elem_exp_loadtxt(FILE *);//, size_t);

#endif
