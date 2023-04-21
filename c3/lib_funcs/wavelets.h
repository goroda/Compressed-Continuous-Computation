// Copyright (c) 2023, University of Michigan

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



/** \file wavelets.h
 * Provides header files and structure definitions for functions in wavelets.c
 */

#ifndef WAVELETS_H
#define WAVELETS_H

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

#include <stddef.h>
#include <stdio.h>
#include <complex.h>

#include "fwrap.h"
#include "quadrature.h"



struct WaveletOpts;
struct WaveletOpts * wavelet_opts_alloc(enum poly_type);
void wavelet_opts_free(struct WaveletOpts *);
void wavelet_opts_free_deep(struct WaveletOpts **);
void wavelet_opts_set_start(struct WaveletOpts *, size_t);
size_t wavelet_opts_get_start(const struct WaveletOpts *);
void wavelet_opts_set_maxnum(struct WaveletOpts *, size_t);
size_t wavelet_opts_get_maxnum(const struct WaveletOpts *);
int wavelet_opts_get_final_round(const struct WaveletOpts * ope);
void wavelet_opts_set_final_round(struct WaveletOpts * ope, int);
void wavelet_opts_set_coeffs_check(struct WaveletOpts *, size_t);
void wavelet_opts_set_tol(struct WaveletOpts *, double);
void wavelet_opts_set_lb(struct WaveletOpts *, double);
double wavelet_opts_get_lb(const struct WaveletOpts *);
void wavelet_opts_set_ub(struct WaveletOpts *, double);
double wavelet_opts_get_ub(const struct WaveletOpts *);
void wavelet_opts_set_mean_and_std(struct WaveletOpts *, double, double);
void wavelet_opts_set_ptype(struct WaveletOpts *, enum poly_type);
enum poly_type wavelet_opts_get_ptype(const struct WaveletOpts *);
void wavelet_opts_set_qrule(struct WaveletOpts *, enum quad_rule);
size_t wavelet_opts_get_nparams(const struct WaveletOpts *);
void wavelet_opts_set_nparams(struct WaveletOpts *, size_t);
void wavelet_opts_set_kristoffel_weight(struct WaveletOpts *, int);


/////////////////////////////////////////////////
/** \struct WaveletExpansion
 * \brief structure to represent a multi-resolution wavelet expansion
 * \var WaveletExpansion::p
 * orthogonal polynomial
 * \var WaveletExpansion::num_poly
 * number of basis functions
 * \var WaveletExpansion::lower_bound
 * lower bound of input space
 * \var WaveletExpansion::upper_bound
 * upper bound of input space
 * \var WaveletExpansion::coeff
 * coefficients of basis functions
 * \var WaveletExpansion::nalloc
 * size of double allocated
 */
struct WaveletExpansion{

    struct OrthPoly * p;
    size_t maximum_order; // maximum order = num_poly-1
	size_t maximum_res; //k=0,1,\ldots, (2^k(l), 2^{k}(l+1)), l=0,\ldots,2^k-1)

    // lower and upper bound of evaluation
    double lower_bound; 
    double upper_bound; 
    double * coeff;
	
    size_t nalloc; // number of coefficients allocated for efficiency
    struct SpaceMapping * space_transform;
};

#define WAVELETCALLOC 50

size_t wavelet_expansion_get_num_params(const struct WaveletExpansion *);
double wavelet_expansion_get_lb(const struct WaveletExpansion *);
double wavelet_expansion_get_ub(const struct WaveletExpansion *);
struct WaveletExpansion * 
wavelet_expansion_init(enum poly_type, size_t, double, double);
struct WaveletExpansion * 
wavelet_expansion_init_from_opts(const struct WaveletOpts *, size_t);
struct WaveletExpansion * 
wavelet_expansion_create_with_params(struct WaveletOpts *, size_t, const double *);
void
wavelet_expansion_pad_params(struct WaveletExpansion * p,
                               size_t num_params);;
size_t wavelet_expansion_get_params(const struct WaveletExpansion *, double *);
double * wavelet_expansion_get_params_ref(const struct WaveletExpansion *, size_t *);

int
wavelet_expansion_update_params(struct WaveletExpansion *,
                                  size_t, const double *);

struct WaveletExpansion * wavelet_expansion_copy(const struct WaveletExpansion *);

enum poly_type 
wavelet_expansion_get_ptype(const struct WaveletExpansion *);

struct WaveletExpansion * 
wavelet_expansion_zero(struct WaveletOpts *, int);
    
struct WaveletExpansion * 
wavelet_expansion_constant(double, struct WaveletOpts *);

struct WaveletExpansion * 
wavelet_expansion_linear(double, double, struct WaveletOpts *);

int
wavelet_expansion_linear_update(struct WaveletExpansion *, double, double);
    
struct WaveletExpansion * 
wavelet_expansion_quadratic(double, double, struct WaveletOpts *);

struct WaveletExpansion * 
wavelet_expansion_genorder(size_t,struct WaveletOpts*);
void
wavelet_expansion_orth_basis(size_t, struct WaveletExpansion **, struct WaveletOpts *);

double wavelet_expansion_deriv_eval(const struct WaveletExpansion *, double);


struct WaveletExpansion *
wavelet_expansion_deriv(const struct WaveletExpansion *);
struct WaveletExpansion *
wavelet_expansion_dderiv(const struct WaveletExpansion *);
struct WaveletExpansion * wavelet_expansion_dderiv_periodic(const struct WaveletExpansion * );

void wavelet_expansion_free(struct WaveletExpansion *);

unsigned char * 
serialize_wavelet(unsigned char *,
        struct WaveletExpansion *, size_t *);
unsigned char *
deserialize_wavelet(unsigned char *, 
            struct WaveletExpansion ** );

void wavelet_expansion_savetxt(const struct WaveletExpansion *,
                                 FILE *, size_t);
struct WaveletExpansion *
wavelet_expansion_loadtxt(FILE *);

struct StandardPoly * 
wavelet_expansion_to_standard_poly(struct WaveletExpansion *);

int wavelet_expansion_arr_eval(size_t,
                                 struct WaveletExpansion **, 
                                 double, double *);

int wavelet_expansion_arr_evalN(size_t,
                                  struct WaveletExpansion **,
                                  size_t,
                                  const double *, size_t,
                                  double *, size_t);
    
double chebyshev_poly_expansion_eval(const struct WaveletExpansion *, double);

double wavelet_expansion_eval(const struct WaveletExpansion *, double);
double wavelet_expansion_get_kristoffel_weight(const struct WaveletExpansion *, double);


void wavelet_expansion_evalN(const struct WaveletExpansion *, size_t,
                               const double *, size_t, double *, size_t);
int wavelet_expansion_param_grad_eval(
    const struct WaveletExpansion *, size_t, const double *, double *);
double wavelet_expansion_param_grad_eval2(const struct WaveletExpansion *, double, double *);

size_t
wavelet_expansion_param_grad_inner(const struct WaveletExpansion * a,
                                     const struct WaveletExpansion * b,
                                     double *inner_vals);

int
wavelet_expansion_squared_norm_param_grad(const struct WaveletExpansion *,
                                            double, double *);
double
wavelet_expansion_rkhs_squared_norm(const struct WaveletExpansion *,
                                      enum coeff_decay_type,
                                      double);
int
wavelet_expansion_rkhs_squared_norm_param_grad(const struct WaveletExpansion *,
                                                 double, enum coeff_decay_type,
                                                 double, double *);


void wavelet_expansion_round(struct WaveletExpansion **);
void wavelet_expansion_roundt(struct WaveletExpansion **,double);

void wavelet_expansion_approx (double (*)(double,void *), void *, 
                                 struct WaveletExpansion *);
int
wavelet_expansion_approx_vec(struct WaveletExpansion *,
                               struct Fwrap *,
                               const struct WaveletOpts *);
                               
    /* int (*A)(size_t, double *,double *,void *), void *, */
    /* struct WaveletExpansion *); */


struct WaveletExpansion * 
wavelet_expansion_approx_adapt(const struct WaveletOpts *,struct Fwrap *);

struct WaveletExpansion * 
wavelet_expansion_randu(enum poly_type, size_t, double, double);

double cheb_integrate2(const struct WaveletExpansion *);
double legendre_integrate(const struct WaveletExpansion *);

struct WaveletExpansion *
wavelet_expansion_prod(const struct WaveletExpansion *,
                         const struct WaveletExpansion *);
struct WaveletExpansion *
wavelet_expansion_sum_prod(size_t, size_t, 
        struct WaveletExpansion **, size_t,
        struct WaveletExpansion **);

double wavelet_expansion_integrate(const struct WaveletExpansion *);
double wavelet_expansion_integrate_weighted(const struct WaveletExpansion *);
double wavelet_expansion_inner_w(const struct WaveletExpansion *,
                                   const struct WaveletExpansion *);
double wavelet_expansion_inner(const struct WaveletExpansion *,
                                 const struct WaveletExpansion *);
double wavelet_expansion_norm_w(const struct WaveletExpansion * p);
double wavelet_expansion_norm(const struct WaveletExpansion * p);

void wavelet_expansion_flip_sign(struct WaveletExpansion *);

void wavelet_expansion_scale(double, struct WaveletExpansion *);
struct WaveletExpansion *
wavelet_expansion_daxpby(double, struct WaveletExpansion *,
                           double, struct WaveletExpansion *);

int wavelet_expansion_axpy(double a, struct WaveletExpansion * x,
                        struct WaveletExpansion * y);
void
wavelet_expansion_sum3_up(double, struct WaveletExpansion *,
                           double, struct WaveletExpansion *,
                           double, struct WaveletExpansion *);

struct WaveletExpansion *
wavelet_expansion_lin_comb(size_t, size_t, 
                             struct WaveletExpansion **, size_t,
                             const double *);


/////////////////////////////////////////////////////////////
// Algorithms
//

double * 
legendre_expansion_real_roots(struct WaveletExpansion *, size_t *);
double * standard_poly_real_roots(struct StandardPoly *, size_t *);
double *
wavelet_expansion_real_roots(struct WaveletExpansion *, size_t *);
double wavelet_expansion_max(struct WaveletExpansion *, double *);
double wavelet_expansion_min(struct WaveletExpansion *, double *);
double wavelet_expansion_absmax(struct WaveletExpansion *, double *,void*);

/////////////////////////////////////////////////////////
// Utilities
void print_wavelet(struct WaveletExpansion *, size_t, void *, FILE*);

#endif
