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



/** \file polynomials.h
 * Provides header files and structure definitions for functions in in polynomials.c
 */

#ifndef POLYNOMIALS_H
#define POLYNOMIALS_H

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

#include <stddef.h>
#include <stdio.h>
#include <complex.h>

#include "fwrap.h"
#include "quadrature.h"


struct SpaceMapping;
double space_mapping_map(struct SpaceMapping *, double);
double space_mapping_map_inverse(struct SpaceMapping *, double);
double space_mapping_map_deriv(struct SpaceMapping *, double);
double space_mapping_map_inverse_deriv(struct SpaceMapping *, double);
    
enum coeff_decay_type {NONE,ALGEBRAIC,EXPONENTIAL};

/** \enum poly_type
 * contains LEGENDRE, CHEBYSHEV, STANDARD, HERMITE
 */
enum poly_type {LEGENDRE, CHEBYSHEV, HERMITE, STANDARD, FOURIER};

struct OpeOpts;
struct OpeOpts * ope_opts_alloc(enum poly_type);
void ope_opts_free(struct OpeOpts *);
void ope_opts_free_deep(struct OpeOpts **);
void ope_opts_set_start(struct OpeOpts *, size_t);
size_t ope_opts_get_start(const struct OpeOpts *);
void ope_opts_set_maxnum(struct OpeOpts *, size_t);
size_t ope_opts_get_maxnum(const struct OpeOpts *);
void ope_opts_set_coeffs_check(struct OpeOpts *, size_t);
void ope_opts_set_tol(struct OpeOpts *, double);
void ope_opts_set_lb(struct OpeOpts *, double);
double ope_opts_get_lb(const struct OpeOpts *);
void ope_opts_set_ub(struct OpeOpts *, double);
double ope_opts_get_ub(const struct OpeOpts *);
void ope_opts_set_mean_and_std(struct OpeOpts *, double, double);
void ope_opts_set_ptype(struct OpeOpts *, enum poly_type);
enum poly_type ope_opts_get_ptype(const struct OpeOpts *);
void ope_opts_set_qrule(struct OpeOpts *, enum quad_rule);
size_t ope_opts_get_nparams(const struct OpeOpts *);
void ope_opts_set_nparams(struct OpeOpts *, size_t);
void ope_opts_set_kristoffel_weight(struct OpeOpts *, int);


/** \struct StandardPoly
 * \brief structure to represent standard polynomials in the monomial basis
 * \var StandardPoly::ptype
 * polynomial type, always STANDARD
 * \var StandardPoly::num_poly
 * number of basis functions
 * \var StandardPoly::lower_bound
 * lower bound of input space
 * \var StandardPoly::upper_bound
 * upper bound of input space
 * \var StandardPoly::coeff
 * coefficients of basis functions (low degree to high degree order)
 */
struct StandardPoly
{
    // polynomial with basis 1 x x^2 x^3 ....
    enum poly_type ptype; 
    size_t num_poly;
    double lower_bound;
    double upper_bound;
    double * coeff; // in low degree to high degree order
};
struct StandardPoly * standard_poly_init(size_t, double, double);
struct StandardPoly * standard_poly_deriv(struct StandardPoly *);
void standard_poly_free (struct StandardPoly *);

/////////////////////////////////////////////////


/** \struct OrthPoly
 * \brief Orthogonal polynomial
 * \var OrthPoly::ptype
 * polynomial type
 * \var OrthPoly::an
 * *a* from three term recurrence of the polynomial
 * \var OrthPoly::bn
 * *b* from three term recurrence of the polynomial
 * \var OrthPoly::cn
 * *c* from three term recurrence of the polynomial
 * \var OrthPoly::lower
 * lower bound of polynomial
 * \var OrthPoly::upper
 * upper bound of polynomial
 * \var OrthPoly::const_term
 * value of the constant polynomial in the family
 * \var OrthPoly::lin_coeff
 * value of the slope of the linear polynomial in family
 * \var OrthPoly::lin_const
 * value of the offset of the linear polynomial in family
 * \var OrthPoly::norm
 * normalizing constants for polynomial family
 */
struct OrthPoly
{
    enum poly_type ptype;
    double (*an)(size_t);
    double (*bn)(size_t);
    double (*cn)(size_t);
    
    double lower; 
    double upper; 

    double const_term;
    double lin_coeff;
    double lin_const;

    double (*norm)(size_t);

};

struct OrthPoly * init_cheb_poly(void);
struct OrthPoly * init_leg_poly(void);
void free_orth_poly(struct OrthPoly *);
unsigned char * serialize_orth_poly(struct OrthPoly *);
struct OrthPoly * deserialize_orth_poly(unsigned char *);

struct StandardPoly * orth_to_standard_poly(struct OrthPoly *, size_t);

/* double eval_orth_poly_wp(const struct OrthPoly *, double, double, *\/ */
/*                              size_t, double); */

double deriv_legen(double, size_t);
double * deriv_legen_upto(double, size_t);
double orth_poly_eval(const struct OrthPoly *, size_t, double);

/** \struct OrthPolyExpansion
 * \brief structure to represent an expansion of orthogonal polynomials
 * \var OrthPolyExpansion::p
 * orthogonal polynomial
 * \var OrthPolyExpansion::num_poly
 * number of basis functions
 * \var OrthPolyExpansion::lower_bound
 * lower bound of input space
 * \var OrthPolyExpansion::upper_bound
 * upper bound of input space
 * \var OrthPolyExpansion::coeff
 * coefficients of basis functions
 * \var OrthPolyExpansion::nalloc
 * size of double allocated
 */
struct OrthPolyExpansion{

    struct OrthPoly * p;
    size_t num_poly; // maximum order = num_poly-1

    // lower and upper bound of evaluation
    double lower_bound; 
    double upper_bound; 
    double * coeff;

    /* double _Complex * ccoeff; */
    /* #ifdef __cplusplus */
    /* double _Complex * ccoeff;     */
    /* #else */
    double complex * ccoeff;
    /* #endif */
    /* void * ccoeff; */
    
    size_t nalloc; // number of coefficients allocated for efficiency

    struct SpaceMapping * space_transform;
    int kristoffel_eval;

};

#define OPECALLOC 50

size_t orth_poly_expansion_get_num_poly(const struct OrthPolyExpansion *);
size_t orth_poly_expansion_get_num_params(const struct OrthPolyExpansion *);
double orth_poly_expansion_get_lb(const struct OrthPolyExpansion *);
double orth_poly_expansion_get_ub(const struct OrthPolyExpansion *);
struct OrthPolyExpansion * 
orth_poly_expansion_init(enum poly_type, size_t, double, double);
struct OrthPolyExpansion * 
orth_poly_expansion_init_from_opts(const struct OpeOpts *, size_t);
struct OrthPolyExpansion * 
orth_poly_expansion_create_with_params(struct OpeOpts *, size_t, const double *);
size_t orth_poly_expansion_get_params(const struct OrthPolyExpansion *, double *);
double * orth_poly_expansion_get_params_ref(const struct OrthPolyExpansion *, size_t *);

int
orth_poly_expansion_update_params(struct OrthPolyExpansion *,
                                  size_t, const double *);

struct OrthPolyExpansion * orth_poly_expansion_copy(const struct OrthPolyExpansion *);

enum poly_type 
orth_poly_expansion_get_ptype(const struct OrthPolyExpansion *);

struct OrthPolyExpansion * 
orth_poly_expansion_zero(struct OpeOpts *, int);
    
struct OrthPolyExpansion * 
orth_poly_expansion_constant(double, struct OpeOpts *);

struct OrthPolyExpansion * 
orth_poly_expansion_linear(double, double, struct OpeOpts *);

int
orth_poly_expansion_linear_update(struct OrthPolyExpansion *, double, double);
    
struct OrthPolyExpansion * 
orth_poly_expansion_quadratic(double, double, struct OpeOpts *);

struct OrthPolyExpansion * 
orth_poly_expansion_genorder(size_t,struct OpeOpts*);
void
orth_poly_expansion_orth_basis(size_t, struct OrthPolyExpansion **, struct OpeOpts *);

double orth_poly_expansion_deriv_eval(const struct OrthPolyExpansion *, double);


struct OrthPolyExpansion *
orth_poly_expansion_deriv(const struct OrthPolyExpansion *);
struct OrthPolyExpansion *
orth_poly_expansion_dderiv(const struct OrthPolyExpansion *);
struct OrthPolyExpansion * orth_poly_expansion_dderiv_periodic(const struct OrthPolyExpansion * );

void orth_poly_expansion_free(struct OrthPolyExpansion *);

unsigned char * 
serialize_orth_poly_expansion(unsigned char *,
        struct OrthPolyExpansion *, size_t *);
unsigned char *
deserialize_orth_poly_expansion(unsigned char *, 
            struct OrthPolyExpansion ** );

void orth_poly_expansion_savetxt(const struct OrthPolyExpansion *,
                                 FILE *, size_t);
struct OrthPolyExpansion *
orth_poly_expansion_loadtxt(FILE *);

struct StandardPoly * 
orth_poly_expansion_to_standard_poly(struct OrthPolyExpansion *);

int orth_poly_expansion_arr_eval(size_t,
                                 struct OrthPolyExpansion **, 
                                 double, double *);

int orth_poly_expansion_arr_evalN(size_t,
                                  struct OrthPolyExpansion **,
                                  size_t,
                                  const double *, size_t,
                                  double *, size_t);
    
double chebyshev_poly_expansion_eval(const struct OrthPolyExpansion *, double);

double orth_poly_expansion_eval(const struct OrthPolyExpansion *, double);
double orth_poly_expansion_get_kristoffel_weight(const struct OrthPolyExpansion *, double);


void orth_poly_expansion_evalN(const struct OrthPolyExpansion *, size_t,
                               const double *, size_t, double *, size_t);
int orth_poly_expansion_param_grad_eval(
    const struct OrthPolyExpansion *, size_t, const double *, double *);
double orth_poly_expansion_param_grad_eval2(const struct OrthPolyExpansion *, double, double *);

int
orth_poly_expansion_squared_norm_param_grad(const struct OrthPolyExpansion *,
                                            double, double *);
double
orth_poly_expansion_rkhs_squared_norm(const struct OrthPolyExpansion *,
                                      enum coeff_decay_type,
                                      double);
int
orth_poly_expansion_rkhs_squared_norm_param_grad(const struct OrthPolyExpansion *,
                                                 double, enum coeff_decay_type,
                                                 double, double *);


void orth_poly_expansion_round(struct OrthPolyExpansion **);
void orth_poly_expansion_roundt(struct OrthPolyExpansion **,double);

void orth_poly_expansion_approx (double (*)(double,void *), void *, 
                                 struct OrthPolyExpansion *);
int
orth_poly_expansion_approx_vec(struct OrthPolyExpansion *,
                               struct Fwrap *,
                               const struct OpeOpts *);
                               
    /* int (*A)(size_t, double *,double *,void *), void *, */
    /* struct OrthPolyExpansion *); */


struct OrthPolyExpansion * 
orth_poly_expansion_approx_adapt(const struct OpeOpts *,struct Fwrap *);

struct OrthPolyExpansion * 
orth_poly_expansion_randu(enum poly_type, size_t, double, double);

double cheb_integrate2(const struct OrthPolyExpansion *);
double legendre_integrate(const struct OrthPolyExpansion *);

struct OrthPolyExpansion *
orth_poly_expansion_prod(const struct OrthPolyExpansion *,
                         const struct OrthPolyExpansion *);
struct OrthPolyExpansion *
orth_poly_expansion_sum_prod(size_t, size_t, 
        struct OrthPolyExpansion **, size_t,
        struct OrthPolyExpansion **);

double orth_poly_expansion_integrate(const struct OrthPolyExpansion *);
double orth_poly_expansion_integrate_weighted(const struct OrthPolyExpansion *);
double orth_poly_expansion_inner_w(const struct OrthPolyExpansion *,
                                   const struct OrthPolyExpansion *);
double orth_poly_expansion_inner(const struct OrthPolyExpansion *,
                                 const struct OrthPolyExpansion *);
double orth_poly_expansion_norm_w(const struct OrthPolyExpansion * p);
double orth_poly_expansion_norm(const struct OrthPolyExpansion * p);

void orth_poly_expansion_flip_sign(struct OrthPolyExpansion *);

void orth_poly_expansion_scale(double, struct OrthPolyExpansion *);
struct OrthPolyExpansion *
orth_poly_expansion_daxpby(double, struct OrthPolyExpansion *,
                           double, struct OrthPolyExpansion *);

int orth_poly_expansion_axpy(double a, struct OrthPolyExpansion * x,
                        struct OrthPolyExpansion * y);
void
orth_poly_expansion_sum3_up(double, struct OrthPolyExpansion *,
                           double, struct OrthPolyExpansion *,
                           double, struct OrthPolyExpansion *);

struct OrthPolyExpansion *
orth_poly_expansion_lin_comb(size_t, size_t, 
                             struct OrthPolyExpansion **, size_t,
                             const double *);


/////////////////////////////////////////////////////////////
// Algorithms
//

double * 
legendre_expansion_real_roots(struct OrthPolyExpansion *, size_t *);
double * standard_poly_real_roots(struct StandardPoly *, size_t *);
double *
orth_poly_expansion_real_roots(struct OrthPolyExpansion *, size_t *);
double orth_poly_expansion_max(struct OrthPolyExpansion *, double *);
double orth_poly_expansion_min(struct OrthPolyExpansion *, double *);
double orth_poly_expansion_absmax(struct OrthPolyExpansion *, double *,void*);

/////////////////////////////////////////////////////////
// Utilities
void print_orth_poly_expansion(struct OrthPolyExpansion *, size_t, void *, FILE*);

#endif
