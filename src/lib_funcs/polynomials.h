// Copyright (c) 2014-2015, Massachusetts Institute of Technology
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

/** \file polynomials.h
 * Provides header files and structure definitions for functions in in polynomials.c
 */

#ifndef POLYNOMIALS_H
#define POLYNOMIALS_H

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

#include <stdlib.h>

/** \enum poly_type
 * contains LEGENDRE, CHEBYSHEV, STANDARD
 */
enum poly_type {LEGENDRE, CHEBYSHEV, STANDARD};

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

struct OrthPoly * init_cheb_poly();
struct OrthPoly * init_leg_poly();
void free_orth_poly(struct OrthPoly *);
unsigned char * serialize_orth_poly(struct OrthPoly *);
struct OrthPoly * deserialize_orth_poly(unsigned char *);

struct StandardPoly * orth_to_standard_poly(struct OrthPoly *, size_t);

double eval_orth_poly_wp(const struct OrthPoly *, double, double, 
                             size_t, double);
double deriv_legen(double, size_t);
double * deriv_legen_upto(double, size_t);
double * orth_poly_deriv_upto(enum poly_type, size_t, double);
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
 */
struct OrthPolyExpansion{

    struct OrthPoly * p;
    size_t num_poly; // maximum order = num_poly-1

    // lower and upper bound of evaluation
    double lower_bound; 
    double upper_bound; 
    double * coeff;
};

struct OrthPolyExpansion * 
orth_poly_expansion_init(enum poly_type, size_t, double, double);

struct OrthPolyExpansion * 
orth_poly_expansion_copy(struct OrthPolyExpansion *);

struct OrthPolyExpansion * 
orth_poly_expansion_constant(double, enum poly_type, double, double);

struct OrthPolyExpansion * 
orth_poly_expansion_linear(double, double, enum poly_type, double, double);

struct OrthPolyExpansion * 
orth_poly_expansion_quadratic(double, double, enum poly_type, double, double);

struct OrthPolyExpansion * 
orth_poly_expansion_genorder(enum poly_type, size_t, double, double);

double orth_poly_expansion_deriv_eval(double, void *);
struct OrthPolyExpansion *
orth_poly_expansion_deriv(struct OrthPolyExpansion *);

void orth_poly_expansion_free(struct OrthPolyExpansion *);

unsigned char * 
serialize_orth_poly_expansion(unsigned char *,
        struct OrthPolyExpansion *, size_t *);
unsigned char *
deserialize_orth_poly_expansion(unsigned char *, 
            struct OrthPolyExpansion ** );

struct StandardPoly * 
orth_poly_expansion_to_standard_poly(struct OrthPolyExpansion *);


double legendre_poly_expansion_eval(struct OrthPolyExpansion *, double);
double orth_poly_expansion_eval(struct OrthPolyExpansion *, double);

void orth_poly_expansion_round(struct OrthPolyExpansion **);

void orth_poly_expansion_approx (double (*)(double,void *), void *, 
                       struct OrthPolyExpansion *);

struct OpeAdaptOpts{
    
    size_t start_num;
    size_t coeffs_check;
    double tol;
};

struct OrthPolyExpansion * orth_poly_expansion_approx_adapt(double (*A)(double,void *), 
        void *, enum poly_type, double, double, struct OpeAdaptOpts *);


double cheb_integrate2(struct OrthPolyExpansion *);
double legendre_integrate(struct OrthPolyExpansion *);


struct OrthPolyExpansion *
orth_poly_expansion_prod(struct OrthPolyExpansion *,
                         struct OrthPolyExpansion *);

double orth_poly_expansion_integrate(struct OrthPolyExpansion *);
double orth_poly_expansion_inner_w(struct OrthPolyExpansion *,
                            struct OrthPolyExpansion *);
double orth_poly_expansion_inner(struct OrthPolyExpansion *,
                            struct OrthPolyExpansion *);
double orth_poly_expansion_norm_w(struct OrthPolyExpansion * p);
double orth_poly_expansion_norm(struct OrthPolyExpansion * p);

void orth_poly_expansion_flip_sign(struct OrthPolyExpansion *);

void orth_poly_expansion_scale(double, struct OrthPolyExpansion *);
struct OrthPolyExpansion *
orth_poly_expansion_daxpby(double, struct OrthPolyExpansion *,
                           double, struct OrthPolyExpansion *);


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
double orth_poly_expansion_absmax(struct OrthPolyExpansion *, double *);

/////////////////////////////////////////////////////////
// Utilities
void print_orth_poly_expansion(struct OrthPolyExpansion *, size_t, void *);

#endif
