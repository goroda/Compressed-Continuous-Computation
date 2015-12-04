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

/** \file piecewisepoly.h
 * Provides header files and structure definitions for functions in piecewisepoly.c
 */

#ifndef PIECEWISEPOLY_H
#define PIECEWISEPOLY_H

#include <stdlib.h>

/** \struct PiecewisePoly
 * \brief Tree structure to represent piecewise polynomials
 * \var PiecewisePoly::leaf
 * 1 if leaf 0 otherwise
 * \var PiecewisePoly::nbranches
 * number of branches extending from current root. may be unspecified if leaf=1
 * \var PiecewisePoly::branches
 * branches from root. If it is a leaf then branches=NULL
 * \var PiecewisePoly::ope
 * Polynomial if leaf, NULL otherwise
*/
struct PiecewisePoly
{   
    int leaf;
    size_t nbranches;
    struct PiecewisePoly ** branches;
    struct OrthPolyExpansion * ope; 
};

struct PwPolyAdaptOpts
{
    enum poly_type ptype;
    size_t maxorder;
    double minsize;  // I think these can conflict
    size_t coeff_check;
    double epsilon;
    size_t nregions;
    double * pts; // could be null then evenly spaced out, (nregions+1,)

    void * other;
};

//allocation and deallocation
struct PiecewisePoly * piecewise_poly_alloc();
struct PiecewisePoly ** piecewise_poly_array_alloc(size_t);
struct PiecewisePoly * piecewise_poly_copy(struct PiecewisePoly *);
void piecewise_poly_free(struct PiecewisePoly *);
void piecewise_poly_array_free(struct PiecewisePoly **, size_t);

struct PiecewisePoly * 
piecewise_poly_constant(double, enum poly_type, double, double) ;
struct PiecewisePoly * 
piecewise_poly_linear(double, double, enum poly_type, double,  double);
struct PiecewisePoly * 
piecewise_poly_quadratic(double,double,double, enum poly_type, double, double);
void piecewise_poly_split(struct PiecewisePoly *, double);
void piecewise_poly_splitn(struct PiecewisePoly *, size_t, double *);

//basic functions to extract information
int piecewise_poly_isflat(struct PiecewisePoly *);
double piecewise_poly_lb(struct PiecewisePoly *);
double piecewise_poly_ub(struct PiecewisePoly *);
void piecewise_poly_nregions_base(size_t *, struct PiecewisePoly *);
size_t piecewise_poly_nregions(struct PiecewisePoly *);
void piecewise_poly_boundaries(struct PiecewisePoly *,size_t *,double **,size_t *);

//operations using one piecewise poly
double piecewise_poly_eval(struct PiecewisePoly *, double);
void piecewise_poly_scale(double, struct PiecewisePoly *);
struct PiecewisePoly * piecewise_poly_deriv(struct PiecewisePoly *);
double piecewise_poly_integrate(struct PiecewisePoly *);
double * piecewise_poly_real_roots(struct PiecewisePoly *, size_t *);
double piecewise_poly_max(struct PiecewisePoly *, double *);
double piecewise_poly_min(struct PiecewisePoly *, double *);
double piecewise_poly_absmax(struct PiecewisePoly *, double *);
double piecewise_poly_norm(struct PiecewisePoly *);
void piecewise_poly_flip_sign(struct PiecewisePoly *);


//operations to modfiy/generate modfied piecewise polynomials
void piecewise_poly_copy_leaves(struct PiecewisePoly *, struct PiecewisePoly ** , size_t *);
void piecewise_poly_ref_leaves(struct PiecewisePoly *, struct PiecewisePoly ** , size_t *);
void piecewise_poly_flatten(struct PiecewisePoly *);
struct PiecewisePoly * 
piecewise_poly_finer_grid(struct PiecewisePoly *, size_t, double *);

//operations using two piecewise polynomials
void piecewise_poly_match(struct PiecewisePoly *, struct PiecewisePoly **,
                     struct PiecewisePoly *, struct PiecewisePoly **);
struct PiecewisePoly *
piecewise_poly_prod(struct PiecewisePoly *, struct PiecewisePoly *);
double piecewise_poly_inner(struct PiecewisePoly *, struct PiecewisePoly *);
struct PiecewisePoly *
piecewise_poly_daxpby(double, struct PiecewisePoly *,
                       double, struct PiecewisePoly *);
struct PiecewisePoly *
piecewise_poly_matched_daxpby(double, struct PiecewisePoly *,
                              double,struct PiecewisePoly *);
struct PiecewisePoly *
piecewise_poly_matched_prod(struct PiecewisePoly *,struct PiecewisePoly *);


// Approximation
struct PiecewisePoly *
piecewise_poly_approx1(double (*)(double, void *), void *, double,
                        double, struct PwPolyAdaptOpts *);
struct PiecewisePoly *
piecewise_poly_approx1_adapt(
                double (*f)(double, void *), void *, double,
                double, struct PwPolyAdaptOpts *);


///////////////////////////////////////////////

struct OrthPolyExpansion * piecewise_poly_trim_left(struct PiecewisePoly **);

int piecewise_poly_check_discontinuity(struct PiecewisePoly *, 
                                       struct PiecewisePoly *, 
                                       int, double);

struct PiecewisePoly *
piecewise_poly_merge_left(struct PiecewisePoly **, struct PwPolyAdaptOpts *);


double minmod_eval(double, double *, double *, size_t,size_t, size_t);
int minmod_disc_exists(double, double *, double *, size_t,size_t, size_t);

void locate_jumps(double (*)(double, void *), void *,
                  double, double, size_t, double, double **, size_t *);

struct PiecewisePoly *
piecewise_poly_approx2(double (*)(double, void *), void *, double,
                        double, struct PwPolyAdaptOpts *);
// serialization and printing
unsigned char * 
serialize_piecewise_poly(unsigned char *, struct PiecewisePoly *, size_t *); 
unsigned char *
deserialize_piecewise_poly(unsigned char *, struct PiecewisePoly ** ); 


void print_piecewise_poly(struct PiecewisePoly * pw, size_t, void *);

#endif


