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

/** \file functions.h
 * Provides header files and structure definitions for functions in functions.c 
 */

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "../lib_array/array.h"

#include "polynomials.h"
#include "piecewisepoly.h"
#include "hpoly.h"
#include "linelm.h"
#include "fwrap.h"
#include "pivoting.h"

/** \enum function_class
 * contains PIECEWISE, POLYNOMIAL, RATIONAL, KERNEL:
 * only POLYNOMIAL is implemented!!!
 */
enum function_class {CONSTANT,PIECEWISE, POLYNOMIAL,
                     LINELM, RATIONAL, KERNEL};

/** \struct Interval
 * \brief A pair of lower and upper bounds
 * \var Interval::lb
 * lower bound
 * \var Interval::ub
 * upper bound
 */
struct Interval
{
    double lb;
    double ub;
};

/** \struct GenericFunction
 * \brief Interface between the world and specific functions such as polynomials, radial
 * basis functions (future), etc (future)
 * \var GenericFunction::dim
 * dimension of the function
 * \var GenericFunction::fc
 * type of function
 * \var GenericFunction::sub_type
 * sub type of function
 * \var GenericFunction::f
 * function
 * \var GenericFunction::fargs
 * function arguments
 */
struct GenericFunction {
    
    size_t dim;
    enum function_class fc;
    /* union */
    /* { */
    /*     enum poly_type ptype; */
    /* } sub_type; */
    void * f;
    void * fargs;
};

struct GenericFunction * generic_function_alloc_base(size_t);
struct GenericFunction ** generic_function_array_alloc(size_t);
struct GenericFunction *
generic_function_alloc(size_t, enum function_class);
struct GenericFunction *
generic_function_copy(const struct GenericFunction *);
void generic_function_copy_pa(const struct GenericFunction *,
                              struct GenericFunction *);

void generic_function_free(struct GenericFunction *);
void generic_function_array_free(struct GenericFunction **,size_t);
unsigned char *
serialize_generic_function(unsigned char *, 
                           const struct GenericFunction *,
                           size_t *);
unsigned char *
deserialize_generic_function(unsigned char *, struct GenericFunction ** );


// special initializers
struct GenericFunction * 
generic_function_constant(double, enum function_class,void *);
struct GenericFunction * 
generic_function_linear(double, double,enum function_class, void *);
struct GenericFunction * 
generic_function_quadratic(double,double,enum function_class,void *);

struct GenericFunction * 
generic_function_poly_randu(enum poly_type,size_t, double, double);
struct GenericFunction * generic_function_deriv(const struct GenericFunction *);


// generic operations
struct GenericFunction *
generic_function_daxpby(double, const struct GenericFunction *,
                        double, const struct GenericFunction *);

double generic_function_inner(const struct GenericFunction *,
                              const struct GenericFunction *);
double generic_function_inner_sum(size_t, size_t, struct GenericFunction **, 
                         size_t, struct GenericFunction **);
double generic_function_norm(const struct GenericFunction *); 
double generic_function_norm2diff(const struct GenericFunction *, 
                                  const struct GenericFunction *);
double generic_function_array_norm2diff(
                size_t, struct GenericFunction **, size_t,
                struct GenericFunction **, size_t);

double generic_function_integral(const struct GenericFunction *);
double * 
generic_function_integral_array(size_t , size_t, struct GenericFunction ** a);


void generic_function_roundt(struct GenericFunction **, double);


double generic_function_1d_eval(const struct GenericFunction *, double);
double generic_function_1d_eval_ind(const struct GenericFunction *, size_t);
double * generic_function_1darray_eval(size_t, 
                                       struct GenericFunction **, 
                                       double);
double generic_function_1darray_eval_piv(struct GenericFunction ** f, 
                                         struct Pivot * piv);
void generic_function_1darray_eval2(size_t, 
                                    struct GenericFunction **, 
                                    double,double *);
void
generic_function_1darray_eval2_ind(size_t, 
                                   struct GenericFunction **, 
                                   size_t, double *);


struct GenericFunction *
generic_function_onezero(enum function_class, double, size_t,
                         double *, double, double);

void generic_function_array_onezero(
    struct GenericFunction **,
    size_t,
    enum function_class,
    size_t,
    size_t *,
    double *,
    void *);

struct GenericFunction *
generic_function_create_nodal(struct GenericFunction *,size_t, double *);
struct GenericFunction *
generic_function_sum_prod(size_t, size_t,  struct GenericFunction **, 
                size_t, struct GenericFunction **);
double generic_function_sum_prod_integrate(size_t, size_t,  
                struct GenericFunction **, size_t, struct GenericFunction **);
struct GenericFunction *
generic_function_prod(struct GenericFunction *, struct GenericFunction *);

double generic_function_array_norm(size_t, size_t, struct GenericFunction **);

void generic_function_flip_sign(struct GenericFunction *);
void generic_function_array_flip_sign(size_t, size_t, struct GenericFunction **);



void generic_function_weighted_sum_pa(double, struct GenericFunction *, 
            double, struct GenericFunction *, struct GenericFunction **);


struct GenericFunction *
generic_function_approximate1d(enum function_class,void *,struct Fwrap *);

// extraction functions
double generic_function_get_lower_bound(const struct GenericFunction * f);
double generic_function_get_upper_bound(const struct GenericFunction * f);
enum function_class generic_function_get_fc(const struct GenericFunction * f);

void
generic_function_sum3_up(double, struct GenericFunction *,
                         double, struct GenericFunction *,
                         double, struct GenericFunction *);

int generic_function_axpy(double, const struct GenericFunction *,
                          struct GenericFunction *);
int generic_function_array_axpy(size_t, double, struct GenericFunction **, 
            struct GenericFunction **);

struct GenericFunction **
generic_function_array_daxpby(size_t, double, size_t, 
        struct GenericFunction **, double, size_t, 
        struct GenericFunction **);

void
generic_function_array_daxpby2(size_t, double, size_t, 
        struct GenericFunction **, double, size_t, 
        struct GenericFunction **, size_t, struct GenericFunction **);

struct GenericFunction *
generic_function_lin_comb(size_t,
                          struct GenericFunction **, const double *);
struct GenericFunction * generic_function_lin_comb2(size_t, size_t, 
                struct GenericFunction **, size_t, const double *);

double generic_function_absmax(const struct GenericFunction *, double *,void *);
double generic_function_absmax_gen(const struct GenericFunction *, 
                                   void *, size_t, void *);
double generic_function_array_absmax(size_t, size_t, 
                                     struct GenericFunction **, 
                                     size_t *, double *, void *);
double 
generic_function_array_absmax_piv(size_t, size_t, 
                                  struct GenericFunction **, 
                                  struct Pivot *,
                                  void *);

void generic_function_scale(double, struct GenericFunction *);
void generic_function_array_scale(double, struct GenericFunction **, size_t);
void generic_function_kronh(int,
                            size_t, size_t, size_t, size_t, 
                            const double *, 
                            struct GenericFunction **,
                            struct GenericFunction **);
void generic_function_kronh2(int, size_t, size_t, size_t, size_t,
        struct GenericFunction **, struct GenericFunction **, 
        struct GenericFunction **);

// more complicated operations

void generic_function_array_orth1d_columns(
    struct GenericFunction **,
    struct GenericFunction **,
    enum function_class,
    void *, size_t,
    size_t, double,
    double);

void generic_function_array_orth(size_t,struct GenericFunction **,
                                 enum function_class,void *);
void 
generic_function_array_orth1d_linelm_columns(struct GenericFunction **,
                                             size_t,size_t,
                                             struct c3Vector *);

////////////////////////////////////////////////////////////////////
// High dimensional helper functions

/** \struct FiberCut
 *  \brief Interface to convert a multidimensional function to a one dimensional function
 *  by fixing all but one dimension
 *  \var FiberCut::totdim
 *  total dimension of the function
 *  \var FiberCut::dimcut
 *  dimension along which function can vary
 *  \var FiberCut::f
 *  function to cut
 *  \var FiberCut::ftype_flag
 *  0 for two dimensional function, 1 for n-dimensional function
 *  \var FiberCut::args
 *  function arguments
 *  \var FiberCut::vals
 *  values at which to fix the function, (totdim) but vals[dimcut] doesnt matter
 */
struct FiberCut
{
    size_t totdim;
    size_t dimcut;
    union func_type {
        double (*fnd)(double *, void *);
        double (*f2d)(double, double, void *);
    } f;
    int ftype_flag; // 0 for f2d, 1 for fnd
    void * args;
    double * vals; // (totdim) but vals[dimcut] doesnt matter
};

struct FiberCut *
fiber_cut_init2d( double (*f)(double, double, void *), void *, size_t, double);

struct FiberCut **
fiber_cut_2darray(double (*f)(double, double, void *), void *,
                            size_t, size_t, const double *);

struct FiberCut **
fiber_cut_ndarray( double (*)(double *, void *), void *,
                            size_t, size_t, size_t, double **);

void fiber_cut_free(struct FiberCut *);
void fiber_cut_array_free(size_t, struct FiberCut **);
double fiber_cut_eval2d(double, void *);
double fiber_cut_eval(double, void *);


/////////////////////////////////////////////////////////
// Utilities

void print_generic_function(const struct GenericFunction *, size_t,void *);
#endif
