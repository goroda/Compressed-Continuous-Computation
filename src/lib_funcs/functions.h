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

/** \file functions.h
 * Provides header files and structure definitions for functions in functions.c 
 */

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "lib_funcs.h"

/** \enum function_class
 * contains PIECEWISE, POLYNOMIAL, RATIONAL, KERNEL:
 * only POLYNOMIAL is implemented!!!
 */
enum function_class {PIECEWISE, POLYNOMIAL, RATIONAL, KERNEL};

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

/** \struct BoundingBox
 * \brief An array of pairs of lower and upper bounds
 * \var BoundingBox::dim
 * dimension of box
 * \var BoundingBox::lb
 * lower bounds
 * \var BoundingBox::ub
 * upper bounds
 */
struct BoundingBox
{
    size_t dim;
    double * lb;
    double * ub;
};

struct BoundingBox * bounding_box_init_std(size_t);
struct BoundingBox * bounding_box_init(size_t,double, double);
struct BoundingBox * bounding_box_vec(size_t, double *, double *);
void bounding_box_free(struct BoundingBox *);

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
    union
    {
        enum poly_type ptype;
    } sub_type;
    void * f;
    void * fargs;
};

struct GenericFunction * generic_function_alloc(size_t, enum function_class, void *);
void generic_function_roundt(struct GenericFunction **, double);

struct GenericFunction * generic_function_deriv(struct GenericFunction *);
struct GenericFunction * generic_function_copy(struct GenericFunction *);

struct GenericFunction * generic_function_approximate1d( 
                double (*f)(double,void *), void *, enum function_class, 
                void *, double lb, double ub, void *);

struct GenericFunction * generic_function_copy(struct GenericFunction *);
void generic_function_free(struct GenericFunction *);
void generic_function_array_free(size_t, struct GenericFunction **);
unsigned char * serialize_generic_function(unsigned char *, 
                    struct GenericFunction *, size_t *);
unsigned char *
deserialize_generic_function(unsigned char *, struct GenericFunction ** );

// extraction functions
double generic_function_get_lower_bound(struct GenericFunction * f);
double generic_function_get_upper_bound(struct GenericFunction * f);
double generic_function_1d_eval(struct GenericFunction *, double);
double * generic_function_1darray_eval(size_t, struct GenericFunction **, 
                                double);

// generic operations
double generic_function_norm(struct GenericFunction *);
double generic_function_norm2diff(struct GenericFunction *, 
                                  struct GenericFunction *);
double generic_function_integral(struct GenericFunction *);
double * 
generic_function_integral_array(size_t , size_t, struct GenericFunction ** a);

struct GenericFunction *
generic_function_sum_prod(size_t, size_t,  struct GenericFunction **, 
                size_t, struct GenericFunction **);
double generic_function_sum_prod_integrate(size_t, size_t,  
                struct GenericFunction **, size_t, struct GenericFunction **);
struct GenericFunction *
generic_function_prod(struct GenericFunction *, struct GenericFunction *);
double generic_function_inner(struct GenericFunction *, struct GenericFunction *);
double generic_function_inner_sum(size_t, size_t, struct GenericFunction **, 
                         size_t, struct GenericFunction **);
double generic_function_array_norm(size_t, size_t, struct GenericFunction **);

void generic_function_flip_sign(struct GenericFunction *);
void generic_function_array_flip_sign(size_t, size_t, struct GenericFunction **);

struct GenericFunction * generic_function_daxpby(double, struct GenericFunction * m,
            double, struct GenericFunction *);

struct GenericFunction **
generic_function_array_daxpby(size_t, double, size_t, 
        struct GenericFunction **, double, size_t, 
        struct GenericFunction **);

void
generic_function_array_daxpby2(size_t, double, size_t, 
        struct GenericFunction **, double, size_t, 
        struct GenericFunction **, size_t, struct GenericFunction **);

struct GenericFunction * generic_function_lin_comb(size_t,
                            struct GenericFunction **, double *);
struct GenericFunction * generic_function_lin_comb2(size_t, size_t, 
                struct GenericFunction **, size_t, double *);

double generic_function_absmax(struct GenericFunction *, double *);
double generic_function_array_absmax(size_t, size_t, 
        struct GenericFunction **, size_t *, double *);

// more complicated operations
void generic_function_scale(double, struct GenericFunction *);
void generic_function_array_scale(double, struct GenericFunction **, size_t);
struct GenericFunction * 
generic_function_constant(double, enum function_class, void *,
            double, double, void *);
struct GenericFunction * 
generic_function_linear(double, double,
            enum function_class, void *,
            double, double, void *);
struct GenericFunction * 
generic_function_quadratic(double, double,
            enum function_class, void *,
            double, double, void *);

void generic_function_array_orth(size_t, enum function_class, void *,
                            struct GenericFunction **, void *);

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
                            size_t, size_t, double *);

struct FiberCut **
fiber_cut_ndarray( double (*)(double *, void *), void *, 
                            size_t, size_t, size_t, double **);

void fiber_cut_free(struct FiberCut *);
void fiber_cut_array_free(size_t, struct FiberCut **);
double fiber_cut_eval2d(double, void *);
double fiber_cut_eval(double, void *);


/////////////////////////////////////////////////////////
// Utilities

void print_generic_function(struct GenericFunction *, size_t,void *);
#endif
