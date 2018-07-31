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





/** \file ft.h 
 Provides header files for ft.c 
*/

#ifndef FT_H
#define FT_H

#include "qmarray.h"
#include "indmanage.h"

struct FTMemSpace;

/** \struct FunctionTrain
 * \brief Function-train
 * \var FunctionTrain::dim
 * dimension of function
 * \var FunctionTrain::ranks
 * function train ranks
 * \var FunctionTrain::cores
 * function train cores
 */
struct FunctionTrain {
    size_t dim;
    size_t * ranks;
    struct Qmarray ** cores;
    
    /* double * evalspace1; */
    /* double * evalspace2; */
    /* double * evalspace3; */

    struct FTMemSpace * evalspace1;
    struct FTMemSpace * evalspace2;
    struct FTMemSpace * evalspace3;

    struct FTMemSpace ** evalspace4;

    double ** evaldd1;
    double ** evaldd2;
    double ** evaldd3;
    double ** evaldd4;
};

struct FunctionTrain * function_train_alloc(size_t);
struct FunctionTrain * function_train_copy(const struct FunctionTrain *);
void function_train_free(struct FunctionTrain *);
unsigned char *
function_train_serialize(unsigned char *, struct FunctionTrain *,
                             size_t *);
unsigned char *
function_train_deserialize(unsigned char *, struct FunctionTrain **);

void function_train_savetxt(struct FunctionTrain *, FILE *,
                            size_t);
struct FunctionTrain * function_train_loadtxt(FILE *);

int function_train_save(struct FunctionTrain *, char *);
struct FunctionTrain * function_train_load(char *);

// getters and setters
size_t function_train_get_dim(const struct FunctionTrain *);
size_t * function_train_get_ranks(const struct FunctionTrain *);
size_t function_train_get_maxrank(const struct FunctionTrain *);
double function_train_get_avgrank(const struct FunctionTrain *);

struct Qmarray* function_train_get_core(const struct FunctionTrain *, size_t);
// evaluators
void function_train_eval_up_to_core(struct FunctionTrain *,
                                    size_t, const double *,
                                    double *, size_t *);
double function_train_eval(struct FunctionTrain *, const double * evalnd_pt);

size_t function_train_func_get_nparams(const struct FunctionTrain *,
                                       size_t, size_t, size_t);
size_t function_train_core_get_nparams(const struct FunctionTrain *,
                                       size_t,size_t *);
size_t function_train_get_nparams(const struct FunctionTrain *);
size_t function_train_core_get_params(const struct FunctionTrain *,
                                      size_t,double *);
size_t function_train_get_params(const struct FunctionTrain *,
                                 double *);
void * function_train_get_uni(const struct FunctionTrain *, size_t, size_t, size_t);
struct GenericFunction * function_train_get_gfuni(const struct FunctionTrain *,
                                                  size_t, size_t, size_t);

void function_train_uni_update_params(struct FunctionTrain *, size_t, size_t, size_t, size_t, const double *);
void function_train_core_update_params(struct FunctionTrain *, size_t,
                                       size_t, const double *);
size_t function_train_update_params(struct FunctionTrain *, const double *);

double function_train_param_grad_sqnorm(struct FunctionTrain *, double *, double *);

double function_train_eval_ind(struct FunctionTrain *, const size_t *);
double function_train_eval_co_perturb(struct FunctionTrain *, 
                                      const double *, const double *, 
                                      double *);
double function_train_eval_co_perturb_ind(struct FunctionTrain *, 
                                          const size_t *, const size_t *, 
                                          double *);
void function_train_eval_fiber_ind(struct FunctionTrain *, 
                                   const size_t *,
                                   size_t, const size_t *,
                                   size_t,
                                   double *);

// generators
struct FunctionTrain *
function_train_poly_randu(enum poly_type,struct BoundingBox *,
                          size_t * ranks, size_t);
struct FunctionTrain *
function_train_rankone(struct MultiApproxOpts *, struct Fwrap *);
struct FunctionTrain *
function_train_initsum(struct MultiApproxOpts *, struct Fwrap *);
struct FunctionTrain *
function_train_zeros(struct MultiApproxOpts *, const size_t *);
struct FunctionTrain *
function_train_constant(double a, struct MultiApproxOpts *);
struct FunctionTrain *
function_train_linear(const double *, size_t, const double *, size_t,
                      struct MultiApproxOpts *);
int
function_train_linear_update(
    struct FunctionTrain * ,
    const double *, size_t, 
    const double *, size_t);

struct FunctionTrain *
function_train_rankone_prod(const double *,
                            struct MultiApproxOpts *);
struct FunctionTrain *
function_train_quadratic(const double *,const double *,
                             struct MultiApproxOpts *);

struct FunctionTrain *
function_train_quadratic_aligned(const double *, const double *,
                                 struct MultiApproxOpts *);
struct FunctionTrain *
function_train_create_nodal(const struct FunctionTrain *, size_t *, double **);

// computations
double
function_train_integrate(const struct FunctionTrain *);
double
function_train_integrate_weighted(const struct FunctionTrain *);
struct FunctionTrain *
function_train_integrate_weighted_subset(
    const struct FunctionTrain * ft, size_t,size_t *);
double function_train_inner(const struct FunctionTrain *, 
                            const struct FunctionTrain * );
double function_train_inner_weighted(const struct FunctionTrain *, 
                                     const struct FunctionTrain *);
double function_train_norm2(const struct FunctionTrain *);
struct FunctionTrain * 
function_train_orthor(struct FunctionTrain *, 
                      struct MultiApproxOpts *);
struct FunctionTrain *
function_train_round(struct FunctionTrain *, double,
                         struct MultiApproxOpts *);
void function_train_truncate(struct FunctionTrain *, double);

struct FunctionTrain * 
function_train_sum(const struct FunctionTrain *,
                       const struct FunctionTrain *);
void function_train_scale(struct FunctionTrain *, double);
struct FunctionTrain *
function_train_product(const struct FunctionTrain *, 
                           const struct FunctionTrain *);
double function_train_norm2diff(const struct FunctionTrain *, 
                                const struct FunctionTrain *);
double function_train_relnorm2diff(const struct FunctionTrain *,
                                       const struct FunctionTrain *);

int function_train_is_kristoffel_active(const struct FunctionTrain *);
void function_train_activate_kristoffel(struct FunctionTrain *);
void function_train_deactivate_kristoffel(struct FunctionTrain *);
double function_train_get_kristoffel_weight(const struct FunctionTrain *,
                                            const double *);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/////////////////////                          ///////////////////////////
/////////////////////    Cross Approximation   ///////////////////////////
/////////////////////                          ///////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

struct FtCrossArgs;
struct FtCrossArgs * ft_cross_args_alloc(size_t, size_t);
void ft_cross_args_set_rank(struct FtCrossArgs *, size_t, size_t);
void ft_cross_args_set_round_tol(struct FtCrossArgs *, double);
void ft_cross_args_set_kickrank(struct FtCrossArgs *, size_t);
void ft_cross_args_set_maxiter(struct FtCrossArgs *, size_t);
void ft_cross_args_set_no_adaptation(struct FtCrossArgs *);
void ft_cross_args_set_adaptation(struct FtCrossArgs *);
void ft_cross_args_set_maxrank_all(struct FtCrossArgs *, size_t);
void ft_cross_args_set_maxrank_ind(struct FtCrossArgs *, size_t, size_t);
void ft_cross_args_set_cross_tol(struct FtCrossArgs *, double);
void ft_cross_args_set_verbose(struct FtCrossArgs *, int);
size_t * ft_cross_args_get_ranks(const struct FtCrossArgs *);
size_t function_train_get_rank(const struct FunctionTrain *, size_t);
struct FtCrossArgs * ft_cross_args_copy(const struct FtCrossArgs *);
void ft_cross_args_free(struct FtCrossArgs *);

struct FunctionTrain *
function_train_init_from_fibers(struct Fwrap *,
                                size_t *,
                                struct CrossIndex **,
                                struct CrossIndex **,
                                struct MultiApproxOpts *);

struct FunctionTrain *
ftapprox_cross(struct Fwrap *,
               struct FtCrossArgs *,
               struct CrossIndex **,
               struct CrossIndex **,
               struct MultiApproxOpts *,
               struct FiberOptArgs *,
               struct FunctionTrain *);

struct FunctionTrain *
ftapprox_cross_rankadapt(struct Fwrap *,
                         struct FtCrossArgs *,
                         struct CrossIndex **, 
                         struct CrossIndex **,
                         struct MultiApproxOpts *,
                         struct FiberOptArgs *,
                         struct FunctionTrain *);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/////////////////////                          ///////////////////////////
/////////////////////    Starting FT1D Array   ///////////////////////////
/////////////////////                          ///////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


/** \struct FT1DArray
 * \brief One dimensional array of function trains
 * \var FT1DArray::size
 * size of array
 * \var FT1DArray::ft
 * array of function trains
 */
struct FT1DArray{
    size_t size;
    struct FunctionTrain ** ft;
};

struct FT1DArray * ft1d_array_alloc(size_t);
struct FT1DArray * function_train_gradient(const struct FunctionTrain *);
void function_train_gradient_eval(const struct FunctionTrain *, const double * evalnd_pt, double * evalnd_out);
unsigned char *
ft1d_array_serialize(unsigned char *, struct FT1DArray *, size_t *);
unsigned char *
ft1d_array_deserialize(unsigned char *, struct FT1DArray **);
int ft1d_array_save(struct FT1DArray *, char *);
struct FT1DArray * ft1d_array_load(char *);
struct FT1DArray * ft1d_array_copy(const struct FT1DArray *);
void ft1d_array_free(struct FT1DArray *);
struct FT1DArray * ft1d_array_jacobian(const struct FT1DArray *);
struct FT1DArray * function_train_hessian(const struct FunctionTrain *);
struct FT1DArray * function_train_hessian_diag(const struct FunctionTrain *);
struct FT1DArray * function_train_hessian_diag_periodic(const struct FunctionTrain *);
void ft1d_array_scale(struct FT1DArray *, size_t, size_t, double);
double * ft1d_array_eval(const struct FT1DArray *, const double *);
void ft1d_array_eval2(const struct FT1DArray *, const double *, double *);
/* struct FunctionTrain * */
/* ft1d_array_sum_prod(size_t, double *, */
/*                     const struct FT1DArray *, const struct FT1DArray *, */
/*                     double); */



//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/////////////////////                          ///////////////////////////
/////////////////////    BLAS TYPE INTERFACE   ///////////////////////////
/////////////////////                          ///////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void c3axpy(double, struct FunctionTrain *, struct FunctionTrain **,double);
void c3vaxpy_arr(size_t, double, struct FT1DArray *, 
                size_t, double *, size_t, double,
                struct FunctionTrain **, double);

double c3dot(struct FunctionTrain *, struct FunctionTrain *);
void c3gemv(double, size_t, struct FT1DArray *, size_t, 
        struct FunctionTrain *,double, struct FunctionTrain *,double);

void c3vaxpy(size_t, double, struct FT1DArray *, size_t, 
            struct FT1DArray **, size_t, double);
void c3vprodsum(size_t, double, struct FT1DArray *, size_t,
                struct FT1DArray *, size_t, double,
                struct FunctionTrain **, double);
void c3vgemv(size_t, size_t, double, struct FT1DArray *, size_t,
        struct FT1DArray *, size_t, double, struct FT1DArray **,
        size_t, double );
void c3vgemv_arr(int, size_t, size_t, double, double *, size_t,
        struct FT1DArray *, size_t, double, struct FT1DArray **, 
        size_t, double);

#endif
