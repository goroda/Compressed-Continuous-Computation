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





/** \file qmarray.h 
 Provides header files for qmarray.c 
*/

#ifndef QMARRAY_H
#define QMARRAY_H

#include "quasimatrix.h"

/* struct Qmarray; */
/** \struct Qmarray
 * \brief Defines a matrix-valued function (a Quasimatrix array)
 * \var Qmarray::nrows
 * number of rows
 * \var Qmarray::ncols
 * number of columns
 * \var Qmarray::funcs
 * functions in column-major order 
 */
struct Qmarray {
    size_t nrows;
    size_t ncols;
    struct GenericFunction ** funcs; // fortran order
};

struct Qmarray * qmarray_alloc(size_t, size_t); 
void qmarray_free(struct Qmarray *); 
struct Qmarray * qmarray_copy(const struct Qmarray *);
struct Qmarray * qmarray_operate_elements(const struct Qmarray *,
                                          const struct Operator *);
unsigned char * 
qmarray_serialize(unsigned char *,const struct Qmarray *, size_t *);
unsigned char * 
qmarray_deserialize(unsigned char *, struct Qmarray **);

void qmarray_savetxt(const struct Qmarray *, FILE *, size_t);
struct Qmarray *  qmarray_loadtxt(FILE *);

struct Qmarray * 
qmarray_approx1d(size_t, size_t, struct OneApproxOpts *,
                 struct Fwrap *);

/* struct Qmarray *  */
/* qmarray_from_fiber_cuts(size_t, size_t,  */
/*                     double (*f)(double, void *),struct FiberCut **,  */
/*                     enum function_class, void *, double, */
/*                     double, void *); */

struct Qmarray *
qmarray_orth1d_rows(size_t, size_t, struct OneApproxOpts *);
struct Qmarray *
qmarray_orth1d_columns(size_t, size_t, struct OneApproxOpts *);
struct Qmarray * 
qmarray_zeros(size_t, size_t,struct OneApproxOpts *);


/* struct Qmarray * */
/* qmarray_orth1d_linelm_grid(size_t,size_t, struct c3Vector *); */
struct Qmarray * 
qmarray_poly_randu(enum poly_type, size_t, size_t, size_t, double, double);

// getters and setters
void * qmarray_get_func_base(const struct Qmarray *, size_t, size_t);
struct GenericFunction *
qmarray_get_func(const struct Qmarray *, size_t, size_t);

//rows
void qmarray_set_row(struct Qmarray *, size_t, const struct Quasimatrix *);
struct Quasimatrix *
qmarray_extract_row(const struct Qmarray *, size_t);

//columns
void qmarray_set_column(struct Qmarray *, size_t, 
                            const struct Quasimatrix *);

void qmarray_set_column(struct Qmarray *, size_t, const struct Quasimatrix *);
void qmarray_set_column_gf(struct Qmarray *, size_t, 
                               struct GenericFunction **);
struct Quasimatrix * 
qmarray_extract_column(const struct Qmarray *, size_t);

size_t qmarray_get_ncols(const struct Qmarray *);
size_t qmarray_get_nrows(const struct Qmarray *);

// qmarrays

// (qmarray - vector multiplication
/* struct Quasimatrix * qmav(struct Qmarray *, double *); */
struct Qmarray * qmam(const struct Qmarray *,const double *, size_t);
struct Qmarray * qmatm(const struct Qmarray *,const double *, size_t);
struct Qmarray * mqma(const double *,const struct Qmarray *, size_t);
struct Qmarray * qmaqma(const struct Qmarray * a, const struct Qmarray * b);
struct Qmarray * qmatqma(const struct Qmarray * a, const struct Qmarray * b);
struct Qmarray * qmaqmat(const struct Qmarray * a, const struct Qmarray * b);
struct Qmarray * qmatqmat(const struct Qmarray * a, const struct Qmarray * b);
double * qmatqma_integrate(const struct Qmarray *,const struct Qmarray *);
double * qmaqmat_integrate(const struct Qmarray *, const struct Qmarray *);
double * qmatqmat_integrate_weighted(const struct Qmarray *,const struct Qmarray *);
double * qmatqmat_integrate(const struct Qmarray *,const  struct Qmarray *);
struct Qmarray * qmarray_kron(const struct Qmarray *, const struct Qmarray *);
double * qmarray_kron_integrate(const struct Qmarray *, const struct Qmarray *);
double * qmarray_kron_integrate_weighted(const struct Qmarray *, const struct Qmarray *);
struct Qmarray * qmarray_vec_kron(const double *, const struct Qmarray *,
                                  const struct Qmarray *);
double * qmarray_vec_kron_integrate(const double *, const struct Qmarray *,
                                    const struct Qmarray *);
double * qmarray_vec_kron_integrate_weighted(
    const double *, const struct Qmarray *,const struct Qmarray *);
struct Qmarray * qmarray_mat_kron(size_t, const double *, const struct Qmarray *,
                                  const struct Qmarray *);
struct Qmarray * qmarray_kron_mat(size_t, const double *, const struct Qmarray *,
                                  const struct Qmarray *);
void qmarray_block_kron_mat(char, int, size_t,
        struct Qmarray **, struct Qmarray *, size_t,
        double *, struct Qmarray *);
double * qmarray_integrate(const struct Qmarray *);
double * qmarray_integrate_weighted(const struct Qmarray *);
double qmarray_norm2diff(const struct Qmarray *,const struct Qmarray *);
double qmarray_norm2(const struct Qmarray *);
void qmarray_axpy(double,const struct Qmarray *, struct Qmarray *);

//struct Quasimatrix * qmmt(struct Quasimatrix *, double *, size_t);
int qmarray_lu1d(struct Qmarray *, struct Qmarray *, double *, size_t *,
                 double *,struct OneApproxOpts*,void *);
int qmarray_maxvol1d(struct Qmarray *, double *, size_t *, double *,struct OneApproxOpts*,void *);

int qmarray_qhouse(struct Qmarray *, struct Qmarray *);
int qmarray_qhouse_rows(struct Qmarray *, struct Qmarray *);
int qmarray_householder(struct Qmarray *, struct Qmarray *, 
                        struct Qmarray *, double *);
int qmarray_householder_rows(struct Qmarray *, struct Qmarray *, 
                             struct Qmarray *, double *);
struct Qmarray *
qmarray_householder_simple(char *, struct Qmarray *, double *,struct OneApproxOpts *);
/* struct Qmarray * */
/* qmarray_householder_simple_grid(char *, struct Qmarray *, double *, */
/*                                 struct c3Vector *); */

int qmarray_svd(struct Qmarray *, struct Qmarray **, double *, double *,
                struct OneApproxOpts *);
size_t qmarray_truncated_svd(struct Qmarray *, struct Qmarray **, 
                             double **, double **, double, struct OneApproxOpts * );


void qmarray_absmax1d(struct Qmarray *, double *, size_t *, 
                      size_t *, double *, void *);
struct Qmarray * qmarray_transpose(struct Qmarray * a);
struct Qmarray * qmarray_stackh(const struct Qmarray *,const struct Qmarray *);
struct Qmarray * qmarray_stackv(const struct Qmarray *,const struct Qmarray *);
struct Qmarray * qmarray_blockdiag(struct Qmarray *, struct Qmarray *);
struct Qmarray * qmarray_deriv(struct Qmarray *);
struct Qmarray * qmarray_dderiv(struct Qmarray *);
struct Qmarray * qmarray_dderiv_periodic(struct Qmarray *);
void qmarray_deriv_eval(const struct Qmarray *, double, double *);
void qmarray_roundt(struct Qmarray **, double);

void qmarray_eval(struct Qmarray *, double, double *);
size_t qmarray_func_get_nparams(const struct Qmarray *,
                                size_t, size_t);
size_t qmarray_get_nparams(const struct Qmarray *,size_t *);
size_t qmarray_get_params(struct Qmarray *,double *);
void qmarray_uni_update_params(struct Qmarray *, size_t, size_t,
                               size_t, const double *);
void qmarray_update_params(struct Qmarray *, size_t, const double *);
void qmarray_param_grad_eval(struct Qmarray *, size_t,
                             const double *, size_t,
                             double *, size_t,
                             double *, size_t,
                             double *);
void qmarray_linparam_eval(struct Qmarray *, size_t,
                           double *, size_t,
                           const double *, size_t);

void qmarray_param_grad_eval_sparse_mult(struct Qmarray *, size_t,
                                         const double *, size_t,
                                         double *, size_t,
                                         double *, size_t,
                                         double *, double *, size_t);
double qmarray_param_grad_sqnorm(struct Qmarray *, double, double *);
struct Qmarray * qmarray_create_nodal(struct Qmarray *, size_t, double *);

int qmarray_is_kristoffel_active(const struct Qmarray *);
void qmarray_activate_kristoffel(struct Qmarray *);
void qmarray_deactivate_kristoffel(struct Qmarray *);
double qmarray_get_kristoffel_weight(const struct Qmarray *,double);

int qmarray_qr(struct Qmarray *, struct Qmarray **, double **, struct OneApproxOpts *);
int qmarray_lq(struct Qmarray *, struct Qmarray **, double **, struct OneApproxOpts *);

int qmarray_qr_gs(struct Qmarray *, double **);
int qmarray_lq_gs(struct Qmarray *, double **);

void print_qmarray(struct Qmarray *, size_t, void *, FILE *);
#endif
