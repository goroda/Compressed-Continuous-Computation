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

/** \file algs.h
 * Provides header files and structure definitions for functions in in algs.c
 */


#ifndef ALGS_H
#define ALGS_H

#include "elements.h"

/** \struct Cross2dargs
 *  \brief Contains arguments for cross approximation
 *  \var Cross2dargs::r 
 *  rank of approximation
 *  \var Cross2dargs::delta
 *  convergence criteria
 *  \var Cross2dargs::fclass
 *  approximation classes for each dimension
 *  \var Cross2dargs::sub_type
 *  approximation sub types for each dimension
 *  \var Cross2dargs::verbose
 *  verbosity level (0,1,2)
 * */
struct Cross2dargs
{
    size_t r;
    double delta;

    enum function_class fclass[2];
    void * sub_type[2];
    void * approx_args [2];
    int verbose;
};

/** \struct FtCrossArgs
 *  \brief Arguments for function-train cross approximation
 *  \var FtCrossArgs::dim
 *  dimension of function
 *  \var FtCrossArgs::ranks
 *  (dim+1,) array of ranks
 *  \var FtCrossArgs::epsilon
 *  cross-approximation convergence criteria
 *  \var FtCrossArgs::maxiter
 *  maximum number of iteration for cross approximation
 *  \var FtCrossArgs::epsround
 *  rounding tolerance for adaptive rank cross approximation
 *  \var FtCrossArgs::kickrank
 *  size of rank increase for adaptation
 *  \var FtCrossArgs::maxiteradapt
 *  maximum number of rank adaptation iterations
 *  \var FtCrossArgs::verbose
 *  verbosity level (0,1,2)
 * */
struct FtCrossArgs
{
    size_t dim;
    size_t * ranks;
    double epsilon;
    size_t maxiter;
    
    double epsround;
    size_t kickrank;
    size_t maxiteradapt;

    int verbose;
};


// (quasimatrix - vector multiplication
struct GenericFunction * qmv(struct Quasimatrix *, double *);
struct Quasimatrix * qmm(struct Quasimatrix *, double *, size_t);
struct Quasimatrix * qmmt(struct Quasimatrix *, double *, size_t);
struct Quasimatrix * quasimatrix_daxpby(double, struct Quasimatrix *,
                                         double, struct Quasimatrix *);

// decompositions
struct Quasimatrix * quasimatrix_householder_simple(struct Quasimatrix *, double *); 
int quasimatrix_householder(struct Quasimatrix *, struct Quasimatrix *, 
        struct Quasimatrix *, double *);
int quasimatrix_qhouse(struct Quasimatrix *, struct Quasimatrix *);

int quasimatrix_lu1d(struct Quasimatrix *, struct Quasimatrix *, double *, double *);

int quasimatrix_maxvol1d(struct Quasimatrix *, double *, double *);

struct SkeletonDecomp * cross_approx_2d(double (*)(double, double, void *), 
        void *, struct BoundingBox *, struct SkeletonDecomp **,  double *,
        double *, struct Cross2dargs *);

// Misc
size_t quasimatrix_rank(struct Quasimatrix *);
double skeleton_decomp_inner_help(struct Quasimatrix *, struct Quasimatrix *,
            struct Quasimatrix *, struct Quasimatrix *);

double quasimatrix_inner(struct Quasimatrix *, struct Quasimatrix *);
double quasimatrix_norm(struct Quasimatrix *);

////////////////////////////////////////////////////////////////////////////
// qmarrays

// (qmarray - vector multiplication
struct Quasimatrix * qmav(struct Qmarray *, double *);
struct Qmarray * qmam(struct Qmarray *, double *, size_t);
struct Qmarray * qmatm(struct Qmarray *, double *, size_t);
struct Qmarray * mqma(double *, struct Qmarray *, size_t);
struct Qmarray * qmaqma(struct Qmarray * a, struct Qmarray * b);
struct Qmarray * qmatqma(struct Qmarray * a, struct Qmarray * b);
struct Qmarray * qmaqmat(struct Qmarray * a, struct Qmarray * b);
struct Qmarray * qmatqmat(struct Qmarray * a, struct Qmarray * b);
double * qmatqma_integrate(struct Qmarray *, struct Qmarray *);
double * qmaqmat_integrate(struct Qmarray *, struct Qmarray *);
double * qmatqmat_integrate(struct Qmarray *, struct Qmarray *);
struct Qmarray * qmarray_kron(struct Qmarray *, struct Qmarray *);
double * qmarray_kron_integrate(struct Qmarray *, struct Qmarray *);
struct Qmarray * qmarray_vec_kron(double *, struct Qmarray *, struct Qmarray *);
double * qmarray_vec_kron_integrate(double *, struct Qmarray *, struct Qmarray *);
struct Qmarray * qmarray_mat_kron(size_t, double *, struct Qmarray *, struct Qmarray *);
struct Qmarray * qmarray_kron_mat(size_t, double *, struct Qmarray *, struct Qmarray *);
void qmarray_block_kron_mat(char, int, size_t,
        struct Qmarray **, struct Qmarray *, size_t,
        double *, struct Qmarray *);
double * qmarray_integrate(struct Qmarray *);
double qmarray_norm2diff(struct Qmarray *, struct Qmarray *);
double qmarray_norm2(struct Qmarray *);

//struct Quasimatrix * qmmt(struct Quasimatrix *, double *, size_t);

int qmarray_lu1d(struct Qmarray *, struct Qmarray *, double *, size_t *,
                    double *);
int qmarray_maxvol1d(struct Qmarray *, double *, size_t *, double *);

int qmarray_qhouse(struct Qmarray *, struct Qmarray *);
int qmarray_qhouse_rows(struct Qmarray *, struct Qmarray *);
int qmarray_householder(struct Qmarray *, struct Qmarray *, 
        struct Qmarray *, double *);
int qmarray_householder_rows(struct Qmarray *, struct Qmarray *, 
        struct Qmarray *, double *);
struct Qmarray *
qmarray_householder_simple(char *, struct Qmarray *, double *);

int qmarray_svd(struct Qmarray *, struct Qmarray **, double *, double *);
size_t qmarray_truncated_svd(struct Qmarray *, struct Qmarray **, 
            double **, double **, double);


void qmarray_absmax1d(struct Qmarray *, double *, size_t *, size_t *, double *);
struct Qmarray * qmarray_transpose(struct Qmarray * a);
struct Qmarray * qmarray_stackh(struct Qmarray *, struct Qmarray *);
struct Qmarray * qmarray_stackv(struct Qmarray *, struct Qmarray *);
struct Qmarray * qmarray_blockdiag(struct Qmarray *, struct Qmarray *);
struct Qmarray * qmarray_deriv(struct Qmarray *);
void qmarray_roundt(struct Qmarray **, double);

void qmarray_eval(struct Qmarray *, double, double *);

////////////////////////////////////////////////////////////////////////////
// function_train

double function_train_eval(struct FunctionTrain *, double *);
struct FunctionTrain * function_train_sum(struct FunctionTrain *, struct FunctionTrain *);
struct FunctionTrain * function_train_afpb(double, double, struct FunctionTrain *, double);
struct FunctionTrain * function_train_orthor(struct FunctionTrain *);
struct FunctionTrain * function_train_round(struct FunctionTrain *, double);
void function_train_scale(struct FunctionTrain *, double);
struct FunctionTrain * function_train_product(struct FunctionTrain *, struct FunctionTrain *);
double function_train_integrate(struct FunctionTrain *);
double function_train_inner(struct FunctionTrain *, struct FunctionTrain *);
double function_train_norm2(struct FunctionTrain *);
double function_train_norm2diff(struct FunctionTrain *, struct FunctionTrain *);
double function_train_relnorm2diff(struct FunctionTrain *, struct FunctionTrain *);

struct FT1DArray * function_train_gradient(struct FunctionTrain *);
struct FT1DArray * ft1d_array_jacobian(struct FT1DArray *);
struct FT1DArray * function_train_hessian(struct FunctionTrain *);
void ft1d_array_scale(struct FT1DArray *, size_t, size_t, double);
double * ft1d_array_eval(struct FT1DArray *, double *);

struct FunctionTrain * 
ft1d_array_sum_prod(size_t, double *, struct FT1DArray *, struct FT1DArray *, 
               double);

struct FunctionTrain *
ftapprox_cross(double (*)(double *, void *), void *, 
    struct BoundingBox *, struct FunctionTrain *, 
    struct IndexSet **, struct IndexSet **, 
    struct FtCrossArgs *, struct FtApproxArgs *);

struct FunctionTrain *
function_train_cross(double (*)(double *, void *), void *, 
        struct BoundingBox *, double *,
        struct FtCrossArgs *, struct FtApproxArgs *);

struct FT1DArray * function_train_hessian(struct FunctionTrain *);

struct FunctionTrain *
ftapprox_cross_rankadapt(double (*)(double *, void *), void *,
                struct BoundingBox *, 
                struct FunctionTrain *, 
                struct IndexSet **, 
                struct IndexSet **,
                double *,
                struct FtCrossArgs *,
                struct FtApproxArgs *
                );


//////////////////////////////////////////////////////////////////////
// Blas type interface 1
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
