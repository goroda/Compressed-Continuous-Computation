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

/** \file elements.h
 * Provides header files and structure definitions for functions in elements.c
 */

#ifndef ELEMENTS_H
#define ELEMENTS_H

#include <stdlib.h>
#include "../lib_funcs/lib_funcs.h"

//////////////////////////////////////////////////////////////
// Quasimatrices

/** \struct Quasimatrix
 *  \brief Defines a vector-valued function
 *  \var Quasimatrix::n
 *  number of functions
 *  \var Quasimatrix::funcs
 *  array of functions
 */
struct Quasimatrix {
    size_t n;
    struct GenericFunction ** funcs;
};

struct Quasimatrix * quasimatrix_alloc(size_t n);
struct Quasimatrix * quasimatrix_init(size_t, size_t, 
                enum function_class *, void **, void **, void **);    

struct Quasimatrix * 
quasimatrix_approx1d(size_t, double (**)(double, void *),
                    void **, enum function_class, void *, double,
                    double, void *);

struct Quasimatrix * 
quasimatrix_approx_from_fiber_cuts(size_t, double (*)(double, void *),
                    struct FiberCut **, enum function_class, void *, double,
                    double, void *);

void quasimatrix_free(struct Quasimatrix *);
struct Quasimatrix * quasimatrix_copy(struct Quasimatrix *);

unsigned char * 
quasimatrix_serialize(unsigned char *, struct Quasimatrix *, size_t *);
unsigned char * 
quasimatrix_deserialize(unsigned char *, struct Quasimatrix ** );

struct Quasimatrix * quasimatrix_orth1d(enum function_class fc, 
                        void * st, size_t n, double lb, double ub);

size_t quasimatrix_absmax(struct Quasimatrix *, double *, double *);

/** \struct SkeletonDecomp
 *  \brief Defines a skeleton decomposition of a two-dimensional function \f$ f(x,y) \f$
 *  \var SkeletonDecomp::r
 *  rank of the function
 *  \var SkeletonDecomp::xqm
 *  quasimatrix representing functions of *x*
 *  \var SkeletonDecomp::yqm
 *  quasimatrix representing functions of *y*
 *  \var SkeletonDecomp::skeleton
 *  skeleton matrix
 */
struct SkeletonDecomp
{
    size_t r;
    struct Quasimatrix * xqm;
    struct Quasimatrix * yqm;
    double * skeleton; // rows are x, cols are y
};
struct SkeletonDecomp * skeleton_decomp_alloc(size_t);
struct SkeletonDecomp * skeleton_decomp_copy(struct SkeletonDecomp *);
void skeleton_decomp_free(struct SkeletonDecomp *);

struct SkeletonDecomp * 
skeleton_decomp_init2d_from_pivots(double (*)(double,double,void *),void *, 
                        struct BoundingBox *, enum function_class *, void **,
                        size_t, double *,double *, void **);
double skeleton_decomp_eval(struct SkeletonDecomp *, double, double);


////////////////////////////////////////////////////////////////////
// qm_array (quasimatrix arrays)

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
struct Qmarray * qmarray_zeros(size_t, size_t,double,double);
struct Qmarray * qmarray_poly_randu(size_t, size_t, size_t, double, double);
struct Qmarray * qmarray_copy(struct Qmarray *);
void qmarray_free(struct Qmarray *); 

struct Qmarray * 
qmarray_approx1d(size_t, size_t, double (**)(double, void *),
                    void **, enum function_class, void *, double,
                    double, void *);
struct Qmarray * 
qmarray_from_fiber_cuts(size_t, size_t, 
                    double (*f)(double, void *),struct FiberCut **, 
                    enum function_class, void *, double,
                    double, void *);

struct Qmarray * 
qmarray_orth1d_columns(enum function_class, 
    void *, size_t, size_t, double, double);
struct Qmarray *
qmarray_orth1d_rows(enum function_class, void *, size_t,
                            size_t, double, double);



struct Quasimatrix * qmarray_extract_column(struct Qmarray *, size_t);
struct Qmarray * qmarray_extract_ncols(struct Qmarray *, size_t);
struct Quasimatrix * qmarray_extract_row(struct Qmarray *, size_t);

void qmarray_set_column(struct Qmarray *, size_t, struct Quasimatrix *);
void qmarray_set_column_gf(struct Qmarray *, size_t, 
                                                struct GenericFunction **);
void qmarray_set_row(struct Qmarray *, size_t, struct Quasimatrix *);

unsigned char * 
qmarray_serialize(unsigned char *, struct Qmarray *, size_t *);
unsigned char * 
qmarray_deserialize(unsigned char *, struct Qmarray ** );

////////////////////////////////////////////////////////////////////
// function_train 

/** \struct FtApproxArgs
 * \brief function train approximation arguments
 * \var FtApproxArgs::dim
 * function dimension
 * \var FtApproxArgs::fc
 * function approximation classes
 * \var FtApproxArgs::sub_type
 * function approximation sub types
 * \var FtApproxArgs::approx_opts
 * function approximation options
 * \var FtApproxArgs::targs
 * type of approximations (0: same in each dimension, 1: specified separately for each  dimension)
 */
struct FtApproxArgs
{
    size_t dim;
    union func_class {
        enum function_class fc0;
        enum function_class * fc1;
    } fc;
    union st {
        void * st0;
        void ** st1;
    } sub_type; 
    union aopts {
        void * ao0;
        void ** ao1;
    } approx_opts;

    int targs; // type of args (0,1)
};

struct FtApproxArgs * ft_approx_args_createpoly(size_t, enum poly_type *,
                struct OpeAdaptOpts * aopts);
struct FtApproxArgs * ft_approx_args_createpwpoly(size_t, enum poly_type *,
                struct PwPolyAdaptOpts * aopts);

enum function_class 
ft_approx_args_getfc(struct FtApproxArgs *, size_t);

void * ft_approx_args_getst(struct FtApproxArgs *, size_t);
void * ft_approx_args_getaopts(struct FtApproxArgs *, size_t);

void ft_approx_args_free(struct FtApproxArgs *);


/** \struct FunctionTrain
 * \brief Functrain train
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
    
    double * evalspace1;
    double * evalspace2;
    double * evalspace3;
   
};

struct BoundingBox * function_train_bds(struct FunctionTrain *);

struct FunctionTrain * function_train_alloc(size_t); 
struct FunctionTrain * function_train_copy(struct FunctionTrain *);
void function_train_free(struct FunctionTrain *);

struct FunctionTrain *
function_train_poly_randu(struct BoundingBox *, size_t *, size_t);

struct FunctionTrain * function_train_initsum(size_t, 
        double (**)(double, void *), void **, struct BoundingBox *,
        struct FtApproxArgs * ); 

struct FunctionTrain * 
function_train_initsum2(size_t,  double (*f)(double, size_t, void *), 
        void *, struct BoundingBox *, struct FtApproxArgs *);

struct FunctionTrain *
function_train_rankone(size_t,  double (*)(double, size_t, void *), 
        void *, struct BoundingBox *, struct FtApproxArgs *);



struct FunctionTrain * 
function_train_linear2(size_t, struct BoundingBox *, 
        double *, size_t, double *, size_t,struct FtApproxArgs *);

struct FunctionTrain * 
function_train_constant(size_t, double, struct BoundingBox *,  
                        struct FtApproxArgs *);

struct FunctionTrain * function_train_linear(size_t, struct BoundingBox *,
                    double *, struct FtApproxArgs * ); 

struct FunctionTrain * function_train_quadratic(size_t, struct BoundingBox *,
                    double *, double *, struct FtApproxArgs * ); 
struct FunctionTrain * 
function_train_quadratic_aligned(struct BoundingBox *, double *, 
                                 double *, struct FtApproxArgs *);

unsigned char * 
function_train_serialize(unsigned char *, struct FunctionTrain *, size_t *);
unsigned char * 
function_train_deserialize(unsigned char *, struct FunctionTrain ** );

int function_train_save(struct FunctionTrain *, char *);
struct FunctionTrain * function_train_load(char *);

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
unsigned char * ft1d_array_serialize(unsigned char *, struct FT1DArray *, size_t *);
unsigned char * ft1d_array_deserialize(unsigned char *, struct FT1DArray ** );
int ft1d_array_save(struct FT1DArray *, char *);
struct FT1DArray * ft1d_array_load(char *);

struct FT1DArray * ft1d_array_copy(struct FT1DArray *);
void ft1d_array_free(struct FT1DArray *);


/////////////////////////////////////////////////////////////////////
// Indices
/** \struct IndexSet
 * \brief Describes index sets used for cross approximation
 * \var IndexSet::type
 * 0 for left, 1 for right
 * \var IndexSet::totdim
 * total dimension of the function
 * \var IndexSet::dim
 * the dimension for which the index set is useful
 * \var IndexSet::rank
 * rank for the dimension sepecified
 * \var IndexSet::inds
 * set of indices
 */
struct IndexSet{
    
    int type; // 0 for left, 1 for right
    size_t totdim;
    size_t dim;
    size_t rank;
    double ** inds;
};

struct IndexSet * index_set_alloc(int, size_t, size_t, size_t);
void index_set_free(struct IndexSet *);
void index_set_array_free(size_t, struct IndexSet **);
struct IndexSet ** index_set_array_rnested(size_t, size_t *, double *);
struct IndexSet ** index_set_array_lnested(size_t, size_t *, double *);
double ** index_set_merge(struct IndexSet *, struct IndexSet *,size_t *);
double ** index_set_merge_fill_end(struct IndexSet *, double **);
double ** index_set_merge_fill_beg(double **, struct IndexSet *);




/////////////////////////////////////////////////////////
// Utilities
void print_quasimatrix(struct Quasimatrix *, size_t, void *);
void print_qmarray(struct Qmarray *, size_t, void *);
void print_index_set_array(size_t, struct IndexSet **);

#endif
