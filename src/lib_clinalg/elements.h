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

/** \file elements.h
 * Provides header files and structure definitions for functions in elements.c
 */

#ifndef ELEMENTS_H
#define ELEMENTS_H

#include <stdlib.h>
#include "../lib_array/array.h"
#include "../lib_funcs/lib_funcs.h"
#include "quasimatrix.h"


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
struct Qmarray * qmarray_zeros(enum poly_type,size_t, size_t,double,double);
struct Qmarray * 
qmarray_poly_randu(enum poly_type, size_t, size_t, size_t, double, double);
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
struct Qmarray *
qmarray_orth1d_linelm_grid(size_t,size_t, struct c3Vector *);

struct Quasimatrix * qmarray_extract_column(const struct Qmarray *, size_t);
struct GenericFunction *
qmarray_get_func(const struct Qmarray *, size_t, size_t);
struct Quasimatrix * qmarray_extract_row(const struct Qmarray *, size_t);
/* struct Qmarray * qmarray_extract_ncols(struct Qmarray *, size_t); */


void qmarray_set_column(struct Qmarray *, size_t, const struct Quasimatrix *);
void qmarray_set_column_gf(struct Qmarray *, size_t, 
                           struct GenericFunction **);
void qmarray_set_row(struct Qmarray *, size_t, const struct Quasimatrix *);

unsigned char * 
qmarray_serialize(unsigned char *, struct Qmarray *, size_t *);
unsigned char * 
qmarray_deserialize(unsigned char *, struct Qmarray ** );

////////////////////////////////////////////////////////////////////
// function_train 

struct FtOneApprox
{
    enum function_class fc;
    void * sub_type;
    void * aopts;
};

/** \struct FtApproxArgs
 * \brief function train approximation arguments
 * \var FtApproxArgs::dim
 * function dimension
 * \var FtApproxArgs::aopts
 * function approximation options
 */
struct FtApproxArgs
{
    size_t dim;
    struct FtOneApprox ** aopts;
};

struct FtApproxArgs * 
ft_approx_args_createpoly(size_t, enum poly_type *,
                          struct OpeAdaptOpts *);
struct FtApproxArgs * 
ft_approx_args_createpwpoly(size_t, enum poly_type *,
                            struct PwPolyAdaptOpts *);
struct FtApproxArgs * 
ft_approx_args_create_le(size_t, 
                         struct LinElemExpAopts *);

struct FtApproxArgs * 
ft_approx_args_create_le2(size_t, 
                         struct LinElemExpAopts **);

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

    double ** evaldd1;
    double ** evaldd2;
    double ** evaldd3;
    double ** evaldd4;
   
};

struct BoundingBox * function_train_bds(struct FunctionTrain *);

struct FunctionTrain * function_train_alloc(size_t); 
struct FunctionTrain * function_train_copy(struct FunctionTrain *);
void function_train_free(struct FunctionTrain *);

struct FunctionTrain *
function_train_poly_randu(enum poly_type,struct BoundingBox *, size_t *, size_t);

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
function_train_constant(enum function_class, void *,
                        size_t, double, struct BoundingBox *,void*);
struct FunctionTrain * 
function_train_constant_d(struct FtApproxArgs *,double,
                          struct BoundingBox *);                      
struct FunctionTrain * 
function_train_linear(enum function_class, void *,size_t, struct BoundingBox *,
                      double *, void* ); 

struct FunctionTrain * 
function_train_linear2(enum function_class,const void *,size_t, const struct BoundingBox *, 
                       const double *, size_t, const double *, size_t,void*);

struct FunctionTrain * 
function_train_quadratic(enum function_class, void *,size_t, struct BoundingBox *,
                         double *, double *,void *); 
struct FunctionTrain * 
function_train_quadratic_aligned(enum function_class, void *,
                                 struct BoundingBox *, double *, 
                                 double *, void *);

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

struct FiberOptArgs
{
    size_t dim;
    void ** opts;
};
struct FiberOptArgs * fiber_opt_args_alloc();
struct FiberOptArgs * fiber_opt_args_init(size_t);
struct FiberOptArgs * fiber_opt_args_bf(size_t,struct c3Vector **);

struct FiberOptArgs * fiber_opt_args_bf_same(size_t,struct c3Vector *);
void fiber_opt_args_free(struct FiberOptArgs *);

/////////////////////////////////////////////////////////
// Utilities
void print_quasimatrix(struct Quasimatrix *, size_t, void *);
void print_qmarray(struct Qmarray *, size_t, void *);

#endif
