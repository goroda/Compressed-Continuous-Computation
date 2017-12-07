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


////////////////////////////////////////////////////////////////////
// function_train 


struct BoundingBox * function_train_bds(struct FunctionTrain *);

struct FunctionTrain * function_train_alloc(size_t); 
struct FunctionTrain * function_train_copy(struct FunctionTrain *);
void function_train_free(struct FunctionTrain *);

struct FunctionTrain *
function_train_poly_randu(enum poly_type,struct BoundingBox *, size_t *, size_t);

struct FunctionTrain *
function_train_rankone(struct MultiApproxOpts *, struct Fwrap *);


struct FunctionTrain * function_train_initsum(size_t, 
        double (**)(double, void *), void **, struct BoundingBox *,
        struct FtApproxArgs * ); 

struct FunctionTrain * 
function_train_initsum2(size_t,  double (*f)(double, size_t, void *), 
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



struct FT1DArray * ft1d_array_alloc(size_t);
unsigned char * ft1d_array_serialize(unsigned char *, struct FT1DArray *, size_t *);
unsigned char * ft1d_array_deserialize(unsigned char *, struct FT1DArray ** );
int ft1d_array_save(struct FT1DArray *, char *);
struct FT1DArray * ft1d_array_load(char *);

struct FT1DArray * ft1d_array_copy(struct FT1DArray *);
void ft1d_array_free(struct FT1DArray *);



/////////////////////////////////////////////////////////
// Utilities
void print_quasimatrix(struct Quasimatrix *, size_t, void *);
void print_qmarray(struct Qmarray *, size_t, void *);

#endif
