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





/** \file quasimatrix.h 
 Provides header files for quasimatrix.c 
*/

#ifndef QUASIMATRIX_H
#define QUASIMATRIX_H

#include "lib_funcs.h"

struct Quasimatrix;

// allocation
struct Quasimatrix *
quasimatrix_alloc(size_t);
void quasimatrix_free_funcs(struct Quasimatrix *);
void quasimatrix_free(struct Quasimatrix *);


// getting and setting
size_t quasimatrix_get_size(const struct Quasimatrix *);
void quasimatrix_set_size(struct Quasimatrix *,size_t);
struct GenericFunction *
quasimatrix_get_func(const struct Quasimatrix *, size_t);
void
quasimatrix_set_func(struct Quasimatrix *,
                     const struct GenericFunction *,
                     size_t);

struct GenericFunction **
quasimatrix_get_funcs(const struct Quasimatrix *);
void quasimatrix_set_funcs(struct Quasimatrix *, struct GenericFunction **);
void
quasimatrix_get_funcs_ref(const struct Quasimatrix *,struct GenericFunction ***);

/* struct Quasimatrix * */
/* quasimatrix_init(size_t,size_t,enum function_class *,void **, */
/*                  void **,void **);     */

struct Quasimatrix * 
quasimatrix_approx1d(size_t, struct Fwrap *,
                     enum function_class *,
                     void *);

/* struct Quasimatrix *  */
/* quasimatrix_approx_from_fiber_cuts(size_t, */
/*                                    double (*)(double, void *), */
/*                                    struct FiberCut **, */
/*                                    enum function_class, */
/*                                    void *, double, */
/*                                    double, void *); */

struct Quasimatrix *
quasimatrix_copy(const struct Quasimatrix *);

unsigned char * 
quasimatrix_serialize(unsigned char *,
                      const struct Quasimatrix *,
                      size_t *);
unsigned char * 
quasimatrix_deserialize(unsigned char *,
                        struct Quasimatrix **);


struct Quasimatrix *
quasimatrix_orth1d(size_t,enum function_class,void *);

size_t
quasimatrix_absmax(struct Quasimatrix *,
                   double *,double *,void*);

///////////////////////////////////////////
// Linear Algebra
///////////////////////////////////////////

// (quasimatrix - vector multiplication
double quasimatrix_inner(const struct Quasimatrix *,
                         const struct Quasimatrix *);
struct GenericFunction *
qmv(const struct Quasimatrix *, const double *);
struct Quasimatrix * qmm(const struct Quasimatrix *,
                         const double *,size_t);
struct Quasimatrix * qmmt(const struct Quasimatrix*,
                          const double *,size_t);
struct Quasimatrix *
quasimatrix_daxpby(double, const struct Quasimatrix *,
                   double, const struct Quasimatrix *);


// decompositions
struct Quasimatrix *
quasimatrix_householder_simple(struct Quasimatrix *,double *,void*); 
int quasimatrix_householder(struct Quasimatrix *,
                            struct Quasimatrix *, 
                            struct Quasimatrix *, double *);

int quasimatrix_qhouse(struct Quasimatrix *,
                       struct Quasimatrix *);

int quasimatrix_lu1d(struct Quasimatrix *,
                     struct Quasimatrix *,
                     double *,double *,void*,void *);

int quasimatrix_maxvol1d(struct Quasimatrix *,
                         double *, double *,void*,void*);

size_t quasimatrix_rank(const struct Quasimatrix *,void*);
double quasimatrix_norm(const struct Quasimatrix *);


// Utilities
void quasimatrix_print(const struct Quasimatrix *,FILE *,
                       size_t, void *);

#endif
