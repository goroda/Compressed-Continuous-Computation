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






#ifndef ARRAY_H
#define ARRAY_H

#include <stddef.h>

// allocation functions
double *  calloc_double(const size_t);
double ** malloc_dd(const size_t);
int    *  calloc_int(const size_t);
size_t *  calloc_size_t(const size_t);
void      copy_dd(size_t, size_t, double **, double **);
void      free_dd(size_t N, double ** arr);

// initialize functions
double * dones(const size_t);
double * drandu(const size_t);
double * darray_val(const size_t, double);
double * dzeros(const size_t);
int    * izeros(const size_t);

// operations on single array
double dprod(const size_t, const double*);
int    iprod(const size_t, const int*);
size_t iprod_sz(const size_t, const size_t*);


// printing functions
void dprint(const size_t, const double *);
void dprint2d(const size_t, const size_t, const double *);
void dprint2dd(const size_t, const size_t,  double **);
void dprint2d_col(const size_t, const size_t, const double *);
void iprint(const size_t, const int *);
void iprint_sz(const size_t, const size_t *);



double * linspace(const double,const double,const size_t);
void dd_row_linspace(double **, size_t, double, double, size_t);
double * logspace(int, int, const size_t);
double * arange(const double,const double,const double, size_t *);
double * diag(const size_t, double *);
double * dconcat_cols(size_t, size_t, size_t, double *, double *);

// serialization functions
char   * serialize_double_to_text(double);
double   deserialize_double_from_text(char *);
char   * serialize_darray_to_text(size_t, const double *);
double * deserialize_darray_from_text(char *, size_t *);

char * serialize_double_packed(double);
double deserialize_double_packed(char *);
char * serialize_double_arr_packed(size_t, const double *);
    

int      darray_save(size_t, size_t, double *, char *, int);
double * darray_load(char *, int);

struct c3Vector
{
    size_t size;
    double * elem;
};

struct c3Vector * c3vector_alloc(size_t, const double * xdata);
struct c3Vector * c3vector_copy(const struct c3Vector *);
void c3vector_free(struct c3Vector *);

struct c3Vector ** c3vector_array_alloc(size_t);
struct c3Vector ** c3vector_array_copy(size_t, struct c3Vector **);
void c3vector_array_free(size_t,struct c3Vector **);

//RANDOM NUMBERS
double randu(void);
double randn(void);
size_t poisson(double);

#endif
