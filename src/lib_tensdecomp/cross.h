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








#ifndef CROSS_H
#define CROSS_H

#include <stdlib.h>

#include "linalg.h"
#include "tensortrain.h"

//#include "listutils.h"

struct index_set {
    size_t dim;
    size_t N;
    size_t ** indices;
};

void create_naive_index_set(struct index_set *, size_t *);
void print_index_set(struct index_set *);
void free_index_set(struct index_set **);

// decompositions
struct cross_fiber_list{
    size_t pre_length;
    size_t post_length;
    size_t * pre_index;
    size_t * post_index;
    double * vals;
    struct cross_fiber_list * next;
};

struct cross_fiber_info{
    size_t nfibers;
    struct cross_fiber_list * head;
};

void AddCrossFiber(struct cross_fiber_list **, size_t, size_t, 
                    size_t *, size_t *, double *, size_t);
int CrossIndexExists(struct cross_fiber_list *, size_t *, size_t *);
double * getCrossFiberListIndex(struct cross_fiber_list *, size_t);
void DeleteCrossFiberList(struct cross_fiber_list **);


struct tt_cross_opts {
    
    size_t maxiter;
    int verbose;
    double epsilon;
    int success;
    size_t dim;
    size_t * ranks;
    size_t * nvals;
    struct index_set ** right;
    struct index_set ** left;
    struct cross_fiber_info ** fibers;
};

struct tt_cross_opts * init_cross_opts(size_t, size_t *, size_t *);
struct tt_cross_opts * init_cross_opts_with_naive_set(size_t, size_t *, 
                                                                size_t *);

void free_cross_opts(struct tt_cross_opts **);

struct tt * tt_cross(void (*A)(size_t, size_t *, size_t, double *, void *), 
                     struct tt_cross_opts *, void *);
struct tt * tt_cross_adapt
(void (*A)(size_t, size_t *, size_t, double *, void *), 
                struct tt_cross_opts *, size_t, size_t, double, void *);

// wrappers for calling cross
struct func_to_array {
    size_t * nvals;
    size_t dim;
    double ** pts;
    double (*f)(double *, void *);
    void * args;
};

void wrap_func_for_cross(size_t, size_t *, size_t, double *, void *);
void wrap_full_tensor_for_cross(size_t, size_t *, size_t, double *, void *);


#endif
