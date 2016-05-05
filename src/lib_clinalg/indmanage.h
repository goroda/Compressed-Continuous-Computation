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

#ifndef INDMANAGE_H
#define INDMANAGE_H

struct CrossIndex
{
    size_t d; // dimension of each node
    size_t n;
    struct CrossNode * nodes;
};

struct CrossIndex * cross_index_alloc(size_t);
void cross_index_free(struct CrossIndex *);
void cross_index_add_index(struct CrossIndex *, size_t, double *);
void cross_index_add_nested(struct CrossIndex *, int, 
                            size_t, double *, double);
double * 
cross_index_get_node_value(struct CrossIndex *,size_t,size_t *);

struct CrossIndex *
cross_index_create_nested(int, int, size_t, size_t,
                          double *, struct CrossIndex *);

struct CrossIndex *
cross_index_create_nested_ind(int, size_t, size_t *,
                              double *, struct CrossIndex *);
double **
cross_index_merge_wspace(struct CrossIndex *, struct CrossIndex *);

double **
cross_index_merge(struct CrossIndex *, struct CrossIndex *);

void cross_index_array_initialize(size_t, struct CrossIndex **,
                                  int, int,size_t *, double **);

void cross_index_copylast(struct CrossIndex *, size_t);
void print_cross_index(struct CrossIndex *);

#endif
