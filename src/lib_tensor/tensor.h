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









#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <stdio.h>

struct tensor {
    size_t dim;
    size_t * nvals;
    double * vals;
};

void init_tensor(struct tensor **, size_t, size_t *);
struct tensor * read_tensor_3d(FILE *);
void free_tensor(struct tensor **);
double tensor_elem(struct tensor *, size_t *);

struct tensor * tensor_stack2h_3d(const struct tensor *, const struct tensor *);
struct tensor * tensor_stack2v_3d(const struct tensor *, const struct tensor *);
struct tensor * tensor_blockdiag_3d(const struct tensor *, const struct tensor *);

struct tensor * tensor_kron_3d(const struct tensor *, const struct tensor *);
void check_right_ortho(struct tensor * core);

double * tensor_sum2(const struct tensor *,const double *);
struct tensor * tensor_ones_3d(size_t, size_t, size_t);
struct tensor * tensor_x_3d(size_t, const double *);
struct tensor * tensor_copy_3d(const struct tensor *, int);


void print_tensor_3d(struct tensor *, int);

#endif
