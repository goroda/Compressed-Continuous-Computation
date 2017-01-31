// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016, Sandia Corporation

// This file is part of the Compressed Continuous Computation (C3) Library
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



/** \file kernels.h
 * Headers for kernels.c
*/

double gauss_kernel_eval(double, double, double, double);
double gauss_kernel_deriv(double, double, double, double);
double gauss_kernel_integrate(double, double, double, double, double);
double gauss_kernel_inner(double, double, double,
                          double, double, double,
                          double, double);

struct Kernel;
struct Kernel * kernel_gaussian(double, double, double);
void kernel_free(struct Kernel *);


struct KernelApproxOpts;
void kernel_approx_opts_free(struct KernelApproxOpts *);
struct KernelApproxOpts *
kernel_approx_opts_gauss(size_t, double *, double, double);
size_t kernel_approx_opts_get_nparams(struct KernelApproxOpts *);
void kernel_approx_opts_set_nparams(struct KernelApproxOpts *, size_t);


struct KernelExpansion;
struct KernelExpansion * kernel_expansion_alloc(size_t);
double kernel_expansion_get_lb(const struct KernelExpansion *);
double kernel_expansion_get_ub(const struct KernelExpansion *);
unsigned char *
serialize_kernel_expansion(unsigned char *, struct KernelExpansion *, size_t *);
unsigned char * 
deserialize_kernel_expansion(unsigned char *, struct KernelExpansion **);
struct KernelExpansion * kernel_expansion_copy(struct KernelExpansion *);
void kernel_expansion_free(struct KernelExpansion *);
void kernel_expansion_set_bounds(struct KernelExpansion *, double, double);
void kernel_expansion_add_kernel(struct KernelExpansion *, double, struct Kernel *);

struct KernelExpansion * kernel_expansion_init(const struct KernelApproxOpts *);
void kernel_expansion_update_params(struct KernelExpansion *, size_t, const double *);
struct KernelExpansion *
kernel_expansion_create_with_params(struct KernelApproxOpts *,
                                    size_t, const double *);
struct KernelExpansion *
kernel_expansion_zero(const struct KernelApproxOpts *, int);
double kernel_expansion_eval(struct KernelExpansion *, double);
void kernel_expansion_evalN(struct KernelExpansion *, size_t,
                            const double *, size_t, double *, size_t);
double kernel_expansion_deriv_eval(struct KernelExpansion *, double);
void kernel_expansion_axpy(double, struct KernelExpansion *, struct KernelExpansion *);
double kernel_expansion_integrate(struct KernelExpansion *);
double kernel_expansion_inner(struct KernelExpansion *, struct KernelExpansion *);
void kernel_expansion_scale(double, struct KernelExpansion *);

size_t kernel_expansion_get_num_params(const struct KernelExpansion *);
size_t kernel_expansion_get_params(const struct KernelExpansion *, double *);
    
int kernel_expansion_param_grad_eval(
    struct KernelExpansion *, size_t, const double *, double *);
int
kernel_expansion_squared_norm_param_grad(const struct KernelExpansion *,
                                         double, double *);

void print_kernel_expansion(struct KernelExpansion *, size_t, void *);
