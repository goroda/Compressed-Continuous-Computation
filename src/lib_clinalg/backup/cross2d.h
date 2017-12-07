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





/** \file cross2d.h
 * Provides header files for cross2d.c
 */

#ifndef CROSS2D_H
#define CROSS2D_H

struct SkeletonDecomp;
struct Cross2dargs;
struct Cross2dargs * cross2d_args_create(size_t,double,enum function_class,
                                         void *, int);
void cross2d_args_destroy(struct Cross2dargs *);
void cross2d_args_set_approx_args(struct Cross2dargs *, void *);
size_t cross2d_args_get_rank(const struct Cross2dargs *);

struct SkeletonDecomp * skeleton_decomp_alloc(size_t);
struct SkeletonDecomp *
skeleton_decomp_copy(const struct SkeletonDecomp *);
void skeleton_decomp_free(struct SkeletonDecomp *);
double * skeleton_get_skeleton(const struct SkeletonDecomp *);

struct SkeletonDecomp * 
skeleton_decomp_init2d_from_pivots(
    double (*)(double,double,void *),
    void *, const struct BoundingBox *,
    const struct Cross2dargs *,
    const double *, const double *);
    


double skeleton_decomp_eval(const struct SkeletonDecomp *,
                            double, double);


struct SkeletonDecomp *
cross_approx_2d(double (*)(double, double, void *), 
                void *, struct BoundingBox *,
                struct SkeletonDecomp **,double *,
                double *, struct Cross2dargs *);

#endif
