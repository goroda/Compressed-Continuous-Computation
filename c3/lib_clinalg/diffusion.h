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





/** \file diffusion.h
 * Provides header files and structure definitions for functions in diffusion.c
 */

#ifndef DIFFUSION_H
#define DIFFUSION_H

#include "dmrg.h"

/// @private
void dmrg_diffusion_midleft(struct Qmarray *, struct Qmarray *,
                 struct Qmarray *, struct Qmarray *, struct Qmarray *,
                 double *, size_t, struct Qmarray *);

/// @private
void dmrg_diffusion_lastleft(struct Qmarray *, struct Qmarray *,
                 struct Qmarray *, struct Qmarray *, struct Qmarray *,
                 double *, size_t, struct Qmarray *);
/// @private
void dmrg_diffusion_midright(struct Qmarray *, struct Qmarray *,
                 struct Qmarray *, struct Qmarray *, struct Qmarray *,
                 double *, size_t, struct Qmarray *);
/// @private
void dmrg_diffusion_firstright(struct Qmarray *, struct Qmarray *,
                 struct Qmarray *, struct Qmarray *, struct Qmarray *,
                 double *, size_t, struct Qmarray *);
/// @private
struct DmDiff
{
    struct FunctionTrain * A;
    struct FunctionTrain * F;
    struct Qmarray ** dA;
    struct Qmarray ** dF;
    struct Qmarray ** ddF;
};


struct FunctionTrain * dmrg_diffusion(
    struct FunctionTrain *,
    struct FunctionTrain *,
    double, size_t, double, int, struct MultiApproxOpts *);

struct FunctionTrain *
exact_diffusion(struct FunctionTrain *, struct FunctionTrain *,
                struct MultiApproxOpts *);

struct FunctionTrain *
exact_laplace(const struct FunctionTrain *,
              struct MultiApproxOpts *);
struct FunctionTrain *
exact_laplace_periodic(struct FunctionTrain *,
                       struct MultiApproxOpts *);
struct FunctionTrain *
exact_laplace_op(struct FunctionTrain *,
                 struct Operator **,
                 struct MultiApproxOpts *);

struct OperatorForLaplace;
struct OperatorForLaplace *
build_lp_operator(size_t dim, size_t len1, const double * evalnd_pt);
struct Operator ** operator_for_laplace_get_op(struct OperatorForLaplace *op);

void destroy_operator_for_laplace(struct OperatorForLaplace * op);

#endif
