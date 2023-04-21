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



/** \file polynomials.h
 * Provides header files and structure definitions for functions in in polynomials.c
 */

#ifndef FOURIER_H
#define FOURIER_H

#include <float.h>
#include "polynomials.h"

struct OrthPoly;
struct OrthPoly * init_fourier_poly(void);
double fourier_expansion_eval(const struct OrthPolyExpansion *, double);
double fourier_expansion_deriv_eval(const struct OrthPolyExpansion *, double);
struct OrthPolyExpansion * fourier_expansion_deriv(const struct OrthPolyExpansion *);
struct OrthPolyExpansion * fourier_expansion_dderiv(const struct OrthPolyExpansion *);
int fourier_expansion_approx_vec(struct OrthPolyExpansion *,
                                 struct Fwrap *,
                                 const struct OpeOpts *);

double fourier_integrate(const struct OrthPolyExpansion *);
struct OrthPolyExpansion *
fourier_expansion_prod(const struct OrthPolyExpansion *,
                       const struct OrthPolyExpansion *);
int
fourier_expansion_axpy(double, const struct OrthPolyExpansion *,
                       struct OrthPolyExpansion *);

#endif
