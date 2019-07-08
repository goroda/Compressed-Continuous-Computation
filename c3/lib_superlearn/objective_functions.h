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


/** \file objective_functions.h
 * Provides prototypes for objective_functions.c
 */

#ifndef C3_OBJECTIVE_FUNCS
#define C3_OBJECTIVE_FUNCS

#include <stddef.h>

#include "lib_optimization.h"
#include "superlearn_util.h"

struct ObjectiveFunction;
void objective_function_add(struct ObjectiveFunction ** obj, double weight,
                            double (*func)(size_t nparam, const double * param, double * grad,
                                           size_t N, size_t * ind,
                                           struct Data * data, struct SLMemManager * ,void * arg),
                            void * arg);
void objective_function_free(struct ObjectiveFunction ** obj);
double objective_eval(size_t, const double *, double *,
                      size_t, size_t *, void *);


//////////////////////////////////////////
// Objective functions
//////////////////////////////////////////
struct LeastSquaresArgs
{
    int (*mapping)(size_t nparam, const double * param, size_t N,size_t * ind,
                   struct SLMemManager * mem, struct Data * data,
                   double ** evals, double ** grads, void * args);
    void * args;
};

double c3_objective_function_least_squares(size_t nparam, const double * param, double * grad,
                                           size_t Ndata, size_t * data_index, struct Data * data,
                                           struct SLMemManager * mem, void * args);
#endif
