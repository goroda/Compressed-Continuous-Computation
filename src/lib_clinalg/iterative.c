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





/** \file iterative.c
 * Provides routines for continuous iterative methods
 */

#include "lib_clinalg.h"

/*
void c3_lanczos(void (*f)(struct FunctionTrain *, struct FunctionTrain **,void *), void * args,
                double epsilon, struct FT1DArray * out, 
                double * alpha, double * beta,  size_t niters)
{
    
    size_t ii;
    ii = 0;
    struct FunctionTrain * new = NULL;
    f(out->ft[ii],&new,args);
    alpha[ii] = function_train_inner(new,out->ft[ii]);
    c3axpy(-alpha[ii],out->ft[ii],&new,epsilon);
    out->ft[ii+1] = function_train_round(new,epsilon);
    beta[ii] = function_train_norm2(out->ft[ii+1]);
    function_train_scale(out->ft[ii+1],1.0/beta[ii]);
    function_train_free(new); new = NULL;
    
    for (ii = 1; ii < niters-1; ii++){
        f(out->ft[ii],&new,args);
        alpha[ii] = function_train_inner(new,out->ft[ii]);
        c3axpy(-alpha[ii],out->ft[ii],&new,epsilon);
        c3axpy(-beta[ii-1],out->ft[ii-1],&new,epsilon);
        out->ft[ii+1] = function_train_round(new,epsilon);
        beta[ii] = function_train_norm2(out->ft[ii+1]);
        function_train_scale(out->ft[ii+1],1.0/beta[ii]);
        function_train_free(new); new = NULL;
    }
    
    f(out->ft[niters-1],&new,args);
    alpha[niters-1] = function_train_inner(new,out->ft[niters-1]);
    function_train_free(new);
    
    //need to call dstegr next
    double waste;
    size_t wastei;
    double * eigs = calloc_double(niters);
    double * evecs = calloc_double(niters*niters);
    //dstegr_('V','A',niters,alpha,beta,waste,waste,wasti,wasti,waste,niters,eigs,evecs,niters)
}
*/



