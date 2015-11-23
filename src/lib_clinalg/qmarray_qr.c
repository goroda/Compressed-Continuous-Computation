// Copyright (c) 2014-2015, Massachusetts Institute of Technology
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

/** \file qmarray_qr.c
 * Provides routines for fast computation of qr decomposition of a quasimatrix array
 */

#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "elements.h"
#include "algs.h"

#include "lib_funcs.h"
#include "array.h"

#define ZERO 1e-16

int qmarray_qr(struct Qmarray * A, struct Qmarray ** Q, double ** R)
{

    size_t ii,kk,ll;

    double lb = generic_function_get_lower_bound(A->funcs[0]);
    double ub = generic_function_get_upper_bound(A->funcs[0]);
    size_t ncols = A->ncols;
    size_t nrows = A->nrows;
    enum poly_type ptype = LEGENDRE;
    *Q = qmarray_orth1d_columns(POLYNOMIAL,&ptype,nrows,ncols, lb, ub); 
    
    *R = calloc_double(ncols*ncols);
    
    struct Qmarray * V = qmarray_alloc(nrows,ncols);
    for (ii = 0; ii < nrows * ncols; ii++){
        V->funcs[ii] = generic_function_alloc_base(1.0);    
    }

    double s, rho, alpha, sigma;
    for (ii = 0; ii < ncols; ii++){
        
        rho = generic_function_array_norm(nrows,1,A->funcs+ii*nrows);
        (*R)[ii*ncols+ii] = rho;
        alpha = generic_function_inner_sum(nrows,1,(*Q)->funcs+ii*nrows,1, A->funcs+ii*nrows);
        
        s = -alpha / fabs(alpha);    
        if (s < 0.0){
            generic_function_array_flip_sign(nrows,1,(*Q)->funcs+ii*nrows);
        }
        
        //should be able to get sigma without doing computation
        sigma = rho*rho - rho * (2.0*s*alpha - 1.0);
        if (fabs(sigma) < ZERO){
            //V=E
            for (kk = 0; kk < nrows; kk++){
                generic_function_copy_pa((*Q)->funcs[ii*nrows+kk],V->funcs[ii*nrows+kk]);
            }
        }
        else{
            for (kk = 0; kk < nrows; kk++){
                generic_function_weighted_sum_pa(
                    rho,(*Q)->funcs[ii*nrows+kk],-1.0,A->funcs[ii*nrows+kk],
                    V->funcs[ii*nrows+kk]);
                sigma += generic_function_inner(V->funcs[ii*nrows+kk],V->funcs[ii*nrows+kk]);
            }
            generic_function_array_scale(1.0/pow(sigma,2),V->funcs+ii*nrows, nrows);
        }
        
        double ev = generic_function_inner_sum(nrows,1,(*Q)->funcs+ii*nrows,1,V->funcs+ii*nrows);
        for (kk = ii+1; kk < ncols; kk++){
            double temp = generic_function_inner_sum(nrows,1,V->funcs+ii*nrows,1,A->funcs+kk*nrows);
            (*R)[ii*nrows+kk] = generic_function_inner_sum(nrows,1,(*Q)->funcs+ii*nrows,1,A->funcs+kk*nrows);
            (*R)[ii*nrows+kk] += (-2.0 * ev * temp);
            for (ll = 0; ll < nrows; ll++){
                generic_function_sum3_up(-2.0*temp,V->funcs[ii*nrows+ll],
                                        (*R)[ii*nrows+kk],(*Q)->funcs[ii*nrows+ll],
                                        1.0,A->funcs[ii*nrows+ll]);
            }

        }
    }
    
    // form Q
    size_t counter = 0;
    size_t ii = ncols-1;
    while (counter < ncols){
        
        for (jj = ii; jj < ncols; jj++){
            double val = generic_function_inner_sum(nrows,1,
                    (*Q)->funcs+ii*nrows, 1, V->funcs+ii*nrows);
            // NEED to update here
        }

        ii = ii - 1;
    }


    return 1;
}

