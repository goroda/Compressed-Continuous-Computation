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





/** \file dmrgprod.c
* Provides routines for computing products of functions using ALSDMRG
*/

#include <assert.h>

#include "dmrgprod.h"
#include "lib_clinalg.h"

/// @private
void dmrg_prod_support(char type, size_t core, size_t r, double * mat,
                        struct Qmarray ** out, void * args)
{
    assert (*out == NULL);
    struct DmProd * ab = args;
    size_t dim = ab->a->dim;
    //printf("in \n");
    if (type == 'L'){
        if (core == 0){
            *out = qmarray_kron(ab->a->cores[0],ab->b->cores[0]);   
        }
        else{
            *out = qmarray_mat_kron(r,mat,ab->a->cores[core],ab->b->cores[core]);
        }
    }
    else if (type == 'R'){
        if (core == dim-1){
            *out = qmarray_kron(ab->a->cores[core],ab->b->cores[core]);
        }
        else{
            *out = qmarray_kron_mat(r,mat,ab->a->cores[core],ab->b->cores[core]);
        }
    }
    //printf("out \n");
}


/***********************************************************//**
    Compute \f$ z(x) = a(x)b(x) \f$ using ALS+DMRG

    \param[in,out] z          - initial guess (destroyed);
    \param[in]     a          - first element of product
    \param[in]     b          - second element of product
    \param[in]     delta      - threshold to stop iterating
    \param[in]     max_sweeps - maximum number of left-right-left sweeps 
    \param[in]     epsilon    - SVD tolerance for rank determination
    \param[in]     verbose    - verbosity level 0 or >0
    \param[in]     opts       - approximation options

***************************************************************/
struct FunctionTrain *
dmrg_product(struct FunctionTrain * z,
             struct FunctionTrain * a, struct FunctionTrain * b,
             double delta, size_t max_sweeps, double epsilon, int verbose,
             struct MultiApproxOpts * opts)
{
    assert(a != NULL);
    assert(b != NULL);
    struct DmProd dm;
    dm.a = a;
    dm.b = b;
    
    struct FunctionTrain * zin;
    if (z == NULL){
        zin = function_train_copy(a);    
    }
    else{
        zin = z;
    }
    struct FunctionTrain * na =
        dmrg_approx(zin,dmrg_prod_support,&dm,delta,max_sweeps,epsilon,verbose,opts);

    if (z == NULL){
        function_train_free(zin);
    }
    return na;
}

