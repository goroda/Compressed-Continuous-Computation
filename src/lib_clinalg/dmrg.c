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

/** \file algs.c
 * Provides routines for dmrg based algorithms for the FT
 */
#include <stdlib.h>
#include <assert.h>

#include "lib_clinalg.h"

/***********************************************************//**
    Update \f$ \Psi \f$ for the dmrg equations

    \param psikp [in] - \f$ \Psi_{k+1} \f$
    \param left [in] - left core for update
    \param right [in] - right core for update

    \return val -  \f$ \Psi_k \f$

    \note
       \f$ \Psi_k = \int left(x) \Psi_{k+1} right^T(x) dx \f$
***************************************************************/
double * dmrg_update_right(double * psikp, struct Qmarray * left, struct Qmarray * right)
{
    struct Qmarray * temp = qmam(left,psikp,right->ncols);
    struct Qmarray * temp2 = qmaqmat(temp,right);
    double * val = qmarray_integrate(temp2);

    qmarray_free(temp); temp = NULL;
    qmarray_free(temp2); temp2 = NULL;
    return val;
}

/***********************************************************//**
    Generate all \f$ \Psi_{i} \f$ for the dmrg for \f$ i = 0,\ldots,d-2\f$

    \param a [in] - function train that is getting optimized
    \param b [in] - right hand side
    \param mats [inout] - allocated space for $d-2$ doubles representing the matrices

    \note
       \f$ \Psi_k = \int left(x) \Psi_{k+1} right^T(x) dx \f$
***************************************************************/
void dmrg_update_all_right(struct FunctionTrain * a, struct FunctionTrain * b, double ** mats)
{
    size_t ii;
    mats[a->dim-2] = calloc_double(1);
    mats[a->dim-2][0] = 1.0;
    for (ii = a->dim-3; ii > 0; ii--){
        mats[ii] = dmrg_update_right(mats[ii+1],a->cores[ii+2],b->cores[ii+2]);
    }
    mats[0] = dmrg_update_right(mats[1],a->cores[2],b->cores[2]);
}

/***********************************************************//**
    Update \f$ \Phi \f$ for the dmrg equations

    \param phik [in] - \f$ \Phi_{k} \f$
    \param left [in] - left core for update
    \param right [in] - right core for update

    \return val -  \f$ \Phi_{k+1} \f$

    \note
       \f$ \Phi_{k+1} = \int left(x)^T \Psi_{k} right(x) dx \f$
***************************************************************/
double * dmrg_update_left(double * phik, struct Qmarray * left, struct Qmarray * right)
{
    struct Qmarray * temp = mqma(phik, right, left->nrows);
    struct Qmarray * temp2 = qmatqma(left,temp);
    double * val = qmarray_integrate(temp2);

    qmarray_free(temp); temp = NULL;
    qmarray_free(temp2); temp2 = NULL;
    return val;
}
 
struct FunctionTrain * dmrg_sweep_lr(struct FunctionTrain * a, struct FunctionTrain * b, double ** phi, double ** psi)
{
    struct FunctionTrain * na = function_train_alloc(a->dim);
    
    phi[0] = calloc_double(1);
    phi[0][0] = 1;
    size_t ii = 0;
    for (ii = 0; ii < a->dim-1; ii++){
        struct Qmarray * newcorer = qmam(b->core[ii+1],psi[ii],a->cores[ii+2]->nrows);
        // take LQ of this
        
        struct Qmarray * newcorel = mqma(phi[ii],b->cores[ii],a->cores[ii-1]->ncols);
        // take QR of this

        // multiply RL
        // svd of RL
        // new left core and new right core
    }

    return na;
}


/*
void fast_kron(size_t n, double * C, struct Qmarray * A, struct Qmarray * B, struct Qmarray ** out)
{
    assert (*out == NULL);
    struct Qmarray * temp = mqma(C,A,n);
    //return out;
}
*/


//void dmrg_core_mid_mult(double * leftmat, double * rightmat, 
//        size_t k, struct FT1DArray * 
