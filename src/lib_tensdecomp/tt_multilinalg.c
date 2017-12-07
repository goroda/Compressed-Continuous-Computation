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








#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "array.h"
#include "tt_multilinalg.h"


/********************************************************//**
    Function tt_round

    Purpose: Round a TT decomp to a certain accuracy
             (Rounding Algorithm from Oseledets 2010)

    Parameters:
        - a (IN) - tt-tensor to compress 
                gets destroyed but not deallocated
        - epsilon (IN) - Relative Error of compression (Fro norm)

    Returns: b - tensor in TT format

***********************************************************/
struct tt * tt_round(struct tt * a, double epsilon)
{
    // STILL NEED TO COMPUTE DELTA 
    double nrm = tt_norm(a);
    double delta = nrm * epsilon;
    if (a->dim > 1) delta = delta / sqrt(a->dim-1.0);

    double * s;
    double * vt;
    double * temp;
    struct mat * sdiag;
    size_t ii;
    struct tt * b;

    right_left_orth(&a);
    init_tt_alloc(&b, a->dim, a->nvals);
    b->ranks[0] = a->ranks[0];
    b->ranks[a->dim] = a->ranks[a->dim];

    //first core
    if (NULL == ( b->cores[0] = malloc(sizeof(struct tensor)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    b->cores[0]->nvals = calloc_size_t(3);
    b->cores[0]->nvals[0] = b->ranks[0];
    b->cores[0]->nvals[1] = a->nvals[0];
    b->ranks[1] = truncated_svd(a->nvals[0]* a->ranks[0], a->ranks[1], 
                                a->nvals[0] * a->ranks[0],a->cores[0]->vals,
                                &(b->cores[0]->vals), &s, &vt, delta); 
    b->cores[0]->nvals[2] = b->ranks[1];
    
       
    for (ii = 1; ii < a->dim-1; ii++){
        sdiag = diagv2m(b->ranks[ii], s);

        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, 
                    CblasNonUnit, b->ranks[ii], a->ranks[ii], 1.0, 
                    sdiag->vals, b->ranks[ii],  vt, b->ranks[ii]);

        freemat(sdiag);
        free(s);
         
        if (NULL == ( b->cores[ii] = malloc(sizeof(struct tensor)))){
            fprintf(stderr, "failed to allocate memory.\n");
            exit(1);
        }
        b->cores[ii]->nvals = calloc_size_t(3);
        b->cores[ii]->nvals[0] = b->ranks[ii];
        b->cores[ii]->nvals[1] = a->nvals[ii];
         
        temp = calloc_double(b->ranks[ii]*a->ranks[ii+1]*a->nvals[ii]);

        // I AM HERE
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, b->ranks[ii], 
                    a->ranks[ii+1]*a->nvals[ii], a->ranks[ii], 1.0, vt,
                    b->ranks[ii], a->cores[ii]->vals, a->ranks[ii],
                    0.0, temp, b->ranks[ii]);
        free(vt);

        b->ranks[ii+1] = truncated_svd(a->nvals[ii]*b->ranks[ii], 
                         a->ranks[ii+1], a->nvals[ii] * b->ranks[ii],
                         temp, &(b->cores[ii]->vals), &s, &vt, delta); 
        b->cores[ii]->nvals[2] = b->ranks[ii+1];
        free(temp);
    }
    // last core
    ii = a->dim-1;
    sdiag = diagv2m(b->ranks[ii], s);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, 
                CblasNonUnit, b->ranks[ii], a->ranks[ii], 1.0, 
                sdiag->vals, b->ranks[ii],  vt, b->ranks[ii]);

    freemat(sdiag);
    free(s);
 
    if (NULL == ( b->cores[ii] = malloc(sizeof(struct tensor)))){
            fprintf(stderr, "failed to allocate memory.\n");
            exit(1);
    }
    b->cores[ii]->nvals = calloc_size_t(3);
    b->cores[ii]->nvals[0] = b->ranks[ii];
    b->cores[ii]->nvals[1] = a->nvals[ii];
    b->cores[ii]->nvals[2] = a->ranks[ii+1];
    b->ranks[ii+1] = a->ranks[ii+1];

    b->cores[ii]->vals = calloc_double(b->ranks[ii]*a->ranks[ii+1]*a->nvals[ii]);

    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, b->ranks[ii], 
                a->ranks[ii+1]*a->nvals[ii], a->ranks[ii], 1.0, vt, 
                b->ranks[ii], a->cores[ii]->vals, a->ranks[ii],
                0.0, b->cores[ii]->vals, b->ranks[ii]);
    free(vt);

    return b;
}

/********************************************************//**
    Function ttscal

    Purpose: scale a tensor in TT form by a scalar

    Parameters:
        - ttc (in/out) - tensor in TT  form
        - scal (in) - scaling factor

    Returns: None. void function
***********************************************************/
void ttscal(struct tt * ttc, double scal)
{
    // just scale first core
    size_t * nvals = ttc->cores[0]->nvals;
    cblas_dscal(nvals[0] * nvals[1] * nvals[2], scal, 
                ttc->cores[0]->vals, 1);
}

/********************************************************//**
    Function ttmult

    Purpose: Hadamard (elementwise) product of two tensors in
             TT format

    Parameters:
        - a (in) - TT 1
        - b (in) - TT

    Returns: tens - elementwise multiplied tensor
***********************************************************/
struct tt * ttmult(const struct tt * a, const struct tt * b)
{
    struct tt * tens; 
    init_tt_alloc(&tens, a->dim, a->nvals);
    
    size_t ii;
    for (ii = 0; ii < tens->dim; ii++){
        tens->ranks[ii] = a->ranks[ii] * b->ranks[ii];
        tens->cores[ii] = tensor_kron_3d(a->cores[ii], b->cores[ii]);
    }

    tens->ranks[tens->dim] = a->ranks[tens->dim] * b->ranks[tens->dim];
    return tens;
}

/********************************************************//**
    Function ttadd

    Purpose: elementwise addition of two tensors in TT format

    Parameters:
        - a (in) - TT 1
        - b (in) - TT 2

    Returns: tens - elementwise added  tensor
***********************************************************/
struct tt * ttadd(const struct tt * a, const struct tt * b)
{
    struct tt * tens;
    init_tt_alloc(&tens, a->dim, a->nvals);
    
    // first core
    tens->cores[0] = tensor_stack2h_3d(a->cores[0], b->cores[0]);
    tens->ranks[0] = a->ranks[0];

    // middle cores
    size_t ii;
    for (ii = 1; ii < tens->dim-1; ii++){
        tens->cores[ii] = tensor_blockdiag_3d(a->cores[ii], b->cores[ii]);
        tens->ranks[ii] = a->ranks[ii] + b->ranks[ii];
    }
    tens->ranks[tens->dim-1] = a->ranks[a->dim-1] + b->ranks[b->dim-1];
    // last core
    tens->cores[tens->dim-1] = tensor_stack2v_3d(a->cores[a->dim-1],
                                b->cores[b->dim-1]);
    tens->ranks[tens->dim] = a->ranks[tens->dim];
    
    return tens;
}


double tt_dot(const struct tt * a, const struct tt * b){
    
    double * v2a = calloc_double(a->ranks[1] * b->ranks[1]);
    size_t ii, jj;
    //printf("here again old friend\n");
    for (ii = 0; ii < a->nvals[0]; ii++){
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    b->ranks[1], a->ranks[1], 1, 1.0,
                    b->cores[0]->vals + ii, b->nvals[0],
                    a->cores[0]->vals + ii, a->nvals[0],
                    1.0, v2a, b->ranks[1]);

    }
    //printf("here again old friend\n");
    //printf("a->ranks [1] =%zu\n", a->ranks[1]);
    //printf("b->ranks [1] =%zu\n", b->ranks[1]);
    //printf("v2a = "); dprint(a->ranks[1] * b->ranks[1], v2a);
    double * v2b = NULL;
    for (ii = 1; ii < a->dim ; ii++){
        if (v2b == NULL){
            /*
            if (isnan(v2a[0])){
                printf("HI!\n");
                printf ("isnan v2a[0]:%d\n", isnan(v2a[0]));
                exit(1);
            }
            if (isinf(v2a[0])){
                printf("HI! ii=%zu \n", ii);
                printf ("isinf v2a[0]:%d\n", isinf(v2a[0]));
                exit(1);
            }
            */
            v2b = calloc_double(a->ranks[ii+1] * b->ranks[ii+1]);
            for (jj = 0; jj < a->nvals[ii]; jj++){
                vec_kron( a->ranks[ii], a->ranks[ii+1], a->cores[ii]->vals + 
                    jj * a->ranks[ii], a->ranks[ii] * a->nvals[ii],
                    b->ranks[ii], b->ranks[ii+1], b->cores[ii]->vals + 
                    jj * b->ranks[ii], b->ranks[ii] * b->nvals[ii],
                    v2a, 1.0, v2b);
                /*
                if (isnan(v2b[0])){
                    printf("HI! ii=%zu, jj=%zu \n",ii,jj);
                    printf("v2a = "); dprint(a->ranks[ii] * b->ranks[ii], v2a);
                    printf ("isnan v2b[0]:%d\n", isnan(v2b[0]));
                    exit(1);
                }
                */
            }
            
            free(v2a);
            v2a = NULL;
        }

        else {
            /*
            if (isnan(v2b[0])){
                printf("HI!\n");
                printf ("isnan v2b[0]:%d\n", isnan(v2b[0]));
                exit(1);
            }
            if (isinf(v2b[0])){
                printf("HI! ii=%zu \n", ii);
                printf ("isinf v2b[0]:%d\n", isinf(v2b[0]));
                exit(1);
            }
            */
            v2a = calloc_double(a->ranks[ii+1] * b->ranks[ii+1]);
            for (jj = 0; jj < a->nvals[ii]; jj++){
                vec_kron( a->ranks[ii], a->ranks[ii+1], a->cores[ii]->vals + 
                    jj * a->ranks[ii], a->ranks[ii] * a->nvals[ii],
                    b->ranks[ii], b->ranks[ii+1], b->cores[ii]->vals + 
                    jj * b->ranks[ii], b->ranks[ii] * b->nvals[ii],
                    v2b,1.0, v2a);
                /*
                if (isinf(v2a[0])){
                    printf("HI! 333 ii=%zu jj=%zu \n", ii,jj);
                    printf("v2b = "); dprint(a->ranks[ii] * b->ranks[ii], v2b);
                    printf ("isinf v2a[0]:%d\n", isinf(v2a[0]));
                    exit(1);
                }
                */
            }

            free(v2b);
            v2b = NULL;
        }
    }

    double out;

    if (v2b == NULL){
        out = v2a[0];
        //printf("final v2a = "); dprint(1, v2a);
        free(v2a);
    }
    else{
        out = v2b[0];
        //printf("final v2b = "); dprint(1, v2b);
        free(v2b);
    }

    return out;
}


double tt_norm(const struct tt * a)
{
    double out = tt_dot(a,a);
    //printf("out=%E\n",out);
    out = sqrt(fabs(out));
    return out;
}

