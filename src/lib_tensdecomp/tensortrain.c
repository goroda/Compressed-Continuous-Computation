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
#include <stdio.h>
#include <math.h>

#include "array.h"
#include "tensortrain.h"

/********************************************************//**
    Function init_tt_alloc

    Purpose: Initialize and allocate all the cores of a tensor
             (everything that can be initialized without knowing
             what the ranks are)

    Parameters:
        - tta (IN)   - uninitialized and un-allocated tensor
        - dim (IN)   - dimension of tensor
        - nvals (IN) - number of elements in each direction of the tensor

    Returns: Nothing.
***********************************************************/
void init_tt_alloc(struct tt ** tta, size_t dim, const size_t * nvals)
{
    if (NULL == ( (*tta) = malloc(sizeof(struct tt)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    (*tta)->dim = dim;
    
    (*tta)->nvals = calloc_size_t(dim);
    memmove((*tta)->nvals, nvals, dim * sizeof(size_t));
    
    (*tta)->ranks = calloc_size_t(dim+1);
    (*tta)->ranks[0] = 1;

    if (NULL == ( (*tta)->cores = malloc(dim * sizeof(struct tensor *)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
}

/********************************************************//**
    Function copy_tt

    Purpose: Copy a tensor traind decomposition

    Parameters:
        - a (IN)   - tensortrain to copy

    Returns: b - copied tensor train
***********************************************************/
struct tt * copy_tt(const struct tt * a){
    
    struct tt * b;
    init_tt_alloc(&b, a->dim, a->nvals);
    memmove(b->ranks, a->ranks, (a->dim+1) * sizeof(size_t));
    size_t ii = 0;
    for (ii = 0; ii < a->dim; ii++){
        b->cores[ii] = tensor_copy_3d(a->cores[ii],0);
    }
    return b;
}


/********************************************************//**
    Function freett

    Purpose: free memory allocated to a TT decomp

    Parameters:
        - ttc - tensor train decomp

    Returns: Nothing. void function.
***********************************************************/
void freett(struct tt * ttc){
    
    free(ttc->nvals);
    free(ttc->ranks);
    size_t ii;
    for (ii = 0; ii < ttc->dim; ii++){
        free_tensor(&(ttc->cores[ii]));
    }
    free(ttc->cores);
    free(ttc);
}

/********************************************************//**
    Function tt_ones

    Purpose: create a TT of ones

    Parameters:
        - dim (in) - dimension of tensor
        - nvals (in) - number of elements in each dimension

    Returns: mytt - TT 
***********************************************************/
struct tt *  tt_ones(size_t dim, const size_t * nvals) 
{
    struct tt * mytt;
    init_tt_alloc(&mytt, dim, nvals);
    size_t ii;
    for (ii = 0; ii < dim; ii++) {
        mytt->ranks[ii] = 1;
        mytt->cores[ii] = tensor_ones_3d(1,nvals[ii],1);
    }
    mytt->ranks[dim] = 1;
    return mytt;
}

struct tt * 
tt_x(const double * x, size_t dimx, size_t dim, size_t * nvals) 
{   
    size_t ii;
    struct tt * new;
    init_tt_alloc(&new, dim, nvals);
    // before dimx
    for (ii = 0; ii < dimx; ii++){
        new->ranks[ii] = 1;
        new->cores[ii] = tensor_ones_3d(1,1,nvals[ii]);
    }
    // dim x
    new->ranks[dimx] = 1;
    new->cores[dimx] = tensor_x_3d(nvals[dimx], x);
    
    // after dimx
    for (ii = dimx+1; ii < dim; ii++){
        new->ranks[ii] = 1;
        new->cores[ii] = tensor_ones_3d(1,1,nvals[ii]);
    }
    new->ranks[dim] = 1;
    return new;
}

double ttelem(struct tt * ttc, size_t * elem)
{
    double out = 1.0;
    size_t ii;
    double * vec = calloc_double(ttc->ranks[2]);
    cblas_dgemv(CblasColMajor, CblasTrans, ttc->ranks[1], ttc->ranks[2],
                1.0, 
                ttc->cores[1]->vals + elem[1] * ttc->ranks[1],
                ttc->ranks[1] * ttc->nvals[1],
                ttc->cores[0]->vals + elem[0] * ttc->ranks[0],
                ttc->ranks[0] * ttc->nvals[0],
                0.0,
                vec, 1);
    double * temp;
    for (ii = 1; ii < ttc->dim-1; ii++){
        temp = calloc_double(ttc->ranks[ii+2]);
        cblas_dgemv(CblasColMajor, CblasTrans, ttc->ranks[ii+1], 
                ttc->ranks[ii+2], 1.0, 
                ttc->cores[ii+1]->vals + elem[ii+1] * ttc->ranks[ii+1],
                ttc->ranks[ii+1] * ttc->nvals[1],
                vec, 1, 0.0, temp, 1);

        free(vec);
        vec = calloc_double(ttc->ranks[ii+2]);
        memmove(vec,temp, ttc->ranks[ii+2] * sizeof(double));
        free(temp);
    }
    
    out = vec[0];
    free(vec);
    return out;
}

/********************************************************//**
    Function full_to_tt

    Purpose: Compress a tensor into its TT format
             (TT-SVD Algorithm from Oseledets 2010)

    Parameters:
        - a (IN)   - tensor to compress
        - epsilon (IN)   - Relative Error of compression (Fro norm)

    Returns: tta - tensor in TT format
***********************************************************/
struct tt * full_to_tt(const struct tensor * a, double epsilon)
{
    
    size_t ii;
    size_t nElements = iprod_sz(a->dim, a->nvals);
    double nrm = norm2(a->vals, nElements);
    double delta = nrm * epsilon;
    if (a->dim > 1) delta = delta / sqrt(a->dim-1.0);
    size_t nl = a->nvals[0];
    size_t nr = nElements/nl;
    
    size_t nvalsc[3];

    struct tt * tta;
    init_tt_alloc(&tta, a->dim, a->nvals);
    tta->ranks[0] = 1;

    double * B = calloc_double(nElements);
    memmove(B, a->vals, nElements * sizeof(double));
    double * u; 
    double * s; 
    double * vt; 
    //printf("On first core nl=%d, nr=%d nElements=%d, \n",nl,nr, nElements);
    tta->ranks[1] = truncated_svd(nl, nr, nl, B, &u, &s, &vt, delta);
    free(B);

    //printf("Rank1=%d\n",tta->ranks[1]);
    //printf("first s:"); dprint(tta->ranks[1], s);

    //printf("Done with truncated_svd \n");
    nvalsc[0] = 1;
    nvalsc[1] = tta->nvals[0];
    nvalsc[2] = tta->ranks[1];
    init_tensor(&(tta->cores[0]),3,nvalsc);
    memmove(tta->cores[0]->vals,u, nl * tta->ranks[1] * sizeof(double));

    //printf("first s:"); dprint(tta->ranks[1], s);
    struct mat * sdiag = diagv2m(tta->ranks[1], s);
    //printmat(sdiag);
    //printf("computed spdiag \n");
    B = calloc_double(tta->ranks[1] * nr);
    //printf("Reallocated B \n");
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, tta->ranks[1], nr, 
                tta->ranks[1], 1.0, sdiag->vals, tta->ranks[1], vt, 
                tta->ranks[1], 0.0, B, tta->ranks[1]);

    //cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, 
    //            CblasNonUnit, tta->ranks[1], nr, 1.0, 
    //            sdiag->vals, tta->ranks[ii],  vt, tta->ranks[1]);

    freemat(sdiag);
    free(u);
    free(s);
    free(vt);
    //printf("Start loop \n");
    for (ii = 1; ii < tta->dim-1; ii++){
        nl = tta->nvals[ii];
        nr = nr/nl;
        //printf("On core=%d, nl=%d, nr=%d \n",ii, nl,nr);
        tta->ranks[ii+1] = truncated_svd(nl*tta->ranks[ii], nr, nl*tta->ranks[ii], B,
                                         &u, &s, &vt, delta); 
        //printf("Rank%d=%d\n",ii+1,tta->ranks[ii+1]);
        //printf("singular values "); dprint(tta->ranks[ii+1], s);
        free(B);
        
        nvalsc[0] = tta->ranks[ii];
        nvalsc[1] = nl;
        nvalsc[2] = tta->ranks[ii+1];
        init_tensor(&(tta->cores[ii]),3,nvalsc);

        memmove(tta->cores[ii]->vals, u, tta->ranks[ii] * nl * 
                        tta->ranks[ii+1] * sizeof(double));

        //printf("finished  nvals loop \n");
        sdiag = diagv2m(tta->ranks[ii+1], s);
        B = calloc_double(tta->ranks[ii+1] * nr);
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, tta->ranks[ii+1], nr, 
                    tta->ranks[ii+1], 1.0, sdiag->vals, tta->ranks[ii+1], vt, 
                    tta->ranks[ii+1], 0.0, B, tta->ranks[ii+1]);
        freemat(sdiag);
        free(u);
        free(s);
        free(vt);
    }
    
    // last core
    tta->ranks[tta->dim]=1;
    nvalsc[0] = tta->ranks[tta->dim-1];
    nvalsc[1] = tta->nvals[tta->dim-1];
    nvalsc[2] = tta->ranks[tta->dim];
    init_tensor(&(tta->cores[tta->dim-1]),3,nvalsc);
    memmove(tta->cores[tta->dim-1]->vals, B, tta->ranks[tta->dim-1] *
            nvalsc[1] * sizeof(double));
    free(B);

    return tta;
}

/********************************************************//**
    Function right_ortho_core

    Purpose: right orthogonalize a core and carry over the remainedr
            to the neighboring core

    Parameters:
        - a (IN/OUT) - tensor train
        - core(IN) - which core (>0) to orthogonalize

    Returns: Nothing. void function
***********************************************************/
void right_ortho_core(struct tt ** a, size_t core){
    
    size_t jj;
    double * r = calloc_double( (*a)->ranks[core] * (*a)->ranks[core] );
    for (jj = 0; jj < (*a)->ranks[core]; jj++){
        r[jj * (*a)->ranks[core] + jj] = 1.0;
    }

    rq_with_rmult((*a)->ranks[core], (*a)->ranks[core+1] * (*a)->nvals[core], 
                        (*a)->cores[core]->vals, (*a)->ranks[core], 
                  (*a)->ranks[core] , (*a)->ranks[core], r , (*a)->ranks[core] );

    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper,CblasNoTrans, CblasNonUnit, 
                (*a)->ranks[core-1] * (*a)->nvals[core-1], (*a)->ranks[core],
                1.0, r, (*a)->ranks[core],
                (*a)->cores[core-1]->vals, 
                (*a)->ranks[core-1] * (*a)->nvals[core-1]);
    free(r);
}

/********************************************************//**
    Function right_left_orth

    Purpose: right orthogonalize cores 1..dim-1 of a TT

    Parameters:
        - a (IN/OUT) - tensor train

    Returns: Nothing. void function
***********************************************************/
void right_left_orth(struct tt ** a){
    size_t ii;
    for (ii = 0; ii < (*a)->dim-1; ii++) {
        right_ortho_core(a,(*a)->dim-ii-1);    
    }
}



