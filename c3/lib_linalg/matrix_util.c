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
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "array.h"
#include "matrix_util.h"


/***********************************************************//**
    Function v2m

    Purpose: Convert a double array to a matrix

    Parameters:
        - N (IN)     - Number of elements in array
        - arr (IN)   - array to convert to a matrix
        - asrow (IN) 
            * 1 to create a row matrix (1 x N)
            * 0 to create a column matrix (N x 1)

    Returns: either row or column Matrix Structure *mat*
***************************************************************/
struct mat * v2m(size_t nVals,double * arr,int asrow)
{
    struct mat * outmat;
    if (NULL == ( outmat = malloc(sizeof(struct mat)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    
    outmat->vals = calloc_double(nVals);
    memmove(outmat->vals, arr, nVals * sizeof(double));
    if (asrow == 0){
        outmat->nrows = nVals;
        outmat->ncols = 1;
    }
    else if (asrow == 1){
        outmat->nrows = 1;
        outmat->ncols = nVals;
    }
    return outmat;
}

/***********************************************************//**
    Function diagv2m

    Purpose: Convert a double array to a diagonal matrix

    Parameters:
        - N (IN)     - Number of elements in array
        - arr (IN)   - array to convert to a matrix

    Returns: diagonal matrix *mat*
***************************************************************/
struct mat * diagv2m(size_t nVals,double * arr)
{
    struct  mat * outmat;
    if (NULL == ( outmat = malloc(sizeof(struct mat)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    outmat->vals = calloc_double(nVals * nVals);
    outmat->nrows = nVals;
    outmat->ncols = nVals;
    size_t ii;
    for (ii = 0; ii < nVals; ii++){
        outmat->vals[ii*nVals+ii] = arr[ii];
    }
    return outmat;
}


/***********************************************************//**
    Function horizcat

    Purpose: Concatenate the columns of two matricies

    Parameters:
        - a (IN) - left matrix
        - b (IN) - right matrix

    Returns: horizontally stacked matrix *mat*
***************************************************************/
struct mat * horizcat(const struct mat * a, const struct mat * b)
{
    if (a->nrows != b->nrows) {
        printf("Should not be concatenating these matrices!!\n");
    }
    
    struct mat * m;
    if (NULL == ( m = malloc(sizeof(struct mat)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }

    m->nrows = a->nrows;
    m->ncols = a->ncols + b->ncols;
    
    m->vals = calloc_double(m->nrows * m->ncols);
    
    size_t siz = sizeof(double);
    size_t ii;
    for (ii = 0; ii < a->nrows; ii++){
        memmove(m->vals + ii * m->ncols, a->vals + ii * a->ncols, a->ncols * siz);
        memmove(m->vals + ii * m->ncols + a->ncols, b->vals + ii * b->ncols, b->ncols * siz);
    }
    return m;
}

/***********************************************************//**
    Function block_diag

    Purpose: Create a blog diagonal matrix

    Parameters:
        - mats (IN)  - array of mats 
        - nMats (IN) - number of matrices

    Returns: block diagonal matrix *m*
***************************************************************/
struct mat * block_diag(const struct mat ** mats, size_t nMats){
        
    struct mat * m;
    if (NULL == ( m = malloc(sizeof(struct mat)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    m->nrows = 0;
    m->ncols = 0;
    size_t ii;
    for (ii = 0; ii < nMats; ii++){
        m->nrows += mats[ii]->nrows;
        m->ncols += mats[ii]->ncols;
    }
    m->vals = calloc_double(m->nrows * m->ncols);

    size_t jj;
    size_t siz = sizeof(double);
    size_t row = 0;
    size_t cols = 0;
    for (ii = 0; ii < nMats; ii++){
        for (jj = 0; jj < mats[ii]->nrows; jj++){ 
            memmove(m->vals + row * m->ncols + cols, 
                    mats[ii]->vals + jj * mats[ii]->ncols, 
                    siz * mats[ii]->ncols);
            row += 1;
        }
        cols = cols + mats[ii]->ncols;
    }
    return m;
}

/***********************************************************//**
    Function freemat

    Purpose: free a matrix structure

    Parameters:
        - matin (IN)  - matrix to free

    Returns: Nothing
***************************************************************/
void freemat(struct mat * matin){
    free(matin->vals);
    free(matin);
}

/***********************************************************//**
    Function printmat

    Purpose: print a matrix to screen

    Parameters:
        - matin (IN)  - matrix to print

    Returns: Nothing
***************************************************************/
void printmat(struct mat * matin){
   
    size_t ii, jj;
    for (ii = 0; ii < matin->nrows; ii++){
        for (jj = 0; jj < matin->ncols; jj++){
            printf("%f ", matin->vals[ii*matin->ncols + jj]);
        }
        printf("\n");
    }
}

void mat_ones(struct mat * m, size_t nrows, size_t ncols)
{
    size_t ii, jj;
    m->nrows = nrows;
    m->ncols = ncols;
    m->vals = calloc_double(nrows*ncols);
    for (ii = 0; ii < nrows; ii++){
        for (jj = 0; jj < ncols; jj++){
            m->vals[ii*ncols + jj] = 1.0;
        }
    }
}


