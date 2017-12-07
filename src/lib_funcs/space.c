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






/** \file space.c
 * Provides basic routines for interfacing specific functions to the outside world through
 * generic functions
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "array.h"

/** \struct BoundingBox
 * \brief An array of pairs of lower and upper bounds
 * \var BoundingBox::dim
 * dimension of box
 * \var BoundingBox::lb
 * lower bounds
 * \var BoundingBox::ub
 * upper bounds
 */
struct BoundingBox
{
    size_t dim;
    double * lb;
    double * ub;
};

/*******************************************************//**
    Initialize a bounds tructure with each dimension bounded by [-1,1]

    \param[in] dim - dimension
        
    \return b - bounds
***********************************************************/
struct BoundingBox * bounding_box_init_std(size_t dim)
{
    struct BoundingBox * b;
    if (NULL == ( b = malloc(sizeof(struct BoundingBox)))){
        fprintf(stderr, "failed to allocate bounds.\n");
        exit(1);
    }
    
    b->dim = dim;
    b->lb = calloc_double(dim);
    b->ub = calloc_double(dim);
    size_t ii;
    for (ii = 0; ii <dim; ii++){
        b->lb[ii] = -1.0;
        b->ub[ii] = 1.0;
    }
    return b;
}

/*******************************************************//**
    Initialize a bound structure with each dimension bounded by [lb,ub]

    \param[in] dim - dimension
    \param[in] lb  - lower bounds
    \param[in] ub  - upper bounds
        
    \return bounds
***********************************************************/
struct BoundingBox * bounding_box_init(size_t dim, double lb, double ub)
{
    struct BoundingBox * b;
    if (NULL == ( b = malloc(sizeof(struct BoundingBox)))){
        fprintf(stderr, "failed to allocate bounds.\n");
        exit(1);
    }
    
    b->dim = dim;
    b->lb = calloc_double(dim);
    b->ub = calloc_double(dim);
    size_t ii;
    for (ii = 0; ii < dim; ii++){
        b->lb[ii] = lb;
        b->ub[ii] = ub;
    }
    return b;
}

/*******************************************************//**
    Initialize a bound structure with each dimension bounded by [lb[i],ub[i]]

    \param[in] dim - dimension
    \param[in] lb  - lower bounds
    \param[in] ub  - upper bounds
        
    \return  bounds
***********************************************************/
struct BoundingBox * bounding_box_vec(size_t dim, const double * lb, const double *ub)
{
    struct BoundingBox * b;
    if (NULL == ( b = malloc(sizeof(struct BoundingBox)))){
        fprintf(stderr, "failed to allocate bounds.\n");
        exit(1);
    }
    
    b->dim = dim;
    b->lb = calloc_double(dim);
    b->ub = calloc_double(dim);
    size_t ii;
    for (ii = 0; ii <dim; ii++){
        b->lb[ii] = lb[ii];
        b->ub[ii] = ub[ii];
    }
    return b;
}

/********************************************************//**
    Free memory allocated for bounding box

    \param[in,out] b - bounds
************************************************************/
void bounding_box_free(struct BoundingBox * b)
{
    if (b != NULL){
        free(b->lb);
        free(b->ub);
        free(b);
    }
}

/********************************************************//**
    Get bounding box dimension
************************************************************/
size_t bounding_box_get_dim(const struct BoundingBox * b)
{
    assert (b != NULL);
    return b->dim;
}

/********************************************************//**
    Return a reference to the lower bounds
************************************************************/
double * bounding_box_get_lb(const struct BoundingBox * b)
{
    assert (b != NULL);
    return b->lb;
}

/********************************************************//**
    Return a reference to the upper bounds
************************************************************/
double * bounding_box_get_ub(const struct BoundingBox * b)
{
    assert (b != NULL);
    return b->ub;
}

/********************************************************//**
    Return the upper bound of some dimension
************************************************************/
double bounding_box_get_ub_dim(const struct BoundingBox * b, size_t ind)
{
    assert (b != NULL);
    return b->ub[ind];
}

/********************************************************//**
    Set the upper bound of some dimension
************************************************************/
void bounding_box_set_ub_dim(struct BoundingBox * b, size_t ind, double val)
{
    assert (b != NULL);
    b->ub[ind] = val;
}


/********************************************************//**
    Return the upper bound of some dimension
************************************************************/
double bounding_box_get_lb_dim(const struct BoundingBox * b, size_t ind)
{
    assert (b != NULL);
    return b->lb[ind];
}

/********************************************************//**
    Set the upper bound of some dimension
************************************************************/
void bounding_box_set_lb_dim(struct BoundingBox * b, size_t ind, double val)
{
    assert (b != NULL);
    b->lb[ind] = val;
}



