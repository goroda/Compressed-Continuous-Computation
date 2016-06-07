// Copyright (c) 2014-2016, Massachusetts Institute of Technology
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

/** \file pivoting.c
 * Provides basic routines for pivoting, or storing (index,value) pairs for
 * arrays of generic functions
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

//#include "array.h"
#include "pivoting.h"

struct Pivot{
    
    size_t ind;
    void * loc;
    size_t n; // number of dimensions of location
    size_t elem_size; // size of the elements in loc
};

/***********************************************************//**
    Create a pivot

    \param[in] n         - dimension of pivot
    \param[in] elem_size - size of the elements of the pivot 
                           (e.g., sizeof(double))
    \return pivot structure
***************************************************************/
struct Pivot * pivot_create(size_t n, size_t elem_size)
{
    struct Pivot * piv = malloc(sizeof(struct Pivot));
    if (piv == NULL){
        fprintf(stderr,"Failure allocating pivot\n");
        exit(1);
    }
    
    piv->n = n;
    piv->elem_size = elem_size;
    piv->loc = malloc( n * elem_size);
    if (piv == NULL){
        fprintf(stderr,"Failure allocating pivot\n");
        exit(1);
    }

    return piv;
}

/***********************************************************//**
    Free pivot                                                            
***************************************************************/
void pivot_free(struct Pivot * piv)
{
    if (piv != NULL){
        free(piv->loc); piv->loc = NULL;
        free(piv); piv = NULL;
    }
}

/***********************************************************//**
    Get size of the element
***************************************************************/
size_t pivot_get_size(const struct Pivot * piv)
{
    assert (piv != NULL);
    return piv->elem_size;
}

/***********************************************************//**
    Get location
***************************************************************/
void * pivot_get_loc(const struct Pivot * piv)
{
    assert (piv != NULL);
    return piv->loc;
}

/***********************************************************//**
    Set index
***************************************************************/
void pivot_set_ind(struct Pivot * piv, size_t ind)
{
    assert (piv != NULL);
    piv->ind = ind;
}

struct PivotSet {
    size_t npivs;
    struct Pivot ** pivots;
};

/***********************************************************//**
    Create a set of pivots

    \param[in] n         - number of pivots
    \param[in] dim       - dimension of each p[ivot]
    \param[in] elem_size - size of the elements of the pivot 
                           (e.g., sizeof(double))
    \return allocate set of pivots
***************************************************************/
struct PivotSet * pivot_set_create(size_t n, size_t dim, size_t elem_size)
{
    struct PivotSet * piv = malloc(sizeof(struct PivotSet));
    if (piv == NULL){
        fprintf(stderr,"Failure allocating PivotSet\n");
        exit(1);
    }
    
    piv->npivs = n;
    piv->pivots = malloc(n * sizeof(struct PivotSet *));
    if (piv == NULL){
        fprintf(stderr,"Failure allocating PivotSet\n");
        exit(1);
    }
    for (size_t ii = 0; ii < n; ii++){
        piv->pivots[ii] = pivot_create(dim,elem_size);
    }
    
    return piv;

}

/***********************************************************//**
    Destroy a set of pivots
***************************************************************/
void pivot_set_destroy(struct PivotSet * piv)
{
    if (piv != NULL){
        for (size_t ii = 0; ii < piv->npivs; ii++){
            pivot_free(piv->pivots[ii]); piv->pivots[ii] = NULL;
        }
        free(piv->pivots); piv->pivots = NULL;
        free(piv); piv = NULL;
    }
}
