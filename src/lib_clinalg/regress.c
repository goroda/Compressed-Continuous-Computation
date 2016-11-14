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

/** \file dmrg.c
 * Provides routines for dmrg based algorithms for the FT
 */

#include <stdlib.h>
#include <assert.h>
#include <math.h>

/* #include "regress.h" */
#include "lib_linalg.h"


/** \struct RegressALS
 *  \brief Alternating least squares regression
 *  \var RegressALS::dim
 *  dimension of approximation
 *  \var RegressALS::N
 *  number of data points
 *  \var RegressALS::y
 *  evaluations
 *  \var RegressALS::x
 *  input locations
 *  \var RegressALS::k
 *  dimension to optimize over
 *  \var RegressALS::pre_eval
 *  evaluations prior to dimension k
 * \var RegressALS::post_eval
 *  evaluations post dimension k
 */
struct RegressALS
{
    size_t dim;
    
    size_t N; 
    const double * y;
    const double * x;

    size_t k;
    double * prev_eval;
    double * post_eval;

    /* double * pre_int; */
    /* double * post_int; */

};

/***********************************************************//**
    Allocate ALS regression
***************************************************************/
struct RegressALS * regress_als_alloc(size_t dim)
{
    struct RegressALS * als = malloc(sizeof(struct RegressALS));
    if (als == NULL){
        fprintf(stderr, "Failure to allocate RegressALS\n");
        return NULL;
    }
    
    als->dim = dim;
    
    als->y = NULL;
    als->x = NULL;

    als->prev_eval = NULL;
    als->post_eval = NULL;

    return als;
}

/***********************************************************//**
    Free ALS regression options
***************************************************************/
void regress_als_free(struct RegressALS * als)
{
    if (als != NULL){
        free(als->prev_eval); als->prev_eval  = NULL;
        free(als->post_eval); als->post_eval = NULL;
        free(als); als = NULL;
    }
}

/***********************************************************//**
    Add data to ALS
***************************************************************/
void regress_als_add_data(struct RegressALS * als, size_t N, const double * y, const double * x)
{
    assert (als != NULL);
    als->N = N;
    als->y = y;
    als->x = x;
}




/***********************************************************//**
    Specify that we are working on the last dimension

***************************************************************/
/* void regress_als_set_last_dim(struct RegressALS * als, struct FunctionTrain * ftrain) */
/* { */
/*     assert (als != NULL); */
/*     als->k = als->dim-1; */
    
/*     if (als->prev_eval == NULL){ // need to precompute everything */
        
/*     } */
/*     else{ */
        
/*     } */

/*     free(als->post_eval); post->eval = NULL; */
/* } */




