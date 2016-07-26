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

/** \file approximate.c
 * Provides interface for approximation
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#include "approximate.h"

struct C3Approx
{
    size_t dim;
    struct BoundingBox * bds;
        
    enum C3ATYPE type;
    
    //if a grid is necessary
    struct c3Vector * grid;

    // approximation stuff
    struct MultiApproxOpts * fapp;

    // optimization stuff
    struct FiberOptArgs * fopt;

    // cross approximation stuff
    struct CrossIndex ** isl;
    struct CrossIndex ** isr;
    struct FtCrossArgs * fca;

    // reference function train (for various uses)
    struct FunctionTrain * ftref;
};

/***********************************************************//**
    Create an approximation structure

    \param[in] type - so far must be CROSS
    \param[in] dim  - number of dimensions
    
    \return approximation structure
***************************************************************/
struct C3Approx * c3approx_create(enum C3ATYPE type, size_t dim)
{
    
    struct C3Approx * c3a = malloc(sizeof(struct C3Approx));
    if (c3a == NULL){
        fprintf(stderr, "Cannot allocate c3 approximation C3Approx structure\n");
        exit(1);
    }

    c3a->type = type;
    c3a->dim = dim;

    c3a->fapp = multi_approx_opts_alloc(dim);
    c3a->fopt = fiber_opt_args_init(dim);
    c3a->fca = NULL;

    c3a->isl = malloc(dim *sizeof(struct CrossIndex * ));
    if (c3a->isl == NULL){
        fprintf(stderr,"Failure allocating memory for C3Approx\n");
    }
    c3a->isr = malloc(dim *sizeof(struct CrossIndex * ));
    if (c3a->isr == NULL){
        fprintf(stderr,"Failure allocating memory for C3Approx\n");
    }

    c3a->ftref = NULL;
    return c3a;
}

/***********************************************************//**
    Free memory allocated to approximation structure
***************************************************************/
void c3approx_destroy(struct C3Approx * c3a)
{
    if (c3a != NULL){
        //multi_approx_opts_free_deep(&(c3a->fapp)); //c3a->fapp = NULL;
        multi_approx_opts_free(c3a->fapp); c3a->fapp = NULL; 
        free(c3a->fapp); c3a->fapp = NULL;
        fiber_opt_args_free(c3a->fopt); c3a->fopt = NULL;
        ft_cross_args_free(c3a->fca); c3a->fca = NULL;
        if (c3a->isl != NULL){
            for (size_t ii = 0; ii < c3a->dim; ii++){
                cross_index_free(c3a->isl[ii]); c3a->isl[ii] = NULL;
            }
            free(c3a->isl); c3a->isl = NULL;
        }
        if (c3a->isr != NULL){
            for (size_t ii = 0; ii < c3a->dim; ii++){
                cross_index_free(c3a->isr[ii]); c3a->isr[ii] = NULL;
            }
            free(c3a->isr); c3a->isr = NULL;
        }

        function_train_free(c3a->ftref); c3a->ftref = NULL;
        free(c3a); c3a = NULL;
    }
}

/***********************************************************//**
    Initialize a polynomial based approximation

    \param[in,out] c3a  - approx structure
    \param[in]     ii   - dimension to set
    \param[in]     opts - options

***************************************************************/
void c3approx_set_approx_opts_dim(struct C3Approx * c3a, size_t ii,
                                  struct OneApproxOpts * opts)
{
    assert (c3a != NULL);
    assert (c3a->fapp != NULL);
    assert (ii < c3a->dim);
    
    multi_approx_opts_set_dim(c3a->fapp,ii,opts);
    /* multi_approx_opts_set_dim_ref(c3a->fapp,ii,opts); */

}

/***********************************************************//**
    Set optimization options

    \param[in,out] c3a  - approx structure
    \param[in]     ii   - dimension to set
    \param[in]     opts - options
***************************************************************/
void c3approx_set_opt_opts_dim(struct C3Approx * c3a, size_t ii,
                               void * opts)
{
    
    assert (c3a != NULL);
    assert (c3a->fopt != NULL);
    assert (ii < c3a->dim);
    fiber_opt_args_set_dim(c3a->fopt,ii,opts);
}

/***********************************************************//**
    Initialize cross approximation arguments

    \param[in,out] c3a       - approx structure
    \param[in]     init_rank - starting rank of approximation
    \param[in]     verbose   - verbosity level from cross approximation
    \param[in]     start     - starting nodes 

    \note
    *starting nodes* are a d dimensional array of 1 dimensional points
    each array of 1 dimensional must have >= *init_rank* nodes
***************************************************************/
void c3approx_init_cross(struct C3Approx * c3a, size_t init_rank, int verbose,
                         double ** start)
{
    assert (c3a != NULL);
    if (c3a->fca != NULL){
        fprintf(stdout,"Initializing cross approximation and\n");
        fprintf(stdout," destroying previous options\n");
        ft_cross_args_free(c3a->fca); c3a->fca = NULL;
    }
    c3a->fca = ft_cross_args_alloc(c3a->dim,init_rank);
    ft_cross_args_set_verbose(c3a->fca,verbose);
    ft_cross_args_set_round_tol(c3a->fca,1e-14);
    
    size_t * ranks = ft_cross_args_get_ranks(c3a->fca);
    cross_index_array_initialize(c3a->dim,c3a->isr,0,1,ranks,(void**)start,sizeof(double));
    cross_index_array_initialize(c3a->dim,c3a->isl,0,0,ranks+1,(void**)start,sizeof(double));
    if (c3a->ftref != NULL){
        function_train_free(c3a->ftref); c3a->ftref = NULL;
    }
    c3a->ftref = function_train_constant(1.0,c3a->fapp);
    /* exit(1); */
     /* function_train_get_lb(c3a->ftref); */
}

/***********************************************************//**
    Set verbosity level
***************************************************************/
void c3approx_set_verbose(struct C3Approx * c3a, int verbose)
{
    assert (c3a != NULL);
    if (c3a->fca == NULL){
        fprintf(stderr,"Must call c3approx_init_cross before setting verbose\n");
        exit(1);
    }
    ft_cross_args_set_verbose(c3a->fca,verbose);
}

/***********************************************************//**
    Set rounding tolerance
***************************************************************/
void c3approx_set_round_tol(struct C3Approx * c3a, double epsround)
{
    assert (c3a != NULL);
    if (c3a->fca == NULL){
        fprintf(stderr,"Must call c3approx_init_cross before setting epsround\n");
        exit(1);
    }
    ft_cross_args_set_round_tol(c3a->fca,epsround);
}

/***********************************************************//**
    Set rounding tolerance
***************************************************************/
void c3approx_set_cross_tol(struct C3Approx * c3a, double tol)
{
    assert (c3a != NULL);
    if (c3a->fca == NULL){
        fprintf(stderr,"Must cal c3approx_init_cross before setting cross tolerance\n");
        exit(1);
    }
    ft_cross_args_set_cross_tol(c3a->fca,tol);
}

/***********************************************************//**
    Set kickrank
***************************************************************/
void c3approx_set_adapt_kickrank(struct C3Approx * c3a, size_t kickrank)
{
    assert (c3a != NULL);
    if (c3a->fca == NULL){
        fprintf(stderr,"Must call c3approx_init_cross before setting adapt_kickrank");
        exit(1);
    }
    ft_cross_args_set_kickrank(c3a->fca,kickrank);
}

/***********************************************************//**
    Set maxiter
***************************************************************/
void c3approx_set_cross_maxiter(struct C3Approx * c3a, size_t maxiter)
{
    assert (c3a != NULL);
    if (c3a->fca == NULL){
        fprintf(stderr,"Must call c3approx_init_cross before setting cross_maxiter");
        exit(1);
    }
    ft_cross_args_set_maxiter(c3a->fca,maxiter);
}

/***********************************************************//**
    Set maximum rank
***************************************************************/
void c3approx_set_adapt_maxrank_all(struct C3Approx * c3a, size_t maxrank)
{
    assert (c3a != NULL);
    if (c3a->fca == NULL){
        fprintf(stderr,"Must call c3approx_init_cross before setting adapt_maxiter");
        exit(1);
    }
    ft_cross_args_set_maxrank_all(c3a->fca,maxrank);
}

/***********************************************************//**
    Perform cross approximation of a function
***************************************************************/
struct FunctionTrain *
c3approx_do_cross(struct C3Approx * c3a, struct Fwrap * fw, int adapt)
{
    assert (c3a != NULL);
    assert (c3a->fca != NULL);
    assert (c3a->isl != NULL);
    assert (c3a->isr != NULL);
    assert (c3a->fapp != NULL);
    assert (c3a->fopt != NULL);
    assert (c3a->ftref != NULL);

    struct FunctionTrain * ft = NULL;
    if (adapt == 0){
        ft = ftapprox_cross(fw,c3a->fca,c3a->isl,c3a->isr,
                            c3a->fapp,c3a->fopt,c3a->ftref);
    }
    else{
        ft = ftapprox_cross_rankadapt(fw,c3a->fca,c3a->isl,c3a->isr,
                                      c3a->fapp,c3a->fopt,c3a->ftref);
    }
    return ft;
}

/////////////////////////////////////////////////////////////////////////


/***********************************************************//**
    Get approximation arguments
***************************************************************/
struct MultiApproxOpts * c3approx_get_approx_args(const struct C3Approx * c3a)
{
    assert (c3a != NULL);
    return c3a->fapp;
}

/***********************************************************//**
    Get dimension
***************************************************************/
size_t c3approx_get_dim(const struct C3Approx * c3a)
{
    assert (c3a != NULL);
    return c3a->dim;
}
