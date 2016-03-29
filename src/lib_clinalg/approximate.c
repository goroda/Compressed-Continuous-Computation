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

#include "array.h"
#include "algs.h"
#include "lib_funcs.h"

enum C3ATYPE { CROSS, REGRESS };
struct C3Approx
{
    size_t dim;
    struct BoundingBox * bds;
        
    enum C3ATYPE type;

    enum poly_type ptype;
    // approximation stuff
    struct OpeAdaptOpts * aopts;
    struct FtApproxArgs * fapp;

    // optimization stuff
    struct c3Vector * optnodes;
    struct FiberOptArgs * fopt;

    // cross approximation stuff
    double ** start;
    struct FtCrossArgs * fca;
    
};

/***********************************************************//**
    Create an approximation structure

    \param[in] type - so far must be CROSS
    \param[in] dim  - number of dimensions
    \param[in] lb   - lower bounds of each dimension
    \param[in] ub   - upper bounds of each dimension
    
    \return approximation structure
***************************************************************/
struct C3Approx * c3approx_create(enum C3ATYPE type, size_t dim, double * lb, double * ub)
{
    
    struct C3Approx * c3a = malloc(sizeof(struct C3Approx));
    if (c3a == NULL){
        fprintf(stderr, "Cannot allocate c3 approximation C3Approx structure\n");
        exit(1);
    }

    c3a->type = type;
    c3a->ptype = LEGENDRE;
    c3a->dim = dim;
    c3a->bds = bounding_box_vec(dim,lb,ub);
    c3a->aopts = NULL;
    c3a->fapp = NULL;

    c3a->optnodes = NULL;
    c3a->fopt = NULL;

    c3a->start = NULL;
    c3a->fca = NULL;

    return c3a;
}

/***********************************************************//**
    Free memory allocated to approximation structure
***************************************************************/
void c3approx_destroy(struct C3Approx * c3a)
{
    if (c3a != NULL){
        bounding_box_free(c3a->bds); c3a->bds = NULL;
        ope_adapt_opts_free(c3a->aopts); c3a->aopts = NULL;
        ft_approx_args_free(c3a->fapp); c3a->fapp = NULL;
        c3vector_free(c3a->optnodes); c3a->optnodes = NULL;
        fiber_opt_args_free(c3a->fopt); c3a->fopt = NULL;
        free_dd(c3a->dim,c3a->start); c3a->start = NULL;
        ft_cross_args_free(c3a->fca); c3a->fca = NULL;
        free(c3a); c3a = NULL;
    }
}

/***********************************************************//**
    Initialize a polynomial based approximation

    \param[in,out] c3a   - approx structure
    \param[in]     ptype - poly type

    \note
    Initialize adaptation and approximation arguments
    if ptype == HERMITE then it also initializes optimization of
    fibers to optimize over a set of discrete nodes
***************************************************************/
void c3approx_init_poly(struct C3Approx * c3a, enum poly_type ptype)
{

    if (c3a != NULL){
        c3a->ptype = ptype;
        c3a->aopts = ope_adapt_opts_alloc();
        ope_adapt_opts_set_start(c3a->aopts,5);
        ope_adapt_opts_set_maxnum(c3a->aopts,30);
        ope_adapt_opts_set_coeffs_check(c3a->aopts,2);
        ope_adapt_opts_set_tol(c3a->aopts,1e-10);
        c3a->fapp = ft_approx_args_createpoly(c3a->dim,&(c3a->ptype),c3a->aopts);

        if (ptype == HERMITE)
        {
            size_t N = 100;
            double * x = linspace(-10.0,10.0,N);
            c3a->optnodes = c3vector_alloc(N,x);
            c3a->fopt = fiber_opt_args_bf_same(c3a->dim,c3a->optnodes);
            free(x); x = NULL;
            if (c3a->fca != NULL){
                ft_cross_args_set_optargs(c3a->fca,c3a->fopt);
            }
        }
    }
}

/***********************************************************//**
    Initialize cross approximation arguments

    \param[in,out] c3a       - approx structure
    \param[in]     init_rank - starting rank of approximation
    \param[in]     verbose   - verbosity level from cross approximation
x***************************************************************/
void c3approx_init_cross(struct C3Approx * c3a, size_t init_rank, int verbose)
{
    if (c3a != NULL){
        c3a->fca = ft_cross_args_alloc(c3a->dim,init_rank);
        ft_cross_args_set_verbose(c3a->fca,verbose);
        ft_cross_args_set_optargs(c3a->fca,c3a->fopt);
        ft_cross_args_set_round_tol(c3a->fca,1e-14);
        if (c3a->ptype == HERMITE){
            c3a->start = malloc_dd(c3a->dim);
            double lbs = -2.0;
            double ubs = 2.0;
            c3a->start[0] = calloc_double(init_rank); 
            for (size_t ii = 0; ii < c3a->dim-1; ii++){
                c3a->start[ii+1] = linspace(lbs,ubs,init_rank);
            }
        }
    }
}

/***********************************************************//**
    Get polynomial type
***************************************************************/
enum poly_type c3approx_get_ptype(const struct C3Approx * c3a)
{
    assert (c3a != NULL);
    return c3a->ptype;
}

/***********************************************************//**
    Get dimension
***************************************************************/
size_t c3approx_get_dim(const struct C3Approx * c3a)
{
    assert (c3a != NULL);
    return c3a->dim;
}

/***********************************************************//**
    Get Bounds
***************************************************************/
struct BoundingBox * c3approx_get_bds(const struct C3Approx * c3a)
{
    assert (c3a != NULL);
    return c3a->bds;
}

/***********************************************************//**
    Set rounding tolerance
***************************************************************/
void c3approx_set_round_tol(struct C3Approx * c3a, double epsround)
{
    assert (c3a != NULL);
    if (c3a->fca == NULL){
        fprintf(stderr,"Must cal c3approx_init_cross before setting epsround\n");
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
    Set nstart for polynomial adaptation
***************************************************************/
void c3approx_set_poly_adapt_nstart(struct C3Approx * c3a, size_t n)
{
    assert (c3a != NULL);
    if (c3a->aopts == NULL){
        fprintf(stderr,"Must run c3approx_init_poly before changing adaptation arguments\n");
        exit(1);
    }
    ope_adapt_opts_set_start(c3a->aopts,n);
}

/***********************************************************//**
    Set maximum number of basis functions for polynomial adaptation
***************************************************************/
void c3approx_set_poly_adapt_nmax(struct C3Approx * c3a, size_t n)
{
    assert (c3a != NULL);
    if (c3a->aopts == NULL){
        fprintf(stderr,"Must run c3approx_init_poly before changing adaptation arguments\n");
        exit(1);
    }
    ope_adapt_opts_set_maxnum(c3a->aopts,n);
        ope_adapt_opts_set_coeffs_check(c3a->aopts,2);
        ope_adapt_opts_set_tol(c3a->aopts,1e-10);
}

/***********************************************************//**
    Set number of coefficients to check for poly adaptation
***************************************************************/
void c3approx_set_poly_adapt_ncheck(struct C3Approx * c3a, size_t n)
{
    assert (c3a != NULL);
    if (c3a->aopts == NULL){
        fprintf(stderr,"Must run c3approx_init_poly before changing adaptation arguments\n");
        exit(1);
    }
    ope_adapt_opts_set_coeffs_check(c3a->aopts,n);
}

/***********************************************************//**
    Set fiber adaptation tolerance for polynomial based adaptation
***************************************************************/
void c3approx_set_poly_adapt_tol(struct C3Approx * c3a, double tol)
{
    assert (c3a != NULL);
    if (c3a->aopts == NULL){
        fprintf(stderr,"Must run c3approx_init_poly before changing adaptation arguments\n");
        exit(1);
    }
    ope_adapt_opts_set_tol(c3a->aopts,tol);
}

/***********************************************************//**
    Perform cross approximation of a function
***************************************************************/
struct FunctionTrain *
c3approx_do_cross(struct C3Approx * c3a, double (*f)(double*,void*),void*arg)
{
    struct FunctionTrain * ft = NULL;
    ft = function_train_cross(f,arg,c3a->bds,c3a->start,c3a->fca,c3a->fapp);
    return ft;
}
