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

/** \file elements.c
 * Provides routines for initializing / using elements of continuous linear algebra
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "stringmanip.h"
#include "array.h"
#include "linalg.h"
#include "elements.h"
#include "algs.h"

// helper functions for function_train_initsum2
struct wrap_spec_func
{
    double (*f)(double, size_t, void *);
    size_t which;
    void * args;
};
double eval_spec_func(double x, void * args)
{
    struct wrap_spec_func * spf = args;
    return spf->f(x,spf->which,spf->args);
}

/* /\**********************************************************\//\** */
/*     Extract a copy of the first nkeep columns of a qmarray */

/*     \param a [in] - qmarray from which to extract columns */
/*     \param nkeep [in] : number of columns to extract */

/*     \return qm - qmarray */
/* **************************************************************\/ */
/* struct Qmarray * qmarray_extract_ncols(struct Qmarray * a, size_t nkeep) */
/* { */
    
/*     struct Qmarray * qm = qmarray_alloc(a->nrows,nkeep); */
/*     size_t ii,jj; */
/*     for (ii = 0; ii < nkeep; ii++){ */
/*         for (jj = 0; jj < a->nrows; jj++){ */
/*             qm->funcs[ii*a->nrows+jj] = generic_function_copy(a->funcs[ii*a->nrows+jj]); */
/*         } */
/*     } */
/*     return qm; */
/* } */

/////////////////////////////////////////////////////////
// qm_array



////////////////////////////////////////////////////////////////////
// function_train 
//




struct BoundingBox * function_train_bds(struct FunctionTrain * ft)
{
    
    struct BoundingBox * bds = bounding_box_init_std(ft->dim);
    size_t ii;
    for (ii = 0; ii < ft->dim; ii++){
        bds->lb[ii] = generic_function_get_lower_bound(ft->cores[ii]->funcs[0]);
        bds->ub[ii] = generic_function_get_upper_bound(ft->cores[ii]->funcs[0]);
    }

    return bds;
}   




/***********************************************************//**
  Allocate fiber optimization options
***************************************************************/
struct FiberOptArgs * fiber_opt_args_alloc()
{
    struct FiberOptArgs * fopt = NULL;
    fopt = malloc(sizeof(struct FiberOptArgs));
    if (fopt == NULL){
        fprintf(stderr,"Memory failure allocating FiberOptArgs\n");
        exit(1);
    }
    return fopt;
}

/***********************************************************//**
    Initialize a baseline optimization arguments class

    \param[in] dim - dimension of problem

    \return fiber optimzation arguments that are NULL in each dimension
***************************************************************/
struct FiberOptArgs * fiber_opt_args_init(size_t dim)
{
    struct FiberOptArgs * fopt = fiber_opt_args_alloc();
    fopt->dim = dim;
    
    fopt->opts = malloc(dim * sizeof(void *));
    if (fopt->opts == NULL){
        fprintf(stderr,"Memory failure initializing fiber opt args\n");
        exit(1);
    }
    for (size_t ii = 0; ii < dim; ii++){
        fopt->opts[ii] = NULL;
    }
    return fopt;
}

/***********************************************************//**
    Initialize a bruteforce optimization with the same nodes 
    in each dimension

    \param[in] dim   - dimension of problem
    \param[in] nodes - nodes over which to optimize 
                       (same ones used for each dimension)

    \return fiber opt args
***************************************************************/
struct FiberOptArgs * 
fiber_opt_args_bf_same(size_t dim, struct c3Vector * nodes)
{
    struct FiberOptArgs * fopt = fiber_opt_args_alloc();
    fopt->dim = dim;
    
    fopt->opts = malloc(dim * sizeof(void *));
    if (fopt->opts == NULL){
        fprintf(stderr,"Memory failure initializing fiber opt args\n");
        exit(1);
    }
    for (size_t ii = 0; ii < dim; ii++){
        fopt->opts[ii] = nodes;
    }
    return fopt;
}

/***********************************************************//**
    Initialize a bruteforce optimization with different nodes
    in each dimension

    \param[in] dim   - dimension of problem
    \param[in] nodes - nodes over which to optimize 
                       (same ones used for each dimension)

    \return fiber opt args
***************************************************************/
struct FiberOptArgs * 
fiber_opt_args_bf(size_t dim, struct c3Vector ** nodes)
{
    struct FiberOptArgs * fopt = fiber_opt_args_alloc();
    fopt->dim = dim;
    
    fopt->opts = malloc(dim * sizeof(void *));
    if (fopt->opts == NULL){
        fprintf(stderr,"Memory failure initializing fiber opt args\n");
        exit(1);
    }
    for (size_t ii = 0; ii < dim; ii++){
        fopt->opts[ii] = nodes[ii];
    }
    return fopt;
}

/***********************************************************//**
    Free memory allocate to fiber optimization arguments

    \param[in,out] fopt - fiber optimization structure
***************************************************************/
void fiber_opt_args_free(struct FiberOptArgs * fopt)
{
    if (fopt != NULL){
        free(fopt->opts); fopt->opts = NULL;
        free(fopt); fopt = NULL;
    }
}


/////////////////////////////////////////////////////////
// Utilities
//


void print_qmarray(struct Qmarray * qm, size_t prec, void * args)
{

    printf("Quasimatrix Array (%zu,%zu)\n",qm->nrows, qm->ncols);
    printf("=========================================\n");
    size_t ii,jj;
    for (ii = 0; ii < qm->nrows; ii++){
        for (jj = 0; jj < qm->ncols; jj++){
            printf("(%zu, %zu)\n",ii,jj);
            print_generic_function(qm->funcs[jj*qm->nrows+ ii],prec,args);
            printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
        }
    }
}

