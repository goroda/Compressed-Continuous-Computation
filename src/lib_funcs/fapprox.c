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






/** \file fapprox.c
 * Provides basic routines for approximating functions
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "fapprox.h"

/***********************************************************//**
  Allocate one dimensional approximations
***************************************************************/
struct OneApproxOpts * 
one_approx_opts_alloc(enum function_class fc, void * aopts)
{

    struct OneApproxOpts * app = malloc(sizeof(struct OneApproxOpts));
    if (app == NULL){
        fprintf(stderr,"Cannot allocate OneApproxOpts\n");
        exit(1);
    }
    app->fc = fc;
    app->aopts = aopts;
    return app;
}

/***********************************************************//**
  Allocate one dimensional approximations
***************************************************************/
struct OneApproxOpts * 
one_approx_opts_alloc_ref(enum function_class fc, void ** aopts)
{

    struct OneApproxOpts * app = malloc(sizeof(struct OneApproxOpts));
    if (app == NULL){
        fprintf(stderr,"Cannot allocate OneApproxOpts\n");
        exit(1);
    }
    app->fc = fc;
    app->aopts = *aopts;
    return app;
}

/***********************************************************//**
  Free one dimensional approximations
***************************************************************/
void one_approx_opts_free(struct OneApproxOpts * oa)
{
    if (oa != NULL){
        free(oa); oa = NULL;
    }
}

/***********************************************************//**
  Free one dimensional approximations (deep)
***************************************************************/
void one_approx_opts_free_deep(struct OneApproxOpts ** oa)
{
    if (*oa != NULL){
        if ((*oa)->fc == PIECEWISE){
            struct PwPolyOpts * opts = (*oa)->aopts;
            pw_poly_opts_free(opts); 
            (*oa)->aopts = NULL;
        }
        else if ((*oa)->fc == POLYNOMIAL){
            struct OpeOpts * opts = (*oa)->aopts;
            ope_opts_free_deep(&opts); (*oa)->aopts = NULL;
            ope_opts_free(opts);
            (*oa)->aopts = NULL;
        }
        else if ((*oa)->fc == LINELM){
            struct LinElemExpAopts * opts = (*oa)->aopts;
            lin_elem_exp_aopts_free(opts);
            (*oa)->aopts = NULL;
        }
        else if ((*oa)->fc == CONSTELM){
            struct ConstElemExpAopts * opts = (*oa)->aopts;
            const_elem_exp_aopts_free(opts);
            (*oa)->aopts = NULL;
        }        
        else if ((*oa)->fc == KERNEL){
            struct KernelApproxOpts * opts = (*oa)->aopts;
            kernel_approx_opts_free(opts);
            (*oa)->aopts = NULL;
        }
        else{
            fprintf(stderr,"Cannot free one_approx options of type %d\n",
                    (*oa)->fc);
        }
        free(*oa); *oa = NULL;
    }
}


/***********************************************************//**
  Get the number of parametrs in the approximation optins
***************************************************************/
size_t one_approx_opts_get_nparams(const struct OneApproxOpts * oa)
{
    assert (oa != NULL);
    assert (oa->aopts != NULL);
    size_t nparams = 0;
    if (oa->fc == POLYNOMIAL){
        nparams = ope_opts_get_nparams(oa->aopts);
    }
    else if (oa->fc == PIECEWISE){
        nparams = pw_poly_opts_get_nparams(oa->aopts);
    }
    else if (oa->fc == LINELM){
        nparams = lin_elem_exp_aopts_get_nparams(oa->aopts);
    }
    else if (oa->fc == CONSTELM){
        nparams = const_elem_exp_aopts_get_nparams(oa->aopts);
    }    
    else if (oa->fc == KERNEL){
        nparams = kernel_approx_opts_get_nparams(oa->aopts);
    }
    else{
        fprintf(stderr,"Cannot get number of parameters for one_approx options of type %d\n",
                oa->fc);
    }
    return nparams;
}

/***********************************************************//**
  Get the lower bounds
***************************************************************/
double one_approx_opts_get_lb(const struct OneApproxOpts * oa)
{
    assert (oa != NULL);
    assert (oa->aopts != NULL);
    double lb = 0.0;
    if (oa->fc == POLYNOMIAL){
        lb = ope_opts_get_lb(oa->aopts);
    }
    else if (oa->fc == PIECEWISE){
        lb = pw_poly_opts_get_lb(oa->aopts);
    }
    else if (oa->fc == LINELM){
        lb = lin_elem_exp_aopts_get_lb(oa->aopts);
    }
    else if (oa->fc == CONSTELM){
        lb = const_elem_exp_aopts_get_lb(oa->aopts);
    }    
    else if (oa->fc == KERNEL){
        lb = kernel_approx_opts_get_lb(oa->aopts);
    }
    else{
        fprintf(stderr,"Cannot get lower_bound for one_approx options of type %d\n",
                oa->fc);
    }
    return lb;
}

/***********************************************************//**
  Get the upper bounds
***************************************************************/
double one_approx_opts_get_ub(const struct OneApproxOpts * oa)
{
    assert (oa != NULL);
    assert (oa->aopts != NULL);
    double ub = 0.0;
    if (oa->fc == POLYNOMIAL){
        ub = ope_opts_get_ub(oa->aopts);
    }
    else if (oa->fc == PIECEWISE){
        ub = pw_poly_opts_get_ub(oa->aopts);
    }
    else if (oa->fc == LINELM){
        ub = lin_elem_exp_aopts_get_ub(oa->aopts);
    }
    else if (oa->fc == CONSTELM){
        ub = const_elem_exp_aopts_get_ub(oa->aopts);
    }    
    else if (oa->fc == KERNEL){
        ub = kernel_approx_opts_get_ub(oa->aopts);
    }
    else{
        fprintf(stderr,"Cannot get upper bound for one_approx options of type %d\n",
                oa->fc);
    }
    return ub;
}


/***********************************************************//**
  Set the number of parametrs in the approximation optins
***************************************************************/
void one_approx_opts_set_nparams(struct OneApproxOpts * oa, size_t num)
{
    assert (oa != NULL);
    assert (oa->aopts != NULL);
    if (oa->fc == POLYNOMIAL){
        ope_opts_set_nparams(oa->aopts,num);
    }
    else if (oa->fc == PIECEWISE){
        pw_poly_opts_set_nparams(oa->aopts,num);
    }
    else if (oa->fc == LINELM){
        lin_elem_exp_aopts_set_nparams(oa->aopts,num);
    }
    else if (oa->fc == CONSTELM){
        const_elem_exp_aopts_set_nparams(oa->aopts,num);
    }    
    else if (oa->fc == KERNEL){
        kernel_approx_opts_set_nparams(oa->aopts,num);
    }
    else{
        fprintf(stderr,"Cannot set number of parameters for one_approx options of type %d\n",
                oa->fc);
    }
}


/***********************************************************//**
  Check whether the unknowns are linearly related to the function output.
  Typically used for regression to extract particular structure
***************************************************************/
int one_approx_opts_linear_p(const struct OneApproxOpts * oa)
{
    assert (oa != NULL);
    assert (oa->aopts != NULL);
    int lin = 0;
    if (oa->fc == POLYNOMIAL){
        lin = 1;
    }
    else if (oa->fc == PIECEWISE){
        assert (1 == 0);
    }
    else if (oa->fc == LINELM){
        lin = 1;
    }
    else if (oa->fc == CONSTELM){
        lin = 1;
    }    
    else if (oa->fc == KERNEL){
        lin = kernel_approx_opts_linear_p(oa->aopts);
    }
    else{
        fprintf(stderr,"Cannot get number of parameters for one_approx options of type %d\n",
                oa->fc);
    }
    return lin;
}




//////////////////////////////////////////////////////
/** \struct MultiApproxOpts
 * \brief Stores approximation options for multiple one dimensional functions
 * \var MultiApproxOpts::dim
 * Number of functions
 * \var MultiApproxOpts::aopts
 * function approximation options
 */
struct MultiApproxOpts
{
    size_t dim;
    struct OneApproxOpts ** aopts;
};

/***********************************************************//**
  Allocate multi_approx_opts

  \param[in] dim - dimension
  
  \return approximation options
***************************************************************/
struct MultiApproxOpts * multi_approx_opts_alloc(size_t dim)
{
    struct MultiApproxOpts * fargs;
    if ( NULL == (fargs = malloc(sizeof(struct MultiApproxOpts)))){
        fprintf(stderr, "Cannot allocate space for MultiApproxOpts.\n");
        exit(1);
    }
    fargs->aopts = malloc(dim * sizeof(struct OneApproxOpts *));
    if (fargs->aopts == NULL){
        fprintf(stderr, "Cannot allocate MultiApproxOpts\n");
        exit(1);
    }
    fargs->dim = dim;
    for (size_t ii = 0; ii < dim; ii++){
        fargs->aopts[ii] = NULL;
    }

    return fargs;
}

/***********************************************************//**
    Free memory allocated to MultiApproxOpts (shallow)

    \param[in,out] fargs - function train approximation arguments
***************************************************************/
void multi_approx_opts_free(struct MultiApproxOpts * fargs)
{
    if (fargs != NULL){
        free(fargs->aopts); fargs->aopts = NULL;
        free(fargs); fargs = NULL;
    }
}
/***********************************************************//**
    Free memory allocated to MultiApproxOpts (deep)

    \param[in,out] fargs - function train approximation arguments
***************************************************************/
void multi_approx_opts_free_deep(struct MultiApproxOpts ** fargs)
{
    if (*fargs != NULL){
        for (size_t ii = 0; ii < (*fargs)->dim; ii++){
            printf("ii freeing deep %zu %d\n",ii, (*fargs)->aopts[ii] == NULL);
           /* one_approx_opts_free((*fargs)->aopts[ii]); */
            printf("here\n");
            /* one_approx_opts_free_deep(&((*fargs)->aopts[ii])); */

            /* one_approx_opts_free((*fargs)->aopts[ii]); */
            if ((*fargs)->aopts[ii] != NULL){
                free((*fargs)->aopts[ii]);
                ((*fargs)->aopts[ii]) = NULL;
            }
            printf("there\n");

            /*     free((*fargs)->aopts[ii]); */
            /*     (*fargs)->aopts[ii] = NULL; */
            /* } */
        }
        free((*fargs)->aopts); (*fargs)->aopts = NULL;
        free(*fargs); *fargs = NULL;
    }
}

/***********************************************************//**
    Set approximation options for a particular dimension
    \param[in,out] fargs - function train approximation arguments
    \param[in]     ind   - set approximation arguments for this dimension
    \param[in]     opts  - approximation arguments
***************************************************************/
void multi_approx_opts_set_dim(struct MultiApproxOpts * fargs,
                               size_t ind,
                               struct OneApproxOpts * opts)
{
    assert (fargs != NULL);
    assert (ind < fargs->dim);
    fargs->aopts[ind] = opts;
}

/***********************************************************//**
    Set (by reference) approximation options for a particular value
    \param[in,out] fargs - function train approximation arguments
    \param[in]     ind   - set approximation arguments for this dimension
    \param[in]     opts  - approximation arguments
***************************************************************/
void multi_approx_opts_set_dim_ref(struct MultiApproxOpts * fargs,
                                   size_t ind,
                                   struct OneApproxOpts ** opts)
{
    assert (fargs != NULL);
    assert (ind < fargs->dim);
    fargs->aopts[ind] = *opts;
}

/***********************************************************//**
    Create the arguments to give to use for approximation
    in the function train. Specifically, legendre polynomials
    for all dimensions

    \param[in,out] mopts - multidimensional options to update
    \param[in]     opts  - options with which to update

***************************************************************/
void
multi_approx_opts_set_all_same(struct MultiApproxOpts * mopts,
                               struct OneApproxOpts * opts)
{
    assert(mopts != NULL);
    for (size_t ii = 0; ii < mopts->dim; ii++){
        mopts->aopts[ii] = opts;
    }
}                               

/***********************************************************//**
    Extract the function class to use for the approximation of the
    *dim*-th dimensional functions 

    \param[in] fargs - function train approximation arguments
    \param[in] dim   - dimension to extract

    \return function_class of the approximation
***************************************************************/
enum function_class 
multi_approx_opts_get_fc(const struct MultiApproxOpts * fargs, size_t dim)
{
    return fargs->aopts[dim]->fc;
}

/***********************************************************//**
    Determine whether the unknowns for a particular dimension
    are linearly related to the output

    \param[in] fargs - function train approximation arguments
    \param[in] dim   - dimension to extract

    \return 1 if linear 0 if not linear
***************************************************************/
int multi_approx_opts_linear_p(const struct MultiApproxOpts * fargs, size_t dim)
{
    return one_approx_opts_linear_p(fargs->aopts[dim]);
}

/***********************************************************//**
    Extract the approximation arguments to use for the approximation of the
    *dim*-th dimensional functions 

    \param[in] fargs - function train approximation arguments
    \param[in] dim   - dimension to extract

    \return approximation arguments
***************************************************************/
void * multi_approx_opts_get_aopts(const struct MultiApproxOpts * fargs, 
                                   size_t dim)
{
    assert (fargs != NULL);
    return fargs->aopts[dim];
}

/***********************************************************//**
    Get number of parameters requested by the approximation
    
    \param[in] f   - multi approx structure
    \param[in] ind - which function to inquire about
***************************************************************/
size_t multi_approx_opts_get_dim_nparams(const struct MultiApproxOpts * f, size_t ind)
{
    assert (f != NULL);
    return one_approx_opts_get_nparams(f->aopts[ind]);
}

/***********************************************************//**
    Get the lower bounds of a particular dimension
    
    \param[in] f   - multi approx structure
    \param[in] ind - which function to inquire about
***************************************************************/
double multi_approx_opts_get_dim_lb(const struct MultiApproxOpts * f, size_t ind)
{
    assert (f != NULL);
    return one_approx_opts_get_lb(f->aopts[ind]);
}

/***********************************************************//**
    Get the upper bounds of a particular dimension
    
    \param[in] f   - multi approx structure
    \param[in] ind - which function to inquire about
***************************************************************/
double multi_approx_opts_get_dim_ub(const struct MultiApproxOpts * f, size_t ind)
{
    assert (f != NULL);
    return one_approx_opts_get_ub(f->aopts[ind]);
}

/***********************************************************//**
    Set the number of parameters 
    
    \param[in,out] f   - multi approx structure
    \param[in]     ind - which function to inquire about
    \param[in]     val - new number of parameters
***************************************************************/
void multi_approx_opts_set_dim_nparams(struct MultiApproxOpts * f, size_t ind, size_t val)
{
    assert (f != NULL);
    one_approx_opts_set_nparams(f->aopts[ind],val);
}


/***********************************************************//**
    Get the dimension
***************************************************************/
size_t multi_approx_opts_get_dim(const struct MultiApproxOpts * f)
{
    assert (f != NULL);
    return f->dim;
}

///////////////////////////////////////////////////////////////////

struct FiberOptArgs
{
    size_t dim;
    void ** opts;
};


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
   set optimization arguments to something
***************************************************************/
void fiber_opt_args_set_dim(struct FiberOptArgs * fopt, size_t ii, void * arg)
{
    assert (fopt != NULL);
    assert (ii < fopt->dim);
    fopt->opts[ii] = arg; 
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
    Get optimization arguments for a certain dimension

    \param[in,out] fopts - fiber optimization structure
    \param[in]     ind   - dimension
***************************************************************/
void * fiber_opt_args_get_opts(const struct FiberOptArgs * fopts, size_t ind)
{
    assert (fopts != NULL);
    assert (ind < fopts->dim);
    return fopts->opts[ind];
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


