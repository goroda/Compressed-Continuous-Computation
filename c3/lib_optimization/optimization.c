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


/** \file optimization.c
 * Provides routines for optimization
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>

#include "array.h"
#include "linalg.h"
#include "optimization.h"

// Line search parameters
/// @private
struct c3LS
{
    enum c3opt_ls_alg alg;
    
    double alpha;
    double beta;
    size_t maxiter;

    int set_initial; // (0 for (quasi) newton methods, 1 otherwise)
};

/***********************************************************//**
    Allocate line-search parameters
***************************************************************/
struct c3LS * c3ls_alloc(enum c3opt_ls_alg alg,int set_initial)
{
    struct c3LS * ls = malloc(sizeof(struct c3LS));
    if (ls == NULL){
        fprintf(stderr,"Error allocating line search params\n");
        exit(1);
    }

    ls->alg = alg;
    if (alg == BACKTRACK){
        ls->alpha = 0.4;
        ls->beta = 0.9;
    }
    else if (alg == STRONGWOLFE){

        ls->alpha = 0.0001;
        ls->beta = 0.9;
    }
    else if (alg == WEAKWOLFE){
        ls->alpha = 0.3;
        ls->beta = 0.9;
    }
    else{
        printf("Line search algorithm, %d, is not recognized\n",alg);
        exit(1);
    }
        
    ls->maxiter = 500;
    ls->set_initial = set_initial;
    
    return ls;
}

/***********************************************************//**
    Copy ls
***************************************************************/
struct c3LS * c3ls_copy(struct c3LS * old)
{
    struct c3LS * new = c3ls_alloc(old->alg,old->set_initial);
    new->alpha   = old->alpha;
    new->beta    = old->beta;
    new->maxiter = old->maxiter;

    return new;
}

/***********************************************************//**
    Free line search parameters
***************************************************************/
void c3ls_free(struct c3LS * ls)
{
    if (ls != NULL){
        free(ls); ls = NULL;
    }
}

int c3ls_get_initial(const struct c3LS * ls)
{
    assert (ls != NULL);
    return ls->set_initial;
}

void c3ls_set_alg(struct c3LS * ls, enum c3opt_ls_alg alg)
{
    assert (ls != NULL);
    ls->alg = alg;
}

enum c3opt_ls_alg c3ls_get_alg(const struct c3LS * ls)
{
    assert (ls != NULL);
    return ls->alg;
}

void c3ls_set_alpha(struct c3LS * ls, double alpha)
{
    assert (ls != NULL);
    ls->alpha = alpha;
}

double c3ls_get_alpha(struct c3LS * ls)
{
    assert (ls != NULL);
    return ls->alpha;
}

void c3ls_set_beta(struct c3LS * ls, double beta)
{
    assert (ls != NULL);
    ls->beta = beta;
}

double c3ls_get_beta(struct c3LS * ls)
{
    assert (ls != NULL);
    return ls->beta;
}

void c3ls_set_maxiter(struct c3LS * ls, size_t maxiter)
{
    assert (ls != NULL);
    ls->maxiter = maxiter;
}

size_t c3ls_get_maxiter(struct c3LS * ls)
{
    assert (ls != NULL);
    return ls->maxiter;
}
///////////////////////////////////////////////////////


struct c3SGD
{
    double validation_fraction;
    
    double learn_rate;
    double learn_rate_decay; // percentage decay, when decay needed
    
    size_t nsamples;

    int do_momentum;
    size_t nepochs_between_increase;
};

struct c3SGD * c3sgd_alloc()
{
    struct c3SGD * sgd = malloc(sizeof(struct c3SGD));
    if (sgd == NULL){
        fprintf(stderr,"Failure to allocate sgd\n");
        exit(1);
    }

    sgd->validation_fraction = 0.1;
    sgd->learn_rate = 1e-3;
    sgd->learn_rate_decay = 0.993;
    /* sgd->learn_rate_decay = 0.999; */

    sgd->nsamples = 0;

    sgd->do_momentum = 1;
    sgd->nepochs_between_increase = 10;

    return sgd;
}

struct c3SGD * c3sgd_copy(struct c3SGD * old)
{
    struct c3SGD * sgd = c3sgd_alloc();
    sgd->validation_fraction = old->validation_fraction;
    sgd->learn_rate = old->learn_rate;
    sgd->learn_rate_decay = old->learn_rate_decay;
    sgd->nsamples = old->nsamples;
    sgd->do_momentum = old->do_momentum;
    sgd->nepochs_between_increase = old->nepochs_between_increase;
    
    return sgd;
}

void c3sgd_free(struct c3SGD * sgd)
{
    if (sgd != NULL){
        free(sgd); sgd = NULL;
    }
}

///////////////////////////////////////////////////////
struct c3Opt
{
    enum c3opt_alg alg;
    size_t d;

    double (*f)(size_t, const double *, double *, void *);
    double (*fsgd)(size_t,size_t, const double *, double *, void *);
    void * farg;
    
    double * lb;
    double * ub;
    int grad;
    struct c3LS * ls;

    // for gradient based methods
    size_t maxiter;
    double relxtol;
    double absxtol;
    double relftol;
    double gtol;

    // for brute force
    size_t nlocs;

    double * workspace;
    int verbose;

    // statistics
    size_t nevals; // number of function evaluations
    size_t ngvals; // number of gradient evaluations
    size_t niters; // number of iterations

    // for different algorithmic details
    double prev_eval;

    // for lbfgs
    size_t nvectors_store;
    int init_scale;

    struct c3SGD * sgd;
    
    // storing traces
    int store_grad;
    int store_func;
    int store_x;
    double * stored_grad;
    double * stored_func;
    double * stored_x;
    

};


/***********************************************************//**
    Allocate optimization algorithm
***************************************************************/
struct c3Opt * c3opt_create(enum c3opt_alg alg)
{
    struct c3Opt * opt = malloc(sizeof(struct c3Opt));
    if (opt == NULL){
        fprintf(stderr,"Error allocating optimization\n");
        exit(1);
    }

    opt->alg = alg;
    opt->d = 0;

    opt->f = NULL;
    opt->farg = NULL;

    opt->lb = NULL;
    opt->ub = NULL;

    opt->verbose = 0;
    opt->maxiter = 1000;
    opt->relxtol = 1e-8;
    opt->absxtol = 1e-8;
    if (alg == SGD){
        opt->absxtol = 1e-20;
    }
    opt->relftol = 1e-8;
    opt->gtol = 1e-12;

    opt->sgd = NULL;
    if (alg == BFGS){
        opt->grad = 1;
        opt->workspace = NULL; 
        opt->ls = c3ls_alloc(WEAKWOLFE,0);
    }
    else if (alg == LBFGS){
        opt->grad = 1;
        opt->workspace = NULL;/* calloc_double(4*d); */
        /* opt->ls = c3ls_alloc(BACKTRACK,0); */
        /* opt->ls = c3ls_alloc(STRONGWOLFE,0); */
        opt->ls = c3ls_alloc(WEAKWOLFE,0);
    }
    else if (alg == SGD){
        opt->grad = 1;
        opt->workspace = NULL;
        opt->ls = NULL;
        opt->sgd = c3sgd_alloc();
    }
    else if (alg==BATCHGRAD){
        opt->grad = 1;
        opt->workspace = NULL;
        opt->ls = c3ls_alloc(STRONGWOLFE,0);
        /* opt->ls = c3ls_alloc(WEAKWOLFE,0); */
    }
    else{
        opt->nlocs = 0;
        opt->grad = 0;
        opt->ls = NULL;
        opt->workspace = NULL;        
    }
    
    opt->nevals = 0;
    opt->ngvals = 0;
    opt->niters = 0;

    opt->nvectors_store = 40;
    opt->init_scale = 0;
        
    opt->store_grad = 0;
    opt->store_func = 0;
    opt->store_x = 0;
    opt->stored_grad = NULL;
    opt->stored_func = NULL;
    opt->stored_x = NULL;

    return opt;
}

/***********************************************************//**
    Set the number of variables in the optimizer

    \note 
    This function overwrites any previously allocated memory
***************************************************************/
void c3opt_set_nvars(struct c3Opt * opt, size_t nvars)
{
    assert (opt != NULL);

    opt->d = nvars;
    
    free(opt->lb); opt->lb = NULL;
    free(opt->ub); opt->ub = NULL;
    
    opt->lb = calloc_double(nvars);
    opt->ub = calloc_double(nvars);
    for (size_t ii = 0; ii < nvars; ii++){
//        printf("lower bounds are small\n");
        opt->lb[ii] = -DBL_MAX;
        opt->ub[ii] = DBL_MAX;
    }

    free(opt->workspace);
    opt->workspace = NULL;
    if (opt->alg == BFGS){
        opt->grad = 1;
        opt->workspace = calloc_double(4*opt->d);
    }
    else if (opt->alg == LBFGS){
        opt->grad = 1;
        opt->workspace = calloc_double(4*opt->d);
    }
    else if (opt->alg == SGD){
        opt->grad = 1;
        opt->workspace = calloc_double(4*opt->d);
    }
    else if (opt->alg==BATCHGRAD){
        opt->grad = 1;
        opt->workspace = calloc_double(2*opt->d);
    }
    else{
        opt->nlocs = 0;
        opt->grad = 0;
        opt->ls = NULL;
        opt->workspace = NULL;        
    }

    opt->nevals = 0;
    opt->ngvals = 0;
    opt->niters = 0;

    free(opt->stored_grad); opt->stored_grad = NULL;
    free(opt->stored_func); opt->stored_func = NULL;
    free(opt->stored_x);    opt->stored_x = NULL;
    
    if (opt->store_grad == 1){
        opt->stored_grad = calloc_double(opt->d*sizeof(double)*opt->maxiter);
    }
    if (opt->store_x == 1){
        opt->stored_x = calloc_double(opt->d*sizeof(double)*opt->maxiter);
    }
    if (opt->store_func == 1){
        opt->stored_func = calloc_double(sizeof(double)*opt->maxiter);
    }

}

/***********************************************************//**
    Allocate optimization algorithm
    (Depracated!)
***************************************************************/
struct c3Opt * c3opt_alloc(enum c3opt_alg alg, size_t d)
{
    struct c3Opt * opt = malloc(sizeof(struct c3Opt));
    if (opt == NULL){
        fprintf(stderr,"Error allocating optimization\n");
        exit(1);
    }

    opt->alg = alg;
    opt->d = d;

    opt->f = NULL;
    opt->farg = NULL;
    
    opt->lb = calloc_double(d);
    opt->ub = calloc_double(d);
    for (size_t ii = 0; ii < d; ii++){
//        printf("lower bounds are small\n");
        opt->lb[ii] = -1e30;
        opt->ub[ii] = 1e30;
    }

    opt->verbose = 0;
    opt->maxiter = 50000;
    opt->relxtol = 1e-8;
    opt->absxtol = 1e-8;
    if (alg == SGD){
        opt->absxtol = 1e-20;
    }
    opt->relftol = 1e-8;
    opt->gtol = 1e-12;

    opt->sgd = NULL;
    if (alg == BFGS){
        opt->grad = 1;
        opt->workspace = calloc_double(4*d);
        /* opt->ls = c3ls_alloc(BACKTRACK,0); */
        /* opt->ls = c3ls_alloc(STRONGWOLFE,0); */
        opt->ls = c3ls_alloc(WEAKWOLFE,0);
    }
    else if (alg == LBFGS){
        opt->grad = 1;
        opt->workspace = calloc_double(4*d);
        /* opt->ls = c3ls_alloc(BACKTRACK,0); */
        /* opt->ls = c3ls_alloc(STRONGWOLFE,0); */
        opt->ls = c3ls_alloc(WEAKWOLFE,0);
    }
    else if (alg==BATCHGRAD){
        opt->grad = 1;
        opt->workspace = calloc_double(2*d);
        /* opt->ls = c3ls_alloc(BACKTRACK,1); */
        /* opt->ls = c3ls_alloc(STRONGWOLFE,0); */
        opt->ls = c3ls_alloc(WEAKWOLFE,0);
    }
    else if (alg == SGD){
        opt->grad = 1;
        opt->workspace = calloc_double(4*d);
        opt->ls = NULL;
        opt->sgd = c3sgd_alloc();
    }
    else{
        opt->nlocs = 0;
        opt->grad = 0;
        opt->ls = NULL;
        opt->workspace = NULL;        
    }
    
    opt->nevals = 0;
    opt->ngvals = 0;
    opt->niters = 0;

    opt->nvectors_store = 40;
    opt->init_scale = 0;
    
    opt->store_grad = 0;
    opt->store_func = 0;
    opt->store_x = 0;
    opt->stored_grad = NULL;
    opt->stored_func = NULL;
    opt->stored_x = NULL;

    
    return opt;
}

/**********************************************************//**
    Copy optimization struct
**************************************************************/
struct c3Opt * c3opt_copy(struct c3Opt * old)
{
    if (old == NULL){
        return NULL;
    }
    struct c3Opt * opt = c3opt_alloc(old->alg,old->d);
    opt->f = old->f;
    opt->farg = old->farg;
    memmove(opt->lb,old->lb,opt->d*sizeof(double));
    memmove(opt->ub,old->ub,opt->d*sizeof(double));
    opt->verbose = old->verbose;
    opt->maxiter = old->maxiter;
    opt->relxtol = old->relxtol;
    opt->absxtol = old->absxtol;
    opt->relftol = old->relftol;
    opt->gtol    = old->gtol;
    opt->nlocs   = old->nlocs;
    if ((opt->alg == BFGS) || (opt->alg == BATCHGRAD) || (opt->alg == LBFGS)){
        opt->grad = 1;
        memmove(opt->workspace,old->workspace,4*opt->d*sizeof(double));
        c3ls_free(opt->ls); opt->ls = NULL;
        opt->ls = c3ls_copy(old->ls);
    }
    else if (opt->alg == SGD){
        /* opt->workspace = calloc_double(opt->nlocs * opt->d); */
        memmove(opt->workspace,old->workspace,4*opt->d*sizeof(double));
        c3sgd_free(opt->sgd); opt->sgd = NULL;
        opt->sgd = c3sgd_copy(old->sgd);
    }
    else if (opt->alg == BRUTEFORCE)
    {
        c3opt_set_brute_force_vals(opt,old->nlocs,old->workspace);
    }

    opt->nevals = old->nevals;
    opt->ngvals = old->ngvals;
    opt->niters = old->niters;

    opt->prev_eval = old->prev_eval;
    opt->nvectors_store = old->nvectors_store;
    opt->init_scale = old->init_scale;

    opt->store_grad  = old->store_grad;
    opt->store_func = old->store_func;
    opt->store_x     = old->store_x;
    if (opt->store_grad == 1){
        opt->stored_grad = calloc_double(opt->d*sizeof(double)*opt->maxiter);
        memmove(opt->stored_grad,old->stored_grad,opt->d*sizeof(double)*opt->maxiter);
    }
    if (opt->store_x == 1){
        opt->stored_x = calloc_double(opt->d*sizeof(double)*opt->maxiter);
        memmove(opt->stored_x,old->stored_x,opt->d*sizeof(double)*opt->maxiter);
    }
    if (opt->store_func == 1){
        opt->stored_func = calloc_double(sizeof(double)*opt->maxiter);
        memmove(opt->stored_func,old->stored_func,sizeof(double)*opt->maxiter);
    }

    return opt;
}

/***********************************************************//**
    Free optimization struct
***************************************************************/
void c3opt_free(struct c3Opt * opt)
{
    if (opt != NULL){
        free(opt->lb); opt->lb = NULL;
        free(opt->ub); opt->ub = NULL;
        free(opt->workspace); opt->workspace = NULL;
        c3ls_free(opt->ls); opt->ls = NULL;
        c3sgd_free(opt->sgd); opt->sgd = NULL;
        free(opt->stored_grad); opt->stored_grad = NULL;
        free(opt->stored_func); opt->stored_func = NULL;
        free(opt->stored_x);    opt->stored_x    = NULL;
        free(opt); opt = NULL;
    }
}

/***********************************************************//**
    Set the number of vectors to store for lbfgs
***************************************************************/
void c3opt_set_nvectors_store(struct c3Opt * opt, size_t nvecs)
{
    assert (opt != NULL);
    opt->nvectors_store = nvecs;
}

/***********************************************************//**
    Get the number of vectors used for lbfgs
***************************************************************/
size_t c3opt_get_nvectors_store(const struct c3Opt * opt)
{
    assert (opt != NULL);
    return opt->nvectors_store;
}

/***********************************************************//**
    Set initial scaling option for LBFGS
    
    \param[in,out] opt - optimization options
    \param[in]     scale - 0:identity
                           1: s^Ty/y^Ty
***************************************************************/
void c3opt_set_lbfgs_scale(struct c3Opt * opt, int scale)
{
    assert (opt != NULL);
    opt->init_scale = scale;
}

/***********************************************************//**
    Get initial scaling option for LBFGS
    
    \param[in,out] opt - optimization options
***************************************************************/
int c3opt_get_lbfgs_scale(struct c3Opt * opt)
{
    assert (opt != NULL);
    return opt->init_scale;
}

/***********************************************************//**
    1 if bruteforce, 0 otherwise
***************************************************************/
int c3opt_is_bruteforce(const struct c3Opt * opt)
{
    assert (opt != NULL);
    return (opt->alg == BRUTEFORCE);
}

/***********************************************************//**
    1 if stochastic gradient descent, 0 otherwise
***************************************************************/
int c3opt_is_sgd(const struct c3Opt * opt)
{
    assert (opt != NULL);
    return (opt->alg == SGD);
}

/***********************************************************//**
    Add lower bound
***************************************************************/
void c3opt_add_lb(struct c3Opt * opt, double * lb)
{
    assert (opt != NULL);
    assert (lb  != NULL);
    memmove(opt->lb,lb,opt->d * sizeof(double));
}

double * c3opt_get_lb(struct c3Opt * opt)
{
    assert (opt != NULL);
    double * lb = opt->lb;
    return lb;
}

/***********************************************************//**
    Add upper bound
***************************************************************/
void c3opt_add_ub(struct c3Opt * opt, double * ub)
{
    assert (opt != NULL);
    assert (ub  != NULL);
    memmove(opt->ub,ub,opt->d * sizeof(double));
}

double * c3opt_get_ub(struct c3Opt * opt)
{
    assert (opt != NULL);
    return opt->ub;
}

size_t c3opt_get_d(struct c3Opt * opt)
{
    assert (opt != NULL);
    return opt->d;
}

void c3opt_set_verbose(struct c3Opt * opt, int verbose)
{
    assert (opt != NULL);
    opt->verbose = verbose;
}

int c3opt_get_verbose(struct c3Opt* opt)
{
    assert (opt != NULL);
    return opt->verbose;
}

void c3opt_set_maxiter(struct c3Opt * opt , size_t maxiter)
{
    assert (opt != NULL);
    opt->maxiter = maxiter;;
}

size_t c3opt_get_maxiter(const struct c3Opt * opt)
{
    assert (opt != NULL);
    return opt->maxiter;
}

void c3opt_set_absxtol(struct c3Opt * opt, double absxtol)
{
    assert (opt != NULL);
    opt->absxtol = absxtol;
}

size_t c3opt_get_niters(struct c3Opt * opt){
    assert (opt != NULL);
    return opt->niters;
}

double c3opt_get_stored_function(const struct c3Opt * opt, size_t ii)
{
    assert (opt->store_func == 1);
    assert (ii < opt->niters);
    return opt->stored_func[ii];
}

size_t c3opt_get_nevals(struct c3Opt * opt){
    assert (opt != NULL);
    return opt->nevals;
}

size_t c3opt_get_ngvals(struct c3Opt * opt){
    assert (opt != NULL);
    return opt->ngvals;
}

double c3opt_get_absxtol(struct c3Opt * opt)
{
    assert (opt != NULL);
    return opt->absxtol;
}

void c3opt_set_relftol(struct c3Opt * opt, double relftol)
{
    assert (opt != NULL);
    opt->relftol = relftol;
}

double c3opt_get_relftol(struct c3Opt * opt)
{
    assert (opt != NULL);
    return opt->relftol;
}

void c3opt_set_gtol(struct c3Opt * opt, double gtol)
{
    assert (opt != NULL);
    opt->gtol = gtol;
}

double c3opt_get_gtol(struct c3Opt * opt)
{
    assert (opt != NULL);
    return opt->gtol;
}

double * c3opt_get_workspace(struct c3Opt * opt)
{
    assert (opt != NULL);
    return opt->workspace;
}


void c3opt_set_storage_options(struct c3Opt * opt, int store_func, int store_grad, int store_x)
{
    assert (opt != NULL);
    opt->store_func = store_func;
    opt->store_grad = store_grad;
    opt->store_x = store_x;

    if (opt->store_grad == 1){
        opt->stored_grad = calloc_double(opt->d*sizeof(double)*opt->maxiter);
    }
    if (opt->store_x == 1){
        opt->stored_x = calloc_double(opt->d*sizeof(double)*opt->maxiter);
    }
    if (opt->store_func == 1){
        opt->stored_func = calloc_double(sizeof(double)*opt->maxiter);
    }
}

int c3opt_get_store_func(const struct c3Opt * opt)
{
    assert (opt != NULL);
    return opt->store_func;
}

void c3opt_print_stored_values(struct c3Opt * opt, FILE * fp, int width, int prec)
{
    assert (opt != NULL);

    fprintf(fp,"# iter ");
    if (opt->store_func == 1){
        fprintf(fp, "F ");
    }
    if (opt->store_grad == 1){
        for (size_t jj = 0; jj < opt->d; jj++){
            fprintf(fp, "G%zu ",jj);
        }
    }
    if (opt->store_x == 1){
        for (size_t jj = 0; jj < opt->d; jj++){
            fprintf(fp, "x%zu ",jj);
        }
    }
    fprintf(fp,"\n");
        
    for (size_t ii = 0; ii < opt->niters; ii++){
        fprintf(fp,"%zu ",ii);
        if (opt->store_func == 1){
            fprintf(fp,"%*.*G ",width,prec,opt->stored_func[ii]);
        }
        if (opt->store_grad == 1){
            for (size_t jj = 0; jj < opt->d; jj++){
                fprintf(fp,"%*.*G ",width,prec,opt->stored_grad[ii*opt->niters+jj]);
            }
        }
        if (opt->store_x== 1){
            for (size_t jj = 0; jj < opt->d; jj++){
                fprintf(fp,"%*.*G ",width,prec,opt->stored_x[ii*opt->niters+jj]);
            }
        }
        fprintf(fp,"\n");
    }
    
}

int c3opt_ls_get_initial(const struct c3Opt * opt)
{
    assert (opt != NULL);
    return c3ls_get_initial(opt->ls);
}

enum c3opt_ls_alg c3opt_ls_get_alg(const struct c3Opt * opt)
{
    assert (opt != NULL);
    return c3ls_get_alg(opt->ls);
}

void c3opt_ls_set_alpha(struct c3Opt * opt, double alpha)
{
    assert (opt != NULL);
    c3ls_set_alpha(opt->ls,alpha);
}

double c3opt_ls_get_alpha(struct c3Opt * opt)
{
    assert (opt != NULL);
    return c3ls_get_alpha(opt->ls);
}

void c3opt_ls_set_beta(struct c3Opt * opt, double beta)
{
    assert (opt != NULL);
    c3ls_set_beta(opt->ls,beta);
}

double c3opt_ls_get_beta(struct c3Opt * opt)
{
    assert (opt != NULL);
    return c3ls_get_beta(opt->ls);
}

void c3opt_ls_set_maxiter(struct c3Opt * opt,size_t maxiter)
{
    assert (opt != NULL);
    c3ls_set_maxiter(opt->ls,maxiter);
}

size_t c3opt_ls_get_maxiter(struct c3Opt * opt)
{
    assert (opt != NULL);
    return c3ls_get_maxiter(opt->ls);
}

void c3opt_ls_set_alg(struct c3Opt * opt, enum c3opt_ls_alg alg){
    assert (opt != NULL);
    c3ls_set_alg(opt->ls,alg);
}


/***********************************************************//**
    Add brute-force optimization locations
    
    \param[in,out] opt   - optimization structure
    \param[in]     nvals - number of locations to test
    \param[in]     loc   - (d*nvals,) flattened array (d first) of options
***************************************************************/
void c3opt_set_brute_force_vals(struct c3Opt * opt, size_t nvals, double * loc)
{
    assert (opt != NULL);
    if (opt->alg != BRUTEFORCE){
        fprintf(stderr,"Must set optimization type to BRUTEFORCE\n");
        fprintf(stderr,"in order to add brute force values\n");
    }
    opt->nlocs = nvals;
    opt->workspace = calloc_double(nvals * opt->d);
    memmove(opt->workspace,loc,opt->d*nvals*sizeof(double));
}

/***********************************************************//**
    Add objective function
***************************************************************/
void c3opt_add_objective(struct c3Opt * opt,
                         double(*f)(size_t,const double *,double *,void *),
                         void * farg)
{
    assert (opt != NULL);
    opt->f = f;
    opt->farg = farg;
}

/***********************************************************//**
    Add objective function
***************************************************************/
void c3opt_add_objective_stoch(struct c3Opt * opt,
                               double(*fsgd)(size_t,size_t,const double *,double *,void *),
                               void * farg)
{
    assert (opt != NULL);
    opt->fsgd = fsgd;
    opt->farg = farg;
}

/***********************************************************//**
    Evaluate the objective function
    
    \param[in]     opt  - optimization structure
    \param[in]     x    - location at which to evaluate
    \param[in,out] grad - gradient of evaluation (evaluates if not NULL)

    \return  Evaluation
***************************************************************/
double c3opt_eval(struct c3Opt * opt, const double * x, double * grad)
{
    assert (opt != NULL);
    assert (opt->f != NULL);
    double out = opt->f(opt->d,x,grad,opt->farg);
    if (isnan(out)){
        fprintf(stderr,"Optimization function value is NaN\n");
        fprintf(stderr, "x = \n");
        for (size_t ii = 0; ii < opt->d; ii++){
            fprintf(stderr, "x[%zu] = %3.5G\n", ii, x[ii]);
        }
        exit(1);
    }
    else if (isinf(out)){
        (void)(1);
        /* fprintf(stderr,"Optimization function value is inf\n");; */
        /* exit(1); */
    }
    
    opt->nevals+=1;
    if (grad != NULL){
        opt->ngvals += 1;
    }
    return out;
}

/***********************************************************//**
    Evaluate the objective function at a single element in the sum
    
    \param[in]     opt  - optimization structure
    \param[in]     ind  - index for sgd
    \param[in]     x    - location at which to evaluate
    \param[in,out] grad - gradient of evaluation (evaluates if not NULL)

    \return  Evaluation
***************************************************************/
double c3opt_eval_stoch(struct c3Opt * opt, size_t ind, const double * x, double * grad)
{
    assert (opt != NULL);
    assert (opt->fsgd != NULL);
    double out = opt->fsgd(opt->d,ind,x,grad,opt->farg);
    opt->nevals+=1;
    if (grad != NULL){
        opt->ngvals += 1;
    }
    return out;
}

/***********************************************************//**
    Compares the numerical and analytic derivatives 

    \param[in] opt - optimization structure (objective assigned)
    \param[in] x   - evaluation location
    \param[in] eps - difference for central difference

    \return ||deriv num - deriv nalytic||/ || deriv_num ||
***************************************************************/
double c3opt_check_deriv(struct c3Opt * opt, const double * x, double eps)
{
    assert (opt != NULL);

    size_t dim = opt->d;
    double * x1 = calloc_double(dim);
    double * x2 = calloc_double(dim);
    double * grad = calloc_double(dim);
    for (size_t ii = 0; ii < dim; ii++){
        x1[ii] = x[ii];
        x2[ii] = x[ii];
    }
    
    c3opt_eval(opt,x,grad);

    double diff = 0.0;
    double v1,v2;
    double norm = 0.0;
    for (size_t ii = 0; ii < dim; ii++){
        x1[ii] += eps;
        x2[ii] -= eps;
        v1 = c3opt_eval(opt,x1,NULL);
        v2 = c3opt_eval(opt,x2,NULL);
        diff += pow( (v1-v2)/2.0/eps - grad[ii], 2 );
        norm += pow( (v1-v2)/2.0/eps,2);
        
        x1[ii] -= eps;
        x2[ii] += eps;
    }
    if (norm > 1){
        diff /= norm;
    }
    free(x1); x1 = NULL;
    free(x2); x2 = NULL;
    free(grad); grad = NULL;
    return sqrt(diff);
}

/***********************************************************//**
    Compares the numerical and analytic derivatives, also outputs
    the difference between numerical and analytic for each  input

    \param[in] opt       - optimization structure 
    \param[in] x         - evaluation location
    \param[in] eps       - difference for central difference
    \param[in,out] diffs - difference for every input

    \return ||deriv num - deriv analytic||/ || deriv_num ||
***************************************************************/
double c3opt_check_deriv_each(struct c3Opt * opt, const double * x, double eps, double * diffs)
{
    assert (opt != NULL);

    size_t dim = opt->d;
    double * x1 = calloc_double(dim);
    double * x2 = calloc_double(dim);
    double * grad = calloc_double(dim);
    for (size_t ii = 0; ii < dim; ii++){
        x1[ii] = x[ii];
        x2[ii] = x[ii];
    }
    
    c3opt_eval(opt,x,grad);

    double diff = 0.0;
    double v1,v2;
    double norm = 0.0;
    for (size_t ii = 0; ii < dim; ii++){
        x1[ii] += eps;
        x2[ii] -= eps;
        v1 = c3opt_eval(opt,x1,NULL);
        v2 = c3opt_eval(opt,x2,NULL);

        double fd = (v1-v2)/2.0/eps;
        diff += pow( fd - grad[ii], 2 );
        diffs[ii] = fd - grad[ii];
        /* printf("ii=%zu, fd =%G, grad=%G, diff=%G\n",ii,fd,grad[ii],diffs[ii]); */
        /* printf("\t v1=%G, v2=%G\n",v1,v2); */

        
        norm += pow(fd,2);
        
        x1[ii] -= eps;
        x2[ii] += eps;
    }
    if (norm > 1){
        diff /= norm;
    }
    free(x1); x1 = NULL;
    free(x2); x2 = NULL;
    free(grad); grad = NULL;
    return sqrt(diff);
}

/***********************************************************//**
    Move along a line during line search

    \param[in]     d    - dimension
    \param[in]     t    - scale of direction
    \param[in]     dir  - direction (d,)
    \param[in]     x    - location to move from (d,)
    \param[in,out] newx - allocated area for new location (d,)
    \param[in]     lb   - lower bounds
    \param[in]     ub   - upper bounds
***************************************************************/
void c3opt_ls_x_move(size_t d, double t, const double * dir,
                     const double * x,
                     double * newx, double * lb, double * ub)
{

    memmove(newx,x,d*sizeof(double));
    cblas_daxpy(d,t,dir,1,newx,1);
    if (lb != NULL){
        assert (ub != NULL);
        for (size_t ii = 0; ii < d; ii++){
            if      (newx[ii] <= lb[ii]){ newx[ii] = lb[ii]; }
            else if (newx[ii] >= ub[ii]){ newx[ii] = ub[ii]; }
        }
    }
}

/***********************************************************//**
    Backtracking Line search with projection onto box constraints
    Uses Armijo condition

    \param[in]     opt     - optimization structure
    \param[in]     x       - base point
    \param[in]     fx      - objective function value at x
    \param[in]     grad    - gradient at x
    \param[in]     dir     - search direction
    \param[in,out] newx    - new location (x + t*p)
    \param[in,out] newf    - objective function value f(newx)
    \param[in,out] info    -  0 - success
                             -1 - alpha not within correct bounds
                             -2 - beta not within correct bounds
                              1 - maximum number of iter reached
    
    \return line search gradient scaling (t*alpha)
    \note See Convex Optimization (boyd and Vandenberghe) page 464
***************************************************************/
double c3opt_ls_box(struct c3Opt * opt, double * x, double fx,
                    double * grad, double * dir,
                    double * newx, double * newf, int *info)
{
    double alpha = c3opt_ls_get_alpha(opt);
    double beta = c3opt_ls_get_beta(opt);
    double * lb = c3opt_get_lb(opt);
    double * ub = c3opt_get_ub(opt);
    size_t d = c3opt_get_d(opt);
    size_t maxiter = c3opt_ls_get_maxiter(opt);
    double absxtol = c3opt_get_absxtol(opt);
    int verbose = c3opt_get_verbose(opt);
    int initial = c3opt_ls_get_initial(opt);
    
    *info = 0;
    if ((alpha <= 0.0) || (alpha >= 0.5)){
        printf("line search alpha (%G) is not (0,0.5)\n",alpha);
        *info = -1;
        return 0.0;
    }
    if ((beta <= 0.0) && (beta >= 1.0)) {
        printf("line search beta (%G) is not (0,1)\n",beta);
        *info = -2;
        return 0.0;
    }
    for (size_t ii = 0; ii < d; ii++){
        if (x[ii] < lb[ii]){
            printf("line search starting point violates constraints\n");
            *info = -3;
            return 0.0;
        }
        if (x[ii] > ub[ii]){
            printf("line search starting point violates constraints\n");
            *info = -3;
            return 0.0;
        }
    }

    double t = 1.0;
    // add a different scale here
    
    size_t iter = 1;
    double dg = cblas_ddot(d,grad,1,dir,1);
    /* printf("initial = %d\n",initial); */
    if (initial == 1){
        t = fabs(2.0 * (fx - opt->prev_eval) / dg);
        /* printf("initial = 1 t = %G dg=%G\n",t,dg); */
        if (1.01 * t > 1.01){
            t = 1.0;
        }
    }
    t = 1.0;
    /* printf("t = %G\n",t); */
    
    c3opt_ls_x_move(d,t,dir,x,newx,lb,ub);
    *newf = c3opt_eval(opt,newx,NULL);

    if (verbose > 2){
        printf("\t LineSearch Iteration:%zu (fval) = (%G)\n",iter,*newf);
        printf("\t\t newx = "); dprint(d,newx);
    }
    
    double checkval = fx + alpha*t*dg;
    while(*newf > checkval)
    {
        if (iter >= maxiter){
            *info = 1;
            if (verbose > 1){
                printf("Warning: maximum number of iterations (%zu) of line search reached\n",iter);
                printf("t = %G\n",t);
            }
            break;
        }
        else if (iter > 1){
            t = beta * t;            
        }

        checkval = fx + alpha*t*dg;
        c3opt_ls_x_move(d,t,dir,x,newx,lb,ub);
        *newf = c3opt_eval(opt,newx,NULL);

        
        iter += 1;
        //printf("absxtol = %G\n",absxtol);
        if (verbose > 2){
            printf("\t LineSearch Iteration:%zu (fval) = (%G)\n",iter,*newf);
            printf("\t\t newx = "); dprint(d,newx);
            /* printf("\t\t oldx = "); dprint(d,x); */
            
        }
        double diff = norm2diff(newx,x,d);
        if (diff < absxtol){
            break;
        }
    } 

    return t*alpha;
}

/***********************************************************//**
    Line Search with bisection for Wolfe conditions

    \param[in]     opt     - optimization structure
    \param[in]     x       - base point
    \param[in]     fx      - objective function value at x
    \param[in,out] grad    - gradient at x (then at new value)
    \param[in]     dir     - search direction
    \param[in,out] newx    - new location (x + t*p)
    \param[in,out] newf    - objective function value f(newx)
    \param[in,out] info    -  0 - success
                             -1 - alpha not within correct bounds
                             -2 - beta not within correct bounds
                             -3 - initial starting point violates constraints
                             -4 - initial direction is not a descent direction
                              1 - maximum number of iter reached
    
                              
***************************************************************/
double c3opt_ls_wolfe_bisect(struct c3Opt * opt, double * x, double fx,
                             double * grad, double * dir,
                             double * newx, double * newf, int *info)
{
    double alpha = c3opt_ls_get_alpha(opt);
    double beta = c3opt_ls_get_beta(opt);
    double * lb = c3opt_get_lb(opt);
    double * ub = c3opt_get_ub(opt);
    size_t d = c3opt_get_d(opt);
    size_t maxiter = c3opt_ls_get_maxiter(opt);
    /* double absxtol = c3opt_get_absxtol(opt); */
    int verbose = c3opt_get_verbose(opt);


    double t = 1.0;
    double tmax = 0.0;
    double tmin = 0.0;
    double dg = cblas_ddot(d,grad,1,dir,1);
    /* double normdir = cblas_ddot(d,dir,1,dir,1); */
    
    *info = 0;
    if ((alpha <= 0.0) || (alpha >= 0.5)){
        printf("line search alpha (%G) is not (0,0.5)\n",alpha);
        *info = -1;
        return 0.0;
    }
    if ((beta <= 0.0) && (beta >= 1.0)) {
        printf("line search beta (%G) is not (0,1)\n",beta);
        *info = -2;
        return 0.0;
    }
    for (size_t ii = 0; ii < d; ii++){
        if (x[ii] < lb[ii]){
            printf("line search starting point violates constraints\n");
            *info = -3;
            return 0.0;
        }
        if (x[ii] > ub[ii]){
            printf("line search starting point violates constraints\n");
            *info = -3;
            return 0.0;
        }
    }
    
    if (dg > 1e-15){
        if (verbose > 4){
            fprintf(stderr,"line search initial direction is not a descent direction, dg=%G\n",dg);
            fprintf(stderr,"gradient norm is %G\n", cblas_ddot(d,grad,1,grad,1));
        }
        
        memmove(newx,x,d*sizeof(double));
        *newf = fx;
        /* exit(1); */
        *info = -4;
        return 0.0;
    }

    double checkval, dg2;
    size_t iter = 1;
    double fval;
    tmin = 0.0;
    tmax = 0.0;
    while(iter < maxiter){

        checkval = fx + alpha*t*dg; // phi(0) + alpha * t * phi'(0)
        c3opt_ls_x_move(d,t,dir,x,newx,lb,ub);
        fval = c3opt_eval(opt,newx,NULL);
        if (verbose > 1){
            printf("Iter=%zu/%zu,t=%G,fx=%G,required=%3.10G,fval=%3.10G\n",iter,maxiter,t,fx,checkval,fval);
        }
        
        if (fval > checkval){
            tmax = t;
            t = 0.5 * (tmin + tmax);
            if (verbose > 1){
                printf("\t Sufficient descent not satisfied, (t,tmax) = (%G,%G)\n",t,tmax);
            }
        }
        else{
            c3opt_eval(opt,newx,grad);
            dg2 = cblas_ddot(d,grad,1,dir,1);
            if (verbose > 1){
                printf("\t p^Td=%G, required=%G, current=%G\n",dg,beta*dg,dg2);
            }
            if (dg2 < beta * dg){
                tmin = t;
                if (fabs(tmax) < 1e-15){
                    t = 2.0 * tmin;
                }
                else{
                    t = 0.5 * (tmin + tmax);
                }
            }
            else{
                *newf = fval;
                break;
            }
        }

        /* if (t*normdir < absxtol){ */
        /*     *newf = c3opt_eval(opt,newx,grad); */
        /*     break; */
        /* } */
        
        iter += 1;
    }
    if (iter == maxiter){
        *newf = c3opt_eval(opt,newx,grad);
        *info = 1;
        if (verbose > 1){
            printf("Warning: maximum number of iterations (%zu) of line search reached\n",iter);
            printf("t = %G\n",t);
        }
    }
    return t*alpha;
}



/***********************************************************//**
    Zoom function for Strong Wolfe
***************************************************************/
double c3opt_ls_zoom(struct c3Opt * opt, double tlow,
                     double flow,
                     double thigh, const double * x,
                     double * dir,
                     double * newx, double fx,
                     double dg, double * grad,
                     double * fval)
{
    double alpha = c3opt_ls_get_alpha(opt);
    double beta = c3opt_ls_get_beta(opt);
    double * lb = c3opt_get_lb(opt);
    double * ub = c3opt_get_ub(opt);
    size_t d = c3opt_get_d(opt);
    double th = thigh;
    double tl = tlow;
    double fl = flow;
    
    // might need to do something smarter here
    double t,checkval,dg2,fcurr;
    /* printf("start new\n"); */
    /* if (fabs(th - tl) < 1e-){ */
    /*     exit(1); */
    /* } */
    while (fabs(th-tl) >= 1e-20){
        /* printf("(tlow,thigh) = (%3.15G,%3.15G),\n",tl,th); */

        t = (tl + th)/2.0;
        c3opt_ls_x_move(d,t,dir,x,newx,lb,ub);

        fcurr = c3opt_eval(opt,newx,grad);
        *fval = fcurr;
        
        checkval = fx + alpha * t * dg;
        /* printf("\t fcurr=%G, fx=%G, checkval=%G\n",fcurr,fx,checkval); */
        if ((fcurr > checkval) || (fcurr >= fl)){
            th = t;
        }
        else{
            dg2 = cblas_ddot(d,grad,1,dir,1);
            if (fabs(dg2) <= -beta * dg){
                *fval = fcurr;
                /* printf("out normal\n"); */
                return t;
            }
            else if (dg2 * (th - tl) >= 0.0){
                th = tl;
            }
            tl = t; // this is weird because t is between?
            fl = fcurr;
        }
    }
    /* printf("out bad\n"); */
    return tlow;
}

/***********************************************************//**
    Line Search with Strong Wolfe

    \param[in]     opt     - optimization structure
    \param[in]     x       - base point
    \param[in]     fx      - objective function value at x
    \param[in,out] grad    - gradient at x
    \param[in]     dir     - search direction
    \param[in,out] newx    - new location (x + t*p)
    \param[in,out] newf    - objective function value f(newx)
    \param[in,out] info    -  0 - success
                             -1 - alpha not within correct bounds
                             -2 - beta not within correct bounds
                             -3 - initial starting point violates constraints
                             -4 - initial direction is not a descent direction
                              1 - maximum number of iter reached
    
                              
***************************************************************/
double c3opt_ls_strong_wolfe(struct c3Opt * opt, double * x, double fx,
                             double * grad, double * dir,
                             double * newx, double * newf, int *info)
{
    double alpha = c3opt_ls_get_alpha(opt);
    double beta = c3opt_ls_get_beta(opt);
    double * lb = c3opt_get_lb(opt);
    double * ub = c3opt_get_ub(opt);
    size_t d = c3opt_get_d(opt);
    size_t maxiter = c3opt_ls_get_maxiter(opt);
    double absxtol = c3opt_get_absxtol(opt);
    int verbose = c3opt_get_verbose(opt);
    int initial = c3opt_ls_get_initial(opt);
    
    *info = 0;
    if ((alpha <= 0.0) || (alpha >= 0.5)){
        printf("line search alpha (%G) is not (0,0.5)\n",alpha);
        *info = -1;
        return 0.0;
    }
    if ((beta <= 0.0) && (beta >= 1.0)) {
        printf("line search beta (%G) is not (0,1)\n",beta);
        *info = -2;
        return 0.0;
    }
    for (size_t ii = 0; ii < d; ii++){
        if (x[ii] < lb[ii]){
            printf("line search starting point violates constraints\n");
            *info = -3;
            return 0.0;
        }
        if (x[ii] > ub[ii]){
            printf("line search starting point violates constraints\n");
            *info = -3;
            return 0.0;
        }
    }

    double tmax = 1.0;
    double t = 1.0;
    double dg = cblas_ddot(d,grad,1,dir,1);
    if (dg > 1e-15){
        if (verbose > 4){
            fprintf(stderr,"line search initial direction is not a descent direction, dg=%G\n",dg);
        }
        *info = -4;
        memmove(newx,x,d*sizeof(double));
        *newf = fx;
        return 0.0;
    }
    if (initial == 1){
        t = fabs(2.0 * (fx - opt->prev_eval) / dg);
        if (1.01 * t > 1.0){
            t = 1.0;
        }
    }
    t = 0.1;
    
    //phi(t) = f(x + t dir)
    double checkval;
    double t_prev = 0.0;
    double f_prev = fx;
    size_t iter = 1;
    while (iter <= maxiter){
        /* printf("(t1,t2) = (%G,%G)\n",t_prev,t); */
        c3opt_ls_x_move(d,t,dir,x,newx,lb,ub);
        *newf = c3opt_eval(opt,newx,grad);
        
        double diff = norm2diff(newx,x,d);
        if (diff < absxtol){ return t*alpha;}


        checkval = fx + alpha*t*dg; // phi(0) + alpha * t * phi'(0)

        if (verbose > 1){
            printf("Iter=%zu,t=%G,fx=%G,checkval=%G,dg=%G\n",iter,t,fx,checkval,dg);
        }
        if ((*newf > checkval) || ( (*newf >= f_prev) && (iter > 1))){
            /* printf("found range (%G,%G) \n",t_prev,t); */
            if (f_prev > *newf){
                /* printf("f_prev = %G\n",f_prev); */
                /* printf("*newf = %G\n",*newf); */
                assert (f_prev < *newf);
            }
            t = c3opt_ls_zoom(opt,t_prev,f_prev,t,x,dir,newx,fx,dg,grad,newf);
            return t * alpha;
        }
        
        double dg2 = cblas_ddot(d,grad,1,dir,1);
        if (verbose > 2){
            printf("\t dg2=%G required = %G x=",dg2,-beta*dg); dprint(d,newx);
            printf("\t \t grad=");dprint(d,grad);
        }
        /* printf("fx=%G, f_prev = %G, *newf=%G\n",fx,f_prev,*newf); */
        if (fabs(dg2) <= -beta*dg){
            /* printf("here?\n"); */
            return t*alpha;
        }
        else if (dg2 >= 0.0){
            assert (*newf < f_prev);
            t = c3opt_ls_zoom(opt,t,*newf,t_prev,x,dir,newx,fx,dg,grad,newf);
            return t * alpha;
        }

        t_prev = t;
        f_prev = *newf;
        /* printf("INCREASE STEP LENGTH\n"); */
        t = t + 0.5*(tmax-t);
        /* if (fabs(tmax-t) < 1e-10){ */
        /*     printf("Cannot increase step length anymore!\n"); */
        /*     return t*alpha; */
        /*     /\* exit(1); *\/ */
        /* } */


        iter++;
    }
    
    *info = 1;
    if (verbose > 1){
        printf("Warning: maximum number of iterations (%zu) of line search reached\n",iter);
        printf("t = %G\n",t);
    }
    
    return t*alpha;
}

/***********************************************************//**
    Set the initial learning rate for SGD
***************************************************************/
void c3opt_set_sgd_learn_rate(struct c3Opt * opt, double learn_rate)
{
    
    assert (opt != NULL);
    assert (opt->sgd != NULL);
    assert (learn_rate <= 1.0);
    opt->sgd->learn_rate = learn_rate;
}

/***********************************************************//**
    Set the number of samples per epoch for SGD
***************************************************************/
void c3opt_set_sgd_nsamples(struct c3Opt * opt, size_t nsamples)
{
    assert (opt != NULL);
    opt->sgd->nsamples = nsamples;
}

/***********************************************************//**
    Get the number of samples per epoch for SGD
***************************************************************/
size_t c3opt_get_sgd_nsamples(const struct c3Opt * opt)
{
    assert (opt != NULL);
    return opt->sgd->nsamples;
}


static void shuffle(size_t N, size_t * orders)
{
    if (N > 1){
        for (size_t ii = 0; ii < N-1; ii++){
            size_t jj = ii + (size_t)rand()/(RAND_MAX / (N - ii) + 1);
            size_t t = orders[jj];
            orders[jj] = orders[ii];
            orders[ii] = t;
            
        }
    }
}

static void sgd_term_check(struct c3Opt * opt,
                           size_t d, const double * next_step, const double * current_step,
                           size_t ndata_train, size_t ndata, const size_t * order,
                           double * xdiff, double * train_error, double * test_error)
{
    *xdiff = 0.0;
    for (size_t ii = 0; ii < d; ii++){
        *xdiff += (next_step[ii]-current_step[ii])*(next_step[ii]-current_step[ii]);
    }

    // check training set error
    *train_error = 0.0;
    for (size_t ii = 0; ii < ndata_train; ii++){
        *train_error += c3opt_eval_stoch(opt,order[ii],next_step,NULL);
    }
    *train_error /= (double)(ndata_train);

    // check test error
    *test_error = 0;
    for (size_t ii = ndata_train; ii < ndata; ii++){
        /* printf("order[%zu] = %zu",ii,order[ii]); */
        *test_error += c3opt_eval_stoch(opt,order[ii],next_step,NULL);
    }
    *test_error /= (double)(ndata - ndata_train);
}

static void sgd_inter_clean(double ** cs, double ** ps, double ** is, double **ns, size_t ** o)
{
    if (*cs != NULL){
        free(*cs); *cs = NULL;
    }
    if (*ps != NULL){
        free(*ps); *ps = NULL;
    }
    if (*is != NULL){
        free(*is); *is = NULL;        
    }
    if (*ns != NULL){
        free(*ns); *ns = NULL;
    }

    if (*o != NULL){
        free(*o); *o = NULL;
    }
}

/***********************************************************//**
    Stochastic gradient descent

    \param[in]     opt        - optimization
    \param[in,out] x          - starting/final point
    \param[in,out] fval       - final function value

    \return  0    - success
            -20+? - failure in gradient (gradient outputs ?)
             1    - maximum number of iterations reached
         other    - error in backward_line_search 
                   (see that function)
***************************************************************/
int c3_opt_sgd(struct c3Opt * opt, double * x, double * fval)
{
    
    size_t d = c3opt_get_d(opt);
    double * workspace = c3opt_get_workspace(opt);
    int verbose = c3opt_get_verbose(opt);
    /* double * lb = c3opt_get_lb(opt); */
    /* double * ub = c3opt_get_ub(opt); */
    size_t maxiter = c3opt_get_maxiter(opt);
    double gtol = c3opt_get_gtol(opt);
    double relftol = c3opt_get_relftol(opt);
    double absxtol = c3opt_get_absxtol(opt);

    size_t ndata = c3opt_get_sgd_nsamples(opt);
    if (ndata < 2){
        fprintf(stderr, "Must set number of samples per epoch for SGD\n");
        exit(1);
    }
    
    opt->nevals = 0;
    opt->ngvals = 0;
    opt->niters = 0;    

    size_t * order = calloc_size_t(ndata);
    for (size_t ii = 0; ii < ndata; ii++){
        order[ii] = ii;
    }


    // shuffle the order, first half will be training set, second will be validation test
    shuffle(ndata,order);

    size_t ndata_train = (size_t) lrint(ndata / 1.1);
    size_t ndata_validate = ndata - ndata_train;

    if (verbose > 0){
        printf("Stochastic Gradient descent\n");
        printf("---------------------------\n\n");
        
        printf("\t ndata = %zu\n",ndata);
        printf("\t ndata_train = %zu\n",ndata_train);
        printf("\t ndata_validate = %zu\n",ndata_validate);
    }
    
    int ret = C3OPT_SUCCESS;
    

    double xdiff;
    double old_train = 0.0;
    /* double old_test = 0.0; */
    double train_error = 0;
    double test_error = 0;
    double learn_rate = opt->sgd->learn_rate;

    if (verbose > 0){
        printf("\nBegin optimization with learn rate = %G\n",learn_rate);
        printf("\n");
        printf("%7s|%22s|%23s|%23s|%10s|%14s\n"," Epoch "," Training Cost function  ", " Validation Cost function ","    ||x-x_p||      "," Learning rate "," g^Tg     ");
        printf("-----------------------------------------------------------------------------------------------------------------------\n");
    }


    int do_momentum = opt->sgd->do_momentum; // 0 standard, 1 ADAM */
    
    // stores the previous iterate
    double * current_step = calloc_double(d);
    double * previous_step = calloc_double(d);
    double * next_step = calloc_double(d);
    double * inter_step = calloc_double(d);
    memmove(next_step,x,d*sizeof(double));
    memmove(current_step,x,d*sizeof(double));
    memmove(previous_step,x,d*sizeof(double));


    if (opt->store_func == 1){
        sgd_term_check(opt,d,next_step,current_step,ndata_train,
                       ndata,order,&xdiff,&train_error,&test_error);
        opt->stored_func[opt->niters] = train_error;
        opt->niters++;
    }

    double * first_moment = calloc_double(d);
    double * bias_corrected_first = calloc_double(d);
    double * second_moment = calloc_double(d);
    double * bias_corrected_second = calloc_double(d);
    double beta_1 = 0.9;
    double beta_2 = 0.999;

    double eps = 1e-8;
    size_t time = 1;
    double avg_ginner;
    double ginner;
    size_t iter;
    for (iter = 0; iter < maxiter; iter++){
        shuffle(ndata,order);
        avg_ginner = 0.0;
        for (size_t ii = 0; ii < ndata_train; ii++){
            memmove(previous_step,current_step,d*sizeof(double));
            memmove(current_step,next_step,d*sizeof(double));
            
            /* printf(" starting step: "); dprint(d,current_step); */
            /* printf("point %zu\n",order[ii]); */
            ginner = cblas_ddot(d,workspace,1,workspace,1);
            *fval = c3opt_eval_stoch(opt,order[ii],current_step,workspace);
            avg_ginner += ginner;
            /* printf("ginner = %3.5G\n", ginner); */
            if ((do_momentum == 0) || (iter < 10)){

                for (size_t jj = 0; jj < d; jj++){
                    /* printf("workspace[%zu]= %G\n",jj,workspace[jj]); */
                    next_step[jj] = current_step[jj] - learn_rate * workspace[jj];
                }
            }
            else{
                // update first moment
                for (size_t jj = 0; jj < d; jj++){
                    first_moment[jj] = beta_1 * first_moment[jj] + (1.0 - beta_1) * workspace[jj];
                    second_moment[jj] = beta_2 * second_moment[jj] + (1.0 - beta_2) * workspace[jj] * workspace[jj];

                    bias_corrected_first[jj] = first_moment[jj] / ( 1.0 - pow(beta_1,time));
                    bias_corrected_second[jj] = second_moment[jj] / ( 1.0 - pow(beta_2,time));
                    next_step[jj] += -learn_rate * bias_corrected_first[jj] / (sqrt(bias_corrected_second[jj]) + eps);
                }
            }
   
        }
        learn_rate *= opt->sgd->learn_rate_decay;


        old_train = train_error;
        /* old_test = test_error; */
        sgd_term_check(opt,d,next_step,current_step,ndata_train,
                       ndata,order,&xdiff,&train_error,&test_error);

        double grad_inner = avg_ginner / (double)(ndata_train);
        double dtrain = fabs(train_error - old_train);
        if (old_train > 1e-10){
            dtrain /= train_error;            
        }

        /* double dtest = test_error - old_test; */

        if (verbose > 0){
            printf("  %-4zu |",iter+1);
            printf("   %-22.7G|   %-23.7G| %-22.7G| %-14.5G| %-14.5G |  %-14.5G",train_error,test_error,xdiff,learn_rate,grad_inner, dtrain);
            /* printf("x = "); dprint(d,x); */
            printf("\n");
        }

        time++;


        if (opt->store_func == 1){
            opt->stored_func[opt->niters] = train_error;
        }
        opt->niters++;


        *fval = train_error;
        memmove(x,next_step,d*sizeof(double));

        if (iter > 5){
            if (grad_inner < gtol){
                ret = C3OPT_GTOL_REACHED;
                break;
            }
            else if (fabs(dtrain) < relftol){
                ret = C3OPT_FTOL_REACHED;
                break;
            }
            else if (fabs(xdiff) < absxtol){
                ret = C3OPT_XTOL_REACHED;
                break;
            }

            if (learn_rate < 1e-20){
                break;
            }
        }

    }

    memmove(x,next_step,d*sizeof(double));

    if (iter > maxiter-1){
        ret = C3OPT_MAXITER_REACHED;
    }
    
    free(first_moment); first_moment = NULL;
    free(second_moment); second_moment = NULL;
    free(bias_corrected_first); bias_corrected_first = NULL;
    free(bias_corrected_second); bias_corrected_second = NULL;
    sgd_inter_clean(&current_step,&previous_step,&inter_step,&next_step,&order);
    if (verbose > 0){
        printf("\n");
    }
    return ret;
}



/***********************************************************//**
    Projected Gradient damped BFGS

    \param[in]     opt        - optimization
    \param[in,out] x          - starting/final point
    \param[in,out] fval       - final function value
    \param[in,out] grad       - gradient at final point
    \param[in,out] invhess    - approx hessian at start and end
                                only upper triangular part used

    \return  0    - success
            -20+? - failure in gradient (gradient outputs ?)
             1    - maximum number of iterations reached
         other    - error in backward_line_search 
                   (see that function)
***************************************************************/
int c3_opt_damp_bfgs(struct c3Opt * opt,
                     double * x, double * fval,
                     double * grad,
                     double * invhess)
                 
{

    size_t d = c3opt_get_d(opt);
    double * workspace = c3opt_get_workspace(opt);
    int verbose = c3opt_get_verbose(opt);
    double * lb = c3opt_get_lb(opt);
    double * ub = c3opt_get_ub(opt);
    size_t maxiter = c3opt_get_maxiter(opt);
    double gtol = c3opt_get_gtol(opt);
    double relftol = c3opt_get_relftol(opt);
    double absxtol = c3opt_get_absxtol(opt);

    opt->nevals = 0;
    opt->ngvals = 0;
    opt->niters = 1;

    *fval = c3opt_eval(opt,x,grad);

    /* //HIHO */
    /* exit(1); */
    if (isnan(*fval)){
        fprintf(stderr,"Initial optimization function valueis NaN\n");
        exit(1);
    }
    else if (isinf(*fval)){
        fprintf(stderr,"Initial optimization function value is inf\n");
        exit(1);
    }
    
    if (opt->store_func == 1){
        opt->stored_func[opt->niters-1] = *fval;
    }
    if (opt->store_grad == 1){
        memmove(opt->stored_grad+(opt->niters-1)*opt->d,grad,d*sizeof(double));
    }
    if (opt->store_x == 1){
        memmove(opt->stored_x+(opt->niters-1)*opt->d,x,d*sizeof(double));
    }

    
    cblas_dsymv(CblasColMajor,CblasUpper,
                d,-1.0,invhess,d,grad,1,0.0,workspace+d,1);

    int ret = C3OPT_SUCCESS;
    
    double grad_norm = sqrt(cblas_ddot(d,grad,1,grad,1));
    double eta;
    if (verbose > 0){

        printf("Initial values:\n \t (fval,||g||) = (%3.5G,%3.5G)\n",*fval,grad_norm);

        if (verbose > 3){
            printf("\t x = "); dprint(d,x);
        }
    }

    /* if (grad_norm > 10 * fabs(*fval)){ */
    /*     for (size_t ii = 0; ii < d; ii++){ */
    /*         grad[ii] *= 1e-2*(fabs(*fval) / grad_norm); */
    /*     } */
    /*     printf("lower grad, new norm = %3.5G\n", sqrt(cblas_ddot(d,grad,1,grad,1))); */
    /* } */

    size_t iter = 0;
    double fvaltemp;
    int onbound = 0;
    int converged = 0;
    int res = 0;
    /* double sc = 0.0;; */
    opt->prev_eval = 0.0;
    enum c3opt_ls_alg alg = c3opt_ls_get_alg(opt);
    while (converged == 0){
        
        memmove(workspace,x,d*sizeof(double));
        // workspace[0:d) is current x
        // workspace[d:2d) is the search direction
        // workspace[2d:3d) is gradient at a new x
        
        fvaltemp = *fval;
        if (alg == BACKTRACK){
            c3opt_ls_box(opt,workspace,fvaltemp,
                         grad,workspace+d,
                         x,fval,&res);
            c3opt_eval(opt,x,workspace+2*d);
        }
        else if (alg == STRONGWOLFE){
            memmove(workspace+2*d,grad,d*sizeof(double));
            c3opt_ls_strong_wolfe(opt,workspace,
                                  fvaltemp,
                                  workspace+2*d,
                                  workspace+d,
                                  x,fval,&res);
        }
        else if (alg == WEAKWOLFE){
            memmove(workspace+2*d,grad,d*sizeof(double));
            res = -4;
            int round=0;
            while (res < 0){
                c3opt_ls_wolfe_bisect(opt,workspace,
                                      fvaltemp,
                                      workspace+2*d,
                                      workspace+d,
                                      x,fval,&res);
                if (res == -4){
                    if (verbose > 4){
                        fprintf(stderr,"Round %d\n",round);
                        fprintf(stderr,"\t Warning line search did not move because lack of\n");
                        fprintf(stderr,"\t descent direction. Changing direction to -gradient\n");
                    }
                    memmove(workspace+d,grad,d*sizeof(double));
                    for (size_t ii = 0; ii < d; ii++){
                        workspace[d+ii] *= -1;
                    }
                    round++;
                }
                else if (res < 0){
                    fprintf(stderr,"Warning: line search returns %d\n",res);
                    exit(1);
                }
            }
        }
        /* assert (*fval < fvaltemp); */
        if (res < 0){
            fprintf(stderr,"Warning: line search returns %d\n",res);
        }
        /* assert (res > -1); */

        // x is now at the nextiterate
        // workspace[2d:3d) is the gradient at the next iterate
        
        opt->prev_eval = fvaltemp;

        if (opt->store_func == 1){
            opt->stored_func[opt->niters] = *fval;
        }
        if (opt->store_grad == 1){
            memmove(opt->stored_grad+opt->niters*opt->d,
                    workspace+2*d,d*sizeof(double));
        }
        if (opt->store_x == 1){
            memmove(opt->stored_x+opt->niters*opt->d,x,d*sizeof(double));
        }

        double xdiff = 0.0;
        double xnorm = 0.0;
        for (size_t ii = 0; ii < d; ii++){
            xdiff += pow(x[ii]-workspace[ii],2);
            xnorm += workspace[ii]*workspace[ii];
        }
        xdiff = sqrt(xdiff);
        xnorm = sqrt(xnorm);
        
        // compute s = xnew - xold
        cblas_dscal(d,-1.0,workspace,1);
        cblas_daxpy(d,1.0,x,1,workspace,1);
        double * s = workspace;

        // compute difference in gradients
        cblas_daxpy(d,-1.0,grad,1,workspace+2*d,1); 
        double * y = workspace+2*d;
        
        // combute BY;
        cblas_dsymv(CblasColMajor,CblasUpper,
                    d,1.0,invhess,d,y,1,0.0,workspace+3*d,1);
        
        double sty = cblas_ddot(d,s,1,y,1);
        double ytBy = cblas_ddot(d,y,1,workspace+3*d,1);
        if (fabs(sty) > 1e-16){
            double a1 = (sty + ytBy)/(sty * sty);
        
            // rank 1 update
            cblas_dsyr(CblasColMajor,CblasUpper,d,a1,
                       s,1,invhess, d);
        
            double a2 = -1.0/sty;
            // symmetric rank 2 updatex
            cblas_dsyr2(CblasColMajor,CblasUpper,d,a2,
                        workspace+3*d,1,
                        /* workspace+d,1,invhess,d); */
                        s,1,invhess,d);
        }

        cblas_daxpy(d,1.0,y,1,grad,1);

        
        // compute next search direction;
        cblas_dsymv(CblasColMajor,CblasUpper,
                d,-1.0,invhess,d,grad,1,0.0,workspace+d,1);
        

        eta = cblas_ddot(d,grad,1,workspace+d,1);
        onbound = 0;
        for (size_t ii = 0; ii < d; ii++){
            if ((x[ii] <= lb[ii]) || (x[ii] >= ub[ii])){
                onbound = 1;
                break;
            }
        }
        double diff = fabs(*fval - fvaltemp);

        /* printf("diff = %G, relftol = %G, diff < relftol = %d\n",diff,relftol,diff<relftol); */
        if (fabs(fvaltemp) > 1e-10){
            diff /= fabs(fvaltemp);
        }

        if (onbound == 1){

            //dprint(2,grad);
            if (diff < relftol){
                //printf("converged close?\n");
                ret = C3OPT_FTOL_REACHED;
                converged = 1;
            }
            else{
                if (xdiff < absxtol){
                    ret = C3OPT_XTOL_REACHED;
                    //printf("converged xclose\n");
                    converged = 1;
                }
            }
        }
        else{
            if ( (eta*eta/2.0) < pow(gtol,2)){
                ret = C3OPT_GTOL_REACHED;
                //printf("converged gradient\n");
                converged = 1;
            }
            else if (diff < relftol){
                /* printf ("HEEERE\n"); */
                ret = C3OPT_FTOL_REACHED;
                /* if (fabs(diff) < 1e-30){ */
                /*     printf("really? %G\n",diff); */
                /*     if (xdiff < 1e-30){ */
                /*         converged = 1; */
                /*     } */
                /* } */
                /* else{ */
                    /* printf("SHOULD BE HERE!\n"); */
                    converged = 1;
                /* } */
            }
            else if (xdiff < absxtol){
                if (iter > 3){
                    ret = C3OPT_XTOL_REACHED;
                    converged = 1;
                }
            }
                
        }

        /* if (fabs(eta) > 10 * fabs(*fval)){ */
        /*     for (size_t ii = 0; ii < d; ii++){ */
        /*         workspace[ii+d] *= 1e-2*(fabs(*fval) / eta); */
        /*     } */
        /*     /\* printf("lower grad, new norm = %3.5G\n", sqrt(cblas_ddot(d,grad,1,grad,1))); *\/ */
        /* } */

        
        if (verbose > 0){
            printf("Iteration:%zu/%zu\n",iter,maxiter);
            printf("\t f(x)          = %3.15G\n",*fval);
            printf("\t |f(x)-f(x_p)| = %3.5G\n",diff);
            printf("\t |x - x_p|/|x_p| = %3.5G\n",xdiff/xnorm);
            printf("\t p^Tg =        = %3.5G\n",eta);
            printf("\t Onbound       = %d\n",onbound);
            if (verbose > 3){
                printf("\t x = "); dprint(d,x);
            }

        }
        

        opt->niters++;
        iter += 1;

        if (iter > maxiter){
            //printf("iter = %zu,verbose=%d\n",iter,verbose);
            ret = C3OPT_MAXITER_REACHED;
            converged = 1;
        }
    }

    return ret;
}


// circular list storing lbfgs vectors
/// @private
struct c3opt_lbfgs_data
{
    size_t iter;
    double * s;
    double * y;
    double rhoinv;
    struct c3opt_lbfgs_data * next;
    struct c3opt_lbfgs_data * prev;
};

/***********************************************************//**
    Allocate low-memory fgs data structure  
***************************************************************/
struct c3opt_lbfgs_data * c3opt_lbfgs_data_alloc(){
    struct c3opt_lbfgs_data* out = malloc(sizeof(struct c3opt_lbfgs_data));
    if (out == NULL){
        fprintf(stderr, "Cannot allocate lbfgs_data to store data\n");
        return NULL;
    }
    out->s = NULL;
    out->y = NULL;
    out->next = NULL;
    out->prev = NULL;
    return out;
}

/***********************************************************//**
    Print lbfgs data structure
***************************************************************/
void c3opt_lbfgs_data_print(size_t dim,
                            struct c3opt_lbfgs_data * data, FILE * fp,int width,int prec)
{
    if (data!= NULL){
        fprintf(fp,"Iter=%zu, rhoinv=%*.*G\n",data->iter,width,prec,data->rhoinv);
        fprintf(fp,"\t s = ");
        for (size_t ii = 0; ii < dim; ii++){
            fprintf(fp,"%*.*G ",width,prec,data->s[ii]);
        }
        fprintf(fp,"\n");
        fprintf(fp," \t (prev==NULL)=%d, (next==NULL)=%d\n",data->prev==NULL,data->next==NULL);
    }
    else{
        fprintf(fp,"NULL\n");
    }
    
}

/***********************************************************//**
    Free contets of the lbfgs data structure
***************************************************************/
void c3opt_lbfgs_data_free_contents(struct c3opt_lbfgs_data * list)
{
    if (list != NULL){
        free(list->s); list->s = NULL;
        free(list->y); list->y = NULL;
    }
}


struct c3opt_lbfgs_list
{
    size_t m; 
    size_t d;
    size_t size;
    struct c3opt_lbfgs_data * most_recent;

    // iterate
    size_t step;
    struct c3opt_lbfgs_data * cstep;
};


/***********************************************************//**
    Allocate a list of lbfgs data.

    \param[in] m - number of steps to remember
    \param[in] d - dimension of memory
***************************************************************/
struct c3opt_lbfgs_list * c3opt_lbfgs_list_alloc(size_t m, size_t d){
    struct c3opt_lbfgs_list* out = malloc(sizeof(struct c3opt_lbfgs_list));
    if (out == NULL){
        fprintf(stderr, "Cannot allocate lbfgs_list to store data\n");
        return NULL;
    }
    out->m = m;
    out->d = d;
    out->size = 0;
    out->most_recent = NULL;

    out->step = 0;
    out->cstep = NULL;
    return out;
}

/***********************************************************//**
    Insert a new element as the first element of the bfgs list
    
    \param[in,out] list      - list to be updated
    \param[in]     iter      - iteration of optimization
    \param[in]     x_next    - next design point
    \param[in]     x_curr    - current design point
    \param[in]     grad_next - gradient at next design point
    \param[in]     grad_curr - gradient at current design point
***************************************************************/
void c3opt_lbfgs_list_insert(struct c3opt_lbfgs_list * list, size_t iter,
                             const double * x_next, const double * x_curr,
                             const double * grad_next, const double * grad_curr)
{

    struct c3opt_lbfgs_data * new = NULL;
    struct c3opt_lbfgs_data * head = list->most_recent;
    
    if (list->size >= list->m){ // full list replace last one
        new = head->prev; // last one becomes newest;
    }
    else{ // need to create a new one
        new = c3opt_lbfgs_data_alloc();
    }

    if (new->s == NULL){
        new->s = calloc_double(list->d);
        new->y = calloc_double(list->d);
    }

    //s = x_next - x_curr
    memmove(new->s,x_next,list->d * sizeof(double));
    cblas_daxpy(list->d,-1.0,x_curr,1,new->s,1);

    //y = g_next - g_curr
    memmove(new->y,grad_next,list->d * sizeof(double));
    cblas_daxpy(list->d,-1.0,grad_curr,1,new->y,1);

    // 1/rho = s^Ty
    new->rhoinv = cblas_ddot(list->d,new->s,1,new->y,1);
    
    // NEED TO CHECK IF HTIS IS BELOW A TOLERANCE!
    /* printf("rhoinv = %G\n ",new->rhoinv); */
    /* assert (fabs(new->rhoinv) > 1e-15); */
    
    new->iter = iter;

    if (list->size >= list->m){ // full list
        new->next = list->most_recent; // next one is going to be the most recent one
        // don't need to set prev as it is already set in first if statement
    }
    else if (list->size > 0){

        list->most_recent->prev = new;
        new->next = list->most_recent;
        list->most_recent = new;
        list->size = list->size + 1;

        // size of the list reaches maximum number kept
        if (list->size == list->m){ // set to last element
            while (head->next != NULL){
                head = head->next;
            }
            new->prev = head;
        }
    }
    else{//(list->size == 0){
        new->next = NULL;
        new->prev = NULL;
        list->size = 1;
    }
    
    // set the head
    list->most_recent = new;
    list->step = 0;    
    
}

/***********************************************************//**
   Reset stepping prior to stepping through list
***************************************************************/
void c3opt_lbfgs_reset_step(struct c3opt_lbfgs_list * list)
{
    list->step = 0;
}

/***********************************************************//**
    Step through a list, from most recent to oldest
    current location measured by list->step
    
    \param[in]     list      - list to step through
    \param[in,out] iter      - iteration of optimization
    \param[in,out] s         - x_iter+1 - x_iter
    \param[in,out] y         - g_iter+1 - g_iter
    \param[in,out] rhoinv    - s^Ty
***************************************************************/
void c3opt_lbfgs_list_step(struct c3opt_lbfgs_list * list,
                           size_t * iter, double ** s,
                           double ** y, double *rhoinv)
{
    assert (list != NULL);

    list->step = list->step+1;
    if (list->step == 1){
        list->cstep = list->most_recent;
    }
    else{
        list->cstep = list->cstep->next;
        if (list->step == list->size){
            list->step = 0;
        }
    }

    *iter = list->cstep->iter;
    *s = list->cstep->s;
    *y = list->cstep->y;
    *rhoinv = list->cstep->rhoinv;
}

/***********************************************************//**
    Step through the bfgs list, from oldest to newest
    current location measured by list->step
    
    \param[in]     list      - list to step through
    \param[in,out] iter      - iteration of optimization
    \param[in,out] s         - x_iter+1 - x_iter
    \param[in,out] y         - g_iter+1 - g_iter
    \param[in,out] rhoinv    - s^Ty
***************************************************************/
void c3opt_lbfgs_list_step_back(struct c3opt_lbfgs_list * list,
                                size_t * iter, double ** s,
                                double ** y, double *rhoinv)
{
    assert (list != NULL);

    list->step = list->step+1;

    if (list->step == 1){
        list->cstep = list->most_recent;
        for (size_t ii = 1; ii < list->size; ii++){
            list->cstep = list->cstep->next;
        }
    }
    else{
        list->cstep = list->cstep->prev;
        if (list->step == list->size){
            list->step = 0;
        }
    }

    *iter = list->cstep->iter;
    *s = list->cstep->s;
    *y = list->cstep->y;
    *rhoinv = list->cstep->rhoinv;
}

/***********************************************************//**
    Print the list                                                            
***************************************************************/
void c3opt_lbfgs_list_print(struct c3opt_lbfgs_list * list, FILE * fp, int width, int prec)
{

    if (list == NULL){
        fprintf(fp,"NULL\n");
    }
    else{
        struct c3opt_lbfgs_data * head = list->most_recent;
        for (size_t ii = 0; ii < list->size; ii++){
            c3opt_lbfgs_data_print(list->d, head,fp,width,prec);
            head = head->next;
        }
    }
}

/***********************************************************//**
    Free the list
***************************************************************/
void c3opt_lbfgs_list_free(struct c3opt_lbfgs_list * list)
{
    if (list != NULL){
        struct c3opt_lbfgs_data * current = list->most_recent;
        struct c3opt_lbfgs_data * next;
        c3opt_lbfgs_data_free_contents(current);
        current->prev = NULL;
        for (size_t ii = 1; ii < list->size; ii++){
            next = current->next;
            free(current); current=NULL;
            current = next;
            c3opt_lbfgs_data_free_contents(current);
            current->prev = NULL;
        }
        free(current); current = NULL;
        free(list); list = NULL;
    }
}


/***********************************************************//**
    Apply Hg, where H is stored as a list and Ho is the identity scaled
    by hoscale

    \param[in]     iter    - iteration
    \param[in]     m       - number of stored vectors
    \param[in]     hoscale - scale for initial Ho
    \param[in]     scale   - 0 then just hoscale, 1 then s^Ty/y^Ty
    \param[in,out] alpha   - space for evlaluation of rho s^T q (at most m)
    \param[in]     g       - object to multiply by
    \param[in,out] q       - space for evaluation of Hg (at least d)
    \param[in]     list    - bfgs list
***************************************************************/
void lbfgs_hg(size_t iter, size_t m, size_t dx, double hoscale, int scale,
              double * alpha, const double * g, double * q,
              struct c3opt_lbfgs_list * list)
{
    // H_0 is identy!!!!
    size_t incr, bound;
    if (iter <= m){
        incr = 0;
        bound = iter;
    }
    else{
        incr = iter-m;
        bound = m;
    }
    memmove(q,g,dx*sizeof(double));

    size_t jj;
    double beta;

    double rhoinv;
    size_t kk;
    double * s = NULL;/* calloc_double(dx); */
    double * y = NULL;/* calloc_double(dx); */
    c3opt_lbfgs_reset_step(list);

    double num_start = 0.0;
    double den_start = 0.0;
    for (int ii = bound-1; ii >= 0; ii--){
        jj = (size_t)ii + incr;

        c3opt_lbfgs_list_step(list,&kk,&s,&y,&rhoinv); // newest to oldest
        if ((size_t)ii == bound-1){
            den_start = cblas_ddot(dx,y,1,y,1);
            num_start = rhoinv;
            if (isnan(num_start)){
                printf("num_start is nan\n");
                exit(1);
            }
        }
        assert (kk == jj); // just to make sure
        alpha[ii] = 1.0/rhoinv * cblas_ddot(dx,s,1,q,1);
        cblas_daxpy(dx,-alpha[ii],y,1,q,1);
    }

    for (size_t zz = 0; zz < dx; zz++){
        if ((den_start > 1e-15) && (scale == 1)){
            q[zz] *= hoscale*num_start/den_start;
        }
        else{
            q[zz] *= hoscale;
        }
    }

    c3opt_lbfgs_reset_step(list);
    for (size_t ii = 0; ii < bound ;ii++){
        jj = ii + incr;
        
        c3opt_lbfgs_list_step_back(list,&kk,&s,&y,&rhoinv); // oldest to newest
        
        beta = 1.0/rhoinv * cblas_ddot(dx,y,1,q,1);
        cblas_daxpy(dx,(alpha[ii]-beta),s,1,q,1);
    }
}

/***********************************************************//**
    Low memory lbfgs

    \param[in]     opt        - optimization
    \param[in,out] x          - starting/final point
    \param[in,out] fval       - final function value
    \param[in,out] grad       - gradient at final point

    \return  0    - success
            -20+? - failure in gradient (gradient outputs ?)
             1    - maximum number of iterations reached
         other    - error in backward_line_search 
                   (see that function)
***************************************************************/
int c3_opt_lbfgs(struct c3Opt * opt,
                double * x, double * fval, double * grad)
{
    size_t d = c3opt_get_d(opt);
    double * workspace = c3opt_get_workspace(opt);
    int verbose = c3opt_get_verbose(opt);
    double * lb = c3opt_get_lb(opt);
    double * ub = c3opt_get_ub(opt);
    size_t maxiter = c3opt_get_maxiter(opt);
    double gtol = c3opt_get_gtol(opt);
    double relftol = c3opt_get_relftol(opt);
    double absxtol = c3opt_get_absxtol(opt);

    size_t m = c3opt_get_nvectors_store(opt);
    int scale = c3opt_get_lbfgs_scale(opt);
    
    opt->nevals = 0;
    opt->ngvals = 0;
    opt->niters = 1;

    *fval = c3opt_eval(opt,x,grad);
    if (opt->store_func == 1){
        opt->stored_func[opt->niters-1] = *fval;
    }
    if (opt->store_grad == 1){
        memmove(opt->stored_grad+(opt->niters-1)*opt->d,grad,d*sizeof(double));
    }
    if (opt->store_x == 1){
        memmove(opt->stored_x+(opt->niters-1)*opt->d,x,d*sizeof(double));
    }

    double grad_norm = sqrt(cblas_ddot(d,grad,1,grad,1));
    int ret = C3OPT_SUCCESS;;

    if (verbose > 0){

        printf("Initial values:\n \t (fval,||g||) = (%3.5G,%3.5G)\n",*fval,grad_norm);
        if (verbose > 1){
            printf("\t x = "); dprint(d,x);
        }
    }

    opt->prev_eval = 0.0;    
    /* size_t m = 10; */
    /* printf("nstore = %zu\n",m); */
    struct c3opt_lbfgs_list * list = c3opt_lbfgs_list_alloc(m,d);
    size_t iter = 0;
    double fvaltemp;
    int onbound = 0;
    int converged = 0;
    int res = 0;
    /* double sc = 0.0;; */
    double eta;
    double * alpha = calloc_double(m);
    memmove(workspace+d,grad,d*sizeof(double));
    for (size_t ii = 0; ii < d; ii++){ workspace[d+ii] *= -1;}
    
    enum c3opt_ls_alg alg = c3opt_ls_get_alg(opt);
    while (converged == 0){
        
        memmove(workspace,x,d*sizeof(double));
        fvaltemp = *fval;
        if (alg == BACKTRACK){
            c3opt_ls_box(opt,workspace,fvaltemp,grad,workspace+d,
                              x,fval,&res);
            c3opt_eval(opt,x,workspace+2*d);
        }
        else if (alg == STRONGWOLFE){
            memmove(workspace+2*d,grad,d*sizeof(double));
            c3opt_ls_strong_wolfe(opt,workspace,fvaltemp,
                                       workspace+2*d,workspace+d,
                                       x,fval,&res);
        }
        else if (alg == WEAKWOLFE){
            memmove(workspace+2*d,grad,d*sizeof(double));
            c3opt_ls_wolfe_bisect(opt,workspace,fvaltemp,
                                  workspace+2*d,workspace+d,
                                  x,fval,&res);
        }
        /* assert (*fval < fvaltemp); */
        assert (res > -1);

        // x is now the next point
        // workspace+2d is the new gradient
        c3opt_lbfgs_list_insert(list,iter,x,workspace,workspace+2*d,grad);

        memmove(grad,workspace+2*d,d*sizeof(double));

        opt->prev_eval = fvaltemp;
        if (opt->store_func == 1){
            opt->stored_func[opt->niters] = *fval;
        }
        if (opt->store_grad == 1){
            memmove(opt->stored_grad+opt->niters*opt->d,workspace+2*d,d*sizeof(double));
        }
        if (opt->store_x == 1){
            memmove(opt->stored_x+opt->niters*opt->d,x,d*sizeof(double));
        }


        // compute next search direction;
        lbfgs_hg(iter+1,m,d,1.0,scale,alpha,grad,workspace+d,list);
        for (size_t ii = 0; ii < d; ii++){ workspace[d+ii] *= -1;}
        eta = cblas_ddot(d,grad,1,workspace+d,1);
        onbound = 0;
        for (size_t ii = 0; ii < d; ii++){
            if ((x[ii] <= lb[ii]) || (x[ii] >= ub[ii])){
                onbound = 1;
                break;
            }
        }
        double diff = fabs(*fval - fvaltemp);
        if (fabs(fvaltemp) > 1e-10){
            diff /= fabs(fvaltemp);
        }

        double xdiff = 0.0;
        for (size_t ii = 0; ii < d; ii++){
            xdiff += pow(x[ii]-workspace[ii],2);
        }
        xdiff = sqrt(xdiff);

        
        if (onbound == 1){
            if (diff < relftol){
                ret = C3OPT_FTOL_REACHED;
                converged = 1;
            }
            else{
                if (xdiff < absxtol){
                    ret = C3OPT_XTOL_REACHED;
                    converged = 1;
                }
            }
        }
        else{
            if ( (eta*eta/2.0) < pow(gtol,2)){
                ret = C3OPT_GTOL_REACHED;
                converged = 1;
            }
            else if (diff < relftol){
                ret = C3OPT_FTOL_REACHED;
                converged = 1;
            }
            else if (xdiff < absxtol){
                ret = C3OPT_XTOL_REACHED;
                converged = 1;
            }
                
        }
        
        if (verbose > 0){
            printf("Iteration:%zu/%zu\n",iter,maxiter);
            printf("\t f(x)          = %3.5G\n",*fval);
            printf("\t |f(x)-f(x_p)| = %3.5G\n",diff);
            printf("\t |x - x_p|     = %3.5G\n",xdiff);
            printf("\t p^Tg =        = %3.5G\n",eta);
            printf("\t Onbound       = %d\n",onbound);
            if (verbose > 3){
                printf("\t x = "); dprint(d,x);
            }

        }

        opt->niters++;
        iter += 1;
        
        if (iter > maxiter){
            ret = C3OPT_MAXITER_REACHED;
            converged = 1;
        }

    }
    free(alpha); alpha = NULL;
    c3opt_lbfgs_list_free(list); list = NULL;
    return ret;
}

/***********************************************************//**
    Projected Gradient (Batch)

    \param[in]     opt        - optimization
    \param[in,out] x          - starting/final point
    \param[in,out] fval       - final function value
    \param[in,out] grad       - gradient at final point

    \return  0    - success
            -20+? - failure in gradient (gradient outputs ?)
             1    - maximum number of iterations reached
         other    - error in backward_line_search 
                   (see that function)
***************************************************************/
int c3_opt_gradient(struct c3Opt * opt,
                    double * x, double * fval, double * grad)
{

    size_t d = c3opt_get_d(opt);
    double * workspace = c3opt_get_workspace(opt);
    int verbose = c3opt_get_verbose(opt);
    double * lb = c3opt_get_lb(opt);
    double * ub = c3opt_get_ub(opt);
    size_t maxiter = c3opt_get_maxiter(opt);
    double gtol = c3opt_get_gtol(opt);
    double relftol = c3opt_get_relftol(opt);
    double absxtol = c3opt_get_absxtol(opt);
    
    opt->nevals = 0;
    opt->ngvals = 0;
    opt->niters = 1;    

    *fval = c3opt_eval(opt,x,grad);
    if (opt->store_func == 1){
        opt->stored_func[opt->niters-1] = *fval;
    }
    if (opt->store_grad == 1){
        memmove(opt->stored_grad+(opt->niters-1)*opt->d,grad,d*sizeof(double));
    }
    if (opt->store_x == 1){
        memmove(opt->stored_x+(opt->niters-1)*opt->d,x,d*sizeof(double));
    }

    double grad_norm = sqrt(cblas_ddot(d,grad,1,grad,1));    
    int ret = C3OPT_SUCCESS;;

    if (verbose > 0){

        printf("Iteration:0 (fval,||g||) = (%3.5G,%3.5G)\n",*fval,grad_norm);
        if (verbose > 1){
            printf("\t x = "); dprint(d,x);
        }
    }

    if ( grad_norm < gtol){
        return C3OPT_GTOL_REACHED;
    }

    size_t iter = 1;
    double fvaltemp;
    int onbound = 0;
    int converged = 0;
    int res = 0;
    double eta;
    opt->prev_eval = 0.0;
    enum c3opt_ls_alg alg = c3opt_ls_get_alg(opt);
    /* printf("alpha = %G\n",c3opt_ls_get_alpha(opt)); */
    while (converged == 0){
        
        memmove(workspace,x,d*sizeof(double));
        for (size_t ii = 0; ii < d; ii++){
            workspace[ii+d] = -grad[ii];
        }
        fvaltemp = *fval;

        if (alg == BACKTRACK){
            c3opt_ls_box(opt,workspace,fvaltemp,grad,
                         workspace+d,
                         x,fval,&res);
            *fval = c3opt_eval(opt,x,grad);
        }
        else if (alg == STRONGWOLFE){
            c3opt_ls_strong_wolfe(opt,workspace,fvaltemp,
                                  grad,workspace+d,
                                  x,fval,&res);
        }
        else if (alg == WEAKWOLFE){
            /* printf("call weak-wolfe, alpha=%G\n",c3opt_ls_get_alpha(opt)); */
            c3opt_ls_wolfe_bisect(opt,workspace,fvaltemp,
                                  grad,workspace+d,
                                  x,fval,&res);
        }
        

        opt->prev_eval = fvaltemp;

        if (opt->store_func == 1){
            opt->stored_func[opt->niters] = *fval;
        }
        if (opt->store_grad == 1){
            memmove(opt->stored_grad+opt->niters*opt->d,grad,d*sizeof(double));
        }
        if (opt->store_x == 1){
            memmove(opt->stored_x+opt->niters*opt->d,x,d*sizeof(double));
        }
        
        opt->niters++;
        iter += 1;
        if (iter > maxiter){
            return C3OPT_MAXITER_REACHED;
        }

        eta = cblas_ddot(d,grad,1,grad,1);
        onbound = 0;
        for (size_t ii = 0; ii < d; ii++){
            if ((x[ii] <= lb[ii]) || (x[ii] >= ub[ii])){
                onbound = 1;
                break;
            }
        }
        double diff = fabs(*fval - fvaltemp);
        if (fabs(fvaltemp) > 1e-10){
            diff /= fabs(fvaltemp);
        }

        double xdiff = 0.0;
        for (size_t ii = 0; ii < d; ii++){
            xdiff += pow(x[ii]-workspace[ii],2);
        }
        xdiff = sqrt(xdiff);
        if (onbound == 1){
            if (diff < relftol){
                ret = C3OPT_FTOL_REACHED;
                converged = 1;
            }
            else{
                if (xdiff < absxtol){
                    ret = C3OPT_XTOL_REACHED;
                    converged = 1;
                }
            }
        }
        else{
            if ( eta < pow(gtol,2)){
                ret = C3OPT_GTOL_REACHED;
                converged = 1;
            }
            else if (diff < relftol){
                ret = C3OPT_FTOL_REACHED;
                converged = 1;
            }
            else if (xdiff < absxtol){
                ret = C3OPT_XTOL_REACHED;
                converged = 1;
            }
        }
        
        if (verbose > 0){
            printf("Iteration:%zu/%zu\n",iter,maxiter);
            printf("\t f(x)          = %3.5G\n",*fval);
            printf("\t |f(x)-f(x_p)| = %3.5G\n",diff);
            printf("\t |x - x_p|     = %3.5G\n",xdiff);
            printf("\t |g| =         = %3.5G\n",eta);
            printf("\t Onbound       = %d\n",onbound);
            if (verbose > 1){
                printf("\t x = "); dprint(d,x);
            }

        }
    }
    return ret;
}










/***********************************************************//**
    Main minimization routine

    \param[in]     opt        - optimization
    \param[in,out] start       - starting/final point
    \param[in,out] minf       - final function value

    \return  0    - success
            -20+? - failure in gradient (gradient outputs ?)
             1    - maximum number of iterations reached
         other    - error in backward_line_search 
                   (see that function)
***************************************************************/
int c3opt_minimize(struct c3Opt * opt, double * start, double *minf)
{
    int res = 0;
    if (opt->alg == BFGS){
        size_t d = c3opt_get_d(opt);
        double * invhess = calloc_double(d*d);
        double * grad = calloc_double(d);
        for (size_t ii = 0; ii < d; ii++){
            invhess[ii*d+ii] = 1.0;
        }
        res = c3_opt_damp_bfgs(opt,start,minf,grad,invhess);
        free(grad); grad = NULL;
        free(invhess); invhess = NULL;
    }
    else if (opt->alg == LBFGS){
        size_t d = c3opt_get_d(opt);
        /* double * invhess = calloc_double(d*d); */
        double * grad = calloc_double(d);
        /* for (size_t ii = 0; ii < d; ii++){ */
        /*     invhess[ii*d+ii] = 1.0; */
        /* } */
        res = c3_opt_lbfgs(opt,start,minf,grad);/* ,invhess); */
        free(grad); grad = NULL;
        /* free(invhess); invhess = NULL; */
    }
    else if (opt->alg == SGD){
        res = c3_opt_sgd(opt,start,minf);
    }
    else if (opt->alg == BATCHGRAD){
        size_t d = c3opt_get_d(opt);
        double * grad = calloc_double(d);
        res = c3_opt_gradient(opt,start,minf,grad);
        free(grad); grad = NULL;
    }
    else if (opt->alg == BRUTEFORCE){
        size_t minind = 0;
        *minf = c3opt_eval(opt, opt->workspace,NULL);
        double val;
        for (size_t ii = 1; ii < opt->nlocs; ii++){
            val = c3opt_eval(opt, opt->workspace + ii*opt->d,NULL);
            if (val < *minf){
                *minf = val;
                minind = ii;
            }
        }
        for (size_t ii = 0; ii < opt->d; ii++){
            start[ii] = opt->workspace[minind*opt->d+ii];
        }
    }
    else{
        fprintf(stderr,"Unknown optimization argument %d\n",opt->alg);
        exit(1);
    }
    return res;
}


























////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////
/////////////   Old and standalone /////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////


/***********************************************************//**
    Projected Gradient damped BFGS

    \param[in]     d          - dimension
    \param[in]     lb         - lower bounds of box constraints
    \param[in]     ub         - upper bounds of box constraints
    \param[in,out] x          - starting/final point
    \param[in,out] fval       - final function value
    \param[in,out] grad       - gradient at final point
    \param[in,out] invhess    - approx hessian at start and end
                                only upper triangular part used
    \param[in,out] space      - allocated space for computation 
                                (size (3*d) array)
    \param[in]     f          - objective function
    \param[in]     fargs      - arguments to object function
    \param[in]     g          - gradient of objective function
    \param[in]     tol        - convergence tolerance for 
                                distance between iterates \f$x\f$
    \param[in]     maxiter    - maximum number of iterations
    \param[in]     maxsubiter - maximum number of backtrack
                                iterations
    \param[in]     alpha      - backtrack parameter (0,0.5)
    \param[in]     beta       - backtrack paremeter (0,1.0)
    \param[in]     verbose    - 0: np output, 1: output

    \return  0    - success
            -20+? - failure in gradient (gradient outputs ?)
             1    - maximum number of iterations reached
         other    - error in backward_line_search 
                   (see that function)
***************************************************************/
int box_damp_bfgs(size_t d, 
                  double * lb, double * ub,
                  double * x, double * fval, double * grad,
                  double * invhess,
                  double * space,
                  double (*f)(double*,void*),void * fargs,
                  int (*g)(double*,double*,void*),
                  double tol,size_t maxiter,size_t maxsubiter,
                  double alpha, double beta, int verbose)
{

    *fval = f(x,fargs);
    
    int res = g(x,grad,fargs);
    if (res != 0){
        return -20+res;
    }
    
    //   printf("grad = "); dprint(2,grad);
    //compute search direction
 
    cblas_dsymv(CblasColMajor,CblasUpper,
                d,-1.0,invhess,d,grad,1,0.0,space+d,1);
    
//    dprint(2,space+d);
    double eta = cblas_ddot(d,grad,1,space+d,1);
    if (verbose > 0){
        printf("Iteration:0 (fval,||g||) = (%3.5G,%3.5G)\n",*fval,eta);
    }
//    printf("eta = %G\n",eta);
    if ( (eta*eta/2.0) < tol){
//        printf("returning \n");
        return 0;
    }

//    printf("iverse hessian =\n");
//    dprint2d_col(2,2,invhess);

    size_t iter = 1;
    double fvaltemp;
    int onbound = 0;
    int converged = 0;
    while (converged == 0){
        memmove(space,x,d*sizeof(double));
        
        fvaltemp = *fval;
        // need a way to output alpha here
        double sc = backtrack_line_search_bc(
                        d,lb,ub,space,
                        fvaltemp,space+d,
                        x,fval,grad,alpha,
                        beta,f,fargs,maxsubiter,
                        &res);

        if (verbose == 1){
            printf("%G,%G\n",space[0],space[1]);
            printf("dir = ");dprint(d,space+d);
            printf("fold,fnew=(%G,%G)\n",fvaltemp,*fval);
            printf("grad ="); dprint(2,grad);
            printf("sc = %G\n",sc);
            printf("%G,%G\n",x[0],x[1]);
        }
        double * s = space+d;
        cblas_dscal(d,sc,s,1);

        if (res != 0){
            return res;
        }
        iter += 1;
        if (iter > maxiter){
            return 1;
        }
        
//        printf("x = "); dprint(2,x);
        res = g(x,space+2*d,fargs);
//        printf("gradient before "); dprint(2,space+2*d);
        if (res != 0){
            return -20 + res;
        }
        
        // compute difference in gradients
        cblas_daxpy(d,-1.0,grad,1,space+2*d,1); 
        double * y = space+2*d;
        
        // combute BY;
        cblas_dsymv(CblasColMajor,CblasUpper,
                    d,1.0,invhess,d,y,1,0.0,space+3*d,1);
        
        double sty = cblas_ddot(d,s,1,y,1);        
        double ytBy = cblas_ddot(d,y,1,space+3*d,1);
        if (fabs(sty) > 1e-16){

            double a1 = (sty + ytBy)/(sty * sty);
        
//        printf("s = "); dprint(2,s);
//        printf("y = "); dprint(2,y);
//        printf("By = "); dprint(2,space+3*d);
//        printf("ytBy = %G\n", ytBy);
//        printf("sty = %G\n",sty);

            // rank 1 update
            cblas_dsyr(CblasColMajor,CblasUpper,d,a1,
                       s,1,invhess, d);
        
            //      printf("hessian after first update\n");
//        dprint2d_col(2,2,invhess);

            double a2 = -1.0/sty;
            // symmetric rank 2 updatex
            cblas_dsyr2(CblasColMajor,CblasUpper,d,a2,
                        space+3*d,1,
                        space+d,1,invhess,d);
        }
//        printf("hessian after second update\n");
//        dprint2d_col(2,2,invhess);
        // get new gradient
        cblas_daxpy(d,1.0,y,1,grad,1);
        //  printf("gradient after "); dprint(2,grad);
        
        // compute next search direction;
        cblas_dsymv(CblasColMajor,CblasUpper,
                d,-1.0,invhess,d,grad,1,0.0,space+d,1);

        eta = cblas_ddot(d,grad,1,space+d,1);
        onbound = 0;
        for (size_t ii = 0; ii < d; ii++){
            if ((x[ii] <= lb[ii]) || (x[ii] >= ub[ii])){
                onbound = 1;
                break;
            }
        }
        double diff = fabs(*fval - fvaltemp);
        if (onbound == 1){
            eta = 0.0;
            for (size_t ii = 0; ii < d; ii++){
                eta += pow(x[ii]-space[ii],2);
            }
            if (eta < tol){
                converged = 1;
            }
        }
        else{
            if (diff < tol){
                converged = 1;
            }
        }
        
        
        if (verbose > 0){

            printf("Iteration:%zu\n",iter);
            printf("\t f(x)          = %3.5G\n",*fval);
            printf("\t |f(x)-f(x_p)| = %3.5G\n",diff);
            printf("\t eta =         = %3.5G\n",eta);
            printf("\t Onbound       = %d\n",onbound);
            
        }
        
    }

    return 0;
}


/***********************************************************//**
    Minimization using Newtons method

    \param[in,out] start - starting point and resulting solution
    \param[in] dim - dimension
    \param[in] step_size - usually 1.0 
    \param[in] tol  - absolute tolerance for gradient convergence
    \param[in] g  - function for gradient evaluation
    \param[in] h  - function for hessian evaluation
    \param[in] args - arguments to gradient and hessian
***************************************************************/
void
newton(double ** start, size_t dim, double step_size, double tol,
        double * (*g)(double *, void *),
        double * (*h)(double *, void *), void * args)
{
        
    int done = 0;
    double * p = calloc_double(dim);
    int * piv = calloc_int(dim);
    int one = 1;
    double diff;
    //double diffrel;
    double den;
    size_t ii;
    while (done == 0)
    {
        double * grad = g(*start,args);
        double * hess = h(*start, args);
        int info;
        dgesv_((int*)&dim,&one,hess,(int*)&dim,piv,grad,(int*)&dim, &info);
        assert(info == 0);
        
        diff = 0.0;
        //diffrel = 0.0;
        den = 0.0;
        for (ii = 0; ii < dim; ii++){
            /* printf("sol[%zu]=%G\n",ii,grad[ii]); */
            den += pow((*start)[ii],2);
            (*start)[ii] = (*start)[ii] - step_size*grad[ii];
            diff += pow(step_size*grad[ii],2);
        }
        diff = sqrt(diff);
        /*
        if (den > 1e-15)
            diffrel = diff / sqrt(den);
        else{
            diffrel = den;
        }
        */
        
        if (diff < tol){
            done = 1;
        }
        free(grad); grad = NULL;
        free(hess); hess = NULL;
    }
    free(piv); piv = NULL;
    free(p); p = NULL;
}

/***********************************************************//**
    Projected Gradient damped newton

    \param[in]     d          - dimension
    \param[in]     lb         - lower bounds of box constraints
    \param[in]     ub         - upper bounds of box constraints
    \param[in,out] x          - starting/final point
    \param[in,out] fval       - final function value
    \param[in,out] grad       - gradient at final point
    \param[in,out] hess       - hessian at final point
    \param[in,out] space      - allocated space for computation 
                                (size (2*d) array)
    \param[in]     f          - objective function
    \param[in]     fargs      - arguments to object function
    \param[in]     g          - gradient of objective function
    \param[in]     gargs      - arguments to gradient function
    \param[in]     h          - hessian 
    \param[in]     hargs      - arguments to hessian
    \param[in]     tol        - convergence tolerance for 
                                distance between iterates \f$x\f$
    \param[in]     maxiter    - maximum number of iterations
    \param[in]     maxsubiter - maximum number of backtrack
                                iterations
    \param[in]     alpha      - backtrack parameter (0,0.5)
    \param[in]     beta       - backtrack paremeter (0,1.0)
    \param[in]     verbose    - 0: np output, 1: output

    \return  0    - success
            -20+? - failure in gradient (gradient outputs ?)
            -30+? - failure in hessian (hessian outputs ?)
             1    - maximum number of iterations reached
         other    - error in backward_line_search 
                   (see that function)
    \note See Convex Optimization 
          (boyd and Vandenberghe) page 466
***************************************************************/
int box_damp_newton(size_t d, double * lb, double * ub,
                    double * x, double * fval, double * grad,
                    double * hess,
                    double * space,
                    double (*f)(double*,void*),void * fargs,
                    int (*g)(double*,double*,void*),void* gargs,
                    int (*h)(double*,double*,void*),void* hargs,
                    double tol,size_t maxiter,size_t maxsubiter,
                    double alpha, double beta, int verbose)
{
    *fval = f(x,fargs);
    
    int res = g(x,grad,gargs);
    if (res != 0){
        return -20+res;
    }
    res = h(x,hess,hargs);
    if (res !=0 ){
        return -30+res;
    }
    
//    printf("here here\n");
    int one=1,info;
    int * piv = calloc_int(d);
    memmove(space+d,grad,d*sizeof(double));
    dgesv_((int*)&d,&one,hess,(int*)&d,piv,space+d,(int*)&d,&info);
    free(piv); piv = NULL;
    assert(info == 0);

    double eta = cblas_ddot(d,grad,1,space+d,1);
    if (verbose > 0){
        printf("Iteration:0 (fval,||g||) = (%3.5G,%3.5G)\n",*fval,eta);
    }
//    printf("eta = %G\n",eta);
    if ( (eta*eta/2.0) < tol){
        return 0;
    }

    size_t iter = 1;
    double fvaltemp;
    int onbound = 0;
    int converged = 0;
    while (converged == 0){
        memmove(space,x,d*sizeof(double));
        for (size_t ii = 0; ii <d; ii++){ space[d+ii] *= -1.0;}
        
        fvaltemp = *fval;
        backtrack_line_search_bc(d,lb,ub,space,
                                 fvaltemp,space+d,
                                 x,fval,grad,alpha,
                                 beta,f,fargs,maxsubiter,
                                 &res);
//        printf("%G,%G\n",x[0],x[1]);
        
        if (res != 0){
            return res;
        }
        iter += 1;
        if (iter > maxiter){
            //          printf("Warning: max iter in newton method reached\n");
//            printf("on bound = %d\n",onbound);
            return 1;
        }
        res = g(x,grad,gargs);
        if (res != 0){
            return -20 + res;
        }
        res = h(x,hess,hargs);
        if (res != 0){
            return -30 + res;
        }

        memmove(space+d,grad,d*sizeof(double));
        piv = calloc_int(d);
        dgesv_((int*)&d,&one,hess,(int*)&d,piv,space+d,(int*)&d,&info);
        free(piv); piv = NULL;
        assert(info == 0);

        eta = cblas_ddot(d,grad,1,space+d,1);
        onbound = 0;
        for (size_t ii = 0; ii < d; ii++){
            if ((x[ii] <= lb[ii]) || (x[ii] >= ub[ii])){
                onbound = 1;
                break;
            }
        }
        if (onbound == 1){
            eta = 0.0;
            for (size_t ii = 0; ii < d; ii++){
                eta += pow(x[ii]-space[ii],2);
            }
            if (eta < tol){
                converged = 1;
            }
        }
        else{
            if ( (eta*eta/2.0) < tol){
                converged = 1;
            }
        }

        if (verbose > 0){
            printf("Iteration:%zu (fval,||g||) = (%3.5G,%3.5G)\n",iter,*fval,eta);
            printf("\t Onbound = %d\n",onbound);
        }
        
    }
    free(piv); piv = NULL;
    return 0;
}

/***********************************************************//**
    Gradient descent (with inexact backtracking)

    \param[in]     d          - dimension
    \param[in,out] x          - starting/final point
    \param[in,out] fval       - final function value
    \param[in,out] grad       - gradient at final point
    \param[in,out] space      - allocated space for computation (size (2*d) array)
    \param[in]     f          - objective function
    \param[in]     fargs      - arguments to object function
    \param[in]     g          - gradient of objective function
    \param[in]     gargs      - arguments to gradient function
    \param[in]     tol        - convergence tolerance for norm of gradient
    \param[in]     maxiter    - maximum number of iterations
    \param[in]     maxsubiter - maximum number of backtrack iterations
    \param[in]     alpha      - backtrack parameter (0,0.5)
    \param[in]     beta       - backtrack paremeter (0,1.0)
    \param[in]     verbose    - 0: np output, 1: output

    \return  0 - success
            -20+? - failure in gradient (gradient function outputs ?)
             1 - maximum number of iterations reached
         other - error in backward_line_search (see that function)
    \note See Convex Optimization (boyd and Vandenberghe) page 466
***************************************************************/
int gradient_descent(size_t d, double * x, double * fval, double * grad,
                     double * space,
                     double (*f)(double *,void*),void * fargs,
                     int (*g)(double *,double*,void*), void * gargs,
                     double tol,size_t maxiter, size_t maxsubiter,
                     double alpha, double beta, int verbose)
{
    int res = g(x,grad,gargs);
    if (res != 0){
        return -20+res;
    }
    *fval = f(x,fargs);
    double eta = sqrt(cblas_ddot(d,grad,1,grad,1));
    if (verbose > 0){
        printf("Iteration:%d (fval,||g||) = (%G,%G)\n",1,*fval,eta);
    }
    if (eta < tol){
        return 0;
    }
    size_t iter = 1;
    double fvaltemp;
    while (eta > tol){
        
        // copy current point
        memmove(space,x,d*sizeof(double));
        // create current search direction (-grad)
        memmove(space+d,grad,d*sizeof(double));
        for (size_t ii = 0; ii <d; ii++){ space[d+ii] *= -1.0;}
        
        fvaltemp = *fval;
        backtrack_line_search(d,space,fvaltemp,space+d,
                              x,fval,grad,alpha,
                              beta,f,fargs,maxsubiter,&res);

        if (res != 0){
            return res;
        }
        iter += 1;
        if (iter > maxiter){
            printf("Warning: max number of iterations (%zu) of gradient descent reached\n",iter);
            return 1;
        }
        res = g(x,grad,gargs);
        if (res != 0){
            return -20 + res;
        }
        eta = sqrt(cblas_ddot(d,grad,1,grad,1));
        if (verbose > 0){
            printf("Iteration:%zu (fval,||g||) = (%G,%G)\n",iter,*fval,eta);
        }
    }
    
    return 0;
}

/***********************************************************//**
    Box constrained Gradient descent (with inexact backtracking 
    projected gradient linesearch) 

    \param[in]     d          - dimension
    \param[in]     lb         - lower bounds
    \param[in]     ub         - upper bounds
    \param[in,out] x          - starting/final point
    \param[in,out] fval       - final function value
    \param[in,out] grad       - gradient at final point
    \param[in,out] space      - allocated space for computation (size (2*d) array)
    \param[in]     f          - objective function
    \param[in]     fargs      - arguments to object function
    \param[in]     g          - gradient of objective function
    \param[in]     gargs      - arguments to gradient function
    \param[in]     tol        - convergence tolerance for iterates
    \param[in]     maxiter    - maximum number of iterations
    \param[in]     maxsubiter - maximum number of backtrack iterations
    \param[in]     alpha      - backtrack parameter (0,0.5)
    \param[in]     beta       - backtrack paremeter (0,1.0)
    \param[in]     verbose    - 0: np output, 1: output

    \return  0 - success
            -20+? - failure in gradient (gradient function outputs ?)
             1 - maximum number of iterations reached
         other - error in backward_line_search (see that function)

***************************************************************/
int box_pg_descent(size_t d, double * lb, double * ub,
                   double * x, double * fval, double * grad,
                   double * space,
                   double (*f)(double *,void*),void * fargs,
                   int (*g)(double *,double*,void*), 
                   void * gargs,
                   double tol,size_t maxiter, size_t maxsubiter,
                   double alpha, double beta, int verbose)
{
    int res = g(x,grad,gargs);
    if (res != 0){
        return -20+res;
    }
    *fval = f(x,fargs);
    double eta = sqrt(cblas_ddot(d,grad,1,grad,1));
    if (verbose > 0){
        printf("Iteration:%d (fval,||g||) = (%G,%G)\n",1,*fval,eta);
    }
    if (eta < tol){
        return 0;
    }
    size_t iter = 1;
    double fvaltemp;
    while (eta > tol){
        
        // copy current point
        memmove(space,x,d*sizeof(double));
        // create current search direction (-grad)
        memmove(space+d,grad,d*sizeof(double));
        for (size_t ii = 0; ii <d; ii++){ space[d+ii] *= -1.0;}
        
        fvaltemp = *fval;
        backtrack_line_search_bc(d,lb,ub,space,fvaltemp,
                                 space+d,x,fval,grad,alpha,
                                 beta,f,fargs,maxsubiter,
                                 &res);

//        printf("newx = (%G,%G)\n",x[0],x[1]);

        if (res != 0){
            printf("Warning: backtrack return %d\n",res);
            return res;
        }
        iter += 1;
        if (iter > maxiter){
            printf("Warning: max number of iterations (%zu) of gradient descent reached\n",iter);
            return 1;
        }
        res = g(x,grad,gargs);
        if (res != 0){
            return -1;
        }
        
        eta = 0.0;
        for (size_t ii = 0; ii < d; ii++){
            eta += pow(x[ii]-space[ii],2);
        }
        eta = sqrt(eta);
        if (verbose > 0){
            printf("Iteration:%zu (fval,||x_k-x_k-1||) = (%G,%G)\n",iter,*fval,eta);
        }
    }
    
    return 0;
}


/***********************************************************//**
    Backtracking Line search                                                           

    \param[in]     d       - dimension
    \param[in]     x       - base point
    \param[in]     fx      - objective function value at x
    \param[in]     p       - search direction
    \param[in,out] newx    - new location (x + t*p)
    \param[in,out] fnx     - objective function value f(newx)
    \param[in]     grad    - gradient at x
    \param[in]     alpha   - specifies accepted decrease (0,0.5) typically (0.01 - 0.3)
    \param[in]     beta    - (0,1) typically (0.1, 0.8) corresponding to (crude search, less crude)
    \param[in]     f       - objective function
    \param[in]     fargs   - arguments to objective function
    \param[in]     maxiter - maximum number of iterations
    \param[in,out] info    - 0 - success
                            -1 - alpha not within correct bounds
                            -2 - beta not within correct bounds
                             1 - maximum number of iter. reached
                             
    \return final scaling of search direction
    \note See Convex Optimization (boyd and Vandenberghe) page 464
***************************************************************/
double backtrack_line_search(size_t d, double * x, double fx, 
                          double * p, double * newx, 
                          double * fnx, double * grad, 
                          double alpha, double beta, 
                          double (*f)(double * x, void * args), 
                          void * fargs, size_t maxiter, 
                          int *info)
{
    *info = 0;
    if ((alpha <= 0.0) || (alpha >= 0.5)){
        printf("line search alpha (%G) is not (0,0.5)\n",alpha);
        *info = -1;
        return 0.0;
    }
    if ((beta <= 0.0) && (beta >= 1.0)) {
        printf("line search beta (%G) is not (0,1)\n",beta);
        *info = -2;
        return 0.0;
    }

    double t = 1.0;
    memmove(newx,x,d*sizeof(double));

    size_t iter = 1;
    
    double dg = cblas_ddot(d,grad,1,p,1);
    double checkval = fx + alpha*t*dg;
    cblas_daxpy(d,t,p,1,newx,1);

    *fnx = f(newx,fargs);
    iter += 1;
    
//    printf("newx (%G,%G)\n",newx[0],newx[1]);
    while (*fnx > checkval){
        if (iter >= maxiter){
            *info = 1;
            printf("Warning: maximum number of iterations (%zu) of line search reached\n",iter);
            break;
        }
        t = beta * t;

        checkval = fx + alpha*t*dg;
       
        memmove(newx,x,d*sizeof(double));
        cblas_daxpy(d,t,p,1,newx,1);
        *fnx = f(newx,fargs);
        iter += 1 ;
    }
    
    return t;
}




/***********************************************************//**
    Backtracking Line search with projection onto box constraints

    \param[in]     d       - dimension
    \param[in]     lb      - lower bounds for each dimension
    \param[in]     ub      - upper bounds for each dimension
    \param[in]     x       - base point
    \param[in]     fx      - objective function value at x
    \param[in]     p       - search direction
    \param[in,out] newx    - new location (x + t*p)
    \param[in,out] fnx     - objective function value f(newx)
    \param[in]     grad    - gradient at x
    \param[in]     alpha   - specifies accepted decrease (0,0.5) typically (0.01 - 0.3)
    \param[in]     beta    - (0,1) typically (0.1, 0.8) corresponding to (crude search, less crude)
    \param[in]     f       - objective function
    \param[in]     fargs   - arguments to objective function
    \param[in]     maxiter - maximum number of iterations
    \param[in,out] info    -  0 - success
                             -1 - alpha not within correct bounds
                             -2 - beta not within correct bounds
                              1 - maximum number of iter reached
    
    \return line search gradient scaling
    \note See Convex Optimization (boyd and Vandenberghe) page 464
***************************************************************/
double backtrack_line_search_bc(size_t d, double * lb, 
                                double * ub,
                                double * x, double fx, 
                                double * p, double * newx, 
                                double * fnx,
                                double * grad, double alpha, 
                                double beta, 
                                double (*f)(double *, void *), 
                                void * fargs, 
                                size_t maxiter, int *info)
{
    *info = 0;
    if ((alpha <= 0.0) || (alpha >= 0.5)){
        printf("line search alpha (%G) is not (0,0.5)\n",alpha);
        *info = -1;
        return 0.0;
    }
    if ((beta <= 0.0) && (beta >= 1.0)) {
        printf("line search beta (%G) is not (0,1)\n",beta);
        *info = -2;
        return 0.0;
    }
    for (size_t ii = 0; ii < d; ii++){
        if (x[ii] < lb[ii]){
            printf("line search starting point violates constraints\n");
            *info = -3;
            return 0.0;
        }
        if (x[ii] > ub[ii]){
            printf("line search starting point violates constraints\n");
            *info = -3;
            return 0.0;
        }
    }

    double t = 1.0;
    memmove(newx,x,d*sizeof(double));

    size_t iter = 1;
    
    double dg = cblas_ddot(d,grad,1,p,1);
//    printf("dg = %G\n",dg);
    double checkval = fx + alpha*t*dg;
    //printf("checkval = %G,fx=%Galpha=%G,t=%G\n",checkval,fx,alpha,t);
    /* printf("beta = %G\n",beta); */
    /* printf("t = %G\n", t); */
    /* printf("alpha = %G\n", alpha); */
    /* printf("dg = %G\n", dg); */
    /* printf("checkval = %G\n",checkval); */
    
    cblas_daxpy(d,t,p,1,newx,1);

    for (size_t ii = 0; ii < d; ii++){
        if (newx[ii] < lb[ii]){
            newx[ii] = lb[ii];
        }
        if (newx[ii] > ub[ii]){
            newx[ii] = ub[ii];
        }
    }
//    printf("nn2 = "); dprint(d,newx);
    *fnx = f(newx,fargs);
//    printf("fx2 = %G\n",*fnx);
    iter += 1;
    
    //printf("newx (%G,%G)\n",newx[0],newx[1]);
    while (*fnx > checkval){
        if (iter >= maxiter){
            *info = 1;
            printf("Warning: maximum number of iterations (%zu) of line search reached\n",iter);
            break;
        }
        t = beta * t;
        
        
        checkval = fx + alpha*t*dg;
        //printf("checkval = %G,fx=%Galpha=%G,t=%G\n",checkval,fx,alpha,t);
        /* printf("beta = %G\n",beta); */
        /* printf("t = %G\n", t); */
        /* printf("alpha = %G\n", alpha); */
        /* printf("dg = %G\n", dg); */
        /* printf("checkval = %G\n",checkval); */
        memmove(newx,x,d*sizeof(double));

        cblas_daxpy(d,t,p,1,newx,1);
//        printf("nn2a = "); dprint(d,newx);
        
        for (size_t ii = 0; ii < d; ii++){
            if (newx[ii] < lb[ii]){
                newx[ii] = lb[ii];
            }
            if (newx[ii] > ub[ii]){
                newx[ii] = ub[ii];
            }
        }
//        printf("nn2 = "); dprint(d,newx);
        *fnx = f(newx,fargs);
//        printf("fx2 = %G\n",*fnx);
        iter += 1 ;
    }
    
    return t*alpha;
}



