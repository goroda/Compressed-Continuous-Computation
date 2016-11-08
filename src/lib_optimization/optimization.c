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
        ls->alpha = 0.1;
        ls->beta = 0.9;
    }
    else{
        printf("Line search algorithm, %d, is not recognized\n",alg);
        exit(1);
    }
        
    
    ls->maxiter = 10000;

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

///////////////////////////////////////////////////////
struct c3Opt
{
    enum c3opt_alg alg;
    size_t d;

    double (*f)(size_t, double *, double *, void *);
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

};

/***********************************************************//**
    Allocate optimization algorithm
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
    opt->maxiter = 1000;
    opt->relxtol = 1e-8;
    opt->absxtol = 1e-8;
    opt->relftol = 1e-8;
    opt->gtol = 1e-12;

    if (alg == BFGS){
        opt->grad = 1;
        opt->workspace = calloc_double(4*d);
        /* opt->ls = c3ls_alloc(BACKTRACK,0); */
        opt->ls = c3ls_alloc(STRONGWOLFE,0);
    }
    else if (alg==BATCHGRAD){
        opt->grad = 1;
        opt->workspace = calloc_double(2*d);
        opt->ls = c3ls_alloc(BACKTRACK,1);
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
    if ((opt->alg == BFGS) || (opt->alg == BATCHGRAD)){
        opt->grad = 1;
        memmove(opt->workspace,old->workspace,4*opt->d*sizeof(double));
        c3ls_free(opt->ls); opt->ls = NULL;
        opt->ls = c3ls_copy(old->ls);
    }
    else if (opt->alg == BRUTEFORCE)
    {
        opt->workspace = calloc_double(opt->nlocs * opt->d);
        memmove(opt->workspace,old->workspace,opt->nlocs*opt->d*sizeof(double));
    }

    opt->nevals = old->nevals;
    opt->ngvals = old->ngvals;
    opt->niters = old->niters;

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
        free(opt); opt = NULL;
    }
}

/***********************************************************//**
    1 if bruteforce, 0 otherwise
***************************************************************/
int c3opt_is_bruteforce(struct c3Opt * opt)
{
    assert (opt != NULL);
    if (opt->alg == BRUTEFORCE){
        return 1;
    }
    return 0;
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

size_t c3opt_get_maxiter(struct c3Opt * opt)
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
                         double(*f)(size_t,double *,double *,void *),
                         void * farg)
{
    assert (opt != NULL);
    opt->f = f;
    opt->farg = farg;
}

/***********************************************************//**
    Evaluate the objective function
***************************************************************/
double c3opt_eval(struct c3Opt * opt, double * x, double * grad)
{
    assert (opt != NULL);
    assert (opt->f != NULL);
    double out = opt->f(opt->d,x,grad,opt->farg);
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
    
    c3opt_eval(opt,(double *)x,grad);

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
    
    c3opt_eval(opt,(double *)x,grad);

    double diff = 0.0;
    double v1,v2;
    double norm = 0.0;
    for (size_t ii = 0; ii < dim; ii++){
        x1[ii] += eps;
        x2[ii] -= eps;
        v1 = c3opt_eval(opt,x1,NULL);
        v2 = c3opt_eval(opt,x2,NULL);
        diff += pow( (v1-v2)/2.0/eps - grad[ii], 2 );
        diffs[ii] =(v1-v2)/2.0/eps - grad[ii];
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
    Zoom function for Strong Wolfe
***************************************************************/
double c3opt_ls_zoom(struct c3Opt * opt, double tlow, double flow,
                     double thigh, const double * x, double * dir,
                     double * newx, double fx, double dg, double * grad,
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
    while (fabs(th-tl) >= 1e-60){
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
            printf("\t dg2=%G x=",dg2); dprint(d,newx);
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
    Projected Gradient damped BFGS with active sets

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

    \note  Domgmin Kim 2010 (with suvrit sra)
***************************************************************/


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
                     double * x, double * fval, double * grad,
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
    
    cblas_dsymv(CblasColMajor,CblasUpper,
                d,-1.0,invhess,d,grad,1,0.0,workspace+d,1);

    /* printf("grad ="); */
    /* dprint(2,grad); */
    /* printf("x = "); */
    /* dprint(2,x); */
    /* printf("lb = "); dprint(2,lb); */
    /* printf("ub = "); dprint(2,ub); */

    int ret = C3OPT_SUCCESS;;
    double eta = cblas_ddot(d,grad,1,workspace+d,1);

    if (verbose > 0){
        printf("Iteration:0 (fval,||g||) = (%3.5G,%3.5G)\n",*fval,eta*eta/2.0);
        if (verbose > 1){
            printf("\t x = "); dprint(d,x);
        }
    }

    if ( (eta*eta/2.0) < gtol){
        return C3OPT_GTOL_REACHED;
    }

    size_t iter = 1;
    double fvaltemp;
    int onbound = 0;
    int converged = 0;
    int res = 0;
    double sc = 0.0;;
    opt->prev_eval = 0.0;
    enum c3opt_ls_alg alg = c3opt_ls_get_alg(opt);
    while (converged == 0){
        
        memmove(workspace,x,d*sizeof(double));
        fvaltemp = *fval;
        if (alg == BACKTRACK){
            sc = c3opt_ls_box(opt,workspace,fvaltemp,grad,workspace+d,
                              x,fval,&res);
            c3opt_eval(opt,x,workspace+2*d);

            /* printf("x = "); dprint(d, x); */
            /* printf("fvaltemp = %G\n",fvaltemp); */
            /* printf("Grad = "); dprint(d,grad); */
            /* printf("dir = "); dprint(d,workspace+d); */
            /* printf("sc = %G\n",sc); */
            /* assert (*fval < fvaltemp); */
        }
        else if (alg == STRONGWOLFE){
            memmove(workspace+2*d,grad,d*sizeof(double));
            sc = c3opt_ls_strong_wolfe(opt,workspace,fvaltemp,
                                       workspace+2*d,workspace+d,
                                       x,fval,&res);

            /* printf("x = "); dprint(d, x); */
            /* printf("fvaltemp = %G\n",fvaltemp); */
            /* printf("Grad = "); dprint(d,grad); */
            /* printf("dir = "); dprint(d,workspace+d); */
            /* printf("sc = %G\n",sc); */
            assert (*fval < fvaltemp);
            /* double * gtest = calloc_double(d); */
            /* /\* double fshould = c3opt_eval(opt,x,gtest); *\/ */
            /* /\* printf("g is"); dprint(d,workspace+2*d); *\/ */
            /* /\* printf("g should"); dprint(d, gtest); *\/ */
            /* /\* printf("f is %G\n", *fval); *\/ */
            /* /\* printf("f should %G\n", fshould); *\/ */

            /* if (fabs(*fval-fshould) > 1e-20){ */
            /*     exit(1); */
            /* } */
        }

        opt->prev_eval = fvaltemp;
        double * s = workspace+d;
        cblas_dscal(d,sc,s,1);

        opt->niters++;
        iter += 1;
        if (iter > maxiter){
            //printf("iter = %zu,verbose=%d\n",iter,verbose);
            return C3OPT_MAXITER_REACHED;
        }

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
                        workspace+d,1,invhess,d);
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
        if (fabs(fvaltemp) > 1e-10){
            diff /= fabs(fvaltemp);
        }

        double xdiff = 0.0;
        for (size_t ii = 0; ii < d; ii++){
            xdiff += pow(x[ii]-workspace[ii],2);
        }
        xdiff = sqrt(xdiff);
        if (onbound == 1){
            //printf("grad = ");
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
            printf("\t eta =         = %3.5G\n",eta);
            printf("\t Onbound       = %d\n",onbound);
            if (verbose > 1){
                printf("\t x = "); dprint(d,x);
            }

        }
        
    }

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
    double sc, eta;
    opt->prev_eval = 0.0;
    while (converged == 0){
        
        memmove(workspace,x,d*sizeof(double));
        for (size_t ii = 0; ii < d; ii++){
            workspace[ii+d] = -grad[ii];
        }
        fvaltemp = *fval;

        sc = c3opt_ls_box(opt,workspace,fvaltemp,grad,
                          workspace+d,
                          x,fval,&res);

        /* printf("gradient is "); dprint(d,grad); */
        opt->prev_eval = fvaltemp;
        
        /* cblas_daxpy(d,-1e-3,grad,1,x,1); */
        
        opt->niters++;
        iter += 1;
        if (iter > maxiter){
            return C3OPT_MAXITER_REACHED;
        }

        *fval = c3opt_eval(opt,x,grad);

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
    size_t * piv = calloc_size_t(dim);
    size_t one = 1;
    double diff;
    //double diffrel;
    double den;
    size_t ii;
    while (done == 0)
    {
        double * grad = g(*start,args);
        double * hess = h(*start, args);
        int info;
        dgesv_(&dim,&one,hess,&dim,piv,grad,&dim, &info);
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
    size_t * piv = calloc_size_t(d);
    memmove(space+d,grad,d*sizeof(double));
    dgesv_(&d,&one,hess,&d,piv,space+d,&d,&info);
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
        piv = calloc_size_t(d);
        dgesv_(&d,&one,hess,&d,piv,space+d,&d,&info);
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



