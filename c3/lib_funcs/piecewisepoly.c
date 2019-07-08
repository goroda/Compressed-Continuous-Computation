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


/** \file piecewisepoly.c
 * Provides routines for using piecewise polynomials
*/

#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "stringmanip.h"
#include "array.h"
#include "futil.h"
#include "polynomials.h"
#include "piecewisepoly.h"
#include "linalg.h"

//#define ZEROTHRESH  2e0 * DBL_EPSILON

struct PwCouple
{
    const struct PiecewisePoly * a;
    const struct PiecewisePoly * b;
    double coeff[2];
};

int pw_eval(size_t N, const double * x, double * out, void * args){
    struct PiecewisePoly * pw = args;
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = piecewise_poly_eval(pw,x[ii]);
    }
    return 0;
}

int pw_eval_prod(size_t N, const double * x, double * out, void * args)
{
    struct PwCouple * c = args;
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = piecewise_poly_eval(c->a,x[ii]) * 
                  piecewise_poly_eval(c->b,x[ii]);
    }

    return 0;
}

double pw_eval_ope(double x, void * args){
    
    struct OrthPolyExpansion * ope = args;
    double out = orth_poly_expansion_eval(ope,x);
    //printf("(%G,%G)\n",x,out);
    return out;
}

int pw_eval_sum(size_t N,const double * x, double * out, void * args)
{
    struct PwCouple * c = args;
 
    if (c->b == NULL){
        assert (c->a != NULL);
        for (size_t ii = 0; ii < N; ii++){
            out[ii] = c->coeff[0] * piecewise_poly_eval(c->a,x[ii]);
        }
    }
    else if (c->a == NULL){
        assert (c->b != NULL);
        for (size_t ii = 0; ii < N; ii++){
            out[ii] = c->coeff[0] * piecewise_poly_eval(c->b,x[ii]);
        }
    }
    else{
        for (size_t ii = 0; ii < N; ii++){
            out[ii] = c->coeff[0] * piecewise_poly_eval(c->a,x[ii]) +
                      c->coeff[1] * piecewise_poly_eval(c->b,x[ii]);
        }
    }
    
    return 0;
}

double pw_eval_neighbor(double x, void * args){
    
    struct PwCouple * c = args;
    double split = piecewise_poly_get_ub(c->a);
    double out;
    if (x <= split){
        out = piecewise_poly_eval(c->a,x);
    }
    else{
        out = piecewise_poly_eval(c->b,x);
    }
    return out;
}

void solveLin(double * x, double * y, double * coeff)
{
    double den = x[0]-x[1];
    coeff[0] = (y[0] - y[1]) / den; // slope
    coeff[1] = (x[0]*y[1] - x[1]*y[0]) /den; // offset;
}

void solveQuad(double * x, double * y, double * coeff){
    
    // high power to lower power
    double den = pow(x[0],2)*(x[1] - x[2]) - 
                 pow(x[1],2)*(x[0] - x[2]) +
                 pow(x[2],2)*(x[0] - x[1]);

    coeff[0] = (y[0] * (x[1] - x[2]) - y[1] * (x[0] - x[2]) + y[2] * (x[0] - x[1]))/den;
    coeff[1] = (pow(x[0],2) * (y[1] - y[2]) - pow(x[1],2) * (y[0] - y[2]) + pow(x[2],2) * (y[0] - y[1]))/den;
    coeff[2] = (pow(x[0],2) * (x[1]*y[2] - x[2]*y[1]) - pow(x[1],2) * (x[0]*y[2] - x[2]*y[0]) + pow(x[2],2) * (x[0]*y[1] - x[1]*y[0]))/den;
}

double pw_eval_lin_func(double x, void * args)
{   
    double * coeff = args;
    double out = coeff[0]*x + coeff[1];
    return out;
}

double pw_eval_quad_func(double x, void * args)
{   
    double * coeff = args;
    double out = coeff[0] * pow(x,2) + coeff[1] * x + coeff[2];
    return out;
}

struct PwPolyOpts
{
    enum poly_type ptype;
    double lb;
    double ub;

    size_t maxorder;
    double minsize;  // I think these can conflict
    size_t coeff_check;
    double epsilon;
    size_t nregions;
    
    int pts_alloc;
    size_t npts;
    double * pts; // could be null then evenly spaced out, (nregions+1,)


    struct OpeOpts * opeopts;
    void * other;
};

struct PwPolyOpts * pw_poly_opts_alloc(enum poly_type ptype, double lb, double ub)
{
    struct PwPolyOpts * ao;
    if ( NULL == (ao = malloc(sizeof(struct PwPolyOpts)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }

    ao->ptype = ptype;
    ao->lb = lb;
    ao->ub = ub;

    ao->maxorder = 10;
    ao->minsize = 1e-2;
    ao->coeff_check = 3;
    ao->epsilon = 1e-15;
    ao->nregions = 5; // number of regions to adapt


    ao->pts_alloc = 0;
    ao->npts = 0;
    ao->pts = NULL;

    ao->opeopts = NULL;
    /* ao->opeopts = ope_opts_alloc(ptype); */
    /* size_t startnum = 3; */
    /* if ((ao->maxorder+1) < startnum){ */
    /*     startnum = ao->maxorder+1; */
    /* } */
    /* ope_opts_set_start(ao->opeopts,startnum); */
    ao->other = NULL;
    
    return ao;
}

void pw_poly_opts_free(struct PwPolyOpts * pw)
{
    if (pw != NULL){
        if (pw->pts_alloc == 1){
            free(pw->pts); pw->pts = NULL;
        }
        ope_opts_free(pw->opeopts); pw->opeopts = NULL;
        free(pw); pw = NULL;
    }
}

void pw_poly_opts_free_deep(struct PwPolyOpts ** pw)
{
    if ((*pw) != NULL){
        if ((*pw)->pts_alloc == 1){
            if ((*pw)->pts != NULL){
                free((*pw)->pts); (*pw)->pts = NULL;
            }
        }
        if ((*pw)->opeopts != NULL){
            ope_opts_free((*pw)->opeopts); (*pw)->opeopts = NULL;
        }
        /* free(*pw); *pw = NULL; */
    }
}

void pw_poly_opts_set_ptype(struct PwPolyOpts * pw, enum poly_type ptype)
{
    assert (pw != NULL);
    pw->ptype = ptype;
    if (pw->opeopts != NULL){
        ope_opts_set_ptype(pw->opeopts,ptype);
    }
}

enum poly_type pw_poly_opts_get_ptype(const struct PwPolyOpts * pw)
{
    assert (pw != NULL);
    return pw->ptype;
}

void pw_poly_opts_set_lb(struct PwPolyOpts * pw, double lb)
{
    assert (pw != NULL);
    pw->lb = lb;
}

double pw_poly_opts_get_lb(const struct PwPolyOpts * pw)
{
    assert (pw != NULL);
    return pw->lb;
}

void pw_poly_opts_set_ub(struct PwPolyOpts * pw, double ub)
{
    assert (pw != NULL);
    pw->ub = ub;
}

double pw_poly_opts_get_ub(const struct PwPolyOpts * pw)
{
    assert (pw != NULL);
    return pw->ub;
}

void pw_poly_opts_set_minsize(struct PwPolyOpts * pw, double minsize)
{
    assert (pw != NULL);
    pw->minsize = minsize;
}

void pw_poly_opts_set_maxorder(struct PwPolyOpts * pw, size_t maxorder)
{
    assert (pw != NULL);
    pw->maxorder = maxorder;
    if (pw->opeopts != NULL){
        size_t startnum = 3;
        if ((pw->maxorder+1) < startnum){
            startnum = pw->maxorder+1;
        }
        ope_opts_set_start(pw->opeopts,startnum);
    }
}

void pw_poly_opts_set_nregions(struct PwPolyOpts * pw, size_t nregions)
{
    assert (pw != NULL);
    pw->nregions = nregions;
}

void pw_poly_opts_set_pts(struct PwPolyOpts * pw, size_t N, double * pts)
{
    assert (pw != NULL);
    if (pw->pts_alloc == 1){
        free(pw->pts); pw->pts = NULL;
    }
    pw->pts_alloc = 0;
    pw->npts = N;
    pw->pts = pts;
}

void pw_poly_opts_set_coeffs_check(struct PwPolyOpts * pw, size_t num)
{
    assert (pw != NULL);
    pw->coeff_check = num;
}

void pw_poly_opts_set_tol(struct PwPolyOpts * pw, double tol)
{
    assert (pw != NULL);
    pw->epsilon = tol;
}


/********************************************************//**
*   Get number of free parameters
*************************************************************/
size_t pw_poly_opts_get_nparams(const struct PwPolyOpts* opts)
{
    (void)(opts);
    NOT_IMPLEMENTED_MSG("pw_poly_opts_get_nparams")
    return 0;
}

/********************************************************//**
*   Set number of free parameters
*************************************************************/
void pw_poly_opts_set_nparams(struct PwPolyOpts* opts, size_t num)
{
    (void)(opts);
    (void)(num);
    NOT_IMPLEMENTED_MSG("pw_poly_opts_set_nparams")
}


//////////////////////////////////////////////////////////////////

/********************************************************//**
*   Allocate memory for a piecewise polynomial
*
*   \return polynomial
*************************************************************/
struct PiecewisePoly *
piecewise_poly_alloc()
{
    struct PiecewisePoly * p;
    if ( NULL == (p = malloc(sizeof(struct PiecewisePoly)))){
        fprintf(stderr,"failed to allocate memory for piecewise polynomial.\n");
        exit(1);
    }
    
    p->nbranches = 0;
    p->leaf = 1;
    p->ope = NULL;
    p->branches = NULL;
    return p;
}

/********************************************************//**
*   Allocate memory for a piecewise polynomial array
*   
*   \param[in] size - size of array
*
*   \return p - pw poly array filled with nulls
*************************************************************/
struct PiecewisePoly **
piecewise_poly_array_alloc(size_t size)
{
    struct PiecewisePoly ** p;
    if ( NULL == (p = malloc(size*sizeof(struct PiecewisePoly *)))){
        fprintf(stderr,"failed to allocate memory for \
                        piecewise polynomial.\n");
        exit(1);
    }
    size_t ii; 
    for (ii = 0; ii < size; ii++){
        p[ii] = NULL;
    }
    return p;
}

static void piecewise_poly_copy_inside(const struct PiecewisePoly * src, struct PiecewisePoly * dst)
{
    if (src->leaf == 1){
        dst->ope = orth_poly_expansion_copy(src->ope);
        dst->leaf = 1;
        dst->nbranches = 0;
        dst->branches = NULL;
    }
    else{
        dst->leaf = 0;
        dst->nbranches = src->nbranches;
        dst->ope = NULL;
        dst->branches = piecewise_poly_array_alloc(dst->nbranches);
        size_t ii;
        for (ii = 0; ii < dst->nbranches; ii++){
            dst->branches[ii] = piecewise_poly_copy(src->branches[ii]);
        }
    }
}


/********************************************************//**
*   Copy a piecewise polynomial
*   
*   \param[in] a - pw polynomial to copy
*
*   \return pw polynomial
*************************************************************/
struct PiecewisePoly *
piecewise_poly_copy(const struct PiecewisePoly * a)
{
    if ( a != NULL ){
        struct PiecewisePoly * p = piecewise_poly_alloc();
        piecewise_poly_copy_inside(a, p);
        return p;
    }
    return NULL;
}

static void piecewise_poly_free_inside(struct PiecewisePoly * poly)
{
    if (poly->leaf == 1){
        orth_poly_expansion_free(poly->ope);
        poly->ope = NULL;
    }
    else{
        size_t ii;
        for (ii = 0; ii < poly->nbranches; ii++){
            piecewise_poly_free(poly->branches[ii]);
            poly->branches[ii] = NULL;
        }
        free(poly->branches);
        poly->branches = NULL;
    }
}

/********************************************************//**
*   Free memory allocated for piecewise polynomial
*
*   \param[in,out] poly - polynomial to free
*************************************************************/
void
piecewise_poly_free(struct PiecewisePoly * poly){
    
    if (poly != NULL)
    {   
        piecewise_poly_free_inside(poly);
        free(poly);
        poly = NULL;
    }
}

/********************************************************//**
*   Free memory allocated for piecewise polynomial array
*
*   \param[in,out] poly - polynomial to free
*   \param[in]     n    - number of elements in the array
*
*************************************************************/
void piecewise_poly_array_free(struct PiecewisePoly ** poly, size_t n)
{
    
    if (poly != NULL)
    {   
        size_t ii;
        for (ii = 0; ii < n; ii++){
            piecewise_poly_free(poly[ii]);
            poly[ii] = NULL;
        }
        free(poly);
        poly = NULL;
    }
}

// some getters and setters
/********************************************************//**
*   Get poly type
*************************************************************/
enum poly_type piecewise_poly_get_ptype(const struct PiecewisePoly * pw)
{
    if (pw->leaf != 1){
        return piecewise_poly_get_ptype(pw->branches[0]);
    }
    else{
        return orth_poly_expansion_get_ptype(pw->ope);
    }
}

struct PiecewisePoly *
piecewise_poly_genorder(size_t order, struct PwPolyOpts * opts)
{
    /* printf("we are in genorder\n"); */
    assert (opts != NULL);
    struct PiecewisePoly * p = piecewise_poly_alloc();

    // only things necessary for const
    struct OpeOpts * ope = ope_opts_alloc(opts->ptype);
    /* printf("(lb,ub) = (%G,%G)\n",opts->lb,opts->ub); */
    ope_opts_set_lb(ope,opts->lb);
    ope_opts_set_ub(ope,opts->ub);

    p->ope = orth_poly_expansion_genorder(order,ope);
    ope_opts_free(ope); ope = NULL;
    return p;
}

/********************************************************//**
*   Construct a piecewise constant function
*
*   \param[in] value - value of the function
*   \param[in] opts  - options
*
*   \return p - piecewise polynomial of one interval
*************************************************************/
struct PiecewisePoly *
piecewise_poly_constant(double value, struct PwPolyOpts * opts)
{
    assert (opts != NULL);
    struct PiecewisePoly * p = piecewise_poly_alloc();

    // only things necessary for const
    struct OpeOpts * ope = ope_opts_alloc(opts->ptype);
    ope_opts_set_lb(ope,opts->lb);
    ope_opts_set_ub(ope,opts->ub);

    p->ope = orth_poly_expansion_constant(value,ope);
    ope_opts_free(ope); ope = NULL;
    return p;
}

/********************************************************//**
    Return a zero function

    \param[in] opts         - extra arguments depending on function_class, sub_type, etc.
    \param[in] force_nparam - if == 1 then approximation will have the number of parameters
                                      defined by *get_nparams, for each approximation type
                              if == 0 then it may be more compressed

    \return p - zero function
************************************************************/
struct PiecewisePoly * 
piecewise_poly_zero(struct PwPolyOpts * opts, int force_nparam)
{
    (void)(opts);
    (void)(force_nparam);
    NOT_IMPLEMENTED_MSG("piecewise_poly_zero")
    return NULL;
}

/********************************************************//**
*   Construct a piecewise linear function
*
*   \param[in] slope  - value of the slope function
*   \param[in] offset - offset
*   \param[in] opts   - upper bound
*
*   \return p - piecewise polynomial of one interval
*************************************************************/
struct PiecewisePoly *
piecewise_poly_linear(double slope, double offset, struct PwPolyOpts * opts)
{
    struct PiecewisePoly * p = piecewise_poly_alloc();
    struct OpeOpts * ope = ope_opts_alloc(opts->ptype);
    ope_opts_set_lb(ope,opts->lb);
    ope_opts_set_ub(ope,opts->ub);

    p->ope = orth_poly_expansion_linear(slope, offset,ope);
    ope_opts_free(ope); ope = NULL;
    return p;
}

/*******************************************************//**
    Update a linear function

    \param[in] f      - existing linear function
    \param[in] a      - slope of the function
    \param[in] offset - offset of the function

    \returns 0 if successfull, 1 otherwise                   
***********************************************************/
int
piecewise_poly_linear_update(struct PiecewisePoly * f,
                             double a, double offset)
{
    (void) f;
    (void) a;
    (void) offset;
    NOT_IMPLEMENTED_MSG("piecewise_poly_linear_update");
    return 1;
}

/* /\********************************************************\//\** */
/* *   Construct a piecewise quadratic function \f$ ax^2 + bx + c \f$ */
/* * */
/* *   \param[in] a    - coefficient of squared term */
/* *   \param[in] b    - coefficient of linear term */
/* *   \param[in] c    - constant term */
/* *   \param[in] opts - opts */
/* * */
/* *   \return p - piecewise polynomial of one interval */
/* *************************************************************\/ */
/* struct PiecewisePoly * */
/* piecewise_poly_quadratic(double a, double b, double c, struct PwPolyOpts * opts) */
/* { */
    
/*     struct PiecewisePoly * p = piecewise_poly_alloc(); */
/*     double coeff[3]; */
/*     coeff[0] = a; coeff[1] = b; coeff[2] = c; */

/*     p->ope = orth_poly_expansion_init(opts->ptype, 3, opts->lb, opts->ub); */
/*     orth_poly_expansion_approx(&pw_eval_quad_func, coeff, p->ope); */
/*     orth_poly_expansion_round(&(p->ope)); */
/*     return p; */
/* } */

/********************************************************//**
*   Generate a quadratic piecewise polynomial expansion
    a * (x-offset)^2
*
*   \param[in] a      - value of the slope function
*   \param[in] offset - offset
*   \param[in] opts   - options
*
*   \return quadratic polynomial
*************************************************************/
struct PiecewisePoly * piecewise_poly_quadratic(double a, double offset, struct PwPolyOpts * opts)
{
    struct PiecewisePoly * p = piecewise_poly_alloc();
    double coeff[3];
    coeff[0] = a; coeff[1] = -2*a*offset; coeff[2] = a*offset*offset;
    p->ope = orth_poly_expansion_init(opts->ptype, 3, opts->lb, opts->ub);
    orth_poly_expansion_approx(&pw_eval_quad_func, coeff, p->ope);
    orth_poly_expansion_round(&(p->ope));
    return p;
}

/********************************************************//**
*   Split a pw-poly leaf into two parts
*
*   \param[in] pw  - polynomial to split
*   \param[in] loc - location to split a
*
*************************************************************/
void
orth_poly_expansion_split(struct PiecewisePoly * pw, double loc)
{
    assert (pw->leaf == 1);
    struct OrthPolyExpansion * ope = pw->ope;

    double lb = ope->lower_bound;
    double ub = ope->upper_bound;
    assert (loc > lb);
    assert (loc < ub);

    pw->leaf = 0;
    pw->nbranches = 2;
    pw->branches = piecewise_poly_array_alloc(2);

    enum poly_type pt = ope->p->ptype;

    pw->branches[0] = piecewise_poly_alloc();
    pw->branches[0]->leaf = 1;
    pw->branches[0]->branches = NULL;
    pw->branches[0]->ope = orth_poly_expansion_init(pt,ope->num_poly,lb,loc);
    orth_poly_expansion_approx(pw_eval_ope,ope,pw->branches[0]->ope);
    orth_poly_expansion_round(&(pw->branches[0]->ope));

    pw->branches[1] = piecewise_poly_alloc();
    pw->branches[1]->leaf = 1;
    pw->branches[1]->branches = NULL;
    pw->branches[1]->ope = orth_poly_expansion_init(pt,ope->num_poly,loc,ub);
    orth_poly_expansion_approx(pw_eval_ope,ope,pw->branches[1]->ope);
    orth_poly_expansion_round(&(pw->branches[1]->ope));
    
    orth_poly_expansion_free(pw->ope);
    pw->ope = NULL;
}

/********************************************************//**
*   Split a pw-poly leaf 
*
*   \param[in,out] pw     - polynomial to split (leaf of a pw polynomial)
*   \param[in]     N      - number of boundaries (including global lb and ub)
*   \param[in]     bounds - boundaries to split it into

*************************************************************/
void piecewise_poly_splitn(struct PiecewisePoly * pw, size_t N, const double * bounds)
{
    assert(pw->leaf == 1);
    struct OrthPolyExpansion * ope = pw->ope;

    /* double lb = ope->lower_bound; */
    /* double ub = ope->upper_bound; */
    assert (fabs(bounds[0] - ope->lower_bound) == 0.0);
    assert (fabs(bounds[N-1] - ope->upper_bound) == 0.0);

    pw->leaf = 0;
    pw->nbranches = N-1;
    pw->branches = piecewise_poly_array_alloc(N-1);

    enum poly_type pt = ope->p->ptype;
    
    size_t ii;
    for (ii = 0; ii < N-1; ii++){
        pw->branches[ii] = piecewise_poly_alloc();
        pw->branches[ii]->leaf = 1;
        pw->branches[ii]->branches = NULL;
        pw->branches[ii]->ope = orth_poly_expansion_init(pt,ope->num_poly,bounds[ii],bounds[ii+1]);
        orth_poly_expansion_approx(pw_eval_ope,ope,pw->branches[ii]->ope);
        orth_poly_expansion_round(&(pw->branches[ii]->ope));
    }
    orth_poly_expansion_free(pw->ope);
    pw->ope = NULL;
}



/********************************************************//**
*   Determine if the tree is flat (only leaves);   
*
*   \param[in] p  - pw poly
*
*   \return 1 if flat, 0 if not
*************************************************************/
int piecewise_poly_isflat(const struct PiecewisePoly * p)
{
    if (p->leaf == 1){
        return 1;
    }

    int flat = 1;
    size_t ii;
    for (ii = 0; ii < p->nbranches; ii++){
        if (p->branches[ii]->leaf != 1){
            flat = 0;
            break;
        }
    }
    return flat;
}

/********************************************************//**
*   Get the lower bound of the space on which a pw polynomial
*   is defined
*
*   \param[in] a - pw poly
*
*   \return pw lower bound
*************************************************************/
double piecewise_poly_get_lb(const struct PiecewisePoly * a)
{
    if (a->leaf  == 1){
        return a->ope->lower_bound;
    }
    else{
        return piecewise_poly_get_lb(a->branches[0]);
    }
}

/********************************************************//**
*   Get the upper bound of the space on which a pw polynomial
*   is defined
*
*   \param[in] a - pw poly
*
*   \return pw upper bound
*************************************************************/
double piecewise_poly_get_ub(const struct PiecewisePoly * a)
{

    if (a->leaf == 1){
        return a->ope->upper_bound;
    }
    else{
        return piecewise_poly_get_ub(a->branches[a->nbranches-1]);
    }

}

/********************************************************//**
*   Get number of pieces in a piecewise poly base function
*
*   \param[in]     a - pw polynomial
*   \param[in,out] N - number of pieces (must preset to 0!!!)
*************************************************************/
void piecewise_poly_nregions_base(size_t * N, const struct PiecewisePoly * a)
{
    if (a->leaf == 1){
        *N = *N + 1;
    }
    else{
        size_t ii;
        for (ii = 0; ii < a->nbranches; ii++){
            piecewise_poly_nregions_base(N,a->branches[ii]);
        }
    }
}

/********************************************************//**
*   Get number of pieces in a piecewise poly 
*
*   \param[in] a - pw polynomial
*************************************************************/
size_t piecewise_poly_nregions(const struct PiecewisePoly * a)
{
    size_t N = 0;
    piecewise_poly_nregions_base(&N,a);
    return N;
}

/********************************************************//**
*   Get boundary nodes between piecewise polynomials
*
*   \param[in]     a     - pw polynomial
*   \param[in,out] N     - number of nodes (including lower and upper)
*   \param[in,out] nodes - allocated nodes (can be NULL)
*   \param[in,out] onNum - node which I am filling
*   
*   \note
*       Call this function as follows to obtain full recursion and answer
*       
*       \code{.c}
*           // struct PiecewisePoly * a = ...; // have a be a piecewise poly
*           double * nodes = NULL;
*           size_t N;
*           piecewise_poly_boundaries(a,&N,&nodes,NULL);
*       \endcode
*************************************************************/
void 
piecewise_poly_boundaries(const struct PiecewisePoly * a, size_t *N, 
                          double ** nodes,
                          size_t * onNum)
{
    if (*nodes == NULL){ // first allocate required number of nodes
        size_t nregions = piecewise_poly_nregions(a);
        *N = nregions + 1;
        *nodes = calloc_double(*N);
        if ((*N) == 2){
            (*nodes)[0] = a->ope->lower_bound;
            (*nodes)[1] = a->ope->upper_bound;
        }
        else{
            size_t start = 0;
            piecewise_poly_boundaries(a,N,nodes,&start);
        }
    }
    else{
        if ( (*onNum) == 0){
            (*nodes)[0] = piecewise_poly_get_lb(a);
            *onNum = 1;
            piecewise_poly_boundaries(a,N,nodes,onNum);
        }
        else{
            if (a->leaf == 1){
                (*nodes)[*onNum] = a->ope->upper_bound;
                *onNum = (*onNum) + 1;
            }
            else{
                size_t ii;
                for (ii = 0; ii < a->nbranches; ii++){
                    piecewise_poly_boundaries(a->branches[ii],N,nodes,onNum);
                }
            }
        }

    }
}

/********************************************************//**
*   Evaluate a piecewise polynomial
*
*   \param[in] poly - pw poly
*   \param[in] x    - location at which to evaluate
*
*   \return out - pw value
*************************************************************/
double
piecewise_poly_eval(const struct PiecewisePoly * poly, double x){
    
    double out = 0.1234567890;
    if (poly->leaf == 1){
        out = orth_poly_expansion_eval(poly->ope, x);
    }
    else{
        for (size_t ii = 0; ii < poly->nbranches; ii++){
            if (x < piecewise_poly_get_ub(poly->branches[ii])+1e-14){
                out = piecewise_poly_eval(poly->branches[ii],x);
                break;
            }
        }
    }
    return out;
}

/********************************************************//**
*   Evaluate the derivative a piecewise polynomial
*
*   \param[in] poly - pw poly
*   \param[in] x    - location at which to evaluate
*
*   \return out 
*************************************************************/
double piecewise_poly_deriv_eval(const struct PiecewisePoly * poly, double x)
{
    
    double out = 0.1234567890;
    if (poly->leaf == 1){
        out = orth_poly_expansion_deriv_eval(poly->ope, x);
    }
    else{
        for (size_t ii = 0; ii < poly->nbranches; ii++){
            if (x < piecewise_poly_get_ub(poly->branches[ii])+1e-14){
                out = piecewise_poly_deriv_eval(poly->branches[ii], x);
                break;
            }
        }
    }
    return out;
}

/********************************************************//**
*   Evaluate a piecewise polynomial
*
*   \param[in]     poly - function
*   \param[in]     N    - number of evaluations
*   \param[in]     x    - location at which to evaluate
*   \param[in]     incx - increment of x
*   \param[in,out] y    - allocated space for evaluations
*   \param[in]     incy - increment of y*
*
*   \note Currently just calls the single evaluation code
*         Note sure if this is optimal, cache-wise
*************************************************************/
void piecewise_poly_evalN(const struct PiecewisePoly * poly, size_t N,
                          const double * x, size_t incx, double * y, size_t incy)
{
    for (size_t ii = 0; ii < N; ii++){
        y[ii*incy] = piecewise_poly_eval(poly,x[ii*incx]);
    }
}

/********************************************************//**
*   Scale a piecewise polynomial
*
*   \param[in]     a    - scale factors
*   \param[in,out] poly - pw poly

*************************************************************/
void piecewise_poly_scale(double a, struct PiecewisePoly * poly){
    
    if (poly->leaf == 1){
        orth_poly_expansion_scale(a, poly->ope);
    }
    else{
        size_t ii;
        for (ii = 0; ii < poly->nbranches; ii++){
            piecewise_poly_scale(a,poly->branches[ii]);
        }
    }
}

/********************************************************//**
*   Differentiate a piecewise polynomial
*   
*   \param[in] p - pw poly to differentiate (from the left)
*
*   \return pnew - polynomial
*************************************************************/
struct PiecewisePoly * 
piecewise_poly_deriv(const struct PiecewisePoly * p)
{

    struct PiecewisePoly * pnew = NULL;
    if (p == NULL){
        return pnew;
    }
    else if (p->leaf == 1){
        pnew = piecewise_poly_alloc();
        assert (p->ope != NULL);
        /* printf("p->ope->nalloc = %zu\n",p->ope->nalloc); */
        /* print_orth_poly_expansion(p->ope,0,NULL); */
        pnew->ope = orth_poly_expansion_deriv(p->ope);
        /* printf("got deriv\n"); */
        pnew->leaf = 1;
    }
    else{
        pnew = piecewise_poly_alloc();
        pnew->leaf = 0;
        pnew->nbranches = p->nbranches;
        pnew->branches = piecewise_poly_array_alloc(p->nbranches);
        size_t ii;
        for (ii = 0; ii < p->nbranches; ii++){
            pnew->branches[ii] = piecewise_poly_deriv(p->branches[ii]);
        }
    }
    return pnew;
}

/********************************************************//**
*   Differentiate a piecewise polynomial twice
*   
*   \param[in] p - pw poly to differentiate (from the left)
*
*   \return pnew - polynomial
*************************************************************/
struct PiecewisePoly * 
piecewise_poly_dderiv(const struct PiecewisePoly * p)
{

    struct PiecewisePoly * pnew = NULL;
    if (p == NULL){
        return pnew;
    }
    else if (p->leaf == 1){
        pnew = piecewise_poly_alloc();
        assert (p->ope != NULL);
        /* printf("p->ope->nalloc = %zu\n",p->ope->nalloc); */
        /* print_orth_poly_expansion(p->ope,0,NULL); */
        pnew->ope = orth_poly_expansion_dderiv(p->ope);
        /* printf("got deriv\n"); */
        pnew->leaf = 1;
    }
    else{
        pnew = piecewise_poly_alloc();
        pnew->leaf = 0;
        pnew->nbranches = p->nbranches;
        pnew->branches = piecewise_poly_array_alloc(p->nbranches);
        size_t ii;
        for (ii = 0; ii < p->nbranches; ii++){
            pnew->branches[ii] = piecewise_poly_dderiv(p->branches[ii]);
        }
    }
    return pnew;
}

/********************************************************//**
   Take a second derivative and enforce periodic bc
**************************************************************/
struct PiecewisePoly * piecewise_poly_dderiv_periodic(const struct PiecewisePoly * f)
{
    (void)(f);
    NOT_IMPLEMENTED_MSG("piecewise_poly_dderiv_periodic");
    exit(1);
}


/********************************************************//**
*   Integrate a piecewise polynomial
*
*   \param[in] poly - pw polynomial to integrate
*
*   \return Integral of approximation
*  
*   \note Should make this tail recursive in the future
*************************************************************/
double
piecewise_poly_integrate(const struct PiecewisePoly * poly)
{
    double out = 0.0;
    if (poly->leaf == 1){
        out = orth_poly_expansion_integrate(poly->ope);
    }
    else{
        size_t ii;
        for (ii = 0; ii < poly->nbranches; ii++){
            out = out + piecewise_poly_integrate(poly->branches[ii]);
        }
    }
    return out;
}

/********************************************************//**
*   Integrate a piecewise polynomial
*
*   \param[in] poly  - pw polynomial to integrate
*
*   \return Integral of approximation
*  
*   \note Should make this tail recursive in the future
*************************************************************/
double piecewise_poly_integrate_weighted(const struct PiecewisePoly * poly)
{
    (void) poly;
    NOT_IMPLEMENTED_MSG("piecewise_poly_integrate_weighted")
    return 0.0;
}

double * piecewise_poly_rr(const struct PiecewisePoly * p, size_t * nkeep)
{
    double * real_roots = NULL;   
    if ( p->leaf == 1){
        real_roots = orth_poly_expansion_real_roots(p->ope, nkeep);
    }
    else{
        real_roots = piecewise_poly_rr(p->branches[0],nkeep);
        size_t ii,n0;
        for (ii = 1; ii < p->nbranches; ii++){
            n0 = 0;
            double * roots2 = piecewise_poly_rr(p->branches[ii], &n0);
            if (n0 > 0){
                real_roots = realloc(real_roots,(*nkeep + n0)*sizeof(double));
                assert(real_roots != NULL);
                memmove(real_roots + *nkeep, roots2, n0*sizeof(double));
                *nkeep = *nkeep + n0;
                free(roots2); roots2 = NULL;
            }
        }
    }
    return real_roots;
}

/********************************************************//**
*   Obtain the real roots of a pw polynomial (only gives 1 of repeating roots)
*
*   \param[in]     p     - piecewise polynomial
*   \param[in,out] nkeep - returns how many real roots there are 
*
*   \return real roots of the pw polynomial
*
*   \note
*       Each root may be repeated twice
*************************************************************/
double *
piecewise_poly_real_roots(const struct PiecewisePoly * p, size_t * nkeep)
{
    *nkeep = 0;    
    //printf("lb = %G\n",piecewise_poly_get_lb(p));
    double * roots = piecewise_poly_rr(p,nkeep);
    
    /*
    size_t ii, jj;
    size_t * keep = calloc_size_t(*nkeep);
    for (ii = 0; ii < *nkeep; ii++){
        for (jj = ii+1; jj < *nkeep; jj++){
            if (fabs(roots[ii] - roots[jj]) <= 1e-10){
                keep[jj] = 1;
            }
        }
    }
    size_t counter = 0;
    for (ii = 0; ii < *nkeep; ii++){
        if (keep[ii] == 0){
            counter++;
        }
    }
    double * real_roots = calloc_double(counter);
    for (ii = 0; ii < *nkeep; ii++){
        if (keep[ii] == 0){
            real_roots[ii] = roots[ii];
        }
    }
    free(keep); keep = NULL;
    free(roots); roots = NULL;
    */
    return roots;
}

/********************************************************//**
*   Obtain the maximum of a pw polynomial
*
*   \param[in]     p - pw polynomial
*   \param[in,out] x - location of maximum value
*
*   \return maximum value
*   
*   \note
*       if constant function then just returns the left most point
*************************************************************/
double piecewise_poly_max(const struct PiecewisePoly * p, double * x)
{
    double locfinal, valfinal;
    if ( p->leaf == 1){
        return orth_poly_expansion_max(p->ope, x);
    }
    else{
        size_t ii = 0;
        double loc2, val2;
        valfinal = piecewise_poly_max(p->branches[0],&locfinal);
        for (ii = 1; ii < p->nbranches;ii++){
            val2 = piecewise_poly_max(p->branches[ii],&loc2);
            if (val2 > valfinal){
                valfinal = val2;
                locfinal = loc2;
            }
        }
    }
    *x = locfinal;
    return valfinal;
}

/********************************************************//**
*   Obtain the minimum of a pw polynomial
*
*   \param[in]     p - pw polynomial
*   \param[in,out] x - location of minimum value
*
*   \return minimum value
*   
*   \note
*       if constant function then just returns the left most point
*************************************************************/
double piecewise_poly_min(const struct PiecewisePoly * p, double * x)
{
    double locfinal, valfinal;
    if ( p->leaf == 1){
        return orth_poly_expansion_min(p->ope, x);
    }
    else{
        size_t ii = 0;
        double loc2, val2;
        valfinal = piecewise_poly_min(p->branches[0],&locfinal);
        for (ii = 1; ii < p->nbranches;ii++){
            val2 = piecewise_poly_min(p->branches[ii],&loc2);
            if (val2 < valfinal){
                valfinal = val2;
                locfinal = loc2;
            }
        }
    }
    *x = locfinal;
    return valfinal;
}

/********************************************************//**
*   Obtain the absolute maximum of a pw polynomial
*
*   \param[in]     p       - pw polynomial
*   \param[in,out] x       - location of absolute maximum
*   \param[in]     optargs - optimization arguments
*
*   \return val - absolute maximum
*   
*   \note
*       if constant function then just returns the left most point
*************************************************************/
double piecewise_poly_absmax(const struct PiecewisePoly * p, double * x, void * optargs)
{
    //printf("here!\n");
    double locfinal, valfinal;
    if ( p->leaf == 1){
        //double lb = piecewise_poly_get_lb(p);
        //double ub = piecewise_poly_get_ub(p);
        //printf("in leaf (%G,%G)\n",lb,ub);
        //if ((ub - lb) < 1000.0*DBL_EPSILON){
        //    return 0.0;
        //}
        //print_orth_poly_expansion(p->ope,3,NULL);
        double maxval = orth_poly_expansion_absmax(p->ope, x,optargs);
        //printf("max is %G \n",maxval);
        return maxval;
    }
    else{
        double loc2, val2;
        size_t ii = 0;
        valfinal = piecewise_poly_absmax(p->branches[0],&locfinal,optargs);
        //printf("nbranches = %zu\n",p->nbranches);
        for (ii = 1; ii < p->nbranches;ii++){
            val2 = piecewise_poly_absmax(p->branches[ii],&loc2,optargs);
            if (val2 > valfinal){
                valfinal = val2;
                locfinal = loc2;
            }
        }
    }
    *x = locfinal;
    return valfinal;
}

/********************************************************//**
*   Compute the norm of piecewise polynomial
*
*   \param[in] p - pw polynomial of which to obtain norm
*
*   \return norm of function
*
*   \note
*        Computes int_a^b f(x)^2 dx
*************************************************************/
double piecewise_poly_norm(const struct PiecewisePoly * p){
    
    double out = piecewise_poly_inner(p,p);
    return sqrt(out);
}

/********************************************************//**
*   Multiply piecewise polynomial by -1
*
*   \param[in,out] p - pw polynomial to multiply by -1
*************************************************************/
void 
piecewise_poly_flip_sign(struct PiecewisePoly * p)
{   
    if (p->leaf == 1){
        orth_poly_expansion_flip_sign(p->ope);
    }
    else{
        size_t ii;
        for (ii = 0; ii < p->nbranches; ii++){
            piecewise_poly_flip_sign(p->branches[ii]);
        }
    }
}

/////////////////////////////////////////////////////////////////////////

/********************************************************//**
*   Create a PW-Poly array of copies of each leaf
*
*   \param[in]     p        - pw poly whose leaves to copy
*   \param[in]     branches - list of references to leaves
*   \param[in,out] onbranch - location of branches
*
*************************************************************/
void piecewise_poly_copy_leaves(const struct PiecewisePoly * p,
                                struct PiecewisePoly ** branches,
                                size_t * onbranch)
{

    if (p->leaf == 1){
        branches[*onbranch] = piecewise_poly_copy(p);
        (*onbranch) = (*onbranch) + 1;
    }
    else{
        size_t ii;
        for (ii = 0; ii < p->nbranches; ii++){
            piecewise_poly_copy_leaves(p->branches[ii],branches,onbranch);
        }
    }
}

/********************************************************//**
*   Create a PW-Poly array of references to each leaf
*
*   \param[in]     p        - pw poly whose leaves to copy
*   \param[in]     branches - list of references to leaves
*   \param[in,out] onbranch - location of branches
*
*************************************************************/
void piecewise_poly_ref_leaves(struct PiecewisePoly * p,
                               struct PiecewisePoly ** branches,
                               size_t * onbranch)
{

    if (p->leaf == 1){
        branches[*onbranch] = p;
        (*onbranch) = (*onbranch) + 1;
    }
    else{
        size_t ii;
        for (ii = 0; ii < p->nbranches; ii++){
            piecewise_poly_ref_leaves(p->branches[ii],branches,onbranch);
        }
    }
}


/********************************************************//**
*   Flatten a piecewise polynomial (make it so each branch is a leaf)
*
*   \param[in,out] p - pw poly whose leaves to copy
*
*   \note
*       Should figure out how to do this without copying the leaves
*       but just referencing them. When I did just reference them I had
*       memory leaks
*************************************************************/
void piecewise_poly_flatten(struct PiecewisePoly * p)
{
    int isflat = piecewise_poly_isflat(p);
    if (isflat == 0){
        size_t nregions = piecewise_poly_nregions(p);
        struct PiecewisePoly ** newbranches = 
            piecewise_poly_array_alloc(nregions);
    
        size_t onregion = 0;
        size_t ii;
        for (ii = 0; ii < p->nbranches; ii++){
            piecewise_poly_copy_leaves(p->branches[ii],
                                        newbranches,&onregion);
            //piecewise_poly_ref_leaves(p->branches[ii],
            //                            newbranches,&onregion);
        }

        piecewise_poly_array_free(p->branches,p->nbranches);
        //free(p->branches); p->branches = NULL;
        p->nbranches = nregions;
        p->branches = newbranches;
        p->ope = NULL;
        p->leaf = 0;
    }
}

/********************************************************//**
*   Reapproximate a piecewise poly on a finer grid.
*
*   \param[in] a     - pw polynomial to reapproximate
*   \param[in] N     - number of nodes (including lb,ub)
*   \param[in] nodes - nodes defining boundaries of each region
*
*   \note
*       Each of the new pieces must be fully encompassed by an old piece
*       NOTE USES LEGENDRE POLYNOMIALS ON EACH LEAF
*************************************************************/
struct PiecewisePoly *
piecewise_poly_finer_grid(const struct PiecewisePoly * a, size_t N, double * nodes)
{

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    /* printf("nbranches of a = %zu\n",a->nbranches); */
    fwrap_set_fvec(fw,pw_eval,(void*)a);

    struct PiecewisePoly * p = NULL;
    if (N == 2){
        assert( a->leaf == 1);
        p = piecewise_poly_copy(a);
    }
    else{

        p = piecewise_poly_alloc();
        p->nbranches = N-1;
        p->leaf = 0;
        p->ope = NULL;
        
        struct OpeOpts * aopts = ope_opts_alloc(LEGENDRE);
        ope_opts_set_start(aopts,8);
        ope_opts_set_coeffs_check(aopts,2);
        ope_opts_set_tol(aopts,1e-14);
        
        p->branches = piecewise_poly_array_alloc(N-1);
        size_t ii;
        for (ii = 0; ii < N-1; ii++){
            /* printf("lb,ub = (%G,%G)\n",nodes[ii],nodes[ii+1]); */
            ope_opts_set_lb(aopts,nodes[ii]);
            ope_opts_set_ub(aopts,nodes[ii+1]);
            p->branches[ii] = piecewise_poly_alloc();
            p->branches[ii]->leaf = 1;
            p->branches[ii]->ope = orth_poly_expansion_approx_adapt(aopts,fw);
            /* LEGENDRE, nodes[ii],nodes[ii+1], aopts); */
        }
        ope_opts_free(aopts); aopts = NULL;
    }
    fwrap_destroy(fw);
    return p;
}

//////////////////////////////////////

/********************************************************//**
*   Inner product between two pw polynomials 
*
*   \param[in] a - first pw polynomial
*   \param[in] b - second pw polynomai
*
*   \return out - inner product
*
*   Notes: 
*          Computes \f$ int_{lb}^ub  a(x)b(x) dx \f$
*************************************************************/
double 
piecewise_poly_inner(const struct PiecewisePoly * a,
                     const struct PiecewisePoly * b)
{
    assert (a != NULL);
    assert (b != NULL);
    if ((a->leaf == 1) && (b->leaf == 1)){
        return orth_poly_expansion_inner(a->ope, b->ope);
    }
    //printf("there!\n");

    struct PiecewisePoly * aa = NULL;
    struct PiecewisePoly * bb = NULL;
    piecewise_poly_match(a,&aa,b,&bb);
    
    piecewise_poly_flatten(aa);
    piecewise_poly_flatten(bb);
    
    double integral = 0;
    for (size_t ii = 0; ii < aa->nbranches; ii++){
        integral += orth_poly_expansion_inner(aa->branches[ii]->ope,bb->branches[ii]->ope);
    }

    piecewise_poly_free(aa); aa = NULL;
    piecewise_poly_free(bb); bb = NULL;
    return integral;
}

static void piecewise_poly_matched_axpy(double a, struct PiecewisePoly * x, struct PiecewisePoly * y)
{
    if (x->leaf == 1){
        orth_poly_expansion_axpy(a, x->ope, y->ope);
    }
    else{
        for (size_t ii = 0; ii < x->nbranches; ii++){
            piecewise_poly_matched_axpy(a, x->branches[ii], y->branches[ii]);
        }
    }
}

/********************************************************//**
*   Add two piecewise polynomials \f$ y \leftarrow ax + y \f$
*
*   \param[in]     a - scaling of first function
*   \param[in,out] x - first function (potentially it is flattened or split)
*   \param[in,out] y - second function
*
*   \return 0 if successful, 1 if error
*
************************************************************/
int piecewise_poly_axpy(double a,struct PiecewisePoly * x, struct PiecewisePoly * y)
{   

    assert (x != NULL);
    assert (y != NULL);

    if ((x->leaf == 1) && (y->leaf == 1)){
        return orth_poly_expansion_axpy(a, x->ope, y->ope);
    }

    struct PiecewisePoly * aa = NULL;
    struct PiecewisePoly * bb = NULL;
    piecewise_poly_match(x,&aa,y,&bb);
    piecewise_poly_matched_axpy(a, aa, bb);
    piecewise_poly_free_inside(y); 
    piecewise_poly_copy_inside(bb, y);

    piecewise_poly_free(aa); aa = NULL;
    piecewise_poly_free(bb); bb = NULL;
    return 0;
}

/********************************************************//**
*   Multiply by scalar and add two PwPolynomials
*
*   \param[in] a - scaling factor for first pw poly
*   \param[in] x - first pw poly
*   \param[in] b - scaling factor for second pw poly
*   \param[in] y - second pw poly
*
*   \return pw poly
*
*   \note 
*       Computes z=ax+by, where x and y are pw polys
*       Requires both polynomials to have the same upper 
*       and lower bounds
*   
*************************************************************/
struct PiecewisePoly *
piecewise_poly_daxpby(double a,const struct PiecewisePoly * x,
                      double b,const struct PiecewisePoly * y)
{

    if ( (x == NULL) && (y == NULL)){
        return NULL;
    }
    if (x == NULL){
        struct PiecewisePoly * p = piecewise_poly_copy(y);
        piecewise_poly_scale(b,p);
        return p;
    }
    else if (y == NULL){
        struct PiecewisePoly * p = piecewise_poly_copy(x);
        piecewise_poly_scale(a,p);
        return p;
    }

    struct PiecewisePoly * xx = NULL;
    struct PiecewisePoly * yy = NULL;
    piecewise_poly_match(x,&xx,y,&yy);
    struct PiecewisePoly * c = piecewise_poly_matched_daxpby(a,xx,b,yy);
    piecewise_poly_free(xx); xx = NULL;
    piecewise_poly_free(yy); yy= NULL;
    return c;    
}


/********************************************************//**
*   Compute the sum of two piecewise polynomials with
*   matching hierarchy
*   
*   \param[in] a - weight of first pw polynomial
*   \param[in] x - first pw polynomial
*   \param[in] b - weight of second pw polynomial
*   \param[in] y - second pw polynomial
*
*   \return c - pw polynomial
*
*   \note 
*        Computes \f$ c = a*x + b*x \f$ where c is same form as a
*************************************************************/
struct PiecewisePoly *
piecewise_poly_matched_daxpby(double a,const struct PiecewisePoly * x,
                              double b,const struct PiecewisePoly * y)
{

    struct PiecewisePoly * c = piecewise_poly_alloc();
    if (y == NULL){
        assert (x != NULL);
        if ( x->leaf == 1){
            c->leaf = 1;
            c->ope = orth_poly_expansion_daxpby(a,x->ope,b,NULL);
        }
        else{
            c->leaf = 0;
            c->nbranches = x->nbranches;
            c->branches = piecewise_poly_array_alloc(c->nbranches);
            size_t ii;
            for (ii = 0; ii < c->nbranches; ii++){
                c->branches[ii] = 
                    piecewise_poly_matched_daxpby(a,x->branches[ii],b,NULL);
            }
        }
    }
    else if ( x == NULL){
        assert ( y != NULL );
        if ( y->leaf == 1){
            c->leaf = 1;
            c->ope = orth_poly_expansion_daxpby(b,y->ope,a,NULL);
        }
        else{
            c->leaf = 0;
            c->nbranches = y->nbranches;
            c->branches = piecewise_poly_array_alloc(c->nbranches);
            size_t ii;
            for (ii = 0; ii < c->nbranches; ii++){
                c->branches[ii] = 
                    piecewise_poly_matched_daxpby(b,y->branches[ii],a,NULL);
            }
        }
    }
    else{
        //printf("in here!\n");
        if ( x->leaf == 1 ){
            assert ( y->leaf == 1);
            c->leaf = 1;
            c->ope = orth_poly_expansion_daxpby(a,x->ope,b,y->ope);
        }
        else{
            if (x->nbranches != y->nbranches){
                printf("x nbranches = %zu\n",x->nbranches);
                printf("y nbranches = %zu\n",y->nbranches);
            }
            assert (x->nbranches == y->nbranches);
            c->leaf = 0;
            c->nbranches = y->nbranches;
            c->branches = piecewise_poly_array_alloc(c->nbranches);
            size_t ii;
            for (ii = 0; ii < c->nbranches; ii++){
                c->branches[ii] = 
                    piecewise_poly_matched_daxpby(a,x->branches[ii],
                                                  b,y->branches[ii]);
            }
        }
    }
    return c;
}

/********************************************************//**
*   Compute the product of two piecewise polynomials with
*   matching hierarchy
*
*   \param[in] a - first pw polynomial
*   \param[in] b - second pw polynomial
*
*   \return pw polynomial
*
*   \note 
*        Computes \f$ c(x) = a(x)b(x)\f$ where c is same form as a
*************************************************************/
struct PiecewisePoly *
piecewise_poly_matched_prod(const struct PiecewisePoly * a,
                            const struct PiecewisePoly * b)
{
    struct PiecewisePoly * c = piecewise_poly_alloc();

    if ( a->leaf == 1){
        assert ( b->leaf == 1);
        c->leaf = 1;
        c->ope = orth_poly_expansion_prod(a->ope,b->ope);
    }
    else{
        assert ( a->nbranches == b->nbranches );
        c->leaf = 0;
        c->nbranches = a->nbranches;
        c->branches = piecewise_poly_array_alloc(c->nbranches);
        size_t ii;
        for (ii = 0; ii < c->nbranches; ii++){
            c->branches[ii] = 
                piecewise_poly_matched_prod(a->branches[ii],b->branches[ii]); 
        }
    }
    return c;
}

/********************************************************//**
*   Compute the product of two piecewise polynomials
*
*   \param[in] a  - first pw polynomial
*   \param[in] b  - second pw polynomial
*
*   \return c - pw polynomial
*
*   \note 
*        Computes c(x) = a(x)b(x) where c is same form as a
*************************************************************/
struct PiecewisePoly *
piecewise_poly_prod(const struct PiecewisePoly * a,
                    const struct PiecewisePoly * b)
{
    struct PiecewisePoly * aa = NULL;
    struct PiecewisePoly * bb = NULL;

    piecewise_poly_match(a,&aa,b,&bb);
    struct PiecewisePoly * out = piecewise_poly_matched_prod(aa,bb);
    piecewise_poly_free(aa); aa = NULL;
    piecewise_poly_free(bb); bb = NULL;
    return out;
}


/********************************************************
*   Match the discretizations of two pw polys
*
*   \param a [inout] - poly1 to match
*   \param b [inout] - poly2 to match
*
*************************************************************/
/* void piecewise_poly_match1(struct PiecewisePoly * a,struct PiecewisePoly * b) */
/* { */
/*     fprintf(stderr, "piecewise_poly_match1 implementation not complete\n"); */
/*     exit(1); */
/*     double lba = piecewise_poly_get_lb(a); */
/*     double lbb = piecewise_poly_get_lb(b); */
/*     assert(fabs(lba-lbb) == 0); */
/*     double uba = piecewise_poly_get_ub(a); */
/*     double ubb = piecewise_poly_get_ub(b); */
/*     assert(fabs(uba-ubb) == 0); */

/*     if ( (a->leaf == 1) && (b->leaf == 0) ) */
/*     { */
/*         double * bounds = NULL; */
/*         size_t nbounds; */
/*         piecewise_poly_boundaries(b,&nbounds,&bounds,NULL); */
/*         piecewise_poly_splitn(a,nbounds,bounds); */
/*         free(bounds); bounds = NULL; */
/*     } */
/*     else if ( (a->leaf == 0) && (b->leaf == 1) ) */
/*     { */
/*         double * bounds = NULL; */
/*         size_t nbounds; */
/*         piecewise_poly_boundaries(a,&nbounds,&bounds,NULL); */
/*         piecewise_poly_splitn(b,nbounds,bounds); */
/*         free(bounds); bounds = NULL; */
/*     } */
/*     else if ( (a->leaf == 0) && (b->leaf == 0)) { */
        
/*         size_t onbrancha = 0; */
/*         size_t onbranchb = 0; */

/*     } */
/* } */

/********************************************************//**
*   Convert two piecewise polynomials to ones with matching
*   splits / hierarchy
*
*   \param[in]     ain  - first pw polynomial
*   \param[in,out] aout - new matched pw polynomial 1 (unallocated)
*   \param[in]     bin  - second pw polynomial
*   \param[in,out] bout - new matched pw polynomial 2 (unallocated)
*
*   \note
*       Should check if matching in the first place
*       New lower bound is highest lower bound, and new upper bound is lowest upper bound
*************************************************************/
void
piecewise_poly_match(const struct PiecewisePoly * ain, struct PiecewisePoly ** aout,
                     const struct PiecewisePoly * bin, struct PiecewisePoly ** bout)
{
    double * nodesa = NULL;
    double * nodesb = NULL;
    size_t Na, Nb;
    piecewise_poly_boundaries(ain, &Na, &nodesa,NULL);
    piecewise_poly_boundaries(bin, &Nb, &nodesb,NULL);

    if (Na == Nb){
        int same = 1;
        for (size_t ii = 0; ii < Na; ii++){
            if (fabs(nodesa[ii] - nodesb[ii]) > 1e-15) {
                same = 0;
                break;
            }
        }
        if (same == 1){
            free(nodesa); nodesa = NULL;
            free(nodesb); nodesb = NULL;
            *aout = piecewise_poly_copy(ain);
            *bout = piecewise_poly_copy(bin);
            piecewise_poly_flatten(*aout);
            piecewise_poly_flatten(*bout);
            return;
        }
    }
    
    
    double lb = nodesa[0] < nodesb[0] ? nodesa[0] : nodesb[0];
    double ub = nodesa[Na-1] > nodesb[Nb-1] ? nodesa[Na-1] : nodesb[Nb-1];
    
    double * nodes = calloc_double(Na + Nb);
    nodes[0] = lb;

    size_t inda = 1;
    while (nodesa[inda] < lb){
        inda++;
    }
    size_t indb = 1;
    while (nodesb[indb] < lb){
        indb++;
    }

    size_t cind = 1;
    while (nodes[cind-1] < ub){
        if (nodesa[inda] <= nodesb[indb]){
            nodes[cind] = nodesa[inda];
        }
        else if (nodesb[indb] < nodesa[inda]){
            nodes[cind] = nodesb[indb];
        }

        if (fabs(nodesb[indb] - nodesa[inda]) < DBL_EPSILON){
            inda++;
            indb++;
        }
        else if (nodesb[indb] < nodesa[inda]){
            indb++;
        }
        else if (nodesa[inda] < nodesb[indb]){
            inda++;
        }
        cind++;
    }


    double * newnodes = realloc(nodes, cind * sizeof(double));
    if (newnodes == NULL){
        fprintf(stderr,"Error (re)allocating memory in piecewise_poly_match");
        exit(1);
    }
    else{
        nodes = newnodes;
    }
    
    *aout = piecewise_poly_finer_grid(ain, cind, nodes);
    *bout = piecewise_poly_finer_grid(bin, cind, nodes);
    free(nodesa); nodesa = NULL;
    free(nodesb); nodesb = NULL;
    free(nodes); nodes = NULL;
}

/********************************************************
*   Remove left-most piece of pw Poly
*
*   \param a [inout] - pw polynomial to trim
*
*   \return poly - left most orthogonal polynomial expansion
*
*   \note
*       If *a* doesn't have a split, the orthogonal expansion is extracted
*       and a is turned to NULL
*************************************************************/
/*
struct OrthPolyExpansion * 
piecewise_poly_trim_left(struct PiecewisePoly ** a)
{   
    struct OrthPolyExpansion * poly = NULL;
    if (a == NULL){
        return poly;
    }
    else if (*a == NULL){
        return poly;
    }
    else if ((*a)->ope != NULL){
        poly = orth_poly_expansion_copy((*a)->ope);
        piecewise_poly_free(*a);
        *a = NULL;
    }
    else if ( (*a)->down[0]->ope != NULL)  // remove the left 
    {
        //printf("removing left lb=%G ub=%G \n", piecewise_poly_get_lb((*a)->left), piecewise_poly_get_ub((*a)->left));
        //printf("new lb should be %G \n", piecewise_poly_get_lb((*a)->right));
        poly = orth_poly_expansion_copy((*a)->down[0]->ope);
        //piecewise_poly_free( (*a)->left);
        
        (*a)->nbranches -= 1;
        size_t ii;
        for (ii = 0; ii < (*a)->nbranches; ii++){
            piecewise_poly_free((*a)->down[ii]);
            (*a)->down[ii] = piecewise_poly_copy((*a)->down[ii+1]);
        }
        piecewise_poly_free((*a)->down[(*a)->nbranches]);
        //(*a)->ope = (*a)->right->ope;
        //if ( (*a)->ope == NULL){
        //    (*a)->split = (*a)->right->split;
        //   // printf("new split is %G\n", (*a)->split);
        //}
        //(*a)->left = (*a)->right->left;
        //(*a)->right = (*a)->right->right;
       // printf("new lb = %G \n", piecewise_poly_get_lb(*a));
    }
    else {
        poly = piecewise_poly_trim_left( &((*a)->down[0]));
    }
    return poly;
}
*/

/********************************************************
*   Check if discontinuity exists between two neighboring
*   piecewise polynomials (upper bound of left == lower bound of right)
*  
*   \param left [in] - left pw polynomial   
*   \param right [in] - right pw polynomial   
*   \param numcheck [in] - number of derivatives to check (if 0 then check only values)
*   \param tol [in] - tolerance defining how big a jump defins discontinuity
*
*   \return 0 if no discontinuity, 1 if discontinuity
*************************************************************/
int piecewise_poly_check_discontinuity(struct PiecewisePoly * left, 
                                       struct PiecewisePoly * right, 
                                       int numcheck, double tol)
{
    if (numcheck == -1){
        return 0;
    }

    double ubl = piecewise_poly_get_ub(left);
    double lbr = piecewise_poly_get_lb(right);
    assert(fabs(ubl-lbr) < DBL_EPSILON*100);

    double val1, val2;
    val1 = piecewise_poly_eval(left, ubl);
    val2 = piecewise_poly_eval(right, lbr);

    double diff = fabs(val1-val2);
    if (fabs(val1) >= 1.0){
        diff /= fabs(val1);
    }
    
    int out;
    if ( diff < tol ){
        struct PiecewisePoly * dleft = piecewise_poly_deriv(left);
        struct PiecewisePoly * dright = piecewise_poly_deriv(right);
        out = piecewise_poly_check_discontinuity(dleft, dright, numcheck-1,tol);
        piecewise_poly_free(dleft); dleft = NULL;
        piecewise_poly_free(dright); dright = NULL;
    }
    else{
        out = 1;
    }
    return out;
}

/*
struct PiecewisePoly *
piecewise_poly_merge_left(struct PiecewisePoly ** p, struct PwPolyOpts * aopts)
{
    struct PiecewisePoly * pnew = NULL;
    //printf("in here\n");
    if (p == NULL){
        return pnew;
    }
    else if (*p == NULL){
        return pnew;
    }
    else if ( (*p)->ope != NULL){
        pnew = piecewise_poly_copy(*p);
        return pnew;
    }
    
    pnew = piecewise_poly_alloc();
    pnew->left = piecewise_poly_alloc();
    //printf("p == NULL = %d\n", p ==NULL);
    pnew->left->ope = piecewise_poly_trim_left(p);
    int disc;
    if ( *p == NULL ){
        pnew->ope = orth_poly_expansion_copy(pnew->left->ope);
        piecewise_poly_free(pnew->left);
        disc = -1;
    }
    else{
        disc = piecewise_poly_check_discontinuity(pnew->left, *p, 2, 1e-1);
    }
    while (disc == 0){
        //printf("discontinuity does not exist at %G \n", piecewise_poly_get_ub(pnew->left));
        struct OrthPolyExpansion * rightcut = piecewise_poly_trim_left(p);

        //printf("p == NULL 2 = %d\n", *p ==NULL);
        //printf("rightcut == NULL = %d\n", rightcut == NULL);

        struct PiecewisePoly temp;
        temp.ope = rightcut;

        struct PwCouple c;
        c.a = pnew->left;
        c.b = &temp;
            
        struct PiecewisePoly * newleft = piecewise_poly_alloc();
        if (aopts == NULL){
            newleft->ope = orth_poly_expansion_approx_adapt(pw_eval_neighbor,&c,
               rightcut->p->ptype, piecewise_poly_get_lb(c.a), piecewise_poly_get_ub(c.b), NULL);

        }
        else{
            struct OpeAdaptOpts adopts;
            adopts.start_num = aopts->maxorder;
            adopts.coeffs_check = aopts->coeff_check;
            adopts.tol = aopts->epsilon;

            newleft->ope = orth_poly_expansion_approx_adapt(pw_eval_neighbor,&c,
                rightcut->p->ptype, piecewise_poly_get_lb(c.a), piecewise_poly_get_ub(c.b), &adopts);
        }
        orth_poly_expansion_free(rightcut);
        rightcut = NULL;
        
        piecewise_poly_free(pnew); pnew = NULL;
        if ( *p == NULL ){
            pnew = piecewise_poly_copy(newleft);
            disc = -1;
        }
        else{
            pnew = piecewise_poly_alloc();
            pnew->left = piecewise_poly_copy(newleft);
            disc = piecewise_poly_check_discontinuity(pnew->left, *p, 2, 1e-6);;
        }
        piecewise_poly_free(newleft); newleft = NULL;
    }
    //printf("p==NULL? %d\n",*p==NULL);
    if (disc == 1){
        pnew->split = piecewise_poly_get_ub(pnew->left);
        pnew->right = piecewise_poly_merge_left(p,aopts);
        pnew->ope = NULL;
    }

    return pnew;
}
*/

// factorial for x = 0...8
static const size_t factorial [] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320};
double eval_coeff(size_t l, double * stencil, size_t nstencil)
{
    assert ( (nstencil-1) <= 8); //only ones for which I have a factorial
    
    // m = nstencil-1
    double out = (double) factorial[nstencil-1];
    size_t ii; 
    for (ii = 0; ii < nstencil; ii++){
        if (ii != l){
            out /= (stencil[l]-stencil[ii]);
        }
    }
    return out;
}

double eval_jump(double x, double * stencil, double * vals, size_t nstencil)
{
    double den = 0.0;
    double out = 0.0;
    double temp;
    size_t ii;
    for (ii = 0; ii < nstencil; ii++){
        temp = eval_coeff(ii,stencil,nstencil);
        if (stencil[ii] > x){
            den += temp;
        }
        out += temp * vals[ii];
    }
    assert (fabs(den) >= DBL_EPSILON);
    out = out / den;
    return out;
}

size_t get_stencil(double x, size_t nstencil, double * total, size_t ntotal)
{
    assert (nstencil <= ntotal);
    assert (x > total[0]);
    assert (x < total[ntotal-1]);
    size_t start;

    size_t ii = 0;
    while ( total[ii] < x){
        ii++;
    }

    if (ii == 1){
        start = 0;
    }
    else if (ii == (ntotal-1)){
        start = ntotal-nstencil;
    }
    else{
        size_t f = ii-1;
        size_t b = ii;
        // now total[ii] is the first element greater than x in total
        double difff, diffb;
        size_t ninc = 2;
        while (ninc < nstencil){
            if (f == 0){
                b++;
            }
            else if (b == ntotal-1){
                f--;
            }
            else{
                difff = total[b+1] - x;
                diffb = x - total[f-1];
                if (difff < diffb){
                    b++;
                }
                else{
                    f--;
                }
            }
            ninc ++;
        }
        start = f;
    }
    return start;
}

/********************************************************//**
*   Evaluate a MinModed jump function based on polynomial annihilation on
*   function values obtained at a sorted set (low to high) of input
*   
*   \param x [in] - location to check
*   \param total [in] - set of sorted points
*   \param evals [in] - function evaluations
*   \param ntotal [in] - number of points
*   \param minm [in] - minimum polynomial order to annihilate
*   \param maxm [in] - maximum polynomial order to annihilate
*
*   \return jump - jump value;
*************************************************************/
double minmod_eval(double x, double * total, double * evals, size_t ntotal, 
                    size_t minm, size_t maxm)
{   
    // note m = nstencil-1
    size_t start = get_stencil(x,minm+1,total, ntotal);
    double jump = eval_jump(x, total+start, evals+start, minm+1);
    int sign = 1;
    if (jump < 0.0){
        sign = -1;
    }
    size_t ii;
    double newjump;
    int newsign;
    for (ii = minm+1; ii <= maxm; ii++){
        newsign = 1;
        start = get_stencil(x,ii+1, total, ntotal);
        newjump = eval_jump(x, total+start, evals+start,ii+1);
        if (newjump < 0.0){
            newsign = -1;
        }
        if (newsign != sign){
            return 0.0;
        }
        else if (sign > 0){
            if (newjump < jump){
                jump = newjump;
            }
        }
        else { // sign < 0
            if (newjump > jump){
                jump = newjump;
            }
        }
    }
    return jump;
}

/********************************************************//**
*   Check whether discontinuity exists based on polynomial annihilation 
*   of function values obtained at a sorted set (low to high) of input
*   
*   \param x [in] - location to check
*   \param total [in] - set of sorted points
*   \param evals [in] - function evaluations
*   \param ntotal [in] - number of points
*   \param minm [in] - minimum polynomial order to annihilate
*   \param maxm [in] - maximum polynomial order to annihilate
*
*   \return disc = 1 if discontinuity exists 0 if it does not
*************************************************************/
int minmod_disc_exists(double x, double * total, double * evals, 
                        size_t ntotal,size_t minm, size_t maxm)
{
    
    double jump = minmod_eval(x,total,evals,ntotal,minm,maxm);
    double h = total[1]-total[0];
    double diff;
    size_t ii = 2;
    for (ii = 2; ii < ntotal; ii++){
        diff = total[ii]-total[0];
        if (diff < h){
            h = diff;
        }
    }

    /* printf("jump=%G h=%G\n",jump,h); */
    double oom = floor(log10(h));
    /* double oom = ceil(log10(h)); */
    /* printf("oom=%G\n",oom); */
    int disc = 1;
    if (fabs(jump) <= pow(10.0,oom)){
    //if (fabs(jump) <= pow(10.0,oom)){
        disc= 0;   
    }
    //printf("(x,jump,tol,disc) = (%G,%G,%G,%d)\n",
    //                    x,jump,pow(10.0,oom),disc);
    return disc;
}

/********************************************************//**
*   Jump detector, locates jumps (discontinuities) in a one dimensional function
*   
*   \param f      [in]    - one dimensional function
*   \param lb     [in]    - lower bound of domain
*   \param ub     [in]    - upper bound of domain
*   \param nsplit [in]    - number of subdomains to split if disc possibly exists
*   \param tol    [in]    - closest distance between two function evaluations, defines the
*                           resolution
*   \param edges  [inout] - array of edges (location of jumps)
*   \param nEdge  [inout] - number of edges 
*
*************************************************************/
void locate_jumps(struct Fwrap * f,
                  double lb, double ub, size_t nsplit, double tol,
                  double ** edges, size_t * nEdge)
{
    size_t minm = 2;
    size_t maxm = 5; // > nsplit ? nsplit-1 : 8 ;
    /* size_t maxm = 5 >= nsplit ? nsplit-1 : 8 ; */
    /* size_t maxm = nsp; */

    int return_val  = 0;
    if ((ub-lb) < tol){
        //printf("add edge between (%G,%G)\n",lb,ub);
        double out = (ub+lb)/2.0;

        double * new_edge = realloc(*edges, (*nEdge+1)*sizeof(double));
        assert (new_edge != NULL);
        *edges = new_edge;
        (*edges)[*nEdge] = out;
        (*nEdge) = (*nEdge)+1;

    }
    else{
        printf("refine from %G - %G\n",lb,ub);
        double * pts = linspace(lb,ub,nsplit+1);
        dprint(nsplit+1,pts);
        double * vals = calloc_double(nsplit+1);
        size_t ii;
        return_val = fwrap_eval(nsplit+1,pts,vals,f);
        assert (return_val == 0);
        double x;
        int disc;
        double jump;
        printf("\n");
        for (ii = 0; ii < nsplit; ii++){
            x = (pts[ii] + pts[ii+1])/2.0;
            disc = minmod_disc_exists(x,pts,vals,nsplit+1,minm,maxm);
            jump = minmod_eval(x,pts,vals,nsplit+1,minm,maxm);
            printf("checking disc at %G = (%G,%d)\n",x,jump,disc);

            if (disc == 1){ // discontinuity potentially exists so refine
                printf("\tdisc exists so refine lb=%G,ub=%G\n",pts[ii],pts[ii+1]);
                locate_jumps(f,pts[ii],pts[ii+1],nsplit,tol,edges,nEdge);
            }
        }
        free(pts); pts = NULL;
        free(vals); vals = NULL;
    }
}

/********************************************************
*   Create Approximation by polynomial annihilation-based splitting
*   
*   \param fw      [in] - function to approximate
*   \param aoptsin [in] - approximation options
*
*   \return p - polynomial
*************************************************************/
/* struct PiecewisePoly * */
/* piecewise_poly_approx2(struct Fwrap * fw, */
/*                        struct PwPolyOpts * aoptsin) */
/* { */
/*     assert (aoptsin != NULL); */
/*     struct PwPolyOpts * aopts = aoptsin; */
/*     struct PiecewisePoly * p = NULL; */
    
/*     double lb = pw_poly_opts_get_lb(aopts); */
/*     double ub = pw_poly_opts_get_ub(aopts); */
    
/*     double * edges = NULL; */
/*     size_t nEdge = 0; */
/*     printf("locate jumps!\n"); */
/*     locate_jumps(fw,lb,ub,aopts->nregions,aopts->minsize,&edges,&nEdge); */
/*     printf("number of edges are %zu\n",nEdge); */
/*     printf("Edges are\n"); */
/*     size_t iii; */
/*     for (iii = 0; iii < nEdge;iii++){ */
/*        printf("(-buf, , buf) (%3.15G,%3.15G,%3.15G)\n", */
/*            edges[iii]-aopts->minsize,edges[iii],edges[iii]+aopts->minsize); */
/*     } */
/*     dprint(nEdge,edges); */
/*     // */

/*     size_t nNodes = nEdge*2+2; */
/*     double * nodes = calloc_double(nNodes); */
/*     nodes[0] = lb; */
/*     size_t ii,jj = 1; */
/*     for (ii = 0; ii < nEdge; ii++){ */
/*         nodes[jj] = edges[ii] - aopts->minsize; */
/*         nodes[jj+1] = edges[ii] + aopts->minsize; */
/*         jj += 2; */
/*     } */
/*     nodes[nEdge*2+1] = ub; */

/*     dprint(nNodes,nodes); */

/*     exit (1); */
/*     /\* struct PiecewisePoly * p =  *\/ */
/*     /\*     piecewise_poly_approx_from_nodes(f, args, aopts->ptype, *\/ */
/*     /\*                      nodes,nEdge*2+2, 2.0*aopts->minsize, &adopts); *\/ */
    
/*     free(edges); edges = NULL; */
/*     free(nodes); nodes = NULL; */
/*     return p; */
/* } */

/********************************************************//**
*   Create Approximation by hierarchical splitting
*   
*   \param[in] aopts - approximation options
*   \param[in] fw    - wrapped function
*
*   \return piecewise - polynomial
*************************************************************/
struct PiecewisePoly *
piecewise_poly_approx1(struct PwPolyOpts * aopts,
                       struct Fwrap * fw)
{

    size_t N = aopts->maxorder+1;
    struct PiecewisePoly * poly = piecewise_poly_alloc();
    /* printf("initializing poly nregions = %zu\n",aopts->nregions); */

    if (aopts->nregions == 1){
        poly->leaf = 1;
        poly->nbranches = 0;
        poly->ope = orth_poly_expansion_init(aopts->ptype, N, aopts->lb, aopts->ub);
        orth_poly_expansion_approx_vec(poly->ope,fw,aopts->opeopts);
        orth_poly_expansion_round(&(poly->ope));
    }
    else{
        /* printf("lb=%G,ub=%G,num=%zu\n",lb,ub,aopts->nregions); */
        poly->leaf = 0;
        double * pts = NULL;
        if (aopts->pts == NULL){
            pts = linspace(aopts->lb,aopts->ub,aopts->nregions+1);
            poly->nbranches = aopts->nregions;
        }
        else{
            poly->nbranches = aopts->npts-1;
        }
        poly->branches = piecewise_poly_array_alloc(poly->nbranches);

        /* printf("nbranches = %zu\n",poly->nbranches); */
        /* dprint(poly->nbranches+1,pts); */
        double clb,cub; 
        size_t ii;
        for (ii = 0; ii < poly->nbranches; ii++){
            if (aopts->pts == NULL){
                clb = pts[ii];
                cub = pts[ii+1];
            }
            else{
                clb = aopts->pts[ii];
                cub = aopts->pts[ii+1];
            }

            //printf("new upper = %G\n",cub);
            
            poly->branches[ii] = piecewise_poly_alloc();
            poly->branches[ii]->leaf = 1;

            if (aopts->opeopts == NULL){
                poly->branches[ii]->ope = 
                    orth_poly_expansion_init(aopts->ptype, N, clb, cub);
                orth_poly_expansion_approx_vec(poly->branches[ii]->ope,fw,aopts->opeopts);
            }
            else{
                ope_opts_set_lb(aopts->opeopts,clb);
                ope_opts_set_ub(aopts->opeopts,cub);
                poly->branches[ii]->ope = orth_poly_expansion_approx_adapt(aopts->opeopts,fw);
                ope_opts_set_lb(aopts->opeopts,pts[0]);
                ope_opts_set_ub(aopts->opeopts,pts[aopts->nregions]);
            }
            /* printf("coeffs = "); dprint(N,poly->branches[ii]->ope->coeff); */
            orth_poly_expansion_round(&(poly->branches[ii]->ope));
            /* printf("coeffs = "); dprint(N,poly->branches[ii]->ope->coeff); */
        }
        free(pts); pts = NULL;
    }

    /* if (aoptsin == NULL){ */
    /*     free(aopts); */
    /*     aopts = NULL; */
    /* } */

    //printf("end approx\n");
    return poly;
}


static void adapt_help(struct PiecewisePoly * pw, struct PwPolyOpts * aopts, struct Fwrap * fw)
{

    size_t N = aopts->maxorder+1;
    double normalization = piecewise_poly_inner(pw,pw);
    /* double normalization = 1; */

    double true_lb = piecewise_poly_get_lb(pw);
    double true_ub = piecewise_poly_get_ub(pw);

    /* printf("refining (%G,%G)\n",true_lb,true_ub); */
    /* printf("number of branches = %zu\n",pw->nbranches); */
    assert (pw->ope == NULL);
    int refined_once = 0;

    for (size_t ii = 0; ii < pw->nbranches; ii++){


        assert (pw->branches[ii]->leaf == 1);
        double lb = piecewise_poly_get_lb(pw->branches[ii]);
        double ub = piecewise_poly_get_ub(pw->branches[ii]);
        /* printf("\t refining branch %zu\n",ii); */
        /* printf("\tlb=%G,ub=%G,diff=%3.5E,minsize=%3.5E\n",lb,ub,ub-lb,aopts->minsize); */
        int refine = 0;
        if ( ( (ub-lb) < aopts->minsize) || (aopts->nregions == 1)){
            refine = 0;
        }
        else{
            size_t ncheck = aopts->coeff_check < N ? aopts->coeff_check : N;
            /* printf("\n\ncheck refine\n"); */
            for (size_t jj = 0; jj < ncheck; jj++){
                double c =  pw->branches[ii]->ope->coeff[N-1-jj];
                /* printf("coeff = %3.15E,sum=%3.15E,epsilon=%3.15E\n",c,normalization,aopts->epsilon); */
                /* printf("cleft = %3.15E,cright=%3.15E\n",c*c,normalization*aopts->epsilon); */
                if (c*c > (aopts->epsilon * normalization)){
                    refine = 1;
                    /* printf("refine!\n"); */
                    break;
                }
            }
        }
    
        if (refine == 1){
            refined_once = 1;
            /* printf("refining branch (%G,%G)\n",lb,ub); */
            /* printf("diff = %G, minsize = %G\n",ub-lb, aopts->minsize); */

            aopts->lb = lb;
            aopts->ub = ub;
            piecewise_poly_free(pw->branches[ii]); pw->branches[ii] = NULL;
            pw->branches[ii] = piecewise_poly_approx1(aopts,fw);
            aopts->lb = true_lb;
            aopts->ub = true_ub;
        }
    }

    piecewise_poly_flatten(pw);
    if (refined_once == 1){ // recurse
        /* printf("\t\t recurse!\n"); */
        adapt_help(pw,aopts,fw);
    }
    /* else{ */
    /*     printf("done!\n"); */
    /* } */
    
}

/********************************************************//**
*   Create Approximation by hierarchical splitting (adaptively)
*   
*   \param[in] aopts - approximation options
*   \param[in] fw    - function wrapper
*
*   \return piecewise polynomial
*************************************************************/
struct PiecewisePoly *
piecewise_poly_approx1_adapt(struct PwPolyOpts * aopts,
                             struct Fwrap * fw)
{
    assert (aopts != NULL);
    assert (aopts->pts == NULL);

    size_t N = aopts->maxorder+1;

    struct PiecewisePoly * poly = piecewise_poly_alloc();

    poly->leaf = 1;
    poly->nbranches = 0;
    poly->ope = orth_poly_expansion_init(aopts->ptype,N,aopts->lb,aopts->ub);
    orth_poly_expansion_approx_vec(poly->ope,fw,aopts->opeopts);
    orth_poly_expansion_round(&(poly->ope));
    /* poly->ope = orth_poly_expansion_approx_adapt(aopts->opeopts,fw); */

    int refine = 0;
    if ( ( (aopts->ub-aopts->lb) < aopts->minsize) || (aopts->nregions == 1)){
        refine = 0;
    }
    else{
        
        size_t npolys = N;
        size_t ncheck = aopts->coeff_check < npolys ? aopts->coeff_check : npolys;
        double normalization = cblas_ddot(npolys,poly->ope->coeff,1,poly->ope->coeff,1);
        /* double normalization = 1.0; */
        for (size_t jj = 0; jj < ncheck; jj++){
            double c =  poly->ope->coeff[npolys-1-jj];
            /* printf("coeff = %3.15E,sum=%3.15E,epsilon=%3.15E\n",c,sum,aopts->epsilon); */
            if (c*c > (aopts->epsilon * normalization)){
                refine = 1;
                break;
            }
        }
    }

    if (refine == 1){
        piecewise_poly_free(poly); poly = NULL;
        poly = piecewise_poly_approx1(aopts,fw);
        adapt_help(poly,aopts,fw);
    }

    /* ope_opts_free(aopts->opeopts); aopts->opeopts = NULL; */
    
    return poly;
    
}

////////////////////////////////////////////////////////

/********************************************************//**
*   Serialize pw polynomial
*   
*   \param[in]     ser       - location to which to serialize
*   \param[in]     p         - polynomial
*   \param[in,out] totSizeIn - if not null then only return total size of array 
                               without serialization! if NULL then serialiaze
*
*   \return ptr : pointer to end of serialization
*************************************************************/
unsigned char *
serialize_piecewise_poly(unsigned char * ser, 
                         struct PiecewisePoly * p,
                         size_t * totSizeIn)
{
    
    size_t totsize; 
    unsigned char * ptr = NULL;
    if (totSizeIn != NULL){
        if (p->leaf == 1){
            ptr = serialize_orth_poly_expansion(ser, p->ope, &totsize);
        }
        else{
            size_t tsize;
            size_t ii;
            ptr = serialize_piecewise_poly(ser, p->branches[0], &totsize);
            for (ii = 1; ii < p->nbranches; ii++){
                tsize = 0;
                ptr = serialize_piecewise_poly(ptr,p->branches[ii], &tsize);
                totsize += tsize;
            }
            totsize += sizeof(size_t); // for nbranches
        }
        *totSizeIn = totsize + sizeof(int); //sizeof(int) is for saying whether  it is a leaf or not
        return ptr;
    }
    else{
        int leaf;
        if (p->leaf == 1){
            leaf = 1;
            ptr = serialize_int(ser,leaf);
            ptr = serialize_orth_poly_expansion(ptr, p->ope, NULL);
        }
        else{
            leaf = 0;
            ptr = serialize_int(ser,leaf);
            ptr = serialize_size_t(ptr, p->nbranches);
            size_t ii;
            for (ii = 0; ii < p->nbranches; ii++){
                ptr = serialize_piecewise_poly(ptr,p->branches[ii], NULL);
            }
        }
    }
    return ptr;
}

/********************************************************//**
*   Deserialize pw polynomial
*
*   \param[in]     ser  - input string
*   \param[in,out] poly - pw polynomial
*
*   \return ptr - ser + number of bytes of poly expansion
*************************************************************/
unsigned char * 
deserialize_piecewise_poly(unsigned char * ser, 
                           struct PiecewisePoly ** poly)
{
    
    *poly = piecewise_poly_alloc();
    int leaf;
    unsigned char * ptr = deserialize_int(ser, &leaf);
    if (leaf == 1){
        (*poly)->leaf = 1;
        ptr = deserialize_orth_poly_expansion(ptr, &((*poly)->ope));
    }
    else{
        (*poly)->leaf = 0;
        //ptr = deserialize_int(ptr, &((*poly)->leaf));
        ptr = deserialize_size_t(ptr, &((*poly)->nbranches));
        (*poly)->branches = piecewise_poly_array_alloc((*poly)->nbranches);
        size_t ii;
        for (ii = 0; ii < (*poly)->nbranches; ii++){
            ptr = deserialize_piecewise_poly(ptr, &((*poly)->branches[ii]));
        }
    }
    
    return ptr;
}

void print_piecewise_poly(struct PiecewisePoly * pw, size_t prec, void *args, FILE* fp)
{
    if (pw->ope != NULL){
        print_orth_poly_expansion(pw->ope,prec,args,fp);
    }
    else{
        fprintf(fp, "Tree structure with %zu branches\n",pw->nbranches);
    }
}


/********************************************************//**
*   Save a generic function in text format
*   
*   \param[in]     p       - function to save
*   \param[in]     stream  - stream to save it to
*   \param[in,out] prec    - precision with which to save
*************************************************************/
void
piecewise_poly_savetxt(const struct PiecewisePoly * p, FILE * stream,
                       size_t prec)
{
    assert (p != NULL);
    if (p->leaf == 1){
        fprintf(stream,"%d ",p->leaf);
        orth_poly_expansion_savetxt(p->ope,stream,prec);
    }
    else{
        fprintf(stream,"%d ",p->leaf);
        fprintf(stream,"%zu ",p->nbranches);
        size_t ii;
        for (ii = 0; ii < p->nbranches; ii++){
            piecewise_poly_savetxt(p->branches[ii],stream,prec);
        }
    }
}


/********************************************************//**
*   Load a piecewise polynomial that is saved as a text file
*
*   \param[in] stream - stream to read
*
*   \return piecwise polynomial
*************************************************************/
struct PiecewisePoly * piecewise_poly_loadtxt(FILE * stream)
{
    
    struct PiecewisePoly * poly = piecewise_poly_alloc();
    int leaf;
    int num = fscanf(stream,"%d ",&leaf);
    assert (num == 1);
    if (leaf == 1){
        poly->leaf = 1;
        poly->ope = orth_poly_expansion_loadtxt(stream);
    }
    else{
        poly->leaf = 0;
        num = fscanf(stream,"%zu ",&(poly->nbranches));
        assert (num == 1);
        poly->branches = piecewise_poly_array_alloc(poly->nbranches);
        size_t ii;
        for (ii = 0; ii < poly->nbranches; ii++){
            poly->branches[ii] = piecewise_poly_loadtxt(stream);
        }
    }
    
    return poly;
}

size_t piecewise_poly_get_num_params(const struct PiecewisePoly * poly)
{
    (void)(poly);
    NOT_IMPLEMENTED_MSG("piecewise_poly_get_num_params")
    return 0;
}


/********************************************************//**
*   Update an expansion's parameters
*            
*   \param[in] ope     - expansion to update
*   \param[in] nparams - number of parameters
*   \param[in] param   - parameters

*   \returns 0 if successful
*************************************************************/
int
piecewise_poly_update_params(struct PiecewisePoly * ope,
                             size_t nparams, const double * param)
{
    (void) (ope);
    (void) (nparams);
    (void) (param);
    NOT_IMPLEMENTED_MSG("piecewise_poly_update_params")
    return 1;
}

/********************************************************//*
*   Evaluate the gradient 
*   with respect to the parameters
*
*   \param[in]     poly - polynomial expansion
*   \param[in]     nx   - number of x points
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N,nx)
*
*   \return 0 success, 1 otherwise
*************************************************************/
int piecewise_poly_param_grad_eval(
    const struct PiecewisePoly * poly, size_t nx, const double * x, double * grad)
{
    (void)(poly);
    (void)(nx);
    (void)(x);
    (void)(grad);
    
    NOT_IMPLEMENTED_MSG("piecewise_poly_param_grad_eval");
    return 1;
}

/********************************************************//*
*   Evaluate the gradient 
*   with respect to the parameters
*
*   \param[in]     poly - polynomial expansion
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N,nx)
*
*   \return evaluation
*************************************************************/
double piecewise_poly_param_grad_eval2(
    const struct PiecewisePoly * poly, double x, double * grad)
{
    (void)(poly);
    (void)(x);
    (void)(grad);

    NOT_IMPLEMENTED_MSG("piecewise_poly_param_grad_eval2");
    return 0.0;
}

/********************************************************//**
    Take a gradient of the squared norm 
    with respect to its parameters, and add a scaled version
    of this gradient to *grad*

    \param[in]     poly  - polynomial
    \param[in]     scale - scaling for additional gradient
    \param[in,out] grad  - gradient, on output adds scale * new_grad

    \return  0 - success, 1 -failure

************************************************************/
int
piecewise_poly_squared_norm_param_grad(const struct PiecewisePoly * poly,
                                       double scale, double * grad)
{
    (void)(poly);
    (void)(scale);
    (void)(grad);

    NOT_IMPLEMENTED_MSG("piecewise_poly_param_grad_eval2");
    return 1;
}


/********************************************************//**
*   Get parameters 
*************************************************************/
size_t piecewise_poly_get_params(const struct PiecewisePoly * pw, double * param)
{
    (void)(pw);
    (void)(param);
    NOT_IMPLEMENTED_MSG("piecewise_poly_get_params")
    return 0;
}

/********************************************************//**
*   Get parameters by reference
*************************************************************/
double * piecewise_poly_get_params_ref(const struct PiecewisePoly * pw, size_t *nparam)
{
    (void)(pw);
    (void)(nparam);
    NOT_IMPLEMENTED_MSG("piecewise_poly_get_params_ref")
    return NULL;
}

/********************************************************//**
*   Initialize a function with certain parameters
*            
*   \param[in] opts    - approximation options
*   \param[in] nparams - number of polynomials
*   \param[in] param   - parameters
*
*   \return p function
*************************************************************/
struct PiecewisePoly * 
piecewise_poly_create_with_params(struct PwPolyOpts * opts,
                                  size_t nparams, const double * param)
{
    (void)(opts);
    (void)(nparams);
    (void)(param);
    NOT_IMPLEMENTED_MSG("piecewise_poly_create_with_params\n");
    return NULL;
}
