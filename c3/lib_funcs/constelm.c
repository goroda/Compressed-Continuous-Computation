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






/** \file constelm.c
 * Provides routines for manipulating constant elements
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <assert.h>

#include "stringmanip.h"
#include "array.h"
#include "linalg.h"
#include "futil.h"
#include "constelm.h"

/** \struct ConstElemExpAopts
 * \brief Approximation options of ConstElemExp
 * \var ConstElemExpAopts::num_nodes
 * number of basis functions or nodes
 * \var ConstElemExpAopts::node_alloc
 * indicator whether nodes were self allocated
 * \var ConstElemExpAopts::nodes
 * nodes
 * \var ConstElemExpAopts::adapt
 * whether or not to adapt (0 or 1)
 * \var ConstElemExpAopts::lb
 * lower bound
 * \var ConstElemExpAopts::ub
 * upper bound
 * \var ConstElemExpAopts::delta
 * adaptation function value tolerance
 * \var ConstElemExpAopts::hmin
 * adaptation node spacing tolerance
 */
struct ConstElemExpAopts{

    size_t num_nodes;
    int node_alloc;
    double * nodes;
    int adapt;

    double lb;
    double ub;
    double delta;
    double hmin;
};

/********************************************************//**
    Allocate approximation arguments (by reference)

    \param[in] N - number of nodes
    \param[in] x - nodes

    \return approximation arguments
*************************************************************/
struct ConstElemExpAopts * const_elem_exp_aopts_alloc(size_t N, double * x)
{
    assert (x != NULL);
    struct ConstElemExpAopts * aopts = NULL;
    aopts = malloc(sizeof(struct ConstElemExpAopts));
    if (aopts == NULL){
        fprintf(stderr,"Memory error allocate ConstElemExpAopts\n");
        exit(1);
    }
    aopts->num_nodes = N;
    aopts->node_alloc = 0;
    aopts->nodes = x;
    aopts->lb = x[0];
    aopts->lb = x[N-1];
        
    aopts->adapt = 0;
    aopts->delta = DBL_MAX;
    aopts->hmin = DBL_MAX;
    return aopts;
}

/********************************************************//**
    Allocate approximation arguments for adaptation

    \param[in] N     - number of nodes
    \param[in] x     - starting nodes reference
    \param[in] lb    - lower bound lb==x[0] if x! NULL
    \param[in] ub    - upper bound ub==x[0] if x! NULL
    \param[in] delta - size of deviation from constant
    \param[in] hmin  - minimum spacing

    \return approximation arguments
*************************************************************/
struct ConstElemExpAopts *
const_elem_exp_aopts_alloc_adapt(size_t N, double * x,
                                 double lb, double ub,
                                 double delta, double hmin)
{
    struct ConstElemExpAopts * aopts = NULL;
    aopts = malloc(sizeof(struct ConstElemExpAopts));
    if (aopts == NULL){
        fprintf(stderr,"Memory error allocate ConstElemExpAopts\n");
        exit(1);
    }
    aopts->num_nodes= N;
    aopts->node_alloc = 0;
    aopts->lb = lb;
    aopts->ub = ub;
    if (N > 0){
        aopts->nodes = x;
    }
    else{
        aopts->nodes = NULL;
    }
    aopts->adapt = 1;
    aopts->delta = delta;
    aopts->hmin = hmin;
    return aopts;
}

/********************************************************//**
    Free memory allocated to approximation arguments

    \param[in,out] aopts - approximation arguments
*************************************************************/
void const_elem_exp_aopts_free(struct ConstElemExpAopts * aopts)
{
    if (aopts != NULL){
        if (aopts->node_alloc == 1){
            free(aopts->nodes); aopts->nodes = NULL;
        }
        free(aopts); aopts = NULL;
    }
}

/********************************************************//**
    (deep)Free memory allocated to approximation arguments

    \param[in,out] aopts - approximation arguments
*************************************************************/
void const_elem_exp_aopts_free_deep(struct ConstElemExpAopts ** aopts)
{
    if (*aopts != NULL){
        if ((*aopts)->node_alloc == 1){
            free((*aopts)->nodes); (*aopts)->nodes = NULL;
        }
        free(*aopts); *aopts = NULL;
    }
}


/********************************************************//**
    Get the lower bound
*************************************************************/
double const_elem_exp_aopts_get_lb(const struct ConstElemExpAopts * aopts)
{
    assert (aopts != NULL);
    return aopts->lb;
}

/********************************************************//**
    Get the upper bound
*************************************************************/
double const_elem_exp_aopts_get_ub(const struct ConstElemExpAopts * aopts)
{
    assert (aopts != NULL);
    return aopts->ub;
}

/********************************************************//**
    Get number of nodes
*************************************************************/
size_t const_elem_exp_aopts_get_num_nodes(const struct ConstElemExpAopts * aopts)
{
    assert (aopts != NULL);
    return aopts->num_nodes;
}

/********************************************************//**
    Sets new nodes (by reference) for approximation options.
    frees old ones if
    previously allocated

    \param[in,out] aopts - approximation arguments
    \param[in]     N     - number of nodes
    \param[in]     nodes - nodes
*************************************************************/
void const_elem_exp_aopts_set_nodes(struct ConstElemExpAopts * aopts,
                                  size_t N, double * nodes)
{

    if (aopts == NULL){
        fprintf(stderr,"Must allocate ConstElemExpAopts before setting nodes\n");
        exit(1);
    }
    if (aopts->node_alloc == 1){
        free(aopts->nodes); aopts->nodes = NULL;
    }
    aopts->num_nodes= N;
    aopts->node_alloc = 0;
    if (N > 0){
        aopts->nodes = nodes;
    }
}

/********************************************************//**
    Set adaptation

    \param[in,out] aopts - approximation arguments
    \param[in]     delta - maximum deviation from constant
    \param[in]     hmin  - minimum distance between nodes
*************************************************************/
void const_elem_exp_aopts_set_adapt(struct ConstElemExpAopts * aopts,
                                  double delta, double hmin)
{
    if (aopts == NULL){
        fprintf(stderr,"Must allocate ConstElemExpAopts before turning on adapting\n");
        exit(1);
    }
    aopts->adapt = 1;
    aopts->delta = delta;
    aopts->hmin = hmin;
}
/********************************************************//**
    Setting delta for adaptation

    \param[in,out] aopts - approximation arguments
    \param[in]     delta - maximum deviation from constant
*************************************************************/
void const_elem_exp_aopts_set_delta(struct ConstElemExpAopts * aopts, double delta)
{
    if (aopts == NULL){
        fprintf(stderr,"Must allocate ConstElemExpAopts before setting delta\n");
        exit(1);
    }
    aopts->delta = delta;
}

/********************************************************//**
    Setting hmin for adaptation

    \param[in,out] aopts - approximation arguments
    \param[in]     hmin  - minimum distance between nodes
*************************************************************/
void const_elem_exp_aopts_set_hmin(struct ConstElemExpAopts * aopts, double hmin)
{
    if (aopts == NULL){
        fprintf(stderr,"Must allocate ConstElemExpAopts before setting hmin\n");
        exit(1);
    }
    aopts->hmin = hmin;
}


///////////////////////////////////////////////


/********************************************************//**
*   Get number of free parameters
*
*   \note Can change this later to include knot locations
*************************************************************/
size_t const_elem_exp_aopts_get_nparams(const struct ConstElemExpAopts* lexp)
{
    assert (lexp != NULL);
    return lexp->num_nodes;
}

/********************************************************//**
*   Set number of free parameters
*
*   \note Can change this later to include knot locations
*************************************************************/
void const_elem_exp_aopts_set_nparams(struct ConstElemExpAopts* lexp, size_t num)
{
    assert (lexp != NULL);
    lexp->num_nodes = num;
    fprintf(stderr,"Warning: setting new nparams in constelem aopts. Do I need to adjust the node locations?\n");
}


/********************************************************//**
*   Allocate a Constant Element Expansion
*
*  \return  Allocated Expansion
*************************************************************/
struct ConstElemExp * const_elem_exp_alloc()
{
    
    struct ConstElemExp * p = NULL;
    if ( NULL == (p = malloc(sizeof(struct ConstElemExp)))){
        fprintf(stderr, "failed to allocate memory for ConstElemExp.\n");
        exit(1);
    }
    p->num_nodes = 0;
    p->nodes = NULL;
    p->coeff = NULL;
    p->diff = NULL;
    return p;
}

/********************************************************//**
    Make a copy of a constant element expansion

    \param[in] lexp - constant element expansion to copy

    \return constant element expansion
*************************************************************/
struct ConstElemExp * const_elem_exp_copy(const struct ConstElemExp * lexp)
{
    
    struct ConstElemExp * p = NULL;
    if (lexp != NULL){
        p = const_elem_exp_alloc();
        p->num_nodes = lexp->num_nodes;
        if (lexp->nodes != NULL){
            p->nodes = calloc_double(p->num_nodes);
            memmove(p->nodes,lexp->nodes,p->num_nodes*sizeof(double));
        }
        if (lexp->coeff != NULL){
            p->coeff = calloc_double(p->num_nodes);
            memmove(p->coeff,lexp->coeff,p->num_nodes*sizeof(double));
        }
    }
    return p;
}

/********************************************************//**
*  Free a Constant Element Expansion
*
*  \param[in,out] exp - expansion to free
*************************************************************/
void const_elem_exp_free(struct ConstElemExp * exp)
{
    if (exp != NULL){
        free(exp->nodes); exp->nodes = NULL;
        free(exp->coeff); exp->coeff = NULL;
        free(exp->diff); exp->diff = NULL;
        free(exp); exp = NULL;
    }
}


/********************************************************//**
    Initialize a constant element expansion

    \param[in] num_nodes - number of nodes/basis functions
    \param[in] nodes     - nodes
    \param[in] coeff     - weights on nodes
  
    \return constant element expansion
    
    \note
    makes a copy of nodes and coefficients
*************************************************************/
struct ConstElemExp * const_elem_exp_init(size_t num_nodes, double * nodes,
                                      double * coeff)
{
    struct ConstElemExp * lexp = const_elem_exp_alloc();
    assert (num_nodes > 1);
    lexp->num_nodes = num_nodes;
    lexp->nodes = calloc_double(num_nodes);
    lexp->coeff = calloc_double(num_nodes);
    memmove(lexp->nodes,nodes,num_nodes*sizeof(double));
    memmove(lexp->coeff,coeff,num_nodes*sizeof(double));
    /* compute_diff(lexp); */
    return lexp;
}

/********************************************************//**
    Initialize a constant element expansion with particular parameters

    \param[in] opts  - options
    \param[in] dim   - number of parameters
    \param[in] param - parameters
  
    \return constant element expansion
    
    \note
    makes a copy of nodes and coefficients
*************************************************************/
struct ConstElemExp *
const_elem_exp_create_with_params(struct ConstElemExpAopts * opts,
                               size_t dim, const double * param)
{
    assert (opts != NULL);
    assert (opts->num_nodes == dim);
    assert (opts->nodes != NULL);
    struct ConstElemExp * lexp = const_elem_exp_alloc();
    
    lexp->num_nodes = dim;;
    lexp->nodes = calloc_double(dim);
    lexp->coeff = calloc_double(dim);

    memmove(lexp->nodes,opts->nodes,dim*sizeof(double));
    memmove(lexp->coeff,param,dim*sizeof(double));
    
    return lexp;
}

/********************************************************//**
    Get number of nodes
*************************************************************/
size_t const_elem_exp_get_num_nodes(const struct ConstElemExp * lexp)
{
    assert (lexp != NULL);
    return lexp->num_nodes;
}

/********************************************************//**
    Get number of params
*************************************************************/
size_t const_elem_exp_get_num_params(const struct ConstElemExp * lexp)
{
    assert (lexp != NULL);
    return lexp->num_nodes;
}

/********************************************************//**
    Get the parameters of a constant element expansion

    \param[in] lexp  - expansion
    \param[in] param - parameters

    \returns number of parameters
*************************************************************/
size_t const_elem_exp_get_params(const struct ConstElemExp * lexp, double * param)
{
    assert (lexp != NULL);
    memmove(param,lexp->coeff,lexp->num_nodes*sizeof(double));
    return lexp->num_nodes;
}

/********************************************************//**
    Get a reference to parameters of a constant element expansion

    \param[in]     lexp   - expansion
    \param[in,out] nparam - parameters

    \returns reference to parameters
*************************************************************/
double * const_elem_exp_get_params_ref(const struct ConstElemExp * lexp, size_t * nparam)
{
    assert (lexp != NULL);
    *nparam = lexp->num_nodes;
    return lexp->coeff;
}

/********************************************************//**
    Update the parameters (coefficients) for a constant element expansion

    \param[in] lexp  - expansion
    \param[in] dim   - number of parameters
    \param[in] param - parameters

    \returns 0 if sucessful
*************************************************************/
int
const_elem_exp_update_params(struct ConstElemExp * lexp,
                             size_t dim, const double * param)
{
    assert (lexp != NULL);
    assert (lexp->num_nodes == dim);
    for (size_t ii = 0; ii < dim; ii++){
        lexp->coeff[ii] = param[ii];
    }

    return 0;
}
    
/********************************************************//**
*   Serialize a ConstElemExp
*
*   \param[in]     ser       - location at which to serialize
*   \param[in]     f         - function to serialize 
*   \param[in,out] totSizeIn - if not NULL then return size of struct 
*                              if NULL then serialiaze
*
*   \return pointer to the end of the serialization
*************************************************************/
unsigned char *
serialize_const_elem_exp(unsigned char * ser, struct ConstElemExp * f,
                       size_t * totSizeIn)
{

    // 3 * sizeof(size_t)-- 1 for num_nodes, 2 for sizes of nodes and coeffs
    size_t totsize = 3*sizeof(size_t) + 2 * f->num_nodes*sizeof(double);

    if (totSizeIn != NULL){
        *totSizeIn = totsize;
        return ser;
    }

    unsigned char * ptr = serialize_size_t(ser, f->num_nodes);
    //serializing a pointer also serializes its size
    ptr = serialize_doublep(ptr, f->nodes, f->num_nodes);
    ptr = serialize_doublep(ptr, f->coeff, f->num_nodes);
    return ptr;      

}

/********************************************************//**
*   Deserialize a constant element expansion
*
*   \param[in]     ser - serialized structure
*   \param[in,out] f - function
*
*   \return ptr - ser + number of bytes of poly expansion
*************************************************************/
unsigned char * deserialize_const_elem_exp(unsigned char * ser, 
                                         struct ConstElemExp ** f)
{

    *f = const_elem_exp_alloc();
        
    unsigned char * ptr = ser;
    ptr = deserialize_size_t(ptr,&((*f)->num_nodes));
    ptr = deserialize_doublep(ptr, &((*f)->nodes), &((*f)->num_nodes));
    ptr = deserialize_doublep(ptr, &((*f)->coeff), &((*f)->num_nodes));

    return ptr;
}

/********************************************************//**
*   Get the index of the coefficient for the closest node
*************************************************************/
static size_t const_elem_exp_find_node(const struct ConstElemExp * f, double x)
{   

    size_t indmin = 0;
    size_t indmax = f->num_nodes-1;
    size_t indmid = indmin + (indmax - indmin)/2;
    
    while (indmid != indmin){
        if (fabs(x - f->nodes[indmid]) <= 1e-15){
            /* printf("eventually here!\n"); */
            return indmid;
        }
        else if (x < f->nodes[indmid]){
            indmax = indmid;
        }
        else { // x > f->nodes[indmid]
            indmin = indmid;
        }
        indmid = indmin + (indmax-indmin)/2;
        //  printf("indmid = %zu, indmin=%zu,indmax=%zu\n",indmid,indmin,indmax);
    }

    if ((x - f->nodes[indmin]) < (f->nodes[indmin+1] - x)){
        return indmin;
    }
    return indmin+1;

}

/********************************************************//**
*   Get value of the coefficient for the closest node
*************************************************************/
inline static double const_elem_exp_find_node_val(const struct ConstElemExp * f, double x)
{
    size_t ind = const_elem_exp_find_node(f,x);
    return f->coeff[ind];
}

/********************************************************//**
*   Evaluate the lin elem expansion
*
*   \param[in] f - function
*   \param[in] x - location
*
*   \return value
*************************************************************/
double const_elem_exp_eval(const struct ConstElemExp * f, double x)
{
    if ((x < f->nodes[0]) || (x > f->nodes[f->num_nodes-1])){
        return 0.0;
    }

    return const_elem_exp_find_node_val(f,x);

}

// evaluate derivative;
double const_elem_exp_deriv_eval(const struct ConstElemExp * f, double x)
{
    (void)(x);
    (void)(f);
    return 0.0;
}

/********************************************************//**
*   Evaluate the lin elem expansion
*
*   \param[in]     poly - function
*   \param[in]     N    - number of evaluations
*   \param[in]     x    - location at which to evaluate
*   \param[in]     incx - increment of x
*   \param[in,out] y    - allocated space for evaluations
*   \param[in]     incy - increment of y
************************************************************/
void const_elem_exp_evalN(const struct ConstElemExp * poly, size_t N,
                          const double * x, size_t incx, double * y, size_t incy)
{
    for (size_t ii = 0; ii < N; ii++){
        y[ii*incy] = const_elem_exp_eval(poly,x[ii*incx]);
    }
}


/********************************************************//**
*   Get the value of the expansion at a particular node
*
*   \param[in] f - function
*   \param[in] node - location
*
*   \return value
*************************************************************/
double const_elem_exp_get_nodal_val(const struct ConstElemExp * f, size_t node)
{
    assert (f != NULL);
    assert (node < f->num_nodes);
    return f->coeff[node];
}



/********************************************************//**
*   Take a derivative same nodes,
*
*   \param[in] f - function
*
*   \return integral
*************************************************************/
struct ConstElemExp * const_elem_exp_deriv(const struct ConstElemExp * f)
{

    struct ConstElemExp * le = const_elem_exp_init(f->num_nodes,
                                                   f->nodes,f->coeff);


    le->coeff[0] = (le->coeff[1] - le->coeff[0]) / (le->nodes[1] - le->nodes[0]);
    for (size_t ii = 1; ii < le->num_nodes-1; ii++){
        le->coeff[ii] = (le->coeff[ii+1] - le->coeff[ii]) / (le->nodes[ii+1] - le->nodes[ii]);
    }
    le->coeff[le->num_nodes-1] = (le->coeff[le->num_nodes-1] - le->coeff[le->num_nodes-2]) /
        (le->nodes[le->num_nodes-1] - le->nodes[le->num_nodes-2]);

    return le;
}

/********************************************************//**
*   Take a second derivative same nodes
*
*   \param[in] f - function
*************************************************************/
struct ConstElemExp * const_elem_exp_dderiv(const struct ConstElemExp * f)
{
    (void)(f);
    NOT_IMPLEMENTED_MSG("const_elem_exp_dderiv");
    exit(1);
}

/********************************************************//**
*   Take a second derivative and enforce periodic bc
*
*   \param[in] f - function
*************************************************************/
struct ConstElemExp * const_elem_exp_dderiv_periodic(const struct ConstElemExp * f)
{
    (void)(f);
    NOT_IMPLEMENTED_MSG("const_elem_exp_dderiv_periodic");
    exit(1);
}

/********************************************************//*
*   Evaluate the gradient of a constant element expansion 
*   with respect to the coefficients of each basis
*
*   \param[in]     f    - polynomial expansion
*   \param[in]     nx   - number of x points
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N,nx)
*
*   \return 0 for success
*************************************************************/
int const_elem_exp_param_grad_eval(
    struct ConstElemExp * f, size_t nx, const double * x, double * grad)
{

    size_t nparam = f->num_nodes;
    /* assert (nparam == lexp->nnodes); */
    for (size_t ii = 0; ii < nx; ii++){
        size_t indmin = const_elem_exp_find_node(f,x[ii]);

        /* printf("x = %G, indmin = %zu\n",x[ii],indmin); */
        for (size_t jj = 0; jj < indmin; jj++)
        {
            grad[ii*nparam+jj] = 0.0;
        }

        for (size_t jj = indmin+2; jj < nparam; jj++)
        {
            grad[ii*nparam+jj] = 0.0;
        }

        grad[ii*nparam+indmin] = 1.0;

    }

    return 0;
}

/********************************************************//*
*   Evaluate the gradient of an constant element expansion
*   with respect to the coefficients of each basis
*
*   \param[in]     f    - polynomial expansion
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N)
*
*   \return out - value
*************************************************************/
double const_elem_exp_param_grad_eval2(
    struct ConstElemExp * f, double x, double * grad)
{
    assert (grad != NULL);

    size_t nparam = const_elem_exp_get_num_params(f);
    size_t indmin = const_elem_exp_find_node(f,x);
    for (size_t jj = 0; jj < indmin; jj++)
    {
        grad[jj] = 0.0;
    }

    for (size_t jj = indmin+2; jj < nparam; jj++)
    {
        grad[jj] = 0.0;
    }

    grad[indmin] = 1.0;

    return f->nodes[indmin];
}

/********************************************************//**
    Take a gradient of the squared norm 
    with respect to its parameters, and add a scaled version
    of this gradient to *grad*

    \param[in]     f     - constant element expansion
    \param[in]     scale - scaling for additional gradient
    \param[in,out] grad  - gradient, on output adds scale * new_grad

    \return  0 - success, 1 -failure

************************************************************/
int
const_elem_exp_squared_norm_param_grad(const struct ConstElemExp * f,
                                     double scale, double * grad)
{
    (void)(f);
    (void)(scale);
    (void)(grad);
    assert(1 == 0);
    return 0;
}

/********************************************************//**
*   Integrate the Constant Element Approximation
*
*   \param[in] f - function
*
*   \return integral
*************************************************************/
double const_elem_exp_integrate(const struct ConstElemExp * f)
{

    assert (f->num_nodes>1 );
    double dx = (f->nodes[1]-f->nodes[0])/2.0;
    double dx2;
    double integral = f->coeff[0] * dx;
    for (size_t ii = 1; ii < f->num_nodes-1;ii++){
        dx2 = (f->nodes[ii+1]-f->nodes[ii])/2.0;
        integral += (f->coeff[ii] * (dx + dx2));
        dx = dx2;
    }
    dx = (f->nodes[f->num_nodes-1]-f->nodes[f->num_nodes-2])*0.5;
    integral += f->coeff[f->num_nodes-1] * dx ;
    return integral;
}

double const_elem_exp_integrate_weighted(const struct ConstElemExp * f)
{
    (void)(f);
    NOT_IMPLEMENTED_MSG("const_elem_exp_integrate_weighted");
    return 0.0;
}

/********************************************************//**
*   Determine if two functions have the same discretization
*
*   \param[in] f - function
*   \param[in] g - function
*
*   \return 1 - yes
*           0 - no
*************************************************************/
static int const_elem_sdiscp(const struct ConstElemExp * f,
                             const struct ConstElemExp * g)
{
    if (f->num_nodes != g->num_nodes){
        return 0;
    }
    for (size_t ii = 0; ii < f->num_nodes;ii++){
        if (fabs(f->nodes[ii] - g->nodes[ii]) > 1e-15){
            return 0;
        }
    }
    return 1;
}

/********************************************************//**
*   Inner product between two functions with the same discretizations
*
*   \param[in] N - number of nodes
*   \param[in] x - nodes
*   \param[in] f - coefficients of first function
*   \param[in] g - coefficients of second function
*
*   \return inner product
*************************************************************/
static double const_elem_exp_inner_same(size_t N, double * x,
                                        double * f, double * g)
{


    double dx = (x[1] - x[0])/2.0;
    double inner = f[0]*g[0]*dx;
    double dx2;
    for (size_t ii = 1; ii < N-1; ii++){
        dx2 = (x[ii+1] - x[ii])/2.0;
        inner += (f[ii]*g[ii]*(dx2+dx));
        dx = dx2;
    }
    dx = (x[N-1]-x[N-2])/2.0;
    inner += (f[N-1]*g[N-1]*dx);
    
    return inner;
}

/********************************************************//**
*   Interpolate two const element expansions onto the same grid
*   keeping only those nodes needed for inner product
*
*   \param[in]     f - function
*   \param[in]     g - function
*   \param[in,out] x - interpolated nodes 
*   \param[in,out] fvals - values of f
*   \param[in,out] gvals - values of g

*   \return number of nodes

*   \note
    x,fvals,gvals must all be previously allocaed with enough space
    at least f->num_nodes + g->num_nodes is enough for a 
    conservative allocation
*************************************************************/
static size_t const_elem_exp_same_interp(
    const struct ConstElemExp * f, const struct ConstElemExp * g, double * x,
    double * fvals, double * gvals)
{
    size_t nnodes = 0;
    size_t inodef = 0;
    size_t inodeg = 0;
    
    if (f->nodes[0] > g->nodes[g->num_nodes-1]){
        return 0;
    }
    if (g->nodes[0] > f->nodes[f->num_nodes-1])
    {
        return 0;
    }
    while(f->nodes[inodef+1] < g->nodes[0]){
        inodef++;
    }
    while (g->nodes[inodeg+1] < f->nodes[0]){
        inodeg++;
    }

    while ((inodef < f->num_nodes) && (inodeg < g->num_nodes)){
        if (fabs(f->nodes[inodef] - g->nodes[inodeg]) < 1e-15){
            x[nnodes] = f->nodes[inodef];
            fvals[nnodes] = f->coeff[inodef];
            gvals[nnodes] = g->coeff[inodeg];
            inodef++;
            inodeg++;
            nnodes++;
        }
        else if (f->nodes[inodef] < g->nodes[inodeg]){
            x[nnodes] = f->nodes[inodef];
            fvals[nnodes] = f->coeff[inodef];
            gvals[nnodes] = const_elem_exp_eval(g,x[nnodes]);
            inodef++;
            nnodes++;
        }
        else if (g->nodes[inodeg] < f->nodes[inodef]){
            x[nnodes] = g->nodes[inodeg];
            fvals[nnodes] = const_elem_exp_eval(f, x[nnodes]);
            gvals[nnodes] = g->coeff[inodeg];
            inodeg++;
            nnodes++;
        }
    }

    return nnodes;
}
/********************************************************//**
*   Inner product between two functions
*
*   \param[in] f - function
*   \param[in] g - function
*
*   \return inner product
*************************************************************/
double const_elem_exp_inner(const struct ConstElemExp * f,
                            const struct ConstElemExp * g)
{
    double value = 0.0;
    int samedisc = const_elem_sdiscp(f,g);
    if (samedisc == 1){
        value = const_elem_exp_inner_same(f->num_nodes,f->nodes,f->coeff, g->coeff);
    }
    else{
        double * x = calloc_double(f->num_nodes + g->num_nodes);
        double * fvals = calloc_double(f->num_nodes + g->num_nodes);
        double * gvals = calloc_double(f->num_nodes + g->num_nodes);        
        size_t N = const_elem_exp_same_interp(f,g,x,fvals,gvals);
        
        value = const_elem_exp_inner_same(N,x,fvals,gvals);
        
        free(x); x= NULL;
        free(fvals); fvals = NULL;
        free(gvals); gvals = NULL;
    }
    
    return value;
}

/********************************************************//**
   Add two functions with same discretization levels
    
   \param[in] a - scaled value
   \param[in] f - function
   \param[in,out] g - function

   \returns 0 successful
            1 error
            
   \note 
   Could be sped up by keeping track of evaluations
*************************************************************/
static int const_elem_exp_axpy_same(double a, const struct ConstElemExp * f,
                                  struct ConstElemExp * g)
{

    cblas_daxpy(g->num_nodes,a,f->coeff,1,g->coeff,1);
    return 0;
}


/********************************************************//**
*   Interpolate two constant element expansions onto the same grid
*
*   \param[in]     f - function
*   \param[in]     g - function
*   \param[in,out] x - interpolated nodes 

*   \return number of nodes

*   \note
    x must be previously allocaed with enough space
    at least f->num_nodes + g->num_nodes is enough for a 
    conservative allocation
*************************************************************/
static size_t const_elem_exp_interp_same_grid(
    const struct ConstElemExp * f,
    const struct ConstElemExp * g, double * x)
{
    size_t nnodes = 0;
    size_t inodef = 0;
    size_t inodeg = 0;
    
    while ((inodef < f->num_nodes) || (inodeg < g->num_nodes)){
        if (inodef == f->num_nodes){
            x[nnodes] = g->nodes[inodeg];
            inodeg++;
            nnodes++;
        }
        else if (inodeg == g->num_nodes){
            x[nnodes] = f->nodes[inodef];
            inodef++;
            nnodes++;
        }
        else if (fabs(f->nodes[inodef] - g->nodes[inodeg]) < 1e-15){
            x[nnodes] = f->nodes[inodef];
            inodef++;
            inodeg++;
            nnodes++;
        }
        else if (f->nodes[inodef] < g->nodes[inodeg]){
            x[nnodes] = f->nodes[inodef];
            inodef++;
            nnodes++;
        }
        else if (g->nodes[inodeg] < f->nodes[inodef]){
            x[nnodes] = g->nodes[inodeg];
            inodeg++;
            nnodes++;
        }
    }

    return nnodes;
}


/********************************************************//**
   Add two functions
    
   \param[in]     a - scaled value
   \param[in]     f - function
   \param[in,out] g - function

   \returns 0 successful
            1 error
            
   \note 
   Could be sped up by keeping track of evaluations
*************************************************************/
int const_elem_exp_axpy(double a, 
                        const struct ConstElemExp * f,
                        struct ConstElemExp * g)
{
    
    int res = 0;
    int samedisc = const_elem_sdiscp(f,g);
    if (samedisc == 1){
        res = const_elem_exp_axpy_same(a,f, g);
    }
    else{
        double * x = calloc_double(f->num_nodes+g->num_nodes);
        double * coeff = calloc_double(f->num_nodes+g->num_nodes);
        size_t num = const_elem_exp_interp_same_grid(f,g,x);
//        printf("interpolated!\n");
        for (size_t ii = 0; ii < num; ii++){
            coeff[ii] = a*const_elem_exp_eval(f,x[ii]) +
                        const_elem_exp_eval(g,x[ii]);
        }
//        printf("good\n");
        g->num_nodes = num;
        free(g->nodes); g->nodes = x;
//        printf("bad!\n");
        free(g->coeff); g->coeff = coeff;
//        printf("word\n");
    }
    return res;
}

/********************************************************//**
   Multiply two functions
    
   \param[in] f   - first function
   \param[in] g   - second function

   \returns product
            
   \note 
*************************************************************/
struct ConstElemExp * const_elem_exp_prod(const struct ConstElemExp * f,
                                          const struct ConstElemExp * g)
{
    
    int samedisc = const_elem_sdiscp(f,g);
    struct ConstElemExp * prod = NULL;    
    if (samedisc == 1){
        prod = const_elem_exp_copy(f);
        for (size_t ii = 0; ii < f->num_nodes; ii++){
            prod->coeff[ii]*= g->coeff[ii];
        }
    }
    else{
        assert (1 == 0);
    }

    return prod;
}

/********************************************************//**
    Compute the norm of a function

    \param[in] f - function
    
    \return norm
*************************************************************/
double const_elem_exp_norm(const struct ConstElemExp * f)
{
    double norm = const_elem_exp_inner(f,f);
    return sqrt(norm);
}

/********************************************************//**
    Compute the maximum of the function

    \param[in]     f - function
    \param[in,out] x - location of maximum
    
    \return value of maximum
*************************************************************/
double const_elem_exp_max(const struct ConstElemExp * f, double * x)
{
    double mval = f->coeff[0];
    *x = f->nodes[0];
    for (size_t ii = 1; ii < f->num_nodes;ii++){
        if (f->coeff[ii] > mval){
            mval = f->coeff[ii];
            *x = f->nodes[ii];
        }
    }
    return mval;
}

/********************************************************//**
    Compute the minimum of the function

    \param[in]     f - function
    \param[in,out] x - location of minimum
    
    \return value of minimum
*************************************************************/
double const_elem_exp_min(const struct ConstElemExp * f, double * x)
{
    double mval = f->coeff[0];
    *x = f->nodes[0];
    for (size_t ii = 1; ii < f->num_nodes;ii++){
        if (f->coeff[ii] < mval){
            mval = f->coeff[ii];
            *x = f->nodes[ii];
        }
    }
    return mval;
}

/********************************************************//**
    Compute the maximum of the absolute value function

    \param[in]     f       - function
    \param[in,out] x       - location of absolute value max
    \param[in]     size    - size of x variable (sizeof(double) or sizeof(size_t))
    \param[in]     optargs - optimization arguments
    
    \return value
*************************************************************/
double const_elem_exp_absmax(const struct ConstElemExp * f, void * x,
                           size_t size,
                           void * optargs)
{
    if (optargs == NULL){
        size_t dsize = sizeof(double);
        double mval = fabs(f->coeff[0]);
        if (size == dsize){
            *(double *)(x) = f->nodes[0];
        }
        else{
            *(size_t *)(x) = 0;
        }
        for (size_t ii = 1; ii < f->num_nodes;ii++){
            if (fabs(f->coeff[ii]) > mval){
                mval = fabs(f->coeff[ii]);
                if (size == dsize){
                    *(double *)(x) = f->nodes[ii];
                }
                else{
                    *(size_t *)(x) = ii;
                }
            }
        }
        return mval;
    }
    else{
        assert (size == sizeof(double));
        struct c3Vector * optnodes = optargs;
        double mval = fabs(const_elem_exp_eval(f,optnodes->elem[0]));
        *(double *)(x) = optnodes->elem[0];
        for (size_t ii = 0; ii < optnodes->size; ii++){
            double val = fabs(const_elem_exp_eval(f,optnodes->elem[ii]));
            if (val > mval){
                mval = val;
                *(double *)(x) = optnodes->elem[ii];
            }
        }
        return mval;
    }
}


/********************************************************//**
    Approximate a function

    \param[in] opts - approximation options
    \param[in] f    - function

    \return Approximated function
*************************************************************/
struct ConstElemExp * 
const_elem_exp_approx(struct ConstElemExpAopts * opts, struct Fwrap * f)
{

    assert(opts != NULL);
    struct ConstElemExp * lexp = const_elem_exp_alloc();
    if (opts->adapt == 0){
        assert (opts->nodes != NULL);

        // allocate nodes and coefficients
        size_t N = opts->num_nodes;
        lexp->num_nodes = N;
        lexp->nodes = calloc_double(N);
        lexp->coeff = calloc_double(N);

        // copy nodes from options
        memmove(lexp->nodes,opts->nodes,N*sizeof(double));
        fwrap_eval(N,lexp->nodes,lexp->coeff,f);

    }
    else{
        assert (1 == 0);
    }
    
    return lexp;
}

/********************************************************//**
    Return a zero function

    \param[in] opts        - extra arguments depending on function_class, sub_type, etc.
    \param[in] force_param - if == 1 then approximation will have the number of parameters
                                     defined by *get_nparams, for each approximation type
                             if == 0 then it may be more compressed

    \return p - zero function
************************************************************/
struct ConstElemExp * 
const_elem_exp_zero(const struct ConstElemExpAopts * opts, int force_param)
{

    struct ConstElemExp * lexp = NULL;
    if (force_param == 0){
        lexp = const_elem_exp_constant(0.0,opts);
    }
    else{
        lexp = const_elem_exp_alloc();    
        if (opts->num_nodes == 0){
            lexp->num_nodes = 2;
            lexp->nodes = linspace(opts->lb,opts->ub,2);
        }
        else{
            lexp->num_nodes = opts->num_nodes;
            lexp->nodes = calloc_double(opts->num_nodes);
            memmove(lexp->nodes,opts->nodes,opts->num_nodes*sizeof(double));
        }
        lexp->coeff = calloc_double(lexp->num_nodes);
        for (size_t ii = 0; ii < lexp->num_nodes; ii++){
            lexp->coeff[ii] = 0.0;
        }
    }
    return lexp;
}


/********************************************************//**
    Create a constant function

    \param[in] a  - function value
    \param[in] opts  - options

    \return function
*************************************************************/
struct ConstElemExp * 
const_elem_exp_constant(double a,
                        const struct ConstElemExpAopts * opts)
{
    
    struct ConstElemExp * lexp = const_elem_exp_alloc();
    if (opts->num_nodes == 0){
        lexp->num_nodes = 2;
        lexp->nodes = linspace(opts->lb,opts->ub,2);
    }
    else{
        lexp->num_nodes = opts->num_nodes;
        lexp->nodes = calloc_double(opts->num_nodes);
        memmove(lexp->nodes,opts->nodes,opts->num_nodes*sizeof(double));
    }
    lexp->coeff = calloc_double(lexp->num_nodes);
    for (size_t ii = 0; ii < lexp->num_nodes; ii++){
        lexp->coeff[ii] = a;
    }
    assert (lexp->num_nodes != 0);
    return lexp;
}

/********************************************************//**
    Return a linear function (NOT possible for const_elem)

    \param[in] a      - slope of the function
    \param[in] offset - offset of the function
    \param[in] opts  - extra arguments depending on function_class, 
                        sub_type, etc.

    \return gf - constant element function
*************************************************************/
struct ConstElemExp * 
const_elem_exp_linear(double a, double offset,
                      const struct ConstElemExpAopts * opts)
{
    (void)(a);
    (void)(offset);
    (void)(opts);
    NOT_IMPLEMENTED_MSG("const_elem_exp_linear");
    return NULL;
}

/*******************************************************//**
    Update a linear function

    \param[in] f      - existing linear function
    \param[in] a      - slope of the function
    \param[in] offset - offset of the function

    \returns 0 if successfull, 1 otherwise                   
***********************************************************/
int
const_elem_exp_linear_update(struct ConstElemExp * f,
                             double a, double offset)
{
    (void) f;
    (void) a;
    (void) offset;
    NOT_IMPLEMENTED_MSG("const_elem_exp_linear_update");
    return 1;
}

/********************************************************//**
    Return a quadratic function a * (x - offset)^2 = a (x^2 - 2offset x + offset^2)

    \param[in] a      - quadratic coefficients
    \param[in] offset - shift of the function
    \param[in] opts   - extra arguments depending on function_class, sub_type,  etc.

    \return quadratic function
*************************************************************/
struct ConstElemExp * 
const_elem_exp_quadratic(double a, double offset,
                        const struct ConstElemExpAopts * opts)
{
    (void)(a);
    (void)(offset);
    (void)(opts);
    NOT_IMPLEMENTED_MSG("const_elem_exp_quadratic");
    return NULL;
}

/********************************************************//**
    Multiply by -1

    \param[in,out] f - function
*************************************************************/
void const_elem_exp_flip_sign(struct ConstElemExp * f)
{
    for (size_t ii = 0; ii < f->num_nodes; ii++){
        f->coeff[ii] *= -1.0;
    }
    assert (f->num_nodes != 0);
}

/********************************************************//**
    Generate an orthonormal basis
    
    \param[in]     n    - number of basis function
    \param[in,out] f    - constant element expansions with allocated nodes
                          and coefficients set to zero
    \param[in]     opts - approximation options

    \note
    Uses modified gram schmidt to determine function coefficients
    Each function f[ii] must have the same nodes
*************************************************************/
void const_elem_exp_orth_basis(size_t n, struct ConstElemExp ** f,
                               struct ConstElemExpAopts * opts)
{
    assert (opts != NULL);

    if (opts->adapt == 0){
        assert (opts->nodes != NULL);
        /* assert (n <= opts->num_nodes); */
        double * zeros = calloc_double(opts->num_nodes);
        /* printf("n = %zu\n", n); */
        /* printf("num_nodes = %zu\n", opts->num_nodes); */
        for (size_t ii = 0; ii < n; ii++){
            f[ii] = const_elem_exp_init(opts->num_nodes,opts->nodes,zeros);
            if (ii < opts->num_nodes){
                f[ii]->coeff[ii] = 1.0;

                if (ii == 0){
                    f[ii]->coeff[ii] /= ((opts->nodes[1]-opts->nodes[0])/2);
                }
                /* else if (ii == n-1){ */
                else if (ii == (opts->num_nodes-1)){                    
                    /* f[ii]->coeff[ii] /= ((opts->nodes[n-1]-opts->nodes[n-2])/2); */
                    f[ii]->coeff[ii] /= ((opts->nodes[ii]-opts->nodes[ii-1])/2);
                }
                else{
                    double dx1 = (opts->nodes[ii] - opts->nodes[ii-1])/2;
                    double dx2 = (opts->nodes[ii+1] - opts->nodes[ii])/2; 
                    f[ii]->coeff[ii] /= (dx1+dx2);
                }
                f[ii]->coeff[ii] = sqrt(f[ii]->coeff[ii]);
            }
            else{
                f[ii]->coeff[0] = 1e-20;
            }                
        }
        /* for (size_t ii = 0; ii < n; ii++){ */
        /*     printf("Orth Function %zu = \n", ii); */
        /*     printf("\t"); dprint(f[ii]->num_nodes, f[ii]->nodes); */
        /*     printf("\t"); dprint(f[ii]->num_nodes, f[ii]->coeff); */
        /* } */
        free(zeros); zeros = NULL;
    }
    else{
        // not on a grid I can do whatever I want
        assert (n > 1);
        double * nodes = linspace(opts->lb,opts->ub,n);
        double * zeros = calloc_double(n);
        for (size_t ii = 0; ii < n; ii++){
            f[ii] = const_elem_exp_init(n,nodes,zeros);
            f[ii]->coeff[ii] = 1.0;
            if (ii == 0){
                f[ii]->coeff[ii] /= ((nodes[1]-nodes[0])/2);
            }
            else if (ii == n-1){
                f[ii]->coeff[ii] /= ((nodes[n-1]-nodes[n-2])/2);
            }
            else{
                double dx1 = (nodes[ii]-nodes[ii-1])/2;
                double dx2 = (nodes[ii+1]-nodes[ii])/2;                
                f[ii]->coeff[ii] /= (dx1+dx2);
            }
            f[ii]->coeff[ii] = sqrt(f[ii]->coeff[ii]);
        }
        free(zeros); zeros = NULL;
        free(nodes); nodes = NULL;
    }
}

/********************************************************//**
   Scale a function 

   \param[in]     a - value
   \param[in,out] f - function
*************************************************************/
void const_elem_exp_scale(double a, struct ConstElemExp * f)
{
    for (size_t ii = 0; ii < f->num_nodes; ii++){
        f->coeff[ii] *= a;
    }
}

/********************************************************//**
    Get lower bound

    \param[in] f - function

    \return lower bound
*************************************************************/
double const_elem_exp_get_lb(struct ConstElemExp * f)
{
    return f->nodes[0];
}

/********************************************************//**
    Get upper bound

    \param[in] f - function

    \return upper bound
*************************************************************/
double const_elem_exp_get_ub(struct ConstElemExp * f)
{
    return f->nodes[f->num_nodes-1];
}


static int compare_le (const void * a, const void * b)
{
    const double * aa = a;
    const double * bb = b;
    if (*aa > *bb){
        return 1;
    }
    else{
        return 0;
    }
}

/********************************************************//**
    Create a constant element function with zeros at particular
    locations and 1 everywhere else.

    \param[in] nzeros    - number of zeros
    \param[in] zero_locs - locations of zeros
    \param[in] opts      - constant expansion options

    \return upper bound
*************************************************************/
struct ConstElemExp *
const_elem_exp_onezero(size_t nzeros, double * zero_locs,
                     struct ConstElemExpAopts * opts)
{
    assert (opts != NULL);
    if (opts->adapt == 1){
        // can do whatever I want
        if (nzeros == 0){
            double * nodes = calloc_double(2);
            double * coeff = calloc_double(2);
            struct ConstElemExp * le = const_elem_exp_init(2,nodes,coeff);
            le->coeff[0] = 1.0;
            free(nodes); nodes = NULL;
            free(coeff); coeff = NULL;
            /* print_const_elem_exp(le,3,NULL,stdout); */
            return le;
        }
        else{
            struct ConstElemExp * le = NULL;
            double * sorted_arr = calloc_double(nzeros);
            memmove(sorted_arr,zero_locs,nzeros*sizeof(double));
            qsort (sorted_arr, nzeros, sizeof(double), compare_le);
            
            int onlb = 0;
            int onub = 0;
            double difflb = fabs(sorted_arr[0]-opts->lb);
            if (difflb < 1e-14){
                onlb = 1;
            }
            double diffub = fabs(zero_locs[nzeros-1]-opts->ub);
            if (diffub < 1e-14){
                onub = 1;
            }
            if ((onlb == 0) && (onub == 0)){
                double * nodes = calloc_double(nzeros+2);
                double * coeff = calloc_double(nzeros+2);
                nodes[0] = opts->lb;
                coeff[0] = 0.0;
                for (size_t jj = 1; jj < nzeros+1; jj++){
                    nodes[jj] = sorted_arr[jj-1];
                    coeff[jj] = 0.0;
                }
                nodes[nzeros+1] = opts->ub;
                coeff[nzeros+1] = 1.0;
                le = const_elem_exp_init(nzeros+2,nodes,coeff);
                free(nodes); nodes = NULL;
                free(coeff); coeff = NULL;
            }
            else if ((onlb == 1) && (onub == 1)){
                double * nodes = calloc_double(nzeros+1);
                double * coeff = calloc_double(nzeros+1);
                nodes[0] = opts->lb;
                coeff[0] = 0.0;
                nodes[1] = (opts->lb + sorted_arr[1])/2.0;
                coeff[1] = 1.0;
                for (size_t jj = 2; jj< nzeros+1; jj++){
                    nodes[jj] = sorted_arr[jj-1];
                    coeff[jj] = 0.0;
                }
                le = const_elem_exp_init(nzeros+1,nodes,coeff);
                free(nodes); nodes = NULL;
                free(coeff); coeff = NULL;
            }
            else if ((onlb == 1) && (onub == 0)){
                double * nodes = calloc_double(nzeros+1);
                double * coeff = calloc_double(nzeros+1);
                nodes[0] = opts->lb;
                coeff[0] = 0.0;
                for (size_t jj = 1; jj < nzeros; jj++){
                    nodes[jj] = sorted_arr[jj-1];
                    coeff[jj] = 0.0;
                }
                nodes[nzeros] = opts->ub;
                coeff[nzeros] = 1.0;
                le = const_elem_exp_init(nzeros+1,nodes,coeff);
                free(nodes); nodes = NULL;
                free(coeff); coeff = NULL;
            }
            else if ((onlb == 0) && (onub == 1)){
                double * nodes = calloc_double(nzeros+1);
                double * coeff = calloc_double(nzeros+1);
                nodes[0] = opts->lb;
                coeff[0] = 1.0;
                for (size_t jj = 1; jj< nzeros+1; jj++){
                    nodes[jj] = sorted_arr[jj-1];
                    coeff[jj] = 0.0;
                }
                le = const_elem_exp_init(nzeros+1,nodes,coeff);
                free(nodes); nodes = NULL;
                free(coeff); coeff = NULL;
            }
            else{
                assert (1 == 0);
            }
            /* print_const_elem_exp(le,3,NULL,stdout); */
            return le;
        }
    }
    else{
        assert(opts->nodes != NULL);
        assert(opts->num_nodes > nzeros);
        double * coeff = calloc_double(opts->num_nodes);
        struct ConstElemExp * le = const_elem_exp_init(opts->num_nodes, opts->nodes, coeff);
        for (size_t ii = 0; ii < opts->num_nodes; ii++){
            int nonzero = 1;
            for (size_t jj = 0; jj < nzeros; jj++){
                if (fabs(le->nodes[ii] - zero_locs[jj]) < 1e-14){
                    nonzero = 0;
                    break;
                }
            }
            if (nonzero == 1){
                le->coeff[ii] = 1.0;
            }
        }
        free(coeff);
        return le;
    }
}

/********************************************************//**
    Print a constant element function

    \param[in] f      - constant element function
    \param[in] args   - extra arguments (not used I think)
    \param[in] prec   - precision with which to save it
    \param[in] stream - stream to print to
************************************************************/
void print_const_elem_exp(const struct ConstElemExp * f, size_t prec, 
                        void * args, FILE * stream)
{
    (void) args;
    if (f == NULL){
        fprintf(stream, "Const Elem Expansion is NULL\n");
    }
    else{
        /* assert (args == NULL); */
        fprintf(stream, "Const Elem Expansion: num_nodes=%zu\n",f->num_nodes);
        for (size_t ii = 0; ii < f->num_nodes; ii++){
            if (prec < 100){
                fprintf(stream, "(%3.*G,%3.*G)  ",(int)prec,f->nodes[ii],
                        (int)prec,f->coeff[ii]);
            }
        }
        fprintf(stream,"\n");
    }
}

/********************************************************//**
    Save a constant element expansion in text format

    \param[in] f      - function to save
    \param[in] stream - stream to save it to
    \param[in] prec   - precision with which to save it
************************************************************/
void const_elem_exp_savetxt(const struct ConstElemExp * f,
                          FILE * stream, size_t prec)
{
    assert (f != NULL);
    fprintf(stream,"%zu ",f->num_nodes);
    for (size_t ii = 0; ii < f->num_nodes; ii++){
        if (prec < 100){
            fprintf(stream, "%3.*G ",(int)prec,f->nodes[ii]);
            fprintf(stream, "%3.*G ",(int)prec,f->coeff[ii]);
        }
    }
}

/********************************************************//**
    Load a constant element expansion in text format

    \param[in] stream - stream to save it to

    \return Constant Element Expansion
************************************************************/
struct ConstElemExp * const_elem_exp_loadtxt(FILE * stream)//, size_t prec)
{
    size_t num_nodes;

    int num = fscanf(stream,"%zu ",&num_nodes);
    assert (num == 1);
    /* printf("number of nodes read = %zu\n",num_nodes); */
    double * nodes = calloc_double(num_nodes);
    double * coeff = calloc_double(num_nodes);
    for (size_t ii = 0; ii < num_nodes; ii++){
        num = fscanf(stream,"%lG",nodes+ii);
        assert (num == 1);
        num = fscanf(stream,"%lG",coeff+ii);
        assert (num == 1);
    };
    struct ConstElemExp * lexp = const_elem_exp_init(num_nodes,nodes,coeff);
    free(nodes); nodes = NULL;
    free(coeff); coeff = NULL;
    return lexp;
}

