// Copyright (c) 2014-2016, Massachusetts Institute of Technology
// Copyright (c) 2016, Sandia National Laboratories

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

/** \file linelm.c
 * Provides routines for manipulating linear elements
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
#include "linelm.h"

/** \struct LinElemExpAopts
 * \brief Approximation options of LinElemExp
 * \var LinElemExpAopts::num_nodes
 * number of basis functions or nodes
 * \var LinElemExpAopts::node_alloc
 * indicator whether nodes were self allocated
 * \var LinElemExpAopts::nodes
 * nodes
 * \var LinElemExpAopts::adapt
 * whether or not to adapt (0 or 1)
 * \var LinElemExpAopts::lb
 * lower bound
 * \var LinElemExpAopts::ub
 * upper bound
 * \var LinElemExpAopts::delta
 * adaptation function value tolerance
 * \var LinElemExpAopts::hmin
 * adaptation node spacing tolerance
 */
struct LinElemExpAopts{

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
struct LinElemExpAopts * lin_elem_exp_aopts_alloc(size_t N, double * x)
{
    assert (x != NULL);
    struct LinElemExpAopts * aopts = NULL;
    aopts = malloc(sizeof(struct LinElemExpAopts));
    if (aopts == NULL){
        fprintf(stderr,"Memory error allocate LinElemExpAopts\n");
        exit(1);
    }
    aopts->num_nodes= N;
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
    \param[in] delta - size of deviation from linear
    \param[in] hmin  - minimum spacing

    \return approximation arguments
*************************************************************/
struct LinElemExpAopts *
lin_elem_exp_aopts_alloc_adapt(size_t N, double * x,
                               double lb, double ub,
                               double delta, double hmin)
{
    struct LinElemExpAopts * aopts = NULL;
    aopts = malloc(sizeof(struct LinElemExpAopts));
    if (aopts == NULL){
        fprintf(stderr,"Memory error allocate LinElemExpAopts\n");
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
void lin_elem_exp_aopts_free(struct LinElemExpAopts * aopts)
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
void lin_elem_exp_aopts_free_deep(struct LinElemExpAopts ** aopts)
{
    if (*aopts != NULL){
        if ((*aopts)->node_alloc == 1){
            free((*aopts)->nodes); (*aopts)->nodes = NULL;
        }
        free(*aopts); *aopts = NULL;
    }
}

/********************************************************//**
    Get number of nodes
*************************************************************/
size_t lin_elem_exp_aopts_get_num_nodes(const struct LinElemExpAopts * aopts)
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
void lin_elem_exp_aopts_set_nodes(struct LinElemExpAopts * aopts,
                                  size_t N, double * nodes)
{

    if (aopts == NULL){
        fprintf(stderr,"Must allocate LinElemExpAopts before setting nodes\n");
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
    \param[in]     delta - maximum deviation from linear
    \param[in]     hmin  - minimum distance between nodes
*************************************************************/
void lin_elem_exp_aopts_set_adapt(struct LinElemExpAopts * aopts,
                                  double delta, double hmin)
{
    if (aopts == NULL){
        fprintf(stderr,"Must allocate LinElemExpAopts before turning on adapting\n");
        exit(1);
    }
    aopts->adapt = 1;
    aopts->delta = delta;
    aopts->hmin = hmin;
}
/********************************************************//**
    Setting delta for adaptation

    \param[in,out] aopts - approximation arguments
    \param[in]     delta - maximum deviation from linear
*************************************************************/
void lin_elem_exp_aopts_set_delta(struct LinElemExpAopts * aopts, double delta)
{
    if (aopts == NULL){
        fprintf(stderr,"Must allocate LinElemExpAopts before setting delta\n");
        exit(1);
    }
    aopts->delta = delta;
}

/********************************************************//**
    Setting hmin for adaptation

    \param[in,out] aopts - approximation arguments
    \param[in]     hmin  - minimum distance between nodes
*************************************************************/
void lin_elem_exp_aopts_set_hmin(struct LinElemExpAopts * aopts, double hmin)
{
    if (aopts == NULL){
        fprintf(stderr,"Must allocate LinElemExpAopts before setting hmin\n");
        exit(1);
    }
    aopts->hmin = hmin;
}

///////////////////////////////////////////////

/********************************************************//**
    Get number of nodes
*************************************************************/
size_t lin_elem_exp_get_num_nodes(const struct LinElemExp * lexp)
{
    assert (lexp != NULL);
    return lexp->num_nodes;
}


/********************************************************//**
*   Allocate a Linear Element Expansion
*
*  \return  Allocated Expansion
*************************************************************/
struct LinElemExp * lin_elem_exp_alloc()
{
    
    struct LinElemExp * p = NULL;
    if ( NULL == (p = malloc(sizeof(struct LinElemExp)))){
        fprintf(stderr, "failed to allocate memory for LinElemExp.\n");
        exit(1);
    }
    p->num_nodes = 0;
    p->nodes = NULL;
    p->coeff = NULL;
    p->diff = NULL;
    p->inner = NULL;
    return p;
}

/********************************************************//**
    Make a copy of a linear element expansion

    \param[in] lexp - linear element expansion to copy

    \return linear element expansion
*************************************************************/
struct LinElemExp * lin_elem_exp_copy(struct LinElemExp * lexp)
{
    
    struct LinElemExp * p = NULL;
    if (lexp != NULL){
        p = lin_elem_exp_alloc();
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
*  Free a Linear Element Expansion
*
*  \param[in,out] exp - expansion to free
*************************************************************/
void lin_elem_exp_free(struct LinElemExp * exp)
{
    if (exp != NULL){
        free(exp->nodes); exp->nodes = NULL;
        free(exp->coeff); exp->coeff = NULL;
        free(exp->diff); exp->diff = NULL;
        free(exp); exp = NULL;
    }
}

/* static void compute_inner_helper(struct LinElemExp * exp) */
/* { */
/*     if (exp != NULL){ */
/*         if (exp->coeff != NULL){ */
/*             if (exp->diff != NULL){ */
/*                 free(exp->diff); exp->diff = NULL; */
/*             } */

/*             //left  */
/*             exp->diff = calloc_double(exp->num_nodes); */
/*             for (size_t ii = 0; ii < exp->num_nodes-1; ii++){ */
/*                 exp->diff[ii] = exp->coeff[ii+1]-exp->coeff[ii]; */
/*             } */
/*         } */
/*     } */
/* } */

/********************************************************//**
    Initialize a linear element expansion

    \param[in] num_nodes - number of nodes/basis functions
    \param[in] nodes     - nodes
    \param[in] coeff     - weights on nodes
  
    \return linear element expansion
    
    \note
    makes a copy of nodes and coefficients
*************************************************************/
struct LinElemExp * lin_elem_exp_init(size_t num_nodes, double * nodes,
                                      double * coeff)
{
    struct LinElemExp * lexp = lin_elem_exp_alloc();
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
    Initialize a linear element expansion with particular parameters

    \param[in] opts  - options
    \param[in] dim   - number of parameters
    \param[in] param - parameters
  
    \return linear element expansion
    
    \note
    makes a copy of nodes and coefficients
*************************************************************/
struct LinElemExp *
lin_elem_exp_create_with_params(struct LinElemExpAopts * opts,
                               size_t dim, const double * param)
{
    assert (opts != NULL);
    assert (opts->num_nodes == dim);
    assert (opts->nodes != NULL);
    struct LinElemExp * lexp = lin_elem_exp_alloc();
    
    lexp->num_nodes = dim;;
    lexp->nodes = calloc_double(dim);
    lexp->coeff = calloc_double(dim);

    memmove(lexp->nodes,opts->nodes,dim*sizeof(double));
    memmove(lexp->coeff,param,dim*sizeof(double));
    
    return lexp;
}

/********************************************************//**
    Get the parameters of a linear element expansion

    \param[in] lexp  - expansion
    \param[in] dim   - number of parameters
    \param[in] param - parameters

    \returns number of parameters
*************************************************************/
size_t lin_elem_exp_get_params(const struct LinElemExp * lexp, double * param)
{
    assert (lexp != NULL);
    memmove(param,lexp->coeff,lexp->num_nodes*sizeof(double));
    return lexp->num_nodes;
}

/********************************************************//**
    Update the parameters (coefficients) for a linear element expansion

    \param[in] lexp  - expansion
    \param[in] dim   - number of parameters
    \param[in] param - parameters
*************************************************************/
void
lin_elem_exp_update_params(struct LinElemExp * lexp,
                           size_t dim, const double * param)
{
    assert (lexp != NULL);
    assert (lexp->num_nodes == dim);
    for (size_t ii = 0; ii < dim; ii++){
        lexp->coeff[ii] = param[ii];
    }
}
    
/********************************************************//**
*   Serialize a LinElemExp
*
*   \param[in]     ser       - location at which to serialize
*   \param[in]     f         - function to serialize 
*   \param[in,out] totSizeIn - if not NULL then return size of struct 
*                              if NULL then serialiaze
*
*   \return pointer to the end of the serialization
*************************************************************/
unsigned char *
serialize_lin_elem_exp(unsigned char * ser, struct LinElemExp * f,
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
*   Deserialize a linear element expansion
*
*   \param[in]     ser - serialized structure
*   \param[in,out] f - function
*
*   \return ptr - ser + number of bytes of poly expansion
*************************************************************/
unsigned char * deserialize_lin_elem_exp(unsigned char * ser, 
                                         struct LinElemExp ** f)
{

    *f = lin_elem_exp_alloc();
        
    unsigned char * ptr = ser;
    ptr = deserialize_size_t(ptr,&((*f)->num_nodes));
    ptr = deserialize_doublep(ptr, &((*f)->nodes), &((*f)->num_nodes));
    ptr = deserialize_doublep(ptr, &((*f)->coeff), &((*f)->num_nodes));

    return ptr;
}

/********************************************************//**
*   Evaluate the lin elem expansion
*
*   \param[in] f - function
*   \param[in] x - location
*
*   \return value
*************************************************************/
double lin_elem_exp_eval(const struct LinElemExp * f, double x)
{
    if ((x < f->nodes[0]) || (x > f->nodes[f->num_nodes-1])){
        return 0.0;
    }
    size_t indmin = 0;
    size_t indmax = f->num_nodes-1;
    size_t indmid = indmin + (indmax - indmin)/2;
    
    while (indmid != indmin){
        if (fabs(x - f->nodes[indmid]) <= 1e-15){
            /* printf("eventually here!\n"); */
            return f->coeff[indmid];
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
    if (fabs(x - f->nodes[indmid]) <= 1e-15){
        /* printf("eventually here!\n"); */
        return f->coeff[indmid];
    }
    
    /* printf("\n\n\nx=%3.15G\n",x); */
    /* dprint(f->num_nodes,f->nodes); */
    /* for (size_t ii = 0; ii < f->num_nodes; ii++){ */
    /*     double diff = fabs(x-f->nodes[ii]); */
    /*     fprintf(stdout,"diff[%zu] = %3.15G, diff<=1e-15=%d\n",ii,diff,diff<1e-15); */
    /* } */
    /* printf("indmid = %zu\n",indmid); */
    /* printf("should not be here!\n"); */
    /* exit(1); */
//    printf("indmin = %zu,x=%G\n",indmin,x);
   
    double den = f->nodes[indmin+1]-f->nodes[indmin];
    double t = (f->nodes[indmid+1]-x)/den;

    double value = f->coeff[indmin] * t + f->coeff[indmin+1]*(1.0-t);
    return value;
}
/********************************************************//**
*   Evaluate the lin elem expansion
*
*   \param[in]     f    - function
*   \param[in]     N    - number of evaluations
*   \param[in]     x    - location at which to evaluate
*   \param[in]     incx - increment of x
*   \param[in,out] y    - allocated space for evaluations
*   \param[in]     incy - increment of y
************************************************************/
void lin_elem_exp_evalN(const struct LinElemExp * poly, size_t N,
                        const double * x, size_t incx, double * y, size_t incy)
{
    for (size_t ii = 0; ii < N; ii++){
        y[ii*incy] = lin_elem_exp_eval(poly,x[ii*incx]);
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
double lin_elem_exp_get_nodal_val(const struct LinElemExp * f, size_t node)
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
struct LinElemExp * lin_elem_exp_deriv(const struct LinElemExp * f)
{
    struct LinElemExp * le = lin_elem_exp_init(f->num_nodes,
                                               f->nodes,f->coeff);

    assert(f->num_nodes > 1);
    if (f->num_nodes <= 5){
        //first order for first part
        le->coeff[0] = (f->coeff[1]-f->coeff[0])/
            (f->nodes[1]-f->nodes[0]);
        // second order centeral
        for (size_t ii = 1; ii < le->num_nodes-1; ii++){
            le->coeff[ii] = (f->coeff[ii+1]-f->coeff[ii-1])/
                (f->nodes[ii+1]-f->nodes[ii-1]);
        }
        // first order last
        le->coeff[le->num_nodes-1] = 
            (f->coeff[le->num_nodes-1] - f->coeff[le->num_nodes-2]) / 
            (f->nodes[f->num_nodes-1] - f->nodes[f->num_nodes-2]);
    }
    else{ // mostly fourth order

        //first order for first part
        le->coeff[0] = (f->coeff[1]-f->coeff[0])/
                       (f->nodes[1]-f->nodes[0]);
        // second order
        le->coeff[1] = (f->coeff[2]-f->coeff[0]) / (f->nodes[2]-f->nodes[0]);

        // fourth order central
        for (size_t ii = 2; ii < le->num_nodes-2; ii++){
            double sum_all = f->nodes[ii+2]-f->nodes[ii-2];
            double sum_mid = f->nodes[ii+1]-f->nodes[ii-1];
            double r_num = pow(f->nodes[ii+2]-f->nodes[ii],3) +
                pow(f->nodes[ii] - f->nodes[ii-2],3);
            double r_den = pow(f->nodes[ii+1]-f->nodes[ii],3) + 
                pow(f->nodes[ii]-f->nodes[ii-1],3);
          
            double r = r_num/r_den;
        
            double den = r*sum_mid - sum_all;
            double num = r*(f->coeff[ii+1]-f->coeff[ii-1]) - 
                           f->coeff[ii+2] + f->coeff[ii-2];
            le->coeff[ii] = num/den;
        }

        // second order
        le->coeff[f->num_nodes-2] = (f->coeff[f->num_nodes-1] - 
                                     f->coeff[f->num_nodes-3]) / 
                                    (f->nodes[f->num_nodes-1] - 
                                     f->nodes[f->num_nodes-3]);

        // first order last
        le->coeff[le->num_nodes-1] = 
            (f->coeff[le->num_nodes-1] - f->coeff[le->num_nodes-2]) / 
            (f->nodes[f->num_nodes-1] - f->nodes[f->num_nodes-2]);
    }

    return le;
}

/********************************************************//*
*   Evaluate the gradient of an orthonormal polynomial expansion 
*   with respect to the coefficients of each basis
*
*   \param[in]     poly - polynomial expansion
*   \param[in]     nx   - number of x points
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N,nx)
*
*   \return out - polynomial value
*************************************************************/
int lin_elem_exp_param_grad_eval(
    struct LinElemExp * lexp, size_t nx, const double * x, double * grad)
{
    (void)(lexp);
    (void)(x);
    (void)(grad);
    (void)(nx);
    /* for (size_t ii = 0; ii < nx; ii++){ */
        fprintf(stderr,"Lin_elem_exp_param_grad_eval IS NOT YET IMPLEMENTED\n");
        exit(1);
    /* } */

    /* return 0; */
}

/********************************************************//**
*   Integrate the Linear Element Approximation
*
*   \param[in] f - function
*
*   \return integral
*************************************************************/
double lin_elem_exp_integrate(const struct LinElemExp * f)
{

    assert (f->num_nodes>1 );
    double dx = f->nodes[1]-f->nodes[0];
    double integral = f->coeff[0] * dx * 0.5;
    for (size_t ii = 1; ii < f->num_nodes-1;ii++){
        dx = f->nodes[ii+1]-f->nodes[ii-1];
        integral += f->coeff[ii] * dx * 0.5;
    }
    dx = f->nodes[f->num_nodes-1]-f->nodes[f->num_nodes-2];
    integral += f->coeff[f->num_nodes-1] * dx * 0.5;
    return integral;
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
static int lin_elem_sdiscp(const struct LinElemExp * f,
                           const struct LinElemExp * g)
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
*   Compute Integrals necessary for inner products for an element
*   \f[
*    left = \int_{x_1}^{x_2} (x_2-x)^2 dx
*   \f]
*   \f[
*    mixed = \int_{x_1}^{x_2} (x_2-x)(x-x_1) dx
*   \f]
*   \f[
*    right = \int_{x_1}^{x_2} (x-x_1)^2 dx
*   \f]
*  
*   \param[in]     x1 - left boundary of element
*   \param[in]     x2 - right boundary of element
*   \param[in,out] left - first integral
*   \param[in,out] mixed - second integral
*   \param[in,out] right - third integral
*************************************************************/
static void lin_elem_exp_inner_element(
    double x1, double x2, double * left, double * mixed, 
    double * right)
{
    double dx = (x2-x1);
    double x2sq = x2*x2;
    double x1sq = x1*x1;
    double x2cu = x2sq * x2;
    double x1cu = x1sq * x1;
    double dx2 = x2sq - x1sq; //pow(x2,2)-pow(x1,2);
    double dx3 = (x2cu - x1cu)/3.0; //(pow(x2,3)-pow(x1,3))/3.0;
          
    *left = x2sq*dx - x2*dx2 + dx3; // pow(x2,2)*dx - x2*dx2 + dx3;
    *mixed = (x2+x1) * dx2/2.0 - x1*x2*dx - dx3;
    *right = dx3 - x1*dx2 + x1sq * dx; //pow(x1,2)*dx;
}

/********************************************************//**
*   Interpolate two linear element expansions onto the same grid
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
static size_t lin_elem_exp_inner_same_grid(
    const struct LinElemExp * f, const struct LinElemExp * g, double * x,
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
            gvals[nnodes] = lin_elem_exp_eval(g,x[nnodes]);
            inodef++;
            nnodes++;
        }
        else if (g->nodes[inodeg] < f->nodes[inodef]){
            x[nnodes] = g->nodes[inodeg];
            fvals[nnodes] = lin_elem_exp_eval(f, x[nnodes]);
            gvals[nnodes] = g->coeff[inodeg];
            inodeg++;
            nnodes++;
        }
    }
    //printf("nodes are "); dprint(nnodes,x);
//    printf("fvalues are "); dprint(nnodes,fvals);
//    printf("gvalues are "); dprint(nnodes, gvals);
    return nnodes;
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
static double lin_elem_exp_inner_same(size_t N, double * x,
                                      double * f, double * g)
{
    double value = 0.0;
    double left,mixed,right,dx2;
    for (size_t ii = 0; ii < N-1; ii++){
        dx2 = (x[ii+1]-x[ii])*(x[ii+1] - x[ii]);
        /* dx2 = f->diff[ii] * f->diff[ii];//(x[ii+1]-x[ii])*(x[ii+1] - x[ii]); */
        lin_elem_exp_inner_element(x[ii],x[ii+1],&left,&mixed,&right);
        value += (f[ii] * g[ii]) / dx2 * left +
                 (f[ii] * g[ii+1]) / dx2 * mixed +
                 (g[ii] * f[ii+1]) / dx2 * mixed +
                 (f[ii+1] * g[ii+1]) / dx2 * right;
    }    
    return value;
}

/********************************************************//**
*   Inner product between two functions
*
*   \param[in] f - function
*   \param[in] g - function
*
*   \return inner product
*************************************************************/
double lin_elem_exp_inner(const struct LinElemExp * f,
                          const struct LinElemExp * g)
{

    double value = 0.0;
    int samedisc = lin_elem_sdiscp(f,g);
    if (samedisc == 1){
//        printf("here?!\n");
        /* if (f->diff == NULL){ */
        /*     compute_diff(f); */
        /* } */
        /* if (g->diff == NULL){ */
        /*     compute_diff(g); */
        /* } */
        value = lin_elem_exp_inner_same(f->num_nodes,f->nodes,
                                        f->coeff, g->coeff);
    }
    else{
        assert ( (f->num_nodes + g->num_nodes) < 1000);
        double xnew[1000];
        double fnew[1000];
        double gnew[1000];
//        printf("here\n");

        size_t nnodes = lin_elem_exp_inner_same_grid(f,g,
                                                     xnew,fnew,gnew);
//        printf("nnodes = %zu\n",nnodes);
        if (nnodes > 2){
            //          printf("nnodes = %zu\n",nnodes);
            // dprint(nnodes,xnew);
            //dprint(nnodes,fnew);
            //dprint(nnodes,gnew);
            value = lin_elem_exp_inner_same(nnodes,xnew,fnew,gnew);
        }
        else{
            printf("weird =\n");

            printf("f = \n");
            print_lin_elem_exp(f,3,NULL,stdout);
            printf("g = \n");
            print_lin_elem_exp(g,3,NULL,stdout);
            
            assert(1 == 0);
        }
        //       printf("there\n");
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
static int lin_elem_exp_axpy_same(double a, const struct LinElemExp * f,
                                  struct LinElemExp * g)
{

    cblas_daxpy(g->num_nodes,a,f->coeff,1,g->coeff,1);
    /* for (size_t ii = 0; ii < g->num_nodes; ii++){ */
    /*     g->coeff[ii] += a*f->coeff[ii]; */
    /* } */
    return 0;
}


/********************************************************//**
*   Interpolate two linear element expansions onto the same grid
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
static size_t lin_elem_exp_interp_same_grid(
    const struct LinElemExp * f,
    const struct LinElemExp * g, double * x)
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
    
   \param[in] a - scaled value
   \param[in] f - function
   \param[in,out] g - function

   \returns 0 successful
            1 error
            
   \note 
   Could be sped up by keeping track of evaluations
*************************************************************/
int lin_elem_exp_axpy(double a, 
                      const struct LinElemExp * f,
                      struct LinElemExp * g)
{
    
    int res = 0;
    int samedisc = lin_elem_sdiscp(f,g);
    if (samedisc == 1){
        res = lin_elem_exp_axpy_same(a,f, g);
    }
    else{
        double * x = calloc_double(f->num_nodes+g->num_nodes);
        double * coeff = calloc_double(f->num_nodes+g->num_nodes);
        size_t num = lin_elem_exp_interp_same_grid(f,g,x);
//        printf("interpolated!\n");
        for (size_t ii = 0; ii < num; ii++){
            coeff[ii] = a*lin_elem_exp_eval(f,x[ii]) +
                        lin_elem_exp_eval(g,x[ii]);
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

struct lefg
{
    const struct LinElemExp * f;
    const struct LinElemExp * g;
};

static int leprod(size_t N, const double * x,double * out, void * arg)
{
    struct lefg * fg = arg;
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = lin_elem_exp_eval(fg->f,x[ii]);
        out[ii] *= lin_elem_exp_eval(fg->g,x[ii]);        
    }
    return 0;
}

/********************************************************//**
   Multiply two functions
    
   \param[in] f   - first function
   \param[in] g   - second function
   \param[in] arg - extra arguments (not yet implemented)

   \returns product
            
   \note 
*************************************************************/
struct LinElemExp * lin_elem_exp_prod(const struct LinElemExp * f,
                                      const struct LinElemExp * g,
                                      void * arg)
{

    (void)(arg);
    double lb = f->nodes[0] < g->nodes[0] ? g->nodes[0] : f->nodes[0];
    double ub = f->nodes[f->num_nodes-1] > g->nodes[g->num_nodes-1] ? g->nodes[g->num_nodes-1] : f->nodes[f->num_nodes-1];

    struct lefg fg;
    fg.f = f;
    fg.g = g;
    struct LinElemExpAopts * opts = NULL;
    double hmin = 1e-3;
    double delta = 1e-4;
    opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,lb,ub,delta,hmin);

    struct Fwrap * fw = fwrap_create(1,"general-vec");
    fwrap_set_fvec(fw,leprod,&fg);
    
    struct LinElemExp * prod = NULL;    
    prod = lin_elem_exp_approx(opts,fw);
    fwrap_destroy(fw);
    
    lin_elem_exp_aopts_free(opts);
    return prod;
}

/********************************************************//**
    Compute the norm of a function

    \param[in] f - function
    
    \return norm
*************************************************************/
double lin_elem_exp_norm(const struct LinElemExp * f)
{
    double norm = lin_elem_exp_inner(f,f);
    return sqrt(norm);
}

/********************************************************//**
    Compute the maximum of the function

    \param[in]     f - function
    \param[in,out] x - location of maximum
    
    \return value of maximum
*************************************************************/
double lin_elem_exp_max(const struct LinElemExp * f, double * x)
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
double lin_elem_exp_min(const struct LinElemExp * f, double * x)
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
double lin_elem_exp_absmax(const struct LinElemExp * f, void * x,
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
        double mval = fabs(lin_elem_exp_eval(f,optnodes->elem[0]));
        *(double *)(x) = optnodes->elem[0];
        for (size_t ii = 0; ii < optnodes->size; ii++){
            double val = fabs(lin_elem_exp_eval(f,optnodes->elem[ii]));
            if (val > mval){
                mval = val;
                *(double *)(x) = optnodes->elem[ii];
            }
        }
        return mval;
    }
}

/********************************************************//**
    Compute an error estimate 

    \param[in]     f    - function
    \param[in,out] errs - errors for each element
    \param[in]     dir  - direction (1 for forward, -1 for backward)
    \param[in]     type - 0 for Linf, 2 for L2
    
    \return maximum error

    \note 
    The error estimate consists of finite difference approximations
    to the second derivative of the function
    Left boundary always uses forward difference and 
    right boundary always uses backward
*************************************************************/
double lin_elem_exp_err_est(struct LinElemExp * f, double * errs, short dir, short type)
{
    assert (f->num_nodes > 2);
    double value = 0.0;

    double dx = (f->nodes[1]-f->nodes[0]);
    double dx2 = (f->nodes[2]-f->nodes[1]);
    double m1 = (f->coeff[1]-f->coeff[0])/dx;
    double mid1 = (f->nodes[1] + f->nodes[0])/2.0;
    double m2 = (f->coeff[2]-f->coeff[1])/dx2;
    double mid2 = (f->nodes[2] + f->nodes[1])/2.0;
    double err = (m2-m1)/(mid2-mid1);
    if (type == 0){
        errs[0] = pow(dx,2)*fabs(err)/8.0;
        value = errs[0];
    }
    else if (type == 2){
        errs[0] = pow(dx,2)*sqrt(pow(err,2)*dx)/8.0;
        value += errs[0];
    }
    dx = dx2;
    double m3,mid3;
    for (size_t ii = 2; ii < f->num_nodes-1; ii++){
        dx2 = (f->nodes[ii+1]-f->nodes[ii]);
        m3 = (f->coeff[ii+1]-f->coeff[ii]);
        mid3 = (f->nodes[ii+1] + f->nodes[ii])/2.0;
        if (dir == 1){
            err = (m3-m2)/(mid3-mid2);
        }
        else{
            err = (m2-m1)/(mid2-mid1);
        }
        if (type == 0){
            errs[ii-1] = pow(dx,2)*fabs(err)/8.0;
            if (errs[ii-1] > value){
                value = errs[ii-1];
            }
        }
        else if (type == 2){
            errs[ii-1] = pow(dx,2)*sqrt(pow(err,2)*dx)/8.0;
            value += errs[ii-1];
        }

        m1 = m2;
        mid1 = mid2;
        m2 = m3;
        mid2 = mid3;
        dx = dx2;
    }
    err = (m2-m1)/(mid2-mid1);
    size_t ii = f->num_nodes-1;
    if (type == 0){
        errs[ii-1] = pow(dx,2)*fabs(err)/8.0;
        if (errs[ii-1] > value){
            value = errs[ii-1];
        }
    }
    else if (type == 2){
        errs[ii-1] = pow(dx,2)*sqrt(pow(err,2)*dx)/8.0;
        value += errs[ii-1];
    }

    return value;
}

struct LinElemXY
{
    double x;
    double y;
    struct LinElemXY * next;
};

struct LinElemXY * xy_init(double x, double y)
{
    struct LinElemXY * xy = malloc(sizeof(struct LinElemXY));
    if (xy == NULL){
        fprintf(stderr,"Cannot allocate LinElemXY struct for lin element adapting\n");
        exit(1);
    }
    
    xy->x = x;
    xy->y = y;
    xy->next = NULL;
    return xy;
}

void xy_append(struct LinElemXY ** xy, double x, double y)
{

    if (*xy == NULL){
        /* printf("xy is null so initialize\n"); */
        *xy = xy_init(x,y);
        /* printf("got it\n"); */
    }
    else{
        /* printf("xy == NULL = %d\n",xy==NULL); */
        /* printf("xy is not null!\n"); */
        struct LinElemXY * temp = *xy;
        /* printf("iterate\n"); */
        while (temp->next != NULL){
            /* printf("iterate\n"); */
            temp = temp->next;
        }
        /* printf("done iterating\n"); */
        temp->next = xy_init(x,y);

        /* if ((*xy)->next == NULL){ */
        /*     printf("next is null!\n"); */
        /*     (*xy)->next = xy_init(x,y); */
        /* } */
        /* else{ */
        /*     printf("next is not null!\n"); */
        /*     xy_append(&((*xy)->next),x,y); */
        /*     printf("appended next!\n"); */
        /* } */
        /* printf("got it2\n"); */
    }
}

void xy_concat(struct LinElemXY ** xy,struct LinElemXY * r)
{

    if (*xy == NULL){
        *xy = r;
    }
    else{
        xy_concat(&((*xy)->next),r);
    }
}

struct LinElemXY * xy_last(struct LinElemXY * xy)
{
    if (xy == NULL){
        return NULL;
    }
    struct LinElemXY * temp = xy;
    while (temp->next != NULL){
        temp = temp->next;
    }
    return temp;
}

double lin_elem_xy_get_x(struct LinElemXY * xy)
{
    return xy->x;
}
double lin_elem_xy_get_y(struct LinElemXY * xy)
{
    return xy->y;
}
struct LinElemXY * lin_elem_xy_next(struct LinElemXY * xy)
{
    return xy->next;
}

size_t lin_elem_xy_length(struct LinElemXY * xy)
{
    size_t count = 0;
    struct LinElemXY * temp = xy;
    if (temp == NULL){
        return count;
    }
    while (temp!= NULL){
        count ++;
        temp = temp->next;
    }
    return count;
}

void lin_elem_xy_free(struct LinElemXY * xy)
{
    if (xy != NULL){
        lin_elem_xy_free(xy->next); xy->next = NULL;
        free(xy); xy = NULL;
    }
}

/********************************************************//**
    Recursively partition

    \param[in] f     - function
    \param[in] xl    - left bound
    \param[in] fl    - left evaluation
    \param[in] xr    - right bound
    \param[in] fr    - right evaluation
    \param[in] delta - value tolerance
    \param[in] hmin  - input tolerance
    \param[in] xy    - xypairs
*************************************************************/
void lin_elem_adapt(struct Fwrap * f,
                    double xl, double fl,double xr, double fr,
                    double delta, double hmin,struct LinElemXY ** xy)
{
    //xy_append(xy,xl,fl);
    if ((xr - xl) <= hmin){
        /* printf("adapt is within bounds!\n"); */
        /* printf("adding left (xl,fl)=(%G,%G)\n",xl,fl); */
        xy_append(xy,xl,fl);
        /* printf("added left (xl,fl)=(%G,%G)\n",xl,fl); */

        xy_append(xy,xr,fr);
        /* printf("done!\n"); */
    }
    else{
        double mid = (xl+xr)/2.0;
        double fmid;
        fwrap_eval(1,&mid,&fmid,f);

        if (fabs( (fl+fr)/2.0 - fmid  )/fabs(fmid) < delta){
//        if (fabs( (fl+fr)/2.0 - fmid) < delta){
            // maybe should add the midpoint since evaluated
            /* printf("finish again! xy==null?=%d\n\n",xy==NULL); */
            /* printf("adding the left %G,%G\n",xl,fl); */
            xy_append(xy,xl,fl);
            xy_append(xy,mid,fmid);
            /* printf("added the left %G,%G\n",xl,fl); */
            /* printf("adding the right %G,%G\n",xr,fr); */
            xy_append(xy,xr,fr);
            /* printf("added the right %G,%G\n",xr,fr); */
        }
        else{
            /* printf("adapt further\n"); */
            lin_elem_adapt(f,xl,fl,mid,fmid,delta,hmin,xy);
            struct LinElemXY * last = NULL;//xy_last(*xy);
            lin_elem_adapt(f,mid,fmid,xr,fr,delta,hmin,&last);
            if (last != NULL){
                xy_concat(xy,last->next);
                free(last);
            }
        }
    }
}


/********************************************************//**
    Approximate a function

    \param[in] opts - approximation options
    \param[in] f    - function

    \return Approximated function
*************************************************************/
struct LinElemExp * 
lin_elem_exp_approx(struct LinElemExpAopts * opts, struct Fwrap * f)
{

    assert(opts != NULL);
    struct LinElemExp * lexp = lin_elem_exp_alloc();
    if (opts->adapt == 0){
        assert (opts->nodes != NULL);

        // allocate nodes and coefficients
        size_t N = opts->num_nodes;
        lexp->num_nodes = N;
        lexp->nodes = calloc_double(N);
        lexp->coeff = calloc_double(N);

        // copy nodes from options
        memmove(lexp->nodes,opts->nodes,N*sizeof(double));

        // evaluate the function
        /* printf("evaluate points\n"); */
        /* dprint(N,lexp->nodes); */
        fwrap_eval(N,lexp->nodes,lexp->coeff,f);

        assert (fabs(lexp->nodes[0]-opts->nodes[0])<1e-15);
        /* printf("cannot evaluate them"); */
    }
    else{
        /* printf("not here!\n"); */
        // adapt
        struct LinElemXY * xy = NULL;
        if (opts->nodes == NULL){ // no nodes yet specified
            double xl = opts->lb;
            double xr = opts->ub;
            double fl,fr;
            fwrap_eval(1,&xl,&fl,f);
            fwrap_eval(1,&xr,&fr,f);
            
            lin_elem_adapt(f,xl,fl,xr,fr,opts->delta,opts->hmin,&xy);
            lexp->num_nodes = lin_elem_xy_length(xy);
            lexp->nodes = calloc_double(lexp->num_nodes);
            lexp->coeff = calloc_double(lexp->num_nodes);
            struct LinElemXY * temp = xy;
            for (size_t ii = 0; ii < lexp->num_nodes; ii++){
                lexp->nodes[ii] = temp->x;
                lexp->coeff[ii] = temp->y;
                temp = temp->next;
            }
            lin_elem_xy_free(xy); xy = NULL;
        }
        else{
            // starting nodes specified
            assert (opts->num_nodes > 1);
            double xl = opts->nodes[0];
            double xr = opts->nodes[1];

            double fl,fr;
            fwrap_eval(1,&xl,&fl,f);
            fwrap_eval(1,&xr,&fr,f);
            
            lin_elem_adapt(f,xl,fl,xr,fr,opts->delta,opts->hmin,&xy);
            for (size_t ii = 2; ii < opts->num_nodes; ii++){
                /* printf("on node = %zu\n",ii); */
                xl = xr;
                fl = fr;
                
                xr = opts->nodes[ii];
                fwrap_eval(1,&xr,&fr,f);

                /* printf("(xl,fl,xr,fr)=(%G,%G,%G,%G)\n",xl,fl,xr,fr); */
                struct LinElemXY * temp = NULL;
                lin_elem_adapt(f,xl,fl,xr,fr,opts->delta,opts->hmin,&temp);
                /* printf("adapted\n"); */
                if (temp != NULL){
                    xy_concat(&xy,temp->next);
                    free(temp);
                }                    
            }
            /* printf("finished here\n"); */
            lexp->num_nodes = lin_elem_xy_length(xy);
            lexp->nodes = calloc_double(lexp->num_nodes);
            lexp->coeff = calloc_double(lexp->num_nodes);
            struct LinElemXY * temp = xy;
            for (size_t ii = 0; ii < lexp->num_nodes; ii++){
                lexp->nodes[ii] = temp->x;
                lexp->coeff[ii] = temp->y;
                temp = temp->next;
            }
            lin_elem_xy_free(xy); xy = NULL;
        }
            
    }
    
    assert (lexp->num_nodes != 0);
    return lexp;
}

/********************************************************//**
    Create a constant function

    \param[in] a  - function value
    \param[in] opts  - options

    \return function
*************************************************************/
struct LinElemExp * 
lin_elem_exp_constant(double a,
                      const struct LinElemExpAopts * opts)
{
    
    struct LinElemExp * lexp = lin_elem_exp_alloc();
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
    Create a linear function
    \f[
        y = ax + b
    \f]

    \param[in] a    - function value
    \param[in] b    - y intercept
    \param[in] opts - options

    \return function
*************************************************************/
struct LinElemExp * 
lin_elem_exp_linear(double a, double b,
                    const struct LinElemExpAopts * opts)
{

    struct LinElemExp * lexp = lin_elem_exp_alloc();
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
        lexp->coeff[ii] = a * lexp->nodes[ii] + b;
    }
    assert (lexp->num_nodes != 0);
    return lexp;
}

/********************************************************//**
    Multiply by -1

    \param[in,out] f - function
*************************************************************/
void lin_elem_exp_flip_sign(struct LinElemExp * f)
{
    for (size_t ii = 0; ii < f->num_nodes; ii++){
        f->coeff[ii] *= -1.0;
    }
    assert (f->num_nodes != 0);
}

/********************************************************//**
    Generate an orthonormal basis
    
    \param[in]     n    - number of basis function
    \param[in,out] f    - linear element expansions with allocated nodes
                          and coefficients set to zero
    \param[in]     opts - approximation options

    \note
    Uses modified gram schmidt to determine function coefficients
    Each function f[ii] must have the same nodes
*************************************************************/
void lin_elem_exp_orth_basis(size_t n, struct LinElemExp ** f, struct LinElemExpAopts * opts)
{
    assert (opts != NULL);

    if (opts->adapt == 0){
        assert (opts->nodes != NULL);
        assert (n <= opts->num_nodes);
        double * zeros = calloc_double(opts->num_nodes);
        for (size_t ii = 0; ii < n; ii++){
            f[ii] = lin_elem_exp_init(opts->num_nodes,opts->nodes,zeros);
            f[ii]->coeff[ii] = 1.0;
        }
        double norm, proj;
        for (size_t ii = 0; ii < n; ii++){
            norm = lin_elem_exp_norm(f[ii]);
            lin_elem_exp_scale(1/norm,f[ii]);
            assert (f[ii]->num_nodes != 0);
            for (size_t jj = ii+1; jj < n; jj++){
                proj = lin_elem_exp_inner(f[ii],f[jj]);
                lin_elem_exp_axpy(-proj,f[ii],f[jj]);
            }
            
        }
        free(zeros); zeros = NULL;
    }
    else{
        // not on a grid I can do whatever I want
        assert (n > 1);
        double * nodes = linspace(opts->lb,opts->ub,n);
        double * zeros = calloc_double(n);
        for (size_t ii = 0; ii < n; ii++){
            f[ii] = lin_elem_exp_init(n,nodes,zeros);
            f[ii]->coeff[ii] = 1.0;
        }
        double norm, proj;
        for (size_t ii = 0; ii < n; ii++){
            norm = lin_elem_exp_norm(f[ii]);
            lin_elem_exp_scale(1/norm,f[ii]);
            assert (f[ii]->num_nodes != 0);
            for (size_t jj = ii+1; jj < n; jj++){
                proj = lin_elem_exp_inner(f[ii],f[jj]);
                lin_elem_exp_axpy(-proj,f[ii],f[jj]);
            }

        }
        free(zeros); zeros = NULL;
        free(nodes); nodes = NULL;
    }

    
    /* double norm, proj; */
    /* for (size_t ii = 0; ii < n; ii++){ */
    /*     assert (f[ii]->num_nodes >= n); */
    /*     f[ii]->coeff[ii] = 1.0;         */
    /* } */
    /* for (size_t ii = 0; ii < n; ii++){ */
    /*     norm = lin_elem_exp_norm(f[ii]); */
    /*     lin_elem_exp_scale(1/norm,f[ii]); */
    /*     for (size_t jj = ii+1; jj < n; jj++){ */
    /*         proj = lin_elem_exp_inner(f[ii],f[jj]); */
    /*         lin_elem_exp_axpy(-proj,f[ii],f[jj]); */
    /*     } */
    /* } */
}

/********************************************************//**
   Scale a function 

   \param[in]     a - value
   \param[in,out] f - function
*************************************************************/
void lin_elem_exp_scale(double a, struct LinElemExp * f)
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
double lin_elem_exp_lb(struct LinElemExp * f)
{
    return f->nodes[0];
}

/********************************************************//**
    Get upper bound

    \param[in] f - function

    \return upper bound
*************************************************************/
double lin_elem_exp_ub(struct LinElemExp * f)
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
    Create a linear element function with zeros at particular
    locations and 1 everyhwhere else.

    \param[in] nzeros    - number of zeros
    \param[in] zero_locs - locations of zeros
    \param[in] opts      - linear expansion options

    \return upper bound
*************************************************************/
struct LinElemExp *
lin_elem_exp_onezero(size_t nzeros, double * zero_locs,
                     struct LinElemExpAopts * opts)
{
    assert (opts != NULL);
    if (opts->adapt == 1){
        // can do whatever I want
        if (nzeros == 0){
            double * nodes = calloc_double(2);
            double * coeff = calloc_double(2);
            struct LinElemExp * le = lin_elem_exp_init(2,nodes,coeff);
            le->coeff[0] = 1.0;
            free(nodes); nodes = NULL;
            free(coeff); coeff = NULL;
            /* print_lin_elem_exp(le,3,NULL,stdout); */
            return le;
        }
        else{
            struct LinElemExp * le = NULL;
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
                le = lin_elem_exp_init(nzeros+2,nodes,coeff);
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
                le = lin_elem_exp_init(nzeros+1,nodes,coeff);
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
                le = lin_elem_exp_init(nzeros+1,nodes,coeff);
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
                le = lin_elem_exp_init(nzeros+1,nodes,coeff);
                free(nodes); nodes = NULL;
                free(coeff); coeff = NULL;
            }
            else{
                assert (1 == 0);
            }
            /* print_lin_elem_exp(le,3,NULL,stdout); */
            return le;
        }
    }
    else{
        assert(opts->nodes != NULL);
        assert(opts->num_nodes > nzeros);
        double * coeff = calloc_double(opts->num_nodes);
        struct LinElemExp * le = lin_elem_exp_init(opts->num_nodes, opts->nodes, coeff);
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
    Print a linear element function

    \param[in] f      - linear element function
    \param[in] args   - extra arguments (not used I think)
    \param[in] prec   - precision with which to save it
    \param[in] stream - stream to print to
************************************************************/
void print_lin_elem_exp(const struct LinElemExp * f, size_t prec, 
                        void * args, FILE * stream)
{
    if (f == NULL){
        fprintf(stream, "Lin Elem Expansion is NULL\n");
    }
    else{
        assert (args == NULL);
        fprintf(stream, "Lin Elem Expansion: num_nodes=%zu\n",f->num_nodes);
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
    Save a linear element expansion in text format

    \param[in] f      - function to save
    \param[in] stream - stream to save it to
    \param[in] prec   - precision with which to save it
************************************************************/
void lin_elem_exp_savetxt(const struct LinElemExp * f,
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
    Load a linear element expansion in text format

    \param[in] stream - stream to save it to

    \return Linear Element Expansion
************************************************************/
struct LinElemExp * lin_elem_exp_loadtxt(FILE * stream)//, size_t prec)
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
    struct LinElemExp * lexp = lin_elem_exp_init(num_nodes,nodes,coeff);
    free(nodes); nodes = NULL;
    free(coeff); coeff = NULL;
    return lexp;
}

