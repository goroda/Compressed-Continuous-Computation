// Copyright (c) 2014-2016, Massachusetts Institute of Technology
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
        free(exp); exp = NULL;
    }

}

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

    return lexp;
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
serialize_lin_elem_exp(unsigned char * ser, struct LinElemExp * f,size_t * totSizeIn)
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
double lin_elem_exp_eval(struct LinElemExp * f, double x)
{
    if ((x < f->nodes[0]) || (x > f->nodes[f->num_nodes-1])){
        return 0.0;
    }
    size_t indmin = 0;
    size_t indmax = f->num_nodes-1;
    size_t indmid = indmin + (indmax - indmin)/2;
    // printf("x=%G\n",x);
    while (indmid != indmin){
        if (fabs(x - f->nodes[indmid]) <= 1e-15){
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

//    printf("indmin = %zu,x=%G\n",indmin,x);
   
    double den = f->nodes[indmin+1]-f->nodes[indmin];
    double t = (f->nodes[indmid+1]-x)/den;

    double value = f->coeff[indmin] * t + f->coeff[indmin+1]*(1.0-t);
    return value;
}

/********************************************************//**
*   Integrate the Linear Element Approximation
*
*   \param[in] f - function
*
*   \return integral
*************************************************************/
double lin_elem_exp_integrate(struct LinElemExp * f)
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
static int lin_elem_sdiscp(struct LinElemExp * f, struct LinElemExp * g)
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
    double dx2 = pow(x2,2)-pow(x1,2);
    double dx3 = (pow(x2,3)-pow(x1,3))/3.0;
          
    *left = pow(x2,2)*dx - x2*dx2 + dx3;
    *mixed = (x2+x1) * dx2/2.0 - x1*x2*dx - dx3;
    *right = dx3 - x1*dx2 + pow(x1,2)*dx;
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
    struct LinElemExp * f, struct LinElemExp * g, double * x,
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
        dx2 = pow(x[ii+1]-x[ii],2);
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
double lin_elem_exp_inner(struct LinElemExp * f,struct LinElemExp * g)
{

    double value = 0.0;
    int samedisc = lin_elem_sdiscp(f,g);
    if (samedisc == 1){
        value = lin_elem_exp_inner_same(f->num_nodes,f->nodes,
                                        f->coeff, g->coeff);
    }
    else{
        double xnew[1000];
        double fnew[1000];
        double gnew[1000];
//        printf("here\n");
        assert ( (f->num_nodes + g->num_nodes) < 1000);
        size_t nnodes = lin_elem_exp_inner_same_grid(f,g,
                                                     xnew,fnew,gnew);
        if (nnodes > 2){
            //          printf("nnodes = %zu\n",nnodes);
            // dprint(nnodes,xnew);
            //dprint(nnodes,fnew);
            //dprint(nnodes,gnew);
            value = lin_elem_exp_inner_same(nnodes,xnew,fnew,gnew);
        }
        else{
            printf("weird =\n");
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
static int lin_elem_exp_axpy_same(double a, struct LinElemExp * f,
                                  struct LinElemExp * g)
{

    for (size_t ii = 0; ii < g->num_nodes; ii++){
        g->coeff[ii] += a*f->coeff[ii];
    }
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
    struct LinElemExp * f, struct LinElemExp * g, double * x)
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
                      struct LinElemExp * f,
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

/********************************************************//**
    Compute the norm of a function

    \param[in] f - function
    
    \return norm
*************************************************************/
double lin_elem_exp_norm(struct LinElemExp * f)
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
double lin_elem_exp_max(struct LinElemExp * f, double * x)
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
double lin_elem_exp_min(struct LinElemExp * f, double * x)
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
    \param[in]     optargs - optimization arguments
    
    \return value
*************************************************************/
double lin_elem_exp_absmax(struct LinElemExp * f, double * x,
                           void * optargs)
{
    if (optargs == NULL){

        double mval = fabs(f->coeff[0]);
        *x = f->nodes[0];
        for (size_t ii = 1; ii < f->num_nodes;ii++){
            if (fabs(f->coeff[ii]) > mval){
                mval = fabs(f->coeff[ii]);
                *x = f->nodes[ii];
            }
        }
        return mval;
    }
    else{
        struct c3Vector * optnodes = optargs;
        double mval = fabs(lin_elem_exp_eval(f,optnodes->elem[0]));
        *x = optnodes->elem[0];
        for (size_t ii = 0; ii < optnodes->size; ii++){
            double val = fabs(lin_elem_exp_eval(f,optnodes->elem[ii]));
            if (val > mval){
                mval = val;
                *x = optnodes->elem[ii];
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

/********************************************************//**
    Allocate approximation arguments

    \param[in] N - number of nodes
    \param[in] x - nodes

    \return approximation arguments
*************************************************************/
struct LinElemExpAopts * lin_elem_exp_aopts_alloc(size_t N, double * x)
{
    struct LinElemExpAopts * aopts = NULL;
    aopts = malloc(sizeof(struct LinElemExpAopts));
    if (aopts == NULL){
        fprintf(stderr,"Memory error allocate LinElemExpAopts\n");
        exit(1);
    }
    aopts->num_nodes= N;
    aopts->nodes = calloc_double(N);
    memmove(aopts->nodes,x,N*sizeof(double));
    aopts->adapt = 0;
    double delta = DBL_MAX;
    double hmin = DBL_MAX;
    return aopts;
}

/********************************************************//**
    Free memory allocated to approximation arguments

    \param[in,out] aopts - approximation arguments
*************************************************************/
void lin_elem_exp_aopts_free(struct LinElemExpAopts * aopts)
{
    if (aopts != NULL){
        free(aopts->nodes); aopts->nodes = NULL;
        free(aopts); aopts = NULL;
    }
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
};

void xy_append(struct LinElemXY ** xy, double x, double y)
{

    if (*xy == NULL){
//        printf("xy is null so initialize\n");
        *xy = xy_init(x,y);
    }
    else{
//        printf("xy is not null!\n");
        xy_append(&((*xy)->next),x,y);
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


void lin_elem_xy_free(struct LinElemXY * xy)
{
    if (xy != NULL){
        lin_elem_xy_free(xy->next); xy->next = NULL;
        free(xy); xy = NULL;

    }
}

/********************************************************//**
    Recursively partition

    \param[in] f    - function
    \param[in] args - function arguments

    \return Approximated function
*************************************************************/
void lin_elem_adapt(double (*f)(double,void*), void * args,
                    double xl, double fl,double xr, double fr,
                    double delta, double hmin,struct LinElemXY ** xy)
{
    //xy_append(xy,xl,fl);
    if ((xr - xl) <= hmin){
        printf("adapt is within bounds!\n");
        xy_append(xy,xl,fl);
        xy_append(xy,xr,fr);
    }
    else{
        double mid = (xl+xr)/2.0;
        double fmid = f(mid,args);

        if (fabs( (fl+fr)/2.0 - fmid  )/fabs(fmid) < delta){
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
            lin_elem_adapt(f,args,xl,fl,mid,fmid,delta,hmin,xy);
            struct LinElemXY * last = NULL;//xy_last(*xy);
            lin_elem_adapt(f,args,mid,fmid,xr,fr,delta,hmin,&last);
            if (last != NULL){
                xy_concat(xy,last->next);
                free(last);
            }
        }
    }
}


/********************************************************//**
    Approximate a function

    \param[in] f    - function
    \param[in] args - function arguments
    \param[in] lb   - lower bound
    \param[in] ub   - upper bound
    \param[in] opts - approximation options

    \return Approximated function
*************************************************************/
struct LinElemExp * 
lin_elem_exp_approx(double (*f)(double,void*), void * args,
                    double lb, double ub,
                    struct LinElemExpAopts * opts)
{

    struct LinElemExp * lexp = lin_elem_exp_alloc();
    if (opts != NULL){
       HERE
    }

    struct LinElemXY * xy = NULL;
    size_t N;
    if (opts == NULL){
        N = 10;
        lexp->num_nodes = N;
        lexp->nodes = linspace(lb,ub,N);
    }
    else{
        assert (opts->nodes != NULL);
        N = opts->num_nodes;
        lexp->num_nodes = N;
        lexp->nodes = calloc_double(N);
        memmove(lexp->nodes,opts->nodes,N*sizeof(double));
    }
    
    lexp->coeff = calloc_double(N);
    for (size_t ii = 0; ii < N; ii++){
        lexp->coeff[ii] = f(lexp->nodes[ii],args);
    }

    return lexp;
}

/********************************************************//**
    Create a constant function

    \param[in] a  - function value
    \param[in] lb - input lower bound
    \param[in] ub - input upper bound
    \param[in] opts  - options

    \return function
*************************************************************/
struct LinElemExp * 
lin_elem_exp_constant(double a, double lb, double ub,
                      const struct LinElemExpAopts * opts)
{
    size_t N;
    if (opts == NULL){
        N = 2;
    }
    else{
        N = opts->num_nodes;
    }
    
    struct LinElemExp * lexp = lin_elem_exp_alloc();
    lexp->num_nodes = N;
    lexp->nodes = linspace(lb,ub,N);
    lexp->coeff = calloc_double(N);
    for (size_t ii = 0; ii < N; ii++){
        lexp->coeff[ii] = a;
    }

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
}

/********************************************************//**
    Generate an orthonormal basis
    
    \param[in]     n - number of basis function
    \param[in,out] f - linear element expansions with allocated nodes
                       and coefficients set to zero

    \note
    Uses modified gram schmidt to determine function coefficients
    Each function f[ii] must have the same nodes
*************************************************************/
void lin_elem_exp_orth_basis(size_t n, struct LinElemExp ** f)
{
    double norm, proj;
    for (size_t ii = 0; ii < n; ii++){
        assert (f[ii]->num_nodes == n);
        f[ii]->coeff[ii] = 1.0;        
    }
    for (size_t ii = 0; ii < n; ii++){
        norm = lin_elem_exp_norm(f[ii]);
        lin_elem_exp_scale(1/norm,f[ii]);
        for (size_t jj = ii+1; jj < n; jj++){
            proj = lin_elem_exp_inner(f[ii],f[jj]);
            lin_elem_exp_axpy(-proj,f[ii],f[jj]);
        }
    }
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


void print_lin_elem_exp(struct LinElemExp * f, size_t prec, 
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

