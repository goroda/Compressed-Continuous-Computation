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
        if ( NULL == (p = malloc(sizeof(struct LinElemExp)))){
            fprintf(stderr, "failed to allocate memory for LinElemExp.\n");
            exit(1);
        }
        p->num_nodes = lexp->num_nodes;
        if (lexp->nodes != NULL){
            p->nodes = calloc_double(p->num_nodes);
            memmove(p->nodes,lexp->nodes,p->num_nodes*sizeof(double));
        }
        if (lexp->coeff != NULL){
            p->coeff = calloc_double(p->num_nodes);
            memmove(p->coeff,lexp->coeff,p->num_nodes*sizeof(double));
        }
        if (lexp->inner != NULL){
            p->inner= calloc_double(p->num_nodes);
            memmove(p->inner,lexp->inner,p->num_nodes*sizeof(double));
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
        free(exp->inner); exp->inner = NULL;
        free(exp); exp = NULL;
    }

}

/********************************************************//**
    Precompute inner products of each basis     

    \param[in,out] f - lin elem expansion function
*************************************************************/
void lin_elem_exp_pinner(struct LinElemExp * f)
{
    assert (f->inner == NULL);
    double * nodes = f->nodes;
    double * coeff = f->coeff;
    f->inner = calloc_double(f->num_nodes);
    double dx = nodes[1]-nodes[0];
    double m = (coeff[1]-coeff[0])/dx;
    double p = (-coeff[1]*nodes[0] + coeff[0]*nodes[1])/dx;
    double t1 = (pow(nodes[1],3)-pow(nodes[0],3))/3.0;
    double t2 = (pow(nodes[1],2)-pow(nodes[0],2));
    f->inner[0] = m*m*t1 + p*m*t2 + p*p*dx;
    for (size_t ii = 1; ii < f->num_nodes-1;ii++){
        dx = nodes[ii+1]-nodes[ii];
        m = (-coeff[ii]+coeff[ii+1])/dx;
        p = (coeff[ii]*nodes[ii+1] - coeff[ii+1]*nodes[ii])/dx;
        t1 = (pow(nodes[ii+1],3)-pow(nodes[ii],3))/3.0;
        t2 = (pow(nodes[ii+1],2)-pow(nodes[ii],2));
        f->inner[ii] = m*m*t1 + p*m*t2 + p*p*dx;
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
struct LinElemExp * lin_elem_exp_init(size_t num_nodes, double * nodes, double * coeff)
{
    struct LinElemExp * lexp = lin_elem_exp_alloc();
    assert (num_nodes > 1);
    lexp->num_nodes = num_nodes;
    lexp->nodes = calloc_double(num_nodes);
    lexp->coeff = calloc_double(num_nodes);
    memmove(lexp->nodes,nodes,num_nodes*sizeof(double));
    memmove(lexp->coeff,coeff,num_nodes*sizeof(double));
    lin_elem_exp_pinner(lexp);

    return lexp;
}
    
/********************************************************//**
*   Serialize a LinElemExp
*
*   \param[in]     ser       - location at which to serialize
*   \param[in]     f         - function to serialize 
*   \param[in,out] totSizeIn - if not NULL then only return total size of struct 
*                              if NULL then serialiaze
*
*   \return pointer to the end of the serialization
*************************************************************/
unsigned char *
serialize_lin_elem_exp(unsigned char * ser, struct LinElemExp * f,size_t * totSizeIn)
{
    
    size_t totsize = sizeof(size_t) + 3 * f->num_nodes*sizeof(double);

    if (totSizeIn != NULL){
        *totSizeIn = totsize;
        return ser;
    }
    unsigned char * ptr = serialize_size_t(ser, f->num_nodes);
    ptr = serialize_doublep(ptr, f->nodes, f->num_nodes);
    ptr = serialize_doublep(ptr, f->coeff, f->num_nodes);
    assert(f->inner != NULL);
    ptr = serialize_doublep(ptr, f->inner, f->num_nodes);
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
    ptr = deserialize_doublep(ptr, &((*f)->inner), &((*f)->num_nodes));
    
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
    double left = f->coeff[indmin] * (f->nodes[indmid+1]-x)/den;
    double right = f->coeff[indmin+1] * (x-f->nodes[indmin])/den;
//    printf ("left = %G, right=%G\n",left,right);
    double value = left+right;
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
*   Inner product between two functions
*
*   \param[in] f - function
*   \param[in] g - function
*
*   \return inner product
*************************************************************/
double lin_elem_exp_inner(struct LinElemExp * f,struct LinElemExp * g)
{

    size_t inodef = 0;
    size_t inodeg = 0;

    double minx, maxx;

    if (f->nodes[0] < g->nodes[0]){
        if (f->nodes[f->num_nodes-1] < g->nodes[0]){
            return 0.0;
        }
        while (f->nodes[inodef+1] < g->nodes[0]){
            inodef += 1;
        }
        minx = f->nodes[inodef];
    }
    else{
        if (g->nodes[g->num_nodes-1] < f->nodes[0]){
            return 0.0;
        }
        while (g->nodes[inodeg+1] < f->nodes[0]){
            inodeg += 1;
        }
        minx = g->nodes[inodeg];
    }

    double inner = 0.0;
    while ( (inodef < f->num_nodes-2) && (inodeg < g->num_nodes-2) ){
        double dx1 = f->nodes[inodef+1] - f->nodes[inodef];
        double m1 = (f->coeff[inodef+1] - f->coeff[inodef])/dx1;
        double p1 = -f->coeff[inodef+1]*f->nodes[inodef] +
                        f->coeff[inodef]*f->nodes[inodef+1];
        p1/= dx1;

        double dx2 = g->nodes[inodeg+1] - g->nodes[inodeg];
        double m2 = (g->coeff[inodeg+1] - g->coeff[inodeg])/dx2;
        double p2 = -g->coeff[inodeg+1]*g->nodes[inodeg] +
                       g->coeff[inodeg]*g->nodes[inodeg+1];
        p2 /= dx2;

        double coeff1 = m1*m2/3.0;
        double coeff2 = (p2*m1 + p1*m2)/2.0;
        double coeff3 = p2*p1;

        if (fabs(f->nodes[inodef+1] - g->nodes[inodeg+1]) < 1e-15){
            inodef += 1;
            inodeg += 1;
            maxx = f->nodes[inodef+1];
        }
        else if (f->nodes[inodef+1] < g->nodes[inodeg+1]){
            maxx = f->nodes[inodef+1];
            inodef += 1;
                        
        }
        else if (g->nodes[inodeg+1] < f->nodes[inodef+1]){
            maxx = g->nodes[inodeg+1];
            inodeg += 1;
        }
        else{
            printf("something weird happened in inner product of lin elem\n");
            exit(1);
        }

        //printf("inodef,inodeg = (%zu,%zu) (%G,%G) \n,",inodef,inodeg,minx,maxx);

        double v1 = pow(maxx,3)-pow(minx,3);
        double v2 = pow(maxx,2)-pow(minx,2);
        double v3 = maxx-minx;
        inner += coeff1*v1 + coeff2*v2 + coeff3*v3;
        
        minx = maxx;
        
    }
 
    return inner;
    
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
int lin_elem_exp_axpy(double a, struct LinElemExp * f,struct LinElemExp * g)
{
    size_t maxN = f->num_nodes + g->num_nodes;
    double * newnodes = calloc_double(maxN);
    double * newcoeff = calloc_double(maxN);
    
    size_t inodef = 0;
    size_t inodeg = 0;
    size_t numnodes = 0;
    while ((inodef < f->num_nodes) && (inodeg < g->num_nodes)){
        if (fabs(f->nodes[inodef]-g->nodes[inodeg]) < 1e-15){
            newnodes[numnodes] = f->nodes[inodef];
            newcoeff[numnodes] = a*f->coeff[inodef] + g->coeff[inodeg];
            inodef++;
            inodeg++;
            numnodes++;
        }
        else if (f->nodes[inodef] < g->nodes[inodeg]){
            newnodes[numnodes] = f->nodes[inodef];
            newcoeff[numnodes] = a*f->coeff[inodef] +
                lin_elem_exp_eval(g,newnodes[numnodes]);
            inodef++;
            numnodes++;
        }
        else{
            newnodes[numnodes] = g->nodes[inodeg];
            newcoeff[numnodes] = a *
                lin_elem_exp_eval(f,g->nodes[inodeg]) +
                g->coeff[inodeg];
            inodeg++;
            numnodes++;
        }
    }
    // There is more work here it is not quite right because
    // I have never added the final element of either
    if ((inodef == f->num_nodes) && (inodeg == g->num_nodes)){
        g->num_nodes = numnodes;
    }
    else if (inodef == f->num_nodes){
        printf("here\n");
        assert(1 == 0);
        while (g->nodes[inodeg] < f->nodes[inodef]){
            newnodes[numnodes] = g->nodes[inodeg];
            newcoeff[numnodes] = g->coeff[inodeg] +
                a * lin_elem_exp_eval(f,newnodes[numnodes]);
            inodeg++;
            numnodes++;
            if (inodeg == g->num_nodes){
                break;
            }
        }

    }
    else if (inodeg == g->num_nodes){
        assert(1 == 0);
        printf("there\n");
        while (inodef < f->num_nodes){
            newnodes[numnodes] = f->nodes[inodef];
            newcoeff[numnodes] = a*f->coeff[inodef];
            inodef++;
            numnodes++;
        }
    }

    g->num_nodes = numnodes;
    free(g->nodes); g->nodes = newnodes;
    free(g->coeff); g->coeff = newcoeff;
    free(g->inner); g->inner = NULL;
    lin_elem_exp_pinner(g);
    return 0;
    
}

/********************************************************//**
    Compute the norm of a function

    \param[in] f - function
    
    \return norm
*************************************************************/
double lin_elem_exp_norm(struct LinElemExp * f)
{
    if (f->inner == NULL){
        lin_elem_exp_pinner(f);
    }

    size_t ii;
    double norm = 0.0;
    for (ii = 0; ii < f->num_nodes; ii++){
        norm += f->inner[ii];
    }
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

    \param[in]     f - function
    \param[in,out] x - location of absolute value max
    
    \return value
*************************************************************/
double lin_elem_exp_absmax(struct LinElemExp * f, double * x)
{
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
    Approximate a function                                                         

    \param[in] f    - function
    \param[in] args - function arguments
    \param[in] lb   - lower bound
    \param[in] ub   - upper bound
    \param[in] opts - approximation options

    \return Approximated function
*************************************************************/
struct LinElemExp * lin_elem_exp_approx(double (*f)(double,void*), void * args,
                                        double lb, double ub,
                                        struct LinElemExpAopts * opts)
{
    size_t N;
    int adapt;
    if (opts == NULL){
        N = 10;
        adapt = 0;
    }
    else{
        N = opts->num_nodes;
        adapt = opts->adapt;
    }
    
    struct LinElemExp * lexp = lin_elem_exp_alloc();
    lexp->num_nodes = N;
    lexp->nodes = linspace(lb,ub,N);
    lexp->coeff = calloc_double(N);
    for (size_t ii = 0; ii < N; ii++){
        lexp->coeff[ii] = f(lexp->nodes[ii],args);
    }
    lin_elem_exp_pinner(lexp);

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
struct LinElemExp * lin_elem_exp_constant(double a, double lb, double ub,
                                          struct LinElemExpAopts * opts)
{
    size_t N;
    int adapt;
    if (opts == NULL){
        N = 2;
        adapt = 0;
    }
    else{
        N = opts->num_nodes;
        adapt = opts->adapt;
    }
    
    struct LinElemExp * lexp = lin_elem_exp_alloc();
    lexp->num_nodes = N;
    lexp->nodes = linspace(lb,ub,N);
    lexp->coeff = calloc_double(N);
    for (size_t ii = 0; ii < N; ii++){
        lexp->coeff[ii] = a;
    }
    lin_elem_exp_pinner(lexp);

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
    free(f->inner); f->inner = NULL;
    lin_elem_exp_pinner(f);
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
    free(f->inner); f->inner = NULL;
    lin_elem_exp_pinner(f);
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
    assert (args == NULL);
    fprintf(stream, "Lin Elem Expansion");
    for (size_t ii = 0; ii < f->num_nodes; ii++){
        if (prec < 100){
            fprintf(stream, "(%3.5G,%3.5G)  ",f->nodes[ii],f->coeff[ii]);
        }
    }
    fprintf(stream,"\n");
}

