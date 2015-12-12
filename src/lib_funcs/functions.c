// Copyright (c) 2014-2015, Massachusetts Institute of Technology
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

/** \file functions.c
 * Provides basic routines for interfacing specific functions to the outside world through
 * generic functions
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "stringmanip.h"
#include "array.h"
#include "functions.h"
#include "polynomials.h"
#include "piecewisepoly.h"

/*******************************************************//**
    Initialize a bounds tructure with each dimension bounded by [-1,1]

    \param dim [in] - dimension
        
    \return b - bounds
***********************************************************/
struct BoundingBox * bounding_box_init_std(size_t dim)
{
    struct BoundingBox * b;
    if (NULL == ( b = malloc(sizeof(struct BoundingBox)))){
        fprintf(stderr, "failed to allocate bounds.\n");
        exit(1);
    }
    
    b->dim = dim;
    b->lb = calloc_double(dim);
    b->ub = calloc_double(dim);
    size_t ii;
    for (ii = 0; ii <dim; ii++){
        b->lb[ii] = -1.0;
        b->ub[ii] = 1.0;
    }
    return b;
}

/*******************************************************//**
    Initialize a bound structure with each dimension bounded by [lb,ub]

    \param dim [in] - dimension
    \param lb [in] - lower bounds
    \param ub [in] - upper bounds
        
    \return b - bounds
***********************************************************/
struct BoundingBox * bounding_box_init(size_t dim, double lb, double ub)
{
    struct BoundingBox * b;
    if (NULL == ( b = malloc(sizeof(struct BoundingBox)))){
        fprintf(stderr, "failed to allocate bounds.\n");
        exit(1);
    }
    
    b->dim = dim;
    b->lb = calloc_double(dim);
    b->ub = calloc_double(dim);
    size_t ii;
    for (ii = 0; ii <dim; ii++){
        b->lb[ii] = lb;
        b->ub[ii] = ub;
    }
    return b;
}

/*******************************************************//**
    Initialize a bound structure with each dimension bounded by [lb[i],ub[i]]

    \param dim [in] - dimension
    \param lb [in] - lower bounds
    \param ub [in] - upper bounds
        
    \return b - bounds
***********************************************************/
struct BoundingBox * bounding_box_vec(size_t dim, double * lb, double *ub)
{
    struct BoundingBox * b;
    if (NULL == ( b = malloc(sizeof(struct BoundingBox)))){
        fprintf(stderr, "failed to allocate bounds.\n");
        exit(1);
    }
    
    b->dim = dim;
    b->lb = calloc_double(dim);
    b->ub = calloc_double(dim);
    size_t ii;
    for (ii = 0; ii <dim; ii++){
        b->lb[ii] = lb[ii];
        b->ub[ii] = ub[ii];
    }
    return b;
}

/********************************************************//**
    Free memory allocated for bounding box

    \param b [inout] - bounds
************************************************************/
void bounding_box_free(struct BoundingBox * b)
{
    if (b != NULL){
        free(b->lb);
        free(b->ub);
        free(b);
    }
}

/********************************************************//**
    Allocate memory for a generic function without specifying class or sub_type

    \param dim [in] - dimension of functions

    \return out - generic function
************************************************************/
struct GenericFunction * generic_function_alloc_base(size_t dim)
{
    struct GenericFunction * out;
    if (NULL == ( out = malloc(sizeof(struct GenericFunction)))){
        fprintf(stderr, "failed to allocate for a generic function.\n");
        exit(1);
    }
    out->dim = dim;
    out->fargs = NULL;
    out->f = NULL;
    return out;
}

/********************************************************//**
    Allocate memory for a generic function array
    
    \param size [in] - size of array

    \return out - generic function
************************************************************/
struct GenericFunction ** generic_function_array_alloc(size_t size)
{
    struct GenericFunction ** out;
    if (NULL == ( out = malloc( size * sizeof(struct GenericFunction *)))){
        fprintf(stderr, "failed to allocate a generic function array.\n");
        exit(1);
    }
    size_t ii;
    for (ii = 0; ii < size; ii++){
        out[ii] = NULL;
    }
    return out;
}



/********************************************************//**
    Allocate memory for a generic function of a particular
    function class

    \param dim [in] - dimension of functions
    \param fc [in] - function class
    \param sub_type [in] - sub type of approximation

    \return out - generic function
************************************************************/
struct GenericFunction *
generic_function_alloc(size_t dim, enum function_class fc, void * sub_type){
    struct GenericFunction * out;
    if (NULL == ( out = malloc(sizeof(struct GenericFunction)))){
        fprintf(stderr, "failed to allocate for a generic function.\n");
        exit(1);
    }
    out->dim = dim;
    out->fc = fc;
    out->fargs = NULL;

    enum poly_type ptype;
    switch (fc){
        case PIECEWISE:
            ptype = *((enum poly_type *) sub_type);
            out->sub_type.ptype = ptype; 
            break;
        case POLYNOMIAL:
            ptype = *((enum poly_type *) sub_type);
            out->sub_type.ptype = ptype; 
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }

    //out->sub_type.ptype = NULL;
    out->f = NULL;
    out->fargs = NULL;
    return out;
}

/********************************************************//**
*  Round an generic function to some tolerance
*
*  \param gf [inout] - generic function
*  \param thresh [in] - threshold (relative) to round to
*
*  \note
*      (UNTESTED, use with care!!!! 
*************************************************************/
void generic_function_roundt(struct GenericFunction ** gf, double thresh)
{
    struct OrthPolyExpansion * ope = NULL;
    switch ((*gf)->fc){
        case PIECEWISE:
            break;
        case POLYNOMIAL:
            ope = (*gf)->f;
            orth_poly_expansion_roundt(&ope,thresh);
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
}

/********************************************************//**
    Create a generic function by approximating a one dimensional function

    \param f [in] - function to approximate
    \param args [in] - function arguments
    \param fc [in] - function approximation class
    \param sub_type [in] - sub type of approximation
    \param lb [in] - lower bound
    \param ub [in] - upper bound
    \param aopts [in] - approximation options

    \return gf - generic function
************************************************************/
struct GenericFunction * 
generic_function_approximate1d( double (*f)(double,void *), void * args, 
            enum function_class fc, void * sub_type, double lb, double ub, 
            void * aopts)
{
    struct GenericFunction * gf = generic_function_alloc(1,fc,sub_type);
    switch (fc){
        case PIECEWISE:
            //printf("approximate\n");
            gf->f = piecewise_poly_approx1_adapt(f,args, lb, ub, aopts);
            //gf->f = piecewise_poly_approx2(f,args, lb, ub, aopts);
            //printf("approximated\n");
            gf->fargs = NULL;
            break;
        case POLYNOMIAL:
            //printf("approximate with poyl type %d \n",gf->sub_type.ptype);
            //enum poly_type * ptype = sub_type;
            //printf("approximate with poyl type %d \n",*ptype);
            gf->f = orth_poly_expansion_approx_adapt(f,args, gf->sub_type.ptype, lb, ub, aopts);
            gf->fargs = NULL;
            //printf("done\n");
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
    //print_generic_function(gf,0,NULL);
    return gf;
}

/********************************************************//**
    Create a pseudo-random polynomial generic function 

*   \param maxorder [in] - maximum order of the polynomial
*   \param lower [in] - lower bound of input
*   \param upper [in] - upper bound of input

    \return gf - generic function
************************************************************/
struct GenericFunction * 
generic_function_poly_randu(size_t maxorder, double lower, double upper)
{
    enum function_class fc = POLYNOMIAL;
    enum poly_type sub_type = LEGENDRE;
    struct GenericFunction * gf = generic_function_alloc(1,fc,&sub_type);
    gf->f = orth_poly_expansion_randu(sub_type,maxorder,lower,upper);
    gf->fargs = NULL;
    return gf;
}

/********************************************************//**
    Take the derivative of a generic function

    \param gf [in] - generic function

    \return out - generic function representing the derivative
************************************************************/
struct GenericFunction * 
generic_function_deriv(struct GenericFunction * gf)
{
    struct GenericFunction * out = NULL; 
    switch (gf->fc){
        case PIECEWISE:
            out = generic_function_alloc(1,gf->fc,&(gf->sub_type.ptype));
            out->f = piecewise_poly_deriv(gf->f);
            out->fargs = NULL;
            //printf("DERIVATIVE OF PIECEWISE POLY IS NOT YET IMPLEMENTED!\n");
            break;
        case POLYNOMIAL:
            out = generic_function_alloc(1,gf->fc,&(gf->sub_type.ptype));
            out->f = orth_poly_expansion_deriv(gf->f);
            out->fargs = NULL;
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
    //print_generic_function(gf,0,NULL);
    return out;
}

/********************************************************//**
    Copy a generic function

    \param gf [in] - generic function

    \return out - generic function
************************************************************/
struct GenericFunction * 
generic_function_copy(struct GenericFunction * gf)
{
    struct GenericFunction * out = NULL; 

    switch (gf->fc){
        case PIECEWISE:
            out = generic_function_alloc(gf->dim, gf->fc, &(gf->sub_type.ptype));
            out->f = piecewise_poly_copy(gf->f);
            break;
        case POLYNOMIAL:
            //ptype = *((enum poly_type *) gf->sub_type.ptype);
            out = generic_function_alloc(gf->dim, gf->fc, &(gf->sub_type.ptype));
            out->f = orth_poly_expansion_copy(gf->f);
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
    return out;
}

/********************************************************//**
    Copy a generic function to a preallocated generic function

    \param gf [in] - generic function
    \param gfpa [inout] - preallocated function

************************************************************/
void generic_function_copy_pa(struct GenericFunction * gf, struct GenericFunction * gfpa)
{
    switch (gf->fc){
        case PIECEWISE:
            gfpa->fc = gf->fc;
            gfpa->sub_type.ptype = gf->sub_type.ptype;
            gfpa->f = piecewise_poly_copy(gf->f);
            break;
        case POLYNOMIAL:
            gfpa->fc = gf->fc;
            gfpa->sub_type.ptype = gf->sub_type.ptype;
            gfpa->f = orth_poly_expansion_copy(gf->f); 
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
}


/********************************************************//**
    Free memory for generic function

    \param gf [inout] - generic function
************************************************************/
void generic_function_free(struct GenericFunction * gf){
    if (gf != NULL){
        switch (gf->fc){
            case PIECEWISE:
                if (gf->f != NULL){
                    piecewise_poly_free((struct PiecewisePoly *) gf->f);
                }
                break;
            case POLYNOMIAL:
                if (gf->f != NULL){
                    orth_poly_expansion_free((struct OrthPolyExpansion *) gf->f);
                }
                break;    
            case RATIONAL:
                break;
            case KERNEL:
                break;
        }
        free(gf); gf = NULL;
    }
}

/********************************************************//**
    Free memory for generic function array

    \param gf [inout] - generic function array
    \param n [in] - number of generic functions in the array
************************************************************/
void generic_function_array_free(struct GenericFunction ** gf, size_t n){
    if (gf != NULL){
        size_t ii;
        for (ii = 0; ii < n; ii++){
            generic_function_free(gf[ii]);
            gf[ii] = NULL;
        }
        free(gf); gf = NULL;
    }
}


/********************************************************//**
*   Serialize a generic function
*
*   \param ser [inout] - location to serialize to
*   \param gf [in] - generic function
*   \param totSizeIn [in] - if not null then only total size in bytes of generic function si returned! if NULL then serialization occurs
*
*   \return ptr - ser + num_bytes
************************************************************/
unsigned char *
serialize_generic_function(unsigned char * ser, 
                struct GenericFunction * gf, size_t * totSizeIn)
{   
    // order = 
    // function_class -> sub_type -> function
    
    size_t totSize = sizeof(int) + sizeof(size_t); // for function class and dim
    size_t sizef;
    unsigned char * ptr = ser;
    struct OrthPolyExpansion * ft = NULL;
    struct PiecewisePoly * pw = NULL;
    switch (gf->fc){
        case PIECEWISE:
            pw = gf->f;
            totSize += sizeof(int); // sub_type;
            serialize_piecewise_poly(NULL,pw, &sizef);
            totSize += sizef;
            if (totSizeIn != NULL){
                *totSizeIn = totSize;
                return ser;
            }
            
            ptr = serialize_size_t(ser, gf->dim);
            ptr = serialize_int(ptr, gf->fc);
            ptr = serialize_int(ptr, gf->sub_type.ptype);
            ptr = serialize_piecewise_poly(ptr,pw,NULL);

            break;
        case POLYNOMIAL:
            ft = gf->f;

            totSize += sizeof(int); // sub_type;
            serialize_orth_poly_expansion(NULL,ft, &sizef);
            totSize += sizef;
            if (totSizeIn != NULL){
                *totSizeIn = totSize;
                return ser;
            }
            
            ptr = serialize_size_t(ser, gf->dim);
            ptr = serialize_int(ptr, gf->fc);
            ptr = serialize_int(ptr, gf->sub_type.ptype);
            ptr = serialize_orth_poly_expansion(ptr,ft,NULL);
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }

    return ptr;
}

/********************************************************//**
*   Deserialize a generic function
*
*   \param ser [in] - serialized function
*   \param gf  [inout] -  generic function
*
*   \return ptr = ser + nBytes of gf
************************************************************/
unsigned char *
deserialize_generic_function(unsigned char * ser, 
                struct GenericFunction ** gf)
{
    
    // function_class -> sub_type -> function
    int fci;
    size_t dim;
    enum function_class fc;
    
    unsigned char * ptr = deserialize_size_t(ser, &dim);
    ptr = deserialize_int(ptr, &fci);
    fc = (enum function_class) fci;

    struct OrthPolyExpansion * ope = NULL;
    struct PiecewisePoly * pw = NULL;
    enum poly_type ptype;
    int pti;
    switch (fc){
        case PIECEWISE:
            ptr = deserialize_int(ptr, &pti);
            ptype = (enum poly_type) pti;
            ptr = deserialize_piecewise_poly(ptr, &pw);

            *gf = generic_function_alloc(dim, fc, &ptype);
            (*gf)->f = pw;
            (*gf)->fargs=NULL;
            break;
        case POLYNOMIAL:
            ptr = deserialize_int(ptr, &pti);
            ptype = (enum poly_type) pti;
            ptr = deserialize_orth_poly_expansion(ptr, &ope);

            *gf = generic_function_alloc(dim, fc, &ptype);
            (*gf)->f = ope;
            (*gf)->fargs=NULL;
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
    return ptr;
}


/////////////////////////////////////////////////////////////
//

/********************************************************//**
*   Compute norm of a generic function
*
*   \param f [in] - generic function
*
*   \return out - norm
************************************************************/
double generic_function_norm(struct GenericFunction * f){
    double out = generic_function_inner(f,f);

    if (out < 0.0){
        fprintf(stderr, "Norm of a function cannot be negative %G\n",out);
        exit(1);
    }
    //assert (out > -1e-15);
    return sqrt(out);
}

/********************************************************//**
*   Compute the norm of the difference between two generic function
*
*   \param f1 [in] - generic function
*   \param f2 [in] - generic function
*
*   \return out - norm of difference
************************************************************/
double generic_function_norm2diff(struct GenericFunction * f1, 
                                    struct GenericFunction * f2)
{
    struct GenericFunction * f3 = generic_function_daxpby(1.0,f1,-1.0,f2);
    double out = generic_function_norm(f3);
    generic_function_free(f3); f3 = NULL;
    return out;
}

/********************************************************//**
*   Compute the norm of the difference between two generic function arrays
*   
*   \param n [in] - number of elements
*   \param f1 [in] - generic function array
*   \param inca [in] - incremenent of first array
*   \param f2 [in] - generic function array
*   \param incb [in] - incremenent of second array
*
*   \return out - norm of difference
************************************************************/
double generic_function_array_norm2diff(
                size_t n, struct GenericFunction ** f1, size_t inca,
                struct GenericFunction ** f2, size_t incb)
{
    double out = 0.0;
    size_t ii;
    for (ii = 0; ii < n; ii++){
        out += pow(generic_function_norm2diff(f1[ii*inca],f2[ii*incb]),2);
    }
    assert (out >= 0.0);
    return sqrt(out);
}



/********************************************************//**
*   Compute the integral of a generic function
*
*   \param f [in] - generic function
*
*   \return out - integral
************************************************************/
double generic_function_integral(struct GenericFunction * f){
    double out = 0.0;   
    switch (f->fc){
        case PIECEWISE:
            out = piecewise_poly_integrate((struct PiecewisePoly *) f->f);
            break;
        case POLYNOMIAL:
            out = orth_poly_expansion_integrate((struct OrthPolyExpansion *) f->f);
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
    return out;
}

/********************************************************//**
*   Compute the integral of all the functions in a generic function array
*
*   \param n [in] - number of functions
*   \param lda [in] - stride
*   \param a [in] - array of generic functions
*
*   \return out - array of integrals
************************************************************/
double * 
generic_function_integral_array(size_t n , size_t lda, struct GenericFunction ** a)
{
    double * out = calloc_double(n);
    size_t ii;
    for (ii = 0; ii < n; ii++){
        out[ii] = generic_function_integral(a[ii*lda]);
    }
    return out;
}

/********************************************************//**
*   Compute the sum of the product between the functions of two function arrays
*
*   \param n [in] - number of functions
*   \param lda [in] - stride of first array
*   \param a [in] - array of generic functions
*   \param ldb [in] - stride of second array
*   \param b [in] - array of generic functions
*
*   \return out - generic function
************************************************************/
struct GenericFunction *
generic_function_sum_prod(size_t n, size_t lda,  struct GenericFunction ** a, 
                size_t ldb, struct GenericFunction ** b)
{
    size_t ii;
    int allpoly = 1;
    for (ii = 0; ii < n; ii++){
        if (a[ii*lda]->fc != POLYNOMIAL){
            allpoly = 0;
            break;
        }
        if (b[ii*ldb]->fc != POLYNOMIAL){
            allpoly = 0;
            break;
        }
    }

    if (allpoly == 1){
        struct OrthPolyExpansion ** aa = NULL;
        struct OrthPolyExpansion ** bb= NULL;
        if (NULL == (aa = malloc(n * sizeof(struct OrthPolyExpansion *)))){
            fprintf(stderr, "failed to allocate memmory in generic_function_sum_prod\n");
            exit(1);
        }
        if (NULL == (bb = malloc(n * sizeof(struct OrthPolyExpansion *)))){
            fprintf(stderr, "failed to allocate memmory in generic_function_sum_prod\n");
            exit(1);
        }
        for  (ii = 0; ii < n; ii++){
            aa[ii] = a[ii*lda]->f;
            bb[ii] = b[ii*ldb]->f;
        }
        
        struct GenericFunction * gf = generic_function_alloc(1,a[0]->fc,&(a[0]->sub_type.ptype));
        gf->f = orth_poly_expansion_sum_prod(n,1,aa,1,bb);
        gf->fargs = NULL;
        free(aa); aa = NULL;
        free(bb); bb = NULL;

        assert (gf->f != NULL);
        return gf;
    }

    ii = 0;
    struct GenericFunction * out = generic_function_prod(a[lda*ii],b[ldb*ii]);
    struct GenericFunction * out2 = NULL;
    struct GenericFunction * temp = NULL;
    for (ii = 1; ii < n; ii++){
        temp = generic_function_prod(a[lda*ii],b[ldb*ii]);
        if (out2 == NULL){
            out2 = generic_function_daxpby(1.0,out, 1.0, temp);
            generic_function_free(out); out = NULL;
        }
        else if (out == NULL){
            out = generic_function_daxpby(1.0,out2, 1.0, temp);
            generic_function_free(out2); out2 = NULL;
        }
        generic_function_free(temp);
    }

    if (out == NULL){
        return out2;
    }
    else if (out2 == NULL){
        return out;
    }
    return NULL;
}

/********************************************************//**
*   Compute the product between two generic functions
*
*   \param a [in] - generic function
*   \param b [in] - generic function
*
*   \return out(x) = a(x)b(x)  - generic function
************************************************************/
struct GenericFunction *
generic_function_prod(struct GenericFunction * a, struct GenericFunction * b){
    enum function_class fc = a->fc;
    int apalloc = 0;
    int bpalloc = 0;
    struct PiecewisePoly * ap = NULL;
    struct PiecewisePoly * bp = NULL;
    if ( (a->fc != b->fc) || (a->fc == PIECEWISE) ){
        // everything to PIECEWISE!
        fc = PIECEWISE;
        if (a->fc == POLYNOMIAL){
            ap = piecewise_poly_alloc();
            apalloc = 1;
            ap->ope = a->f;
        }
        else{
            ap = a->f;
        }
        if (b->fc == POLYNOMIAL){
            bp = piecewise_poly_alloc();
            bpalloc = 1;
            bp->ope = b->f;
        }
        else{
            bp = b->f;
        }
    }
    struct GenericFunction * out = NULL;
    switch (fc){
        case PIECEWISE:
            out = generic_function_alloc(a->dim,PIECEWISE,&(a->sub_type.ptype));
            out->fargs = a->fargs;
            out->f = piecewise_poly_prod(ap, bp);
            break;
        case POLYNOMIAL:
            out = generic_function_alloc(a->dim,a->fc,&(a->sub_type.ptype));
            out->fargs = a->fargs;
            out->f = orth_poly_expansion_prod(
                            (struct OrthPolyExpansion *) a->f,
                            (struct OrthPolyExpansion *) b->f);
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
    
    if (apalloc == 1){
        free(ap);
    }
    if (bpalloc == 1){
        free(bp);
    }
    return out;
}

/********************************************************//**
*   Compute the inner product between two generic functions
*
*   \param a [in] - generic function
*   \param b [in] - generic function
*
*   \return out -  int a(x) b(x) dx 
************************************************************/
double generic_function_inner(struct GenericFunction * a, struct GenericFunction * b){
    double out = 0.123456789;   
    enum function_class fc = a->fc;
    int apalloc = 0;
    int bpalloc = 0;
    struct PiecewisePoly * ap = NULL;
    struct PiecewisePoly * bp = NULL;
    if ( (a->fc != b->fc) || (a->fc == PIECEWISE) ){
        // everything to PIECEWISE!
        fc = PIECEWISE;
        if (a->fc == POLYNOMIAL){
            ap = piecewise_poly_alloc();
            apalloc = 1;
            ap->ope = a->f;
        }
        else{
            ap = a->f;
        }
        if (b->fc == POLYNOMIAL){
            bp = piecewise_poly_alloc();
            bpalloc = 1;
            bp->ope = b->f;
        }
        else{
            bp = b->f;
        }
    }

    switch (fc){
        case PIECEWISE:
            if (ap == NULL){
                //printf("anull\n");
                ap = a->f;
            }
            if (bp == NULL){
                //printf("bnull\n");
                bp = b->f;
            }
            //printf("in here, ap==NULL=%d, bp==NULL=%d\n",ap==NULL,bp==NULL);
            //print_piecewise_poly(ap,3,NULL);
            //print_piecewise_poly(bp,3,NULL);
            out = piecewise_poly_inner(ap, bp);
            //printf("piecewise inner = %G\n",out);
            break;
        case POLYNOMIAL:
            out = orth_poly_expansion_inner(
                            (struct OrthPolyExpansion *) a->f,
                            (struct OrthPolyExpansion *) b->f);
            //printf("poly inner = %G\n",out);
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }

    if (apalloc == 1){
        free(ap);
    }
    if (bpalloc == 1){
        free(bp);
    }

    return out;
}

/********************************************************//**
*   Compute the sum of the inner products between
*   two arrays of generic functions
*
*   \param n [in] - number of inner products
*   \param lda [in] - stride of functions to use in a
*   \param a [in] - first array of generic functions
*   \param ldb [in] - stride of functions to use in b
*   \param b [in] - second array of generic functions
*
*   \return val - sum_{i=1^N} int a[ii*lda](x) b[ii*ldb](x) dx
************************************************************/
double generic_function_inner_sum(size_t n, size_t lda, struct GenericFunction ** a, 
                         size_t ldb, struct GenericFunction ** b)
{
    double val = 0.0;
    size_t ii;
    for (ii = 0; ii < n; ii++){
        //printf("gf1 = \n");
        //print_generic_function(a[ii*lda], 0, NULL);
        //printf("gf2 = \n");
        //print_generic_function(b[ii*ldb], 0, NULL);
        val += generic_function_inner(a[ii*lda], b[ii*ldb]);
    }
    return val;
}

/********************************************************//**
*   Compute the norm of an array of generic functions
*
*   \param n [in] - number of functions
*   \param lda [in] - stride of functions to use in a
*   \param a [in] - functions
*
*   \return val -sqrt(sum_{i=1^N} int a[ii*lda](x)^2 ) dx)
************************************************************/
double generic_function_array_norm(size_t n, size_t lda, 
                                        struct GenericFunction ** a)
{   
    
    double val = 0.0;
    size_t ii;
    for (ii = 0; ii < n; ii++){
        val += pow(generic_function_norm(a[lda*ii]),2.0);
    }
    //val = generic_function_inner_sum(n,lda,a,lda,a);

    return sqrt(val);
}

/********************************************************//**
*   Flip the sign of a generic function f(x) to -f(x)
*
*   \param f [inout] - number of functions
************************************************************/
void generic_function_flip_sign(struct GenericFunction * f){
    switch (f->fc){
        case PIECEWISE:
            piecewise_poly_flip_sign((struct PiecewisePoly *) f->f);
            break;
        case POLYNOMIAL:
            orth_poly_expansion_flip_sign((struct OrthPolyExpansion *) f->f);
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
}

/********************************************************//**
*   Flip the sign of each generic function in an array
*
*   \param n [in] - number of functions
*   \param lda [in] - stride of array
*   \param a [inout] - array of functions
************************************************************/
void 
generic_function_array_flip_sign(size_t n, size_t lda, struct GenericFunction ** a){
    size_t ii;
    for (ii = 0; ii < n; ii++){
        generic_function_flip_sign(a[ii*lda]);
    }
}

/********************************************************//**
*   Get the lower bound of a generic function 
*
*   \param f [in] - function
*
*   \return lb - lower bound
************************************************************/
double generic_function_get_lower_bound(struct GenericFunction * f){
    double lb = -0.123456789;
    switch (f->fc){
        case PIECEWISE:
            lb = piecewise_poly_lb((struct PiecewisePoly *) f->f);
            break;
        case POLYNOMIAL:
            lb = ((struct OrthPolyExpansion *) f->f)->lower_bound;
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
    return lb;
}

/********************************************************//**
*   Get the upper bound of a generic function 
*
*   \param f [in] - function
*
*   \return ub - lower bound
************************************************************/
double generic_function_get_upper_bound(struct GenericFunction * f){
    double ub = 0.123456789;
    switch (f->fc){
        case PIECEWISE:
            ub = piecewise_poly_ub((struct PiecewisePoly *) f->f);
            break;
        case POLYNOMIAL:
            ub = ((struct OrthPolyExpansion *) f->f)->upper_bound;
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
    return ub;
}

/********************************************************//**
*   Evaluate a generic function
*
*   \param f [in] - function
*   \param x [in] - location at which to evaluate
*
*   \return out - evaluation
************************************************************/
double generic_function_1d_eval(struct GenericFunction * f, double x){
    double out = 0.1234567890;
    struct OrthPolyExpansion * op;
    struct PiecewisePoly * pw;
    switch (f->fc){
        case PIECEWISE:
            pw = f->f;
            out = piecewise_poly_eval(pw,x);
            break;
        case POLYNOMIAL:
            op = f->f;
            out = orth_poly_expansion_eval(op,x);
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
    return out;
}

/********************************************************//**
*   Evaluate an array of generic functions
*
*   \param n [in] - number of functions
*   \param f [in] - array of functions
*   \param x [in] - location at which to evaluate
*
*   \return out - array of values
************************************************************/
double * 
generic_function_1darray_eval(size_t n, struct GenericFunction ** f, double x)
{
    double * out = calloc_double(n);
    size_t ii;
    for (ii = 0; ii < n; ii++){
        out[ii] = generic_function_1d_eval(f[ii],x);
    }
    return out;
}

/********************************************************//**
*   Multiply and add 3 functions \f$ z \leftarrow ax + by + cz \f$
*
*   \param a [in] - first scaling factor 
*   \param x [in] - first function
*   \param b [in] - second scaling factor 
*   \param y [in] - second function
*   \param c [in] - third scaling factor 
*   \param z [in] - third function
*
*************************************************************/
void
generic_function_sum3_up(double a, struct GenericFunction * x,
                         double b, struct GenericFunction * y,
                         double c, struct GenericFunction * z)
{
    if ( x->fc != POLYNOMIAL){
        fprintf(stderr, "Have not yet implemented generic_function_sum3_up \n");
        fprintf(stderr, "for functions other than polynomials\n");
        exit(1);
    }
    assert (x->fc == y->fc);
    assert (y->fc == z->fc);
    
    orth_poly_expansion_sum3_up(a,x->f,b,y->f,c,z->f);
}

/********************************************************//**
*   Add two generic functions \f$ y \leftarrow ax + y \f$
*
*   \param a [in] - scaling of first function
*   \param x [in] - first function
*   \param y [inout] - second function
*
*   \return 0 if successfull, 1 if error
*
*   \note
*       Handling the function class of the output is not very smart
************************************************************/
int generic_function_axpy(double a, struct GenericFunction * x, 
            struct GenericFunction * y)
{
    //printf("in here! a =%G b = %G\n",a,b);

    assert (y != NULL);
    assert (x != NULL);
    assert (x->fc == y->fc);

    int success = 1;
    switch (x->fc){
        case PIECEWISE:
            fprintf(stderr,"Error: axpy not implemented for piecewise polynomials\n");
            exit(1);
            break;
        case POLYNOMIAL:
            success = orth_poly_expansion_axpy(a,x->f,y->f);
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
    
    return success;
}

/********************************************************//**
*   Add generic functions \f$ y[i] \leftarrow a x[i] + y[i] \f$
*
*   \param n [in] - number of functions
*   \param a [in] - scaling of the first functions
*   \param x [in] - first function array
*   \param y [inout] - second function array
*
*   \return 0 if successfull, 1 if error
*
*   \note
*       Handling the function class of the output is not very smart
************************************************************/
int generic_function_array_axpy(size_t n, double a, struct GenericFunction ** x, 
            struct GenericFunction ** y)
{
    //printf("in here! a =%G b = %G\n",a,b);

    assert (y != NULL);
    assert (x != NULL);

    int success = 1;
    size_t ii;
    for (ii = 0; ii < n; ii++){
        success = generic_function_axpy(a,x[ii],y[ii]);
        if (success == 1){
            break;
        }
    }
    
    return success;
}

/********************************************************//**
*   Add two generic functions z = ax + by
*
*   \param a [in] - scaling of first function
*   \param x [in] - first function
*   \param b [in] - scaling of second function
*   \param y [in] - second function
*
*   \return out - generic function 
*
*   \note
*       Handling the function class of the output is not very smart
************************************************************/
struct GenericFunction * 
generic_function_daxpby(double a, struct GenericFunction * x, 
            double b, struct GenericFunction * y)
{
    //printf("in here! a =%G b = %G\n",a,b);

    struct GenericFunction * out = NULL;
    struct OrthPolyExpansion * p = NULL;
    struct PiecewisePoly * pw = NULL;
    struct OrthPolyExpansion * opetemp = NULL;
    struct PiecewisePoly * pwtemp = NULL;
    if (x == NULL){
        assert ( y != NULL);
        switch (y->fc) {
            case PIECEWISE:
                pwtemp = (struct PiecewisePoly *) y->f;
                pw = piecewise_poly_daxpby(a, NULL, b, pwtemp);
                out = generic_function_alloc(y->dim,y->fc,&(y->sub_type.ptype));
                out->f = pw;
                out->fargs = NULL;
                break;
            case POLYNOMIAL:
                opetemp = (struct OrthPolyExpansion *) y->f;
                p = orth_poly_expansion_daxpby(a, NULL, b, opetemp );
                out = generic_function_alloc(y->dim, y->fc, &(p->p->ptype));
                out->f = p;
                out->fargs = NULL;
                break;    
            case RATIONAL:
                break;
            case KERNEL:
                break;
        }
    }
    else if (y == NULL){
        assert ( x != NULL );
        switch (x->fc) {
            case PIECEWISE:
                //printf("piecewise hereherehere\n");
                pwtemp = (struct PiecewisePoly *) x->f;
                pw = piecewise_poly_daxpby(a, pwtemp, b, NULL);
                out = generic_function_alloc(x->dim,x->fc,&(x->sub_type.ptype));
                out->f = pw;
                out->fargs = NULL;
                //printf("out of it hereherehere\n");
                break;
            case POLYNOMIAL:
                opetemp = (struct OrthPolyExpansion *) x->f;
                p = orth_poly_expansion_daxpby(a, opetemp, b, NULL );
                out = generic_function_alloc(x->dim, x->fc, &(p->p->ptype));
                out->f = p;
                out->fargs = NULL;
                break;    
            case RATIONAL:
                break;
            case KERNEL:
                break;
        }
    }
    else {
        //printf("in the else!\n");
        if (x->fc == y->fc){
            switch (x->fc){
                case PIECEWISE:
                    //printf("get pw_daxpby \n");
                    pw = piecewise_poly_daxpby(
                        a, (struct PiecewisePoly *) x->f,
                        b, (struct PiecewisePoly *) y->f );
                    //printf("got pw_daxpby \n");
                    out = generic_function_alloc(x->dim,x->fc,
                            &(x->sub_type.ptype));

                    out->f = pw;
                    out->fargs = NULL;
                    break;
                case POLYNOMIAL:
                    p = orth_poly_expansion_daxpby(
                        a, (struct OrthPolyExpansion *) x->f,
                        b, (struct OrthPolyExpansion *) y->f );
                    out = generic_function_alloc(x->dim,POLYNOMIAL,&(p->p->ptype));
                    out->f = p;
                    out->fargs = NULL;
                    break;    
                case RATIONAL:
                    break;
                case KERNEL:
                    break;
            }
        }
        else if (x->fc != y->fc){
            //printf("dont match! a=%G, b=%G\n",a,b);
            int apalloc = 0;
            int bpalloc = 0;
            struct PiecewisePoly * ap = NULL;
            struct PiecewisePoly * bp = NULL;
            if (x->fc == POLYNOMIAL){
                ap = piecewise_poly_alloc();
                apalloc = 1;
                ap->ope = x->f;
            }
            else{
                ap = x->f;
            }
            if (y->fc == POLYNOMIAL){
                bp = piecewise_poly_alloc();
                bpalloc = 1;
                bp->ope = y->f;
            }
            else{
                bp = y->f;
            }
            pw = piecewise_poly_daxpby(a, ap, b, bp);
            //printf("got it pw is null? %d\n",pw==NULL);
            /*
            printf("---\n");
            printf("a=%G\n",a);
            print_piecewise_poly(ap,2,NULL);
            printf("b=%G\n",b);
            print_piecewise_poly(bp,2,NULL);
            printf("total \n");
            print_piecewise_poly(pw,2,NULL);
            printf("---\n");
            */
            out = generic_function_alloc(x->dim,PIECEWISE,
                                        &(x->sub_type.ptype));

            out->f = pw;
            out->fargs = NULL;
            if (apalloc == 1){
                free(ap);
            }
            if (bpalloc == 1){
                free(bp);
            }
        }
    }

    //printf("in there!|n");
    return out;
}

/********************************************************//**
*   Add two generic functions z = ax + by where z is preallocated (pa)
*
*   \param a [in] - scaling of first function 
*   \param x [in] - first function (NOT NULL)
*   \param b [in] - scaling of second function
*   \param y [in] - second function (NOT NULL)
*   \param z [inout] - Generic function is allocated but sub function (may) not be
*
*   \note
*       Handling when dealing with PW poly is not yet good.
************************************************************/
void
generic_function_weighted_sum_pa(double a, struct GenericFunction * x, 
        double b, struct GenericFunction * y, struct GenericFunction ** z)
{
    assert (x != NULL);
    assert (y != NULL);
    if  (x->fc != y->fc){
        //printf("here\n");
        generic_function_free(*z); *z = NULL;
        *z = generic_function_daxpby(a,x,b,y);
        assert ( (*z)->f != NULL);
        //printf("there %d\n",(*z)->f == NULL);
        //printf("xnull? %d ynull? %d\n",x->f==NULL,y->f==NULL);

        //fprintf(stderr, "generic_function_weighted_sum_pa cannot yet handle generic functions with different function classes\n");
        //fprintf(stderr, "type of x is %d and type of y is %d --- (0:PIECEWISE,1:POLYNOMIAL)\n",x->fc,y->fc);
        //fprintf(stderr, "type of z is %d\n",z->fc);
        //exit(1);
    }
    else{
        enum function_class fc = x->fc;
        if (fc == POLYNOMIAL){
            if ((*z)->f == NULL){
                (*z)->fc = POLYNOMIAL;
                (*z)->f = orth_poly_expansion_daxpby(
                            a, (struct OrthPolyExpansion *) x->f,
                            b, (struct OrthPolyExpansion *) y->f );
                (*z)->sub_type.ptype = LEGENDRE;
                (*z)->fargs = NULL;
            }
            else{
                fprintf(stderr, "cant handle overwriting functions yet\n");
                exit(1);
            }
        }
        else{
            generic_function_free(*z); (*z) = NULL;
            *z = generic_function_daxpby(a,x,b,y);
        }
    }
}


/********************************************************//**
*   Compute axpby for a an array of generic functions
*
*   \param n [in] - number of functions
*   \param a [in] - scaling for x
*   \param ldx [in] - stride of functions to use in a
*   \param x [in] - functions
*   \param b [in] - scaling for y
*   \param ldy [in] - stride of functions to use in a
*   \param y [in] - functions
*
*   \return fout - array of generic functions
*************************************************************/
struct GenericFunction **
generic_function_array_daxpby(size_t n, double a, size_t ldx, 
        struct GenericFunction ** x, double b, size_t ldy, 
        struct GenericFunction ** y)
{
    struct GenericFunction ** fout = NULL;   
    if (NULL == ( fout = malloc(n*sizeof(struct GenericFunction *)))){
        fprintf(stderr, "failed to allocate in generic_function_array_daxpby.\n");
        exit(1);
    }
    //printf("in daxpby here!\n");
    size_t ii;
    if ( y == NULL){
        for (ii = 0; ii < n ;ii++){
            fout[ii] = generic_function_daxpby(a,x[ii*ldx],0.0, NULL);
        }
    }
    else if (x == NULL){
        for (ii = 0; ii < n ;ii++){
            fout[ii] = generic_function_daxpby(b,y[ii*ldy],0.0, NULL);
        }
    }
    else{
        for (ii = 0; ii < n ;ii++){
            //printf("array daxpby ii=(%zu/%zu)!\n",ii,n);
            fout[ii] = generic_function_daxpby(a,x[ii*ldx],b, y[ii*ldy]);
            //printf("outhere ii=(%zu/%zu)!\n",ii,n);
        }
    }
    //printf("return \n");
    return fout;
}

/********************************************************//**
*   Compute axpby for a an array of generic functions and overwrite into z
*
*   \param n [in] - number of functions
*   \param a [in] - scaling for x
*   \param ldx [in] -  stride of functions to use in a
*   \param x [in] - functions
*   \param b [in] - scaling for y
*   \param ldy [in] - stride of functions to use in a
*   \param y [in] - functions
*   \param ldz [in] - stride for z
*   \param z [inout] -  locations for resulting functions
*************************************************************/
void
generic_function_array_daxpby2(size_t n, double a, size_t ldx, 
        struct GenericFunction ** x, double b, size_t ldy, 
        struct GenericFunction ** y, size_t ldz, struct GenericFunction ** z)
{
    size_t ii;
    if ( y == NULL){
        for (ii = 0; ii < n ;ii++){
            z[ii*ldz] = generic_function_daxpby(a,x[ii*ldx],0.0, NULL);
        }
    }
    else if (x == NULL){
        for (ii = 0; ii < n ;ii++){
            z[ii*ldz] = generic_function_daxpby(b,y[ii*ldy],0.0, NULL);
        }
    }
    else{
        for (ii = 0; ii < n ;ii++){
            z[ii*ldz] = generic_function_daxpby(a,x[ii*ldx],b, y[ii*ldy]);
        }
    }
}


/********************************************************//**
*   Compute a linear combination of generic functions
*
*   \param n [in] - number of functions
*   \param gfarray [in] - array of functions
*   \param coeffs [in] - scaling coefficients
*
*   \return out  = sum_i=1^n coeff[i] * gfarray[i]
************************************************************/
struct GenericFunction *
generic_function_lin_comb(size_t n, struct GenericFunction ** gfarray, 
                            double * coeffs)
{
    // this function is not optimal
    struct GenericFunction * out = NULL;
    struct GenericFunction * temp1 = NULL;
    struct GenericFunction * temp2 = NULL;
    size_t ii;
    if (n == 1){
        out = generic_function_daxpby(coeffs[0],gfarray[0], 0.0, NULL);
    }
    else{
        temp1 = generic_function_daxpby(coeffs[0],gfarray[0],coeffs[1],gfarray[1]);
        for (ii = 2; ii < n; ii++){
            if (ii % 2 == 0){
                temp2 = generic_function_daxpby(coeffs[ii],gfarray[ii],1.0,temp1);
                generic_function_free(temp1);
                temp1 = NULL;
            }
            else{
                temp1 = generic_function_daxpby(coeffs[ii],gfarray[ii],1.0,temp2);
                generic_function_free(temp2);
                temp2 = NULL;
            }
        }
    }
    if (temp1 != NULL){
        return temp1;
    }
    else if (temp2 != NULL){
        return temp2;
    }
    else{
        assert (out != NULL);
        return out;
    }
}

/********************************************************//**
*   Compute a linear combination of generic functions
*
*   \param n [in] - number of functions
*   \param ldgf [in] - stride of array to use
*   \param gfa [in] - array of functions
*   \param ldc [in] - stride of coefficents
*   \param c [in] - scaling coefficients
*
*   \return out  = \f$ \sum_{i=1}^n coeff[ldc[i]] * gfa[ldgf[i]] \f$
************************************************************/
struct GenericFunction *
generic_function_lin_comb2(size_t n, size_t ldgf, 
        struct GenericFunction ** gfa, size_t ldc, double * c)
{
    // this function is not optimal
    struct GenericFunction * out = NULL;
    struct GenericFunction * temp1 = NULL;
    struct GenericFunction * temp2 = NULL;
    size_t ii;
    if (n == 1){
        out = generic_function_daxpby(c[0],gfa[0], 0.0, NULL);
    }
    else{

        int allpoly = 1;
        for (ii = 0; ii < n; ii++){
            if (gfa[ii*ldgf]->fc != POLYNOMIAL){
                allpoly = 0;
                break;
            }
        }

        if (allpoly == 1){
            struct OrthPolyExpansion ** xx = NULL;
            if (NULL == (xx = malloc(n * sizeof(struct OrthPolyExpansion *)))){
                fprintf(stderr, "failed to allocate memmory in generic_function_lin_comb2\n");
                exit(1);
            }
            for  (ii = 0; ii < n; ii++){
                xx[ii] = gfa[ii*ldgf]->f;
            }
            
            struct GenericFunction * gf = generic_function_alloc(1,gfa[0]->fc,&(gfa[0]->sub_type.ptype));
            gf->f = orth_poly_expansion_lin_comb(n,1,xx,ldc,c);
            gf->fargs = NULL;
            free(xx); xx = NULL;

            assert (gf->f != NULL);
            return gf;
        }


        temp1 = generic_function_daxpby(c[0],gfa[0],c[ldc],gfa[ldgf]);
        for (ii = 2; ii < n; ii++){
            if (ii % 2 == 0){
                temp2 = generic_function_daxpby(c[ii*ldc],gfa[ii*ldgf],
                                        1.0,temp1);
                generic_function_free(temp1);
                temp1 = NULL;
            }
            else{
                temp1 = generic_function_daxpby(c[ii*ldc],gfa[ii*ldgf],
                                        1.0,temp2);
                generic_function_free(temp2);
                temp2 = NULL;
            }
        }
    }
    if (temp1 != NULL){
        return temp1;
    }
    else if (temp2 != NULL){
        return temp2;
    }
    else{
        assert (out != NULL);
        return out;
    }
}

/********************************************************//**
    Compute the location and value of the maximum, in absolute value, element of a generic function 

    \param f [in] - function
    \param x [inout] - location of maximum

    \return maxval - absolute value of the maximum
************************************************************/
double generic_function_absmax(struct GenericFunction * f, double * x)
{
    double out = 0.123456789;
    struct OrthPolyExpansion * op = NULL;
    struct PiecewisePoly * pw = NULL;
    switch (f->fc){
        case PIECEWISE:
            pw = f->f;
            out = piecewise_poly_absmax(pw,x);
            break;
        case POLYNOMIAL:
            op = f->f;
            out = orth_poly_expansion_absmax(op,x);
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
    return out;
}




/********************************************************//**
    Compute the index, location and value of the maximum, in absolute value, element of a generic function array

    \param n [in] - number of functions
    \param lda [in] - stride
    \param a [in] - array of functions
    \param ind [inout] - index of maximum
    \param x [inout] - location of maximum

    \return maxval - absolute value of the maximum
************************************************************/
double 
generic_function_array_absmax(size_t n, size_t lda, 
            struct GenericFunction ** a, size_t * ind,  double * x)
{
    size_t ii = 0;
    *ind = ii;
    //printf("do absmax\n");
    //print_generic_function(a[ii],0,NULL);
    double maxval = generic_function_absmax(a[ii],x);
    //printf("maxval=%G\n",maxval);
    double tempval, tempx;
    for (ii = 1; ii < n; ii++){
        tempval = generic_function_absmax(a[ii*lda],&tempx);
        if (tempval > maxval){
            maxval = tempval;
            *x = tempx;
            *ind = ii;
        }
    }
    return maxval;
}

/********************************************************//**
    Scale a generic function

    \param a [in] - value with which to scale the functions
    \param gf [inout] - function to scale
************************************************************/
void generic_function_scale(double a, struct GenericFunction * gf)
{
     switch (gf->fc){
        case PIECEWISE:
            piecewise_poly_scale(a,gf->f);
            break;
        case POLYNOMIAL:
            orth_poly_expansion_scale(a,gf->f);
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
}

/********************************************************//**
    Scale a generic function array

    \param a [in] - value with which to scale the functions
    \param gf [inout] - functions to scale
    \param N [in] - number of functions
************************************************************/
void generic_function_array_scale(double a, struct GenericFunction ** gf,
                                    size_t N)
{
    size_t ii;
    for (ii = 0; ii < N; ii++){
        generic_function_scale(a,gf[ii]);
    }
}

/********************************************************//**
    Helper function for computing the first part of
    \f[
       a kron(\cdot,c)
    \f]
    if left = 1,
    otherwise
    \f[
       kron(\cdot,c)a
    \f]
    
    \param left [in]
    \param r [in]
    \param m [in]
    \param n [in]
    \param l [in]
    \param a [in] - if left = 1 (r, m * n) otherwise (l * m,r)
    \param c [in] - (n,l)
    \param ldc [in] - stride length of c
    \param d [inout] - (rm,l)

************************************************************/
void generic_function_kronh(int left,
                            size_t r, size_t m, size_t n, size_t l, 
                            double * a,
                            struct GenericFunction ** c,
                            struct GenericFunction ** d)
{
    size_t ii,jj,kk;
    if (left == 1){
        for (kk = 0; kk < l; kk++){
            for (jj = 0; jj < m; jj++){
                for (ii = 0; ii < r; ii++){
                    //printf("(%zu/%zu,%zu/%zu,%zu/%zu)\n",jj,m-1,kk,l-1,ii,r-1);
                    d[kk*r*m + jj*r + ii] =
                        generic_function_lin_comb2(
                            n, 1, c + n*kk, r, a + ii + jj*n*r);
                }
            }
        }
    }
    else{
        for (ii = 0; ii < r; ii++){
            for (jj = 0; jj < n; jj++){
                for (kk = 0; kk < m; kk++){
                    d[jj +  kk*n + ii*n*m] = 
                        generic_function_lin_comb2(
                            l, n, c + jj, 1, a + kk*l + ii*l*m);
                }
            }
        }
    }
}

void generic_function_kronh2(int left, size_t r, size_t m, size_t n, size_t l,
        struct GenericFunction ** b, struct GenericFunction ** t,
        struct GenericFunction ** out)
{
    if (left == 1){
        size_t ii,jj, kk;
        for (jj = 0; jj < l; jj++){
            for (kk = 0; kk < m; kk++){
                for (ii = 0; ii < r; ii++){
                    out[ii + kk*r + jj*r*m] =
                        generic_function_sum_prod(
                                n, 1, b + jj*n, r, t + ii + kk*r*n);
                }
            }
        }
    }
    else {
        size_t ii,jj, kk;
        for (ii = 0; ii < r; ii++){
            for (jj = 0; jj < n; jj++){
                for (kk = 0; kk < m; kk++){
                    out[kk + jj*m + ii*n*m] =
                         generic_function_sum_prod(
                                 l, n, b + jj, m, t + kk + ii * l * m);
                }
            }
        }
    }
}

/********************************************************//**
    Return a constant function

    \param a [in] - value of the function
    \param fc [in] - function class
    \param st [in] - function class sub_type
    \param lb [in] - lower bound of function
    \param ub [in] - upper bound of function
    \param aopts [in] - extra arguments depending on function_class, sub_type, etc.

    \return gf [out] - constant function
************************************************************/
struct GenericFunction * 
generic_function_constant(double a, enum function_class fc, void * st,
            double lb, double ub, void * aopts)
{   
    
    struct GenericFunction * gf = NULL;
    switch (fc){
        case PIECEWISE:
            assert (aopts == NULL); // am not expecting this here
            gf = generic_function_alloc(1,fc,st);
            gf->f = piecewise_poly_constant(a,gf->sub_type.ptype,lb,ub);
            gf->fargs = NULL;
            break;
        case POLYNOMIAL:
            assert (aopts == NULL); // am not expecting this here
            gf = generic_function_alloc(1,fc,st);
            gf->f = orth_poly_expansion_constant(a,gf->sub_type.ptype,lb,ub);
            gf->fargs = NULL;
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }

    return gf;

}

/*******************************************************//**
    Return a quadratic function a * (x - offset)^2 = a (x^2 - 2offset x + offset^2)

    \param a [in] - quadratic coefficients
    \param offset [in] - shift of the function
    \param fc [in] - function class
    \param st [in] - function class sub_type
    \param lb [in] - lower bound of function
    \param ub [in] - upper bound of function
    \param aopts [in] - extra arguments depending on function_class, sub_type,  etc.

    \return gf - quadratic
************************************************************/
struct GenericFunction * 
generic_function_quadratic(double a, double offset,
            enum function_class fc, void * st,
            double lb, double ub, void * aopts)
{   
    
    struct GenericFunction * gf = NULL;
    switch (fc){
        case PIECEWISE:
            assert (aopts == NULL); // am not expecting this here
            gf = generic_function_alloc(1,fc,st);
            gf->f = piecewise_poly_quadratic(a,a*(-2)*offset,
                        a*offset*offset, gf->sub_type.ptype,lb,ub);
            gf->fargs = NULL;
            break;
        case POLYNOMIAL:
            assert (aopts == NULL); // am not expecting this here
            gf = generic_function_alloc(1,fc,st);
            gf->f = orth_poly_expansion_quadratic(a,offset,gf->sub_type.ptype,lb,ub);
            gf->fargs = NULL;
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }

    return gf;
}

/*******************************************************//**
    Return a linear function

    \param a [in] -  slope of the function
    \param offset [in] - offset of the function
    \param fc [in] - function class
    \param st [in] - function class sub_type
    \param lb [in] - lower bound of function
    \param ub [in] - upper bound of function
    \param aopts [in] - extra arguments depending on function_class, sub_type, etc.

    \return gf - linear function
***********************************************************/
struct GenericFunction * 
generic_function_linear(double a, double offset,
            enum function_class fc, void * st,
            double lb, double ub, void * aopts)
{   
    
    struct GenericFunction * gf = NULL;
    switch (fc){
        case PIECEWISE:
            assert (aopts == NULL); // am not expecting this here
            gf = generic_function_alloc(1,fc,st);
            gf->f = piecewise_poly_linear(a,offset,  gf->sub_type.ptype,
                            lb,ub);
            gf->fargs = NULL;
            break;
        case POLYNOMIAL:
            assert (aopts == NULL); // am not expecting this here
            gf = generic_function_alloc(1,fc,st);
            gf->f = orth_poly_expansion_linear(a,offset, gf->sub_type.ptype,lb,ub);
            gf->fargs = NULL;
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }

    return gf;

}

/*******************************************************//**
    Fill a generic_function array with orthonormal functions of a particular class and sub_type

    \param n [in] - number of columns
    \param fc [in] - function class
    \param st [in] - function class sub_type
    \param gfarray [inout] - array to fill with functions
    \param args [in] - extra arguments depending on function_class, sub_type, etc.
************************************************************/
void generic_function_array_orth(size_t n, enum function_class fc, void * st,
                            struct GenericFunction ** gfarray, void * args)
{
    size_t ii;
    double lb, ub;
    switch (fc){
        case PIECEWISE:
            lb = ((struct Interval *) args)->lb;
            ub = ((struct Interval *) args)->ub;
            for (ii = 0; ii < n; ii++){
                gfarray[ii] = generic_function_alloc(1,fc,st);
                ((struct PiecewisePoly *)gfarray[ii]->f)->ope =  
                    orth_poly_expansion_genorder(
                            gfarray[ii]->sub_type.ptype,ii,lb,ub);
                gfarray[ii]->fargs = NULL;
            }
            break;
        case POLYNOMIAL:
            lb = ((struct Interval *) args)->lb;
            ub = ((struct Interval *) args)->ub;
            for (ii = 0; ii < n; ii++){
                gfarray[ii] = generic_function_alloc(1,fc,st);
                gfarray[ii]->f = orth_poly_expansion_genorder(gfarray[ii]->sub_type.ptype,ii,lb,ub);
                gfarray[ii]->fargs = NULL;
            }
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }

}

////////////////////////////////////////////////////////////////////
// High dimensional helper functions

/********************************************************//**
    Allocate memory for a fiber cut

    \param totdim [in] - total dimension of underlying function
    \param dim [in] - dimension along which fiber is obtained
************************************************************/
struct FiberCut * alloc_fiber_cut(size_t totdim, size_t dim)
{
    struct FiberCut * fcut;
    if (NULL == ( fcut = malloc(sizeof(struct FiberCut)))){
        fprintf(stderr, "failed to allocate fiber_cut.\n");
        exit(1);
    }
    fcut->totdim = totdim;
    fcut->dimcut = dim;
    fcut->vals = calloc_double(totdim);

    return fcut;
}

/********************************************************//**
    Free memory allocated to a fiber cut

    \param fc [inout] - fiber cut
************************************************************/
void fiber_cut_free(struct FiberCut * fc)
{
    free(fc->vals); fc->vals = NULL;
    free(fc); fc = NULL;
}

/********************************************************//**
    Free memory allocated to an array of fiber cuts

    \param n [in] - number of fiber cuts
    \param fc [inout] - array of fiber cuts
************************************************************/
void fiber_cut_array_free(size_t n, struct FiberCut ** fc)
{
    size_t ii;
    for (ii = 0; ii < n; ii++){
        fiber_cut_free(fc[ii]);
        fc[ii] = NULL;
    }
    free(fc); fc = NULL;
}

/********************************************************//**
    Generate a fibercut of a two dimensional function

    \param f [in] - function to cut
    \param args [in] - function arguments
    \param dim [in] - dimension along which we obtain the cut
    \param val [in] - value of the input which is not *dim*

    \return fcut - struct necessary for computing values in the cut
************************************************************/
struct FiberCut *
fiber_cut_init2d( double (*f)(double, double, void *), void * args, 
                            size_t dim, double val)
{
    struct FiberCut * fcut = alloc_fiber_cut(2,dim);
    fcut->f.f2d = f;
    fcut->args = args;
    fcut->ftype_flag=0;
    if (dim == 0){
        fcut->vals[1] = val;
    }
    else{
        fcut->vals[0] = val;
    }
    return fcut;
}

/********************************************************//**
    Generate an array fibercuts of a two dimensional function

    \param f [in] -  function to cut
    \param args [in] - function arguments
    \param dim [in] - dimension along which we obtain the cut
    \param n [in] - number of fibercuts
    \param val [in] - values of the input for each fibercut 

    \return fcut - array of struct necessary for computing values in the cut
***************************************************************/
struct FiberCut **
fiber_cut_2darray( double (*f)(double, double, void *), void * args, 
                            size_t dim, size_t n, double * val)
{   
    struct FiberCut ** fcut;
    if (NULL == ( fcut = malloc(n *sizeof(struct FiberCut *)))){
        fprintf(stderr, "failed to allocate fiber_cut.\n");
        exit(1);
    }
    size_t ii;
    for (ii = 0; ii < n; ii++){
        fcut[ii] = alloc_fiber_cut(2,dim);
        fcut[ii]->f.f2d = f;
        fcut[ii]->args = args;
        fcut[ii]->ftype_flag = 0;
        if (dim == 0){
            fcut[ii]->vals[1] = val[ii];
        }
        else{
            fcut[ii]->vals[0] = val[ii];
        }

    }
    return fcut;
}

/********************************************************//**
    Generate an array fibercuts of a n-dimensional functions

    \param f [in] -  function to cut
    \param args [in] - function arguments
    \param totdim [in] - total number of dimensions
    \param dim [in] - dimension along which we obtain the cut
    \param n [in] - number of fibercuts
    \param val [in] - array of values of the inputs for each fibercut 

    \return fcut -array of struct necessary for computing values in the cut
***************************************************************/
struct FiberCut **
fiber_cut_ndarray( double (*f)(double *, void *), void * args, 
                            size_t totdim, size_t dim, size_t n, double ** val)
{   
    struct FiberCut ** fcut;
    if (NULL == ( fcut = malloc(n *sizeof(struct FiberCut *)))){
        fprintf(stderr, "failed to allocate fiber_cut.\n");
        exit(1);
    }
    size_t ii;
    for (ii = 0; ii < n; ii++){
        fcut[ii] = alloc_fiber_cut(totdim,dim);
        fcut[ii]->f.fnd = f;
        fcut[ii]->args = args;
        fcut[ii]->ftype_flag = 1;
        memmove(fcut[ii]->vals, val[ii], totdim*sizeof(double));

    }
    return fcut;
}

/********************************************************//**
    Evaluate a fiber of a two dimensional function

    \param x [in] - value at which to evaluate
    \param vfcut [in] - void pointer to fiber_cut structure

    \return val - value of the function
************************************************************/
double fiber_cut_eval2d(double x, void * vfcut){
    
    struct FiberCut * fcut = vfcut;
    
    double val;
    if (fcut->dimcut == 0){
        val = fcut->f.f2d(x, fcut->vals[1], fcut->args);
    }
    else{
        val = fcut->f.f2d(fcut->vals[0], x, fcut->args);
    }
    return val;
}

/********************************************************//**
    Evaluate a fiber of an n dimensional function

    \param x [in] - value at which to evaluate
    \param vfcut [in] - void pointer to fiber_cut structure

    \return val - value of the function
************************************************************/
double fiber_cut_eval(double x, void * vfcut){
    
    struct FiberCut * fcut = vfcut;
    
    double val;
    fcut->vals[fcut->dimcut] = x;
    val = fcut->f.fnd(fcut->vals, fcut->args);
    return val;
}


/////////////////////////////////////////////////////////
// Utilities
void print_generic_function(struct GenericFunction * gf, size_t prec,void * args){
    switch (gf->fc){
        case PIECEWISE:
            print_piecewise_poly((struct PiecewisePoly *)gf->f,prec,args);
            break;
        case POLYNOMIAL:
            print_orth_poly_expansion((struct OrthPolyExpansion *)gf->f,prec,args);
            break;    
        case RATIONAL:
            break;
        case KERNEL:
            break;
    }
}
