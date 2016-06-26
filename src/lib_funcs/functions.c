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
#include "linelm.h"


/********************************************************//**
    Allocate memory for a generic function without specifying class or sub_type

    \param[in] dim - dimension of functions

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
    
    \param[in] size - size of array

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

    \param[in] dim      - dimension of functions
    \param[in] fc       - function class

    \return generic function
************************************************************/
struct GenericFunction *
generic_function_alloc(size_t dim, enum function_class fc)
{
    
    struct GenericFunction * out;
    if (NULL == ( out = malloc(sizeof(struct GenericFunction)))){
        fprintf(stderr, "failed to allocate for a generic function.\n");
        exit(1);
    }
    out->dim = dim;
    out->fc = fc;
    out->f = NULL;
    out->fargs = NULL;
    return out;
}

/********************************************************//**
    (Deep) Copy a generic function

    \param[in] gf - generic function

    \return generic function
************************************************************/
struct GenericFunction * 
generic_function_copy(const struct GenericFunction * gf)
{
    struct GenericFunction * out = NULL; 

    out = generic_function_alloc(gf->dim, gf->fc);
    switch (gf->fc){
    case CONSTANT:                                             break;
    case PIECEWISE:  out->f = piecewise_poly_copy(gf->f);      break;
    case POLYNOMIAL: out->f = orth_poly_expansion_copy(gf->f); break;
    case LINELM:     out->f = lin_elem_exp_copy(gf->f);        break;
    case RATIONAL:                                             break;
    case KERNEL:                                               break;
    }
    
    return out;
}

/********************************************************//**
    Copy a generic function to a preallocated generic function

    \param[in]     gf   - generic function
    \param[in,out] gfpa - preallocated function

************************************************************/
void generic_function_copy_pa(const struct GenericFunction * gf, 
                              struct GenericFunction * gfpa)
{
    gfpa->fc = gf->fc;
    switch (gf->fc){
    case CONSTANT:                                              break;
    case PIECEWISE:  gfpa->f = piecewise_poly_copy(gf->f);      break;
    case POLYNOMIAL: gfpa->f = orth_poly_expansion_copy(gf->f); break;
    case LINELM:     gfpa->f = lin_elem_exp_copy(gf->f);        break;
    case RATIONAL:                                              break;
    case KERNEL:                                                break;
    }
}

/********************************************************//**
    Free memory for generic function

    \param[in,out] gf - generic function
************************************************************/
void generic_function_free(struct GenericFunction * gf){
    if (gf != NULL){
        if (gf->f != NULL) {
            switch (gf->fc){
            case CONSTANT:                                    break;
            case PIECEWISE:  piecewise_poly_free(gf->f);      break;
            case POLYNOMIAL: orth_poly_expansion_free(gf->f); break;
            case LINELM:     lin_elem_exp_free(gf->f);        break;
            case RATIONAL:                                    break;
            case KERNEL:                                      break;
            }
            gf->f = NULL;
        }
        free(gf); gf = NULL;
    }
}

/********************************************************//**
    Free memory for generic function array

    \param[in,out] gf - generic function array
    \param[in]     n  - number of generic functions in the array
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
*   \param[in,out] ser       - location to serialize to
*   \param[in]     gf        - generic function
*   \param[in]     totSizeIn - if not null then only total size 
                               in bytes of generic function si returned 
*                              if NULL then serialization occurs
*
*   \return ptr - ser + num_bytes
************************************************************/
unsigned char *
serialize_generic_function(unsigned char * ser, 
                           const struct GenericFunction * gf, 
                           size_t * totSizeIn)
{   
    // order = 
    // function_class -> sub_type -> function
    unsigned char * ptr = ser;
    size_t totSize = sizeof(int) + sizeof(size_t); // for function class and dim
    if (totSizeIn != NULL){
        size_t sizef = 0;
        switch (gf->fc){
        case CONSTANT:   printf("not ser const correct\n");                 break;
        case PIECEWISE:  serialize_piecewise_poly(NULL,gf->f, &sizef);      break;
        case POLYNOMIAL: serialize_orth_poly_expansion(NULL,gf->f, &sizef); break;
        case LINELM:     serialize_lin_elem_exp(NULL,gf->f, &sizef);        break;
        case RATIONAL:                                                      break;
        case KERNEL:                                                        break;
        }
        totSize += sizef;
        *totSizeIn = totSize;
        return ptr;
    }
    else{
        ptr = serialize_size_t(ptr, gf->dim);
        ptr = serialize_int(ptr, gf->fc);
        switch (gf->fc){
        case CONSTANT:   printf("not ser const correct\n");                    break;
        case PIECEWISE:  ptr = serialize_piecewise_poly(ptr,gf->f, NULL);      break;
        case POLYNOMIAL: ptr = serialize_orth_poly_expansion(ptr,gf->f, NULL); break;
        case LINELM:     ptr = serialize_lin_elem_exp(ptr,gf->f, NULL);        break;
        case RATIONAL:                                                         break;
        case KERNEL:                                                           break;
        }
    }
    return ptr;

}

/********************************************************//**
*   Deserialize a generic function
*
*   \param[in]     ser - serialized function
*   \param[in,out] gf  -  generic function
*
*   \return ptr = ser + nBytes of gf
************************************************************/
unsigned char *
deserialize_generic_function(unsigned char * ser, 
                             struct GenericFunction ** gf)
{
    
    int fci;
    size_t dim;
    enum function_class fc;
    
    unsigned char * ptr = deserialize_size_t(ser, &dim);
    ptr = deserialize_int(ptr, &fci);
    fc = (enum function_class) fci;
    *gf = generic_function_alloc(dim,fc);

    struct PiecewisePoly * pw = NULL;
    struct OrthPolyExpansion * ope = NULL;
    struct LinElemExp * le = NULL;
    switch (fc){
    case CONSTANT:   printf("not ser const correct\n");                               break;
    case PIECEWISE:  ptr = deserialize_piecewise_poly(ptr,&pw);       (*gf)->f = pw;  break;
    case POLYNOMIAL: ptr = deserialize_orth_poly_expansion(ptr,&ope); (*gf)->f = ope; break;
    case LINELM:     ptr = deserialize_lin_elem_exp(ptr,&le);         (*gf)->f = le;  break;
    case RATIONAL:                                                                    break;
    case KERNEL:                                                                      break;
    }
    return ptr;
}

////////////////////////////////////////////////////////////
// Various Initializers

/********************************************************//**
    Return a constant function

    \param[in] a     - value of the function
    \param[in] fc    - function class
    \param[in] aopts - extra arguments depending on function_class, sub_type, etc.

    \return gf  - constant function
************************************************************/
struct GenericFunction * 
generic_function_constant(double a, enum function_class fc, void * aopts)
{   
    
    struct GenericFunction * gf = generic_function_alloc(1,fc);
    switch (fc){
    case CONSTANT:                                                  break;
    case PIECEWISE:  gf->f = piecewise_poly_constant(a,aopts);      break;
    case POLYNOMIAL: gf->f = orth_poly_expansion_constant(a,aopts); break;
    case LINELM:     gf->f = lin_elem_exp_constant(a,aopts);        break;
    case RATIONAL:                                                  break;
    case KERNEL:                                                    break;
    }
    return gf;
}

/*******************************************************//**
    Return a linear function

    \param[in] a      - slope of the function
    \param[in] offset - offset of the function
    \param[in] fc     - function class
    \param[in] aopts  - extra arguments depending on function_class, 
                        sub_type, etc.

    \return gf - linear function
***********************************************************/
struct GenericFunction * 
generic_function_linear(double a, double offset,
                        enum function_class fc, void * aopts)
{   
    struct GenericFunction * gf = generic_function_alloc(1,fc);
    switch (fc){
    case CONSTANT:                                                       break;
    case PIECEWISE:  gf->f = piecewise_poly_linear(a,offset,aopts);      break;
    case POLYNOMIAL: gf->f = orth_poly_expansion_linear(a,offset,aopts); break;
    case LINELM:     gf->f = lin_elem_exp_linear(a,offset,aopts);        break;
    case RATIONAL:                                                       break;
    case KERNEL:                                                         break;
    }
    return gf;
}

/*******************************************************//**
    Return a quadratic function a * (x - offset)^2 = a (x^2 - 2offset x + offset^2)

    \param[in] a      - quadratic coefficients
    \param[in] offset - shift of the function
    \param[in] fc     - function class
    \param[in] aopts  - extra arguments depending on function_class, sub_type,  etc.

    \return gf - quadratic
************************************************************/
struct GenericFunction * 
generic_function_quadratic(double a, double offset,
                           enum function_class fc, void * aopts)
{   
    struct GenericFunction * gf = generic_function_alloc(1,fc);
    switch (fc){
    case CONSTANT:                                                                            break;
    case PIECEWISE:  gf->f = piecewise_poly_quadratic(a,a*(-2)*offset,a*offset*offset,aopts); break;
    case POLYNOMIAL: gf->f = orth_poly_expansion_quadratic(a,offset,aopts);                   break;
    case LINELM:     fprintf(stderr,"Cannot make quadratic out of LINELM\n");                 exit(1);        
    case RATIONAL:                                                                            break;
    case KERNEL:                                                                              break;
    }
    return gf;
}

/********************************************************//**
    Create a pseudo-random polynomial generic function 

*   \param[in] ptype    - polynomial type
*   \param[in] maxorder - maximum order of the polynomial
*   \param[in] lower    - lower bound of input
*   \param[in] upper    - upper bound of input

    \return gf - generic function
************************************************************/
struct GenericFunction * 
generic_function_poly_randu(enum poly_type ptype,
                            size_t maxorder, double lower, 
                            double upper)
{
    enum function_class fc = POLYNOMIAL;
    struct GenericFunction * gf = generic_function_alloc(1,fc);
    gf->f = orth_poly_expansion_randu(ptype,maxorder,lower,upper);
    gf->fargs = NULL;
    return gf;
}

/********************************************************//**
    Take the derivative of a generic function

    \param[in] gf - generic function

    \return generic function representing the derivative
************************************************************/
struct GenericFunction * 
generic_function_deriv(const struct GenericFunction * gf)
{
    struct GenericFunction * out = generic_function_alloc(1,gf->fc);
    switch (gf->fc){
    case CONSTANT:                                              break;
    case PIECEWISE:  out->f = piecewise_poly_deriv(gf->f);      break;
    case POLYNOMIAL: out->f = orth_poly_expansion_deriv(gf->f); break;
    case LINELM:     out->f = lin_elem_exp_deriv(gf->f);        break;
    case RATIONAL:                                              break;
    case KERNEL:                                                break;
    }
    return out;
}



/////////////////////////////////////////////////////////////
// Linear Algebra stuff

 /********************************************************//**
 *   Add two generic functions z = ax + by
 *
 *   \param[in] a - scaling of first function
 *   \param[in] x - first function
 *   \param[in] b - scaling of second function
 *   \param[in] y - second function
 *
 *   \return generic function 
 *
 *   \note
 *       Handling the function class of the output is not very smart
 ************************************************************/
 struct GenericFunction * 
 generic_function_daxpby(double a, const struct GenericFunction * x, 
                         double b, const struct GenericFunction * y)
 {
     //printf("in here! a =%G b = %G\n",a,b);

     struct GenericFunction * out = NULL;
     struct PiecewisePoly * pw = NULL;
     if (x == NULL){
         assert ( y != NULL);
         out = generic_function_alloc(y->dim,y->fc);
         switch (y->fc) {
         case CONSTANT:                                                                 break;
         case PIECEWISE: out->f = piecewise_poly_daxpby(a, NULL, b, y->f);              break;
         case POLYNOMIAL: out->f = orth_poly_expansion_daxpby(a, NULL, b, y->f);        break;
         case LINELM: out->f = lin_elem_exp_copy(y->f); lin_elem_exp_scale(b,out->f);   break;
         case RATIONAL:                                                                 break;
         case KERNEL:                                                                   break;
         }
     }
     else if (y == NULL){
         assert ( x != NULL );
         out = generic_function_alloc(x->dim,x->fc);
         switch (x->fc) {
         case CONSTANT:                                                                 break;
         case PIECEWISE: out->f = piecewise_poly_daxpby(a, x->f, b, NULL);              break;
         case POLYNOMIAL: out->f = orth_poly_expansion_daxpby(a, x->f, b, NULL);        break;
         case LINELM: out->f = lin_elem_exp_copy(x->f); lin_elem_exp_scale(a,out->f);   break;
         case RATIONAL:                                                                 break;
         case KERNEL:                                                                   break;
         }
    }
    else {
        /* printf("in the else!\n"); */
        if (x->fc == y->fc){
            out = generic_function_alloc(x->dim,x->fc);
            switch (x->fc) {
            case CONSTANT:                                                                 break;
            case PIECEWISE: out->f = piecewise_poly_daxpby(a, x->f, b, y->f);              break;
            case POLYNOMIAL: out->f = orth_poly_expansion_daxpby(a, x->f, b, y->f);        break;
            case LINELM:
                out->f = lin_elem_exp_copy(x->f);
                lin_elem_exp_scale(a,out->f);
                lin_elem_exp_axpy(b,y->f,out->f);
                break;
            case RATIONAL:                                                                 break;
            case KERNEL:                                                                   break;
            }
        }
        else if (x->fc != y->fc){
            if ((x->fc == LINELM) || (y->fc == LINELM)){
                fprintf(stderr,
                        "Can't add linear elements with other stuff\n");
                exit(1);
            }
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
            out = generic_function_alloc(x->dim,PIECEWISE);


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
 *   Compute the inner product between two generic functions
 *
 *   \param[in] a  - generic function
 *   \param[in] b  - generic function
 *
 *   \return out -  int a(x) b(x) dx 
 ************************************************************/
 double generic_function_inner(const struct GenericFunction * a, 
                               const struct GenericFunction * b)
{
     double out = 0.123456789;   
     enum function_class fc = a->fc;
     int apalloc = 0;
     int bpalloc = 0;
     struct PiecewisePoly * ap = NULL;
     struct PiecewisePoly * bp = NULL;
     if ( (a->fc != b->fc) || (a->fc == PIECEWISE) ){
         assert (a->fc != LINELM);
         assert (b->fc != LINELM);
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
     case CONSTANT:
         assert (1 == 0);
         break;
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
     case LINELM:
         out = lin_elem_exp_inner(a->f,b->f);
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
 *   \param[in] n   - number of inner products
 *   \param[in] lda - stride of functions to use in a
 *   \param[in] a   - first array of generic functions
 *   \param[in] ldb - stride of functions to use in b
 *   \param[in] b   - second array of generic functions
 *
 *   \return val - sum_{i=1^N} int a[ii*lda](x) b[ii*ldb](x) dx
 ************************************************************/
 double generic_function_inner_sum(size_t n, size_t lda, 
                                   struct GenericFunction ** a, 
                                   size_t ldb, 
                                   struct GenericFunction ** b)
 {
     double val = 0.0;
     size_t ii;
     for (ii = 0; ii < n; ii++){
         val += generic_function_inner(a[ii*lda], b[ii*ldb]);
     }
     return val;
 }

/********************************************************//**
*   Compute norm of a generic function
*
*   \param[in] f  - generic function
*
*   \return out - norm
************************************************************/
double generic_function_norm(const struct GenericFunction * f){
    
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
 *   \param[in] f1 - generic function
 *   \param[in] f2 - generic function
 *
 *   \return out - norm of difference
 ************************************************************/
 double generic_function_norm2diff(const struct GenericFunction * f1, 
                                   const struct GenericFunction * f2)
 {
     struct GenericFunction * f3 = generic_function_daxpby(1.0,f1,-1.0,f2);
     double out = generic_function_norm(f3);
     generic_function_free(f3); f3 = NULL;
     return out;
 }

 /********************************************************//**
 *   Compute the norm of the difference between two generic function arrays
 *   
 *   \param[in] n    - number of elements
 *   \param[in] f1   - generic function array
 *   \param[in] inca - incremenent of first array
 *   \param[in] f2   - generic function array
 *   \param[in] incb - incremenent of second array
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
 *   \param[in] f - generic function
 *
 *   \return out - integral
 ************************************************************/
 double generic_function_integral(const struct GenericFunction * f){
     
     assert (f != NULL);
     /* printf("integrating fc = %d\n",f->fc); */
     double out = 0.0;
     switch (f->fc){
     case CONSTANT:   assert(1==0);                              break;
     case PIECEWISE:  out = piecewise_poly_integrate(f->f);      break;
     case POLYNOMIAL: out = orth_poly_expansion_integrate(f->f); break;
     case LINELM:     out = lin_elem_exp_integrate(f->f);        break;
     case RATIONAL:                                              break;
     case KERNEL:                                                break;
     }
     return out;
 }

 /********************************************************//**
 *   Compute the integral of all the functions in a generic function array
 *
 *   \param[in] n   - number of functions
 *   \param[in] lda - stride
 *   \param[in] a   - array of generic functions
 *
 *   \return out - array of integrals
 ************************************************************/
 double * 
 generic_function_integral_array(size_t n,size_t lda,struct GenericFunction ** a)
 {
     double * out = calloc_double(n);
     size_t ii;
     for (ii = 0; ii < n; ii++){
         out[ii] = generic_function_integral(a[ii*lda]);
     }
     return out;
 }

 /********************************************************//**
 *   Create a nodal basis at particular points
 *
 *   \param[in] f   - function to interpolate
 *   \param[in] N   - number of nodes
 *   \param[in] x   - locations of nodes
 *
 *   \return nodal basis (LINELM) function
 ************************************************************/
struct GenericFunction *
generic_function_create_nodal(struct GenericFunction * f,size_t N, double * x)
{
    struct GenericFunction * out = NULL;
    out = generic_function_alloc(f->dim,LINELM);
    out->fargs = NULL;
    double * fvals = calloc_double(N);
    for (size_t ii = 0; ii < N; ii++){
        fvals[ii] = generic_function_1d_eval(f,x[ii]);
    }
    out->f = lin_elem_exp_init(N,x,fvals);
    free(fvals); fvals = NULL;

    return out;
}

struct GenericFunction * 
generic_function_onezero2(
    enum function_class fc,
    size_t nzeros,
    double * zero_locations,
    void * opts
    )
{
    assert (fc == LINELM);
    struct GenericFunction * gf = generic_function_alloc(1,fc);
    gf->f = lin_elem_exp_onezero(nzeros, zero_locations, opts);
    return gf;
}


void generic_function_array_onezero(
    struct GenericFunction ** L,
    size_t n,
    enum function_class fc,
    size_t upto,
    size_t * piv,
    double * px,
    void * opts)
//    void * opt_args)
{
    //create an arbitrary array that has zeros at piv[:upto-1],px[:upto-1]
    // and one at piv[upto],piv[upto] less than one every else
    
    // note that need to set piv[upto] and px[upto] in this function
    // number of pivots per array

    size_t * npiv = calloc_size_t(n); 
    double ** x = malloc_dd(n); // pivots per function
    for (size_t ii = 0; ii < upto; ii++){
        npiv[piv[ii]]++;
    }

    for (size_t ii = 0; ii < n; ii++){
        x[ii] = calloc_double(npiv[ii]);
        size_t on = 0;
        for (size_t jj = 0; jj < upto; jj++){
            if (piv[jj] == ii){
                x[ii][on] = px[jj];
                on++;
            }
        }
    }

    for (size_t ii = 0; ii < n; ii++){
        L[ii] = generic_function_onezero2(fc,npiv[ii],x[ii],opts);
    }

    double xval;
    size_t amind;
    generic_function_array_absmax(n, 1, L,&amind, &xval,NULL);//optargs);
    px[upto] = xval;
    piv[upto] = amind;
    double val = generic_function_1d_eval(L[piv[upto]],px[upto]);
    generic_function_array_scale(1.0/val,L,n);

    free_dd(n,x);
    free(npiv); npiv = NULL;
}

struct GenericFunction *
generic_function_onezero(enum function_class fc, double one, size_t nz,
                         double * zeros, double lb, double ub)
{
    assert (fc == LINELM);
    struct GenericFunction * f = 
        generic_function_alloc(1, fc);

    struct LinElemExp * lexp = lin_elem_exp_alloc();
    lexp->num_nodes = nz+3;
    lexp->nodes = calloc_double(nz+3);
    lexp->coeff = calloc_double(nz+3);
    
    lexp->nodes[0] = lb;
    size_t ind = 1;
    int alloc = 0;
    for (size_t ii = 0; ii < nz; ii++){
        if (zeros[ii] < one){
            lexp->nodes[ind] = zeros[ii];
            ind++;
        }
        else if (alloc == 0){
//            printf("lets go\n");
            lexp->nodes[ind] = one;
            lexp->coeff[ind] = 1.0;
            ind++;
            lexp->nodes[ind] = zeros[ii];
            ind++;
            alloc = 1;
        }
        else{
            lexp->nodes[ind] = zeros[ii];
            ind++;
        }
    }
    if (alloc == 0){
        lexp->nodes[ind] = one;
        lexp->coeff[ind] = 1.0;
        ind++;
    }
    assert (ind == nz+2);
    lexp->nodes[nz+2] = ub;
    f->f = lexp;

    return f;
}





 /********************************************************//**
 *   Compute the sum of the product between the functions of two function arrays
 *
 *   \param[in] n   - number of functions
 *   \param[in] lda - stride of first array
 *   \param[in] a   - array of generic functions
 *   \param[in] ldb - stride of second array
 *   \param[in] b   - array of generic functions
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

         struct GenericFunction * gf = generic_function_alloc(1,a[0]->fc);
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
 *   \param[in] a  - generic function
 *   \param[in] b  - generic function
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
     case CONSTANT: break;
     case PIECEWISE:
         out = generic_function_alloc(a->dim,PIECEWISE);
         out->fargs = a->fargs;
         out->f = piecewise_poly_prod(ap, bp);
         break;
     case POLYNOMIAL:
         out = generic_function_alloc(a->dim,a->fc);
         out->fargs = a->fargs;
         out->f = orth_poly_expansion_prod(
             (struct OrthPolyExpansion *) a->f,
             (struct OrthPolyExpansion *) b->f);
         break;
     case LINELM:
         out = generic_function_alloc(a->dim,a->fc);
         out->fargs = a->fargs;
         out->f = lin_elem_exp_prod(a->f,b->f,NULL);
         /* fprintf(stderr,"lin_elem_exp_prod not yet implemeted\n"); */
         /* exit(1); */
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
 *   Compute the norm of an array of generic functions
 *
 *   \param[in] n   - number of functions
 *   \param[in] lda - stride of functions to use in a
 *   \param[in] a   - functions
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
 *   \param[in,out] f - number of functions
 ************************************************************/
 void generic_function_flip_sign(struct GenericFunction * f)
 {
     switch (f->fc){
     case CONSTANT:                                        break;
     case PIECEWISE:  piecewise_poly_flip_sign(f->f);      break;
     case POLYNOMIAL: orth_poly_expansion_flip_sign(f->f); break;
     case LINELM:     lin_elem_exp_flip_sign(f->f);        break;
     case RATIONAL:                                        break;
     case KERNEL  :                                        break;
     }
 }

 /********************************************************//**
 *   Flip the sign of each generic function in an array
 *
 *   \param[in]     n   - number of functions
 *   \param[in]     lda - stride of array
 *   \param[in,out] a   - array of functions
 ************************************************************/
 void 
 generic_function_array_flip_sign(size_t n, size_t lda, 
                                  struct GenericFunction ** a){
     size_t ii;
     for (ii = 0; ii < n; ii++){
         generic_function_flip_sign(a[ii*lda]);
     }
 }

 /********************************************************//**
 *   Get the lower bound of a generic function 
 *
 *   \param[in] f - function
 *
 *   \return lower bound
 ************************************************************/
 double generic_function_get_lower_bound(const struct GenericFunction * f){
     
     double lb = -0.123456789;

     switch (f->fc){
     case CONSTANT:                                        break;
     case PIECEWISE:  lb = piecewise_poly_lb(f->f);             break;
     case POLYNOMIAL: lb = ((struct OrthPolyExpansion *) f->f)->lower_bound; break;
     case LINELM:     lb = lin_elem_exp_lb(f->f);          break;
     case RATIONAL:                                        break;
     case KERNEL:                                        break;
     }
     return lb;
 }

 /********************************************************//**
 *   Get the upper bound of a generic function 
 *
 *   \param[in] f - function
 *
 *   \return upper bound
 ************************************************************/
 double generic_function_get_upper_bound(const struct GenericFunction * f){
     double ub = 0.123456789;
     switch (f->fc){
     case CONSTANT:                                        break;
     case PIECEWISE:  ub = piecewise_poly_ub(f->f);        break;
     case POLYNOMIAL: ub = ((struct OrthPolyExpansion *) f->f)->upper_bound; break;
     case LINELM:     ub = lin_elem_exp_ub(f->f);          break;
     case RATIONAL:                                        break;
     case KERNEL:                                          break;
     }

     return ub;
 }

 /********************************************************//**
 *   Get the function class
 ************************************************************/
enum function_class generic_function_get_fc(const struct GenericFunction * f)
{
    assert (f != NULL);
    return f->fc;
}


 /********************************************************//**
 *   Evaluate a generic function
 *
 *   \param[in] f  - function
 *   \param[in] x  - location at which to evaluate
 *
 *   \return evaluation
 ************************************************************/
 double generic_function_1d_eval(const struct GenericFunction * f, double x){
     assert (f != NULL);
     double out = 0.1234567890;
     /* printf("f->fc = %d\n",f->fc); */
     switch (f->fc){
     case CONSTANT:   assert(1 == 0);                         break;
     case PIECEWISE:  out = piecewise_poly_eval(f->f,x);      break;
     case POLYNOMIAL: out = orth_poly_expansion_eval(f->f,x); break;
     case LINELM:     out = lin_elem_exp_eval(f->f,x);        break;
     case RATIONAL:                                           break;
     case KERNEL:                                             break;
     }

     return out;
 }

 /********************************************************//**
 *   Evaluate a generic function consisting of nodal
 *   basis functions at some node
 *
 *   \param[in] f   - function
 *   \param[in] ind - location at which to evaluate
 *
 *   \return evaluation
 ************************************************************/
 double generic_function_1d_eval_ind(const struct GenericFunction * f, size_t ind)
 {
     assert (f != NULL);
     assert (f->fc == LINELM);
     double out = 0.1234567890;
     out = lin_elem_exp_get_nodal_val(f->f,ind);

     return out;
 }

 /********************************************************//**
 *   Evaluate a generic function consisting of nodal
 *   basis functions at some node
 *
 *   \param[in] f    - function
 *   \param[in] x    - location at which to Evaluate
 *   \param[in] size - byte size of location (sizeof(double) or (sizeof(size_t)))
 *
 *   \return evaluation
 ************************************************************/
double generic_function_1d_eval_gen(const struct GenericFunction * f, 
                                    void * x, size_t size)
{
     assert (f != NULL);

     size_t dsize = sizeof(double);
     size_t stsize = sizeof(size_t);
     double out;
     if (size == dsize){
         out = generic_function_1d_eval(f,*(double *)x);
     }
     else if (size == stsize){
         out = generic_function_1d_eval_ind(f,*(size_t *)x);
     }
     else{
         fprintf(stderr, "Cannot evaluate generic function at \n");
         fprintf(stderr, "input of byte size %zu\n ", size);
         exit(1);
     }

     return out;
 }

 /********************************************************//**
 *   Evaluate an array of generic functions
 *
 *   \param[in] n - number of functions
 *   \param[in] f - array of functions
 *   \param[in] x - location at which to evaluate
 *
 *   \return array of values
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
   Evaluate a generic function array at a given pivot
************************************************************/
double generic_function_1darray_eval_piv(struct GenericFunction ** f, 
                                         struct Pivot * piv)
{
    size_t size = pivot_get_size(piv);
    size_t ind = pivot_get_ind(piv);
    void * loc = pivot_get_loc(piv);
    double out = generic_function_1d_eval_gen(f[ind],loc,size);
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
 *   \param[in]     a - scaling of first function
 *   \param[in]     x - first function
 *   \param[in,out] y - second function
 *
 *   \return 0 if successfull, 1 if error
 *
 *   \note
 *   Handling the function class of the output is not very smart
 ************************************************************/
 int generic_function_axpy(double a, const struct GenericFunction * x, 
                           struct GenericFunction * y)
 {
     //printf("in here! a =%G b = %G\n",a,b);

     assert (y != NULL);
     assert (x != NULL);
     assert (x->fc == y->fc);

     int out = 1;
     switch (x->fc){
     case CONSTANT:                                                break;
     case PIECEWISE:                                               break;
     case POLYNOMIAL: out = orth_poly_expansion_axpy(a,x->f,y->f); break;
     case LINELM:     out = lin_elem_exp_axpy(a,x->f,y->f);        break;
     case RATIONAL:                                                break;
     case KERNEL:                                                  break;
     }
     return out;
 }

 /********************************************************//**
 *   Add generic functions \f$ y[i] \leftarrow a x[i] + y[i] \f$
 *
 *   \param[in]     n - number of functions
 *   \param[in]     a - scaling of the first functions
 *   \param[in]     x - first function array
 *   \param[in,out] y - second function array
 *
 *   \return 0 if successfull, 1 if error
 *
 *   \note
 *       Handling the function class of the output is not very smart
 ************************************************************/
 int generic_function_array_axpy(size_t n, double a, 
                                 struct GenericFunction ** x, 
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
*   Add two generic functions z = ax + by where z is preallocated (pa)
*
*   \param[in]     a - scaling of first function 
*   \param[in]     x - first function (NOT NULL)
*   \param[in]     b - scaling of second function
*   \param[in]     y - second function (NOT NULL)
*   \param[in,out] z - Generic function is allocated but sub function (may) not be
*
*   \note
*   Handling when dealing with PW poly is not yet good.
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
                (*z)->f = orth_poly_expansion_daxpby(a, x->f,b, y->f );
                (*z)->fargs = NULL;
            }
            else{
                fprintf(stderr, "cant handle overwriting functions yet\n");
                exit(1);
            }
        }
        else if (fc == LINELM){
            assert ((*z)->f == NULL);
            (*z)->fc = LINELM;
            (*z)->f = lin_elem_exp_copy(y->f);
            lin_elem_exp_scale(b,(*z)->f);
            lin_elem_exp_axpy(a,x->f,(*z)->f);
            (*z)->fargs = NULL;
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
*   \param[in] n   - number of functions
*   \param[in] a   - scaling for x
*   \param[in] ldx - stride of functions to use in a
*   \param[in] x   - functions
*   \param[in] b   - scaling for y
*   \param[in] ldy - stride of functions to use in a
*   \param[in] y   - functions
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
*   \param[in]     n   - number of functions
*   \param[in]     a   - scaling for x
*   \param[in]     ldx -  stride of functions to use in a
*   \param[in]     x   - functions
*   \param[in]     b   - scaling for y
*   \param[in]     ldy - stride of functions to use in a
*   \param[in]     y   - functions
*   \param[in]     ldz - stride for z
*   \param[in,out] z   -  locations for resulting functions
*************************************************************/
void
generic_function_array_daxpby2(size_t n, double a, size_t ldx, 
        struct GenericFunction ** x, double b, size_t ldy, 
        struct GenericFunction ** y, size_t ldz, 
                               struct GenericFunction ** z)
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
*   \param[in] n       - number of functions
*   \param[in] gfarray - array of functions
*   \param[in] coeffs  - scaling coefficients
*
*   \return out  = sum_i=1^n coeff[i] * gfarray[i]
************************************************************/
struct GenericFunction *
generic_function_lin_comb(size_t n,struct GenericFunction ** gfarray, 
                          const double * coeffs)
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
*  Round an generic function to some tolerance
*
*  \param[in,out] gf     - generic function
*  \param[in]     thresh - threshold (relative) to round to
*
*  \note
*  (UNTESTED, use with care!!!! 
*************************************************************/
void generic_function_roundt(struct GenericFunction ** gf, double thresh)
{
    struct OrthPolyExpansion * ope = NULL;
    switch ((*gf)->fc){
    case CONSTANT:                                                            break;
    case PIECEWISE:                                                           break;
    case POLYNOMIAL: ope = (*gf)->f; orth_poly_expansion_roundt(&ope,thresh); break;
    case LINELM:                                                              break;
    case RATIONAL:                                                            break;
    case KERNEL:                                                              break;
    }
}

/********************************************************//**
*   Evaluate an array of generic functions
*
*   \param[in]     n   - number of functions
*   \param[in]     f   - array of functions
*   \param[in]     x   - location at which to evaluate
*   \param[in,out] out - array of values
************************************************************/
void
generic_function_1darray_eval2(size_t n, 
                               struct GenericFunction ** f, 
                               double x, double * out)
{
    int allpoly = 1;
    struct OrthPolyExpansion * parr[1000];
    for (size_t ii = 0; ii < n; ii++){
        if (f[ii]->fc != POLYNOMIAL){
            allpoly = 0;
            break;
        }
        parr[ii] = f[ii]->f;
    }
    if ((allpoly == 1) && (n <= 1000)){
        int res = legendre_poly_expansion_arr_eval(n,parr,x,out);
        if (res == 1){ //something when wrong
            size_t ii;
            for (ii = 0; ii < n; ii++){
                out[ii] = generic_function_1d_eval(f[ii],x);
            }
        }
    }
    else{
        size_t ii;
        for (ii = 0; ii < n; ii++){
            out[ii] = generic_function_1d_eval(f[ii],x);
        }
    }
}

/********************************************************//**
*   Evaluate an array of generic functions which should be
*   of nodal basis class at particular nodal locations
*
*   \param[in]     n   - number of functions
*   \param[in]     f   - array of functions
*   \param[in]     ind - location at which to evaluate
*   \param[in,out] out - array of values
************************************************************/
void
generic_function_1darray_eval2_ind(size_t n, 
                                   struct GenericFunction ** f, 
                                   size_t ind, double * out)
{

    size_t ii;
    for (ii = 0; ii < n; ii++){
        out[ii] = generic_function_1d_eval_ind(f[ii],ind);
    }
}

/********************************************************//**
*   Compute a linear combination of generic functions
*
*   \param[in] n    - number of functions
*   \param[in] ldgf - stride of array to use
*   \param[in] gfa  - array of functions
*   \param[in] ldc  - stride of coefficents
*   \param[in] c    - scaling coefficients
*
*   \return function representing
*   \f$ \sum_{i=1}^n coeff[ldc[i]] * gfa[ldgf[i]] \f$
************************************************************/
struct GenericFunction *
generic_function_lin_comb2(size_t n, size_t ldgf, 
                           struct GenericFunction ** gfa,
                           size_t ldc, const double * c)
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
            
            struct GenericFunction * gf = generic_function_alloc(1,gfa[0]->fc);
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
    Compute the location and value of the maximum, 
    in absolute value, element of a generic function 

    \param[in]     f       - function
    \param[in,out] x       - location of maximum
    \param[in]     optargs - optimization arguments

    \return absolute value of the maximum
************************************************************/
double generic_function_absmax(const struct GenericFunction * f, double * x, void * optargs)
{
    double out = 0.123456789;
    size_t dsize = sizeof(double);
    switch (f->fc){
    case CONSTANT:                                                     break;
    case PIECEWISE:  out = piecewise_poly_absmax(f->f,x,optargs);      break;
    case POLYNOMIAL: out = orth_poly_expansion_absmax(f->f,x,optargs); break;
    case LINELM:     out = lin_elem_exp_absmax(f->f,x,dsize,optargs);  break;
    case RATIONAL:                                                     break;
    case KERNEL:                                                       break;
    }
    return out;
}

/********************************************************//**
    Compute the (generic) location and value of the maximum, 
    in absolute value, element of a generic function 

    \param[in]     f       - function
    \param[in,out] x       - location of maximum
    \param[in]     size    - number of bytes of x
    \param[in]     optargs - optimization arguments

    \return absolute value of the maximum
************************************************************/
double generic_function_absmax_gen(const struct GenericFunction * f, 
                                   void * x, size_t size, void * optargs)
{
    double out = 0.123456789;
    size_t dsize = sizeof(double);
    switch (f->fc){
    case CONSTANT:                                                     break;
    case PIECEWISE:  assert (size == dsize); out = piecewise_poly_absmax(f->f,x,optargs);      break;
    case POLYNOMIAL: assert (size == dsize); out = orth_poly_expansion_absmax(f->f,x,optargs); break;
    case LINELM:     out = lin_elem_exp_absmax(f->f,x,size,optargs);        break;
    case RATIONAL:                                                     break;
    case KERNEL:                                                       break;
    }
    return out;
}

/********************************************************//**
    Compute the index, location and value of the maximum, in absolute value, 
    element of a generic function array

    \param[in]     n   - number of functions
    \param[in]     lda - stride
    \param[in]     a   - array of functions
    \param[in,out] ind - index of maximum
    \param[in,out] x   - location of maximum

    \return maxval - absolute value of the maximum
************************************************************/
double 
generic_function_array_absmax(size_t n, size_t lda, 
                              struct GenericFunction ** a, 
                              size_t * ind,  double * x,
                              void * optargs)
{
    size_t ii = 0;
    *ind = ii;
    //printf("do absmax\n");
    //print_generic_function(a[ii],0,NULL);
    double maxval = generic_function_absmax(a[ii],x,optargs);
    //printf("maxval=%G\n",maxval);
    double tempval, tempx;
    for (ii = 1; ii < n; ii++){
        tempval = generic_function_absmax(a[ii*lda],&tempx,optargs);
        if (tempval > maxval){
            maxval = tempval;
            *x = tempx;
            *ind = ii;
        }
    }
    return maxval;
}

/********************************************************//**
    Compute the index, location and value of the maximum, in absolute value, 
    element of a generic function array (Pivot Based)

    \param[in]     n   - number of functions
    \param[in]     lda - stride
    \param[in]     a   - array of functions
    \param[in,out] piv - pivot

    \return maxval - absolute value of the maximum
************************************************************/
double 
generic_function_array_absmax_piv(size_t n, size_t lda, 
                                  struct GenericFunction ** a, 
                                  struct Pivot * piv,
                                  void * optargs)
{
    size_t ii = 0;
    pivot_set_ind(piv,ii);
    //printf("do absmax\n");
    //print_generic_function(a[ii],0,NULL);
    size_t size = pivot_get_size(piv);
    void * x = pivot_get_loc(piv);
    double maxval = generic_function_absmax_gen(a[ii],x,size,optargs);
    //printf("maxval=%G\n",maxval);
    if (size == sizeof(double)){
        double tempval, tempx;
        for (ii = 1; ii < n; ii++){
            tempval = generic_function_absmax_gen(a[ii*lda],&tempx,size,optargs);
            if (tempval > maxval){
                maxval = tempval;
                *(double *)(x) = tempx;
                pivot_set_ind(piv,ii);
            }
        }
    }
    else if (size == sizeof(size_t)){
        double tempval;
        size_t tempx;
        for (ii = 1; ii < n; ii++){
            tempval = generic_function_absmax_gen(a[ii*lda],&tempx,size,optargs);
            if (tempval > maxval){
                maxval = tempval;
                *(size_t *)(x) = tempx;
                pivot_set_ind(piv,ii);
            }
        }  
    }
    else{
        fprintf(stderr, "Cannot perform generic_function_array_absmax_piv\n");
        fprintf(stderr, "with the specified elements of size %zu\n",size);
        exit(1);
    }
    return maxval;
}

/********************************************************//**
    Scale a generic function

    \param[in]     a  - value with which to scale the functions
    \param[in,out] gf - function to scale
************************************************************/
void generic_function_scale(double a, struct GenericFunction * gf)
{
    switch (gf->fc){
    case CONSTANT:                                       break;
    case PIECEWISE:  piecewise_poly_scale(a,gf->f);      break;
    case POLYNOMIAL: orth_poly_expansion_scale(a,gf->f); break;
    case LINELM:     lin_elem_exp_scale(a,gf->f);        break;
    case RATIONAL:                                       break;
    case KERNEL:                                         break;
    }
}

/********************************************************//**
    Scale a generic function array

    \param[in]     a - value with which to scale the functions
    \param[in,out] gf  - functions to scale
    \param[in]     N - number of functions
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
    \param d [inout] - (rm,l)

************************************************************/
void generic_function_kronh(int left,
                            size_t r, size_t m, size_t n, size_t l, 
                            const double * a,
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

    \param[in] f    -  function to cut
    \param[in] args - function arguments
    \param[in] dim  - dimension along which we obtain the cut
    \param[in] n    - number of fibercuts
    \param[in] val  - values of the input for each fibercut 

    \return fcut - array of struct necessary for computing values in the cut
***************************************************************/
struct FiberCut **
fiber_cut_2darray( double (*f)(double, double, void *), void * args, 
                            size_t dim, size_t n, const double * val)
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

    \param[in] f      -  function to cut
    \param[in] args   - function arguments
    \param[in] totdim - total number of dimensions
    \param[in] dim    - dimension along which we obtain the cut
    \param[in] n      - number of fibercuts
    \param[in] val    - array of values of the inputs for each fibercut 

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
//    printf("vals are \n");
    for (ii = 0; ii < n; ii++){
        fcut[ii] = alloc_fiber_cut(totdim,dim);
        fcut[ii]->f.fnd = f;
        fcut[ii]->args = args;
        fcut[ii]->ftype_flag = 1;
        memmove(fcut[ii]->vals, val[ii], totdim*sizeof(double));
//        dprint(totdim,val[ii]);

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

    \param[in] x     - value at which to evaluate
    \param[in] vfcut - void pointer to fiber_cut structure

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
void print_generic_function(const struct GenericFunction * gf, size_t prec,void * args){

    
    switch (gf->fc){
    case CONSTANT:                                               break;
    case PIECEWISE:  print_piecewise_poly(gf->f,prec,args);      break;
    case POLYNOMIAL: print_orth_poly_expansion(gf->f,prec,args); break;
    case LINELM:     print_lin_elem_exp(gf->f,prec,args,stdout); break;
    case RATIONAL:                                               break;
    case KERNEL:                                                 break;
    }
}


/********************************************************//**
    Create a generic function by approximating a one dimensional function

    \param[in] fc    - function approximation class
    \param[in] f     - wrapped function
    \param[in] aopts - approximation options

    \return gf - generic function
************************************************************/
struct GenericFunction * 
generic_function_approximate1d(enum function_class fc, void * aopts,
                               struct Fwrap * f)
{
    struct GenericFunction * gf = generic_function_alloc(1,fc);
    switch (fc){
    case CONSTANT:                                                       break;
    case PIECEWISE:  gf->f = piecewise_poly_approx1_adapt(aopts,f);      break;
    case POLYNOMIAL: gf->f = orth_poly_expansion_approx_adapt(aopts,f);  break;
    case LINELM:     gf->f = lin_elem_exp_approx(aopts,f);               break;
    case RATIONAL:                                                       break;
    case KERNEL:                                                         break;
    }

    //print_generic_function(gf,0,NULL);
    return gf;
}

/***********************************************************//**
    Generate a set of orthonormal arrays of functions for helping
    generate an orthonormal qmarray 

    \param[in] fc    - function class
    \param[in] st    - function class sub_type
    \param[in] nrows - number of rows
    \param[in] ncols - number of columns
    \param[in] lb    - lower bound on 1d functions
    \param[in] ub    - upper bound on 1d functions

    \note
    - Not super efficient because of copies
***************************************************************/
/* void  */
/* generic_function_array_orth1d_columns(struct GenericFunction ** f, */
/*                                       struct GenericFunction ** funcs, */
/*                                       enum function_class fc, */
/*                                       void * st, size_t nrows, */
/*                                       size_t ncols, double lb, */
/*                                       double ub) */
/* { */
    
/*     struct Interval ob; */
/*     ob.lb = lb; */
/*     ob.ub = ub; */
/*     size_t jj,kk; */
/*     generic_function_array_orth(ncols, fc, st, funcs, &ob); */
/*     struct GenericFunction * zero =  */
/*         generic_function_constant(0.0,fc,st,lb,ub,NULL); */
/*     size_t onnon = 0; */
/*     size_t onorder = 0; */
/*     for (jj = 0; jj < ncols; jj++){ */
/*         f[jj*nrows+onnon] = generic_function_copy(funcs[onorder]); */
/*         for (kk = 0; kk < onnon; kk++){ */
/*             f[jj*nrows+kk] = generic_function_copy(zero); */
/*         } */
/*         for (kk = onnon+1; kk < nrows; kk++){ */
/*             f[jj*nrows+kk] = generic_function_copy(zero); */
/*         } */
/*         onnon = onnon+1; */
/*         if (onnon == nrows){ */
/*             //generic_function_free(funcs[onorder]); */
/*             //funcs[onorder] = NULL; */
/*             onorder = onorder+1; */
/*             onnon = 0; */
/*         } */
/*     } */
/*     generic_function_free(zero); zero = NULL; */

/* } */

/***********************************************************//**
    Generate a set of orthonormal arrays of functions for helping
    generate an orthonormal qmarray of nodal basis functions
    on a grid

    \param[in] f     - array of functions
    \param[in] nrows - number of rows
    \param[in] ncols - number of columns
    \param[in] grid  - nodes


    \note
    - Not super efficient because of copies
***************************************************************/
void
generic_function_array_orth1d_linelm_columns(struct GenericFunction ** f,
                                              size_t nrows,size_t ncols,
                                              struct c3Vector * grid)
{
    (void)(f);
    (void)(nrows);
    (void)(ncols);
    (void)(grid);
    assert (1 == 0);
    /* struct LinElemExp ** le = malloc(grid->size * sizeof(struct LinElemExp *)); */
    /* assert (f != NULL); */
    /* double * zero = calloc_double(grid->size); */
    /* for (size_t ii = 0; ii < grid->size; ii++){ */
    /*     le[ii] = lin_elem_exp_init(grid->size,grid->elem,zero); */
    /* } */
    /* lin_elem_exp_orth_basis(grid->size,le); */
    
    /* size_t onnon = 0; */
    /* size_t onorder = 0; */
    /* for (jj = 0; jj < ncols; jj++){ */
    /*     f[jj*nrows+onnon] = generic_function_copy(funcs[onorder]); */
    /*     for (kk = 0; kk < onnon; kk++){ */
    /*         f[jj*nrows+kk] = generic_function_copy(zero); */
    /*     } */
    /*     for (kk = onnon+1; kk < nrows; kk++){ */
    /*         f[jj*nrows+kk] = generic_function_copy(zero); */
    /*     } */
    /*     onnon = onnon+1; */
    /*     if (onnon == nrows){ */
    /*         onorder = onorder+1; */
    /*         onnon = 0; */
    /*     } */
    /* } */

    /* free(zero); zero = NULL; */
    /* for (size_t ii = 0; ii < grid->size; ii++){ */
    /*     lin_elem_exp_free(le[ii]); le[ii] = NULL; */
    /* } */
    /* le[ii] = NULL; */

}

/*******************************************************//**
    Fill a generic_function array with orthonormal functions 
    of a particular class and sub_type

    \param[in]     n       - number of columns
    \param[in,out] gfarray - array to fill with functions
    \param[in]     fc      - function class
    \param[in]     args    - extra arguments depending on 
                             function_class, sub_type, etc.
************************************************************/
void
generic_function_array_orth(size_t n,
                            struct GenericFunction ** gfarray,
                            enum function_class fc,
                            void * args)
{
    size_t ii;
    /* double lb, ub; */
    struct LinElemExp ** b = NULL;
    switch (fc){
    case CONSTANT: break;
    case PIECEWISE:
        /* printf("generating orthonormal piecewise\n"); */
        for (ii = 0; ii < n; ii++){
            gfarray[ii] = generic_function_alloc(1,fc);
            gfarray[ii]->f = piecewise_poly_genorder(ii,args);
            gfarray[ii]->fargs = NULL;
        }
        break;
    case POLYNOMIAL:
        for (ii = 0; ii < n; ii++){
            /* printf("on ii = %zu, fc=%d\n",ii,fc); */
            gfarray[ii] = generic_function_alloc(1,fc);
            gfarray[ii]->f = orth_poly_expansion_genorder(ii,args);
            gfarray[ii]->fargs = NULL;
        }
        break;
    case LINELM:
        b = malloc(n * sizeof(struct LinElemExp *));
        for (ii = 0 ; ii < n; ii++){
            gfarray[ii] = generic_function_alloc(1,fc);
            b[ii] = NULL;
        }
        lin_elem_exp_orth_basis(n,b,args);
        for (ii = 0; ii < n; ii++){
            gfarray[ii]->f = b[ii];
        }
        free(b); b = NULL;
        break;
    case RATIONAL:
        break;
    case KERNEL:
        break;
    }

}
