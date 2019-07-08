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


/** \file funcs_mixed.c
 * Provides basic routines for generic function operations that may be with mixed types
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "stringmanip.h"
#include "functions.h"
#include "futil.h"


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
         assert (a->fc != CONSTELM);
         assert (b->fc != CONSTELM);
         assert (a->fc != KERNEL);
         assert (b->fc != KERNEL);
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

         out = piecewise_poly_inner(ap, bp);
     }
     else{
         GF_SWITCH_TWOOUT(inner, fc, out, a->f, b->f)         
     }


     if (apalloc == 1){ free(ap); }
     if (bpalloc == 1){ free(bp); }

     return out;
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
generic_function_prod(struct GenericFunction * a, struct GenericFunction * b)
{

    assert ( a != NULL);
    assert ( b != NULL);
    enum function_class fc = a->fc;
    struct GenericFunction * out = generic_function_alloc(a->dim, fc);
    out->fargs = a->fargs;

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

        out->f = piecewise_poly_prod(ap, bp);
    }
    else{
        GF_SWITCH_TWOOUT(prod, fc, out->f, a->f, b->f);
    }
    
    if (apalloc == 1){  free(ap); }
    if (bpalloc == 1){  free(bp); }
    
    return out;
}

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
    if (x == NULL){
        assert ( y != NULL);
        out = generic_function_copy(y);
        generic_function_scale(b, out);
    }
    else if (y == NULL){
        assert ( x != NULL );
        out = generic_function_copy(x);
        generic_function_scale(a, out);
    }
    else {
        /* printf("in the else!\n"); */
        if (x->fc == y->fc){
            out = generic_function_copy(x);
            generic_function_scale(a, out);
            generic_function_axpy(b, y, out);
        }
        else if (x->fc != y->fc){
            if ((x->fc == LINELM) || (y->fc == LINELM)){
                fprintf(stderr,
                        "Can't add linear elements with other stuff\n");
                exit(1);
            }
            if ((x->fc == CONSTELM) || (y->fc == CONSTELM)){
                fprintf(stderr,
                        "Can't add piecewise constant elements with other stuff\n");
                exit(1);
            }
            else if ((x->fc == KERNEL) || (y->fc == KERNEL)){
                fprintf(stderr,
                        "Can't add kernel expansions with other stuff\n");
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
            out = generic_function_alloc(x->dim, PIECEWISE);
            out->f = piecewise_poly_daxpby(a, ap, b, bp);
            out->fargs = NULL;
            if (apalloc == 1){ free(ap); };
            if (bpalloc == 1){ free(bp); };
        }
    }

    //printf("in there!|n");
    return out;
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
        else if (fc == CONSTELM){
            assert ((*z)->f == NULL);
            (*z)->fc = CONSTELM;
            (*z)->f = const_elem_exp_copy(y->f);
            const_elem_exp_scale(b,(*z)->f);
            const_elem_exp_axpy(a,x->f,(*z)->f);
            (*z)->fargs = NULL;            
        }
        else{
            generic_function_free(*z); (*z) = NULL;
            *z = generic_function_daxpby(a,x,b,y);
        }
    }
}
