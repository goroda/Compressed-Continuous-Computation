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


/** \file funcs_not_standard.c
 * Provides routines for generic functions that may need to be modified by hand when 
 * introducing new types
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
    case PIECEWISE:  gf->f = piecewise_poly_approx1_adapt(aopts,f);      break;
    case POLYNOMIAL: gf->f = orth_poly_expansion_approx_adapt(aopts,f);  break;
    case LINELM:     gf->f = lin_elem_exp_approx(aopts,f);               break;
    case CONSTELM:   gf->f = const_elem_exp_approx(aopts,f);             break;        
    case KERNEL: assert (1 == 0);                                        break;
    }

    return gf;
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

    /* GF_SWITCH_TWOOUT_FRONT(deserialize,fc,ptr,ptr,&((*gf)->f)) */
    /* /\* printf("deserialize generic function %zu, %d\n",dim,fc); *\/ */
    struct PiecewisePoly * pw = NULL;
    struct OrthPolyExpansion * ope = NULL;
    struct LinElemExp * le = NULL;
    struct ConstElemExp * ce = NULL;
    struct KernelExpansion * ke = NULL;
    switch (fc){
    case PIECEWISE:  ptr = deserialize_piecewise_poly(ptr,&pw);       (*gf)->f = pw;  break;
    case POLYNOMIAL: ptr = deserialize_orth_poly_expansion(ptr,&ope); (*gf)->f = ope; break;
    case LINELM:     ptr = deserialize_lin_elem_exp(ptr,&le);         (*gf)->f = le;  break;
    case CONSTELM:   ptr = deserialize_const_elem_exp(ptr,&ce);       (*gf)->f = ce;  break;
    case KERNEL:     ptr = deserialize_kernel_expansion(ptr,&ke);     (*gf)->f = ke;  break;
    }
    return ptr;
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
    case PIECEWISE:  out = piecewise_poly_absmax(f->f,x,optargs);        break;
    case POLYNOMIAL: out = orth_poly_expansion_absmax(f->f,x,optargs);   break;
    case LINELM:     out = lin_elem_exp_absmax(f->f,x,dsize,optargs);    break;
    case CONSTELM:   out = const_elem_exp_absmax(f->f,x,dsize,optargs);  break;        
    case KERNEL:     out = kernel_expansion_absmax(f->f, x, optargs);    break;
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
    /* size_t dsize = sizeof(double); */
    switch (f->fc){
    case PIECEWISE:  out = piecewise_poly_absmax(f->f,x,optargs);        break;
    case POLYNOMIAL: out = orth_poly_expansion_absmax(f->f,x,optargs);   break;
    case LINELM:     out = lin_elem_exp_absmax(f->f,x,size,optargs);     break;
    case CONSTELM:   out = const_elem_exp_absmax(f->f,x,size,optargs);   break;        
    case KERNEL:     out = kernel_expansion_absmax(f->f, x, optargs);    break;
    }
    return out;
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
    struct OrthPolyExpansion ** pp = NULL;
    struct LinElemExp ** b = NULL;
    struct ConstElemExp ** ce = NULL;    
    struct KernelExpansion ** ke = NULL;
    switch (fc){
    case PIECEWISE:
        /* printf("generating orthonormal piecewise\n"); */
        for (ii = 0; ii < n; ii++){
            gfarray[ii] = generic_function_alloc(1,fc);
            gfarray[ii]->f = piecewise_poly_genorder(ii,args);
            gfarray[ii]->fargs = NULL;
        }
        break;
    case POLYNOMIAL:
        pp = malloc(n * sizeof(struct OrthPolyExpansion *));
        for (size_t zz = 0; zz < n; zz++){
            pp[zz] = NULL;
        }
        orth_poly_expansion_orth_basis(n, pp, args);
        
        for (ii = 0; ii < n; ii++){
            /* printf("on ii = %zu, fc=%d\n",ii,fc); */
            gfarray[ii] = generic_function_alloc(1,fc);
            /* gfarray[ii]->f = orth_poly_expansion_genorder(ii,args); */
            gfarray[ii]->f = pp[ii]; /* orth_poly_expansion_genorder(ii,args); */
            gfarray[ii]->fargs = NULL;
        }
        free(pp); pp = NULL;
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
    case CONSTELM:
        ce = malloc(n * sizeof(struct LinElemExp *));
        for (ii = 0 ; ii < n; ii++){
            gfarray[ii] = generic_function_alloc(1,fc);
            ce[ii] = NULL;
        }
        const_elem_exp_orth_basis(n,ce,args);
        for (ii = 0; ii < n; ii++){
            gfarray[ii]->f = ce[ii];
        }
        free(ce); ce = NULL;
        break;        
    case KERNEL:
        /* assert(1==0); */
        ke = malloc(n * sizeof(struct KernelExpansion *));
        for (ii = 0 ; ii < n; ii++){
            gfarray[ii] = generic_function_alloc(1,fc);
            ke[ii] = NULL;
        }
        kernel_expansion_orth_basis(n,ke,args);
        for (ii = 0; ii < n; ii++){
            gfarray[ii]->f = ke[ii];
        }
        free(ke); ke = NULL;
 
        break;
    }
}
