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
#include "functions.h"
#include "futil.h"

/** \struct Regress1DOpts
 * \brief One dimensional regression options
 * \var Regress1DOpts:atype
 * approximation type
 * \var Regress1DOpts:rtype
 * regression problem
 * \var Regress1DOpts:fc
 * function class of the approximation
 * \var Regress1DOpts:reg_param_set
 * indicator of whethe the regularization parameter is set
 * \var Regress1DOpts:lambda
 * regularization parameter
 * \var Regress1DOpts:decay_type
 * decay type (used for regularized RKHS regression)
 * \var Regress1DOpts:coeff_decay_opt
 * parameter specifying decay rate
 * \var Regress1DOpts:N
 * number of training samples
 * \var Regress1DOpts:x
 * location of training samples
 * \var Regress1DOpts:y
 * value of training samples
 * \var Regress1DOpts:aopts
 * approximation options for function class
 * \var Regress1DOpts:nparam
 * number of parameters for parametric regression
 * \var Regress1DOpts:init_param
 * initial parameters
 * \var Regress1DOpts:gf
 * Generic function currently being worked with
 * \var Regress1DOpts:eval
 * Storage locations for evaluation of current guess
 * \var Regress1DOpts:grad
 * Storage location for gradient
 * \var Regress1DOpts:resid
 * Storage location for residual
 */
struct Regress1DOpts
{
    enum approx_type  atype;
    enum regress_type rtype;
    enum function_class fc;

    // Regularization options
    int reg_param_set;
    double lambda;
    enum coeff_decay_type decay_type;
    double coeff_decay_param;

    size_t N;
    const double * x;
    const double * y;

    void * aopts; // approximation options

    // parameteric stuff
    size_t nparam; // for parametric
    const double * init_param;

    // store current generic funciton
    struct GenericFunction * gf;

    // stuff to speed up storage
    double * eval;
    double * grad;
    double * resid;
    
};


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
    Free memory for generic function

    \param[in,out] gf - generic function
************************************************************/
void generic_function_free(struct GenericFunction * gf){
    if (gf != NULL){
        if (gf->f != NULL) {
            GF_SWITCH_NO_OUT(free)
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

/* (Deep) Copy a generic function */
GF_IN_OUT(copy)   

/********************************************************//**
    Copy a generic function to a preallocated generic function

    \param[in]     gf   - generic function
    \param[in,out] gfpa - preallocated function

************************************************************/
void generic_function_copy_pa(const struct GenericFunction * gf, 
                              struct GenericFunction * gfpa)
{
    gfpa->fc = gf->fc;
    void * temp = NULL;
    GF_SWITCH_TEMPOUT(copy)
    gfpa->f = temp;
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
        GF_SWITCH_THREE_FRONT(serialize,gf->fc,NULL,gf->f,&sizef)
        totSize += sizef;
        *totSizeIn = totSize;
        return ptr;
    }
    else{
        ptr = serialize_size_t(ptr, gf->dim);
        ptr = serialize_int(ptr, gf->fc);
        GF_SWITCH_THREEOUT_FRONT(serialize,gf->fc,ptr,ptr,gf->f, NULL)
    }
    return ptr;

}



/********************************************************//**
    Save a generic function in text format

    \param[in] gf     - generic function to save
    \param[in] stream - stream to save it to
    \param[in] prec   - precision with which to save it

************************************************************/
void generic_function_savetxt(const struct GenericFunction * gf,
                              FILE * stream, size_t prec)
{
    assert (gf != NULL);
    fprintf(stream,"%zu ",gf->dim);
    fprintf(stream,"%d ",(int)(gf->fc));
    GF_SWITCH_NO_THREEOUT(savetxt,gf->fc,gf->f,stream,prec)
}

/********************************************************//**
    Load a generic function in text format

    \param[in] stream - stream to save it to

    \return Generic function
************************************************************/
struct GenericFunction *
generic_function_loadtxt(FILE * stream)
{
    size_t dim;
    int num = fscanf(stream,"%zu ",&dim);
    assert (num == 1);
    struct GenericFunction * gf = generic_function_alloc_base(dim);
    int fcint;
    num = fscanf(stream,"%d ",&fcint);
    gf->fc = (enum function_class)fcint;
    assert (num = 1);

    GF_SWITCH_ONEOUT(loadtxt, gf->fc, gf->f, stream)
    return gf;
}

void print_generic_function(const struct GenericFunction * gf, size_t prec,void * args, FILE * fp)
{
    GF_SWITCH_NO_FOUROUT_FRONT(print,gf->fc,gf->f,prec,args,fp)
}

/*******************************************************//**
    Update a linear function

    \param[in] gf     - existing linear function
    \param[in] a      - slope of the function
    \param[in] offset - offset of the function

    \returns 0 if successfull, 1 otherwise                   
    \note 
    Existing function must be linear
***********************************************************/
int
generic_function_linear_update(struct GenericFunction * gf,
                               double a, double offset)
{
    int temp = 0;
    GF_SWITCH_THREEOUT(linear_update,gf->fc,temp,gf->f,a,offset)
    return temp;
}


/********************************************************//**
    Create a generic function through regression of data

    \return gf - generic function
************************************************************/
struct GenericFunction *
generic_function_regress1d(struct Regress1DOpts * opts, struct c3Opt * optimizer, int *info)
{

    struct GenericFunction * func = NULL;
    // perform linear regression to generate the starting point

    // Initialize generic function to this linear function

    double val;
    if (opts->atype == PARAMETRIC){
        double * start = calloc_double(opts->nparam);
        memmove(start,opts->init_param,opts->nparam*sizeof(double));
        if (opts->rtype == LS) {
            c3opt_add_objective(optimizer,param_LSregress_cost,opts);
        }
        else if (opts->rtype == RLS2){
            if (opts->reg_param_set == 0){
                printf("Must set regularization parameter for RLS2 regression\n");
                free(start); start = NULL;
                return NULL;
            }
            c3opt_add_objective(optimizer,param_RLS2regress_cost,opts);
        }
        else if (opts->rtype == RLSD2){
            if (opts->reg_param_set == 0){
                printf("Must set regularization parameter for RLSD2 regression\n");
                free(start); start = NULL;
                return NULL;
            }
            c3opt_add_objective(optimizer,param_RLSD2regress_cost,opts);
        }
        else if (opts->rtype == RLSRKHS){
            if (opts->reg_param_set == 0){
                printf("Must set regularization parameter for RLSRKHS regression\n");
                free(start); start = NULL;
                return NULL;
            }
            else if (opts->decay_type == NONE){
                printf("Must set decay type for parameter for RLSRKHS regression\n");
                free(start); start = NULL;
                return NULL;
            }
            c3opt_add_objective(optimizer,param_RLSRKHSregress_cost,opts);
        }
        else if (opts->rtype == RLS1){
            printf("L1 regularization not yet implemented\n");
            free(start); start = NULL;
            return NULL;
            /* c3opt_add_objective(optimizer,param_RLS1regress_cost,opts); */
        }
        else{
            printf("Parameteric regression type %d is not recognized\n",opts->rtype);
            free(start); start = NULL;
            return NULL;
        }
        

        *info = c3opt_minimize(optimizer,start,&val);
        /* if (*info > -1){ */
            func = generic_function_create_with_params(opts->fc,opts->aopts,opts->nparam,start);
        /* } */
        free(start); start = NULL;
    }
    else if (opts->atype == NONPARAMETRIC){
        printf("Non-parametric regression is not yet implemented\n");
        return NULL;
    }
    else{
        printf("Regression of type %d is not recognized\n",opts->atype);
        return NULL;
    }

    return func;
}

/********************************************************//**
    Create a generic function with particular parameters

    \param[in] fc    - function class
    \param[in] aopts - approximation options
    \param[in] dim   - number of parameters
    \param[in] param - parameter values to set

    \return generic function
************************************************************/
struct GenericFunction *
generic_function_create_with_params(enum function_class fc, void * aopts, size_t dim,
                                    const double * param)
{

    struct GenericFunction * gf = generic_function_alloc(1,fc);
    GF_SWITCH_THREEOUT(create_with_params, fc, gf->f, aopts, dim, param);
    return gf;
}

/********************************************************//**
    Return a zero function

    \param[in] fc           - function class
    \param[in] aopts        - extra arguments depending on function_class, sub_type, etc.
    \param[in] force_nparam - if == 1 then approximation will have the number of parameters
                                      defined by *get_nparams, for each approximation type
                              if == 0 then it may be more compressed

    \return gf - zero function
************************************************************/
struct GenericFunction * 
generic_function_zero(enum function_class fc, void * aopts, int force_nparam)
{   
    struct GenericFunction * gf = generic_function_alloc(1,fc);
    if (force_nparam == 0){ GF_SWITCH_TWOOUT(constant, fc, gf->f, 0, aopts) }
    else { GF_SWITCH_TWOOUT(zero, fc, gf->f, aopts, 1) }
    return gf;
}


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
    GF_SWITCH_TWOOUT(constant, fc, gf->f, a, aopts)
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

    \note 
    For kernel, this is only approximate
***********************************************************/
struct GenericFunction * 
generic_function_linear(double a, double offset,
                        enum function_class fc, void * aopts)
{   
    struct GenericFunction * gf = generic_function_alloc(1,fc);
    GF_SWITCH_THREEOUT(linear, fc, gf->f, a, offset, aopts)
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
    GF_SWITCH_THREEOUT(quadratic, fc, gf->f, a, offset, aopts);
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

struct GenericFunction *
generic_function_onezero(enum function_class fc, double one, size_t nz,
                         double * zeros, double lb, double ub)
{
    assert ((fc == LINELM) || (fc == CONSTELM));

    struct GenericFunction * f = 
        generic_function_alloc(1, fc);


    if (fc == LINELM){
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
    }
    else if (fc == CONSTELM){
        struct ConstElemExp * lexp = const_elem_exp_alloc();
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
    }
    return f;
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
*   Compute a linear combination of generic functions
*
*   \param[in] n    - number of functions
*   \param[in] ldgf - stride of array to use
*   \param[in] gfa  - array of functions
*   \param[in] ldc  - stride of coefficents
*   \param[in] c    - scaling coefficients
*
*   \return function representing
*   \f$ \sum_{i=1}^n coeff[ldc[i]] * gfa[ldgf[i]] \f$
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


/* Take the derivative of a generic function */
GF_IN_OUT(deriv)
GF_IN_OUT(dderiv)
GF_IN_OUT(dderiv_periodic) 


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
    /* out = generic_function_alloc(f->dim,LINELM); */
    out = generic_function_alloc(f->dim,f->fc);    
    out->fargs = NULL;
    double * fvals = calloc_double(N);
    for (size_t ii = 0; ii < N; ii++){
        fvals[ii] = generic_function_1d_eval(f,x[ii]);
    }
    if (f->fc == LINELM){
        out->f = lin_elem_exp_init(N,x,fvals);
    }
    else if (f->fc == CONSTELM){
        out->f = const_elem_exp_init(N,x,fvals);        
    }
    else{
        fprintf(stderr,"Cannot create nodal function of this type\n");
        exit(1);
    }
    free(fvals); fvals = NULL;

    return out;
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

GF_IN_GENOUT(get_num_params, size_t, 0)        // Get the number of parameters describing the generic function
GF_IN_GENOUT(get_lb, double, -0.123456789)     // Get the lower bound of the input space
GF_IN_GENOUT(get_ub, double, 0.123456789)     // Get the lower bound of the input space

/********************************************************//**
*   Get the function class
************************************************************/
enum function_class generic_function_get_fc(const struct GenericFunction * f)
{
    assert (f != NULL);
    return f->fc;
}

/***********************************************************//**
    Determine whether kristoffel weighting is active

    \param[in] gf - generic function

    \return 1 if active, 0 otherwise
***************************************************************/
int generic_function_is_kristoffel_active(const struct GenericFunction * gf)
{
    if (gf->fc != POLYNOMIAL){
        return 0;
    }
    else{
        struct OrthPolyExpansion * ope = gf->f;
        return ope->kristoffel_eval;
    }
}

void generic_function_activate_kristoffel(struct GenericFunction * gf)
{
    if (gf->fc != POLYNOMIAL){
        fprintf(stderr,"Cannot activate kristoffel for non polynomial basis\n");
        exit(1);
    }
    else{
        struct OrthPolyExpansion * ope = gf->f;
        ope->kristoffel_eval = 1;
    }
}

void generic_function_deactivate_kristoffel(struct GenericFunction * gf)
{
    if (gf->fc == POLYNOMIAL){
        struct OrthPolyExpansion * ope = gf->f;
        ope->kristoffel_eval = 0;
    }
}


/***********************************************************//**
    Get the kristoffel normalization factor                                                            

    \param[in] gf - generic function
    \param[in] x  - location at which to obtain normalization

    \return normalization factor
***************************************************************/
double generic_function_get_kristoffel_weight(const struct GenericFunction * gf,
                                              double x)
{
    if (gf->fc != POLYNOMIAL){
        fprintf(stderr, "Cannot get the kristoffel weight of a function that is not a polynomial\n");
        exit(1);
    }
    else{
        double weight = orth_poly_expansion_get_kristoffel_weight(gf->f,x);
        return weight;
    }
}

/********************************************************//**
    Get the parameters of generic function

    \param[in] gf         - generic function
    \param[in,out] params - location to write parameters

    \returns number of parameters
************************************************************/
size_t generic_function_get_params(const struct GenericFunction * gf, double * params)
{

    assert (gf != NULL);
    size_t nparam = 0;
    GF_SWITCH_TWOOUT(get_params, gf->fc, nparam, gf->f, params)
    return nparam;
}

/********************************************************//**
    Get the parameters of generic function

    \param[in] gf         - generic function
    \param[in,out] nparam - location to write parameters

    \returns reference to parameters
************************************************************/
double * generic_function_get_params_ref(const struct GenericFunction * gf, size_t * nparam)
{

    assert (gf != NULL);
    double * params = NULL;
    GF_SWITCH_TWOOUT(get_params_ref, gf->fc, params, gf->f, nparam)

    return params;
}


/********************************************************//**
    Update a generic function with particular parameters

    \param[in] f     - function to update
    \param[in] dim   - number of parameters
    \param[in] param - parameter values to set

    \returns 0 if successfull, 1 otherwise
************************************************************/
int 
generic_function_update_params(struct GenericFunction * f, size_t dim,
                               const double * param)
{

    for (size_t ii = 0; ii < dim; ii++){
        if (isnan(param[ii])){
            fprintf(stderr,"Updating generic functions with params that are NaN\n");
            exit(1);
        }
        else if (isinf(param[ii])){
            fprintf(stderr,"Updating generic functions with params that are inf\n");
            exit(1);
        }
    }

    int res = 0;
    GF_SWITCH_THREEOUT(update_params, f->fc, res, f->f, dim, param)
    return res;
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
    assert ( (*gf)->fc == POLYNOMIAL);
    ope = (*gf)->f;
    orth_poly_expansion_roundt(&ope,thresh);
}

static struct GenericFunction * 
generic_function_onezero2(
    enum function_class fc,
    size_t nzeros,
    double * zero_locations,
    void * opts
    )
{
    struct GenericFunction * gf = generic_function_alloc(1,fc);
    if (fc == LINELM){
        gf->f = lin_elem_exp_onezero(nzeros, zero_locations, opts);
    }
    else if (fc == CONSTELM){
        gf->f = const_elem_exp_onezero(nzeros, zero_locations, opts);        
    }
    else{
        fprintf(stderr,"Cannot create a onezero generic function for non-nodal basis\n");
        exit(1);
    }
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

/********************************************************//**
*   Flip the sign of a generic function f(x) to -f(x)
*
*   \param[in,out] gf - number of functions
************************************************************/
void generic_function_flip_sign(struct GenericFunction * gf)
{
    GF_SWITCH_NO_OUT(flip_sign)
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
    GF_SWITCH_THREEOUT(axpy, x->fc, out, a, x->f, y->f);
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
    Scale a generic function

    \param[in]     a  - value with which to scale the functions
    \param[in,out] gf - function to scale
************************************************************/
void generic_function_scale(double a, struct GenericFunction * gf)
{
    GF_SWITCH_NO_ONEOUT(scale, gf->fc, a, gf->f)
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
     
    GF_SWITCH_TWOOUT(eval, f->fc, out, f->f, x)

        if (isnan(out)){
            fprintf(stderr,"Warning, evaluation of generic_function is nan\n");
            exit(1);
        }
        else if (isinf(out)){
            fprintf(stderr,"Warning, evaluation of generic_function is inf\n");
            exit(1);
        }
    return out;
}

/********************************************************//**
    Evaluate the derivative of a generic function

    \param[in] gf - generic function
    \param[in] x  - location at which to evaluate

    \return value of the derivative
************************************************************/
double generic_function_deriv_eval(const struct GenericFunction * gf, double x)
{
    double out = 0.1234567890;
    GF_SWITCH_TWOOUT(deriv_eval, gf->fc, out, gf->f, x);
    return out;
}


/********************************************************//**
*   Evaluate a generic function at multiple locations
*
*   \param[in]     f    - function
*   \param[in]     N    - number of evaluations
*   \param[in]     x    - location at which to evaluate
*   \param[in]     incx - increment of x
*   \param[in,out] y    - allocated space for evaluations
*   \param[in]     incy - increment of y
************************************************************/
void generic_function_1d_evalN(const struct GenericFunction * f, size_t N,
                               const double * x, size_t incx, double * y, size_t incy)
{
     assert (f != NULL);
     assert (f->f != NULL);
     GF_SWITCH_SIX(evalN,f->fc,f->f,N,x,incx,y,incy)
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
    double out = 0.1234567890;
    if (f->fc == LINELM){
        out = lin_elem_exp_get_nodal_val(f->f,ind);
    }
    else if (f->fc == CONSTELM){
        out = const_elem_exp_get_nodal_val(f->f,ind);
    }
    else{
        assert (1 == 0);
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
*   Evaluate a generic function consisting of nodal
*   basis functions at some node
*
*   \param[in] f    - function
*   \param[in] x    - location at which to Evaluate
*   \param[in] size - byte size of location (sizeof(double) or (sizeof(size_t)))
*
*   \return evaluation
************************************************************/
static double generic_function_1d_eval_gen(const struct GenericFunction * f,
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
        int res = orth_poly_expansion_arr_eval(n,parr,x,out);
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
*   Evaluate an array of functions at an array of points
*
*   \param[in]     n    - number of functions
*   \param[in]     f    - function
*   \param[in]     N    - number of evaluations
*   \param[in]     x    - location at which to evaluate
*   \param[in]     incx - increment of x
*   \param[in,out] y    - allocated space for evaluations
*   \param[in]     incy - increment of y*
*
*   \note Currently just calls the single evaluation code
*         Note sure if this is optimal, cache-wise
*************************************************************/
void
generic_function_1darray_eval2N(size_t n, 
                                struct GenericFunction ** f,
                                size_t N, const double * x, size_t incx,
                                double * y, size_t incy)
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
        /* printf("generic function, kristoffel_active = %d\n",generic_function_is_kristoffel_active(f[0])); */
        int res = orth_poly_expansion_arr_evalN(n,parr,N,x,incx,y,incy);
        /* for (size_t ii = 0; ii < N*n; ii++){ */
        /*     printf("y[%zu] = %G\n",ii,y[ii]); */
        /* } */
        if (res == 1){ //something when wrong
            size_t ii;
            for (ii = 0; ii < n; ii++){
                generic_function_1darray_eval2(n,f,x[ii*incx],y+ii*incy);
            }
        }
    }
    else{
        for (size_t ii = 0; ii < N; ii++){
            generic_function_1darray_eval2(n,f,x[ii*incx],y+ii*incy);
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
    Take a gradient with respect to function parameters

    \param[in]     gf   - generic function
    \param[in]     nx   - number of x values
    \param[in]     x    - x values
    \param[in,out] grad - gradient (N,nx)

    \return  0 - success, 1 -failure
************************************************************/
int generic_function_param_grad_eval(const struct GenericFunction * gf,
                                     size_t nx, const double * x,
                                     double * grad)
{

    enum function_class fc = generic_function_get_fc(gf);
    int res = 1;
    GF_SWITCH_FOUROUT(param_grad_eval, fc, res, gf->f, nx, x, grad)
    assert (res == 0);
    return res;
}


/********************************************************//**
    Take a gradient with respect to function parameters

    \param[in]     gf   - generic function
    \param[in]     x    - x values
    \param[in,out] grad - gradient (N)

    \return  evaluation
************************************************************/
double generic_function_param_grad_eval2(const struct GenericFunction * gf,
                                         double x,double * grad)
                                        
{

    enum function_class fc = generic_function_get_fc(gf);
    double ret = 0.1234;    
    GF_SWITCH_THREEOUT(param_grad_eval2, fc, ret, gf->f, x, grad)
    return ret;
}


GF_IN_GENOUT(integrate, double, 0.0)           // Compute an integral
GF_IN_GENOUT(integrate_weighted, double, 0.0)  // Take the derivative of a generic function

/********************************************************//**
*   Compute norm of a generic function
*
*   \param[in] f  - generic function
*
*   \return out - norm
************************************************************/
double generic_function_norm(const struct GenericFunction * f)
{
    double out = generic_function_inner(f,f);

    if (out < 0.0){
        fprintf(stderr, "Norm of a function cannot be negative %G\n",out);
        exit(1);
        /* generic_function_scale(0.0, f); */
        /* out = 0.0; */
    }
    //assert (out > -1e-15);
    return sqrt(out);
}


/********************************************************//**
*   Compute the weighted inner product between two generic functions
*
*   \param[in] a  - generic function
*   \param[in] b  - generic function
*
*   \return out -  int a(x) b(x) w(x) dx 
************************************************************/
double generic_function_inner_weighted(const struct GenericFunction * a, 
                                       const struct GenericFunction * b)
{
    assert(a->fc == POLYNOMIAL);
    assert(b->fc == POLYNOMIAL);
    double out = orth_poly_expansion_inner_w(a->f,b->f);
       
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
     for (size_t ii = 0; ii < n; ii++){
         val += generic_function_inner(a[ii*lda], b[ii*ldb]);
     }
     return val;
 }

/********************************************************//**
*   Compute the sum of the (weighted) inner products between
*   two arrays of generic functions
*
*   \param[in] n   - number of inner products
*   \param[in] lda - stride of functions to use in a
*   \param[in] a   - first array of generic functions
*   \param[in] ldb - stride of functions to use in b
*   \param[in] b   - second array of generic functions
*
*   \return val - sum_{i=1^N} int a[ii*lda](x) b[ii*ldb](x) w(x) dx
************************************************************/
double generic_function_inner_weighted_sum(size_t n, size_t lda, 
                                           struct GenericFunction ** a, 
                                           size_t ldb, 
                                           struct GenericFunction ** b)
{
     double val = 0.0;
     size_t ii;
     for (ii = 0; ii < n; ii++){
         val += generic_function_inner_weighted(a[ii*lda], b[ii*ldb]);
     }
     return val;
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
   Compute the integral of a generic function

   \param[in] f - generic function
 
   \return out - integral

   \note Computes \f$ \int f(x) w(x) dx\f$ for every univariate function
   in the qmarray
   
   w(x) depends on underlying parameterization
   for example, it is 1/2 for legendre (and default for others),
   gauss for hermite,etc
************************************************************/
double generic_function_integral_weighted(
    const struct GenericFunction * f){
     
    assert (f != NULL);
    assert (f->fc == POLYNOMIAL);
     
    double out = orth_poly_expansion_integrate_weighted(f->f);
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
        out[ii] = generic_function_integrate(a[ii*lda]);
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
***********************************************************/
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
    Compute the index, location and value of the maximum, in absolute value, 
    element of a generic function array

    \param[in]     n       - number of functions
    \param[in]     lda     - stride
    \param[in]     a       - array of functions
    \param[in,out] ind     - index of maximum
    \param[in,out] x       - location of maximum
    \param[in]     optargs - optimization arguments

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

    \param[in]     n       - number of functions
    \param[in]     lda     - stride
    \param[in]     a       - array of functions
    \param[in,out] piv     - pivot
    \param[in]     optargs - optimization arguments

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
    fcut->fpoint.f2d = f;
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
        fcut[ii]->fpoint.f2d = f;
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
        fcut[ii]->fpoint.fnd = f;
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
        val = fcut->fpoint.f2d(x, fcut->vals[1], fcut->args);
    }
    else{
        val = fcut->fpoint.f2d(fcut->vals[0], x, fcut->args);
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
    val = fcut->fpoint.fnd(fcut->vals, fcut->args);
    return val;
}


/////////////////////////////////////////////////////////
// Utilities



/***********************************************************
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




//////////////////////////////////////////////////
//////////////////////////////////////////////////
//////////////////////////////////////////////////
/////               Regression          //////////
//////////////////////////////////////////////////
//////////////////////////////////////////////////
//////////////////////////////////////////////////


/********************************************************//**
    Create a regression options

    \param[in] atype - approximation type
    \param[in] rtype - regression problem type
    \param[in] N     - number of training samples
    \param[in] x     - location of training samples
    \param[in] y     - values at training samples

    \return opts     - regression options
************************************************************/
struct Regress1DOpts *
regress_1d_opts_create(enum approx_type atype, enum regress_type rtype,
                       size_t N, const double * x, const double * y)
{
    struct Regress1DOpts * opts = malloc(sizeof(struct Regress1DOpts));
    if (opts == NULL){
        fprintf(stderr, "Error allocating regression options\n");
        exit(1);
    }
    opts->atype = atype;
    opts->rtype = rtype;
    opts->N = N;
    opts->x = x;
    opts->y = y;
    
    opts->aopts = NULL;

    opts->nparam = 0;
    opts->init_param = NULL;

    opts->gf = NULL;
    
    opts->eval  = calloc_double(N);
    opts->grad  = NULL;
    opts->resid = calloc_double(N);

    // regularization options
    opts->reg_param_set     = 0;
    opts->lambda            = 0.0;
    opts->decay_type        = NONE;
    opts->coeff_decay_param = 1.0;
    
    return opts;
}

/********************************************************//**
    Destroy regression options
************************************************************/
void regress_1d_opts_destroy(struct Regress1DOpts * opts)
{
    if (opts != NULL){
        generic_function_free(opts->gf); opts->gf    = NULL;
        free(opts->eval);                opts->eval  = NULL;
        free(opts->grad);                opts->grad  = NULL;
        free(opts->resid);               opts->resid = NULL; 
        free(opts);                      opts        = NULL;
    }
}

/********************************************************//**
    Add a parametric form to learn

    \param[in] opts  - regression options structure
    \param[in] fc    - regression problem type
    \param[in] aopts - parametric approximation options
************************************************************/
void regress_1d_opts_set_parametric_form(
    struct Regress1DOpts * opts, enum function_class fc, void * aopts)
{
    assert(opts != NULL);
    assert(aopts != NULL);

    opts->fc = fc;
    opts->aopts = aopts;

    GF_OPTS_SWITCH_ONEOUT(get_nparams, fc, opts->nparam, aopts)    
    opts->grad = calloc_double(opts->nparam);
}

/********************************************************//**
    Add starting parameters for optimization   
************************************************************/
void regress_1d_opts_set_initial_parameters(
    struct Regress1DOpts * opts, const double * param)
{
    assert (opts != NULL);
    opts->init_param = param;
    
    opts->gf = generic_function_create_with_params(opts->fc,opts->aopts,opts->nparam,param);
}

/********************************************************//**
    Set regularization penalty
************************************************************/
void regress_1d_opts_set_regularization_penalty(
    struct Regress1DOpts * opts, double lambda)
{
    assert (opts != NULL);
    opts->reg_param_set = 1;
    opts->lambda = lambda;
}

/********************************************************//**
    Set RKHS decay
************************************************************/
void regress_1d_opts_set_RKHS_decay_rate(
    struct Regress1DOpts * opts, enum coeff_decay_type decay_type, double lambda)
{
    assert (opts != NULL);
    opts->decay_type = decay_type;

    if (decay_type == ALGEBRAIC){
        if ((lambda < 1e-15) || (lambda > 1)){
            fprintf(stderr,"For algebraic decay of RKHS must specify decay rate in (0,1)\n");
            fprintf(stderr,"\t Currently specified as %G\n",lambda);
            exit(1);
        }
        else{
            opts->coeff_decay_param = lambda;
        }
    }
    else if (decay_type == EXPONENTIAL){
        if (lambda < 0){
            fprintf(stderr,"For exponential decay of RKHS must specify decay rate > 0\n");
            fprintf(stderr,"\t Currently specified as %G\n",lambda);
            exit(1);
        }
        else{
            opts->coeff_decay_param = lambda;
        }
    }
    else{
        fprintf(stderr,"Do not recognized RKHS decay type %d\n",decay_type);
        exit(1);
    }
}









/********************************************************//**
    Take a gradient of the squared norm of a generic function
    with respect to its parameters, and add a scaled version
    of this gradient to *grad*

    \param[in]     gf    - generic function
    \param[in]     scale - scaling for additional gradient
    \param[in,out] grad  - gradient, on output adds scale * new_grad

    \return  0 - success, 1 -failure
************************************************************/
int
generic_function_squared_norm_param_grad(const struct GenericFunction * gf,
                                         double scale, double * grad)
{

    enum function_class fc = generic_function_get_fc(gf);
    int res = 1;
    GF_SWITCH_THREEOUT(squared_norm_param_grad, fc, res, gf->f, scale, grad)
    return res;
}

/********************************************************//**
    Norm in the RKHS (instead of L2)

    \param[in]     gf          - generic function
    \param[in]     decay_type  - type of decay
    \param[in]     decay_param - parameter of decay

    \return  0 - success, 1 -failure
************************************************************/
double
generic_function_rkhs_squared_norm(const struct GenericFunction * gf,
                                   enum coeff_decay_type decay_type,
                                   double decay_param)
{

    enum function_class fc = generic_function_get_fc(gf);
    assert (fc == POLYNOMIAL);
    double out = orth_poly_expansion_rkhs_squared_norm(gf->f,decay_type,decay_param);

    return out;
}


/********************************************************//**
    Take a gradient of the norm in the RKHS (instead of L2)

    \param[in]     gf          - generic function
    \param[in]     scale       - scaling for additional gradient
    \param[in]     decay_type  - type of decay
    \param[in]     decay_param - parameter of decay
    \param[in,out] grad        - gradient, on output adds scale * new_grad

    \return  0 - success, 1 -failure
************************************************************/
int
generic_function_rkhs_squared_norm_param_grad(const struct GenericFunction * gf,
                                         double scale, enum coeff_decay_type decay_type,
                                         double decay_param, double * grad)
{

    enum function_class fc = generic_function_get_fc(gf);
    assert (fc == POLYNOMIAL);
    int res = orth_poly_expansion_rkhs_squared_norm_param_grad(
        gf->f,scale,decay_type,decay_param,grad);


    return res;
}




/********************************************************//**
    LS regression objective function
************************************************************/
double param_LSregress_cost(size_t dim, const double * param, double * grad, void * arg)
{

    struct Regress1DOpts * opts = arg;

    assert (opts->nparam == dim);
    assert (opts->gf != NULL);
    // update function
    /* printf("update param\n"); */
    /* printf("\t old = "); */
    /* print_generic_function(opts->gf,0,NULL); */
    /* printf("\t param = "); dprint(dim,param); */
    generic_function_update_params(opts->gf,dim,param);

    /* printf("evaluate\n"); */
    for (size_t ii = 0; ii < opts->N; ii++){
        opts->eval[ii] = generic_function_1d_eval(opts->gf,opts->x[ii]);
    }

    /* printf("compute resid\n"); */
    double out = 0.0;
    for (size_t ii = 0; ii < opts->N; ii++){
        opts->resid[ii] = opts->y[ii]-opts->eval[ii];
        out += opts->resid[ii] * opts->resid[ii];
    }
    out *= 0.5;
    
    if (grad != NULL){
        /* printf("grad is not null!\n"); */
        for (size_t ii = 0; ii < dim; ii++){
            grad[ii] = 0.0;
        }
        for (size_t jj = 0; jj < opts->N; jj++){
            int res = generic_function_param_grad_eval(opts->gf,1,
                                                       opts->x+jj,
                                                       opts->grad);
            assert (res == 0);
            for (size_t ii = 0; ii < dim; ii++){
                grad[ii] += opts->resid[jj] * (-1.0)*opts->grad[ii];
            }
        }
        /* printf("done\n"); */
    }

    return out;
}

/********************************************************//**
    Ridge regression
************************************************************/
double param_RLS2regress_cost(size_t dim, const double * param, double * grad, void * arg)
{

    struct Regress1DOpts * opts = arg;
    
    // first part (recall this function updates parameters already!)
    double ls_portion = param_LSregress_cost(dim,param,grad,arg);

    // second part
    double regularization  = generic_function_inner(opts->gf,opts->gf);

    double out = ls_portion + 0.5*opts->lambda * regularization;
    
    if (grad != NULL){
        int res = generic_function_squared_norm_param_grad(opts->gf,0.5*opts->lambda,grad);
        assert (res == 0);
    }

    return out;
}

/********************************************************//**
    Ridge regression penalizing second derivative
************************************************************/
double param_RLSD2regress_cost(size_t dim, const double * param, double * grad, void * arg)
{

    struct Regress1DOpts * opts = arg;
    
    // first part (recall this function updates parameters already!)
    double ls_portion = param_LSregress_cost(dim,param,grad,arg);

    // second part
    struct GenericFunction * gf1 = generic_function_deriv(opts->gf);
    struct GenericFunction * gf2 = generic_function_deriv(gf1);
    double regularization = generic_function_inner(gf2,gf2);

    double out = ls_portion + 0.5*opts->lambda * regularization;
    
    if (grad != NULL){
        int res = generic_function_squared_norm_param_grad(gf2,0.5*opts->lambda,grad);
        assert (res == 0);
    }

    generic_function_free(gf1); gf1 = NULL;
    generic_function_free(gf2); gf2 = NULL;
    return out;
}

/********************************************************//**
    Ridge regression with an RKHS penalty
************************************************************/
double param_RLSRKHSregress_cost(size_t dim, const double * param, double * grad, void * arg)
{

    struct Regress1DOpts * opts = arg;
    
    // first part (recall this function updates parameters already!)
    double ls_portion = param_LSregress_cost(dim,param,grad,arg);

    // second part
    double regularization =
        generic_function_rkhs_squared_norm(opts->gf,
                                           opts->decay_type,
                                           opts->coeff_decay_param);

    double out = ls_portion + 0.5*opts->lambda * regularization;
    
    if (grad != NULL){
        int res = generic_function_rkhs_squared_norm_param_grad(opts->gf,opts->lambda,
                                                                opts->decay_type,
                                                                opts->coeff_decay_param,grad);
        assert (res == 0);
    }

    return out;
}


