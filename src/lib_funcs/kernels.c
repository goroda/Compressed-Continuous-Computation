// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016, Sandia Corporation

// This file is part of the Compressed Continuous Computation (C3) Library
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



/** \file kernels.c
 * Provides routines for manipulating kernels
*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <assert.h>

//#define ZEROTHRESH 1e-20
#define ZEROTHRESH  1e0 * DBL_EPSILON
//#define ZEROTHRESH 0.0
//#define ZEROTHRESH  1e2 * DBL_EPSILON
//#define ZEROTHRESH  1e-12

#include "stringmanip.h"
#include "array.h"
#include "polynomials.h"
#include "hpoly.h"
#include "lib_quadrature.h"
#include "linalg.h"
#include "legtens.h"


/********************************************************//**
*   Evaluate a Gaussian kernel at *x*
*
*   \return  scale * exp( -(x-center)^2 / width_squared)
*************************************************************/
double gauss_kernel_eval(double scale, double width_squared, double center, double x)
{
    double dx = x-center;
    
    double inner =  - dx * dx / width_squared;
    return scale * exp (inner);
}

/********************************************************//**
*   Evaluate the derivative of a Gaussian kernel at *x*
*
*   \return  scale * exp( -(x-center)^2 / width_squared) * (-(2(x-center)/width_squared))
*************************************************************/
double gauss_kernel_deriv(double scale, double width_squared, double center, double x)
{

    double dx = x - center;
    double dx_over_width = - dx / width_squared;
    double inner = dx * dx_over_width;
    
    double dinner = 2.0 * dx_over_width;

    return scale * exp(inner) * dinner;
}

enum KernelType {KernGauss, KernNone};
struct Kernel
{
    enum KernelType type;
    size_t nparams;
    double * params;
};


/********************************************************//**
*   Allocate space for the kernel
*************************************************************/
struct Kernel * kernel_alloc(size_t nparams)
{
    struct Kernel * kern = malloc(sizeof(struct Kernel ));
    if (kern == NULL){
        fprintf(stderr,"Failure to allocate struct Kernel\n");
        exit(1);
    }

    kern->type = KernNone;
    kern->nparams = nparams;
    kern->params = calloc_double(nparams);
    return kern;
}

/********************************************************//**
*   Serialize a kernel
*
*   \param[in] ser       - location to which to serialize
*   \param[in] kern      - kernel
*   \param[in] totSizeIn - if not null then only return total size 
*                          of array without serialization! if NULL then serialiaze
*
*   \return ptr : pointer to end of serialization
*************************************************************/
unsigned char *
serialize_kernel(unsigned char * ser, struct Kernel * kern, size_t * totSizeIn)
{
    // type + nparams + (params+size)
    size_t totsize = sizeof(int) + sizeof(size_t) + kern->nparams * sizeof(double) + sizeof(size_t);
    if (totSizeIn != NULL){
        *totSizeIn = totsize;
        return ser;
    }
    
    unsigned char * ptr = serialize_int(ser,kern->type);
    ptr = serialize_size_t(ptr,kern->nparams);
    ptr = serialize_doublep(ptr,kern->params,kern->nparams);
    return ptr;
}

/********************************************************//**
*   Deserialize a kernel
*
*   \param[in]     ser  - input string
*   \param[in,out] kern - kernel
*
*   \return ptr - ser + number of bytes of kernel
*************************************************************/
unsigned char * 
deserialize_kernel(
    unsigned char * ser, 
    struct Kernel ** kern)
{

    int type;
    size_t nparam;
    unsigned char * ptr = ser;
    ptr = deserialize_int(ptr,&type);
    ptr = deserialize_size_t(ptr,&nparam);


    *kern = kernel_alloc(nparam);
    (*kern)->type = type;
    free((*kern)->params); (*kern)->params = NULL;
    ptr = deserialize_doublep(ptr,&((*kern)->params),&((*kern)->nparams));

    return ptr;
    
}


/********************************************************//**
*   Copy a kernel
*************************************************************/
struct Kernel * kernel_copy(struct Kernel * kern)
{
    if (kern == NULL){
        return NULL;
    }
    struct Kernel * out = kernel_alloc(kern->nparams);
    out->type = kern->type;
    out->nparams = kern->nparams;
    memmove(out->params,kern->params, kern->nparams * sizeof(double));
    return out;
}

/********************************************************//**
*   Free space allocated to the kernel
*************************************************************/
void kernel_free(struct Kernel * kern)
{
    if (kern != NULL){
        free(kern->params); kern->params = NULL;
        free(kern); kern = NULL;
    }
}


/********************************************************//**
*   Create a gaussian kernel
*************************************************************/
struct Kernel * kernel_gaussian(double scale, double width, double center)
{
    struct Kernel * kern = kernel_alloc(3);
    kern->type = KernGauss;
    kern->params[0] = scale;
    kern->params[1] = width*width; // note width squared!!
    kern->params[2] = center;
    return kern;
}

/********************************************************//**
*   Evaluate a kernel
*************************************************************/
double kernel_eval(struct Kernel * kern, double x)
{
    double out = 0.0;
    switch (kern->type){
    case KernGauss:
        out = gauss_kernel_eval(kern->params[0],kern->params[1],kern->params[2],x);
        break;
    case KernNone:
        fprintf(stderr,"No kernel type detected for evaluation\n");
        exit(1);
    }

    return out;
}

/********************************************************//**
*   Evaluate the derivative of a kernel at x
*************************************************************/
double kernel_deriv(struct Kernel * kern, double x)
{
    double out = 0.0;
    switch (kern->type){
    case KernGauss:
        out = gauss_kernel_deriv(kern->params[0],kern->params[1],kern->params[2],x);
        break;
    case KernNone:
        fprintf(stderr,"No kernel type detected for evaluation\n");
        exit(1);
    }
    return out;
}

/********************************************************//**
*   Allocate an array of kernels with the same number of params
*************************************************************/
struct Kernel ** kernel_array_alloc(size_t nkerns)
{
    struct Kernel ** kerns = malloc(nkerns * sizeof(struct Kernel *));
    if (kerns == NULL){
        fprintf(stderr,"Failure to allocate an array of struct Kernel\n");
        exit(1);
    }
    for (size_t ii = 0; ii < nkerns; ii++){
        kerns[ii] = NULL;
    }

    return kerns;
}

/********************************************************//**
*   Free an array of kernels
*************************************************************/
void kernel_array_free(size_t nkerns, struct Kernel ** kerns)
{
    if (kerns != NULL)
    {
        for (size_t ii = 0; ii < nkerns; ii++){
            kernel_free(kerns[ii]); kerns[ii] = NULL;
        }
        free(kerns); kerns = NULL;
    }
}



struct KernelApproxOpts
{
    size_t nnodes;
    double * centers;
    enum KernelType type;
    size_t nother_params;
    double * other_params;
    size_t nregress_params;
};

/********************************************************//**
* Allocate kernel approximation options
*************************************************************/
struct KernelApproxOpts * kernel_approx_opts_alloc()
{
    struct KernelApproxOpts * ko = malloc(sizeof(struct KernelApproxOpts));
    if (ko == NULL){
        fprintf(stderr,"Failure to allocate struct KernelApproxOpts\n");
        exit(1);
    }

    ko->nnodes = 0;
    ko->centers = NULL;
    ko->type = KernNone;
    ko->nother_params = 0;
    ko->other_params = NULL;

    ko->nregress_params = 0;
    return ko;
}

/********************************************************//**
* Free memory for kernel approximation options
*************************************************************/
void kernel_approx_opts_free(struct KernelApproxOpts * opts)
{
    if (opts != NULL){
        free(opts->centers); opts->centers = NULL;
        free(opts->other_params); opts->other_params = NULL;
        free(opts); opts = NULL;
    }
}

/********************************************************//**
*   A set of gaussian radial basis functions
*************************************************************/
struct KernelApproxOpts *
kernel_approx_opts_gauss_rbf(size_t ncenters, double * centers, double scale, double width)
{
    struct KernelApproxOpts * o = kernel_approx_opts_alloc();
    o->nnodes = ncenters;
    o->type = KernGauss;
    o->centers = calloc_double(ncenters);
    memmove(o->centers,centers,ncenters*sizeof(double));

    o->nother_params = 2;
    o->other_params = calloc_double(2);
    o->other_params[0] = scale;
    o->other_params[1] = width;

    o->nregress_params = ncenters;

    return o;
}

/********************************************************//**
*   Get number of parameters used for regression
*************************************************************/
size_t kernel_approx_opts_get_nparams(struct KernelApproxOpts * opts)
{
    assert (opts != NULL);
    return opts->nregress_params;
        
}


struct KernelExpansion
{
    size_t nalloc;
    
    size_t nkernels;
    double * coeff;
    struct Kernel ** kernels;
};

/********************************************************//**
*   Allocate a kernel expansion 
*************************************************************/
struct KernelExpansion * kernel_expansion_alloc(size_t nalloc)
{
    assert (nalloc > 0);
    struct KernelExpansion * ke = malloc(sizeof(struct KernelExpansion));
    if (ke == NULL){
        fprintf(stderr,"Failure to allocate struct KernelExpansion \n");
        exit(1);
    }

    ke->nalloc = nalloc;
    ke->nkernels = 0;
    ke->coeff = calloc_double(ke->nalloc);
    ke->kernels = kernel_array_alloc(nalloc);
    return ke;
}

/********************************************************//**
*   Serialize a kernel expansion
*
*   \param[in] ser       - location to which to serialize
*   \param[in] kern      - kernel expansion
*   \param[in] totSizeIn - if not null then only return total size 
*                          of array without serialization! if NULL then serialiaze
*
*   \return ptr : pointer to end of serialization
*************************************************************/
unsigned char *
serialize_kernel_expansion(unsigned char * ser, struct KernelExpansion * kern, size_t * totSizeIn)
{
    assert (kern->nkernels > 0);
    // nkernels, (coeff+size), kernels
    size_t totsize = sizeof(size_t) + kern->nkernels * sizeof(double) + sizeof(size_t);
    for (size_t ii = 0; ii < kern->nkernels; ii++){
        size_t ksize = 0;
        serialize_kernel(ser,kern->kernels[ii],&ksize);
        totsize+=ksize;
    }
    if (totSizeIn != NULL){
        *totSizeIn = totsize;
        return ser;
    }
    
    unsigned char * ptr = serialize_size_t(ser,kern->nkernels);
    ptr = serialize_doublep(ptr,kern->coeff,kern->nkernels);
    for (size_t ii = 0; ii < kern->nkernels; ii++){
        ptr = serialize_kernel(ptr,kern->kernels[ii],NULL);
    }
    return ptr;
}

/********************************************************//**
*   Deserialize a kernel expansion
*
*   \param[in]     ser  - input string
*   \param[in,out] kern - kernel expansion
*
*   \return ptr - ser + number of bytes of kernel expansion
*************************************************************/
unsigned char * 
deserialize_kernel_expansion( unsigned char * ser, struct KernelExpansion ** kern)
{

    size_t nkernels;
    unsigned char * ptr = ser;
    ptr = deserialize_size_t(ptr,&nkernels);

    *kern = kernel_expansion_alloc(nkernels);

    
    free((*kern)->coeff); (*kern)->coeff = NULL;
    ptr = deserialize_doublep(ptr,&((*kern)->coeff),&((*kern)->nkernels));

    for (size_t ii = 0; ii < nkernels; ii++){
        ptr = deserialize_kernel(ptr,&((*kern)->kernels[ii]));
    }

    return ptr;
    
}


/********************************************************//**
*   Copy a kernel expansion
*************************************************************/
struct KernelExpansion * kernel_expansion_copy(struct KernelExpansion * ke)
{
    if (ke == NULL){
        return NULL;
    }
    struct KernelExpansion * out = kernel_expansion_alloc(ke->nalloc);
    out->nkernels = ke->nkernels;
    for (size_t ii = 0; ii < ke->nalloc; ii++){
        out->coeff[ii] = ke->coeff[ii];
        out->kernels[ii] = kernel_copy(ke->kernels[ii]);
    }
    return out;
}

/********************************************************//**
*   Free a kernel expansion
*************************************************************/
void kernel_expansion_free(struct KernelExpansion * ke)
{
    if (ke != NULL){
        free(ke->coeff); ke->coeff = NULL;
        kernel_array_free(ke->nalloc,ke->kernels);
        free(ke); ke = NULL;
    }
}

/********************************************************//**
*   Add a kernel with specified weight
*************************************************************/
void kernel_expansion_add_kernel(struct KernelExpansion * ke, double weight, struct Kernel * kern)
{

    if (ke->nkernels == ke->nalloc)
    {

        // reallocate coefficients
        double * coeffs = realloc(ke->coeff,2 * ke->nalloc * sizeof(double));
        if (coeffs == NULL){
            fprintf(stderr, "Failure to adding a kernel to a kernel expansion\n");
            exit(1);
        }
        else{
            ke->coeff = coeffs;
            for (size_t ii = ke->nalloc; ii < 2 * ke->nalloc; ii++){
                ke->coeff[ii] = 0.0;
            }
                
        }

        // reallocate kernels
        struct Kernel ** karr = kernel_array_alloc(ke->nalloc);
        for (size_t ii = 0; ii < ke->nalloc; ii++){
            karr[ii] = kernel_copy(ke->kernels[ii]);
        }

        kernel_array_free(ke->nalloc,ke->kernels);
        ke->kernels = kernel_array_alloc(2*ke->nalloc);
        for (size_t ii = 0; ii < ke->nalloc; ii++){
            ke->kernels[ii] = kernel_copy(karr[ii]);
        }
        kernel_array_free(ke->nalloc,karr);  karr = NULL;

        // specify the size of allocated space
        ke->nalloc = 2 * ke->nalloc;
        kernel_expansion_add_kernel(ke,weight,kern);
    }
    else{
        ke->coeff[ke->nkernels] = weight;
        if (ke->kernels[ke->nkernels] == NULL){
            kernel_free(ke->kernels[ke->nkernels]);
        }
        ke->kernels[ke->nkernels] = kernel_copy(kern);
        ke->nkernels++;
    }
}


/********************************************************//**
*   Evaluate a kernel expansion
*************************************************************/
double kernel_expansion_eval(struct KernelExpansion * kern, double x)
{
    double out = 0.0;
    for (size_t ii = 0; ii < kern->nkernels; ii++)
    {
        out += kern->coeff[ii] * kernel_eval(kern->kernels[ii],x);
    }
    return out;
}

/********************************************************//**
*   Evaluate a kernel expansion consisting of sequentially increasing 
*   order kernels from the same family.
*
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
void kernel_expansion_evalN(struct KernelExpansion * poly, size_t N,
                            const double * x, size_t incx, double * y, size_t incy)
{
    for (size_t ii = 0; ii < N; ii++){
        y[ii*incy] = kernel_expansion_eval(poly,x[ii*incx]);
    }
}


/********************************************************//**
*   Evaluate the derivative of a kernel expansion (useful for gradients)
*************************************************************/
double kernel_expansion_deriv_eval(double x, void * kernin)
{
    struct KernelExpansion * kern = kernin;
    double out = 0.0;
    for (size_t ii = 0; ii < kern->nkernels; ii++)
    {
        out += kern->coeff[ii] * kernel_deriv(kern->kernels[ii],x);
    }
    return out;
}


/********************************************************//**
*   Multiply by scalar and overwrite expansion
*
*   \param[in] a - scaling factor for first polynomial
*   \param[in] k - kernel expansion to scale
*************************************************************/
void kernel_expansion_scale(double a, struct KernelExpansion * x)
{
    
    size_t ii;
    for (ii = 0; ii < x->nkernels; ii++){
        x->coeff[ii] *= a;
    }
}


/********************************************************//*
*   Evaluate the gradient of an orthonormal kernel expansion 
*   with respect to the parameters
*
*   \param[in]     ke   - kernel expansion
*   \param[in]     nx   - number of x points
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N,nx)
*
*   \return 0 success, 1 failure

*************************************************************/
int kernel_expansion_param_grad_eval(
    struct KernelExpansion * ke, size_t nx, const double * x, double * grad)
{
    int res = 0;
    size_t nparams = ke->nkernels;
    for (size_t ii = 0; ii < nx; ii++){
        for (size_t jj = 0; jj < ke->nkernels; jj++){
            grad[ii*nparams + jj] = kernel_eval(ke->kernels[jj],x[ii]);
        }
    }
    return res;
}



/********************************************************//**
*   Get number of parameters
*************************************************************/
size_t kernel_expansion_get_num_params(const struct KernelExpansion * ke)
{
    assert (ke != NULL);
    return ke->nkernels;
}

/********************************************************//**
*   Get parameters defining kernel (for now just coefficients)
*************************************************************/
size_t kernel_expansion_get_params(const struct KernelExpansion * ke, double * param)
{
    assert (ke != NULL);
    memmove(param,ke->coeff,ke->nkernels * sizeof(double));
    return ke->nkernels;
}

void print_kernel_expansion(struct KernelExpansion * k, size_t prec, 
                            void * args)
{

    (void)(prec);
    if (args == NULL){
        printf("Kernel Expansion:\n");
        printf("--------------------------------\n");
        printf("Number of kernels = %zu\n",k->nkernels);
        for (size_t ii = 0; ii < k->nkernels; ii++){
            printf("Kernel %zu: weight=%G \n\n",ii,k->coeff[ii]);
            /* print_kernel(k); */
        }
    }
}






