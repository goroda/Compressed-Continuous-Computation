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
/* #define ZEROTHRESH  1e0 * DBL_EPSILON */
//#define ZEROTHRESH 0.0
//#define ZEROTHRESH  1e2 * DBL_EPSILON
//#define ZEROTHRESH  1e-12

#include "stringmanip.h"
#include "array.h"
#include "futil.h"
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

/********************************************************//**
*   Integrate a gaussian kernel from lb to ub
*************************************************************/
double gauss_kernel_integrate(double scale, double width_squared, double center, double lb, double ub)
{
    double width = sqrt(width_squared);
    return 0.5 * sqrt(M_PI) * width * scale * (erf((center-lb)/width) - erf((center-ub)/width));
}

/********************************************************//**
*   Integrate a product of gaussian kernels from lb to ub
*************************************************************/
double gauss_kernel_inner(double s1, double w1, double c1,
                          double s2, double w2, double c2,
                          double lb, double ub)
{

    double width1 = sqrt(w1);
    double width2 = sqrt(w2);
    double wsum = w1+w2;
    double sqrt_wsum = sqrt(wsum);
        
    double pre = sqrt(M_PI) * width1 * width2 * s1 * s2;
    double den = 2.0 * sqrt_wsum;

    double cdiff = (c1-c2);
    double n1 = exp(- cdiff * cdiff / wsum);
    pre *= n1;


    double c1w2 = c1*w2;
    double w1c2 = w1*c2;
    double c1w2_plus_w1c2 = c1w2 + w1c2;
    double w1w2sqrt_wsum = width1*width2*sqrt_wsum;
    double mid1 = erf((c1w2_plus_w1c2 - lb * wsum)/(w1w2sqrt_wsum));
    double mid2 = erf((c1w2_plus_w1c2 - ub * wsum)/(w1w2sqrt_wsum));
    
    return pre / den * (mid1 -mid2);
}


enum KernelType {KernGauss, KernNone};
struct Kernel
{
    enum KernelType type;
    size_t nparams;
    double * params;
};



/********************************************************//**
*   Check if same kernel
*************************************************************/
int same_kernel(struct Kernel * x, struct Kernel * y)
{
    if (x->type != y->type){
        return 0;
    }
    else if (x->nparams != y->nparams){
        return 0;
    }
    int ret = 1;
    for (size_t ii = 0; ii < x->nparams; ii++){
        if (fabs(x->params[ii] - y->params[ii]) > 1e-15){
            ret = 0;
            break;
        }
    }
    return ret;
}

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
    /* printf("serializing nparam = %zu\n",kern->nparams); */
    
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
deserialize_kernel(unsigned char * ser, struct Kernel ** kern)
{

    int type;
    size_t nparam;
    unsigned char * ptr = ser;
    ptr = deserialize_int(ptr,&type);
    ptr = deserialize_size_t(ptr,&nparam);

    /* printf("deserializing nparam = %zu\n",nparam); */
    *kern = kernel_alloc(nparam);
    (*kern)->type = (enum KernelType) type;
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

void kernel_to_string(struct Kernel * kern, char * output)
{
    if (kern->type == KernGauss){
        sprintf(output,"Squared Exponential: scale=%G, width^2=%G, center=%G",
                kern->params[0],kern->params[1],kern->params[2]);
    }
}

/********************************************************//**
*   Update the center of a kernel
*************************************************************/
void kernel_update_center(struct Kernel * kern, double center)
{
    switch (kern->type){
    case KernGauss:
        kern->params[2] = center;
        break;
    case KernNone:
        fprintf(stderr,"No kernel type detected for updating center\n");
        exit(1);
    }
}

/********************************************************//**
*   Get the center of a kernel
*************************************************************/
double kernel_get_center(const struct Kernel * kern)
{
    double out = 0.123456789;
    switch (kern->type){
    case KernGauss:
        out = kern->params[2];
        break;
    case KernNone:
        fprintf(stderr,"No kernel type detected for updating center\n");
        exit(1);
    }
    return out;
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
*   Evaluate the gradient of a kernel at x with respect to the center
*************************************************************/
double kernel_grad_center(struct Kernel * kern, double x)
{
    double out = 0.0;
    switch (kern->type){
    case KernGauss: // just switch center and x because of symmetry!
        out = gauss_kernel_deriv(kern->params[0],kern->params[1],x,kern->params[2]);
        break;
    case KernNone:
        fprintf(stderr,"No kernel type detected for evaluation\n");
        exit(1);
    }
    return out;
}

/********************************************************//**
*   Integrate a kernel
*************************************************************/
double kernel_integrate(struct Kernel * kern, double lb, double ub)
{
    double out = 0.0;
    switch (kern->type){
    case KernGauss:
        out = gauss_kernel_integrate(kern->params[0],kern->params[1],kern->params[2],lb,ub);
        break;
    case KernNone:
        fprintf(stderr,"No kernel type detected for evaluation\n");
        exit(1);
    }
    return out;
}


/********************************************************//**
*   Integrate a kernel
*************************************************************/
double kernel_inner(struct Kernel * k1, struct Kernel * k2, double lb, double ub)
{
    double out = 0.0;
    if ((k1->type == KernGauss) && (k2->type == KernGauss)){
        out = gauss_kernel_inner(k1->params[0],k1->params[1],k1->params[2],
                                 k2->params[0],k2->params[1],k2->params[2],
                                 lb,ub);
    }
    else{
        fprintf(stderr,"Cannot integrate product of specified kernel types\n");
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


    // bounds for some operations like generating linear functions
    double prac_lb; //lower bound for some operations
    double prac_ub; //upper bound for some operations

    // bounds for approximation
    double lb; 
    double ub;

    // additional
    int adapt_center;
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

    ko->prac_lb = -DBL_MAX;
    ko->prac_ub = DBL_MAX;

    ko->lb = -DBL_MAX;
    ko->ub = DBL_MAX;

    ko->adapt_center = 0;
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
kernel_approx_opts_gauss(size_t ncenters, double * centers, double scale, double width)
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

    o->prac_lb = DBL_MAX;
    o->prac_ub = -DBL_MAX;
    for (size_t ii = 0; ii < ncenters; ii++){
        if (centers[ii] < o->prac_lb){
            o->prac_lb = centers[ii];
        }

        if (centers[ii] > o->prac_ub){
            o->prac_ub = centers[ii];
        }
    }

    o->lb = -DBL_MAX;
    o->ub = DBL_MAX;
    return o;
}


/********************************************************//**
*   Set location adaptation (1 = yes, 0 = no)
*************************************************************/
void kernel_approx_opts_set_center_adapt(struct KernelApproxOpts * opts,int adapt)
{
    assert (opts != NULL);
    opts->adapt_center = adapt;
    if (adapt == 1){
        opts->nregress_params = 2*opts->nnodes;
    }
}

/********************************************************//**
*   Check if linear parameterization (0 if no, 1 if yes)
*************************************************************/
int kernel_approx_opts_linear_p(const struct KernelApproxOpts * opts)
{
    assert (opts != NULL);
    int lin = 1;
    /* printf(" adapt->center = %d\n",opts->adapt_center); */
    if (opts->adapt_center == 1){
        lin = 0;
    }
    return lin;
}

/********************************************************//**
*   Get number of parameters used for regression
*************************************************************/
size_t kernel_approx_opts_get_nparams(struct KernelApproxOpts * opts)
{
    assert (opts != NULL);
    return opts->nregress_params;
        
}

/********************************************************//**
*   Get number of parameters used for regression
*************************************************************/
void kernel_approx_opts_set_nparams(struct KernelApproxOpts * opts, size_t nparam)
{
    assert (opts != NULL);
    opts->nregress_params = nparam;
    if (opts->adapt_center == 0){
        opts->nnodes = nparam;
    }
    else{
        if (nparam %2 != 0){
            fprintf(stderr, "Adapting of kernel centers is turned on\n");
            fprintf(stderr, "Therefore must set an even number of parameters\n");
            exit(1);
        }
        opts->nnodes = nparam/2;
    }

    free(opts->centers); opts->centers = NULL;
    assert (opts->prac_lb < DBL_MAX);
    assert (opts->prac_ub > -DBL_MAX);
    opts->centers = linspace(opts->prac_lb,opts->prac_ub,opts->nnodes);
}


/********************************************************//**
*   Set lower bounds
*************************************************************/
void kernel_approx_opts_set_lb(struct KernelApproxOpts * opts, double lb)
{
    opts->lb = lb;
    opts->prac_lb = lb;
}

/********************************************************//**
*   Get lower bounds
*************************************************************/
double  kernel_approx_opts_get_lb(struct KernelApproxOpts * opts)
{
    return opts->lb;
}

/********************************************************//**
*   Set upper bounds
*************************************************************/
void kernel_approx_opts_set_ub(struct KernelApproxOpts * opts, double ub)
{
    opts->ub = ub;
    opts->prac_ub = ub;
}

/********************************************************//**
*   Get upper bounds
*************************************************************/
double  kernel_approx_opts_get_ub(struct KernelApproxOpts * opts)
{
    return opts->ub;
}

struct KernelExpansion
{
    size_t nalloc;
    
    size_t nkernels;
    double * coeff;
    struct Kernel ** kernels;

    double lb;
    double ub;

    size_t include_kernel_param;
};

/********************************************************//**
*   Get the lower bound
*************************************************************/
double kernel_expansion_get_lb(const struct KernelExpansion * k)
{
    assert (k != NULL);
    return k->lb;
}

/********************************************************//**
*   Get the upper bound
*************************************************************/
double kernel_expansion_get_ub(const struct KernelExpansion * k)
{
    assert (k != NULL);
    return k->ub;
}

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
    ke->lb = -DBL_MAX;
    ke->ub = DBL_MAX;

    ke->include_kernel_param = 0;
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
    // nkernels, (coeff+size), kernels, lb, ub, include_kernel_param
    size_t totsize =
        sizeof(size_t) + kern->nkernels * sizeof(double) +
        sizeof(size_t) + 2 * sizeof(double) +
        sizeof(size_t);
    
    for (size_t ii = 0; ii < kern->nkernels; ii++){
        size_t ksize = 0;
        serialize_kernel(ser,kern->kernels[ii],&ksize);
        totsize+=ksize;
    }
    if (totSizeIn != NULL){
        *totSizeIn = totsize;
        return ser;
    }

    /* printf("serializing %zu\n",kern->nkernels); */
    unsigned char * ptr = serialize_size_t(ser,kern->nkernels);
    ptr = serialize_doublep(ptr,kern->coeff,kern->nkernels);
    for (size_t ii = 0; ii < kern->nkernels; ii++){
        ptr = serialize_kernel(ptr,kern->kernels[ii],NULL);
    }
    ptr = serialize_double(ptr,kern->lb);
    ptr = serialize_double(ptr,kern->ub);
    ptr = serialize_size_t(ptr,kern->include_kernel_param);
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
    /* printf("deseiralizing nkernels = %zu\n",nkernels); */
    *kern = kernel_expansion_alloc(nkernels);
    (*kern)->nkernels = nkernels;
    (*kern)->nalloc = nkernels;

    free((*kern)->coeff); (*kern)->coeff = NULL;

    double * coeff = NULL;
    size_t n2;
    ptr = deserialize_doublep(ptr,&coeff,&n2);
    /* printf("n2,nk = (%zu,%zu)\n",n2,nkernels); */
    (*kern)->coeff = coeff;
    
    for (size_t ii = 0; ii < nkernels; ii++){
        ptr = deserialize_kernel(ptr,&((*kern)->kernels[ii]));
    }

    ptr = deserialize_double(ptr,&((*kern)->lb));
    ptr = deserialize_double(ptr,&((*kern)->ub));
    ptr = deserialize_size_t(ptr,&((*kern)->include_kernel_param));
    /* printf("lb,ub=%G,%G\n",(*kern)->lb,(*kern)->ub); */
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
    out->lb = ke->lb;
    out->ub = ke->ub;

    out->include_kernel_param = ke->include_kernel_param;
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
*   Set lower and upper bounds
*************************************************************/
void kernel_expansion_set_bounds(struct KernelExpansion * ke, double lb, double ub)
{
    assert(ke != NULL);
    ke->lb = lb;
    ke->ub = ub;
}

/********************************************************//**
*   Get number of kernels
*************************************************************/
size_t kernel_expansion_get_nkernels(const struct KernelExpansion * ke)
{
    assert (ke != NULL);
    return ke->nkernels;
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
    Initialize a kernel expansion with some options

    \param[in] opts  - options
  
    \return kernel expansion
*************************************************************/
struct KernelExpansion *
kernel_expansion_init(const struct KernelApproxOpts * opts)
{
    struct KernelExpansion * ke = kernel_expansion_alloc(1);
    struct Kernel * kern = NULL;
    for (size_t ii = 0; ii < opts->nnodes; ii++)
    {
        switch (opts->type){
        case KernGauss:
            kern = kernel_gaussian(opts->other_params[0],opts->other_params[1],opts->centers[ii]);
            kernel_expansion_add_kernel(ke,0.0,kern);
            break;
        case KernNone:
            fprintf(stderr, "Kernel Approximation Options don't sepcify a kernel type\n");
            exit(1);
        }
        kernel_free(kern); kern = NULL;
    }

    ke->lb = opts->lb;
    ke->ub = opts->ub;

    ke->include_kernel_param = (size_t)opts->adapt_center;
    /* kernel_expansion_set_bounds(ke,opts->centers[0],opts->centers[opts->nnodes-1]); */

    return ke;
}

/********************************************************//**
    Update the params of a kernel expansion
*************************************************************/
int kernel_expansion_update_params(struct KernelExpansion * ke, size_t dim, const double * param)
{

    if (ke->include_kernel_param == 0){
        assert (ke->nkernels == dim);
        memmove(ke->coeff,param,dim*sizeof(double));
    }
    else{
        if (dim == ke->nkernels){ // still just update the coeffs
            memmove(ke->coeff,param,dim*sizeof(double));
        }
        else{
            assert (ke->nkernels = dim/2);
            memmove(ke->coeff,param,dim/2 * sizeof(double));
            for (size_t ii = 0; ii < ke->nkernels; ii++){
                kernel_update_center(ke->kernels[ii],param[ii+dim/2]);
            }
        }
    }

    return 0;
}

/********************************************************//**
    Initialize a kernel expansion with particular parameters

    \param[in] opts  - options
    \param[in] dim   - number of parameters
    \param[in] param - parameters
  
    \return kernel expansion
    
    \note makes a copy of nodes and coefficients
*************************************************************/
struct KernelExpansion *
kernel_expansion_create_with_params(struct KernelApproxOpts * opts,
                                    size_t dim, const double * param)
{
    assert (opts != NULL);
    assert (opts->nregress_params == dim);

    struct KernelExpansion * ke = kernel_expansion_init(opts);
    kernel_expansion_update_params(ke,dim,param);
    
    return ke;
}


/********************************************************//**
    Return a zero function

    \param[in] opts        - options
    \param[in] force_param - nothing yet

    \return ke - zero function
************************************************************/
struct KernelExpansion *
kernel_expansion_zero(const struct KernelApproxOpts * opts, int force_param)
{

    (void) (force_param);
    assert(opts != NULL);
    
    struct KernelExpansion * ke = kernel_expansion_init(opts);
    for (size_t ii = 0; ii < ke->nkernels; ii++){
        ke->coeff[ii] = 0.0;
    }

    return ke;
}


/********************************************************//**
    Return an approximately linear function

    \param[in] a      - value of the slope function
    \param[in] offset - offset
    \param[in] opts   - upper bound

    \return ke - approximately linear function

    \note computed by linear regression
************************************************************/
struct KernelExpansion *
kernel_expansion_linear(double a, double offset, const struct KernelApproxOpts * opts)
{
    assert(opts != NULL);
    
    struct KernelExpansion * ke = kernel_expansion_init(opts);
    for (size_t ii = 0; ii < ke->nkernels; ii++){
        ke->coeff[ii] = 0.0;
    }

    // weird bounds
    size_t nregress = 100 * ke->nkernels;
    double lb = opts->prac_lb;
    double ub = opts->prac_ub;
    
    double * x = linspace(lb,ub,nregress);
    double * y = calloc_double(nregress);
    double * A = calloc_double(nregress * ke->nkernels);

    for (size_t ii = 0; ii < nregress; ii++){
        y[ii] = a * x[ii] + offset;
    }
    
    for (size_t jj = 0; jj < ke->nkernels; jj++){
        for (size_t ii = 0; ii < nregress; ii++){
            A[jj*nregress+ii] = kernel_eval(ke->kernels[jj],x[ii]);
        }
    }

    /* dprint2d_col(nregress,ke->nkernels,A); */
    // perform linear least squares for the coefficients
    linear_ls(nregress,ke->nkernels,A,y,ke->coeff);

    free(x); x = NULL;
    free(A); A = NULL;
    free(y); y = NULL;
    return ke;
}

/*******************************************************//**
    Update a linear function

    \param[in] f      - existing linear function
    \param[in] a      - slope of the function
    \param[in] offset - offset of the function

    \returns 0 if successfull, 1 otherwise                   
***********************************************************/
int kernel_expansion_linear_update(struct KernelExpansion * f, double a, double offset)
{
    (void) f;
    (void) a;
    (void) offset;
    NOT_IMPLEMENTED_MSG("kernel_expansion_linear_update");
    return 1;
}


/********************************************************//**
    Return a constant function

    \param[in] val  - value
    \param[in] opts - options

    \return ke - zero function
************************************************************/
struct KernelExpansion *
kernel_expansion_constant(double val, const struct KernelApproxOpts * opts)
{
    return kernel_expansion_linear(0.0, val, opts);
}

/*******************************************************//**
    Return a quadratic function a * (x - offset)^2 = a (x^2 - 2offset x + offset^2)

    \param[in] a      - quadratic coefficients
    \param[in] offset - shift of the function
    \param[in] aopts  - extra arguments depending on function_class, sub_type,  etc.

    \return gf - quadratic
************************************************************/
struct KernelExpansion *
kernel_expansion_quadratic(double a, double offset, void * aopts)
{
    (void)(a);
    (void)(offset);
    (void)(aopts);
    NOT_IMPLEMENTED_MSG("kernel_expansion_quadratic")
    return NULL;
}
                           
    
/********************************************************//**
*   Evaluate a kernel expansion
*************************************************************/
double kernel_expansion_eval(const struct KernelExpansion * kern, double x)
{
    double out = 0.0;
    for (size_t ii = 0; ii < kern->nkernels; ii++)
    {
        out += kern->coeff[ii] * kernel_eval(kern->kernels[ii],x);
    }
    return out;
}


/********************************************************//**
*   Check that the two expansions have identical bases
*************************************************************/
int check_same_nodes_kernels(struct KernelExpansion * x, struct KernelExpansion * y)
{
    if (x->nkernels != y->nkernels){
        return 0;
    }

    int ret = 1;
    for (size_t ii = 0; ii < x->nkernels; ii++){
        ret = same_kernel(x->kernels[ii],y->kernels[ii]);
        if (ret == 0){
            break;
        }
    }
    return ret;
}



/********************************************************//**
*   Evaluate a kernel expansion consisting of sequentially increasing 
*   order kernels from the same family.
*
*   \param[in]     ke   - function
*   \param[in]     N    - number of evaluations
*   \param[in]     x    - location at which to evaluate
*   \param[in]     incx - increment of x
*   \param[in,out] y    - allocated space for evaluations
*   \param[in]     incy - increment of y*
*
*   \note Currently just calls the single evaluation code
*         Note sure if this is optimal, cache-wise
*************************************************************/
void kernel_expansion_evalN(struct KernelExpansion * ke, size_t N,
                            const double * x, size_t incx, double * y, size_t incy)
{
    for (size_t ii = 0; ii < N; ii++){
        y[ii*incy] = kernel_expansion_eval(ke,x[ii*incx]);
    }
}

/********************************************************//**
*   Obtain the derivative of a kernel
*************************************************************/
struct KernelExpansion * kernel_expansion_deriv(const struct KernelExpansion * ke)
{
    (void) (ke);
    NOT_IMPLEMENTED_MSG("kernel_expansion_deriv")
    exit(1);
}

/********************************************************//**
*   Obtain the second derivative of a kernel
*************************************************************/
struct KernelExpansion * kernel_expansion_dderiv(const struct KernelExpansion * ke)
{
    (void) (ke);
    NOT_IMPLEMENTED_MSG("kernel_expansion_dderiv")
    exit(1);
}

/********************************************************//**
*   Obtain the second derivative of a kernel with periodic boundary conditions
*************************************************************/
struct KernelExpansion * kernel_expansion_dderiv_periodic(const struct KernelExpansion * ke)
{
    (void) (ke);
    NOT_IMPLEMENTED_MSG("kernel_expansion_dderiv_periodic")
    exit(1);
}

/********************************************************//**
*   Evaluate the derivative of a kernel expansion (useful for gradients)
*************************************************************/
double kernel_expansion_deriv_eval(const struct KernelExpansion * kern, double x)    
{
    assert (kern != NULL);
    double out = 0.0;
    for (size_t ii = 0; ii < kern->nkernels; ii++)
    {
        out += kern->coeff[ii] * kernel_deriv(kern->kernels[ii],x);
    }
    return out;
}

/********************************************************//**
*   Add two kernels
*************************************************************/
int kernel_expansion_axpy(double a, struct KernelExpansion * x, struct KernelExpansion * y)
{
    int same_nodes_and_kernels = check_same_nodes_kernels(x,y);

    if (same_nodes_and_kernels == 1){
        for (size_t ii = 0; ii < y->nkernels; ii++)
        {
            y->coeff[ii] += (a * x->coeff[ii]);
        }
    }
    else{
        fprintf(stderr, "Cannot axpy kernel expansions that have different structures\n");
        exit(1);
    }

    return 0;
}

/********************************************************//**
*   Integrate a kernel expansion
*************************************************************/
double kernel_expansion_integrate(struct KernelExpansion * a)
{
    double out = 0.0;
    for (size_t ii = 0; ii < a->nkernels; ii++){
        out += a->coeff[ii]*kernel_integrate(a->kernels[ii],a->lb,a->ub);
    }
    return out;
}

/********************************************************//**
*   Integrate a kernel expansion on a weighted domain
*************************************************************/
double kernel_expansion_integrate_weighted(struct KernelExpansion * a)
{
    return kernel_expansion_integrate(a);
}

/********************************************************//**
   Multiply two functions
    
   \param[in] f   - first function
   \param[in] g   - second function

   \returns product
            
   \note 
*************************************************************/
struct KernelExpansion * kernel_expansion_prod(const struct KernelExpansion * f,
                                               const struct KernelExpansion * g)
{
    (void)(f);
    (void)(g);
    NOT_IMPLEMENTED_MSG("kernel_expansion_prod")
    return NULL;
}

/********************************************************//**
*   Inner product between two kernel expansions of the same type
*
*   \param[in] a - first kernel
*   \param[in] b - second kernel
*
*   \return  inner product
*
*   \note
*   Computes  \f$ \int_{lb}^ub  a(x)b(x) dx \f$ 
*   where the bounds are the tightest bounds
*************************************************************/
double
kernel_expansion_inner(struct KernelExpansion * a,
                       struct KernelExpansion * b)
{

    double lb = a->lb;
    if (a->lb < b->lb){
        lb = b->lb;
    }

    double ub = a->ub;
    if (a->ub > b->ub){
        ub = b->ub;
    }

    /* printf("a coeffs: "); dprint; */

    double out = 0.0;
    for (size_t ii = 0; ii < a->nkernels; ii++){
        for (size_t jj = 0; jj < b->nkernels; jj++){
            out += a->coeff[ii]*b->coeff[jj]*kernel_inner(a->kernels[ii],b->kernels[jj],lb,ub);
        }
    }
    return out;
}



/********************************************************//**
*   Multiply by scalar and overwrite expansion
*
*   \param[in] a - scaling factor for first polynomial
*   \param[in] x - kernel expansion to scale
*************************************************************/
void kernel_expansion_scale(double a, struct KernelExpansion * x)
{
    size_t ii;
    for (ii = 0; ii < x->nkernels; ii++){
        x->coeff[ii] *= a;
    }
}

/* Multiply a kernel expansion by -1 */
void kernel_expansion_flip_sign(struct KernelExpansion * x)
{
    kernel_expansion_scale(-1.0, x);
}

/********************************************************//*
*   Evaluate the gradient of a kernel expansion 
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
    if (ke->include_kernel_param == 1){
        nparams *= 2;
    }
    for (size_t ii = 0; ii < nx; ii++){
        for (size_t jj = 0; jj < ke->nkernels; jj++){
            grad[ii*nparams + jj] = kernel_eval(ke->kernels[jj],x[ii]);
        }
        if (ke->include_kernel_param == 1){
            for (size_t jj = 0; jj < ke->nkernels; jj++){
                grad[ii*nparams+jj+ke->nkernels] =
                    ke->coeff[jj]*kernel_grad_center(ke->kernels[jj],x[ii]);
            }
        }
    }
    return res;
}

/********************************************************//*
*   Evaluate the gradient of an kernel expansion 
*   with respect to the parameters
*
*   \param[in]     ke   - kernel expansion
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N)
*
*   \return value

*************************************************************/
double kernel_expansion_param_grad_eval2(
    struct KernelExpansion * ke, double x, double * grad)
{

    size_t nparams = ke->nkernels;
    if (ke->include_kernel_param == 1){
        nparams *= 2;
    }

    double eval = 0.0;
    for (size_t jj = 0; jj < ke->nkernels; jj++){
        grad[jj] = kernel_eval(ke->kernels[jj],x);
        eval += grad[jj] * ke->coeff[jj];
    }
    if (ke->include_kernel_param == 1){
        for (size_t jj = 0; jj < ke->nkernels; jj++){
            grad[jj+ke->nkernels] = ke->coeff[jj]*kernel_grad_center(ke->kernels[jj],x);
        }
    }

    return eval;
}

/********************************************************//**
    Take a gradient of the squared norm 
    with respect to its parameters, and add a scaled version
    of this gradient to *grad*

    \param[in]     ke    - kernel
    \param[in]     scale - scaling for additional gradient
    \param[in,out] grad  - gradient, on output adds scale * new_grad

    \return  0 - success, 1 -failure

************************************************************/
int
kernel_expansion_squared_norm_param_grad(const struct KernelExpansion * ke,
                                         double scale, double * grad)
{

    if (ke->include_kernel_param == 1){
        assert (1 == 0);
    }
    
    int res = 1;
    for (size_t ii = 0; ii < ke->nkernels; ii++){
        double g1 = 0.0;
        for (size_t jj = 0; jj < ke->nkernels; jj++){
            g1 += ke->coeff[jj]*kernel_inner(ke->kernels[ii],ke->kernels[jj],ke->lb,ke->ub);
        }
        /* g1 *= ke->coeff[ii]; */
        grad[ii] += 2.0 * scale * g1;
    }
    res = 0;
    return res;
}


/********************************************************//**
*   Get number of parameters
*************************************************************/
size_t kernel_expansion_get_num_params(const struct KernelExpansion * ke)
{
    assert (ke != NULL);
    if (ke->include_kernel_param == 0){
        return ke->nkernels;        
    }
    else{
        return 2*ke->nkernels;
    }
}

/********************************************************//**
*   Get parameters defining kernel (for now just coefficients)
*************************************************************/
size_t kernel_expansion_get_params(const struct KernelExpansion * ke, double * param)
{
    assert (ke != NULL);
    memmove(param,ke->coeff,ke->nkernels * sizeof(double));
    if (ke->include_kernel_param == 0){
        return ke->nkernels;
    }
    else{
        for (size_t ii = 0; ii < ke->nkernels; ii++){
            param[ii+ke->nkernels] = kernel_get_center(ke->kernels[ii]);
        }
        return 2 * ke->nkernels;
    }
}

/********************************************************//**
*   Get parameters defining kernel (for now just coefficients)
*************************************************************/
double * kernel_expansion_get_params_ref(const struct KernelExpansion * ke, size_t * nparam)
{
    assert (ke != NULL);
    assert (ke->include_kernel_param == 0);
    *nparam = ke->nkernels;
    return ke->coeff;
}


/* static double numint(struct KernelExpansion * a, struct KernelExpansion * b, size_t nint) */
/* { */

/*     double lb = a->lb; */
/*     if (a->lb < b->lb){ */
/*         lb = b->lb; */
/*     } */

/*     double ub = a->ub; */
/*     if (a->ub > b->ub){ */
/*         ub = b->ub; */
/*     } */

/*     double * x = linspace(lb,ub,nint); */
/*     double num_int = 0.0; */
/*     for (size_t ii = 0; ii < nint-1; ii++){ */
/*         double val1 = kernel_expansion_eval(a,x[ii]); */
/*         double val2 = kernel_expansion_eval(b,x[ii]); */
/*         num_int += val1*val2*(x[ii+1]-x[ii]); */
/*     } */
/*     free(x); x = NULL; */
/*     return num_int; */
/* } */


/********************************************************//**
    Generate an orthonormal basis
    
    \param[in]     n    - number of basis function
    \param[in,out] f    - linear element expansions that are nulled
    \param[in]     opts - approximation options

    \note
    Uses modified gram schmidt to determine function coefficients
*************************************************************/
void kernel_expansion_orth_basis(size_t n, struct KernelExpansion ** f, struct KernelApproxOpts * opts)
{
    assert (opts != NULL);
    assert (opts->nnodes >= n);

    for (size_t ii = 0; ii < n; ii++){
        f[ii] = kernel_expansion_init(opts); // they all have zero coefficients;
        f[ii]->coeff[ii] = 1.0;        
        /* for (size_t jj = 0; jj<= ii; jj++){ */
        /*     f[ii]->coeff[jj] = 1.0; */
        /* } */
    }

    double inner, norm, proj;
    for (size_t ii = 0; ii < n; ii++){
        inner = kernel_expansion_inner(f[ii],f[ii]);
        /* inner = numint(f[ii],f[ii],60000); */
        if (inner < 0){
            fprintf(stderr,"Cannot generate %zu orthonormal basis with this kernel\n",n);
            fprintf(stderr,"Consider making the correlation length smaller\n");
            exit(1);
            /* inner = 0; */
            /* kernel_expansion_scale(0.0,f[ii]); */
            /* for (size_t jj = ii; jj < n; jj++){ */
            /*     kernel_expansion_scale(0.0,f[jj]); */
            /* } */
            /* break; */
        }
        else{
            norm = sqrt(inner);
            /* printf("norm = %G\n",norm); */
            if (isnan(norm)){
                fprintf(stderr,"inner is %G\n",inner);
                fprintf(stderr,"norm is NAN\n");
                exit(1);
            }
            kernel_expansion_scale(1/norm,f[ii]);
        }
        for (size_t jj = ii+1; jj < n; jj++){
            proj = kernel_expansion_inner(f[ii],f[jj]);
            /* proj = numint(f[ii],f[jj],60000); */

            if (isnan(proj)){
                fprintf(stderr,"proj is NAN\n");
                exit(1);
            }
            if (isinf(proj)){
                fprintf(stderr,"proj is NAN\n");
                exit(1);
            }
            int iter = 0;
            while (fabs(proj) > 1e-14)
            {
                
                /* printf("proj = %G\n",proj); */
                kernel_expansion_axpy(-proj,f[ii],f[jj]);

                //check inner
                proj = kernel_expansion_inner(f[ii],f[jj]);
                /* proj = numint(f[ii],f[jj],60000); */
                if (fabs(proj) <= 1e-14){
                    break;
                }
                /* else{ */
                /*     fprintf(stderr,"projection too large, going again: = %G\n",proj); */
                /*     /\* assert (proj < 1e-14);                     *\/ */
                /* } */
                iter++;
                /* if (iter > 0){ */
                /*     fprintf(stderr,"\t on iter %d\n",iter); */
                /* } */
                if (iter > 0){
                    /* exit(1); */
                    break;
                }
            }
        }
    }
}

void print_kernel_expansion(struct KernelExpansion * k, size_t prec, 
                            void * args, FILE *fp)
{

    (void)(prec);
    char kern[256];
    if (args == NULL){
        fprintf(fp, "Kernel Expansion:\n");
        fprintf(fp, "--------------------------------\n");
        fprintf(fp, "Number of kernels = %zu\n",k->nkernels);
        for (size_t ii = 0; ii < k->nkernels; ii++){
            kernel_to_string(k->kernels[ii],kern);
            fprintf(fp, "Kernel %zu: weight=%G \n",ii,k->coeff[ii]);
            fprintf(fp, "\t %s\n\n",kern);
            /* print_kernel(k); */
        }
    }
}




/********************************************************//**
    Compute the location and value of the maximum, 
    in absolute value,

    \param[in]     f       - function
    \param[in,out] x       - location of maximum
    \param[in]     optargs - optimization arguments

    \return absolute value of the maximum
************************************************************/
double kernel_expansion_absmax(const struct KernelExpansion * f, double * x, void * optargs)
{
    if (optargs == NULL){
        fprintf(stderr, "Must provide optimization arguments to kernel_expansion_absmax\n");
        exit(1);
    }
    
    struct c3Vector * optnodes = optargs;
    double mval = fabs(kernel_expansion_eval(f, optnodes->elem[0]));
    *x = optnodes->elem[0];
    double cval = mval;
    *x = optnodes->elem[0];
    for (size_t ii = 0; ii < optnodes->size; ii++){
        double val = fabs(kernel_expansion_eval(f, optnodes->elem[ii]));
        double tval = val;
        if (val > mval){
            mval = val;
            cval = tval;
            *x = optnodes->elem[ii];
        }
    }
    return cval;
}

/********************************************************//**
    Save a unction in text format

    \param[in] f     -  function to save
    \param[in] stream - stream to save it to
    \param[in] prec   - precision with which to save it

************************************************************/
void kernel_expansion_savetxt(const struct KernelExpansion * f,
                              FILE * stream, size_t prec)
{
    (void)(f);
    (void)(stream);
    (void)(prec);
    NOT_IMPLEMENTED_MSG("kernel_expansion_savetxt");
}


/********************************************************//**
    Load a function in text format

    \param[in] stream - stream to save it to

    \return kernel expansion
************************************************************/
struct KernelExpansion * kernel_expansion_loadtxt(FILE * stream)
{
    (void)(stream);
    NOT_IMPLEMENTED_MSG("kernel_expansion_loadtxt")
    return NULL;
}


