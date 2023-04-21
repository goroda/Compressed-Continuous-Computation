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


/** \file Wavelets.c
 * Provides routines for manipulating Wavelets
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <assert.h>

#include "futil.h"
#include "stringmanip.h"
#include "array.h"
#include "polynomials.h"
#include "hpoly.h"
#include "lib_quadrature.h"
#include "linalg.h"
#include "legtens.h"

#define ZEROTHRESH  1e0 * DBL_EPSILON

// replace in this file Wavelets with the name of the approximation
// and wavelet with a lowercase name

struct WaveletsOpts{

    /* Fill In */
};

struct WaveletsOpts * wavelet_opts_alloc(enum poly_type ptype)
{
    NOT_IMPLEMENTED_MSG("wavelet_opts_alloc");
    return NULL;
}

void wavelet_opts_free(struct WaveletsOpts * wavelet)
{
    NOT_IMPLEMENTED_MSG("wavelet_opts_free");
}

void wavelet_opts_free_deep(struct WaveletsOpts ** wavelet)
{
    NOT_IMPLEMENTED_MSG("wavelet_opts_free_deep");
}

void wavelet_opts_set_start(struct WaveletsOpts * wavelet, size_t start)
{
    NOT_IMPLEMENTED_MSG("wavelet_opts_set_start");
}

void wavelet_opts_set_maxnum(struct WaveletsOpts * wavelet, size_t maxnum)
{
    NOT_IMPLEMENTED_MSG("wavelet_opts_set_maxnum");
}

size_t wavelet_opts_get_maxnum(const struct WaveletsOpts * wavelet)
{
    NOT_IMPLEMENTED_MSG("wavelet_opts_get_maxnum");
    return 0;
}

void wavelet_opts_set_coeffs_check(struct WaveletsOpts * wavelet, size_t num)
{
    NOT_IMPLEMENTED_MSG("wavelet_opts_set_coeffs_check");
}

void wavelet_opts_set_tol(struct WaveletsOpts * wavelet, double tol)
{
    NOT_IMPLEMENTED_MSG("wavelet_opts_set_tol");    
}

void wavelet_opts_set_lb(struct WaveletsOpts * wavelet, double lb)
{
    NOT_IMPLEMENTED_MSG("wavelet_opts_set_lb");    
}

double wavelet_opts_get_lb(const struct WaveletsOpts * wavelet)
{
    NOT_IMPLEMENTED_MSG("wavelet_opts_get_lb");
    return 0;
}

void wavelet_opts_set_ub(struct WaveletsOpts * wavelet, double ub)
{
    NOT_IMPLEMENTED_MSG("wavelet_opts_set_ub");    
}

double wavelet_opts_get_ub(const struct WaveletsOpts * wavelet)
{
    NOT_IMPLEMENTED_MSG("wavelet_opts_get_ub");
    return 0;
}

/********************************************************//**
*   Get number of free parameters
*************************************************************/
size_t wavelet_opts_get_nparams(const struct WaveletsOpts * opts)
{
    NOT_IMPLEMENTED_MSG("wavelet_opts_get_nparams");
    return 0;    
}

/********************************************************//**
*   Set number of free parameters
*************************************************************/
void wavelet_opts_set_nparams(struct WaveletsOpts * opts, size_t num)
{
    NOT_IMPLEMENTED_MSG("wavelet_opts_set_nparams");
}


/********************************************************//**
*   Get number of parameters
*************************************************************/
size_t wavelet_expansion_get_num_params(const struct WaveletExpansion * wavelet)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_get_num_params");
    return 0;
}

/********************************************************//**
*   Get lower bounds
*************************************************************/
double wavelet_expansion_get_lb(const struct WaveletExpansion * wavelet)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_get_lb");
    return 0;
}

/********************************************************//**
*   Get upper bounds
*************************************************************/
double wavelet_expansion_get_ub(const struct WaveletExpansion * wavelet)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_get_ub");
    return 0;
}


/********************************************************//**
*   Initialize WaveletExpansion with parameters
*            
*   \param[in] opts    - approximation options
*   \param[in] nparams - number of parameters
*   \param[in] param   - parameters
*
*   \return p - WaveletExpansion
*************************************************************/
struct WaveletExpansion * 
wavelet_expansion_create_with_params(struct WaveletExpansionOpts * opts,
                          size_t nparams, const double * param)
{

    NOT_IMPLEMENTED_MSG("wavelet_expansion_create_with_params");
    return NULL;
}

/********************************************************//**
*   Get parameters defining polynomial (for now just coefficients)
*************************************************************/
size_t wavelet_expansion_get_params(const struct WaveletExpansion * wavelet, double * param)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_get_params");
    return 0;
}

/********************************************************//**
*   Get parameters defining polynomial (for now just coefficients)
*************************************************************/
double * wavelet_expansion_get_params_ref(const struct WaveletExpansion * wavelet, size_t *nparam)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_get_params_res");
    return NULL;
}

/********************************************************//**
*   Update an expansion's parameters
*            
*   \param[in] wavelet     - expansion to update
*   \param[in] nparams - number of polynomials
*   \param[in] param   - parameters

*   \returns 0 if successful
*************************************************************/
int wavelet_expansion_update_params(struct WaveletExpansion * wavelet,
                         size_t nparams, const double * param)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_update_params");
    return 0;
}

/********************************************************//**
*   Copy an WaveletExpansion
*            
*   \param[in] pin
*
*   \return p - WaveletExpansion
*************************************************************/
struct WaveletExpansion * 
wavelet_expansion_copy(struct WaveletExpansion * pin)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_copy");
    return NULL;

}

/********************************************************//**
    Return a zero function

    \param[in] opts         - extra arguments depending on function_class, sub_type, etc.
    \param[in] force_nparam - if == 1 then approximation will have the number of parameters
                                      defined by *get_nparams, for each approximation type
                              if == 0 then it may be more compressed

    \return p - zero function
************************************************************/
struct WaveletExpansion * 
wavelet_expansion_zero(struct WaveletExpansionOpts * opts, int force_nparam)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_zero");
    return NULL;
}


/********************************************************//**
*   Generate a constant function
*
*   \param[in] a    - value of the function
*   \param[in] opts - opts
*
*   \return p 
*************************************************************/
struct WaveletExpansion * 
wavelet_expansion_constant(double a, struct WaveletExpansionOpts * opts)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_constant");
    return NULL;
}

/********************************************************//**
*   Generate a linear function
*
*   \param[in] a      - value of the slwavelet function
*   \param[in] offset - offset
*   \param[in] opts   - options
*
*   \return p - orthogonal polynomial expansion
*************************************************************/
struct WaveletExpansion * 
wavelet_expansion_linear(double a, double offset, struct WaveletExpansionOpts * opts)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_linear");
    return NULL;
}

/********************************************************//**
*   Update a linear orthonormal polynomial expansion
*
*   \param[in] p      - existing linear function
*   \param[in] a      - value of the slwavelet function
*   \param[in] offset - offset
*
*   \return 0 if succesfull, 1 otherwise
*
*************************************************************/
int
wavelet_expansion_linear_update(struct WaveletExpansion * p, double a, double offset)
{
    assert (isnan(a) == 0);
    assert (isinf(a) == 0);
    assert (isnan(offset) == 0);
    assert (isinf(offset) == 0);

    NOT_IMPLEMENTED_MSG("wavelet_expansion_linear_update");
    return NULL;
}

/********************************************************//**
*   Generate a quadratic orthonormal polynomial expansion
    a * (x-offset)^2
*
*   \param[in] a      - value of the slwavelet function
*   \param[in] offset - offset
*   \param[in] opts   - options
*
*   \return quadratic polynomial
*************************************************************/
struct WaveletExpansion * 
wavelet_expansion_quadratic(double a, double offset, struct WaveletExpansionOpts * opts)
{
    assert (isnan(a) == 0);
    assert (isinf(a) == 0);
    assert (isnan(offset) == 0);
    assert (isinf(offset) == 0);
    NOT_IMPLEMENTED_MSG("wavelet_expansion_quadratic");
    return NULL;
}


/********************************************************//**
*   Evaluate the derivative of an orthogonal polynomial expansion
*
*   \param[in] poly - pointer to orth poly expansion
*   \param[in] x    - location at which to evaluate
*
*
*   \return out - value of derivative
*************************************************************/
double wavelet_expansion_deriv_eval(const struct WaveletExpansion * poly, double x)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_deriv_eval");
    return 0.0;
}

/********************************************************//**
*   Evaluate the derivative of WaveletExpansion
*
*   \param[in] p - 
*   
*   \return derivative
*
*************************************************************/
struct WaveletExpansion * wavelet_expansion_deriv(struct WaveletExpansion * p)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_deriv");
    return NULL;
}

/********************************************************//**
*   free the memory of WaveletExpansion
*
*   \param[in,out] p 
*************************************************************/
void wavelet_expansion_free(struct WaveletExpansion * p)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_free");
}

/********************************************************//**
*   Serialize wavelet
*   
*   \param[in] ser       - location to which to serialize
*   \param[in] p         - polynomial
*   \param[in] totSizeIn - if not null then only return total size of 
*                          array without serialization! if NULL then serialiaze
*
*   \return ptr : pointer to end of serialization
*************************************************************/
unsigned char *
serialize_wavelet(unsigned char * ser, struct WaveletExpansion * p, size_t * totSizeIn)
{
    // order is  ptype->lower_bound->upper_bound->orth_poly->coeff
    NOT_IMPLEMENTED_MSG("serialize_wavelet");
    return NULL;
}

/********************************************************//**
*   Deserialize wavelet
*
*   \param[in]     ser  - input string
*   \param[in,out] poly - poly expansion
*
*   \return ptr - ser + number of bytes of WaveletExpansion
*************************************************************/
unsigned char * 
deserialize_wavelet(unsigned char * ser, struct WaveletExpansion ** poly)
{
    NOT_IMPLEMENTED_MSG("deserialize_wavelet");
    return NULL;
}

/********************************************************//**
    Save an WaveletExpansion in text format

    \param[in] f      - function to save
    \param[in] stream - stream to save it to
    \param[in] prec   - precision with which to save it
************************************************************/
void wavelet_expansion_savetxt(const struct WaveletExpansion * f,
                                 FILE * stream, size_t prec)
{
    assert (f != NULL);
    NOT_IMPLEMENTED_MSG("wavelet_expansion_savetxt");
}

/********************************************************//**
    Load an WaveletExpansion in text format

    \param[in] stream - stream to save it to

    \return WaveletExpansion
************************************************************/
struct WaveletExpansion * wavelet_expansion_loadtxt(FILE * stream)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_loadtxt");
    return NULL;
}

/********************************************************//**
*   Evaluate WaveletExpansion 
*
*   \param[in] f - WaveletExpansion
*   \param[in] x - location at which to evaluate
*
*   \return out - value
*************************************************************/
double wavelet_expansion_eval(const struct WaveletExpansion * f, double x)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_eval");
    return 0.0;
}

/********************************************************//**
*   Evaluate WaveletExpansion
*
*   \param[in]     f    - WaveletExpansion
*   \param[in]     N    - number of evaluations
*   \param[in]     x    - location at which to evaluate
*   \param[in]     incx - increment of x
*   \param[in,out] y    - allocated space for evaluations
*   \param[in]     incy - increment of y*
*
*   \note Currently just calls the single evaluation code
*         Note sure if this is optimal, cache-wise
*************************************************************/
void wavelet_expansion_evalN(const struct WaveletExpansion *f, size_t N,
                  const double * x, size_t incx, double * y, size_t incy)
{
    for (size_t ii = 0; ii < N; ii++){
        y[ii*incy] = wavelet_expansion_eval(f,x[ii*incx]);
    }
}

/********************************************************//*
*   Evaluate the gradient of WaveletExpansion with respect to the parameters
*
*   \param[in]     f    - WaveletExpansion
*   \param[in]     nx   - number of x points
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N,nx)
*
*   \return 0 success, 1 otherwise
*************************************************************/
int wavelet_expansion_param_grad_eval(const struct WaveletExpansion * f, size_t nx, const double * x, double * grad)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_param_grad_eval");
    return 0;
}

/********************************************************//*
*   Evaluate the gradient of WaveletExpansion with respect to the parameters
*
*   \param[in]     f    - function
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N,nx)
*
*   \return evaluation
*************************************************************/
double wavelet_expansion_param_grad_eval2(const struct WaveletExpansion * f, double x, double * grad)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_param_grad_eval2");
    return 0;
}

/********************************************************//**
    Take a gradient of the squared norm 
    with respect to its parameters, and add a scaled version
    of this gradient to *grad*

    \param[in]     f     - WaveletExpansion
    \param[in]     scale - scaling for additional gradient
    \param[in,out] grad  - gradient, on output adds scale * new_grad

    \return  0 - success, 1 -failure

************************************************************/
int
wavelet_expansion_squared_norm_param_grad(const struct WaveletExpansion * f,
                               double scale, double * grad)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_squared_norm_param_grad");
    return 0;
}


/********************************************************//**
*  Approximate a function in WaveletExpansion format
*  
*  \param[in,out] fa  -  approximated format
*  \param[in]     f    - wrapped function
*  \param[in]     opts - approximation options
*
*  \return 0 - no problems, > 0 problem

*************************************************************/
int
wavelet_expansion_approx(struct WaveletExpansion * poly, struct Fwrap * f,
              const struct WaveletExpansionOpts * opts)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_approx");
    return 0;
}

/********************************************************//**
*   Compute the product of two WaveletExpansions
*
*   \param[in] a - first 
*   \param[in] b - second 
*
*   \return c - WaveletExpansion
*
*   \note 
*   Computes c(x) = a(x)b(x) where c is same form as a
*************************************************************/
struct WaveletExpansion * wavelet_expansion_prod(const struct WaveletExpansion * a, const struct WaveletExpansion * b)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_prod");
    return NULL;
}

/********************************************************//**
*   Compute the sum of the product between the functions in two arrays
*
*   \param[in] n   - number of functions
*   \param[in] lda - stride of first array
*   \param[in] a   - array of WaveletExpansions
*   \param[in] ldb - stride of second array
*   \param[in] b   - array of WaveletExpansions
*
*   \return c - WaveletExpansion
*
*   \note 
*       All the functions need to have the same lower 
*       and upper bounds and be of the same type*
*************************************************************/
struct WaveletExpansion *
wavelet_expansion_sum_prod(size_t n, size_t lda, 
                struct WaveletExpansion ** a, size_t ldb,
                struct WaveletExpansion ** b)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_sum_prod");
    return NULL;
}

/********************************************************//**
*   Integrate an WaveletExpansion
*
*   \param[in] f - WaveletExpansion
*
*   \return out - Integral of the function
*************************************************************/
double
wavelet_expansion_integrate(const struct WaveletExpansion * f)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_integrate");
    return 0.0;
}

/********************************************************//**
*   Integrate a WaveletExpansion with weighted measure
*
*   \param[in] f - function to integrate
*
*   \return out - Integral of approximation
*
    \note Computes  \f$ \int f(x) w(x) dx \f$ 
    in the qmarray
*************************************************************/
double
wavelet_expansion_integrate_weighted(const struct WaveletExpansion * f)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_integrate_weighted");
    return 0.0;
}


/********************************************************//**
*   Weighted inner product between two WaveletExpansions
*
*   \param[in] a - first WaveletExpansion
*   \param[in] b - second WaveletExpansion
*
*   \return inner product
*
*   \note
*       Computes \f[ \int_{lb}^ub  a(x)b(x) w(x) dx \f]
*
*************************************************************/
double wavelet_expansion_inner_w(const struct WaveletExpansion * a, const struct WaveletExpansion * b)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_inner_w");
    return 0.0;
}

/********************************************************//**
*   Inner product between two polynomial expansions of the same type
*
*   \param[in] a - first polynomial
*   \param[in] b - second polynomai
*
*   \return  inner product
*
*   \note
*   If the polynomial is NOT HERMITE then
*   Computes  \f$ \int_{lb}^ub  a(x)b(x) dx \f$ by first
*   converting each polynomial to a Legendre polynomial
*   Otherwise it computes the error with respect to gaussia weight
*************************************************************/
double
wavelet_expansion_inner(const struct WaveletExpansion * a, const struct WaveletExpansion * b)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_inner");
    return 0.0;
}

/********************************************************//**
*   Compute the norm of an orthogonal polynomial
*   expansion with respect to family weighting 
*   function
*
*   \param[in] p - polynomial to integrate
*
*   \return out - norm of function
*
*   \note
*        Computes  \f$ \sqrt(\int f(x)^2 w(x) dx) \f$
*************************************************************/
double wavelet_expansion_norm_w(const struct WaveletExpansion * p)
{
    double out = sqrt(wavelet_expansion_inner_w(p,p));
    return sqrt(out);
}

/********************************************************//**
*   Compute the norm of WaveletExpansion
*
*   \param[in] f - function
*
*   \return out - norm of function
*
*   \note
*        Computes \f$ \sqrt(\int_a^b f(x)^2 dx) \f$
*************************************************************/
double wavelet_expansion_norm(const struct WaveletExpansion * f)
{

    double out = 0.0;
    out = sqrt(wavelet_expansion_inner(f,f));
    return out;
}

/********************************************************//**
*   Multiply WaveletExpansion by -1
*************************************************************/
void  wavelet_expansion_flip_sign(struct WaveletExpansion * f)
{   
    NOT_IMPLEMENTED_MSG("wavelet_expansion_flip_sign");
}

/********************************************************//**
*   Multiply by scalar and overwrite expansion
*
*   \param[in] a - scaling factor
*   \param[in] f - WaveletExpansion to scale
*************************************************************/
void wavelet_expansion_scale(double a, struct WaveletExpansion * f)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_scale");
}

/********************************************************//**
*   Multiply by scalar and add two orthgonal 
*   expansions of the same family together \f[ y \leftarrow ax + y \f]
*
*   \param[in] a  - scaling factor
*   \param[in] x  - first
*   \param[in] y  - second

*   \return 0 if successfull 1 if error with allocating more space for y
*
**************************************************************/
int wavelet_expansion_axpy(double a, struct WaveletExpansion * x, struct WaveletExpansion * y)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_axpy");
    return 0;
}

////////////////////////////////////////////////////////////////////////////
// Algorithms


/********************************************************//**
*   Obtain the maximum of WaveletExpansion
*
*   \param[in]     f - WaveletExpansion
*   \param[in,out] x - location of maximum value
*
*   \return maxval - maximum value
*************************************************************/
double wavelet_expansion_max(struct WaveletExpansion * f, double * x)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_max");
    return 0;
}

/********************************************************//**
*   Obtain the minimum of WaveletExpansion
*
*   \param[in]     f - WaveletExpansion
*   \param[in,out] x - location of maximum value
*
*   \return val - min value
*************************************************************/
double wavelet_expansion_min(struct WaveletExpansion * f, double * x)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_min");
    return 0;
}


/********************************************************//**
*   Obtain the maximum in absolute value of WaveletExpansion
*
*   \param[in]     f     - WaveletExpansion
*   \param[in,out] x     - location of maximum
*   \param[in]     oargs - optimization arguments 
*                          required for HERMITE otherwise can set NULL
*
*   \return maxval : maximum value (absolute value)
*
*************************************************************/
double wavelet_expansion_absmax(struct WaveletExpansion * f, double * x, void * oargs)
{
    NOT_IMPLEMENTED_MSG("wavelet_expansion_absmax");
    return 0;
}


/////////////////////////////////////////////////////////
// Utilities
void print_wavelet(struct WaveletExpansion * p, size_t prec, 
                  void * args, FILE *fp)
{
   NOT_IMPLEMENTED_MSG("print_wavelet");
}
