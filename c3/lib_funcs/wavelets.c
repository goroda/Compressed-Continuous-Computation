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
size_t wavelet_get_num_params(const struct Wavelets * wavelet)
{
    NOT_IMPLEMENTED_MSG("wavelet_get_num_params");
    return 0;
}

/********************************************************//**
*   Get lower bounds
*************************************************************/
double wavelet_get_lb(const struct Wavelets * wavelet)
{
    NOT_IMPLEMENTED_MSG("wavelet_get_lb");
    return 0;
}

/********************************************************//**
*   Get upper bounds
*************************************************************/
double wavelet_get_ub(const struct Wavelets * wavelet)
{
    NOT_IMPLEMENTED_MSG("wavelet_get_ub");
    return 0;
}


/********************************************************//**
*   Initialize Wavelets with parameters
*            
*   \param[in] opts    - approximation options
*   \param[in] nparams - number of parameters
*   \param[in] param   - parameters
*
*   \return p - Wavelets
*************************************************************/
struct Wavelets * 
wavelet_create_with_params(struct WaveletsOpts * opts,
                          size_t nparams, const double * param)
{

    NOT_IMPLEMENTED_MSG("wavelet_create_with_params");
    return NULL;
}

/********************************************************//**
*   Get parameters defining polynomial (for now just coefficients)
*************************************************************/
size_t wavelet_get_params(const struct Wavelets * wavelet, double * param)
{
    NOT_IMPLEMENTED_MSG("wavelet_get_params");
    return 0;
}

/********************************************************//**
*   Get parameters defining polynomial (for now just coefficients)
*************************************************************/
double * wavelet_get_params_ref(const struct Wavelets * wavelet, size_t *nparam)
{
    NOT_IMPLEMENTED_MSG("wavelet_get_params_res");
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
int wavelet_update_params(struct Wavelets * wavelet,
                         size_t nparams, const double * param)
{
    NOT_IMPLEMENTED_MSG("wavelet_update_params");
    return 0;
}

/********************************************************//**
*   Copy an Wavelets
*            
*   \param[in] pin
*
*   \return p - Wavelets
*************************************************************/
struct Wavelets * 
wavelet_copy(struct Wavelets * pin)
{
    NOT_IMPLEMENTED_MSG("wavelet_copy");
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
struct Wavelets * 
wavelet_zero(struct WaveletsOpts * opts, int force_nparam)
{
    NOT_IMPLEMENTED_MSG("wavelet_zero");
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
struct Wavelets * 
wavelet_constant(double a, struct WaveletsOpts * opts)
{
    NOT_IMPLEMENTED_MSG("wavelet_constant");
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
struct Wavelets * 
wavelet_linear(double a, double offset, struct WaveletsOpts * opts)
{
    NOT_IMPLEMENTED_MSG("wavelet_linear");
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
wavelet_linear_update(struct Wavelets * p, double a, double offset)
{
    assert (isnan(a) == 0);
    assert (isinf(a) == 0);
    assert (isnan(offset) == 0);
    assert (isinf(offset) == 0);

    NOT_IMPLEMENTED_MSG("wavelet_linear_update");
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
struct Wavelets * 
wavelet_quadratic(double a, double offset, struct WaveletsOpts * opts)
{
    assert (isnan(a) == 0);
    assert (isinf(a) == 0);
    assert (isnan(offset) == 0);
    assert (isinf(offset) == 0);
    NOT_IMPLEMENTED_MSG("wavelet_quadratic");
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
double wavelet_deriv_eval(const struct Wavelets * poly, double x)
{
    NOT_IMPLEMENTED_MSG("wavelet_deriv_eval");
    return 0.0;
}

/********************************************************//**
*   Evaluate the derivative of Wavelets
*
*   \param[in] p - 
*   
*   \return derivative
*
*************************************************************/
struct Wavelets * wavelet_deriv(struct Wavelets * p)
{
    NOT_IMPLEMENTED_MSG("wavelet_deriv");
    return NULL;
}

/********************************************************//**
*   free the memory of Wavelets
*
*   \param[in,out] p 
*************************************************************/
void wavelet_free(struct Wavelets * p)
{
    NOT_IMPLEMENTED_MSG("wavelet_free");
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
serialize_wavelet(unsigned char * ser, struct Wavelets * p, size_t * totSizeIn)
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
*   \return ptr - ser + number of bytes of Wavelets
*************************************************************/
unsigned char * 
deserialize_wavelet(unsigned char * ser, struct Wavelets ** poly)
{
    NOT_IMPLEMENTED_MSG("deserialize_wavelet");
    return NULL;
}

/********************************************************//**
    Save an Wavelets in text format

    \param[in] f      - function to save
    \param[in] stream - stream to save it to
    \param[in] prec   - precision with which to save it
************************************************************/
void wavelet_savetxt(const struct Wavelets * f,
                                 FILE * stream, size_t prec)
{
    assert (f != NULL);
    NOT_IMPLEMENTED_MSG("wavelet_savetxt");
}

/********************************************************//**
    Load an Wavelets in text format

    \param[in] stream - stream to save it to

    \return Wavelets
************************************************************/
struct Wavelets * wavelet_loadtxt(FILE * stream)
{
    NOT_IMPLEMENTED_MSG("wavelet_loadtxt");
    return NULL;
}

/********************************************************//**
*   Evaluate Wavelets 
*
*   \param[in] f - Wavelets
*   \param[in] x - location at which to evaluate
*
*   \return out - value
*************************************************************/
double wavelet_eval(const struct Wavelets * f, double x)
{
    NOT_IMPLEMENTED_MSG("wavelet_eval");
    return 0.0;
}

/********************************************************//**
*   Evaluate Wavelets
*
*   \param[in]     f    - Wavelets
*   \param[in]     N    - number of evaluations
*   \param[in]     x    - location at which to evaluate
*   \param[in]     incx - increment of x
*   \param[in,out] y    - allocated space for evaluations
*   \param[in]     incy - increment of y*
*
*   \note Currently just calls the single evaluation code
*         Note sure if this is optimal, cache-wise
*************************************************************/
void wavelet_evalN(const struct Wavelets *f, size_t N,
                  const double * x, size_t incx, double * y, size_t incy)
{
    for (size_t ii = 0; ii < N; ii++){
        y[ii*incy] = wavelet_eval(f,x[ii*incx]);
    }
}

/********************************************************//*
*   Evaluate the gradient of Wavelets with respect to the parameters
*
*   \param[in]     f    - Wavelets
*   \param[in]     nx   - number of x points
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N,nx)
*
*   \return 0 success, 1 otherwise
*************************************************************/
int wavelet_param_grad_eval(const struct Wavelets * f, size_t nx, const double * x, double * grad)
{
    NOT_IMPLEMENTED_MSG("wavelet_param_grad_eval");
    return 0;
}

/********************************************************//*
*   Evaluate the gradient of Wavelets with respect to the parameters
*
*   \param[in]     f    - function
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N,nx)
*
*   \return evaluation
*************************************************************/
double wavelet_param_grad_eval2(const struct Wavelets * f, double x, double * grad)
{
    NOT_IMPLEMENTED_MSG("wavelet_param_grad_eval2");
    return 0;
}

/********************************************************//**
    Take a gradient of the squared norm 
    with respect to its parameters, and add a scaled version
    of this gradient to *grad*

    \param[in]     f     - Wavelets
    \param[in]     scale - scaling for additional gradient
    \param[in,out] grad  - gradient, on output adds scale * new_grad

    \return  0 - success, 1 -failure

************************************************************/
int
wavelet_squared_norm_param_grad(const struct Wavelets * f,
                               double scale, double * grad)
{
    NOT_IMPLEMENTED_MSG("wavelet_squared_norm_param_grad");
    return 0;
}


/********************************************************//**
*  Approximate a function in Wavelets format
*  
*  \param[in,out] fa  -  approximated format
*  \param[in]     f    - wrapped function
*  \param[in]     opts - approximation options
*
*  \return 0 - no problems, > 0 problem

*************************************************************/
int
wavelet_approx(struct Wavelets * poly, struct Fwrap * f,
              const struct WaveletsOpts * opts)
{
    NOT_IMPLEMENTED_MSG("wavelet_approx");
    return 0;
}

/********************************************************//**
*   Compute the product of two Waveletss
*
*   \param[in] a - first 
*   \param[in] b - second 
*
*   \return c - Wavelets
*
*   \note 
*   Computes c(x) = a(x)b(x) where c is same form as a
*************************************************************/
struct Wavelets * wavelet_prod(const struct Wavelets * a, const struct Wavelets * b)
{
    NOT_IMPLEMENTED_MSG("wavelet_prod");
    return NULL;
}

/********************************************************//**
*   Compute the sum of the product between the functions in two arrays
*
*   \param[in] n   - number of functions
*   \param[in] lda - stride of first array
*   \param[in] a   - array of Waveletss
*   \param[in] ldb - stride of second array
*   \param[in] b   - array of Waveletss
*
*   \return c - Wavelets
*
*   \note 
*       All the functions need to have the same lower 
*       and upper bounds and be of the same type*
*************************************************************/
struct Wavelets *
wavelet_sum_prod(size_t n, size_t lda, 
                struct Wavelets ** a, size_t ldb,
                struct Wavelets ** b)
{
    NOT_IMPLEMENTED_MSG("wavelet_sum_prod");
    return NULL;
}

/********************************************************//**
*   Integrate an Wavelets
*
*   \param[in] f - Wavelets
*
*   \return out - Integral of the function
*************************************************************/
double
wavelet_integrate(const struct Wavelets * f)
{
    NOT_IMPLEMENTED_MSG("wavelet_integrate");
    return 0.0;
}

/********************************************************//**
*   Integrate a Wavelets with weighted measure
*
*   \param[in] f - function to integrate
*
*   \return out - Integral of approximation
*
    \note Computes  \f$ \int f(x) w(x) dx \f$ 
    in the qmarray
*************************************************************/
double
wavelet_integrate_weighted(const struct Wavelets * f)
{
    NOT_IMPLEMENTED_MSG("wavelet_integrate_weighted");
    return 0.0;
}


/********************************************************//**
*   Weighted inner product between two Waveletss
*
*   \param[in] a - first Wavelets
*   \param[in] b - second Wavelets
*
*   \return inner product
*
*   \note
*       Computes \f[ \int_{lb}^ub  a(x)b(x) w(x) dx \f]
*
*************************************************************/
double wavelet_inner_w(const struct Wavelets * a, const struct Wavelets * b)
{
    NOT_IMPLEMENTED_MSG("wavelet_inner_w");
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
wavelet_inner(const struct Wavelets * a, const struct Wavelets * b)
{
    NOT_IMPLEMENTED_MSG("wavelet_inner");
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
double wavelet_norm_w(const struct Wavelets * p)
{
    double out = sqrt(wavelet_inner_w(p,p));
    return sqrt(out);
}

/********************************************************//**
*   Compute the norm of Wavelets
*
*   \param[in] f - function
*
*   \return out - norm of function
*
*   \note
*        Computes \f$ \sqrt(\int_a^b f(x)^2 dx) \f$
*************************************************************/
double wavelet_norm(const struct Wavelets * f)
{

    double out = 0.0;
    out = sqrt(wavelet_inner(f,f));
    return out;
}

/********************************************************//**
*   Multiply Wavelets by -1
*************************************************************/
void  wavelet_flip_sign(struct Wavelets * f)
{   
    NOT_IMPLEMENTED_MSG("wavelet_flip_sign");
}

/********************************************************//**
*   Multiply by scalar and overwrite expansion
*
*   \param[in] a - scaling factor
*   \param[in] f - Wavelets to scale
*************************************************************/
void wavelet_scale(double a, struct Wavelets * f)
{
    NOT_IMPLEMENTED_MSG("wavelet_scale");
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
int wavelet_axpy(double a, struct Wavelets * x, struct Wavelets * y)
{
    NOT_IMPLEMENTED_MSG("wavelet_axpy");
    return 0;
}

////////////////////////////////////////////////////////////////////////////
// Algorithms


/********************************************************//**
*   Obtain the maximum of Wavelets
*
*   \param[in]     f - Wavelets
*   \param[in,out] x - location of maximum value
*
*   \return maxval - maximum value
*************************************************************/
double wavelet_max(struct Wavelets * f, double * x)
{
    NOT_IMPLEMENTED_MSG("wavelet_max");
    return 0;
}

/********************************************************//**
*   Obtain the minimum of Wavelets
*
*   \param[in]     f - Wavelets
*   \param[in,out] x - location of maximum value
*
*   \return val - min value
*************************************************************/
double wavelet_min(struct Wavelets * f, double * x)
{
    NOT_IMPLEMENTED_MSG("wavelet_min");
    return 0;
}


/********************************************************//**
*   Obtain the maximum in absolute value of Wavelets
*
*   \param[in]     f     - Wavelets
*   \param[in,out] x     - location of maximum
*   \param[in]     oargs - optimization arguments 
*                          required for HERMITE otherwise can set NULL
*
*   \return maxval : maximum value (absolute value)
*
*************************************************************/
double wavelet_absmax(struct Wavelets * f, double * x, void * oargs)
{
    NOT_IMPLEMENTED_MSG("wavelet_absmax");
    return 0;
}


/////////////////////////////////////////////////////////
// Utilities
void print_wavelet(struct Wavelets * p, size_t prec, 
                  void * args, FILE *fp)
{
   NOT_IMPLEMENTED_MSG("print_wavelet");
}
