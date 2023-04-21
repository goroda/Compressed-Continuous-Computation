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


/** \file NAME.c
 * Provides routines for manipulating NAME
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <assert.h>

#include "stringmanip.h"
#include "array.h"
#include "polynomials.h"
#include "hpoly.h"
#include "lib_quadrature.h"
#include "linalg.h"
#include "legtens.h"

#define ZEROTHRESH  1e0 * DBL_EPSILON

// replace in this file NAME with the name of the approximation
// and NLOWER with a lowercase name

struct NAMEOpts{

    /* Fill In */
};

struct NAMEOpts * NLOWER_opts_alloc(enum poly_type ptype)
{
    NOT_IMPLEMENTED_MSG("NLOWER_opts_alloc");
    return NULL;
}

void NLOWER_opts_free(struct NAMEOpts * NLOWER)
{
    NOT_IMPLEMENTED_MSG("NLOWER_opts_free");
}

void NLOWER_opts_free_deep(struct NAMEOpts ** NLOWER)
{
    NOT_IMPLEMENTED_MSG("NLOWER_opts_free_deep");
}

void NLOWER_opts_set_start(struct NAMEOpts * NLOWER, size_t start)
{
    NOT_IMPLEMENTED_MSG("NLOWER_opts_set_start");
}

void NLOWER_opts_set_maxnum(struct NAMEOpts * NLOWER, size_t maxnum)
{
    NOT_IMPLEMENTED_MSG("NLOWER_opts_set_maxnum");
}

size_t NLOWER_opts_get_maxnum(const struct NAMEOpts * NLOWER)
{
    NOT_IMPLEMENTED_MSG("NLOWER_opts_get_maxnum");
    return 0;
}

void NLOWER_opts_set_coeffs_check(struct NAMEOpts * NLOWER, size_t num)
{
    NOT_IMPLEMENTED_MSG("NLOWER_opts_set_coeffs_check");
}

void NLOWER_opts_set_tol(struct NAMEOpts * NLOWER, double tol)
{
    NOT_IMPLEMENTED_MSG("NLOWER_opts_set_tol");    
}

void NLOWER_opts_set_lb(struct NAMEOpts * NLOWER, double lb)
{
    NOT_IMPLEMENTED_MSG("NLOWER_opts_set_lb");    
}

double NLOWER_opts_get_lb(const struct NAMEOpts * NLOWER)
{
    NOT_IMPLEMENTED_MSG("NLOWER_opts_get_lb");
    return 0
}

void NLOWER_opts_set_ub(struct NAMEOpts * NLOWER, double ub)
{
    NOT_IMPLEMENTED_MSG("NLOWER_opts_set_ub");    
}

double NLOWER_opts_get_ub(const struct NAMEOpts * NLOWER)
{
    NOT_IMPLEMENTED_MSG("NLOWER_opts_get_ub");
    return 0;
}

/********************************************************//**
*   Get number of free parameters
*************************************************************/
size_t NLOWER_opts_get_nparams(const struct NAMEOpts * opts)
{
    NOT_IMPLEMENTED_MSG("NLOWER_opts_get_nparams");
    return 0;    
}

/********************************************************//**
*   Set number of free parameters
*************************************************************/
void NLOWER_opts_set_nparams(struct NAMEOpts * opts, size_t num)
{
    NOT_IMPLEMENTED_MSG("NLOWER_opts_set_nparams");
}


/********************************************************//**
*   Get number of parameters
*************************************************************/
size_t NLOWER_get_num_params(const struct NAME * NLOWER)
{
    NOT_IMPLEMENTED_MSG("NLOWER_get_num_params");
    return 0;
}

/********************************************************//**
*   Get lower bounds
*************************************************************/
double NLOWER_get_lb(const struct NAME * NLOWER)
{
    NOT_IMPLEMENTED_MSG("NLOWER_get_lb");
    return 0;
}

/********************************************************//**
*   Get upper bounds
*************************************************************/
double NLOWER_get_ub(const struct NAME * NLOWER)
{
    NOT_IMPLEMENTED_MSG("NLOWER_get_ub");
    return 0;
}


/********************************************************//**
*   Initialize NAME with parameters
*            
*   \param[in] opts    - approximation options
*   \param[in] nparams - number of parameters
*   \param[in] param   - parameters
*
*   \return p - NAME
*************************************************************/
struct NAME * 
NLOWER_create_with_params(struct NAMEOpts * opts,
                          size_t nparams, const double * param)
{

    NOT_IMPLEMENTED_MSG("NLOWER_create_with_params");
    return NULL;
}

/********************************************************//**
*   Get parameters defining polynomial (for now just coefficients)
*************************************************************/
size_t NLOWER_get_params(const struct NAME * NLOWER, double * param)
{
    NOT_IMPLEMENTED_MSG("NLOWER_get_params");
    return 0;
}

/********************************************************//**
*   Get parameters defining polynomial (for now just coefficients)
*************************************************************/
double * NLOWER_get_params_ref(const struct NAME * NLOWER, size_t *nparam)
{
    NOT_IMPLEMENTED_MSG("NLOWER_get_params_res");
    return NULL;
}

/********************************************************//**
*   Update an expansion's parameters
*            
*   \param[in] NLOWER     - expansion to update
*   \param[in] nparams - number of polynomials
*   \param[in] param   - parameters

*   \returns 0 if successful
*************************************************************/
int NLOWER_update_params(struct NAME * NLOWER,
                         size_t nparams, const double * param)
{
    NOT_IMPLEMENTED_MSG("NLOWER_update_params");
    return 0;
}

/********************************************************//**
*   Copy an NAME
*            
*   \param[in] pin
*
*   \return p - NAME
*************************************************************/
struct NAME * 
NLOWER_copy(struct NAME * pin)
{
    NOT_IMPLEMENTED_MSG("NLOWER_copy");
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
struct NAME * 
NLOWER_zero(struct NAMEOpts * opts, int force_nparam)
{
    NOT_IMPLEMENTED_MSG("NLOWER_zero");
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
struct NAME * 
NLOWER_constant(double a, struct NAMEOpts * opts)
{
    NOT_IMPLEMENTED_MSG("NLOWER_constant");
    return NULL;
}

/********************************************************//**
*   Generate a linear function
*
*   \param[in] a      - value of the slNLOWER function
*   \param[in] offset - offset
*   \param[in] opts   - options
*
*   \return p - orthogonal polynomial expansion
*************************************************************/
struct NAME * 
NLOWER_linear(double a, double offset, struct NAMEOpts * opts)
{
    NOT_IMPLEMENTED_MSG("NLOWER_linear");
    return NULL;
}

/********************************************************//**
*   Update a linear orthonormal polynomial expansion
*
*   \param[in] p      - existing linear function
*   \param[in] a      - value of the slNLOWER function
*   \param[in] offset - offset
*
*   \return 0 if succesfull, 1 otherwise
*
*************************************************************/
int
NLOWER_linear_update(struct NAME * p, double a, double offset)
{
    assert (isnan(a) == 0);
    assert (isinf(a) == 0);
    assert (isnan(offset) == 0);
    assert (isinf(offset) == 0);

    NOT_IMPLEMENTED_MSG("NLOWER_linear_update");
    return NULL;
}

/********************************************************//**
*   Generate a quadratic orthonormal polynomial expansion
    a * (x-offset)^2
*
*   \param[in] a      - value of the slNLOWER function
*   \param[in] offset - offset
*   \param[in] opts   - options
*
*   \return quadratic polynomial
*************************************************************/
struct NAME * 
NLOWER_quadratic(double a, double offset, struct NAMEOpts * opts)
{
    assert (isnan(a) == 0);
    assert (isinf(a) == 0);
    assert (isnan(offset) == 0);
    assert (isinf(offset) == 0);
    NOT_IMPLEMENTED_MSG("NLOWER_quadratic");
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
double NLOWER_deriv_eval(const struct NAME * poly, double x)
{
    NOT_IMPLEMENTED_MSG("NLOWER_deriv_eval");
    return 0.0;
}

/********************************************************//**
*   Evaluate the derivative of NAME
*
*   \param[in] p - 
*   
*   \return derivative
*
*************************************************************/
struct NAME * NLOWER_deriv(struct NAME * p)
{
    NOT_IMPLEMENTED_MSG("NLOWER_deriv");
    return NULL;
}

/********************************************************//**
*   free the memory of NAME
*
*   \param[in,out] p 
*************************************************************/
void NLOWER_free(struct NAME * p)
{
    NOT_IMPLEMENTED_MSG("NLOWER_free");
}

/********************************************************//**
*   Serialize NLOWER
*   
*   \param[in] ser       - location to which to serialize
*   \param[in] p         - polynomial
*   \param[in] totSizeIn - if not null then only return total size of 
*                          array without serialization! if NULL then serialiaze
*
*   \return ptr : pointer to end of serialization
*************************************************************/
unsigned char *
serialize_NLOWER(unsigned char * ser, struct NAME * p, size_t * totSizeIn)
{
    // order is  ptype->lower_bound->upper_bound->orth_poly->coeff
    NOT_IMPLEMENTED_MSG("serialize_NLOWER");
    return NULL;
}

/********************************************************//**
*   Deserialize NLOWER
*
*   \param[in]     ser  - input string
*   \param[in,out] poly - poly expansion
*
*   \return ptr - ser + number of bytes of NAME
*************************************************************/
unsigned char * 
deserialize_NLOWER(unsigned char * ser, struct NAME ** poly)
{
    NOT_IMPLEMENTED_MSG("deserialize_NLOWER");
    return NULL;
}

/********************************************************//**
    Save an NAME in text format

    \param[in] f      - function to save
    \param[in] stream - stream to save it to
    \param[in] prec   - precision with which to save it
************************************************************/
void NLOWER_savetxt(const struct NAME * f,
                                 FILE * stream, size_t prec)
{
    assert (f != NULL);
    NOT_IMPLEMENTED_MSG("NLOWER_savetxt");
}

/********************************************************//**
    Load an NAME in text format

    \param[in] stream - stream to save it to

    \return NAME
************************************************************/
struct NAME * NLOWER_loadtxt(FILE * stream)
{
    NOT_IMPLEMENTED_MSG("NLOWER_loadtxt");
    return NULL;
}

/********************************************************//**
*   Evaluate NAME 
*
*   \param[in] f - NAME
*   \param[in] x - location at which to evaluate
*
*   \return out - value
*************************************************************/
double NLOWER_eval(const struct NAME * f, double x)
{
    NOT_IMPLEMENTED_MSG("NLOWER_eval");
    return NULL;
}
/********************************************************//**
*   Evaluate NAME
*
*   \param[in]     f    - NAME
*   \param[in]     N    - number of evaluations
*   \param[in]     x    - location at which to evaluate
*   \param[in]     incx - increment of x
*   \param[in,out] y    - allocated space for evaluations
*   \param[in]     incy - increment of y*
*
*   \note Currently just calls the single evaluation code
*         Note sure if this is optimal, cache-wise
*************************************************************/
void NLOWER_evalN(const struct NAME *f, size_t N,
                  const double * x, size_t incx, double * y, size_t incy)
{
    for (size_t ii = 0; ii < N; ii++){
        y[ii*incy] = NLOWER_eval(f,x[ii*incx]);
    }
}

/********************************************************//*
*   Evaluate the gradient of NAME with respect to the parameters
*
*   \param[in]     f    - NAME
*   \param[in]     nx   - number of x points
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N,nx)
*
*   \return 0 success, 1 otherwise
*************************************************************/
int NLOWER_param_grad_eval(const struct NAME * f, size_t nx, const double * x, double * grad)
{
    NOT_IMPLEMENTED_MSG("NLOWER_param_grad_eval");
    return 0;
}

/********************************************************//*
*   Evaluate the gradient of NAME with respect to the parameters
*
*   \param[in]     f    - function
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N,nx)
*
*   \return evaluation
*************************************************************/
double NLOWER_param_grad_eval2(const struct NAME * f, double x, double * grad)
{
    NOT_IMPLEMENTED_MSG("NLOWER_param_grad_eval2");
    return 0;
}

/********************************************************//**
    Take a gradient of the squared norm 
    with respect to its parameters, and add a scaled version
    of this gradient to *grad*

    \param[in]     f     - NAME
    \param[in]     scale - scaling for additional gradient
    \param[in,out] grad  - gradient, on output adds scale * new_grad

    \return  0 - success, 1 -failure

************************************************************/
int
NLOWER_squared_norm_param_grad(const struct NAME * f,
                               double scale, double * grad)
{
    NOT_IMPLEMENTED_MSG("NLOWER_squared_norm_param_grad");
    return 0;
}


/********************************************************//**
*  Approximate a function in NAME format
*  
*  \param[in,out] fa  -  approximated format
*  \param[in]     f    - wrapped function
*  \param[in]     opts - approximation options
*
*  \return 0 - no problems, > 0 problem

*************************************************************/
int
NLOWER_approx(struct NAME * poly, struct Fwrap * f,
              const struct NAMEOpts * opts)
{
    NOT_IMPLEMENTED_MSG("NLOWER_approx");
    return 0;
}

/********************************************************//**
*   Compute the product of two NAMEs
*
*   \param[in] a - first 
*   \param[in] b - second 
*
*   \return c - NAME
*
*   \note 
*   Computes c(x) = a(x)b(x) where c is same form as a
*************************************************************/
struct NAME * NLOWER_prod(const struct NAME * a, const struct NAME * b)
{
    NOT_IMPLEMENTED_MSG("NLOWER_prod");
    return NULL;
}

/********************************************************//**
*   Compute the sum of the product between the functions in two arrays
*
*   \param[in] n   - number of functions
*   \param[in] lda - stride of first array
*   \param[in] a   - array of NAMEs
*   \param[in] ldb - stride of second array
*   \param[in] b   - array of NAMEs
*
*   \return c - NAME
*
*   \note 
*       All the functions need to have the same lower 
*       and upper bounds and be of the same type*
*************************************************************/
struct NAME *
NLOWER_sum_prod(size_t n, size_t lda, 
                struct NAME ** a, size_t ldb,
                struct NAME ** b)
{
    NOT_IMPLEMENTED_MSG("NLOWER_sum_prod");
    return NULL;
}

/********************************************************//**
*   Integrate an NAME
*
*   \param[in] f - NAME
*
*   \return out - Integral of the function
*************************************************************/
double
NLOWER_integrate(const struct NAME * f)
{
    NOT_IMPLEMENTED_MSG("NLOWER_integrate");
    return 0.0;
}

/********************************************************//**
*   Integrate a NAME with weighted measure
*
*   \param[in] f - function to integrate
*
*   \return out - Integral of approximation
*
    \note Computes  \f$ \int f(x) w(x) dx \f$ 
    in the qmarray
*************************************************************/
double
NLOWER_integrate_weighted(const struct NAME * f)
{
    NOT_IMPLEMENTED_MSG("NLOWER_integrate_weighted");
    return 0.0;
}


/********************************************************//**
*   Weighted inner product between two NAMEs
*
*   \param[in] a - first NAME
*   \param[in] b - second NAME
*
*   \return inner product
*
*   \note
*       Computes \f[ \int_{lb}^ub  a(x)b(x) w(x) dx \f]
*
*************************************************************/
double NLOWER_inner_w(const struct NAME * a, const struct NAME * b)
{
    NOT_IMPLEMENTED_MSG("NLOWER_inner_w");
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
NLOWER_inner(const struct NAME * a, const struct NAME * b)
{
    NOT_IMPLEMENTED_MSG("NLOWER_inner");
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
double NLOWER_norm_w(const struct NAME * p)
{
    double out = sqrt(NLOWER_inner_w(p,p));
    return sqrt(out);
}

/********************************************************//**
*   Compute the norm of NAME
*
*   \param[in] f - function
*
*   \return out - norm of function
*
*   \note
*        Computes \f$ \sqrt(\int_a^b f(x)^2 dx) \f$
*************************************************************/
double NLOWER_norm(const struct NAME * f)
{

    double out = 0.0;
    out = sqrt(NLOWER_inner(p,p));
    return out;
}

/********************************************************//**
*   Multiply NAME by -1
*************************************************************/
void  NLOWER_flip_sign(struct NAME * f)
{   
    NOT_IMPLEMENTED_MSG("NLOWER_flip_sign");
}

/********************************************************//**
*   Multiply by scalar and overwrite expansion
*
*   \param[in] a - scaling factor
*   \param[in] f - NAME to scale
*************************************************************/
void NLOWER_scale(double a, struct NAME * f)
{
    NOT_IMPLEMENTED_MSG("NLOWER_scale");
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
int NLOWER_axpy(double a, struct NAME * x, struct NAME * y)
{
    NOT_IMPLEMENTED_MSG("NLOWER_axpy");
    return 0;
}

////////////////////////////////////////////////////////////////////////////
// Algorithms


/********************************************************//**
*   Obtain the maximum of NAME
*
*   \param[in]     f - NAME
*   \param[in,out] x - location of maximum value
*
*   \return maxval - maximum value
*************************************************************/
double NLOWER_max(struct NAME * f, double * x)
{
    NOT_IMPLEMENTED_MSG("NLOWER_max");
    return 0;
}

/********************************************************//**
*   Obtain the minimum of NAME
*
*   \param[in]     f - NAME
*   \param[in,out] x - location of maximum value
*
*   \return val - min value
*************************************************************/
double NLOWER_min(struct NAME * f, double * x)
{
    NOT_IMPLEMENTED_MSG("NLOWER_min");
    return 0;
}


/********************************************************//**
*   Obtain the maximum in absolute value of NAME
*
*   \param[in]     f     - NAME
*   \param[in,out] x     - location of maximum
*   \param[in]     oargs - optimization arguments 
*                          required for HERMITE otherwise can set NULL
*
*   \return maxval : maximum value (absolute value)
*
*************************************************************/
double NLOWER_absmax(struct NAME * f, double * x, void * oargs)
{
    NOT_IMPLEMENTED_MSG("NLOWER_absmax");
    return 0;
}


/////////////////////////////////////////////////////////
// Utilities
void print_NLOWER(struct NAME * p, size_t prec, 
                  void * args, FILE *fp)
{
   NOT_IMPLEMENTED_MSG("print_NLOWER");
}

