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





/** \file probability_utils.c
 * Provides utilities for working with statistics and probability
 */


#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "array.h"

#include "lib_optimization.h"
#include "lib_probability.h"
#include "linalg.h"
#include "lib_clinalg.h"



// Coefficients in rational approximations for erfinv
static const double a1 = -3.969683028665376e+01;
static const double a2 = 2.209460984245205e+02;
static const double a3 = -2.759285104469687e+02;
static const double a4 = 1.383577518672690e+02;
static const double a5 = -3.066479806614716e+01;
static const double a6 = 2.506628277459239e+00;

static const double b1 = -5.447609879822406e+01;
static const double b2 = 1.615858368580409e+02;
static const double b3 = -1.556989798598866e+02;
static const double b4 = 6.680131188771972e+01;
static const double b5 = -1.328068155288572e+01;

static const double c1 = -7.784894002430293e-03;
static const double c2 = -3.223964580411365e-01;
static const double c3 = -2.400758277161838e+00;
static const double c4 = -2.549732539343734e+00;
static const double c5 = 4.374664141464968e+00;
static const double c6 = 2.938163982698783e+00;

static const double d1 = 7.784695709041462e-03;
static const double d2 = 3.224671290700398e-01;
static const double d3 = 2.445134137142996e+00;
static const double d4 = 3.754408661907416e+00;

/***********************************************************//*r
    Compute the inverse cdf of a standard normal distribution using rational approximation
    
    \param p [in] - cdf value

    \return val - invcdf(x)
***************************************************************/
double icdfnorm_rational(double pin)
{
    double p_low = 0.02425;
    double p_high = 1.0 - p_low;
    double p = pin;
    if (pin == 0.0){
        p = DBL_EPSILON;
    }
    else if (p == 1.0){
        p = 1.0-DBL_EPSILON;
    }
    
    double val;
    if (p <= p_low){
        double q = sqrt(-2.0 * log(p));
        val = ((((c1*q+c2)*q + c3)*q + c4)*q + c5)*q + c6;
        val /= ( (((d1*q + d2)*q + d3)*q + d4)*q + 1.0 );
    }
    else if (p <= p_high){
        double r = p - 0.5;
        double q = r*r;
        val = (((((a1*q+a2)*q + a3)*q + a4)*q + a5)*q + a6)*q;
        val /= (((((b1*q + b2)*q + b3)*q + b4)*q + b5)*q + 1.0);

    }
    else{
        double q = sqrt(-2.0 * log(1-p));
        val = ((((c1*q+c2)*q + c3)*q + c4)*q + c5)*q + c6;
        val /= ( (((d1*q + d2)*q + d3)*q + d4)*q + 1.0 );
        val = -val;
    }
    return val;
}

/***********************************************************//*r
    Compute the inverse cdf of a standard normal distribution by refining
    rational approximation result to machine precision using Halley's rational method
    
    \param p [in] - cdf value

    \return val - invcdf(x)
***************************************************************/
double icdfnorm(double p)
{
    double val = icdfnorm_rational(p);
    double e = 0.5 * erfc(-val/sqrt(2.0))-p;
    double u = e * sqrt(2.0*M_PI) * exp(val * val / 2.0);
    double dval = u/(1+val*u/2.0);
    //printf("dval = %G\n",dval);
    val -= dval;

    while (dval > 1e-15){

        e = 0.5 * erfc(-val/sqrt(2.0))-p;
        u = e * sqrt(2.0*M_PI) * exp(val * val / 2.0);
        dval = u/(1+val*u/2.0);
        //printf("dval = %G\n",dval);
        val -= dval;
    }
    return val;
}


/***********************************************************//*r
    Compute the Cumulative distribution function of a normal distribution
    
    \param mean [in] - mean of distribution
    \param std [in] - standard deviation of normal
    \param x [in] - location at which to compute the CDF

    \return val - cdf(x)
***************************************************************/
double cdf_normal(double mean, double std, double x)
{
    
    double temp = (x - mean) / std / sqrt(2.0);
    double val =  0.5 * (1.0 + erf(temp));
    
    return val;
}



/***********************************************************//**
    Compute the Inverse cumulative distribution function of a 
    normal distribution
    
    \param mean [in] - mean of distribution
    \param std [in] - standard deviation of normal
    \param x [in] - location at which to compute the inverse CDF (0,1)

    \return guess - icdf(x)

    \note
        Tested for up to about +- 4 std
***************************************************************/
double icdf_normal(double mean, double std, double x)
{
    
    //double out = mean + std * icdfnorm_rational(x);
    double out = mean + std * icdfnorm(x);
    return out;
}



