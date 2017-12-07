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





/** \file hpoly.c
 * Provides routines for manipulating hermite polynomials
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "hpoly.h"

//for orthogonal
inline static double zero_seq11(size_t n){ (void) n; return 0.0; }
/* inline static double one_seq11(size_t n) { (void) n; return 1.0; } */
/* inline static double hermc_seq(size_t n) */
/* { */
/*     return -((double)n - 1.0); */
/* } */

// for orthonormal
inline static double an(size_t n){ return 1.0 / sqrt((double)n);}
inline static double cn(size_t n){ return - sqrt(((double)n-1)/(double)n);}


static inline double hermortho(size_t n)
{
    (void)n;
    return 1.0;
    /* double val = 1.0; */
    /* for (size_t ii = 1; ii < n; ii++){ */
    /*     val *= (ii+1); */
    /* } */
    /* //return sqrt(2*M_PI)*val; */
    /* return val; */
}

/********************************************************//**
*   Initialize a Hermite polynomial
*
*   \return p - polynomial
*************************************************************/
struct OrthPoly * init_hermite_poly(){
    
    struct OrthPoly * p;
    if ( NULL == (p = malloc(sizeof(struct OrthPoly)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    p->ptype = HERMITE;
    /* p->an = &one_seq11; */
    p->an = an;
    p->bn = &zero_seq11;
    /* p->cn = &hermc_seq; */
    p->cn = cn;
    
    p->lower = -DBL_MAX;
    p->upper = DBL_MAX;

    p->const_term = 1.0;
    p->lin_coeff = 1.0;
    p->lin_const = 0.0;

    p->norm = hermortho;

    return p;
}

/********************************************************//**
*   Integrate a Hermite approximation
*
*   \param[in] poly - polynomial to integrate
*
*   \return out - integral of approximation
*************************************************************/
double hermite_integrate(const struct OrthPolyExpansion * poly)
{
    double out = poly->coeff[0];//sqrt(2.0*M_PI);
    return out;
}
