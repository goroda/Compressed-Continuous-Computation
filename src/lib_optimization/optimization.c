// Copyright (c) 2014-2015, Massachusetts Institute of Technology
//
// This file is part of the Compressed Continuous Computation (C3) toolbox
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


/** \file optimization.c
 * Provides routines for optimization
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "array.h"
#include "linalg.h"
#include "optimization.h"

/***********************************************************//**
    Minimization using Newtons method

    \param start [inout] - starting point and resulting solution
    \param dim [in] - dimension
    \param step_size [in] - usually 1.0 
    \param tol [in] - absolute tolerance for gradient convergence
    \param g [in] - function for gradient evaluation
    \param h [in] - function for hessian evaluation
    \param args [in] - arguments to gradient and hessian
***************************************************************/
void
newton(double ** start, size_t dim, double step_size, double tol,
        double * (*g)(double *, void *),
        double * (*h)(double *, void *), void * args)
{
        
    int done = 0;
    double * p = calloc_double(dim);
    size_t * piv = calloc_size_t(dim);
    size_t one = 1;
    double diff;
    //double diffrel;
    double den;
    size_t ii;
    while (done == 0)
    {
        double * grad = g(*start,args);
        double * hess = h(*start, args);
        int info;
        dgesv_(&dim,&one,hess,&dim,piv,grad,&dim, &info);
        assert(info == 0);
        
        diff = 0.0;
        //diffrel = 0.0;
        den = 0.0;
        for (ii = 0; ii < dim; ii++){
            //printf("sol[%zu]=%G\n",ii,grad[ii]);
            den += pow((*start)[ii],2);
            (*start)[ii] = (*start)[ii] - step_size*grad[ii];
            diff += pow(step_size*grad[ii],2);
        }
        diff = sqrt(diff);
        /*
        if (den > 1e-15)
            diffrel = diff / sqrt(den);
        else{
            diffrel = den;
        }
        */
        
        if (diff < tol){
            done = 1;
        }
        free(grad); grad = NULL;
        free(hess); hess = NULL;
    }
    free(piv); piv = NULL;
    free(p); p = NULL;
}

