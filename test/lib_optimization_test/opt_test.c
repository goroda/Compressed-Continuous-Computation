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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "array.h"
#include "linalg.h"
#include "lib_optimization.h"

#include "CuTest.h"

double quad2d(double * x, void * args)
{
    assert(args == NULL);

    double out =  pow(x[0]-3.0,2.0) + pow(x[1]-2.0,2.0);
    return out;
}

double * quad2dGrad(double * x, void * args)
{
    assert(args == NULL);
    double * grad = calloc_double(2);
    grad[0] = 2.0 * (x[0] - 3.0);
    grad[1] = 2.0 * (x[1] - 2.0);
    return grad;
}

double * quad2dHess(double * x, void * args)
{
    assert(args == NULL);
    
    double * hess = calloc_double(2 * 2);
    hess[0] = 2.0 + 0.0 * x[0];
    hess[1] = 0.0;
    hess[2] = 0.0;
    hess[3] = 2.0;
    return hess;
}

double rosen2d(double * x, void * args)
{
    assert(args == NULL);

    double out = pow(1.0-x[0],2.0) + 100.0*pow(x[1]-pow(x[0],2.0),2.0);
    return out;
}

double * rosen2dGrad(double * x, void * args)
{
    assert(args == NULL);
    double * grad = calloc_double(2);
    grad[0] = -2.0 * (1.0 - x[0]) + 100.0 * 2.0 * (x[1] - pow(x[0],2.0)) * (-2.0 * x[0]);
    grad[1] = 100.0 * 2.0 * (x[1] - pow(x[0],2.0));
    return grad;
}

double * rosen2dHess(double * x, void * args)
{
    assert(args == NULL);
    
    double * hess = calloc_double(2 * 2);
    hess[0] = 2.0 + 100.0 * 2.0 * ( -2.0 * (x[1] - pow(x[0],2.0)) ) * ( -2.0 * x[0] * (-2.0 * x[0]));
    hess[1] = 100.0 * 2.0 * x[1]; //d / dy ( df / dx)
    hess[2] = 100.0 * 2.0 * (-2.0 * x[0]); // d / dx (df / dy)
    hess[3] = 100.0 * 2.0;
    return hess;
}

void Test_newton(CuTest * tc)
{
    printf("Testing Function: newton \n");
    double * start = calloc_double(2);
    size_t dim = 2;
    double step_size = 1.0;
    double tol = 1e-8;
    
    newton(&start,dim,step_size,tol,quad2dGrad, quad2dHess, NULL);
    double val = quad2d(start,NULL);
    CuAssertDblEquals(tc,3.0,start[0],1e-14);
    CuAssertDblEquals(tc,2.0,start[1],1e-14);
    CuAssertDblEquals(tc,0.0,val,1e-14);

    newton(&start,dim,step_size,1e-10,rosen2dGrad, rosen2dHess, NULL);
    val = rosen2d(start,NULL);
    CuAssertDblEquals(tc,1.0,start[0],1e-3);
    CuAssertDblEquals(tc,1.0,start[1],1e-3);
    CuAssertDblEquals(tc,0.0,val,1e-14);


    free(start); start = NULL;
}


CuSuite * OptGetSuite(){
    //printf("----------------------------\n");

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_newton);
    return suite;
}
