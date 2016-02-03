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

int quad2dGrad2(double * x, double * grad, void * args)
{
    assert(args == NULL);
    grad[0] = 2.0 * (x[0] - 3.0);
    grad[1] = 2.0 * (x[1] - 2.0);
    
    return 0;
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

int quad2dHess2(double * x, double * hess, void * args)
{
    assert(args == NULL);
    
    hess[0] = 2.0 + 0.0 * x[0];
    hess[1] = 0.0;
    hess[2] = 0.0;
    hess[3] = 2.0;
    return 0;
}


double rosen2d(double * x, void * args)
{
    assert(args == NULL);

    double out = pow(1.0-x[0],2.0) + 
                 100.0*pow(x[1]-pow(x[0],2.0),2.0);
    return out;
}

double * rosen2dGrad(double * x, void * args)
{
    assert(args == NULL);
    double * grad = calloc_double(2);
    grad[0] = -2.0 * (1.0 - x[0]) + 
              100.0*2.0*(x[1] - pow(x[0],2.0)) * (-2.0 * x[0]);
    grad[1] = 100.0 * 2.0 * (x[1] - pow(x[0],2.0));
    return grad;
}

int rosen2dGrad2(double * x,double* grad,void* args)
{
    assert(args == NULL);
    grad[0] = -2.0 * (1.0 - x[0]) + 100.0 * 2.0 * (x[1] - pow(x[0],2.0)) * (-2.0 * x[0]);
    grad[1] = 100.0 * 2.0 * (x[1] - pow(x[0],2.0));
    return 0;
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

int rosen2dHess2(double * x, double * hess, void * args)
{
    assert(args == NULL);
    
    hess[0] = 1200*x[0]*x[0] - 400*x[1] + 2;
    hess[1] = -400*x[0]; //d / dy ( df / dx)
    hess[2] = -400*x[0];
    hess[3] = 200;
    
    return 0;
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

void Test_pg_newton(CuTest * tc)
{
    printf("Testing Function: newton projected gradient \n");
    
    size_t dim = 2;
    double tol = 1e-12;
    size_t maxiter = 1000;
    size_t maxsubiter = 1000;
    double alpha = 0.3;
    double beta = 0.7;
    int verbose = 0;
    
    double lb[2] = {-5.0,-5.0};
    double ub[2] = {5.0,5.0};
    double start[2] = {2.0,1.0};
    double grad[2];
    double hess[4];
    double space[4];

    double val;
    
    int res = box_damp_newton(dim,lb,ub,start,&val,grad,hess,
                              space,rosen2d,NULL,rosen2dGrad2,
                              NULL,rosen2dHess2,NULL, 
                              tol,maxiter,maxsubiter, 
                              alpha,beta,verbose); 
    CuAssertIntEquals(tc,0,res);
    CuAssertDblEquals(tc,1.0,start[0],1e-3);
    CuAssertDblEquals(tc,1.0,start[1],1e-3);
    CuAssertDblEquals(tc,0.0,val,1e-14);


    start[0] = 1.0;
    start[1] = -2.0;
    ub[0] = 2.0;
    ub[1] = 1.0;
    res = box_damp_newton(dim,lb,ub,start,&val,grad,hess,
                          space,quad2d,NULL,quad2dGrad2,
                          NULL,quad2dHess2,NULL,
                          tol,maxiter,maxsubiter,
                          alpha,beta,verbose);
    CuAssertIntEquals(tc,0,res);
    CuAssertDblEquals(tc,2,start[0],1e-3);
    CuAssertDblEquals(tc,1,start[1],1e-3);
    CuAssertDblEquals(tc,quad2d(start,NULL),val,1e-14);

}

double f_grad_desc(double * x, void * args)
{
    assert(args == NULL);
    double out =  pow(x[0]-3.0,2.0) + pow(x[1]-2.0,2.0);
    return out;
}

int g_grad_desc(double * x, double * out, void * args)
{
    assert(args == NULL);
//    dprint(2,x);
    out[0] = 2.0 * (x[0] - 3.0);
    out[1] = 2.0 * (x[1] - 2.0);
    return 0;
}


void Test_grad_descent(CuTest * tc)
{
    printf("Testing Function: gradient descent with inexact backtrack \n");

    size_t dim = 2;
    double start[2] = {0.0,0.0};
    double grad[2];
    double space[4];
    double tol = 1e-15;
    double alpha = 0.4;
    double beta = 0.9;
    int verbose = 0;

    double val = quad2d(start,NULL);    
    int res = gradient_descent(dim,start,&val,grad,space,
                               f_grad_desc,NULL,g_grad_desc,
                               NULL,tol,10000,
                               10000,alpha,beta,verbose);

//    printf("diff = %G\n",start[0]-3.0);
    CuAssertDblEquals(tc,3.0,start[0],1e-13);
    CuAssertDblEquals(tc,2.0,start[1],1e-13);
    CuAssertDblEquals(tc,0.0,val,1e-13);
    CuAssertIntEquals(tc,0,res);

}

void Test_box_grad_descent(CuTest * tc)
{
    printf("Testing Function: box-constrained gradient descent \n");

    size_t dim = 2;
    double lb[2] = {-1.0,-1.0};
    double ub[2] = {1.0,0.5};
    double start[2] = {0.0,0.0};
    double grad[2];
    double space[4];
    double tol = 1e-15;
    double alpha = 0.4;
    double beta = 0.9;
    int verbose = 0;

    double val = quad2d(start,NULL);    
    int res = box_pg_descent(dim,lb,ub,start,&val,grad,space,f_grad_desc,NULL,
                               g_grad_desc,NULL,tol,10000,10000,alpha,beta,
                               verbose);

//    printf("diff = %G\n",start[0]-3.0);
    CuAssertDblEquals(tc,1.0,start[0],1e-13);
    CuAssertDblEquals(tc,0.5,start[1],1e-13);
    CuAssertIntEquals(tc,0,res);

}


CuSuite * OptGetSuite(){
    //printf("----------------------------\n");

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_newton);
    SUITE_ADD_TEST(suite, Test_pg_newton);
    SUITE_ADD_TEST(suite, Test_grad_descent);
    SUITE_ADD_TEST(suite, Test_box_grad_descent);
    return suite;
}
