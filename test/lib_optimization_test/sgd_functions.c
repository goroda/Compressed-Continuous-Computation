// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016-2017 Sandia Corporation

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



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "array.h"
#include "linalg.h"
#include "lib_optimization.h"

#include "sgd_functions.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

////////////////////////////////////////////////
//Collection of test problems for testing sgd within machine learning applications

////////////////////////////////////////////////
// 2dimensional linear least squares 
static double lin_ls_2d_start[2] = {-5.0, 3.0};
static double lin_ls_2d_sol[3] = {0.3,-0.2, 0.0}; // last one is value at minimum
static double lin_ls_2d(size_t dim, size_t ind, const double * x, double * grad, void * arg)
{

    (void)(dim);

    struct SgdData * data = arg;
    double p1 = 0.3;
    double p2 = -0.2;
    double guess = data->x[ind*2]*x[0] + data->x[ind*2+1]*x[1];

    double true_val = p1*data->x[ind*2] + p2*data->x[ind*2+1];
    double resid = true_val - guess;
    double loss = resid * resid;

    if (grad != NULL){
        grad[0] = -2 * resid * data->x[ind*2];
        grad[1] = -2 * resid * data->x[ind*2+1];
    }

    return loss;
}

static double lin_ls_5d_start[5] = {-5.0, 3.0, 0.9, 10.0, -0.3};
static double lin_ls_5d_sol[6] = {0.3,-0.2, 0.6, 12.0, 14.0, 0.0}; // last one is value at minimum
static double lin_ls_5d(size_t dim, size_t ind, const double * x, double * grad, void * arg)
{

    (void)(dim);

    struct SgdData * data = arg;

    double guess = 0.0;
    double true_val = 0.0;
    for (size_t ii = 0; ii < 5; ii++){
        guess += data->x[ind*5+ii]*x[ii];
        true_val +=  data->x[ind*5+ii] * lin_ls_5d_sol[ii];
    }

    double resid = true_val - guess;
    double loss = resid * resid;

    if (grad != NULL){
        grad[0] = -2 * resid * data->x[ind*5];
        grad[1] = -2 * resid * data->x[ind*5+1];
        grad[2] = -2 * resid * data->x[ind*5+2];
        grad[3] = -2 * resid * data->x[ind*5+3];
        grad[4] = -2 * resid * data->x[ind*5+4];
    }

    return loss;
}


struct SgdTestProblem sprobs[34];
void create_sgd_probs(){
        
    sprobs[0].dim = 2;
    sprobs[0].eval = lin_ls_2d;
    sprobs[0].start = lin_ls_2d_start;
    sprobs[0].sol = lin_ls_2d_sol;

    sprobs[1].dim = 5;
    sprobs[1].eval = lin_ls_5d;
    sprobs[1].start = lin_ls_5d_start;
    sprobs[1].sol = lin_ls_5d_sol;
}


// other functions
size_t sgd_test_problem_get_dim(void * arg)
{
    struct SgdTestProblem * p = arg;
    return p->dim;
}

double * sgd_test_problem_get_start(void * arg)
{
    struct SgdTestProblem * p = arg;
    return p->start;
}

double * sgd_test_problem_get_sol(void * arg)
{
    struct SgdTestProblem * p = arg;
    return p->sol;
}

double sgd_test_problem_eval(size_t dim, size_t ind, const double * x,double * grad,void *arg)
{
    struct SgdData * data = arg;

    return data->prob.eval(dim,ind,x,grad,data);
}

