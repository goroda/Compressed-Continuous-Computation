// Copyright (c) 2014-2016, Massachusetts Institute of Technology
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

#include "uncon_test.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

////////////////////////////////////////////////
//Collection of test problems from More, Garbow and Hillstrom 1981
////////////////////////////////////////////////
// Rosenbrock function
double rosen_brock_func(size_t dim, double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);
    double f1 = 10.0 * (x[1] - pow(x[0],2));
    double f2 = (1.0 - x[0]);
    
    double out = pow(f1,2) + pow(f2,2);
    
    if (grad != NULL){
        grad[0] = 2.0*f1 * (-20.0 * x[0]) + 2.0 * f2 * (-1.0);
        grad[1] = 2 * f1 * 10.0;
    }
    
    return out;
}

double rosen_brock_start[2] = {-1.2, 1.0};
double rosen_brock_sol[3] = {1.0, 1.0, 0.0};

// Freudenstein and Roth
double f1(size_t dim, double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);
    double f1 = -13.0 + x[0] + ( (5.0 - x[1]) * x[1] - 2.0)*x[1];
    double f2 = -29.0 + x[0] + ( (x[1] + 1.0) * x[1] - 14.0) * x[1];
    
    double out = pow(f1,2) + pow(f2,2);
    
    if (grad != NULL){
        grad[0] = 2.0*f1*1 + 2.0*f2*1;
        grad[1] = 2.0*f1 * (-3.0*pow(x[1],2) + 10.0 * x[1] - 2.0) 
            + 2.0 * f2 * (3.0*pow(x[1],2) + 2.0*x[1] - 14.0);
    }
    
    return out;
}

double f1start[2] = {0.5, -2.0};
double f1sol[3] = {11.41, -0.8968, 48.9842};
// alternative
/* double f1sol[3] = {5.0, 4.0, 0.0}; */

// Powell badly scaled
double f2(size_t dim, double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);
    double f1 = pow(10.0,4)*x[0]*x[1] - 1.0;
    double f2 = exp(-x[0]) + exp(-x[1]) - 1.001;
    
    double out = pow(f1,2) + pow(f2,2);
    
    if (grad != NULL){
        grad[0] = 2.0*f1*pow(10.0,4)*x[1] + 2.0*f2*(-1.0)*exp(-x[0]);
        grad[1] = 2.0*f1*pow(10.0,4)*x[0] + 2.0*f2*(-1.0)*exp(-x[1]);
    }
    
    return out;
}

double f2start[2] = {0.0,1.0};
double f2sol[3] = {1.098e-5, 9.106,0.0};

// Brown badly scaled
double ff3(size_t dim, double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);
    double f1 = x[0] - pow(10,6);
    double f2 = x[1] - 2 * pow(10,-6.0);
    double f3 = x[0]*x[1] - 2.0;
    
    double out = pow(f1,2) + pow(f2,2) + pow(f3,2);
    
    if (grad != NULL){
        grad[0] = 2.0*f1 + 2.0*f3*x[1];
        grad[1] = 2.0*f2 + 2.0*f3*x[0];
    }
    return out;
}

double f3start[2] = {1.0,1.0};
double f3sol[3] = {1e6, 2e-6, 0.0};

// Beale function
double ff4(size_t dim, double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);
    double f1 = 1.5 - x[0] * (1 - pow(x[1],1));
    double f2 = 2.25 - x[0] * (1 - pow(x[1],2));
    double f3 = 2.625 - x[0] * (1 - pow(x[1],3));

    double out = pow(f1,2) + pow(f2,2) + pow(f3,2);
    
    if (grad != NULL){
        grad[0] = 2 * f1 * (-1) * (1 - pow(x[1],1)) + 
                  2 * f2 * (-1) * (1 - pow(x[1],2)) +
                  2 * f3 * (-1) * (1 - pow(x[1],3)); 

        grad[1] = 2 * f1 * x[0] + 
                  2 * f2 * 2*x[0]*x[1] + 2 * f3 * 3*x[0]*x[1]*x[1];

    }
    return out;
}

double f4start[2] = {1.0,1.0};
double f4sol[3] = {3.0, 0.5, 0.0};

// Jennrich and Sampson function
double ff5(size_t dim, double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);
    
    if (isnan(x[0])){
        printf("stop\n");
        exit(1);
    }

    if (isnan(x[1])){
        printf("stop\n");
        exit(1);
    }
    double f[10];
    double out = 0.0;
    for (size_t ii = 0; ii < 10; ii++){
        f[ii] = 2.0 + 2.0 * (double) (ii+1) - 
            (exp((double)(ii+1)*x[0]) + exp((double)(ii+1) * x[1]));
        out += pow(f[ii],2);
    }

    if (grad != NULL){
        grad[0] = 0.0;
        grad[1] = 0.0;
        for (size_t ii = 0; ii < 10; ii++){
            grad[0] += 2.0 * f[ii] * (-(double)(ii+1))*exp((double)(ii+1)*x[0]);
            grad[1] += 2.0 * f[ii] * (-(double)(ii+1))*exp((double)(ii+1)*x[1]);
        }
    }
    
    if (isnan(out)){
        dprint(2,x);
        exit(1);
    }
    return out;
}

double f5start[2] = {0.3,0.4};
double f5sol[3] = {0.2578, 0.2578, 124.362};

// Helical valley function
double ff6(size_t dim, double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);
    

    double theta;
    if (x[0] > 0){
        theta = 1.0 / 2.0 / M_PI * atan(x[1]/x[0]);
    }
    else{
        theta = 1.0 / 2.0 / M_PI * atan(x[1]/x[0])+0.5;
    }
    
    double f1 = 10 * (x[2] - 10 * theta);
    double f2 = 10 * (sqrt( x[0]*x[0] + x[1]*x[1]) - 1.0);
    double f3 = x[2];

    double out = f1 * f1 + f2 * f2 + f3 * f3;

    if (grad != NULL){
        
        double dtheta1 = 1.0/2.0/M_PI * (-x[1] / (x[0]*x[0] + x[1]*x[1]));
        double dtheta2 = 1.0/2.0/M_PI * (x[0] / (x[0]*x[0] + x[1]*x[1]));

        grad[0] = 0.0;
        grad[0] += 2* f1 * (-100) * dtheta1;
        grad[0] += 2 * f2 * 5 * 1.0/sqrt(x[0]*x[0] + x[1]*x[1]) * 2.0 * x[0];

        grad[1] = 0.0;
        grad[1] += 2.0 * f1 *(-100) * dtheta2;
        grad[1] += 2 * f2 * 5 * 1.0/sqrt(x[0]*x[0] + x[1]*x[1]) * 2.0 * x[1];

        grad[2] = 0.0;
        grad[2] += 2.0 * f1 * 10.0;
        grad[2] += 2.0 * f3;
    }

    
    return out;
}

double f6start[3] = {-1.0,0.0,0.0};
double f6sol[4] = {1.0,0.0,0.0,0.0};

// Helical valley function
double ff7(size_t dim, double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    double y[15] = {0.14, 0.18, 0.22, 0.25, 0.29,
                    0.32, 0.35, 0.39, 0.37, 0.58,
                    0.73, 0.96, 1.34, 2.10, 4.39};

    double f[15];
    double u[15];
    double v[15];
    double w[15];
    double out = 0.0;
    for (size_t ii = 0; ii < 15; ii++){
        u[ii] = (double)(ii+1);
        v[ii] = 16.0 - (double)(ii+1);
        w[ii] = u[ii];
        if (v[ii] < u[ii]) w[ii] = v[ii];

        f[ii] = y[ii] - (x[0] + u[ii]/(v[ii] * x[1] + w[ii] * x[2]));
        out += f[ii] * f[ii];
    }

    if (grad != NULL){
        
        grad[0] = 0.0;
        grad[1] = 0.0;
        grad[2] = 0.0;
        for (size_t ii = 0; ii < 15; ii++){
            grad[0] += 2*f[ii]*(-1.0);
            grad[1] += 2*f[ii]*u[ii]*1.0/pow(v[ii]*x[1] + w[ii]*x[2],2)*v[ii];
            grad[2] += 2*f[ii]*u[ii]*1.0/pow(v[ii]*x[1] + w[ii]*x[2],2)*w[ii];
        }
    }

    
    return out;
}
double f7start[3] = {1.0,1.0,1.0}; 
double f7sol[4] = {0.0,0.0,0.0,8.21487e-3};// confused

// Gaussian Function
double ff8(size_t dim, double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    double y[15];
    y[0] = 0.0009;
    y[1] = 0.0044;
    y[2] = 0.0175;
    y[3] = 0.0540;
    y[4] = 0.1295;
    y[5] = 0.2420;
    y[6] = 0.3521;
    y[7] = 0.3989;
    y[14] = 0.0009;
    y[13] = 0.0044;
    y[12] = 0.0175;
    y[11] = 0.0540;
    y[10] = 0.1295;
    y[9] = 0.2420;
    y[8] = 0.3521;

    double f[15];
    double t[15];
    double out = 0.0;
    for (size_t ii = 0; ii < 15; ii++){
        t[ii] = (8.0 - (double)(ii+1))/2.0;
        f[ii] = x[0] * exp( -0.5 * x[1] * pow(t[ii] - x[2],2)) - y[ii];
        out += f[ii] * f[ii];
    }

    if (grad != NULL){
        grad[0] = 0.0;
        grad[1] = 0.0;
        grad[2] = 0.0;
        for (size_t ii = 0; ii < 15; ii++){
            grad[0] += 2*f[ii]*exp( -0.5 * x[1] * pow(t[ii] - x[2],2));
            grad[1] += 2*f[ii]*x[0]*exp( -0.5 * x[1] * pow(t[ii] - x[2],2)) * 
                       -0.5 * pow(t[ii]-x[2],2);
            grad[2] += 2*f[ii]*x[0]*exp( -0.5 * x[1] * pow(t[ii] - x[2],2)) * 
                       -0.5 * x[1] * (t[ii] - x[2]) * (-1.0);
        }
    }

    
    return out;
}
double f8start[3] = {0.4, 1.0, 0.0};
double f8sol[4] = {0.0,0.0,0.0,1.12793e-8};// confused


// Meyer function
double ff9(size_t dim, double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    double y[16] = { 34780.0, 28610.0, 23650.0, 19630.0, 16370.0,
                     13720.0, 11540.0,  9744.0,  8261.0,  7030.0,
                      6005.0,  5147.0,  4427.0,  3820.0,  3307.0,
                      2872.0};
    double f[16];
    double t[16];
    double out = 0.0;
    for (size_t ii = 0; ii < 16; ii++){
        t[ii] = 45.0 + 5.0 * ii;
        f[ii] = x[0] * exp( x[1] / (t[ii] + x[2])) - y[ii];
        out += f[ii] * f[ii];
    }

    if (grad != NULL){
        grad[0] = 0.0;
        grad[1] = 0.0;
        grad[2] = 0.0;
        for (size_t ii = 0; ii < 16; ii++){
            grad[0] += 2*f[ii]*exp( x[1] / (t[ii] + x[2]));
            grad[1] += 2*f[ii]*x[0] * exp( x[1] / (t[ii] + x[2])) * 
                          1.0/(t[ii] + x[2]);
            grad[2] += 2*f[ii]*x[0] * exp( x[1] / (t[ii] + x[2])) * 
                -x[1]/pow(t[ii]+x[2],2);
        }
    }

    
    return out;
}
double f9start[3] = {0.02, 4000.0, 250.0};
double f9sol[4]   = {0.0,0.0,0.0,87.9458};// confused

/* // Gulf research and development function */
/* double ff10(size_t dim, double * x, double * grad, void * arg) */
/* { */
/*     (void)(dim); */
/*     (void)(arg); */

/*     double f[100]; */
/*     double t[100]; */
/*     double y[100]; */
/*     double m = 100.0; */
/*     double out = 0.0; */
/*     for (size_t ii = 0; ii < 100; ii++){ */
/*         t[ii] = (double)(ii+1)/100.0; */
/*         y[ii] = 25.0 + pow(-50.0 * log(t[ii]),2.0/3.0); */
/*         f[ii] = exp(-1.0/x[0] * pow( fabs(y[ii]*m*(double)(ii+1)*x[2]),x[2])); */
/*         out += f[ii] * f[ii]; */
/*     } */

/*     if (grad != NULL){ */
/*         grad[0] = 0.0; */
/*         grad[1] = 0.0; */
/*         grad[2] = 0.0; */
/*         for (size_t ii = 0; ii < 100; ii++){ */
/*             grad[0] += exp(-1.0/x[0] * pow( fabs(y[ii]*m*(double)(ii+1)*x[2]),x[2])) * -1.0/pow(x[0],2) * pow( fabs(y[ii]*m*(double)(ii+1)*x[2]),x[2]); */
/*             grad[1] += exp(-1.0/x[0] * pow( fabs(y[ii]*m*(double)(ii+1)*x[2]),x[2])) * (-1.0/x[0]) *  */
/*         } */
/*     } */

    
/*     return out; */
/* } */
/* double f10start[3] = {5.0, 2.5, 0.15}; */
/* double f10sol[4]   = {50.0,25.0,1.50,0.0}; */


///////////////////////////////////////////////////////////////////////////
//assign problems

void create_unc_probs(){
    tprobs[0].dim = 2;
    tprobs[0].eval = rosen_brock_func;
    tprobs[0].start = rosen_brock_start;
    tprobs[0].sol = rosen_brock_sol;

    tprobs[1].dim   = 2;
    tprobs[1].eval  = f1;
    tprobs[1].start = f1start;
    tprobs[1].sol   = f1sol;

    tprobs[2].dim   = 2;
    tprobs[2].eval  = f2;
    tprobs[2].start = f2start;
    tprobs[2].sol   = f2sol;

    tprobs[3].dim   = 2;
    tprobs[3].eval  = ff3;
    tprobs[3].start = f3start;
    tprobs[3].sol   = f3sol;

    tprobs[4].dim   = 2;
    tprobs[4].eval  = ff4;
    tprobs[4].start = f4start;
    tprobs[4].sol   = f4sol;

    tprobs[5].dim   = 2;
    tprobs[5].eval  = ff5;
    tprobs[5].start = f5start;
    tprobs[5].sol   = f5sol;

    tprobs[6].dim   = 3;
    tprobs[6].eval  = ff6;
    tprobs[6].start = f6start;
    tprobs[6].sol   = f6sol;

    tprobs[7].dim   = 3;
    tprobs[7].eval  = ff7;
    tprobs[7].start = f7start;
    tprobs[7].sol   = f7sol;

    tprobs[8].dim   = 3;
    tprobs[8].eval  = ff8;
    tprobs[8].start = f8start;
    tprobs[8].sol   = f8sol;

    tprobs[9].dim   = 3;
    tprobs[9].eval  = ff9;
    tprobs[9].start = f9start;
    tprobs[9].sol   = f9sol;
}


// other functionsx
size_t unc_test_problem_get_dim(void * arg)
{
    struct UncTestProblem * p = arg;
    return p->dim;
}

double * unc_test_problem_get_start(void * arg)
{
    struct UncTestProblem * p = arg;
    return p->start;
}

double * unc_test_problem_get_sol(void * arg)
{
    struct UncTestProblem * p = arg;
    return p->sol;
}

double unc_test_problem_eval(size_t dim, double * x,double * grad,void *arg)
{
    struct UncTestProblem * p = arg;
    
    return p->eval(dim,x,grad,arg);
}





