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

#include "unconstrained_functions.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

////////////////////////////////////////////////
//Collection of test problems from More, Garbow and Hillstrom 1981
////////////////////////////////////////////////
// Rosenbrock function
double rosen_brock_func(size_t dim, const double * x, double * grad, void * arg)
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

static double rosen_brock_start[2] = {-1.2, 1.0};
static double rosen_brock_sol[3] = {1.0, 1.0, 0.0};

// Freudenstein and Roth
double f1(size_t dim, const double * x, double * grad, void * arg)
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

static double f1start[2] = {0.5, -2.0};
static double f1sol[3] = {11.41, -0.8968, 48.9842};
// alternative
/* double f1sol[3] = {5.0, 4.0, 0.0}; */

// Powell badly scaled
double f2(size_t dim, const double * x, double * grad, void * arg)
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

static double f2start[2] = {0.0,1.0};
static double f2sol[3] = {1.098e-5, 9.106,0.0};

// Brown badly scaled
double ff3(size_t dim, const double * x, double * grad, void * arg)
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

static double f3start[2] = {1.0,1.0};
static double f3sol[3] = {1e6, 2e-6, 0.0};

// Beale function
double ff4(size_t dim, const double * x, double * grad, void * arg)
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

static double f4start[2] = {1.0,1.0};
static double f4sol[3] = {3.0, 0.5, 0.0};

// Jennrich and Sampson function
double ff5(size_t dim, const double * x, double * grad, void * arg)
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

static double f5start[2] = {0.3,0.4};
static double f5sol[3] = {0.2578, 0.2578, 124.362};

// Helical valley function
double ff6(size_t dim, const double * x, double * grad, void * arg)
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

static double f6start[3] = {-1.0,0.0,0.0};
static double f6sol[4] = {1.0,0.0,0.0,0.0};

// Bard function
double ff7(size_t dim, const double * x, double * grad, void * arg)
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
static double f7start[3] = {1.0,1.0,1.0}; 
static double f7sol[4] = {0.0,0.0,0.0,8.21487e-3};// confused

// Gaussian Function
double ff8(size_t dim, const double * x, double * grad, void * arg)
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
static double f8start[3] = {0.4, 1.0, 0.0};
static double f8sol[4] = {0.0,0.0,0.0,1.12793e-8};// confused


// Meyer function
double ff9(size_t dim, const double * x, double * grad, void * arg)
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
static double f9start[3] = {0.02, 4000.0, 250.0};
static double f9sol[4]   = {0.0,0.0,0.0,87.9458};// confused

// Gulf research and development function
double ff10(size_t dim, const double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    double f[100];
    double t[100];
    double y[100];
    double m = 100.0;
    double out = 0.0;
    for (size_t ii = 0; ii < 100; ii++){
        t[ii] = (double)(ii+1)/100.0;
        y[ii] = 25.0 + pow(-50.0 * log(t[ii]),2.0/3.0);
        double g = y[ii]*m*(double)(ii+1)*x[1];
        f[ii] = exp(-1.0/x[0] * pow(fabs(g),x[2]));
        out += f[ii] * f[ii];
    }

    if (grad != NULL){
        grad[0] = 0.0;
        grad[1] = 0.0;
        grad[2] = 0.0;
        for (size_t ii = 0; ii < 100; ii++){
            double g = y[ii]*m*(double)(ii+1)*x[1];
            double preexp = exp(-1.0/x[0] * pow( fabs(g) ,x[2]));
            grad[0] += 2*f[ii]*preexp * 1.0/pow(x[0],2) * pow(fabs(g),x[2]);
            double gp = y[ii]*m*(double)(ii+1);
            grad[1] += 2*f[ii]*preexp * (-1.0/x[0]) * x[2] * g * gp * pow(fabs(g),x[2]-2);
            grad[2] += 2*f[ii]*preexp * (-1.0/x[0]) * pow(g,x[2]) * log(fabs(g));
        }
    }
    
    return out;
}
static double f10start[3] = {5.0, 2.5, 0.15};
static double f10sol[4]   = {50.0,25.0,1.50,0.0};

// Box three-dimensional 
double ff11(size_t dim, const double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    double f[20];
    double t[20];
    size_t m = 20;
    double out = 0.0;
    for (size_t ii = 0; ii < m; ii++){
        t[ii] = (double)(ii+1)/10.0;
        f[ii] = exp(-t[ii]*x[0]) - exp(-t[ii]*x[1]) - x[2]*(exp(-t[ii]) - exp(-10.0*t[ii]));
        out += f[ii] * f[ii];
    }

    if (grad != NULL){
        grad[0] = 0.0;
        grad[1] = 0.0;
        grad[2] = 0.0;
        for (size_t ii = 0; ii < m; ii++){
            grad[0] += 2*f[ii]*exp(-t[ii]*x[0])*(-t[ii]);
            grad[1] += 2*f[ii]*exp(-t[ii]*x[1])*(t[ii]);
            grad[2] += 2*f[ii]*-(exp(-t[ii]) - exp(-10.0*t[ii]));
        }
    }
    
    return out;
}
static double f11start[3] = {0.0, 10.0, 20.0};
static double f11sol[4]   = {1.0,10.0,1.0,0.0}; // multiple minimum exist

// Powell singular function 
double ff12(size_t dim, const double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    double f[20];
    double out = 0.0;
    f[0] = x[0] + 10.0 * x[1];
    f[1] = pow(5,0.5) * (x[2]-x[3]);
    f[2] = pow(x[1]-2*x[2],2.0);
    f[3] = pow(10.0,0.5)*pow(x[0]-x[3],2);
    out = f[0]*f[0] + f[1]*f[1] + f[2]*f[2] + f[3]*f[3];
    
    if (grad != NULL){
        grad[0] = 2.0 * f[0] + 2.0 * f[3] * pow(10.0,0.5)*2.0 *(x[0]-x[3]);
        grad[1] = 2.0 * f[0] * 10.0 + 2.0 * f[2] * 2.0 *(x[1] - 2*x[2]);
        grad[2] = 2.0 * f[1] * pow(5.0,0.5) + 2.0 * f[2] * 2.0 *(x[1] - 2*x[2]) *-2.0;
        
        grad[3] = 2.0 * f[1] * pow(5.0,0.5)*-1.0 + 2.0 * f[3] * pow(10.0,0.5)*2.0 *(x[0]-x[3])*-1.0;
    }
    
    return out;
}
static double f12start[4] = {3.0,-1.0,0.0,1.0};
static double f12sol[5]   = {0.0, 0.0, 0.0, 0.0, 0.0};

// Wood function
double ff13(size_t dim, const double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    double f[20];
    double out = 0.0;
    
    f[0] = 10.0 * (x[1] - pow(x[0],2));
    f[1] = 1.0-x[0];
    f[2] = pow(90.0,0.5)*(x[3]-pow(x[2],2));
    f[3] = 1.0 - x[2];
    f[4] = pow(10.0,0.5)*(x[1] + x[3] - 2.0);
    f[5] = pow(10.0,-0.5) * (x[1]-x[3]);
    
    out = f[0]*f[0] + f[1]*f[1] + f[2]*f[2] + f[3]*f[3] + f[4]*f[4] + f[5]*f[5];
    
    if (grad != NULL){
        grad[0] = 2.0 * f[0] * 10.0 * -2.0 * x[0] + 2.0 * f[1] * -1.0;
        grad[1] = 2.0 * f[0] * 10.0  + 2.0 * f[4] * pow(10.0,0.5) + 2.0 * f[5] * pow(10,-0.5);
        grad[2] = 2.0 * f[2] * pow(90.0,0.5) * -2.0 * x[2] + 2.0 * f[3] * -1.0;
        grad[3] = 2.0 * f[2] * pow(90.0,0.5)  + 2.0 * f[4] * pow(10.0,0.5) + 2.0 * f[5] * pow(10,-0.5) * -1.0;
    }

    /* printf("out=%G\n",out); */
    return out;
}
static double f13start[4] = {-3.0,-1.0,-3.0,-1.0};
static double f13sol[5]   = {1.0, 1.0, 1.0, 1.0, 0.0};

// Kowalic and Osborne function
double ff14(size_t dim, const double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    double f[11];
    double y[11] = {0.1957,0.1947,0.1735,0.1600,0.0844,0.0627,
                    0.0456,0.0342,0.0323,0.0235,0.0246};
    double u[11] = {4.0,2.0,1.0,0.5,0.25,0.167,0.125,0.1,0.0833,0.0714,0.0625};
    size_t m = 11;

    double out = 0.0;
    for (size_t ii = 0; ii < m; ii++){
        f[ii] = y[ii] - x[0]*(pow(u[ii],2) + u[ii]*x[1]) / (pow(u[ii],2) + u[ii]*x[2] + x[3]);
        out += f[ii]*f[ii];
    }


    if (grad != NULL){
        grad[0] = 0.0;
        grad[1] = 0.0;
        grad[2] = 0.0;
        grad[3] = 0.0;
        for (size_t ii = 0; ii < m; ii++){
            grad[0] += 2.0 * f[ii] * (pow(u[ii],2) + u[ii]*x[1]) / (pow(u[ii],2) + u[ii]*x[2] + x[3]) * -1.0;
            grad[1] += 2.0 * f[ii] * x[0]*u[ii] / (pow(u[ii],2) + u[ii]*x[2] + x[3]) * (-1.0);
            grad[2] += 2.0 * f[ii] * x[0]*(pow(u[ii],2) + u[ii]*x[1]) / pow(pow(u[ii],2) + u[ii]*x[2] + x[3],2) * u[ii];
            grad[3] += 2.0 * f[ii] * x[0]*(pow(u[ii],2) + u[ii]*x[1]) / pow(pow(u[ii],2) + u[ii]*x[2] + x[3],2);
        }
    }

    return out;
}
static double f14start[4] = {0.25,0.39,0.415,0.39};
static double f14sol[5]   = {0.0,0.0,0.0,0.0,3.07505e-4}; // unknown minimizer

// Brown and Dennis
double ff15(size_t dim, const double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    double f[20];
    double t[20];
    size_t m = 20;

    double out = 0.0;
    for (size_t ii = 0; ii < m; ii++){
        t[ii] = (double)(ii+1)/5.0;
        f[ii] = pow(x[0] + t[ii]*x[1] - exp(t[ii]),2) + pow(x[2] + x[3]*sin(t[ii]) - cos(t[ii]),2);
        out += f[ii]*f[ii];
    }


    if (grad != NULL){
        grad[0] = 0.0;
        grad[1] = 0.0;
        grad[2] = 0.0;
        grad[3] = 0.0;
        for (size_t ii = 0; ii < m; ii++){
            grad[0] += 2.0 * f[ii] * 2.0 * (x[0] + t[ii]*x[1] - exp(t[ii])); 
            grad[1] += 2.0 * f[ii] * 2.0 * (x[0] + t[ii]*x[1] - exp(t[ii])) * t[ii]; 
            grad[2] += 2.0 * f[ii] * 2.0 * (x[2] + x[3]*sin(t[ii]) - cos(t[ii]));
            grad[3] += 2.0 * f[ii] * 2.0 * (x[2] + x[3]*sin(t[ii]) - cos(t[ii]))*sin(t[ii]);
        }
    }

    return out;
}

static double f15start[4] = {25.0,5.0,-5.0,-1.0};
static double f15sol[5]   = {0.0,0.0,0.0,0.0,85822.2}; // unknown minimizer

// Osborne 1
double ff16(size_t dim, const double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    double f[33];
    double t[33];
    size_t m = 33;

    double y[33] = {0.844,0.908,0.932,0.936,0.925,0.908,0.881,0.850,0.818,
                    0.784,0.751,0.718,0.685,0.658,0.628,0.603,0.580,0.558,
                    0.538,0.522,0.506,0.490,0.478,0.467,0.457,0.448,0.438,
                    0.431,0.424,0.420,0.414,0.411,0.406};
    double out = 0.0;
    for (size_t ii = 0; ii < m; ii++){
        t[ii] = (double)ii * 10.0;
        f[ii] = y[ii] - (x[0] + x[1]*exp(-t[ii]*x[3]) + x[2] * exp(-t[ii]*x[4]));
        out += f[ii]*f[ii];
    }


    if (grad != NULL){
        grad[0] = 0.0;
        grad[1] = 0.0;
        grad[2] = 0.0;
        grad[3] = 0.0;
        grad[4] = 0.0;
        for (size_t ii = 0; ii < m; ii++){
            grad[0] += 2.0 * f[ii] * -1.0;
            grad[1] += 2.0 * f[ii] * exp(-t[ii]*x[3])*-1.0;
            grad[2] += 2.0 * f[ii] * exp(-t[ii]*x[4])*-1.0;
            grad[3] += 2.0 * f[ii] * x[1] * -t[ii] * exp(-t[ii]*x[3])*-1.0;
            grad[4] += 2.0 * f[ii] * x[2] * -t[ii] * exp(-t[ii]*x[4])*-1.0;
        }
    }

    return out;
}

static double f16start[5] = {0.5,1.5,-1.0,0.01,0.02};
static double f16sol[6]   = {0.0,0.0,0.0,0.0,0.0,5.46489e-5}; // unknown minimizer

// Biggs EXP6
double ff17(size_t dim, const double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    double f[13];
    double y[13];
    double t[13];
    size_t m = 13;

    double out = 0.0;
    for (size_t ii = 0; ii < m; ii++){
        t[ii] = (double)(ii+1)/10.0;
        y[ii] = exp(-t[ii]) - 5.0*exp(-10.0*t[ii]) + 3.0 * exp(-4.0*t[ii]);
        f[ii] = x[2]*exp(-t[ii]*x[0]) - x[3]*exp(-t[ii]*x[1]) + x[5]*exp(-t[ii]*x[4]) - y[ii];
        out += f[ii]*f[ii];
    }


    if (grad != NULL){
        
        grad[0] = 0.0;
        grad[1] = 0.0;
        grad[2] = 0.0;
        grad[3] = 0.0;
        grad[4] = 0.0;
        grad[5] = 0.0;
        for (size_t ii = 0; ii < m; ii++){
            grad[0] += 2.0 * f[ii] * x[2] * -t[ii] * exp(-t[ii]*x[0]);
            grad[1] += 2.0 * f[ii] * x[3] * -t[ii] * exp(-t[ii]*x[1])*-1.0;
            grad[2] += 2.0 * f[ii] * exp(-t[ii]*x[0]);
            grad[3] += 2.0 * f[ii] * exp(-t[ii]*x[1])*-1.0;
            grad[4] += 2.0 * f[ii] * x[5] * exp(-t[ii]*x[4]) * -t[ii];
            grad[5] += 2.0 * f[ii] * exp(-t[ii] * x[4]);
        }
    }

    return out;
}

static double f17start[6] = {1.0,2.0,1.0,1.0,1.0,1.0};
static double f17sol[7]   = {0.0,0.0,0.0,0.0,0.0,0.0,5.65565e-3};

// Osborne 2 function
double ff18(size_t dim, const double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    double f[65];
    double y[65] = {1.366,1.191,1.112,1.013,0.991,0.885,0.831,0.847,0.786,0.725,
                    0.746,0.679,0.608,0.655,0.616,0.606,0.602,0.626,0.651,0.724,
                    0.649,0.649,0.694,0.644,0.624,0.661,0.612,0.558,0.533,0.495,
                    0.500,0.423,0.395,0.375,0.372,0.391,0.396,0.405,0.428,0.429,
                    0.523,0.562,0.607,0.653,0.672,0.708,0.633,0.668,0.645,0.632,
                    0.591,0.559,0.597,0.625,0.739,0.710,0.729,0.720,0.636,0.581,
                    0.428,0.292,0.162,0.098,0.054};
    double t[65];
    size_t m = 65;

    double out = 0.0;
    for (size_t ii = 0; ii < m; ii++){
        t[ii] = (double)(ii)/10.0;
        f[ii] = y[ii] - (x[0] * exp(-t[ii]*x[4]) +
                         x[1]*exp(-pow(t[ii]-x[8],2)*x[5]) +
                         x[2]*exp(-pow(t[ii]-x[9],2)*x[6]) +
                         x[3]*exp(-pow(t[ii]-x[10],2)*x[7]));
        out += f[ii]*f[ii];
    }


    if (grad != NULL){

        for (size_t ii = 0; ii < 11; ii++){
            grad[ii] = 0.0;
        }
        
        for (size_t ii = 0; ii < m; ii++){
            grad[0] -= 2.0 * f[ii] * exp(-t[ii]*x[4]);
            grad[1] -= 2.0 * f[ii] * exp(-pow(t[ii]-x[8],2)*x[5]);
            grad[2] -= 2.0 * f[ii] * exp(-pow(t[ii]-x[9],2)*x[6]);
            grad[3] -= 2.0 * f[ii] * exp(-pow(t[ii]-x[10],2)*x[7]);
            grad[4] -= 2.0 * f[ii] * x[0]*-t[ii]*exp(-t[ii]*x[4]);
            grad[5] -= 2.0 * f[ii] * x[1]*exp(-pow(t[ii]-x[8],2)*x[5])*-pow(t[ii]-x[8],2);
            grad[6] -= 2.0 * f[ii] * x[2]*exp(-pow(t[ii]-x[9],2)*x[6])*-pow(t[ii]-x[9],2);
            grad[7] -= 2.0 * f[ii] * x[3]*exp(-pow(t[ii]-x[10],2)*x[7])*-pow(t[ii]-x[10],2);
            grad[8] -= 2.0 * f[ii] * x[1]*exp(-pow(t[ii]-x[8],2)*x[5]) * -(t[ii]-x[8])*x[5] * -1.0;
            grad[9] -= 2.0 * f[ii] * x[2]*exp(-pow(t[ii]-x[9],2)*x[6]) * -(t[ii]-x[9])*x[6] * -1.0;
            grad[10] -= 2.0 * f[ii] * x[3]*exp(-pow(t[ii]-x[10],2)*x[7]) * -(t[ii]-x[10])*x[7] * -1.0;
        }
    }

    return out;
}

static double f18start[11] = {1.3,0.65,0.65,0.7,0.6,3.0,5.0,7.0,2.0,4.5,5.5};
static double f18sol[12]   = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.01377e-2};

// Watson function
double ff19(size_t dim, const double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    double f[31];
    double t[31];
    double summer[29];
    size_t d = 9;
    
    double out = 0.0;
    for (size_t ii = 0; ii < 29; ii++){
        t[ii] = (double)(ii+1)/29.0;
        f[ii] = 0.0;
        for (size_t jj=2; jj < d+1;jj++){
            f[ii] += (double)(jj-1)*x[jj-1] * pow(t[ii],jj-2);
        }
        summer[ii] = 0.0;
        for (size_t jj = 1; jj < d+1; jj++){
            summer[ii] += x[jj-1]*pow(t[ii],jj-1);
        }
        f[ii] -= pow(summer[ii],2);
        f[ii] -= 1.0;
        out += f[ii]*f[ii];
    }
    f[29] = x[0];
    f[30] = x[1] - pow(x[0],2) - 1.0;
    out += f[29]*f[29] + f[30]*f[30];

    if (grad != NULL){

        for (size_t ii = 0; ii < d; ii++){
            grad[ii] = 0.0;
        }
        
        for (size_t jj = 2; jj < d+1; jj++){
            for (size_t ii = 0; ii < 29; ii++){
                grad[jj-1] += 2*f[ii]*(double)(jj-1) * pow(t[ii],jj-2);
            }
        }
        for (size_t jj = 1; jj < d+1; jj++){
            for (size_t ii = 0; ii < 29; ii++){
                grad[jj-1] -= 2*f[ii]*2 * summer[ii] * pow(t[ii],jj-1);
            }
        }
        grad[0] += 2.0*f[29];
        grad[0] -= 2.0*f[30]*2.0*x[0];
        grad[1] += 2.0*f[30];
    }

    return out;
}

static double f19start[12] = {0.0,0.0,0.0,0.0,0.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0};
static double f19sol[10]   = {0.0,0.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,1.39976e-6}; // minimizer unknown

// Extended Rosenbrock function
double ff20(size_t dim, const double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    size_t m = 30;
    double out = 0.0;
    for (size_t ii = 1; ii <= 15; ii++){
        double f1 = 10.0 * (x[2*ii-1] - pow(x[2*ii-2],2));
        double f2 = 1.0 - x[2*ii-2];
        out += f1*f1 + f2*f2;
    }
    if (grad != NULL){

        for (size_t ii = 0; ii < m; ii = ii+1){
            grad[ii] = 0.0;
        }
        for (size_t ii = 1; ii <= 15; ii++){
            grad[2*ii-2] += 2.0*(1.0 - x[2*ii-2]) * -1.0;
            grad[2*ii-2] += 2.0 * 10.0 * (x[2*ii-1] - pow(x[2*ii-2],2)) * -10.0 * 2.0 * x[2*ii-2];
            grad[2*ii-1] += 2.0 * 10.0 * (x[2*ii-1] - pow(x[2*ii-2],2))*10.0;
        }

    }

    return out;
}

// specified below
static double f20start[30];
static double f20sol[31];

// Extended Powell singular function
double ff21(size_t dim, const double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    size_t m = 400;
    double out = 0.0;
    for (size_t ii = 1; ii <= m/4; ii++){
        double f1 = x[4*ii-4] + 10.0*x[4*ii-3];
        double f2 = pow(5.0,0.5)*(x[4*ii-2] - x[4*ii-1]);
        double f3 = pow(x[4*ii-3]-2.0*x[4*ii-2],2);
        double f4 = pow(10.0,0.5)*pow(x[4*ii-4] - x[4*ii-1],2);
        out += f1*f1 + f2*f2 + f3*f3 + f4*f4;
    }
    if (grad != NULL){

        for (size_t ii = 0; ii < m; ii = ii+1){
            grad[ii] = 0.0;
        }
        for (size_t ii = 1; ii <= m/4; ii++){
            double f1 = x[4*ii-4] + 10.0*x[4*ii-3];
            double f2 = pow(5.0,0.5)*(x[4*ii-2] - x[4*ii-1]);
            double f3 = pow(x[4*ii-3]-2.0*x[4*ii-2],2);
            double f4 = pow(10.0,0.5)*pow(x[4*ii-4] - x[4*ii-1],2);

            grad[4*ii-4] += 2.0*f1 + 2.0*f4 * 2.0*pow(10.0,0.5) *(x[4*ii-4] - x[4*ii-1]);
            grad[4*ii-3] += 2.0*f1*10.0 + 2.0 * f3 * (x[4*ii-3]-2.0*x[4*ii-2])*2.0;
            grad[4*ii-2] += 2.0*f2* pow(5.0,0.5) + 2.0*f3*2.0*(x[4*ii-3]-2.0*x[4*ii-2])*-2.0;
            grad[4*ii-1] += 2.0*f2* pow(5.0,0.5)*-1 + 2.0*f4*pow(10.0,0.5)*-2.0*(x[4*ii-4] - x[4*ii-1]);
        }

    }

    return out;
}

// specified below
static double f21start[400];
static double f21sol[401];

// Penalty I
double ff22(size_t dim, const double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    size_t d = 4;
    double f[11];
    double out = 0.0;

    double a = 1e-5;
    for (size_t ii = 0; ii < d; ii++){
        f[ii] = sqrt(a)*(x[ii]-1);
        out += f[ii]*f[ii];
    }
    f[d] = 0.0;
    for (size_t ii = 0; ii < d; ii++){
        f[d] += pow(x[ii],2);
    }
    f[d] -= 0.25;
    out += f[d]*f[d];
    
    if (grad != NULL){

        for (size_t ii = 0; ii < d; ii++){
            grad[ii] = 2*f[ii]*sqrt(a) + 2.0 * f[d] * x[ii]*2.0;
        }

    }

    return out;
}

/* static double f22start[10] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0}; */
/* static double f22sol[11]   = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,7.08765e-5}; */
static double f22start[4] = {1.0,2.0,3.0,4.0};
static double f22sol[5]   = {0.0,0.0,0.0,0.0,2.24997e-5};

// Penalty II (DOESNT WORK)
double ff23(size_t dim, const double * x, double * grad, void * arg)
{
    (void)(dim);
    (void)(arg);

    size_t d = 4;
    double f[8] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    double y[8];
    double out = 0.0;

    double a = 1e-5;
    /* double a = 1e-1; */

    f[0] = x[0]-0.2;
    out += f[0]*f[0];
    for (size_t ii = 1; ii < d; ii++){
        y[ii] = exp((double)(ii+1)/10.0) + exp((double)(ii)/10.0);
        f[ii] = pow(a,0.5)*(exp(x[ii]/10.0) + exp(x[ii-1]/10.0) - y[ii]);
        out += f[ii]*f[ii];
    }
    
    for (size_t ii = 4; ii < 7; ii++){
    /* for (size_t ii = 4; ii < 5; ii++){ */
        f[ii] += pow(a,0.5)*(exp(x[ii-d+1]/10.0) - exp(-1.0/10.0));
        out += f[ii]*f[ii];
    }
    
    f[7] = -1.0;
    for (size_t jj = 0; jj < d; jj++){
        f[2*d-1] += (double)(d-jj)*pow(x[jj],2);
    }
    out += f[2*d-1]*f[2*d-1];
    
    if (grad != NULL){

        for (size_t ii = 0; ii < d; ii++){
            grad[ii] = 0.0;
        }
        grad[0] += 2.0*f[0];
        for (size_t ii = 1; ii < d; ii++){
            grad[ii]   += 2.0*f[ii]*sqrt(a)*exp(x[ii]/10.0)/10.0;
            grad[ii-1] += 2.0*f[ii]*sqrt(a)*exp(x[ii-1]/10.0)/10.0;
        }
        
        for (size_t ii = d; ii < 2*d-1; ii++){
        /* for (size_t ii = 4; ii < 5; ii++){ */
            grad[ii-d] += 2.0*f[ii]*pow(a,0.5)*exp(x[ii-d+1]/10.0)/10.0;
        }
        
        for (size_t jj = 0; jj < d; jj++){
            grad[jj] += 2.0 * f[2*d-1] * (double)(d-jj)*2.0*x[jj];
        }
    }

    return out;
}

static double f23start[4] = {0.5,0.5,0.5,0.5};
static double f23sol[5]   = {0.0,0.0,0.0,0.0,9.37629e-6};


///////////////////////////////////////////////////////////////////////////
//assign problems

struct UncTestProblem tprobs[34];
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

    tprobs[10].dim   = 3;
    tprobs[10].eval  = ff10;
    tprobs[10].start = f10start;
    tprobs[10].sol   = f10sol;

    tprobs[11].dim   = 3;
    tprobs[11].eval  = ff11;
    tprobs[11].start = f11start;
    tprobs[11].sol   = f11sol;

    tprobs[12].dim   = 4;
    tprobs[12].eval  = ff12;
    tprobs[12].start = f12start;
    tprobs[12].sol   = f12sol;

    tprobs[13].dim   = 4;
    tprobs[13].eval  = ff13;
    tprobs[13].start = f13start;
    tprobs[13].sol   = f13sol;

    tprobs[14].dim   = 4;
    tprobs[14].eval  = ff14;
    tprobs[14].start = f14start;
    tprobs[14].sol   = f14sol;

    tprobs[15].dim   = 4;
    tprobs[15].eval  = ff15;
    tprobs[15].start = f15start;
    tprobs[15].sol   = f15sol;

    tprobs[16].dim   = 5;
    tprobs[16].eval  = ff16;
    tprobs[16].start = f16start;
    tprobs[16].sol   = f16sol;

    tprobs[17].dim   = 6;
    tprobs[17].eval  = ff17;
    tprobs[17].start = f17start;
    tprobs[17].sol   = f17sol;

    tprobs[18].dim   = 11;
    tprobs[18].eval  = ff18;
    tprobs[18].start = f18start;
    tprobs[18].sol   = f18sol;

    tprobs[19].dim   = 9;
    tprobs[19].eval  = ff19;
    tprobs[19].start = f19start;
    tprobs[19].sol   = f19sol;

    for (size_t ii = 0; ii < 15; ii++){
        f20start[2*ii] = -1.2;
        f20start[2*ii+1] = 1;
        f20sol[2*ii] = 1.0;
        f20sol[2*ii+1] = 1.0;
    }
    f20sol[30] = 0.0;
    
    tprobs[20].dim   = 30;
    tprobs[20].eval  = ff20;
    tprobs[20].start = f20start;
    tprobs[20].sol   = f20sol;

    for (size_t ii = 1; ii <= 100; ii++){
        f21start[4*ii-4] = 3.0;
        f21start[4*ii-3] = -1.0;
        f21start[4*ii-2] = 0.0;
        f21start[4*ii-1] = 1.0;
    }
    for (size_t ii = 0; ii < 400; ii++ ){
        f21sol[ii] = 0.0;
    }
    f21sol[400] = 0.0;

    tprobs[21].dim   = 400;
    tprobs[21].eval  = ff21;
    tprobs[21].start = f21start;
    tprobs[21].sol   = f21sol;

    tprobs[22].dim   = 4;
    tprobs[22].eval  = ff22;
    tprobs[22].start = f22start;
    tprobs[22].sol   = f22sol;

    tprobs[23].dim   = 4;
    tprobs[23].eval  = ff23;
    tprobs[23].start = f23start;
    tprobs[23].sol   = f23sol;

}


// other functions
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

double unc_test_problem_eval(size_t dim, const double * x,double * grad,void *arg)
{
    struct UncTestProblem * p = arg;
    
    return p->eval(dim,x,grad,arg);
}





