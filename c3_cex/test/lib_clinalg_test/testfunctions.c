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




// first function

#include "testfunctions.h"
#include <assert.h>
#include <math.h>

int func(size_t n, const double * x, double * out, void * args)
{
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = 1.0 + 0.0*x[ii];
    }
    if (args != NULL){
        int * N = args;
        *N += n;
    }
    return 0;
}

int func2(size_t n, const double * x, double * out, void * args)
{
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = x[ii];
    }
    if (args != NULL){
        int * N = args;
        *N += n;
    }
    return 0;
}

int funcp(size_t n, const double * x, double * out, void * args)
{
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = sin(2*3.14159*x[ii]);
    }
    if (args != NULL){
        int * N = args;
        *N += n;
    }
    return 0;
}

int func3(size_t n, const double * x, double * out, void * args)
{
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = pow(x[ii],2.0) + sin(3.14159*x[ii]);
    }
    if (args != NULL){
        int * N = args;
        *N += n;
    }
    return 0;
}

int func2p(size_t n, const double * x, double * out, void * args)
{
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = sin(pow(x[ii],2.0));
    }
    if (args != NULL){
        int * N = args;
        *N += n;
    }
    return 0;
}

int func4(size_t n, const double * x, double * out, void * args)
{
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = 3.0*pow(x[ii],4.0) - 2.0*pow(x[ii],2.0);
    }
    if (args != NULL){
        int * N = args;
        *N += n;
    }
    return 0;
}

int func3p(size_t n, const double * x, double * out, void * args)
{
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = sin(3.0*pow(x[ii],4.0) - 2.0*pow(x[ii],2.0));
    }
    if (args != NULL){
        int * N = args;
        *N += n;
    }
    return 0;
}

int func5(size_t n, const double * x, double * out, void * args)
{
    for (size_t ii = 0; ii < n; ii++){
        //return 3.0*cos(M_PI*x) - 2.0*pow(x,0.5);
        out[ii] = x[ii];
    }
    if (args != NULL){
        int * N = args;
        *N += n;
    }
    return 0;
}

int func6(size_t n, const double * x, double * out, void * args)
{
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = exp(5.0*x[ii]);
    }
    if (args != NULL){
        int * N = args;
        *N += n;
    }
    return 0;
}


int funcnda(size_t n, const double * x, double * out, void * args)
{
    (void)(args);
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = x[ii];
    }
    
    return 0;
}

int funcndb(size_t n, const double * x, double * out, void * args)
{
    (void)(args);
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = pow(x[ii],2);
    }

    return 0;
}

// two dimensions
int funcnd1(size_t n, const double * x, double * out, void * args)
{
    (void)(args);

    for (size_t ii = 0; ii < n; ii++){
        double vala, valb;
        funcnda(1,x+ii*2,&vala,NULL);
        funcndb(1,x+ii*2+1,&valb,NULL);
        out[ii] = vala + valb;
    }

    return 0;
}

int funcnd2(size_t n, const double * x, double * out, void * args)
{
    (void)(args);

    for (size_t ii = 0; ii < n; ii++){
        out[ii] = x[ii*4] + x[ii*4+1] + x[ii*4+2] + x[ii*4+3];        
    }
    return 0;
}

int disc2d(size_t n, const double * xy, double * out, void * args)
{
    (void)(args);
     
    for (size_t ii = 0; ii < n; ii++){
        double x = xy[ii*2];
        double y = xy[ii*2+1];
        if ((x > 0.5) || (y > 0.4)){
            out[ii] = 0.0;    
        }
        else{
            out[ii] = exp(5.0 * x + 5.0 * y);
        }
    }
    return 0;
}

//4 dimensional
int funcH4(size_t n, const double * x, double * out, void * args)
{
    (void)(args);
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = 2.0*x[ii*4] + x[ii*4+1]*pow(x[ii*4+2],4) +  x[ii*4+3]*pow(x[ii*4+0],2);
    }
    return 0;
}

// 4 dimensional
int funch1(size_t n, const double * x, double * out, void * arg)
{
    (void)(arg);
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = x[ii*4+0] + x[ii*4+1] + x[ii*4+2] + x[ii*4+3];
    }

    return 0;
}

// 4 dimensional
int funch2(size_t n, const double * x, double * out, void * arg)
{
    (void)(arg);
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = x[ii*4+0]*x[ii*4+1] + x[ii*4+2]*x[ii*4+3] +
            x[ii*4+1]*pow(x[ii*4+2],5) + pow(x[ii*4+1],8)*pow(x[ii*4+3],2);        
    }

//    double out = x[0]*x[1] + x[2]*x[3] + x[0]*exp(-x[2]*x[3]);
//    double out = x[0]*x[1] + x[2]*x[3] + x[0]*sin(x[1]*x[2]);
    /* printf("x = "); dprint(4,x); */
    /* printf("out = %G\n",out); */
    return 0;
}


// 6 dimensional
int func_not_all(size_t n, const double * x, double * out, void * args)
{
    (void)(args);
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = x[ii*6+1] + x[ii*6+4];
    }
    return 0;
}


int sin10d(size_t n, const double * x, double * out, void * args)
{
    
    (void)(args);

    for (size_t jj = 0; jj < n; jj++ ){
        out[jj] = 0.0;
        for (size_t ii = 0; ii < 10; ii++){
            out[jj] += x[jj*10+ii];
        }
        out[jj] = sin(out[jj]);        
    }
    return 0;
}

int sin100d(size_t n, const double * x, double * out, void * args)
{
    
    (void)(args);

    for (size_t jj = 0; jj < n; jj++ ){
        out[jj] = 0.0;
        for (size_t ii = 0; ii < 100; ii++){
            out[jj] += x[jj*100+ii];
        }
        out[jj] = sin(out[jj]);        
    }
    return 0;
}

int sin1000d(size_t n, const double * x, double * out, void * args)
{
    
    (void)(args);

    for (size_t jj = 0; jj < n; jj++ ){
        out[jj] = 0.0;
        for (size_t ii = 0; ii < 1000; ii++){
            out[jj] += x[jj*1000+ii];
        }
        out[jj] = sin(out[jj]);        
    }
    return 0;
}

int funcGrad(size_t n, const double * x, double * out, void * args)
{
    (void)(args);
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = x[ii*4+0] * x[ii*4+1] + x[ii*4+2] * x[ii*4+3];
    }
    return 0;
}

// 3 dimensional
int funcHess(size_t n, const double * x, double * out, void * args)
{
    (void)(args);
    //double out = 2.0*x[0] + x[1]*pow(x[2],4) + x[3]*pow(x[0],2);
    for (size_t ii = 0; ii < n; ii++){
        out[ii] =  x[ii*3+0] + pow(x[ii*3+0],2)*x[ii*3+2] +  x[ii*3+1] * pow(x[ii*3+2],4);// + x[3]*pow(x[0],2);
    }
    return 0;
}


//4 dimensional
int funcCheck2(size_t n, const double * x, double * out, void * args)
{
    (void)(args);
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = pow(x[ii*4+ 0] * x[ii*4 + 1],2) + x[ii*4 + 2] * x[ii*4+3] +
                  x[ii*4+1]*sin(x[ii*4+3]); 
    }
    return 0;
}

//5 dimensional
int funcCheck3(size_t n, const double * x, double * out, void * args)
{
    (void)(args);
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = sin(x[ii*5 + 0] +  x[ii*5 + 1] + x[ii*5 + 2]  +  x[ii*5 + 3] +
                      x[ii*5 + 4]);
    }
    return 0;
}


int quad2d(size_t N, const double * x,double * out, void * arg)
{
    (void)(arg);
    (void)(x);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = x[2*ii+0]*x[2*ii+0] + x[2*ii+1]*x[2*ii+1];
        /* out[ii] = 0.2; */
    }
    return 0;
}

int gauss2d(size_t n, const double *xin, double * out, void * args)
{
    (void) args;
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = exp(-pow(xin[ii*2 + 0],2) / 0.1 - pow(xin[ii*2+1],2) * 0.1);
    }

    return 0;
}
