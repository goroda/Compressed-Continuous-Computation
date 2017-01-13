// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016, Sandia Corporation

// This file is part of the Compressed Continuous Computation (C3) Library
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


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>

#include "CuTest.h"
#include "testfunctions.h"

#include "array.h"
#include "lib_linalg.h"

#include "lib_funcs.h"


void Test_gauss_eval(CuTest * tc){
    
    printf("Testing functions: gauss_kernel_eval(deriv) \n");

    double scale = 1.2;
    double width = 0.2;
    double center = -0.3;

    double h = 1e-8;
    double x = 0.2;
    double xh = x+h;
    double valh = gauss_kernel_eval(scale,width*width,center,xh);
    double x2h = x-h;
    double val2h = gauss_kernel_eval(scale,width*width,center,x2h);

    double numerical_deriv = (valh-val2h)/(2.0*h);
    double analytical_deriv = gauss_kernel_deriv(scale,width*width,center,x);
    CuAssertDblEquals(tc,numerical_deriv,analytical_deriv,1e-5);
}


void Test_kernel_expansion_mem(CuTest * tc)
{

    printf("Testing functions: kernel allocation and deallocation memory \n");
    
    double scale = 1.2;
    double width = 0.2;
    size_t N = 20;
    double * centers = linspace(-1,1,N);

    struct KernelExpansion * ke = kernel_expansion_alloc(1);
    CuAssertIntEquals(tc,1,ke!=NULL);
    for (size_t ii = 0; ii < N; ii++){
        struct Kernel * kern = kernel_gaussian(scale,width*width,centers[ii]);
        kernel_expansion_add_kernel(ke,0.0,kern);
        kernel_free(kern); kern = NULL;
    }
    
    kernel_expansion_free(ke); ke = NULL;

    free(centers); centers = NULL;
}

void Test_kernel_expansion_copy(CuTest * tc)
{

    printf("Testing functions: kernel_expansion_copy \n");
    
    double scale = 1.2;
    double width = 0.2;
    size_t N = 20;
    double * centers = linspace(-1,1,N);

    struct KernelExpansion * ke = kernel_expansion_alloc(1);
    for (size_t ii = 0; ii < N; ii++){
        struct Kernel * kern = kernel_gaussian(scale,width*width,centers[ii]);
        kernel_expansion_add_kernel(ke,randu(),kern);
        kernel_free(kern); kern = NULL;
    }

    struct KernelExpansion * ke2 = kernel_expansion_copy(ke);

    double * x = linspace(-1,1,200);
    for (size_t ii = 0; ii < 200; ii++){
        double val1 = kernel_expansion_eval(ke,x[ii]);
        double val2 = kernel_expansion_eval(ke2,x[ii]);
        CuAssertDblEquals(tc,val1,val2,1e-15);
    }

    free(x); x = NULL;
    
    kernel_expansion_free(ke2); ke2 = NULL;
    kernel_expansion_free(ke); ke = NULL;

    free(centers); centers = NULL;
}

void Test_serialize_kernel_expansion(CuTest * tc){
    
    printf("Testing functions: serialize_kernel_expansion \n");
    double scale = 1.2;
    double width = 0.2;
    size_t N = 18;
    double * centers = linspace(-1,1,N);

    struct KernelExpansion * ke = kernel_expansion_alloc(1);
    for (size_t ii = 0; ii < N; ii++){
        struct Kernel * kern = kernel_gaussian(scale,width*width,centers[ii]);
        kernel_expansion_add_kernel(ke,randu(),kern);
        kernel_free(kern); kern = NULL;
    }

    unsigned char * text = NULL;
    size_t size_to_be;
    serialize_kernel_expansion(text, ke, &size_to_be);
    text = malloc(size_to_be * sizeof(char));

    serialize_kernel_expansion(text, ke, NULL);

    struct KernelExpansion * k2 = NULL;
    deserialize_kernel_expansion(text, &k2);
    free(text); text = NULL;
    
    double * x = linspace(-1,1,200);
    for (size_t ii = 0; ii < 200; ii++){
        double val1 = kernel_expansion_eval(ke,x[ii]);
        double val2 = kernel_expansion_eval(k2,x[ii]);
        CuAssertDblEquals(tc,val1,val2,1e-15);
    }
    free(x); x = NULL;
    
    kernel_expansion_free(k2); k2 = NULL;
    kernel_expansion_free(ke); ke = NULL;

    free(centers); centers = NULL;
}



CuSuite * KernGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_gauss_eval);
    SUITE_ADD_TEST(suite, Test_kernel_expansion_mem);
    SUITE_ADD_TEST(suite, Test_kernel_expansion_copy);
    SUITE_ADD_TEST(suite, Test_serialize_kernel_expansion);

    return suite;
}
