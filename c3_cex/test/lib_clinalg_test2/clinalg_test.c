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




#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "array.h"
#include "CuTest.h"

#include "lib_funcs.h"
#include "lib_clinalg.h"
#include "lib_linalg.h"

// 5 dimensional
double func1(double * x, void * args)
{
    assert ( args == NULL );
    return x[0] + x[2] + x[4];
}

double func2(double * x, void * args)
{
    assert ( args == NULL );
    return exp(x[0]*x[1]);
}

void Test_c3axpy(CuTest * tc)
{
    printf("Testing Function: c3axpy\n");
    struct BoundingBox * bds = bounding_box_init_std(5);
    struct FunctionTrain * ft1 = function_train_cross(func1,NULL,bds,NULL,NULL,NULL);
    struct FunctionTrain * ft2 = function_train_cross(func2,NULL,bds,NULL,NULL,NULL);
    
    c3axpy(3.0,ft1,&ft2,1e-10);
    
    size_t ii,jj, N = 10000;
    double pt[5];
    double sum = 0.0;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < 5; jj++){
            pt[jj] = randu()*2.0-1.0;
        }
        double val = func1(pt,NULL)*3.0 + func2(pt,NULL);
        double vala = function_train_eval(ft2,pt);
        double diff = val-vala;
        sum += pow(diff,2);
    }

    CuAssertDblEquals(tc, 0.0, sum / N, 1e-9);
    bounding_box_free(bds); bds = NULL;
    function_train_free(ft1); ft1 = NULL;
    function_train_free(ft2); ft2 = NULL;
}

CuSuite * CLinalgSuite()
{
    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_c3axpy);
    return suite;
}

void RunAllTests(void) {
    
    printf("Running Test Suite: lib_clinalg\n");

    CuString * output = CuStringNew();
    CuSuite * suite = CuSuiteNew();
    
    CuSuite * clin = CLinalgSuite();
    CuSuiteAddSuite(suite, clin);
    CuSuiteRun(suite);
    CuSuiteSummary(suite, output);
    CuSuiteDetails(suite, output);
    printf("%s \n", output->buffer);
    
    CuSuiteDelete(clin);
    CuStringDelete(output);
    free(suite);
}

int main(void) {
    RunAllTests();
}
