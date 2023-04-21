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

#include "array.h"
#include "CuTest.h"

void TestLinspace(CuTest * tc) {
    printf("Testing Function: linspace\n");
    
    size_t N = 5;
    double min = 0.0;
    double max  = 1.0;
    double * actual = linspace(min,max,N);
    double expected[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
    size_t ii;
    for (ii = 0; ii < N; ii++){
        CuAssertDblEquals(tc, expected[ii], actual[ii], 1e-14);
    }

    free(actual);
}

void TestArange(CuTest * tc) {
    printf("Testing Function: arange\n");

    size_t N;
    double start = 0.0;
    double stop  = 1.0;
    double step = 0.3;
    double * actual = arange(start,stop,step,&N);
    double expected[4] = {0.0, 0.3, 0.6, 0.9};
    size_t ii;
    CuAssertIntEquals(tc,4,N);
    for (ii = 0; ii < 4; ii++){
        CuAssertDblEquals(tc, expected[ii], actual[ii], 1e-14);
    }

    free(actual);
}

void Test_dprod(CuTest * tc) {
    printf("Testing Function: dprod\n");

    double vals[4] = { 1.0, 2.0, 3.0, 4.0 };
    double actual = dprod(4,vals);
    double expected = 24.0;
    CuAssertDblEquals(tc, expected, actual, 1e-14);
}

void Test_iprod(CuTest * tc) {
    printf("Testing Function: iprod\n");

    int vals[4] = { 1, 2, 3, 4 };
    int actual = iprod(4,vals);
    int expected = 24;
    CuAssertIntEquals(tc, expected, actual);
}

CuSuite * VectorUtilGetSuite(){
    //printf("Generating Suite: VectorUtil\n");
    //printf("----------------------------\n");

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, TestLinspace);
    SUITE_ADD_TEST(suite, TestArange);
    SUITE_ADD_TEST(suite, Test_dprod);
    SUITE_ADD_TEST(suite, Test_iprod);
    return suite;
}

void RunAllTests(void) {
    
    printf("Running Test Suite: lib_array\n");

    CuString * output = CuStringNew();
    CuSuite * suite = CuSuiteNew();
    
    CuSuite * vec = VectorUtilGetSuite();
    CuSuiteAddSuite(suite, vec);
    CuSuiteRun(suite);
    CuSuiteSummary(suite, output);
    CuSuiteDetails(suite, output);
    printf("%s \n", output->buffer);
    
    CuSuiteDelete(vec);
    CuStringDelete(output);
    free(suite);
   
}

int main(void) {
    RunAllTests();
}
