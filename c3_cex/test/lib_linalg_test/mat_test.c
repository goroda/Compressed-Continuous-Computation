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
#include "matrix_util.h"

#include "CuTest.h"

void Test_v2m(CuTest * tc)
{

    printf("Testing Function: v2m\n");
    
    size_t ii;
    size_t N = 4;
    double val[4] = { 1.0, 2.0, 3.0, 4.0};
    struct mat * m1 = v2m(N,val,1);
    CuAssertIntEquals(tc,1,m1->nrows);
    CuAssertIntEquals(tc,4,m1->ncols);
    
    for (ii = 0; ii < 4; ii++){
        CuAssertDblEquals(tc, val[ii], m1->vals[ii], 1e-14);
    }

    struct mat * m2 = v2m(N,val,0);
    CuAssertIntEquals(tc,4,m2->nrows);
    CuAssertIntEquals(tc,1,m2->ncols);
    for (ii = 0; ii < 4; ii++){
        CuAssertDblEquals(tc, val[ii], m2->vals[ii], 1e-14);
    }

    freemat(m1);
    freemat(m2);
}

void Test_diagv2m(CuTest * tc)
{

    printf("Testing Function: diagv2m\n");
    
    int ii, jj;
    double val[4] = { 1.0, 2.0, 3.0, 4.0};
    struct mat * m1 = diagv2m(4,val);
    CuAssertIntEquals(tc,4,m1->nrows);
    CuAssertIntEquals(tc,4,m1->ncols);
    
    for (ii = 0; ii < 4; ii++){
        for (jj = 0; jj < 4; jj++){
            if (ii != jj){
                CuAssertDblEquals(tc, 0.0, m1->vals[ii*4+jj], 1e-14);
            }
            else{
                CuAssertDblEquals(tc, val[ii], m1->vals[ii*4 + ii], 1e-14);
            } 
        }
    }

    freemat(m1);
}

CuSuite * MatrixUtilGetSuite(){
    //printf("Generating Suite: MatrixUtil\n");
    //printf("----------------------------\n");

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_v2m);
    SUITE_ADD_TEST(suite, Test_diagv2m);
    return suite;
}
