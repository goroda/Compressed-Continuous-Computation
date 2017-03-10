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

#include "CuTest.h"
#include "array.h"
#include "linalg.h"
#include "tensor.h"

void Test_stack2h(CuTest * tc){

    printf("Testing function: tensor_stack2h_3d\n");

    size_t ii,jj,kk;
    size_t nvals_a[3];
    nvals_a[0] = 3;
    nvals_a[1] = 2;
    nvals_a[2] = 4;
    struct tensor * a; 
    init_tensor(&a, 3, nvals_a);
    for (ii = 0; ii < nvals_a[0]*nvals_a[1]*nvals_a[2]; ii++){
        a->vals[ii] = randu();
    }

    size_t nvals_b[3];
    nvals_b[0] = 3;
    nvals_b[1] = 2;
    nvals_b[2] = 6;
    struct tensor * b; 
    init_tensor(&b, 3, nvals_b);
    for (ii = 0; ii < nvals_b[0]*nvals_b[1]*nvals_b[2]; ii++){
        b->vals[ii] = randu();
    }
    
    struct tensor * c = tensor_stack2h_3d(a,b);
    
    
    CuAssertIntEquals(tc, 3, c->dim);
    CuAssertIntEquals(tc, nvals_a[0], c->nvals[0]);
    CuAssertIntEquals(tc, nvals_a[1], c->nvals[1]);
    CuAssertIntEquals(tc, nvals_a[2] + nvals_b[2], c->nvals[2]);
    
    size_t elem1[3];
    size_t elem2[3];
    double v1;
    double v2;
    for (ii = 0; ii < nvals_a[0]; ii++){
        elem1[0] = ii; elem2[0] = ii;
        for (jj = 0; jj < nvals_a[1]; jj++){
            elem1[1] = jj; elem2[1] = jj;
            for (kk = 0; kk < nvals_a[2]; kk++){
                elem1[2] = kk; 
                v1 = tensor_elem(c,elem1);
                v2 = tensor_elem(a,elem1);
                CuAssertDblEquals(tc, v2, v1, 1e-14);
            }
            for (kk = nvals_a[2]; kk < c->nvals[2]; kk++){
                elem2[2] = kk - nvals_a[2];
                elem1[2] = kk;
                v1 = tensor_elem(c,elem1);
                v2 = tensor_elem(b,elem2);
                CuAssertDblEquals(tc, v2, v1, 1e-14);
            }
        }
    }

    free_tensor(&a);
    free_tensor(&b);
    free_tensor(&c);
}

void Test_stack2v(CuTest * tc){

    printf("Testing function: tensor_stack2v_3d\n");

    size_t ii,jj,kk;
    size_t nvals_a[3];
    nvals_a[0] = 3;
    nvals_a[1] = 2;
    nvals_a[2] = 4;
    struct tensor * a; 
    init_tensor(&a, 3, nvals_a);
    for (ii = 0; ii < nvals_a[0]*nvals_a[1]*nvals_a[2]; ii++){
        a->vals[ii] = randu();
    }

    size_t nvals_b[3];
    nvals_b[0] = 9;
    nvals_b[1] = 2;
    nvals_b[2] = 4;
    struct tensor * b; 
    init_tensor(&b, 3, nvals_b);
    for (ii = 0; ii < nvals_b[0]*nvals_b[1]*nvals_b[2]; ii++){
        b->vals[ii] = randu();
    }
    
    struct tensor * c = tensor_stack2v_3d(a,b);
    
    CuAssertIntEquals(tc, 3, c->dim);
    CuAssertIntEquals(tc, nvals_a[0] + nvals_b[0], c->nvals[0]);
    CuAssertIntEquals(tc, nvals_a[1], c->nvals[1]);
    CuAssertIntEquals(tc, nvals_a[2], c->nvals[2]);
    
    size_t elem1[3];
    size_t elem2[3]; 
    double v1;
    double v2;
    for (ii = 0; ii < nvals_a[0]; ii++){
        elem1[0] = ii; 
        for (jj = 0; jj < nvals_a[1]; jj++){
            elem1[1] = jj; 
            for (kk = 0; kk < nvals_a[2]; kk++){
                elem1[2] = kk; 
                v1 = tensor_elem(c,elem1);
                v2 = tensor_elem(a,elem1);
                CuAssertDblEquals(tc, v2, v1, 1e-14);
            }
        }
    }
    for (ii = nvals_a[0]; ii < c->nvals[0]; ii++){
        elem1[0] = ii-nvals_a[0]; elem2[0] = ii;
        for (jj = 0; jj < nvals_a[1]; jj++){
            elem1[1] = jj; elem2[1] = jj;
            for (kk = 0; kk < nvals_a[2]; kk++){
                elem1[2] = kk; elem2[2] = kk;
                v1 = tensor_elem(c,elem2);
                v2 = tensor_elem(b,elem1);
                CuAssertDblEquals(tc, v2, v1, 1e-14);
            }
        }
    }

    free_tensor(&a);
    free_tensor(&b);
    free_tensor(&c);
}

void Test_blockdiag(CuTest * tc){

    printf("Testing function: tensor_blockdiag_3d\n");

    size_t ii,jj,kk;
    size_t nvals_a[3];
    nvals_a[0] = 7;
    nvals_a[1] = 2;
    nvals_a[2] = 4;
    struct tensor * a; 
    init_tensor(&a, 3, nvals_a);
    for (ii = 0; ii < nvals_a[0]*nvals_a[1]*nvals_a[2]; ii++){
        a->vals[ii] = randu();
    }

    size_t nvals_b[3];
    nvals_b[0] = 10;
    nvals_b[1] = 2;
    nvals_b[2] = 9;
    struct tensor * b; 
    init_tensor(&b, 3, nvals_b);
    for (ii = 0; ii < nvals_b[0]*nvals_b[1]*nvals_b[2]; ii++){
        b->vals[ii] = randu();
    }
    
    struct tensor * c = tensor_blockdiag_3d(a,b);
    
    CuAssertIntEquals(tc, 3, c->dim);
    CuAssertIntEquals(tc, nvals_a[0] + nvals_b[0], c->nvals[0]);
    CuAssertIntEquals(tc, nvals_a[1], c->nvals[1]);
    CuAssertIntEquals(tc, nvals_a[2] + nvals_b[2], c->nvals[2]);

    size_t elem1[3];
    size_t elem2[3]; // c tensor
    double v1;
    double v2;
    for (ii = 0; ii < nvals_a[0]; ii++){
        elem1[0] = ii;  elem2[0] = ii;
        for (jj = 0; jj < nvals_a[1]; jj++){
            elem1[1] = jj;  elem2[1] = jj;
            for (kk = 0; kk < nvals_a[2]; kk++){
                elem1[2] = kk;  elem2[2] = kk;
                v1 = tensor_elem(c,elem2);
                v2 = tensor_elem(a,elem1);
                CuAssertDblEquals(tc, v2, v1, 1e-14);
            }
            for (kk = nvals_a[2]; kk < c->nvals[2]; kk++){
                elem2[2] = kk;
                v1 = tensor_elem(c,elem2);
                CuAssertDblEquals(tc, 0.0, v1, 1e-14);
            }
        }
    }
    for (ii = nvals_a[0]; ii < c->nvals[0]; ii++){
        elem1[0] = ii-nvals_a[0]; elem2[0] = ii;
        for (jj = 0; jj < nvals_a[1]; jj++){
            elem1[1] = jj; elem2[1] = jj;
            for (kk = 0; kk < a->nvals[2]; kk++){
                elem2[2] = kk;
                v1 = tensor_elem(c,elem2);
                CuAssertDblEquals(tc, 0.0, v1, 1e-14);
            }
            for (kk = nvals_a[2]; kk < c->nvals[2]; kk++){
                elem1[2] = kk-nvals_a[2]; elem2[2] = kk;
                v1 = tensor_elem(c,elem2);
                v2 = tensor_elem(b,elem1);
                CuAssertDblEquals(tc, v2, v1, 1e-14);
            }
        }
    }

    free_tensor(&a);
    free_tensor(&b);
    free_tensor(&c);
}

CuSuite * TensorGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_stack2h);
    SUITE_ADD_TEST(suite, Test_stack2v);
    SUITE_ADD_TEST(suite, Test_blockdiag);
    return suite;
}

void RunAllTests(void) {
    
    printf("Running Test Suite: lib_tensor\n");

    CuString * output = CuStringNew();
    CuSuite * suite = CuSuiteNew();
    
    CuSuite * tensor = TensorGetSuite();
    CuSuiteAddSuite(suite, tensor);
    CuSuiteRun(suite);
    CuSuiteSummary(suite, output);
    CuSuiteDetails(suite, output);
    printf("%s \n", output->buffer);
    
    CuSuiteDelete(tensor);
    CuStringDelete(output);
    free(suite);
   
}

int main(void) {
    RunAllTests();
}
