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

#include "array.h"
#include "CuTest.h"
#include "quadrature.h"

struct counter{
    int N;
};

void Test_cheb_gauss(CuTest * tc){

    printf("Testing function: cheb_gauss\n");

    size_t N = 2;
    size_t N2 = 3;
    double p1[2];
    double w1[2];
    double p2[3];
    double w2[3];

    double p1s[2] = { -0.5 * sqrt(2), 0.5 * sqrt(2) };
    double w1s[2] = { 0.5 * M_PI, 0.5 * M_PI };
    cheb_gauss(N,p1,w1);

    CuAssertDblEquals(tc, p1s[1], p1[0], 1e-14);
    CuAssertDblEquals(tc, p1s[0], p1[1], 1e-14);
    CuAssertDblEquals(tc, w1s[0], w1[0], 1e-14);
    CuAssertDblEquals(tc, w1s[1], w1[1], 1e-14);

    double p2s[3] = {-0.5 * sqrt(3), 0.0, 0.5 * sqrt(3)};
    double w2s[3] = {M_PI/3.0, M_PI/3.0, M_PI/3.0};
    cheb_gauss(N2,p2,w2);

    CuAssertDblEquals(tc, p2s[2], p2[0], 1e-14);
    CuAssertDblEquals(tc, p2s[1], p2[1], 1e-14);
    CuAssertDblEquals(tc, p2s[0], p2[2], 1e-14);
    CuAssertDblEquals(tc, w2s[0], w2[0], 1e-14);
    CuAssertDblEquals(tc, w2s[1], w2[1], 1e-14);
    CuAssertDblEquals(tc, w2s[2], w2[2], 1e-14);
}

void Test_cc_nestedness(CuTest * tc){

    printf("Logic test: are cc nodes nested?\n");
    double p[9];
    double w[9];
    clenshaw_curtis(9,p,w);
    //printf("9 points \n");
    //dprint(9, p);

    double p2[17];
    double w2[17];
    clenshaw_curtis(17,p2,w2);
    //printf("17 points \n");
   // dprint(17, p2);
    
    size_t ii;
    for (ii = 0; ii < 9; ii++){
        CuAssertDblEquals(tc,p[ii],p2[2*ii],1e-13);
    }
}

void Test_clenshaw_curtis(CuTest * tc){
    printf("Testing function: clenshaw_curtis\n");

    size_t ii;
    double int1;
    size_t N = 2;
    double p1[2];
    double w1[2];

    clenshaw_curtis(N,p1,w1);
    int1 = 0.0;
    for (ii = 0; ii < N; ii++){
        int1 += w1[ii] * p1[ii];
    }
    CuAssertDblEquals(tc, 0.0, int1, 1e-14);

    size_t N2 = 7;
    double p2[7];
    double w2[7];
    clenshaw_curtis(N2,p2,w2);
    int1 = 0.0;
    for (ii = 0; ii < N2; ii++){
        int1 += w2[ii] * pow(p2[ii],6);
    }
    CuAssertDblEquals(tc, 2.0/7.0, int1, 1e-14);
}

void Test_fejer_nestedness(CuTest * tc){

    printf("Logic test: are fejer2 nodes nested?\n");
    double p[9];
    double w[9];
    fejer2(9,p,w);
    //printf("9 points \n");
    //dprint(9, p);

    double p2[19];
    double w2[19];
    fejer2(19,p2,w2);
    //printf("17 points \n");
   // dprint(17, p2);
    
    size_t ii;
    for (ii = 0; ii < 9; ii++){
        CuAssertDblEquals(tc,p[ii],p2[2*ii+1],1e-13);
    }
}

void Test_fejer2(CuTest * tc){
    printf("Testing function: fejer2\n");

    size_t ii;
    double int1;
    size_t N = 2;
    double p1[2];
    double w1[2];

    fejer2(N,p1,w1);
    int1 = 0.0;
    for (ii = 0; ii < N; ii++){
        int1 += w1[ii] * p1[ii];
    }
    CuAssertDblEquals(tc, 0.0, int1, 1e-14);

    size_t N2 = 7;
    double p2[7];
    double w2[7];
    clenshaw_curtis(N2,p2,w2);
    int1 = 0.0;
    for (ii = 0; ii < N2; ii++){
        int1 += w2[ii] * pow(p2[ii],6);
    }
    CuAssertDblEquals(tc, 2.0/7.0, int1, 1e-14);
}

void Test_gauss_legendre(CuTest * tc){
    printf("Testing function: gauss_legendre\n");
    size_t N = 2;
    size_t N2 = 3;
    double p1[2];
    double w1[2];
    double p2[3];
    double w2[3];

    //size_t N3 = 4;
    //double p3[4];
    //double w3[4];

    double p1s[2] = { sqrt(3.0)/3.0, -sqrt(3.0)/3.0 };
    double w1s[2] = { 1.0, 1.0 };
    gauss_legendre(N,p1,w1);

    CuAssertDblEquals(tc, p1s[1], p1[0], 1e-14);
    CuAssertDblEquals(tc, p1s[0], p1[1], 1e-14);
    CuAssertDblEquals(tc, w1s[0]/2.0, w1[0], 1e-14);
    CuAssertDblEquals(tc, w1s[1]/2.0, w1[1], 1e-14);
    
    double p2s[3] = {sqrt(15.0)/5.0, 0.0, -sqrt(15.0)/5.0};
    double w2s[3] = {5.0/9.0, 8.0/9.0, 5.0/9.0};
    gauss_legendre(N2,p2,w2);

    CuAssertDblEquals(tc, p2s[2], p2[0], 1e-14);
    CuAssertDblEquals(tc, p2s[1], p2[1], 1e-14);
    CuAssertDblEquals(tc, p2s[0], p2[2], 1e-14);
    CuAssertDblEquals(tc, w2s[0]/2.0, w2[0], 1e-14);
    CuAssertDblEquals(tc, w2s[1]/2.0, w2[1], 1e-14);
    CuAssertDblEquals(tc, w2s[2]/2.0, w2[2], 1e-14);

    //gauss_legendre(N3,p3,w3);
    
    //printf("pts = "); dprint(N3,p3);
    //printf("wts = "); dprint(N3, w3);
    //double p3s[4] = {sqrt(15.0 - )/5.0, 0.0, -sqrt(15.0)/5.0};
    //double w3s[4] = {5.0/9.0, 8.0/9.0, 5.0/9.0};
}


void Test_gauss_hermite(CuTest * tc){
    printf("Testing function: gauss_hermite\n");
    size_t N = 10;
    double p1[10];
    double w1[10];
    
    gauss_hermite(N,p1,w1);

    /* printf("x = "); dprint(N,p1); */
    /* printf("w = "); dprint(N,w1); */
    double val = 0.0;
    double val2 = 0.0;
    for (size_t ii = 0; ii < N; ii++){
        val += w1[ii]*p1[ii]*p1[ii]*p1[ii];
        val2 += w1[ii]*p1[ii]*p1[ii];
    }

    CuAssertDblEquals(tc,0.0, val, 1e-14);
    CuAssertDblEquals(tc,1, val2, 1e-14);

}

CuSuite * QuadGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_cheb_gauss);
    SUITE_ADD_TEST(suite, Test_cc_nestedness);
    SUITE_ADD_TEST(suite, Test_clenshaw_curtis);
    SUITE_ADD_TEST(suite, Test_fejer_nestedness);
    SUITE_ADD_TEST(suite, Test_fejer2);
    SUITE_ADD_TEST(suite, Test_gauss_legendre);
    SUITE_ADD_TEST(suite, Test_gauss_hermite);
    return suite;
}




void RunAllTests(void) {
    
    printf("Running Test Suite: lib_quadrature\n");

    CuString * output = CuStringNew();
    CuSuite * suite = CuSuiteNew();
    
    CuSuite * quad = QuadGetSuite();

    CuSuiteAddSuite(suite, quad);

    CuSuiteRun(suite);
    CuSuiteSummary(suite, output);
    CuSuiteDetails(suite, output);
    printf("%s \n", output->buffer);
    
    CuSuiteDelete(quad);

    CuStringDelete(output);
    free(suite);
}

int main(void) {
    RunAllTests();
}
