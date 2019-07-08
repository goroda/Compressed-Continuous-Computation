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
#include <string.h>
#include <math.h>

#include "CuTest.h"
#include "array.h"
#include "tensortrain.h"
#include "tt_multilinalg.h"
#include "tt_integrate.h"
#include "candecomp.h"
#include "convert_decomp.h"
#include "quadrature.h"
#include "cross.h"


struct counter{
    int N;
};

void Test_tt_integrate(CuTest * tc)
{
    printf("Testing Functions: tt_integrate\n");
    size_t N1, N2, N3;
    N1 = 10;
    N2 = 11;
    N3 = 12;
    double * x = linspace(0.0,1.0,N1);
    double * y = linspace(0.0,1.0,N2);
    double * z = linspace(0.0,1.0,N3);
    double h1 = x[1]-x[0];
    double h2 = y[1]-y[0];
    double h3 = z[1]-z[0];

    struct mat **cores = malloc(sizeof(struct mat )*3);
    cores[0] = v2m(N1,x,0);
    cores[1] = v2m(N2,y,0);
    cores[2] = v2m(N3,z,0);
     
    struct candecomp cd;
    cd.dim = 3;
    cd.rank = 1;
    cd.cores = cores;
    
    struct tt * tensortrain = cd2tt(&cd);
    ttscal(tensortrain, 2.0);

    size_t shape[3];
    shape[0] = N1;
    shape[1] = N2;
    shape[2] = N3;
    struct tt * ones = tt_ones(3, shape);
    struct tt * ttuse = ttadd(tensortrain, ones);
    
    double ** weights = malloc(3*sizeof(double *));
    weights[0] = trap_w(N1,h1);
    weights[1] = trap_w(N2,h2);
    weights[2] = trap_w(N3,h3);
    
    double int1 = tt_integrate(ttuse, weights);
    CuAssertDblEquals(tc, 1.25, int1, 1e-10);

    free(weights[0]);
    free(weights[1]);
    free(weights[2]);
    weights[0] = simpson_w(N1,h1);
    weights[1] = simpson_w(N2,h2);
    weights[2] = simpson_w(N3,h3);

    double int2 = tt_integrate(ttuse, weights);
    CuAssertDblEquals(tc, 1.25, int2, 1e-10);


    freett(ttuse);
    freett(tensortrain);


    double p1[10];
    double p2[11];
    double p3[12];
    clenshaw_curtis(N1,p1, weights[0]);
    rescale_cc(N1, p1, weights[0], 0.0,1.0);

    clenshaw_curtis(N2,p2, weights[1]);
    rescale_cc(N2, p2, weights[1], 0.0,1.0);

    clenshaw_curtis(N3,p3, weights[2]);
    rescale_cc(N3, p3, weights[2], 0.0,1.0);

    freemat(cores[2]);
    freemat(cores[1]);
    freemat(cores[0]);
    cores[0] = v2m(N1,p1,0);
    cores[1] = v2m(N2,p2,0);
    cores[2] = v2m(N3,p3,0);
     
    tensortrain = cd2tt(&cd);
    ttscal(tensortrain, 2.0);

    ttuse = ttadd(tensortrain, ones);
    double int3 = tt_integrate(ttuse, weights);
    CuAssertDblEquals(tc, 1.25, int3, 1e-10);
    
    freett(ttuse);
    freett(tensortrain);

    freett(ones);

    freemat(cores[2]);
    freemat(cores[1]);
    freemat(cores[0]);
    free(x);
    free(y);
    free(z);
    free(cores);
    free(weights[0]);
    free(weights[1]);
    free(weights[2]);
    free(weights);

}

double sinesum(double * x, void * args){
    
    size_t dim = * (size_t *) args;
    size_t ii;
    double sum = 0;
    for (ii = 0; ii < dim; ii++){
        sum += x[ii];
    }
    return sin(sum);
}

void Test_sineintegrate(CuTest * tc)
{
    printf("Testing Functions: tt_integrate with cross \n");
    size_t ii;
    //size_t jj;
    size_t dim = 2000;
    size_t N = 5;
    double ** pts = malloc(dim*sizeof(double *));
    double ** weights = malloc(dim*sizeof(double *));
    for (ii = 0; ii < dim; ii++){
        pts[ii] = calloc_double(N);
        weights[ii] = calloc_double(N);
        clenshaw_curtis(N,pts[ii], weights[ii]);
        rescale_cc(N, pts[ii], weights[ii], 0.0,1.0);
    }
    
    size_t * nvals = calloc_size_t(dim);
    for (ii = 0; ii < dim; ii++) { nvals[ii] = N; }

    struct func_to_array a;
    a.dim = dim;
    a.nvals = nvals;
    a.pts = pts;
    a.f = &sinesum;
    a.args = (void *) (&dim);

    size_t * ranks = calloc_size_t(dim+1);
    ranks[0] = 1;
    for (ii = 1; ii < dim; ii++ ){ ranks[ii] = 2; };
    ranks[dim] = 1;
    struct tt_cross_opts  * opts = 
            init_cross_opts_with_naive_set(dim, ranks, nvals);
    opts->epsilon = 1e-7;
    opts->verbose = 1;
    opts->maxiter = 1;
    
    struct tt * integrand;
    integrand = tt_cross(&wrap_func_for_cross, opts, (void *) (&a));

    double int1 = tt_integrate(integrand, weights);
    //printf("integral =%3.8E \n", int1);
    if (dim == 100){
        CuAssertDblEquals(tc, -3.926795e-3, int1, 1e-6);
    }
    else if (dim == 500){
        CuAssertDblEquals(tc, -7.287664e-10, int1, 1e-6);
    }
    else if (dim == 2000){
        CuAssertDblEquals(tc, 2.628834e-37, int1, 1e-6);
    }
    else if (dim == 4000){
        CuAssertDblEquals(tc, 9.400335e-37, int1, 1e-6);
    }

    free(ranks);
    free_cross_opts(&opts);
    freett(integrand);
    free(nvals);
    for (ii = 0; ii < dim; ii++){
        free(pts[ii]);
        free(weights[ii]);
    }
    free(pts);
    free(weights);
}

#define CHADDIM 42
double chad(double * x, void * args)
{
    struct counter * c = (struct counter *)args;
    c->N = c->N+1;
    size_t ii = 0;
    double sum = 0.0;
    for (ii = 0; ii < CHADDIM; ii++){
        sum = sum + x[ii] * x[ii];
    }
    sum = sum + x[0] * x[1];

    return sum;
}

void Test_chad(CuTest * tc)
{
    printf("Testing Functions: chad \n");

    CuAssertIntEquals(tc,1,1);

    size_t ii;
    //size_t jj;
    size_t dim = CHADDIM;
    size_t N = 10;
    double ** pts = malloc(dim*sizeof(double *));
    double ** weights = malloc(dim*sizeof(double *));
    for (ii = 0; ii < dim; ii++){
        pts[ii] = calloc_double(N);
        weights[ii] = calloc_double(N);
        clenshaw_curtis(N,pts[ii], weights[ii]);
        rescale_cc(N, pts[ii], weights[ii], 0.0,1.0);
    }
    
    size_t * nvals = calloc_size_t(dim);
    for (ii = 0; ii < dim; ii++) { nvals[ii] = N; }
    
    struct counter c;
    c.N = 0;
    struct func_to_array a;
    a.dim = dim;
    a.nvals = nvals;
    a.pts = pts;
    a.f = &chad;
    a.args = (void *) (&c);

    size_t * ranks = calloc_size_t(dim+1);
    ranks[0] = 1;
    for (ii = 1; ii < dim; ii++ ){ ranks[ii] = 8; };
    ranks[dim] = 1;
    struct tt_cross_opts  * opts = 
            init_cross_opts_with_naive_set(dim, ranks, nvals);
    opts->epsilon = 1e-7;
    opts->verbose = 1;
    opts->maxiter = 1;
    
    struct tt * integrand;
    integrand = tt_cross(&wrap_func_for_cross, opts, (void *) (&a));

    double int1 = tt_integrate(integrand, weights);
    printf("integral =%3.8E \n", int1);

    free(ranks);
    free_cross_opts(&opts);
    freett(integrand);
    free(nvals);
    for (ii = 0; ii < dim; ii++){
        free(pts[ii]);
        free(weights[ii]);
    }
    free(pts);
    free(weights);
}


CuSuite * IntegrateGetSuite(){
    //printf("----------------------------\n");

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_tt_integrate);
    SUITE_ADD_TEST(suite, Test_sineintegrate);
    SUITE_ADD_TEST(suite, Test_chad);
    return suite;
}

void RunAllTests(void) {
    
    printf("Running Test Suite: tt_integrate\n");

    CuString * output = CuStringNew();
    CuSuite * suite = CuSuiteNew();
    
    CuSuite * inte = IntegrateGetSuite();

    CuSuiteAddSuite(suite, inte);

    CuSuiteRun(suite);
    CuSuiteSummary(suite, output);
    CuSuiteDetails(suite, output);
    printf("%s \n", output->buffer);
    
    CuSuiteDelete(inte);
    CuStringDelete(output);
    free(suite);
}

int main(void) {
    RunAllTests();
}
