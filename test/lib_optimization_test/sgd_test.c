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

#include "CuTest.h"

#include "sgd_functions.h"

static unsigned int seed = 1;
void Test_sgd_p1(CuTest * tc)
{
    srand(seed);
    printf("\n////////////////////////////////////////////\n");
    printf("SGD Test: 1\n");

    size_t f1 = 0;
    struct SgdTestProblem p = sprobs[f1];
    size_t dim = sgd_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,2,dim);

    size_t N = 10000;
    double x[20000];

    for (size_t ii = 0; ii < N; ii++){
        x[ii*2+0]= 2.0*randu()-1;
        x[ii*2+1]= 2.0*randu()-1;
    }
    
    struct SgdData data;
    data.N = N;
    data.x = x;
    data.prob = p;
    
    struct c3Opt * opt = c3opt_alloc(SGD,dim);
    c3opt_set_sgd_nsamples(opt,N);
    c3opt_add_objective_stoch(opt,sgd_test_problem_eval,&data);
    c3opt_set_maxiter(opt,10);
    /* c3opt_set_verbose(opt,1); */

    double * start_ref = sgd_test_problem_get_start(&p);

    double val;
    double * start = calloc_double(dim);
    memmove(start,start_ref, dim * sizeof(double));
    int res = c3opt_minimize(opt,start,&val);
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = sgd_test_problem_get_sol(&p);

    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    CuAssertDblEquals(tc,0.0,err,1e-8);
    
    //minimizer
    for (size_t ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,soll[ii],start[ii],1e-4);
    }

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
    free(start); start = NULL;
}

void Test_sgd_p2(CuTest * tc)
{
    srand(seed);
    printf("\n////////////////////////////////////////////\n");
    printf("SGD Test: 2\n");

    size_t f1 = 1;
    struct SgdTestProblem p = sprobs[f1];
    size_t dim = sgd_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,5,dim);

    size_t N = 10000;
    double x[50000];

    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = 2.0*randu()-1;
        }
    }
    
    struct SgdData data;
    data.N = N;
    data.x = x;
    data.prob = p;
    
    struct c3Opt * opt = c3opt_alloc(SGD,dim);
    c3opt_set_sgd_learn_rate(opt,1e-1);
    c3opt_set_sgd_nsamples(opt,N);
    c3opt_add_objective_stoch(opt,sgd_test_problem_eval,&data);
    c3opt_set_maxiter(opt,10);
    /* c3opt_set_verbose(opt,1); */

    double * start_ref = sgd_test_problem_get_start(&p);

    double val;
    double * start = calloc_double(dim);
    memmove(start,start_ref, dim * sizeof(double));
    int res = c3opt_minimize(opt,start,&val);
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = sgd_test_problem_get_sol(&p);

    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    CuAssertDblEquals(tc,0.0,err,1e-8);
    
    //minimizer
    for (size_t ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,soll[ii],start[ii],1e-4);
    }

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
    free(start); start = NULL;
}

void Test_sgd_p3(CuTest * tc)
{
    srand(seed);
    printf("\n////////////////////////////////////////////\n");
    printf("SGD Test: 3\n");

    size_t f1 = 2;
    struct SgdTestProblem p = sprobs[f1];
    size_t dim = sgd_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,2,dim);

    size_t N = 10;
    double x[50000];

    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = 2.0*randu()-1;
        }
    }
    
    struct SgdData data;
    data.N = N;
    data.x = x;
    data.prob = p;
    
    struct c3Opt * opt = c3opt_alloc(SGD,dim);
    c3opt_set_sgd_learn_rate(opt,1e-3);
    c3opt_set_gtol(opt,1e-11);
    c3opt_set_sgd_nsamples(opt,N);
    c3opt_set_absxtol(opt,1e-30);
    c3opt_add_objective_stoch(opt,sgd_test_problem_eval,&data);
    c3opt_set_maxiter(opt,10000);
    /* c3opt_set_verbose(opt,1); */

    double * start_ref = sgd_test_problem_get_start(&p);

    double val;
    double * start = calloc_double(dim);
    memmove(start,start_ref, dim * sizeof(double));
    int res = c3opt_minimize(opt,start,&val);
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = sgd_test_problem_get_sol(&p);

    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    CuAssertDblEquals(tc,0.0,err,1e-6);
    
    //minimizer
    for (size_t ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,soll[ii],start[ii],1e-4);
    }

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
    free(start); start = NULL;
}


void Test_sgd_p4(CuTest * tc)
{
    srand(seed);
    printf("\n////////////////////////////////////////////\n");
    printf("SGD Test: 4\n");

    size_t f1 = 3;
    struct SgdTestProblem p = sprobs[f1];
    size_t dim = sgd_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,5,dim);

    size_t N = 500;
    double x[50000];

    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = 2.0*randu()-1;
        }
    }
    
    struct SgdData data;
    data.N = N;
    data.x = x;
    data.prob = p;
    
    struct c3Opt * opt = c3opt_alloc(SGD,dim);
    c3opt_set_sgd_learn_rate(opt,1e-3);
    c3opt_set_sgd_nsamples(opt,N);
    c3opt_add_objective_stoch(opt,sgd_test_problem_eval,&data);
    c3opt_set_maxiter(opt,200);
    /* c3opt_set_verbose(opt,1); */

    double * start_ref = sgd_test_problem_get_start(&p);

    double val;
    double * start = calloc_double(dim);
    memmove(start,start_ref, dim * sizeof(double));
    int res = c3opt_minimize(opt,start,&val);
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = sgd_test_problem_get_sol(&p);

    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    CuAssertDblEquals(tc,0.0,err,1e-6);
    
    //minimizer
    for (size_t ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,soll[ii],start[ii],1e-4);
    }

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
    free(start); start = NULL;
}


void Test_sgd_p5(CuTest * tc)
{
    srand(seed);
    printf("\n////////////////////////////////////////////\n");
    printf("SGD Test: 5 (logistic regression)\n");

    size_t f1 = 4;
    struct SgdTestProblem p = sprobs[f1];
    size_t dim = sgd_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,3,dim);

    size_t N = 100;
    double x[250000];

    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < dim-1; jj++){
            x[ii*2+jj] = 2.0*randu()-1;
        }
    }
    
    struct SgdData data;
    data.N = N;
    data.x = x;
    data.prob = p;
    
    struct c3Opt * opt = c3opt_alloc(SGD,dim);
    c3opt_set_sgd_learn_rate(opt,1e-3);
    c3opt_set_sgd_nsamples(opt,N);
    c3opt_add_objective_stoch(opt,sgd_test_problem_eval,&data);
    c3opt_set_gtol(opt,2e-3);
    c3opt_set_maxiter(opt,10000);
    c3opt_set_verbose(opt,1);

    double * start_ref = sgd_test_problem_get_start(&p);

    double val;
    double * start = calloc_double(dim);
    memmove(start,start_ref, dim * sizeof(double));
    int res = c3opt_minimize(opt,start,&val);
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = sgd_test_problem_get_sol(&p);

    //minimum -> dont check because maximizing log likelihood)
    
    //minimizer
    dprint(dim,soll);
    dprint(dim,start);
    /* for (size_t ii = 0; ii < dim; ii++){ */
    /*     CuAssertDblEquals(tc,soll[ii],start[ii],1e-4); */
    /* } */

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
    free(start); start = NULL;
}




CuSuite * SGDGetSuite(){
    //printf("----------------------------\n");

    CuSuite * suite = CuSuiteNew();

    SUITE_ADD_TEST(suite, Test_sgd_p1);
    SUITE_ADD_TEST(suite, Test_sgd_p2);
    SUITE_ADD_TEST(suite, Test_sgd_p3);
    SUITE_ADD_TEST(suite, Test_sgd_p4);



    // dont know what this is upposed to do
    /* SUITE_ADD_TEST(suite, Test_sgd_p5); */

    return suite;
}
