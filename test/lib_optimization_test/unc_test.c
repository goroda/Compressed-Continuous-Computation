// Copyright (c) 2014-2016, Massachusetts Institute of Technology
//
// This file is part of the Compressed Continuous Computation (C3) toolbox
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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "array.h"
#include "linalg.h"
#include "lib_optimization.h"

#include "CuTest.h"

#include "uncon_test.h"

void Test_unc1(CuTest * tc)
{

    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 1\n");

    size_t f1 = 0;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,2,dim);

    struct c3Opt * opt = c3opt_alloc(BFGS,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    /* c3opt_set_verbose(opt,1); */

    double * start = unc_test_problem_get_start(&p);

    double val;
    int res = c3opt_minimize(opt,start,&val);
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = unc_test_problem_get_sol(&p);

    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    CuAssertDblEquals(tc,err,0.0,1e-8);
    
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
}

void Test_unc2(CuTest * tc)
{

    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 2\n");


    size_t f1 = 1;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,2,dim);

    struct c3Opt * opt = c3opt_alloc(BFGS,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    /* c3opt_set_verbose(opt,1); */

    double * start = unc_test_problem_get_start(&p);

    double val;
    int res = c3opt_minimize(opt,start,&val);
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = unc_test_problem_get_sol(&p);

    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    CuAssertDblEquals(tc,err,0.0,1e-4);
    
    //minimizer
    for (size_t ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,soll[ii],start[ii],1e-2);
    }

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_unc3(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 3\n");

    size_t f1 = 2;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,2,dim);

    struct c3Opt * opt = c3opt_alloc(BFGS,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    /* c3opt_set_verbose(opt,2); */
    c3opt_set_gtol(opt,1e-20);
    c3opt_set_absxtol(opt,1e-20);
    c3opt_set_relftol(opt,1e-14);
    c3opt_set_maxiter(opt,1000);

    double * start = unc_test_problem_get_start(&p);

    double val;
    int res = c3opt_minimize(opt,start,&val);
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = unc_test_problem_get_sol(&p);

    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    CuAssertDblEquals(tc,0.0,err,1e-9);
    
    //minimizer
    /* for (size_t ii = 0; ii < dim; ii++){ */
    /*     CuAssertDblEquals(tc,soll[ii],start[ii],1e-2); */
    /* } */

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_unc4(CuTest * tc)
{

    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 4\n");


    size_t f1 = 3;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);

    struct c3Opt * opt = c3opt_alloc(BFGS,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_ls_set_maxiter(opt,400);
    /* c3opt_set_verbose(opt,4); */
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-14); */
    /* c3opt_set_maxiter(opt,1000); */

    double * start = unc_test_problem_get_start(&p);
    double gerr = c3opt_check_deriv(opt,start,1e-6);
    CuAssertDblEquals(tc,0.0,gerr,1e-5);


    double val;
    int res = c3opt_minimize(opt,start,&val);
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = unc_test_problem_get_sol(&p);

    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    CuAssertDblEquals(tc,0.0,err,1e-9);
    
    //minimizer
    for (size_t ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,soll[ii],start[ii],1e-2);
    }

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_unc5(CuTest * tc)
{

    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 5\n");

    size_t f1 = 4;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);

    struct c3Opt * opt = c3opt_alloc(BFGS,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    /* c3opt_set_verbose(opt,2); */
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-14); */
    /* c3opt_set_maxiter(opt,1000); */
    double * start = unc_test_problem_get_start(&p);

    double val;
    int res = c3opt_minimize(opt,start,&val);
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = unc_test_problem_get_sol(&p);

    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    CuAssertDblEquals(tc,0.0,err,1e-9);
    
    //minimizer
    for (size_t ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,soll[ii],start[ii],1e-2);
    }

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_unc6(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 6\n");


    size_t f1 = 5;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);

    struct c3Opt * opt = c3opt_alloc(BFGS,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);


    c3opt_set_verbose(opt,0);
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-14); */
    c3opt_set_maxiter(opt,400);
    c3opt_ls_set_maxiter(opt,1000);

    double * start = unc_test_problem_get_start(&p);
    /* printf("start\n"); */
    /* dprint(2,start); */

    double gerr = c3opt_check_deriv(opt,start,1e-8);
    CuAssertDblEquals(tc,0.0,gerr,1e-5);
    /* printf("gerr = %G\n",gerr); */
    
    double val;
    int res = c3opt_minimize(opt,start,&val);
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = unc_test_problem_get_sol(&p);

    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    CuAssertDblEquals(tc,0.0,err,1e-4);
    
    //minimizer
    for (size_t ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,soll[ii],start[ii],1e-2);
    }

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_unc7(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 7\n");


    size_t f1 = 6;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,3,dim);

    struct c3Opt * opt = c3opt_alloc(BFGS,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-14); */
    c3opt_set_maxiter(opt,400);
    c3opt_ls_set_maxiter(opt,1000);

    double * start = unc_test_problem_get_start(&p);
    double gerr = c3opt_check_deriv(opt,start,1e-8);
    CuAssertDblEquals(tc,0.0,gerr,1e-5);
    /* printf("gerr = %G\n",gerr); */
    
    double val;
    int res = c3opt_minimize(opt,start,&val);
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = unc_test_problem_get_sol(&p);

    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    CuAssertDblEquals(tc,0.0,err,1e-10);
    
    //minimizer
    for (size_t ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,soll[ii],start[ii],1e-2);
    }

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_unc8(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 8\n");


    size_t f1 = 7;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,3,dim);

    struct c3Opt * opt = c3opt_alloc(BFGS,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-15); */
    c3opt_set_maxiter(opt,400);
    c3opt_ls_set_maxiter(opt,1000);

    double * start = unc_test_problem_get_start(&p);
    double gerr = c3opt_check_deriv(opt,start,1e-8);
    CuAssertDblEquals(tc,0.0,gerr,1e-5);
    /* printf("gerr = %G\n",gerr); */
    
    double val;
    int res = c3opt_minimize(opt,start,&val);
    /* printf("res = %d\n",res); */
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = unc_test_problem_get_sol(&p);

    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    CuAssertDblEquals(tc,0.0,err,1e-4);
    
    //minimizer //not sure for this problem]
    /* for (size_t ii = 0; ii < dim; ii++){ */
    /*     CuAssertDblEquals(tc,soll[ii],start[ii],1e-2); */
    /* } */

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_unc9(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 9\n");


    size_t f1 = 8;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,3,dim);

    struct c3Opt * opt = c3opt_alloc(BFGS,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-15); */
    c3opt_set_maxiter(opt,400);
    c3opt_ls_set_maxiter(opt,1000);

    double * start = unc_test_problem_get_start(&p);
    double gerr = c3opt_check_deriv(opt,start,1e-8);
    CuAssertDblEquals(tc,0.0,gerr,1e-5);
    /* printf("gerr = %G\n",gerr); */
    
    double val;
    int res = c3opt_minimize(opt,start,&val);
    /* printf("res = %d\n",res); */
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = unc_test_problem_get_sol(&p);

    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    CuAssertDblEquals(tc,0.0,err,1e-4);
    
    //minimizer //not sure for this problem]
    /* for (size_t ii = 0; ii < dim; ii++){ */
    /*     CuAssertDblEquals(tc,soll[ii],start[ii],1e-2); */
    /* } */

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}
CuSuite * UnGetSuite(){
    //printf("----------------------------\n");

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_unc1);
    SUITE_ADD_TEST(suite, Test_unc2);
    SUITE_ADD_TEST(suite, Test_unc3);
    SUITE_ADD_TEST(suite, Test_unc4);
    SUITE_ADD_TEST(suite, Test_unc5);
    SUITE_ADD_TEST(suite, Test_unc6);
    SUITE_ADD_TEST(suite, Test_unc7);
    SUITE_ADD_TEST(suite, Test_unc8);
    SUITE_ADD_TEST(suite, Test_unc9);
    return suite;
}
