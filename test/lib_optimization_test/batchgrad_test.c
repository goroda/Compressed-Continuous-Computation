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

#include "unconstrained_functions.h"

void Test_bgrad1(CuTest * tc)
{

    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 1\n");

    size_t f1 = 0;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,2,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_set_maxiter(opt,200000);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    /* c3opt_ls_set_alpha(opt,0.001); */
    /* c3opt_ls_set_beta(opt,0.9); */
    /* c3opt_set_gtol(opt,1e-18); */
    /* c3opt_set_relftol(opt,1e-18); */
    c3opt_set_verbose(opt,0);

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

void Test_bgrad2(CuTest * tc)
{

    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 2\n");

    size_t f1 = 1;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,2,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_set_maxiter(opt,100000);
    c3opt_set_gtol(opt,1e-18);
    c3opt_set_absxtol(opt,1e-18);
    c3opt_set_relftol(opt,1e-18);
    c3opt_set_verbose(opt,1);
    
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    

    double * start = unc_test_problem_get_start(&p);

    double val;
    int res = c3opt_minimize(opt,start,&val);
    CuAssertIntEquals(tc,1,res>=0);

    double * soll = unc_test_problem_get_sol(&p);

    
    // minimizer
    for (size_t ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,soll[ii],start[ii],1e-2);
    }

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    /* CuAssertDblEquals(tc,err,0.0,1e-4); */

    
    c3opt_free(opt); opt = NULL;
}

void Test_bgrad3(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 3\n");

    size_t f1 = 2;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,2,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,2);
    c3opt_set_gtol(opt,1e-20);
    c3opt_set_absxtol(opt,1e-20);
    c3opt_set_relftol(opt,1e-14);    
    c3opt_set_maxiter(opt,10000);

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
    /* CuAssertDblEquals(tc,0.0,err,1e-9); */
    
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

void Test_bgrad4(CuTest * tc)
{

    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 4\n");


    size_t f1 = 3;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_ls_set_maxiter(opt,100);
    c3opt_ls_set_alpha(opt,1e-3);
    c3opt_ls_set_beta(opt,0.9);
    c3opt_set_verbose(opt,3);
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-14); */
    c3opt_set_maxiter(opt,10);

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
    /* CuAssertDblEquals(tc,0.0,err,1e-9); */
    
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

void Test_bgrad5(CuTest * tc)
{

    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 5\n");

    size_t f1 = 4;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    /* c3opt_set_verbose(opt,2); */
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-14); */
    /* c3opt_set_maxiter(opt,20000); */
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

void Test_bgrad6(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 6\n");


    size_t f1 = 5;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);


    c3opt_set_verbose(opt,0);
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-14); */
    c3opt_set_maxiter(opt,400);
    c3opt_ls_set_maxiter(opt,20000);

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

void Test_bgrad7(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 7\n");


    size_t f1 = 6;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,3,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-14); */
    c3opt_set_maxiter(opt,10000);
    c3opt_ls_set_maxiter(opt,200000);

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

void Test_bgrad8(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 8\n");


    size_t f1 = 7;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,3,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-15); */
    c3opt_set_maxiter(opt,20000);
    c3opt_ls_set_maxiter(opt,20000);

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

void Test_bgrad9(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 9\n");


    size_t f1 = 8;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,3,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-15); */
    c3opt_set_maxiter(opt,400);
    c3opt_ls_set_maxiter(opt,20000);

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

void Test_bgrad10(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 10\n");

    size_t f1 = 9;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,3,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    c3opt_set_maxiter(opt,200000);
    c3opt_ls_set_maxiter(opt,10000);

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

void Test_bgrad11(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 11\n");

    size_t f1 = 10;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,3,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    c3opt_set_gtol(opt,1e-20);
    c3opt_set_absxtol(opt,1e-20);
    c3opt_set_relftol(opt,1e-15);
    c3opt_set_maxiter(opt,20000);
    c3opt_ls_set_maxiter(opt,20000);

    double * start = unc_test_problem_get_start(&p);
    double * deriv_diff = calloc_double(dim);
    double gerr = c3opt_check_deriv_each(opt,start,1e-8,deriv_diff);
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-5);
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-5);
    free(deriv_diff); deriv_diff = NULL;
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
    
    //minimizer is different but same minimum
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

void Test_bgrad12(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 12\n");

    size_t f1 = 11;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,3,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-15); */
    c3opt_set_maxiter(opt,200000);
    c3opt_ls_set_maxiter(opt,2000);

    double * start = unc_test_problem_get_start(&p);
    double * deriv_diff = calloc_double(dim);
    double gerr = c3opt_check_deriv_each(opt,start,1e-8,deriv_diff);
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3);
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
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
    
    //minimizer is different but same minimum
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

void Test_bgrad13(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 13\n");

    size_t f1 = 12;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,4,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-15) */;
    c3opt_set_maxiter(opt,1000000);
    
    /* c3opt_ls_set_beta(opt,0.99);  */
    c3opt_ls_set_maxiter(opt,20000);

    double * start = unc_test_problem_get_start(&p);
    double * deriv_diff = calloc_double(dim);
    double gerr = c3opt_check_deriv_each(opt,start,1e-8,deriv_diff);
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3);
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
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
    
    //minimizer is different but same minimum
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

void Test_bgrad14(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 14\n");

    size_t f1 = 13;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,4,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    c3opt_set_gtol(opt,1e-30);
    c3opt_set_absxtol(opt,1e-20);
    c3opt_set_relftol(opt,1e-20);
    c3opt_set_maxiter(opt,20000);

    /* c3opt_ls_set_beta(opt,0.99); // NEEDED HERE! */
    c3opt_ls_set_alpha(opt,1e-1);
    c3opt_ls_set_maxiter(opt,20000);

    double * start = unc_test_problem_get_start(&p);
    double * deriv_diff = calloc_double(dim);
    double gerr = c3opt_check_deriv_each(opt,start,1e-8,deriv_diff);
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3);
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
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
    
    //minimizer 
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu, solution=%G\n",ii,start[ii]); */
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

void Test_bgrad15(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 15\n");

    size_t f1 = 14;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,4,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    /* c3opt_set_gtol(opt,1e-30); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-20); */
    c3opt_set_maxiter(opt,20000);

    /* c3opt_ls_set_alpha(opt,0.3); */
    c3opt_ls_set_maxiter(opt,20000);

    double * start = unc_test_problem_get_start(&p);
    double * deriv_diff = calloc_double(dim);
    double gerr = c3opt_check_deriv_each(opt,start,1e-8,deriv_diff);
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3);
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
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
    

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    /* printf("\n\n\t Note: needed to set line search beta to 0.99\n"); */
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_bgrad16(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 16\n");

    size_t f1 = 15;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,4,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    c3opt_set_maxiter(opt,20000);

    /* c3opt_ls_set_alpha(opt,0.3); */
    c3opt_ls_set_maxiter(opt,20000);

    double * start = unc_test_problem_get_start(&p);    
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
    

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    /* printf("\n\n\t Note: needed to set line search beta to 0.99\n"); */
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_bgrad17(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 17\n");

    size_t f1 = 16;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,5,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_gtol(opt,1e-20);
    c3opt_set_absxtol(opt,1e-20);
    c3opt_set_relftol(opt,1e-20);
    c3opt_set_verbose(opt,0);
    c3opt_set_maxiter(opt,200000);

    /* c3opt_ls_set_beta(opt,0.99); */
    c3opt_ls_set_maxiter(opt,2000);

    double * start = unc_test_problem_get_start(&p);
    double * deriv_diff = calloc_double(dim);
    double gerr = c3opt_check_deriv_each(opt,start,1e-8,deriv_diff);
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3);
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    
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
    

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    /* printf("\n\n\t Note: needed to set line search beta to 0.99\n"); */
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_bgrad18(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 18\n");

    size_t f1 = 17;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,6,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    c3opt_set_maxiter(opt,2000000);
    c3opt_set_gtol(opt,1e-20);
    /* c3opt_ls_set_beta(opt,0.99); */
    c3opt_ls_set_alpha(opt,0.3);
    c3opt_ls_set_maxiter(opt,2000);

    double * start = unc_test_problem_get_start(&p);
    double * deriv_diff = calloc_double(dim);
    double gerr = c3opt_check_deriv_each(opt,start,1e-8,deriv_diff);
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3);
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    
    double val;
    int res = c3opt_minimize(opt,start,&val);
    printf("res = %d\n",res);
    /* CuAssertIntEquals(tc,1,res>=0); */
    
    double * soll = unc_test_problem_get_sol(&p);
    
    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    CuAssertDblEquals(tc,0.0,err,1e-4);
    

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("\n\n\t Note: needed to set line search beta to 0.99\n");
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_bgrad19(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 19\n");

    size_t f1 = 18;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,11,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    c3opt_set_maxiter(opt,100000);

    /* c3opt_ls_set_beta(opt,0.99); */
    c3opt_ls_set_maxiter(opt,20000);

    double * start = unc_test_problem_get_start(&p);
    
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
    

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_bgrad20(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 20\n");

    size_t f1 = 19;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,9,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    c3opt_set_maxiter(opt,100000);

    /* c3opt_set_gtol(opt,1e-30); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,1e-20); */
    /* c3opt_ls_set_beta(opt,0.7); */
    c3opt_ls_set_maxiter(opt,20000);

    double * start = unc_test_problem_get_start(&p);
    double * deriv_diff = calloc_double(dim);
    double gerr = c3opt_check_deriv_each(opt,start,1e-8,deriv_diff);
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3);
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
    
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
    

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_bgrad21(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 21\n");

    size_t f1 = 20;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,30,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    c3opt_set_maxiter(opt,20000);

    c3opt_ls_set_maxiter(opt,20000);

    double * start = unc_test_problem_get_start(&p);
    double * deriv_diff = calloc_double(dim);
    double gerr = c3opt_check_deriv_each(opt,start,1e-8,deriv_diff);
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3);
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
    
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
    

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_bgrad22(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 22\n");

    size_t f1 = 21;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,400,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    c3opt_set_maxiter(opt,1000000);

    c3opt_ls_set_maxiter(opt,20000);

    double * start = unc_test_problem_get_start(&p);
    double * deriv_diff = calloc_double(dim);
    double gerr = c3opt_check_deriv_each(opt,start,1e-8,deriv_diff);
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3);
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
    
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
    

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_bgrad23(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 23\n");

    size_t f1 = 22;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,4,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    c3opt_set_maxiter(opt,100000);
    c3opt_ls_set_maxiter(opt,20000);

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
    CuAssertDblEquals(tc,0.0,err,1e-4);
    

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

void Test_bgrad24(CuTest * tc)
{
    printf("////////////////////////////////////////////\n");
    printf("\t Unconstrained Test: 24\n");

    size_t f1 = 23;
    struct UncTestProblem p = tprobs[f1];
    size_t dim = unc_test_problem_get_dim(&p);
    CuAssertIntEquals(tc,4,dim);

    struct c3Opt * opt = c3opt_alloc(BATCHGRAD,dim);
    c3opt_add_objective(opt,unc_test_problem_eval,&p);
    c3opt_set_verbose(opt,0);
    /* c3opt_set_gtol(opt,1e-20); */
    /* c3opt_set_absxtol(opt,1e-20); */
    /* c3opt_set_relftol(opt,0.1e-20); */
    c3opt_set_maxiter(opt,100000);
    c3opt_ls_set_beta(opt,0.9900);
    c3opt_ls_set_alpha(opt,0.1);
    c3opt_ls_set_maxiter(opt,1000);

    double * start = unc_test_problem_get_start(&p);
    double * deriv_diff = calloc_double(dim);
    double gerr = c3opt_check_deriv_each(opt,start,1e-8,deriv_diff);
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3);
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    double val;
    int res = c3opt_minimize(opt,start,&val);
    printf("res = %d\n",res);
    CuAssertIntEquals(tc,1,res>=0);
    
    double * soll = unc_test_problem_get_sol(&p);
    
    //minimum
    double err = fabs(soll[dim]-val);
    if (fabs(soll[dim]) > 1){
        err /= fabs(soll[dim]);
    }
    /* CuAssertDblEquals(tc,0.0,err,1e-4); */

    printf("\t\t *True* Minimum:               : %3.6E\n", soll[dim]);
    printf("\t\t Minimum Found:                : %3.6E\n\n",val);

    printf("\t\t Number of Function Evaluations: %zu\n",c3opt_get_nevals(opt));
    printf("\t\t Number of Gradient Evaluations: %zu\n",c3opt_get_ngvals(opt));
    printf("\t\t Number of iterations:           %zu\n",c3opt_get_niters(opt));
    printf("\n\n *_*_*_*_*_*_*_*_* DOES NOT WORK *_*_*_*_*_*_*_*_*_\n");
    printf("////////////////////////////////////////////\n");

    c3opt_free(opt); opt = NULL;
}

CuSuite * BGradGetSuite(){
    //printf("----------------------------\n");

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_bgrad1);
    /* SUITE_ADD_TEST(suite, Test_bgrad2); */
    /* SUITE_ADD_TEST(suite, Test_bgrad3); */
    SUITE_ADD_TEST(suite, Test_bgrad4);
    /* SUITE_ADD_TEST(suite, Test_bgrad5); */
    /* /\* SUITE_ADD_TEST(suite, Test_bgrad6); *\/ */
    /* SUITE_ADD_TEST(suite, Test_bgrad7); */
    /* SUITE_ADD_TEST(suite, Test_bgrad8); */
    /* SUITE_ADD_TEST(suite, Test_bgrad9); */
    /* /\* SUITE_ADD_TEST(suite, Test_bgrad10); *\/ */
    /* /\* SUITE_ADD_TEST(suite, Test_bgrad11); *\/ */
    /* SUITE_ADD_TEST(suite, Test_bgrad12); */
    /* SUITE_ADD_TEST(suite, Test_bgrad13); */
    /* SUITE_ADD_TEST(suite, Test_bgrad14); */
    /* SUITE_ADD_TEST(suite, Test_bgrad15); */
    /* SUITE_ADD_TEST(suite, Test_bgrad16); */
    /* /\* SUITE_ADD_TEST(suite, Test_bgrad17); *\/ */
    /* /\* SUITE_ADD_TEST(suite, Test_bgrad18); *\/ */
    /* SUITE_ADD_TEST(suite, Test_bgrad19); */
    /* /\* SUITE_ADD_TEST(suite, Test_bgrad20); *\/ */
    /* SUITE_ADD_TEST(suite, Test_bgrad21); */
    /* SUITE_ADD_TEST(suite, Test_bgrad22); */
    /* SUITE_ADD_TEST(suite, Test_bgrad23); */
    /* SUITE_ADD_TEST(suite, Test_bgrad24); */
    return suite;
}

