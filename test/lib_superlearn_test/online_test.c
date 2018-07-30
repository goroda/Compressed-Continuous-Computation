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
#include <time.h>
#include <string.h>
#include <linalg.h>

#include "array.h"
#include "CuTest.h"
#include "regress.h"
#include "online.h"
#include "objective_functions.h"

static unsigned int seed = 3;

void Test_online1(CuTest * tc)
{
    srand(seed);
    printf("\n Testing online learning on a randomly generated low rank function \n");
    printf("\t  Dimensions: 5\n");
    printf("\t  Ranks:      [1 2 3 2 3 1]\n");
    printf("\t  LPOLY order: 3\n");
    printf("\t  nunknowns:   92\n");
    printf("\t  ndata:       350\n");

    size_t dim = 5;

    size_t ranks[6] = {1,2,3,2,3,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    size_t nunknown = 0;
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        nunknown += (maxorder+1)*ranks[ii]*ranks[ii+1];
    }

    double * param = calloc_double(nunknown);
    double * grad = calloc_double(nunknown);
    for (size_t ii = 0; ii < nunknown; ii++){
        param[ii] = randu()*2.0-1.0;
        /* param[ii] = randn(); */
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,param,ranks);
    struct RegressOpts* ropts = regress_opts_alloc(dim);
    struct StochasticUpdater * su = stochastic_updater_alloc(SU_ADADELTA);
    int res = setup_least_squares_online_learning(su, 0.9, ftp, ropts);
    CuAssertIntEquals(tc, 0, res);
    CuAssertIntEquals(tc, nunknown, su->nparams);

    // create data
    size_t ndata = 1000000;
    double x[5];
    double y;

    double diffs = function_train_relnorm2diff(ftp->ft,a);
    printf("\n\t Relative Error Start: ||f - f_approx||/||f|| = %G\n",diffs);
    struct Data * data = data_alloc(1, dim);
    for (size_t jj = 0; jj < dim; jj++){
        x[jj] = randu()*(ub-lb) + lb;
    }
    for (size_t ii = 0 ; ii < ndata; ii++){
        /* printf("ii = %zu\n", ii); */
        for (size_t jj = 0; jj < dim; jj++){
            x[jj] = randu()*(ub-lb) + lb;
        }

        y = function_train_eval(a,x);

        data_set_xy(data, x, &y);

        /* printf("update\n"); */
        double eval = stochastic_update_step(su, param, grad, data);
        (void) eval;
        ft_param_update_params(ftp, param);
        /* printf("pt = "); dprint(dim, x); */
        /* printf("\t y = %3.5G\n", y); */
        /* printf("\t pred y = %3.5G\n", function_train_eval(ftp->ft, x)); */
        /* printf("\t param = "); dprint(nunknown, param); */
        /* printf("\t\t obj = %3.15G\n", eval); */
    }

    ft_param_update_params(ftp, param);

    double diff = function_train_relnorm2diff(ftp->ft,a);
    printf("\n\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    /* CuAssertDblEquals(tc,0.0,diff,1e-2); */

    data_free(data); data = NULL;
    stochastic_updater_free(su); su = NULL;
    ft_param_free(ftp);            ftp      = NULL;
    regress_opts_free(ropts);      ropts    = NULL;
    bounding_box_free(bds);        bds      = NULL;
    function_train_free(a);        a        = NULL;

    free(param); param = NULL;
    free(grad); grad = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}

static double ff_reg(double * x)
{
    return sin(x[0]*2.0) + x[1] + x[2] + x[3]*x[4] + x[3]*x[2]*x[2];
}

void Test_online2(CuTest * tc)
{
    srand(seed);
    srand(seed);
    printf("\nTesting Function: online regression with kernel basis \n");
    printf("\t  Dimensions: 5\n");


    size_t dim = 5;

    size_t ranks[6] = {1,7,5,4,3,1};
    double lb = -1.0;
    double ub = 1.0;

    // Initialize Approximation Structure
    size_t nparams = 10;
    double * centers = linspace(lb,ub,nparams);
    double scale = 1.0;
    double width = pow(nparams,-0.2)*2.0/12.0;
    width *= 10;

    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,centers,scale,width);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(KERNEL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);

    size_t nunknown = 0;
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        nunknown += nparams*ranks[ii]*ranks[ii+1];
    }

    double * param = calloc_double(nunknown);
    double * grad = calloc_double(nunknown);
    for (size_t ii = 0; ii < nunknown; ii++){
        /* param[ii] = randu()*2.0-1.0; */
        param[ii] = randn();
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,param,ranks);

    /* ft_param_create_from_lin_ls(ftp,ndata,x,y,1e-3); */
    struct RegressOpts* ropts = regress_opts_alloc(dim);
    struct StochasticUpdater * su = stochastic_updater_alloc(SU_ADADELTA);
    int res = setup_least_squares_online_learning(su, 9.99e-1, ftp, ropts);       
    CuAssertIntEquals(tc, 0, res);
    CuAssertIntEquals(tc, nunknown, su->nparams);

    // create data
    size_t ndata = 100000;
    double x[5];
    double y;

    // // add noise
    struct Data * data = data_alloc(1, dim);
    for (size_t jj = 0; jj < dim; jj++){
        x[jj] = randu()*(ub-lb) + lb;
    }
    for (size_t ii = 0 ; ii < ndata; ii++){
        /* printf("ii = %zu\n", ii); */
        if (ii % 1 == 0){
            for (size_t jj = 0; jj < dim; jj++){
                x[jj] = randu()*(ub-lb) + lb;
            }
        }
        
        y = ff_reg(x);
        data_set_xy(data, x, &y);

        double eval = stochastic_update_step(su, param, grad, data);
        (void) eval;
        ft_param_update_params(ftp, param);
        /* printf("pt = "); dprint(dim, x); */
        /* printf("\t y = %3.5G\n", y); */
        /* printf("\t pred y = %3.5G\n", function_train_eval(ftp->ft, x)); */
        /* /\* printf("\t param = "); dprint(nunknown, param); *\/ */
        /* printf("\t\t obj = %3.15G\n", eval); */
    }

    ft_param_update_params(ftp, param);

    double diff = 0.0;
    double norm = 0.0;
    for (size_t ii = 0; ii < 1000; ii++){
        double pt[5];
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }
        double v1 = ff_reg(pt);
        double v2 = function_train_eval(ftp->ft,pt);

        diff += pow(v1-v2,2);
        norm += pow(v1,2);
    }
    printf("\n\t Error = %G, norm = %G, rat = %G\n",diff,norm,diff/norm);
    CuAssertDblEquals(tc,0.0,diff/norm,1e-1);

    
    data_free(data); data = NULL;
    stochastic_updater_free(su); su = NULL;
    
    ft_param_free(ftp);            ftp      = NULL;
    regress_opts_free(ropts);      ropts    = NULL;

    free(centers); centers = NULL;
    free(param); param = NULL;
    free(grad); grad = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}

CuSuite * OnlineGetSuite()
{
    CuSuite * suite = CuSuiteNew();

    SUITE_ADD_TEST(suite, Test_online1);
    SUITE_ADD_TEST(suite, Test_online2);

    return suite;
}


