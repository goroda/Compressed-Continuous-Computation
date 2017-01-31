// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016, Sandia Corporation

// This file is part of the Compressed Continuous Computation (C3) Library
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


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "array.h"

#include "CuTest.h"
#include "testfunctions.h"

#include "lib_funcs.h"
#include "lib_linalg.h"

#include "lib_clinalg.h"

#include "lib_optimization.h"

static int seed = 3;


void Test_LS_ALS(CuTest * tc)
{
    srand(seed);
    printf("Testing Function: regress_als_sweep_lr (5 dimensional, max rank = 8, max order = 3)\n");

    size_t dim = 5;
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[6] = {1,2,8,3,2,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 1000;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = function_train_eval(a,x+ii*dim);
        /* y[ii] += randn(); */
    }


    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(dim * 25 * (maxorder+1));
    size_t onparam=0;

    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        
        /* double * a_param = calloc_double(25 * (maxorder+1)); */
        /* size_t nn = function_train_core_get_params(a,ii,a_param); */
        /* dprint(nn,a_param); */
        /* for (size_t ll = 0; ll  < nn; ll++){ */
        /*     param_space[onparam] = a_param[ll]; */
        /*     onparam++; */
        /* } */
        /* free(a_param); a_param = NULL; */
        for (size_t jj = 0; jj < ranks[ii]; jj++){
            for (size_t kk = 0; kk < ranks[ii+1]; kk++){
                for (size_t ll = 0; ll < maxorder+1; ll++){
                    param_space[onparam] += randu()*2.0-1.0;
                    onparam++;
                }
            }
        }
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    struct RegressOpts* ropts = regress_opts_create(ALS,FTLS,ndata,dim,x,y);
    regress_opts_set_verbose(ropts,1);

    struct FunctionTrain * ft_final = c3_regression_run(ftp,ropts);
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-3);
    
    ft_param_free(ftp);            ftp         = NULL;
    regress_opts_free(ropts);      ropts       = NULL;
    free(param_space);             param_space = NULL;
    bounding_box_free(bds);        bds         = NULL;
    function_train_free(a);        a           = NULL;
    function_train_free(ft_final); ft_final    = NULL;
    
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
}

void Test_LS_ALS2(CuTest * tc)
{
    srand(seed);
    printf("Testing Function: regress_als_sweep_lr (5 dimensional, max rank = 5, max order = 8) \n");

    size_t dim = 5;
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[6] = {1,2,3,2,5,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 8;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 4000;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = function_train_eval(a,x+ii*dim);
        /* y[ii] += randn(); */
    }

        // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(dim * 25 * (maxorder+1));
    size_t onparam=0;

    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        
        for (size_t jj = 0; jj < ranks[ii]; jj++){
            for (size_t kk = 0; kk < ranks[ii+1]; kk++){
                for (size_t ll = 0; ll < maxorder+1; ll++){
                    param_space[onparam] += randu()*2.0-1.0;
                    onparam++;
                }
            }
        }
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    struct RegressOpts* ropts = regress_opts_create(ALS,FTLS,ndata,dim,x,y);
    regress_opts_set_verbose(ropts,1);
    regress_opts_set_als_maxsweep(ropts,10);
    regress_opts_set_convtol(ropts,1e-10);
    struct FunctionTrain * ft_final = c3_regression_run(ftp,ropts);
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff);


    
    struct RegressOpts * aio_opts = regress_opts_create(AIO,FTLS,ndata,dim,x,y);
    regress_opts_set_verbose(ropts,0);
    regress_opts_set_convtol(ropts,1e-10);
    struct FunctionTrain * ft_final2 = c3_regression_run(ftp,aio_opts);
    double diff2 = function_train_relnorm2diff(ft_final2,a);
    printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff2);


    struct RegressOpts * als_opts = regress_opts_create(ALS,FTLS,ndata,dim,x,y);
    regress_opts_set_verbose(ropts,0);
    regress_opts_set_convtol(ropts,1e-10);
    struct FunctionTrain * ft_final3 = c3_regression_run(ftp,als_opts);
    double diff3 = function_train_relnorm2diff(ft_final3,a);
    printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff3);
    CuAssertDblEquals(tc,0.0,diff3,1e-3);
    
    ft_param_free(ftp);             ftp         = NULL;
    regress_opts_free(ropts);       ropts       = NULL;
    regress_opts_free(aio_opts);    aio_opts    = NULL;
    regress_opts_free(als_opts);    als_opts    = NULL;
    free(param_space);              param_space = NULL;
    bounding_box_free(bds);         bds         = NULL;
    function_train_free(a);         a           = NULL;
    function_train_free(ft_final);  ft_final    = NULL;
    function_train_free(ft_final2); ft_final2   = NULL;
    function_train_free(ft_final3); ft_final3   = NULL;

    
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
}

void Test_LS_ALS_SPARSE2(CuTest * tc)
{
    srand(seed);
    printf("Testing Function: regress_als with sparse reg (5 dimensional, max rank = 5, max order = 8) \n");

    size_t dim = 5;
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[6] = {1,2,3,2,5,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 8;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 4000;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = function_train_eval(a,x+ii*dim);
        /* y[ii] += randn(); */
    }

    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(dim * 25 * (maxorder+1));
    size_t onparam=0;

    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        
        for (size_t jj = 0; jj < ranks[ii]; jj++){
            for (size_t kk = 0; kk < ranks[ii+1]; kk++){
                for (size_t ll = 0; ll < maxorder+1; ll++){
                    param_space[onparam] += randu()*2.0-1.0;
                    onparam++;
                }
            }
        }
    }

    double regweight = 1e-2;
    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    struct RegressOpts* ropts = regress_opts_create(ALS,FTLS_SPARSEL2,ndata,dim,x,y);
    regress_opts_set_verbose(ropts,1);
    regress_opts_set_als_maxsweep(ropts,20);
    regress_opts_set_convtol(ropts,1e-10);
    regress_opts_set_regweight(ropts,regweight);
    struct FunctionTrain * ft_final = c3_regression_run(ftp,ropts);
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff);


    
    struct RegressOpts * aio_opts = regress_opts_create(AIO,FTLS_SPARSEL2,ndata,dim,x,y);
    regress_opts_set_verbose(ropts,0);
    regress_opts_set_convtol(ropts,1e-10);
    regress_opts_set_regweight(ropts,regweight);
    struct FunctionTrain * ft_final2 = c3_regression_run(ftp,aio_opts);
    double diff2 = function_train_relnorm2diff(ft_final2,a);
    printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff2);


    struct RegressOpts * als_opts = regress_opts_create(ALS,FTLS_SPARSEL2,ndata,dim,x,y);
    regress_opts_set_verbose(ropts,0);
    regress_opts_set_convtol(ropts,1e-10);
    regress_opts_set_regweight(ropts,regweight);
    struct FunctionTrain * ft_final3 = c3_regression_run(ftp,als_opts);
    double diff3 = function_train_relnorm2diff(ft_final3,a);
    printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff3);
    CuAssertDblEquals(tc,0.0,diff3,1e-3);
    
    ft_param_free(ftp);             ftp         = NULL;
    regress_opts_free(ropts);       ropts       = NULL;
    regress_opts_free(aio_opts);    aio_opts    = NULL;
    regress_opts_free(als_opts);    als_opts    = NULL;
    free(param_space);              param_space = NULL;
    bounding_box_free(bds);         bds         = NULL;
    function_train_free(a);         a           = NULL;
    function_train_free(ft_final);  ft_final    = NULL;
    function_train_free(ft_final2); ft_final2   = NULL;
    function_train_free(ft_final3); ft_final3   = NULL;

    
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
}

void Test_function_train_core_param_grad_eval1(CuTest * tc)
{
    printf("Testing Function: function_train_param_grad_eval1 \n");
    srand(seed);
    size_t dim = 4;    
    
    size_t ranks[5] = {1,2,3,8,1};
    /* size_t ranks[5] = {1,2,2,2,1}; */
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10; // 10
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 10; // 10
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    struct RunningCoreTotal * runeval_lr = ftutil_running_tot_space(a);
    struct RunningCoreTotal * runeval_rl = ftutil_running_tot_space(a);
    struct RunningCoreTotal * rungrad = ftutil_running_tot_space(a);

    /* size_t core = 2; */
    for (size_t core = 0; core < dim; core++){
        printf("core = %zu\n",core);
        running_core_total_restart(runeval_lr);
        running_core_total_restart(runeval_rl);
        running_core_total_restart(rungrad);

        /* function_train_core_pre_post_run(a,core,ndata,x,runeval_lr,runeval_rl); */
        
        size_t max_param_within_func;
        size_t nparam;
        nparam = function_train_core_get_nparams(a,core,&max_param_within_func);
        CuAssertIntEquals(tc,(maxorder+1)*ranks[core]*ranks[core+1],nparam);
        CuAssertIntEquals(tc,(maxorder+1),max_param_within_func);

        size_t core_space_size = nparam * ranks[core] * ranks[core+1];
        double * core_grad_space = calloc_double(core_space_size * ndata);
        double * max_func_param_space = calloc_double(max_param_within_func);

        double * vals = calloc_double(ndata);
        double * grad = calloc_double(ndata*nparam);

        double * guess = calloc_double(nparam);
        for (size_t jj = 0; jj < nparam; jj++){
            guess[jj] = randn();
        }
        function_train_core_update_params(a,core,nparam,guess);

        for (size_t ii = 0 ; ii < ndata; ii++){
            for (size_t jj = 0; jj < dim; jj++){
                x[ii*dim+jj] = randu()*(ub-lb) + lb;
            }
            y[ii] = function_train_eval(a,x+ii*dim);
        }

        /* printf("x = \n"); */
        /* dprint2d_col(dim,ndata,x); */
        /* printf("y = "); */
        /* dprint(ndata,y); */

    
        printf("\t Testing Evaluation\n");
        function_train_core_pre_post_run(a,core,ndata,x,runeval_lr,runeval_rl);
        function_train_core_param_grad_eval(a,core,ndata,x,runeval_lr,runeval_rl,NULL,nparam,vals,NULL,
                                             NULL,0,NULL);

        for (size_t ii = 0; ii < ndata; ii++){
            CuAssertDblEquals(tc,y[ii],vals[ii],1e-13);
        }
    
        printf("\t Testing Gradient\n");
        function_train_core_param_grad_eval(a,core,ndata,x,runeval_lr,runeval_rl,rungrad,nparam,vals,grad,
                                             core_grad_space,core_space_size,max_func_param_space);
        for (size_t ii = 0; ii < ndata; ii++){
            CuAssertDblEquals(tc,y[ii],vals[ii],1e-13);
        }

        running_core_total_restart(rungrad);
        function_train_core_param_grad_eval(a,core,ndata,x,runeval_lr,runeval_rl,rungrad,nparam,vals,grad,
                                             core_grad_space,core_space_size,max_func_param_space);

        running_core_total_restart(rungrad);
        function_train_core_param_grad_eval(a,core,ndata,x,runeval_lr,runeval_rl,rungrad,nparam,vals,grad,
                                             core_grad_space,core_space_size,max_func_param_space);

        running_core_total_restart(rungrad);
        function_train_core_param_grad_eval(a,core,ndata,x,runeval_lr,runeval_rl,rungrad,nparam,vals,grad,
                                             core_grad_space,core_space_size,max_func_param_space);

        /* printf("grad = "); dprint(totparam,grad); */
        for (size_t zz = 0; zz < ndata; zz++){
            double h = 1e-8;
            for (size_t jj = 0; jj < nparam; jj++){
                guess[jj] += h;
                function_train_core_update_params(a,core,nparam,guess);
            
                double val2 = function_train_eval(a,x+zz*dim);
                double fd = (val2-y[zz])/h;
                CuAssertDblEquals(tc,fd,grad[jj + zz * nparam],1e-5);
                guess[jj] -= h;
                function_train_core_update_params(a,core,nparam,guess);
            }
        }

        free(guess); guess = NULL;
        free(vals); vals = NULL;
        free(grad); grad = NULL;
    
        free(core_grad_space);      core_grad_space = NULL;
        free(max_func_param_space); max_func_param_space = NULL;
    }        
    running_core_total_free(runeval_lr); runeval_lr = NULL;
    running_core_total_free(runeval_rl); runeval_rl = NULL;
    running_core_total_free(rungrad);    rungrad = NULL;
    
    bounding_box_free(bds); bds = NULL;
    function_train_free(a); a   = NULL;
    free(x);                x   = NULL;
    free(y);                y   = NULL;
}


void Test_function_train_param_grad_eval(CuTest * tc)
{
    printf("Testing Function: function_train_param_grad_eval \n");
    srand(seed);
    size_t dim = 4;    
    
    size_t ranks[5] = {1,2,3,8,1};
    /* size_t ranks[5] = {1,2,2,2,1}; */
    double lb = -0.5;
    double ub = 1.0;
    size_t maxorder = 10; // 10
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 10; // 10
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    struct RunningCoreTotal * runeval_lr = ftutil_running_tot_space(a);
    struct RunningCoreTotal * runeval_rl = ftutil_running_tot_space(a);
    struct RunningCoreTotal ** rungrad = ftutil_running_tot_space_eachdim(a);

    size_t nparam[4];
    size_t max_param_within_func=0, temp_nparam;
    size_t totparam = 0;
    for (size_t ii = 0; ii < dim; ii++){
        nparam[ii] = function_train_core_get_nparams(a,ii,&temp_nparam);
        if (temp_nparam > max_param_within_func){
            max_param_within_func = temp_nparam;
        }
        CuAssertIntEquals(tc,(maxorder+1)*ranks[ii]*ranks[ii+1],nparam[ii]);
        totparam += nparam[ii];
    }
    CuAssertIntEquals(tc,(maxorder+1),max_param_within_func);

    size_t core_space_size = 0;
    for (size_t ii = 0; ii < dim; ii++){
        if (nparam[ii] * ranks[ii] * ranks[ii+1] > core_space_size){
            core_space_size = nparam[ii] * ranks[ii] * ranks[ii+1];
        }
    }
    double * core_grad_space = calloc_double(core_space_size * ndata);
    double * max_func_param_space = calloc_double(max_param_within_func);

    double * vals = calloc_double(ndata);
    double * grad = calloc_double(ndata*totparam);

    double * guess = calloc_double(totparam);
    size_t runtot = 0;
    size_t running = 0;
    for (size_t zz = 0; zz < dim; zz++){
        /* printf("nparam[%zu] = %zu\n",zz,nparam[zz]); */
        for (size_t jj = 0; jj < nparam[zz]; jj++){
            guess[running+jj] = randn();
            /* printf("guess[%zu] = %G\n",runtot,guess[runtot]); */
            runtot++;
        }
        function_train_core_update_params(a,zz,nparam[zz],guess+running);
        running+=nparam[zz];
    }

    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        y[ii] = function_train_eval(a,x+ii*dim);
    }

    /* printf("x = \n"); */
    /* dprint2d_col(dim,ndata,x); */
    /* printf("y = "); */
    /* dprint(ndata,y); */

    
    printf("\t Testing Evaluation\n");
    function_train_param_grad_eval(a,ndata,x,runeval_lr,NULL,NULL,nparam,vals,NULL,
                                   core_grad_space,core_space_size,max_func_param_space);

    for (size_t ii = 0; ii < ndata; ii++){
        CuAssertDblEquals(tc,y[ii],vals[ii],1e-15);
    }

    
    printf("\t Testing Gradient\n");
    //purposely doing so many evaluations to test restart!!
    running_core_total_restart(runeval_lr);
    running_core_total_restart(runeval_rl);
    function_train_param_grad_eval(a,ndata,x,runeval_lr,runeval_rl,rungrad,nparam,vals,grad,
                                   core_grad_space,core_space_size,max_func_param_space);
    for (size_t ii = 0; ii < ndata; ii++){
        CuAssertDblEquals(tc,y[ii],vals[ii],1e-15);
    }

    running_core_total_restart(runeval_lr);
    running_core_total_restart(runeval_rl);
    running_core_total_arr_restart(a->dim,rungrad);
    function_train_param_grad_eval(a,ndata,x,runeval_lr,runeval_rl,rungrad,nparam,vals,grad,
                                   core_grad_space,core_space_size,max_func_param_space);
    
    running_core_total_restart(runeval_lr);
    running_core_total_restart(runeval_rl);
    running_core_total_arr_restart(a->dim,rungrad);
    function_train_param_grad_eval(a,ndata,x,runeval_lr,runeval_rl,rungrad,nparam,vals,grad,
                                   core_grad_space,core_space_size,max_func_param_space);

    running_core_total_restart(runeval_lr);
    running_core_total_restart(runeval_rl);
    running_core_total_arr_restart(a->dim,rungrad);
    function_train_param_grad_eval(a,ndata,x,runeval_lr,runeval_rl,rungrad,nparam,vals,grad,
                                   core_grad_space,core_space_size,max_func_param_space);

    /* printf("grad = "); dprint(totparam,grad); */
    for (size_t zz = 0; zz < ndata; zz++){
        running = 0;
        double h = 1e-8;
        for (size_t ii = 0; ii < dim; ii++){
            /* printf("dim = %zu\n",ii); */
            for (size_t jj = 0; jj < nparam[ii]; jj++){
                /* printf("\t jj = %zu\n", jj); */
                guess[running+jj] += h;
                function_train_core_update_params(a,ii,nparam[ii],guess + running);
            
                double val2 = function_train_eval(a,x+zz*dim);
                double fd = (val2-y[zz])/h;
                /* printf("val2=%G, y[0]=%G\n",val2,y[0]); */
                /* printf("fd = %3.15G, calc is %3.15G\n",fd,grad[running+jj + zz * totparam]); */
                /* printf("\t fd[%zu,%zu] = %3.5G\n",jj,zz,fd); */
                CuAssertDblEquals(tc,fd,grad[running+jj + zz * totparam],1e-5);
                guess[running+jj] -= h;
                function_train_core_update_params(a,ii,nparam[ii],guess + running);
            }
            running += nparam[ii];
        }
    }

    free(guess); guess = NULL;
    
    free(vals); vals = NULL;
    free(grad); grad = NULL;
    
    free(core_grad_space);      core_grad_space = NULL;
    free(max_func_param_space); max_func_param_space = NULL;
        
    running_core_total_free(runeval_lr); runeval_lr = NULL;
    running_core_total_free(runeval_rl); runeval_rl = NULL;
    running_core_total_arr_free(dim,rungrad);    rungrad = NULL;
    
    bounding_box_free(bds); bds = NULL;
    function_train_free(a); a   = NULL;
    free(x);                x   = NULL;
    free(y);                y   = NULL;
}

void Test_LS_AIO(CuTest * tc)
{
    srand(seed);
    printf("Testing Function: regress_aio_ls (5 dimensional, max rank = 3, max order = 3) \n");
    printf("\t Num degrees of freedom = O(5 * 3 * 3 * 4) = O(180)\n");

    size_t dim = 5;

    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[6] = {1,2,3,2,3,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 300;
    /* size_t ndata = 200; */
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = function_train_eval(a,x+ii*dim);
        /* y[ii] += randn(); */
    }


    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(dim * 25 * (maxorder+1));
    size_t onparam=0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);

        for (size_t jj = 0; jj < ranks[ii]; jj++){
            for (size_t kk = 0; kk < ranks[ii+1]; kk++){
                for (size_t ll = 0; ll < maxorder+1; ll++){
                    param_space[onparam] += 0.1 * pow(0.1,ll);
                    onparam++;
                }
            }
        }
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    struct RegressOpts* ropts = regress_opts_create(AIO,FTLS,ndata,dim,x,y);

    struct FunctionTrain * ft_final = c3_regression_run(ftp,ropts);
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-3);
    
    ft_param_free(ftp);      ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;
    free(param_space); param_space = NULL;
    bounding_box_free(bds); bds       = NULL;
    function_train_free(a); a         = NULL;
    function_train_free(ft_final); ft_final = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
}

void Test_LS_AIO2(CuTest * tc)
{
    printf("Testing Function: regress_aio_ls (5 dimensional, max rank = 8, max order = 3) \n");
    printf("\t Num degrees of freedom = O(5 * 8 * 8 * 4) = O(1280)\n");

    srand(seed);
    size_t dim = 5;

    size_t ranks[6] = {1,2,3,2,8,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);
    
    // create data
    /* size_t ndata = dim * 8 * 8 * (maxorder+1); // slightly more than degrees of freedom */
    size_t ndata = 500;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = function_train_eval(a,x+ii*dim);
        /* y[ii] += randn(); */
    }

    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(dim * 25 * (maxorder+1));
    size_t onparam=0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);

        for (size_t jj = 0; jj < ranks[ii]; jj++){
            for (size_t kk = 0; kk < ranks[ii+1]; kk++){
                for (size_t ll = 0; ll < maxorder+1; ll++){
                    param_space[onparam] += randu()*2.0-1.0;
                    onparam++;
                }
            }
        }
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    struct RegressOpts* ropts = regress_opts_create(AIO,FTLS,ndata,dim,x,y);
    size_t * npercore = ft_param_get_num_params_per_core(ftp);
    regress_opts_initialize_memory(ropts, npercore,
                                   ranks, maxorder+1,LINEAR_ST);


    struct FunctionTrain * ft_final = c3_regression_run(ftp,ropts);
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-3);
    
    ft_param_free(ftp);      ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;
    free(param_space); param_space = NULL;
    bounding_box_free(bds); bds       = NULL;
    function_train_free(a); a         = NULL;
    function_train_free(ft_final); ft_final = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;

}

void Test_LS_AIO3(CuTest * tc)
{
    printf("Testing Function: regress_aio_ls (5 dimensional, max rank = 8, max order = 10) \n");
    printf("\t Num degrees of freedom = O(5 * 8 * 8 * 11) = O(3520)\n");

    srand(seed);
    
    size_t dim = 5;

    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[6] = {1,2,3,2,8,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 5* dim * 8 * 8 * (maxorder+1);
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = function_train_eval(a,x+ii*dim);
        /* y[ii] += randn(); */
    }

    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(dim * 8*8 * (maxorder+1));
    size_t onparam=0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);

        for (size_t jj = 0; jj < ranks[ii]; jj++){
            for (size_t kk = 0; kk < ranks[ii+1]; kk++){
                for (size_t ll = 0; ll < maxorder+1; ll++){
                    param_space[onparam] += randu()*2.0-1.0;
                    onparam++;
                }
            }
        }
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    struct RegressOpts* ropts = regress_opts_create(AIO,FTLS,ndata,dim,x,y);
    size_t * npercore = ft_param_get_num_params_per_core(ftp);
    regress_opts_initialize_memory(ropts, npercore,
                                   ranks, maxorder+1,LINEAR_ST);


    struct FunctionTrain * ft_final = c3_regression_run(ftp,ropts);
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-3);
    
    ft_param_free(ftp);      ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;
    free(param_space); param_space = NULL;
    bounding_box_free(bds); bds       = NULL;
    function_train_free(a); a         = NULL;
    function_train_free(ft_final); ft_final = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
}

void Test_LS_c3approx_interface(CuTest * tc)
{
    printf("Testing Function: c3approx_interface\n");
    /* srand(seed); */
    
    size_t dim = 5;
    size_t ranks[6] = {1,2,2,2,2,1};
    size_t maxrank = 3;
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;

    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);
    
    // create data
    size_t ndata = 1000;//dim * 8 * 8 * (maxorder+1);
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = function_train_eval(a,x+ii*dim);
    }


    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct C3Approx * c3a = c3approx_create(REGRESS,dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
    }

    // Set regression parameters
    c3approx_set_regress_type(c3a,AIO);
    c3approx_set_regress_start_ranks(c3a,maxrank);
    c3approx_set_regress_num_param_per_func(c3a,maxorder+1);
    c3approx_init_regress(c3a);

    // Perform regression
    struct FunctionTrain * ft_final = c3approx_do_regress(c3a,ndata,x,1,y,1,FTLS);
    
    /* printf("done!\n"); */
    
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-3);

    function_train_free(ft_final); ft_final = NULL;
    bounding_box_free(bds); bds       = NULL;
    function_train_free(a); a         = NULL;
    free(x); x = NULL;
    free(y); y = NULL;

    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
}


double funccv(double * x)
{
    double ret = 0.0;
    for (size_t ii = 0; ii < 5; ii++){
        ret += x[ii]*x[ii];
    }

    return ret;
}


void Test_LS_cross_validation(CuTest * tc)
{
    printf("Testing Function: cross validation\n");
    srand(seed);
    
    size_t dim = 5;
    size_t maxrank = 2;
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 2;

    // create data
    size_t ndata = 40;//dim * 8 * 8 * (maxorder+1);
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = funccv(x+ii*dim);
    }

    // Regression
    size_t kfold = 5;
    struct CrossValidate * cv = cross_validate_init(ndata,dim,x,y,kfold);

    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
    }

    struct FTRegress * ftr = ft_regress_alloc(dim,fapp);
    ft_regress_set_type(ftr,AIO);
    ft_regress_set_obj(ftr,FTLS);
    ft_regress_set_data(ftr,ndata,x,1,y,1);

    size_t nperfunc = maxorder+1;
    size_t opt_maxiter = 1000;
    ft_regress_set_parameter(ftr,"rank",&maxrank);
    ft_regress_set_parameter(ftr,"num_param",&nperfunc);
    ft_regress_set_parameter(ftr,"opt maxiter",&opt_maxiter);
    ft_regress_process_parameters(ftr);

    double err = cross_validate_run(cv,ftr);

    printf("\t CV Error estimate = %G\n",err);

    // Increase rank and order -> more sensitive so CV error should go up

    size_t newrank = maxrank+2;
    nperfunc = maxorder+2;
    opt_maxiter = 1000;
    ft_regress_set_parameter(ftr,"rank",&newrank);
    ft_regress_set_parameter(ftr,"num_param",&nperfunc);
    ft_regress_set_parameter(ftr,"opt maxiter",&opt_maxiter);
    ft_regress_process_parameters(ftr);

    double err2 = cross_validate_run(cv,ftr);

    printf("\t CV Error estimate = %G\n",err2);

    /* CuAssertIntEquals(tc,err<err2,1); */

    // set options for parameters
    size_t norder_ops = 6;
    size_t order_ops[6] = {1,2,3,4,5,6};
    size_t nranks = 4;
    size_t rank_ops[4] ={1,2,3,4};
    /* size_t nmiters = 3; */
    /* size_t miter_ops[3]={500,1000,2000}; */

    cross_validate_add_discrete_param(cv,"num_param",norder_ops,order_ops);
    cross_validate_add_discrete_param(cv,"rank",nranks,rank_ops);
    /* cross_validate_add_discrete_param(cv,"opt maxiter",nmiters,sizeof(size_t),miter_ops); */
    cross_validate_opt(cv,ftr,2);
    
    struct FunctionTrain * ft = ft_regress_run(ftr);
    /* struct FunctionTrain * ft = NULL; */

    // check to make sure options are set to optimal ones
    size_t nparams_per_func;
    function_train_core_get_nparams(ft,0,&nparams_per_func);
    CuAssertIntEquals(tc,3,nparams_per_func);
    size_t * ranks = function_train_get_ranks(ft);
    iprint_sz(dim+1,ranks);
    for (size_t jj = 1; jj < dim; jj++){
        /* CuAssertIntEquals(tc,2,ranks[jj]); */
        function_train_core_get_nparams(ft,jj,&nparams_per_func);
        CuAssertIntEquals(tc,3,nparams_per_func); // this is just for regression testing (to match prev code)
        
    }

    size_t ntest = 1000;
    double norm = 0.0;
    double err3 = 0.0;
    double pt[5];
    for (size_t ii = 0; ii < ntest; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }

        double val = funccv(pt);
        double val2 = function_train_eval(ft,pt);
        double diff = val-val2;
        err3 += diff*diff;
        norm += val*val;
    }

    printf("error = %G\n",err3/norm);
    cross_validate_free(cv); cv = NULL;
    
    free(x); x = NULL;
    free(y); y = NULL;

    function_train_free(ft); ft = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    ft_regress_free(ftr); ftr = NULL;

    CuAssertIntEquals(tc,0,0);
}


void Test_LS_AIO_new(CuTest * tc)
{
    srand(seed);
    printf("Testing Function_new interface: regress_aio_ls (5 dimensional, max rank = 4, max order = 3) \n");

    size_t dim = 5;
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    /* size_t ranks[3] = {1,3,1}; */
    size_t ranks[6] = {1,3,2,4,2,1};
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a =
        function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);
    
    // create data
    size_t ndata = 500;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = function_train_eval(a,x+ii*dim);
        /* y[ii] += randn(); */
    }


    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(dim * 25 * (maxorder+1));
    double * true_params = calloc_double(dim * 25 * (maxorder+1));
    size_t onparam=0;
    size_t incr, running=0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);

        incr = function_train_core_get_params(a,ii,
                                              param_space + running);
        function_train_core_get_params(a,ii,
                                       true_params + running);
        /* for (size_t jj = 0; jj < incr; jj++){ */
        /*     printf("%zu,%G\n",running+jj,true_params[running+jj]); */
        /* } */
        running += incr;
        for (size_t jj = 0; jj < ranks[ii]; jj++){
            for (size_t kk = 0; kk < ranks[ii+1]; kk++){
                param_space[onparam] += randn()*0.01;
                onparam++;
            }
        }
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    size_t * npercore = ft_param_get_num_params_per_core(ftp);
    for (size_t jj = 0; jj < dim; jj++){
        CuAssertIntEquals(tc,ranks[jj]*ranks[jj+1]*(maxorder+1),npercore[jj]);
                          
    }
    
    struct RegressOpts* ropts = regress_opts_create(AIO,FTLS,ndata,dim,x,y);
    regress_opts_initialize_memory(ropts, npercore,
                                   ranks, maxorder+1,LINEAR_ST);

    // Some tests
    ft_param_update_params(ftp,true_params);
    double val = ft_param_eval_objective_aio(ftp,ropts,NULL);
    double * check_param = calloc_double(dim * 25 * (maxorder+1));
    running=0;
    /* printf("\n\n\n"); */
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        incr = function_train_core_get_params(ft_param_get_ft(ftp),ii,
                                              check_param);
        for (size_t jj = 0; jj < incr; jj++){
            /* printf("%zu,%G,%G\n",running+jj,true_params[running+jj],check_param[jj]); */
            CuAssertDblEquals(tc,true_params[running+jj],check_param[jj],1e-20);
        }
        
        running+= incr;
    }
    free(check_param); check_param = NULL;
    CuAssertDblEquals(tc,0.0,val,1e-20);


    ft_param_update_params(ftp,param_space);
    struct FunctionTrain * ft = c3_regression_run(ftp,ropts);
    double diff = function_train_relnorm2diff(ft,a);
    printf("Difference = %G\n",diff);
    /* CuAssertDblEquals(tc,0.0,diff,1e-7); */


    size_t nperfunc = maxorder+1;
    struct FTRegress * reg = ft_regress_alloc(dim,fapp);
    ft_regress_set_type(reg,AIO);
    ft_regress_set_obj(reg,FTLS);
    ft_regress_set_parameter(reg,"num_param",&nperfunc);
    ft_regress_set_start_ranks(reg,ranks);
    ft_regress_set_data(reg,ndata,x,1,y,1);
    ft_regress_process_parameters(reg);

    /* ft_regress_update_params(reg,param_space); */
    struct FunctionTrain * ft2 = ft_regress_run(reg);

    diff = function_train_relnorm2diff(ft2,a);
    printf("Difference = %G\n",diff);
    /* CuAssertDblEquals(tc,0.0,diff,1e-7); */
    
    ft_regress_free(reg);     reg = NULL;
    function_train_free(ft2); ft2 = NULL;
    function_train_free(ft);  ft = NULL;
    free(param_space);        param_space = NULL;
    free(true_params);        true_params = NULL;
    ft_param_free(ftp);       ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;
    bounding_box_free(bds);   bds  = NULL;
    function_train_free(a);   a    = NULL;

    free(x); x = NULL;
    free(y); y = NULL;

    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}


void Test_function_train_param_grad_sqnorm(CuTest * tc)
{
    printf("Testing Function: function_train_param_grad_sqnorm \n");
    srand(seed);
    size_t dim = 4;    
    double weights[4] = {1.0,1.0,1.0,1.0};
    
    size_t ranks[5] = {1,2,3,8,1};

    /* size_t ranks[5] = {1,2,2,2,1}; */
    double lb = -0.6;
    double ub = 1.0;
    size_t maxorder = 10; // 10
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    size_t nparam = function_train_get_nparams(a);
    double * guess = calloc_double(nparam);
    function_train_get_params(a,guess);
    double * grad = calloc_double(nparam);
    
    double val = function_train_param_grad_sqnorm(a,weights,grad);

    /* printf("val = %G\n",val); */
    size_t running = 0;
    size_t notused;
    for (size_t zz = 0; zz < dim; zz++){
        double h = 1e-8;
        /* printf("nparam[%zu] = %zu\n",zz,nparam[zz]); */
        size_t nparam = function_train_core_get_nparams(a,zz,&notused);
        for (size_t jj = 0; jj < nparam; jj++){
            /* printf("jj = %zu\n",jj); */
            guess[running+jj] += h;
            function_train_core_update_params(a,zz,nparam,guess + running);
            /* printf("here?!\n"); */
            double val2 = function_train_param_grad_sqnorm(a,weights,NULL);
            /* printf("val2 = %3.15G\n",val2); */
            double fd = (val2-val)/h;
            /* printf("\t (%3.5G,%3.5G)\n",fd,grad[running+jj]); */
            CuAssertDblEquals(tc,fd,grad[running+jj],1e-5);
            guess[running+jj] -= h;
            function_train_core_update_params(a,zz,nparam,guess + running);
        }
        running += nparam;
        
    }

    free(guess); guess = NULL;
    free(grad); grad = NULL;
    
    bounding_box_free(bds); bds = NULL;
    function_train_free(a); a   = NULL;
}

void Test_SPARSELS_AIO(CuTest * tc)
{
    srand(seed);
    printf("Testing Function: regress_sparse_ls (5 dimensional, max rank = 3, max order = 3) \n");
    printf("\t Num degrees of freedom = O(5 * 3 * 3 * 4) = O(180)\n");

    size_t dim = 5;

    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[6] = {1,2,3,2,3,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 200;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = function_train_eval(a,x+ii*dim);
        /* y[ii] += randn(); */
    }


    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(dim * 25 * (maxorder+1));
    size_t onparam=0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);

        for (size_t jj = 0; jj < ranks[ii]; jj++){
            for (size_t kk = 0; kk < ranks[ii+1]; kk++){
                for (size_t ll = 0; ll < maxorder+1; ll++){
                    param_space[onparam] += randu()*2.0-1.0;
                    onparam++;
                }
            }
        }
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    struct RegressOpts* ropts = regress_opts_create(AIO,FTLS_SPARSEL2,ndata,dim,x,y);
    regress_opts_set_regweight(ropts,3e-12);
    struct FunctionTrain * ft_final = c3_regression_run(ftp,ropts);
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-3);
    
    ft_param_free(ftp);      ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;
    free(param_space); param_space = NULL;
    bounding_box_free(bds); bds       = NULL;
    function_train_free(a); a         = NULL;
    function_train_free(ft_final); ft_final = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
}

void Test_SPARSELS_AIOCV(CuTest * tc)
{
    srand(seed);
    printf("Testing Function: regress_sparse_ls (5 dimensional, max rank = 3, max order = 3) \n");
    printf("\t Num degrees of freedom = O(5 * 3 * 3 * 4) = O(180)\n");

    size_t dim = 5;

    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[6] = {1,2,3,2,3,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 200;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = function_train_eval(a,x+ii*dim);
        /* y[ii] += randn(); */
    }


    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);
    }

    struct FTRegress * ftr = ft_regress_alloc(dim,fapp);
    ft_regress_set_data(ftr,ndata,x,1,y,1);
    ft_regress_set_type(ftr,AIO);
    ft_regress_set_start_ranks(ftr,ranks);
    ft_regress_set_obj(ftr,FTLS_SPARSEL2);
    ft_regress_process_parameters(ftr);
    
    /* // set options for parameters */
    size_t kfold = 10;
    size_t nweight = 12;
    double weight_ops[12]={1e-14, 1e-13, 1e-12,
                           1e-11, 1e-10, 1e-9,
                           1e-8, 1e-7, 1e-6,
                           1e-5, 1e-4, 1e-3};
    struct CrossValidate * cv = cross_validate_init(ndata,dim,x,y,kfold);
    cross_validate_add_discrete_param(cv,"reg_weight",nweight,weight_ops);
    cross_validate_opt(cv,ftr,2);

    ft_regress_process_parameters(ftr);
    struct FunctionTrain * ft = ft_regress_run(ftr);

    double diff = function_train_relnorm2diff(ft,a);
    printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-3);

    
    ft_regress_free(ftr); ftr = NULL;
    cross_validate_free(cv); cv = NULL;

    bounding_box_free(bds); bds = NULL;
    function_train_free(a); a   = NULL;
    function_train_free(ft); ft = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
}


void Test_SPARSELS_cross_validation(CuTest * tc)
{
    printf("Testing Function: cross validation\n");
    srand(seed);
    
    size_t dim = 5;
    double lb = -1.0;
    double ub = 1.0;

    // create data
    size_t ndata = 40;//dim * 8 * 8 * (maxorder+1);
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = funccv(x+ii*dim);
    }

    // Regression
    size_t kfold = 5;
    struct CrossValidate * cv = cross_validate_init(ndata,dim,x,y,kfold);

    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
    }

    struct FTRegress * ftr = ft_regress_alloc(dim,fapp);
    ft_regress_set_data(ftr,ndata,x,1,y,1);
    ft_regress_set_type(ftr,AIO);
    ft_regress_set_obj(ftr,FTLS_SPARSEL2);

    // set options for parameters
    size_t norder_ops = 6;
    size_t order_ops[6] = {1,2,3,4,5,6};
    size_t nranks = 4;
    size_t rank_ops[4] ={1,2,3,4};
    size_t nweight = 11;
    double weight_ops[11]={1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4};

    cross_validate_add_discrete_param(cv,"num_param",norder_ops,order_ops);
    cross_validate_add_discrete_param(cv,"rank",nranks,rank_ops);
    cross_validate_add_discrete_param(cv,"reg_weight",nweight,weight_ops);
    cross_validate_opt(cv,ftr,2);
    
    struct FunctionTrain * ft = ft_regress_run(ftr);
    /* struct FunctionTrain * ft = NULL; */

    // check to make sure options are set to optimal ones
    size_t nparams_per_func;
    function_train_core_get_nparams(ft,0,&nparams_per_func);
    CuAssertIntEquals(tc,3,nparams_per_func);
    size_t * ranks = function_train_get_ranks(ft);
    iprint_sz(dim+1,ranks);
    /* for (size_t jj = 1; jj < dim; jj++){ */
    /*     CuAssertIntEquals(tc,2,ranks[jj]); */
    /*     function_train_core_get_nparams(ft,jj,&nparams_per_func); */
    /*     CuAssertIntEquals(tc,3,nparams_per_func); */
    /* } */

    size_t ntest = 1000;
    double norm = 0.0;
    double err3 = 0.0;
    double pt[5];
    for (size_t ii = 0; ii < ntest; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }

        double val = funccv(pt);
        double val2 = function_train_eval(ft,pt);
        double diff = val-val2;
        err3 += diff*diff;
        norm += val*val;
    }

    printf("error = %G\n",err3/norm);
    cross_validate_free(cv); cv = NULL;
    
    free(x); x = NULL;
    free(y); y = NULL;

    function_train_free(ft); ft = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    ft_regress_free(ftr); ftr = NULL;

    CuAssertIntEquals(tc,0,0);
}


double ff_reg(double * x)
{

    return sin(x[0]*2.0) + x[1] + x[2] + x[3]*x[4] + x[3]*x[2]*x[2]
            + sin(x[0]*x[5]*300.0);
}
void Test_LS_AIO_kernel(CuTest * tc)
{
    srand(seed);
    printf("Testing Function: least squares regression with kernel basis \n");

    size_t dim = 5;
    double lb = -1.0;
    double ub = 1.0;
    /* size_t maxorder = 2; */
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    /* size_t ranks[3] = {1,3,1}; */
    /* size_t ranks[6] = {1,2,3,5,5,1}; */
    size_t ranks[6] = {1,7,3,4,2,1};
    
    // create data
    size_t ndata = 300;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);


    // // add noise
    double maxy = 0.0;
    double miny = 0.0;
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = ff_reg(x+ii*dim);
        if (y[ii] > maxy){
            maxy = y[ii];
        }
        else if (y[ii] < miny){
            miny = y[ii];
        }
        y[ii] += 0.1*randn();
    }
    
    /* for (size_t ii = 0; ii < ndata; ii++){ */
    /*     y[ii] = (2.0*y[ii]-maxy-miny)/(maxy-miny); */
    /* } */
    

    // Initialize Approximation Structure
    size_t nparams = 10;
    double * centers = linspace(lb,ub,nparams);
    double scale = 1.0;
    double width = pow(nparams,-0.2)*2.0/12.0;
    width *= 20;
    printf("width = %G\n",width);
    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,centers,scale,width);


    struct OneApproxOpts * qmopts = one_approx_opts_alloc(KERNEL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(dim * 25 * nparams);
    size_t onparam=0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);

        for (size_t jj = 0; jj < ranks[ii]; jj++){
            for (size_t kk = 0; kk < ranks[ii+1]; kk++){
                for (size_t ll = 0; ll < nparams; ll++){
                    param_space[onparam] = (randu()*2.0-1.0);
                    onparam++;
                }
            }
        }
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    size_t * npercore = ft_param_get_num_params_per_core(ftp);
    for (size_t jj = 0; jj < dim; jj++){
        CuAssertIntEquals(tc,ranks[jj]*ranks[jj+1]*nparams,npercore[jj]);
                          
    }
    
    /* struct RegressOpts* ropts = regress_opts_create(AIO,FTLS_SPARSEL2,ndata,dim,x,y); */
    struct RegressOpts* ropts = regress_opts_create(AIO,FTLS,ndata,dim,x,y);
    regress_opts_set_verbose(ropts,1);
    regress_opts_set_convtol(ropts,1e-3);
    regress_opts_set_regweight(ropts,1e-9); // great for AIO


    double resid = 0.0;
    struct FunctionTrain * ft = c3_regression_run(ftp,ropts);
    for (size_t ii = 0; ii < ndata; ii++){
        double eval = function_train_eval(ft,x+ii*dim);
        /* printf("x = ");dprint(dim,x+ii*dim); */
        /* printf(" y = %G, ft = %G\n",y[ii],eval); */
        resid += pow(y[ii]-eval,2);
    }
    resid /= (double) ndata;
    printf("resid = %G\n",resid);
    /* CuAssertDblEquals(tc,0,resid,1e-3); */

    double diff = 0.0;
    double norm = 0.0;
    for (size_t ii = 0; ii < 1000; ii++){
        double pt[5];
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }
        double v1 = ff_reg(pt);
        /* v1 = (2*v1-maxy-miny)/(maxy-miny); */
        double v2 = function_train_eval(ft,pt);

        /* dprint(5,pt); */
        /* printf(" v1 = %G, v2 = %G\n",v1,v2); */
        /* v2 *= (maxy-miny); */
        /* v2 += maxy + miny; */
        /* v2 /= 2.0; */
        diff += pow(v1-v2,2);

        norm += pow(v1,2);

        /* (2.0*y[ii]-maxy-miny)/(maxy-miny); */

    }
    printf("Difference = %G, norm = %G, rat = %G\n",diff,norm,diff/norm);
    /* CuAssertDblEquals(tc,0.0,diff,1e-7); */
                                         
    free(centers);
    function_train_free(ft);  ft = NULL;
    free(param_space);        param_space = NULL;
    ft_param_free(ftp);       ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;

    free(x); x = NULL;
    free(y); y = NULL;

    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}



double ff_reg2(double * x)
{
    double out = 0.0;
    for (size_t ii = 0; ii < 10; ii++){
        out += x[ii];
    }

    return sin(out);
}

void Test_LS_AIO_kernel2(CuTest * tc)
{
    srand(seed);
    printf("Testing Function: least squares regression with kernel basis \n");

    size_t dim = 10;
    double lb = 0.0;
    double ub = 1.0;
    size_t ranks[11] = {1,2,2,2,2,2,
                        2,2,2,2,1};
    /* size_t ranks[3] = {1,3,1}; */
    /* size_t ranks[6] = {1,2,3,5,5,1}; */
    /* size_t ranks[11] = {1,4,3,4,2, */
    /*                     5,4,3,4,3,1}; */
    
    // create data
    size_t ndata = 600;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);


    // // add noise
    double maxy = 0.0;
    double miny = 0.0;
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = ff_reg2(x+ii*dim);
        if (y[ii] > maxy){
            maxy = y[ii];
        }
        else if (y[ii] < miny){
            miny = y[ii];
        }
        /* y[ii] += 0.1*randn(); */
    }
    
    /* for (size_t ii = 0; ii < ndata; ii++){ */
    /*     y[ii] = (2.0*y[ii]-maxy-miny)/(maxy-miny); */
    /* } */
    

    // Initialize Approximation Structure
    size_t nparams = 5;
    double * centers = linspace(lb,ub,nparams);
    double scale = 1.0;
    double width = pow(nparams,-0.2)*2.0/12.0;
    width *= 20;
    /* width *= 10; */
    /* printf("width = %G\n",width); */
    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,centers,scale,width);


    struct OneApproxOpts * qmopts = one_approx_opts_alloc(KERNEL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(dim * 25 * nparams);
    size_t onparam=0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);

        for (size_t jj = 0; jj < ranks[ii]; jj++){
            for (size_t kk = 0; kk < ranks[ii+1]; kk++){
                for (size_t ll = 0; ll < nparams; ll++){
                    param_space[onparam] = (randu()*2.0-1.0);
                    onparam++;
                }
            }
        }
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    size_t * npercore = ft_param_get_num_params_per_core(ftp);
    for (size_t jj = 0; jj < dim; jj++){
        CuAssertIntEquals(tc,ranks[jj]*ranks[jj+1]*nparams,npercore[jj]);
                          
    }
    
    struct RegressOpts* ropts = regress_opts_create(AIO,FTLS,ndata,dim,x,y);
    regress_opts_set_verbose(ropts,1);
    regress_opts_set_convtol(ropts,1e-9);
    regress_opts_set_regweight(ropts,1e-9); // great for AIO


    double resid = 0.0;
    struct FunctionTrain * ft = c3_regression_run(ftp,ropts);
    for (size_t ii = 0; ii < ndata; ii++){
        double eval = function_train_eval(ft,x+ii*dim);
        /* printf("x = ");dprint(dim,x+ii*dim); */
        /* printf(" y = %G, ft = %G\n",y[ii],eval); */
        resid += pow(y[ii]-eval,2);
    }
    resid /= (double) ndata;
    printf("resid = %G\n",resid);
    /* CuAssertDblEquals(tc,0,resid,1e-3); */

    double diff = 0.0;
    double norm = 0.0;
    for (size_t ii = 0; ii < 1000; ii++){
        double pt[10];
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }
        double v1 = ff_reg2(pt);
        /* v1 = (2*v1-maxy-miny)/(maxy-miny); */
        double v2 = function_train_eval(ft,pt);

        /* dprint(5,pt); */
        /* printf(" v1 = %G, v2 = %G\n",v1,v2); */
        /* v2 *= (maxy-miny); */
        /* v2 += maxy + miny; */
        /* v2 /= 2.0; */
        diff += pow(v1-v2,2);

        norm += pow(v1,2);

        /* (2.0*y[ii]-maxy-miny)/(maxy-miny); */

    }
    printf("Difference = %G, norm = %G, rat = %G\n",diff,norm,diff/norm);
    CuAssertDblEquals(tc,0.0,diff/norm,1e-2);
                                         
    free(centers);
    function_train_free(ft);  ft = NULL;
    free(param_space);        param_space = NULL;
    ft_param_free(ftp);       ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;

    free(x); x = NULL;
    free(y); y = NULL;

    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}


double kern_rosen(double * x)
{
    double f1 = 10.0 * (x[1] - pow(x[0],2));
    double f2 = (1.0 - x[0]);
    double out = pow(f1,2) + pow(f2,2);

    return out;
}

void Test_LS_AIO_kernel3(CuTest * tc)
{
    srand(seed);
    printf("Testing Function: least squares regression with kernel basis (on rosen) \n");

    size_t dim = 2;
    double lb = -2.0;
    double ub = 2.0;
    size_t ranks[3] = {1,4,1};
    
    // create data
    size_t ndata = 1000;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);


    // // add noise
    double maxy = 0.0;
    double miny = 0.0;
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = kern_rosen(x+ii*dim);
        if (y[ii] > maxy){
            maxy = y[ii];
        }
        else if (y[ii] < miny){
            miny = y[ii];
        }
        /* y[ii] += 0.1*randn(); */
    }
    
    /* for (size_t ii = 0; ii < ndata; ii++){ */
    /*     y[ii] = (2.0*y[ii]-maxy-miny)/(maxy-miny); */
    /* } */
    

    // Initialize Approximation Structure
    size_t nparams = 10;
    double * centers = linspace(lb,ub,nparams);
    double scale = 1.0;
    double width = pow(nparams,-0.2)*2.0/12.0;
    width *= 20;
    /* width *= 10; */
    /* printf("width = %G\n",width); */
    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,centers,scale,width);


    struct OneApproxOpts * qmopts = one_approx_opts_alloc(KERNEL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(dim * 25 * nparams);
    size_t onparam=0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);

        for (size_t jj = 0; jj < ranks[ii]; jj++){
            for (size_t kk = 0; kk < ranks[ii+1]; kk++){
                for (size_t ll = 0; ll < nparams; ll++){
                    param_space[onparam] = (randu()*2.0-1.0);
                    onparam++;
                }
            }
        }
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    size_t * npercore = ft_param_get_num_params_per_core(ftp);
    for (size_t jj = 0; jj < dim; jj++){
        CuAssertIntEquals(tc,ranks[jj]*ranks[jj+1]*nparams,npercore[jj]);
                          
    }
    
    struct RegressOpts* ropts = regress_opts_create(AIO,FTLS_SPARSEL2,ndata,dim,x,y);
    regress_opts_set_verbose(ropts,3);
    regress_opts_set_convtol(ropts,1e-9);
    regress_opts_set_regweight(ropts,1e-9); // great for AIO


    double resid = 0.0;
    struct FunctionTrain * ft = c3_regression_run(ftp,ropts);
    for (size_t ii = 0; ii < ndata; ii++){
        double eval = function_train_eval(ft,x+ii*dim);
        /* printf("x = ");dprint(dim,x+ii*dim); */
        /* printf(" y = %G, ft = %G\n",y[ii],eval); */
        resid += pow(y[ii]-eval,2);
    }
    resid /= (double) ndata;
    printf("resid = %G\n",resid);
    /* CuAssertDblEquals(tc,0,resid,1e-3); */

    double diff = 0.0;
    double norm = 0.0;
    for (size_t ii = 0; ii < 1000; ii++){
        double pt[2];
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }
        double v1 = kern_rosen(pt);
        /* v1 = (2*v1-maxy-miny)/(maxy-miny); */
        double v2 = function_train_eval(ft,pt);

        /* dprint(2,pt); */
        /* printf(" v1 = %G, v2 = %G\n",v1,v2); */
        /* v2 *= (maxy-miny); */
        /* v2 += maxy + miny; */
        /* v2 /= 2.0; */
        diff += pow(v1-v2,2);

        norm += pow(v1,2);

        /* (2.0*y[ii]-maxy-miny)/(maxy-miny); */

    }
    printf("Difference = %G, norm = %G, rat = %G\n",diff,norm,diff/norm);
    CuAssertDblEquals(tc,0.0,diff/norm,1e-2);
                                         
    free(centers);
    function_train_free(ft);  ft = NULL;
    free(param_space);        param_space = NULL;
    ft_param_free(ftp);       ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;

    free(x); x = NULL;
    free(y); y = NULL;

    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}



CuSuite * CLinalgRegressGetSuite()
{
    CuSuite * suite = CuSuiteNew();
    /* SUITE_ADD_TEST(suite, Test_LS_ALS); */
    /* SUITE_ADD_TEST(suite, Test_LS_ALS2); */
    /* SUITE_ADD_TEST(suite, Test_LS_ALS_SPARSE2); */

    /* SUITE_ADD_TEST(suite, Test_function_train_param_grad_eval); */
    /* SUITE_ADD_TEST(suite, Test_function_train_core_param_grad_eval1); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO2); */
    /* SUITE_ADD_TEST(suite, Test_LS_c3approx_interface); */
    /* SUITE_ADD_TEST(suite, Test_LS_cross_validation); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_new); */

    /* SUITE_ADD_TEST(suite, Test_function_train_param_grad_sqnorm); */
    /* SUITE_ADD_TEST(suite, Test_SPARSELS_AIO); */
    /* SUITE_ADD_TEST(suite, Test_SPARSELS_AIOCV); */
    /* SUITE_ADD_TEST(suite, Test_SPARSELS_cross_validation); */

    /* SUITE_ADD_TEST(suite, Test_LS_AIO_kernel); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_kernel2); */
    SUITE_ADD_TEST(suite, Test_LS_AIO_kernel3);
        
    // takes too many points
    /* SUITE_ADD_TEST(suite, Test_LS_AIO3); */
    return suite;
}
