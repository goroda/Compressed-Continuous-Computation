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


static unsigned int seed = 3;

void Test_LS_ALS(CuTest * tc)
{
    srand(seed);
    printf("LS_ALS: Testing ALS regression on a randomly generated low rank function \n");
    printf("\t  Dimensions: 5\n");
    printf("\t  Ranks:      [1 2 8 3 2 1]\n");
    printf("\t  LPOLY order: 3\n");
    printf("\t  nunknowns:   200\n");
    printf("\t  ndata:       600\n");


    size_t dim = 5;
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[6] = {1,2,8,3,2,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 600;
    /* printf("\t ndata = %zu\n",ndata); */
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

    size_t onparam=0;
    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){nunknowns += (maxorder+1)*ranks[ii]*ranks[ii+1];}
    double * param_space = calloc_double(nunknowns);
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

    /* printf("\t nunknowns = %zu\n",nunknowns); */
    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    struct RegressOpts* ropts = regress_opts_create(dim,ALS,FTLS);
    regress_opts_set_verbose(ropts,0);
    regress_opts_set_als_conv_tol(ropts,1e-5);
    regress_opts_set_max_als_sweep(ropts,70);
    struct c3Opt* optimizer = c3opt_create(BFGS);
    c3opt_set_relftol(optimizer,1e-9);
    /* c3opt_set_relftol(optimizer,1e-9); */

    struct FunctionTrain * ft_final = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\n\t  Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-3);
    
    ft_param_free(ftp);            ftp         = NULL;
    regress_opts_free(ropts);      ropts       = NULL;
    free(param_space);             param_space = NULL;
    bounding_box_free(bds);        bds         = NULL;
    function_train_free(a);        a           = NULL;
    function_train_free(ft_final); ft_final    = NULL;

    c3opt_free(optimizer); optimizer = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
}

void Test_LS_ALS2(CuTest * tc)
{
    srand(seed);
    printf("\nLS_ALS2: Testing ALS regression on a randomly generated low rank function \n");
    printf("\t  Dimensions: 5\n");
    printf("\t  Ranks:      [1 2 3 2 5 1]\n");
    printf("\t  LPOLY order: 8\n");
    printf("\t  nunknowns:   261\n");
    printf("\t  ndata:       4000\n");

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
    double * param_space = calloc_double(261);
    size_t onparam=0;

    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        
        for (size_t jj = 0; jj < ranks[ii]; jj++){
            for (size_t kk = 0; kk < ranks[ii+1]; kk++){
                for (size_t ll = 0; ll < maxorder+1; ll++){
                    param_space[onparam] += (randu()*2.0-1.0);
                    onparam++;
                }
            }
        }
    }

    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_gtol(optimizer,1e-4);
    
    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    struct RegressOpts* als_opts = regress_opts_create(dim,ALS,FTLS);
    regress_opts_set_verbose(als_opts,0);
    regress_opts_set_als_conv_tol(als_opts,1e-6);
    regress_opts_set_max_als_sweep(als_opts,100);
    
    struct FunctionTrain * ft_final = c3_regression_run(ftp,als_opts,optimizer,ndata,x,y);
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\n\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-3);
    
    /* struct RegressOpts * aio_opts = regress_opts_create(dim,AIO,FTLS); */
    /* struct FunctionTrain * ft_final2 = c3_regression_run(ftp,aio_opts,optimizer,ndata,x,y); */
    /* double diff2 = function_train_relnorm2diff(ft_final2,a); */
    /* printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff2); */

    /* c3opt_set_gtol(optimizer,1e-10); */
    /* struct FunctionTrain * ft_final3 = c3_regression_run(ftp,als_opts,optimizer,ndata,x,y); */
    /* double diff3 = function_train_relnorm2diff(ft_final3,a); */
    /* printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff3); */
    /* CuAssertDblEquals(tc,0.0,diff3,1e-3); */
    
    ft_param_free(ftp);             ftp         = NULL;
    regress_opts_free(als_opts);    als_opts    = NULL;
    free(param_space);              param_space = NULL;
    bounding_box_free(bds);         bds         = NULL;
    function_train_free(a);         a           = NULL;
    function_train_free(ft_final);  ft_final    = NULL;

    /* regress_opts_free(aio_opts);    aio_opts    = NULL; */
    /* function_train_free(ft_final2); ft_final2   = NULL; */
    /* function_train_free(ft_final3); ft_final3   = NULL; */

    c3opt_free(optimizer); optimizer = NULL;
    
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
}

void Test_LS_ALS_SPARSE2(CuTest * tc)
{
    srand(seed);
    printf("\nLS_ALS_SPARSE2: Testing ALS regression on a randomly generated low rank function \n");
    printf("\t  Dimensions: 5\n");
    printf("\t  Ranks:      [1 2 3 2 5 1]\n");
    printf("\t  LPOLY order: 5\n");
    printf("\t  nunknowns:   174\n");
    printf("\t  ndata:       2000\n");

    /* printf("Testing Function: regress_als with sparse reg (5 dimensional, max rank = 5, max order = 8) \n"); */

    size_t dim = 5;
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[6] = {1,2,3,2,5,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 5;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);


    // create data
    size_t ndata = 2000;
    /* printf("\t ndata = %zu\n",ndata); */
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

    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){ nunknowns += (maxorder+1)*ranks[ii]*ranks[ii+1];}
    /* printf("\t nunknowns = %zu \n",nunknowns); */
    double * param_space = calloc_double(nunknowns);
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

    struct c3Opt * optimizer = c3opt_create(BFGS);
    /* c3opt_set_gtol(optimizer,1e-7); */
    
    double regweight = 1e-4;
    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    struct RegressOpts* als_opts = regress_opts_create(dim,ALS,FTLS_SPARSEL2);
    regress_opts_set_verbose(als_opts,0);
    regress_opts_set_max_als_sweep(als_opts,100);
    regress_opts_set_als_conv_tol(als_opts,1e-3);
    regress_opts_set_regularization_weight(als_opts,regweight);
    struct FunctionTrain * ft_final = c3_regression_run(ftp,als_opts,optimizer,ndata,x,y);
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\n\t Relative Error after first round of ALS: ||f - f_approx||/||f|| = %G\n",diff);

    
    struct RegressOpts * aio_opts = regress_opts_create(dim,AIO,FTLS_SPARSEL2);
    regress_opts_set_verbose(aio_opts,0);
    regress_opts_set_regularization_weight(aio_opts,regweight);
    struct FunctionTrain * ft_final2 = c3_regression_run(ftp,aio_opts,optimizer,ndata,x,y);
    double diff2 = function_train_relnorm2diff(ft_final2,a);
    printf("\t Relative Error after continuing with AIO: ||f - f_approx||/||f|| = %G\n",diff2);


    regress_opts_set_verbose(als_opts,0);
    regress_opts_set_regularization_weight(als_opts,regweight);
    struct FunctionTrain * ft_final3 = c3_regression_run(ftp,als_opts,optimizer,ndata,x,y);
    double diff3 = function_train_relnorm2diff(ft_final3,a);
    printf("\t Relative Error after second round of ALS: ||f - f_approx||/||f|| = %G\n",diff3);
    CuAssertDblEquals(tc,0.0,diff3,1e-3);
    
    ft_param_free(ftp);             ftp         = NULL;
    regress_opts_free(aio_opts);    aio_opts    = NULL;
    regress_opts_free(als_opts);    als_opts    = NULL;
    free(param_space);              param_space = NULL;
    bounding_box_free(bds);         bds         = NULL;
    function_train_free(a);         a           = NULL;
    function_train_free(ft_final);  ft_final    = NULL;
    function_train_free(ft_final2); ft_final2   = NULL;
    function_train_free(ft_final3); ft_final3   = NULL;


    c3opt_free(optimizer); optimizer = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
}

void Test_function_train_core_param_grad_eval1(CuTest * tc)
{
    printf("\nTesting Function: function_train_param_grad_eval1 \n");
    printf("\t  Dimensions: 4\n");
    printf("\t  Ranks:      [1 2 3 8 1]\n");
    printf("\t  LPOLY order: 10\n");

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
    size_t ndata = 1; // 10
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    struct RunningCoreTotal * runeval_lr = ftutil_running_tot_space(a);
    struct RunningCoreTotal * runeval_rl = ftutil_running_tot_space(a);
    struct RunningCoreTotal * rungrad = ftutil_running_tot_space(a);

    /* size_t core = 2; */
    for (size_t core = 0; core < dim; core++){
        /* printf("core = %zu\n",core); */
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

    
        /* printf("\t Testing Evaluation\n"); */


        
        function_train_core_pre_post_run(a,core,ndata,x,runeval_lr,runeval_rl);
        /* printf("\n\n"); */
        /* for (size_t jj = 0; jj < ndata; jj++){ */
        /*     double * core1 = calloc_double(ranks[0]*ranks[1]); */
        /*     double * core2 = calloc_double(ranks[1]*ranks[2]); */
        /*     double * core3 = calloc_double(ranks[2]*ranks[3]); */
        /*     double * core4 = calloc_double(ranks[3]*ranks[4]); */

        /*     qmarray_eval(a->cores[0],x[jj*dim+0],core1); */
        /*     qmarray_eval(a->cores[1],x[jj*dim+1],core2); */
        /*     qmarray_eval(a->cores[2],x[jj*dim+2],core3); */
        /*     qmarray_eval(a->cores[3],x[jj*dim+3],core4); */
        /*     double * left_eval = calloc_double(8); // max rank */
        /*     double * right_eval = calloc_double(8); // max rank; */
        /*     double * temp = calloc_double(8); */
        /*     if (core == 0){ */
        /*         memmove(right_eval,core4,ranks[3]*sizeof(double));                 */
        /*         cblas_dgemv(CblasColMajor,CblasNoTrans,ranks[2],ranks[3],1.0,core3,ranks[2], */
        /*                     right_eval,1,0.0,temp,1); */
        /*         /\* printf("after first right eval\n "); dprint(ranks[2],temp); *\/ */
                
        /*         memmove(right_eval,temp,ranks[2]*sizeof(double)); */

        /*         cblas_dgemv(CblasColMajor,CblasNoTrans,ranks[1],ranks[2],1.0,core2,ranks[1], */
        /*                     right_eval,1,0.0,temp,1); */
        /*         /\* printf("after second right eval\n "); dprint(ranks[1],temp); *\/ */
        /*         memmove(right_eval,temp,ranks[1]*sizeof(double)); */

        /*         double * rlvals = running_core_total_get_vals(runeval_rl); */
        /*         for (size_t zz = 0; zz < ranks[1]; zz++){ */
        /*             CuAssertDblEquals(tc,right_eval[zz],rlvals[zz],1e-15); */
        /*         } */

        /*     } */

        /*     free(core1); core1 = NULL; */
        /*     free(core2); core2 = NULL; */
        /*     free(core3); core3 = NULL; */
        /*     free(core4); core4 = NULL; */
        /*     free(left_eval); left_eval = NULL; */
        /*     free(right_eval); right_eval = NULL; */
        /*     free(temp); temp = NULL; */
            
        /* } */
        
        
        function_train_core_param_grad_eval(a,core,ndata,x,runeval_lr,runeval_rl,NULL,nparam,vals,NULL,
                                             NULL,0,NULL);

        for (size_t ii = 0; ii < ndata; ii++){
            double reldiff = (y[ii]-vals[ii])/(y[ii]);
            CuAssertDblEquals(tc,0.0,reldiff,1e-14);
            
        }
    
        /* printf("\t Testing Gradient\n"); */
        function_train_core_param_grad_eval(a,core,ndata,x,runeval_lr,runeval_rl,rungrad,nparam,vals,grad,
                                             core_grad_space,core_space_size,max_func_param_space);
        for (size_t ii = 0; ii < ndata; ii++){
            double reldiff = (y[ii]-vals[ii])/(y[ii]);
            CuAssertDblEquals(tc,0.0,reldiff,1e-14);
            /* CuAssertDblEquals(tc,y[ii],vals[ii],1e-14); */
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
            double h = 1e-7;
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
    printf("\nTesting Function: function_train_param_grad_eval \n");
    printf("\t  Dimensions: 4\n");
    printf("\t  Ranks:      [1 2 3 8 1]\n");
    printf("\t  LPOLY order: 10\n");

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

    
    /* printf("\t Testing Evaluation\n"); */
    function_train_param_grad_eval(a,ndata,x,runeval_lr,NULL,NULL,nparam,vals,NULL,
                                   core_grad_space,core_space_size,max_func_param_space);

    for (size_t ii = 0; ii < ndata; ii++){
        CuAssertDblEquals(tc,y[ii],vals[ii],1e-15);
    }

    
    /* printf("\t Testing Gradient\n"); */
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
        double h = 1e-6;
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
                double reldiff = (fd - grad[running+jj + zz * totparam])/fd;
                CuAssertDblEquals(tc,0.0,fabs(reldiff),2e-5);
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

void Test_function_train_param_grad_eval_simple(CuTest * tc)
{
    printf("\nTesting Function: function_train_param_grad_eval_simple \n");
    printf("\t  Dimensions: 4\n");
    printf("\t  Ranks:      [1 2 3 8 1]\n");
    printf("\t  LPOLY order: 10\n");

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

    size_t temp_nparam, totparam = 0;
    size_t nparam[4];
    for (size_t ii = 0; ii < dim; ii++){
        nparam[ii] = function_train_core_get_nparams(a,ii,&temp_nparam);
        totparam += nparam[ii];
    }

    double * vals = calloc_double(ndata);
    double * grad = calloc_double(ndata*totparam);

    double * guess = calloc_double(totparam);
    size_t runtot = 0;
    size_t running = 0;
    for (size_t zz = 0; zz < dim; zz++){
        for (size_t jj = 0; jj < nparam[zz]; jj++){
            guess[running+jj] = randn();
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

    function_train_param_grad_eval_simple(a,ndata,x,vals,NULL);

    for (size_t ii = 0; ii < ndata; ii++){
        CuAssertDblEquals(tc,y[ii],vals[ii],1e-15);
    }

    
    /* printf("\t Testing Gradient\n"); */
    function_train_param_grad_eval_simple(a,ndata,x,vals,grad);
    for (size_t ii = 0; ii < ndata; ii++){
        CuAssertDblEquals(tc,y[ii],vals[ii],1e-15);
    }


    /* printf("grad = "); dprint(totparam,grad); */
    for (size_t zz = 0; zz < ndata; zz++){
        running = 0;
        double h = 1e-6;
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
                double reldiff = (fd - grad[running+jj + zz * totparam])/fd;
                CuAssertDblEquals(tc,0.0,reldiff,2e-5);
                guess[running+jj] -= h;
                function_train_core_update_params(a,ii,nparam[ii],guess + running);
            }
            running += nparam[ii];
        }
    }

    free(guess); guess = NULL;
    free(vals); vals = NULL;
    free(grad); grad = NULL;
    
    bounding_box_free(bds); bds = NULL;
    function_train_free(a); a   = NULL;
    free(x);                x   = NULL;
    free(y);                y   = NULL;
}

void Test_LS_AIO(CuTest * tc)
{
    srand(seed);
    /* printf("Testing Function: regress_aio_ls (5 dimensional, max rank = 3, max order = 3) \n"); */
    /* printf("\t Num degrees of freedom = O(5 * 3 * 3 * 4) = O(180)\n"); */
    printf("\nLS_AIO: Testing AIO regression on a randomly generated low rank function \n");
    printf("\t  Dimensions: 5\n");
    printf("\t  Ranks:      [1 2 3 2 3 1]\n");
    printf("\t  LPOLY order: 3\n");
    printf("\t  nunknowns:   92\n");
    printf("\t  ndata:       300\n");

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
    /* printf("\t ndata = %zu\n",ndata); */
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
    size_t nunknown = 0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        nunknown += (maxorder+1)*ranks[ii]*ranks[ii+1];
    }

    /* printf("\t nunknown = %zu\n",nunknown); */
    /* struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks); */
    struct FTparam* ftp = ft_param_alloc(dim,fapp,NULL,ranks);
    ft_param_create_from_lin_ls(ftp,ndata,x,y,1e-10);
    struct RegressOpts* ropts = regress_opts_create(dim,AIO,FTLS);
    struct c3Opt * optimizer = c3opt_create(BFGS);
    /* c3opt_set_verbose(optimizer,1); */
    /* c3opt_set_relftol(optimizer,1e-5); */
    /* c3opt_set_relftol(optimizer,1e-12); */
    /* c3opt_set_gtol(optimizer,1e-12); */
    
    struct FunctionTrain * ft_final = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\n\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-3);
    
    ft_param_free(ftp);      ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    function_train_free(ft_final); ft_final = NULL;
    bounding_box_free(bds); bds       = NULL;
    function_train_free(a); a         = NULL;

    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
}

void Test_LS_AIO2(CuTest * tc)
{
    printf("\nLS_AIO2: Testing AIO regression on a randomly generated low rank function \n");
    printf("\t  Dimensions: 5\n");
    printf("\t  Ranks:      [1 2 3 2 8 1]\n");
    printf("\t  LPOLY order: 3\n");
    printf("\t  nunknowns:   152\n");
    printf("\t  ndata:       1000\n");
    
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
    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        nunknowns += (maxorder+1)*(ranks[ii]*ranks[ii+1]);
    }

    /* printf("\t nunknowns = %zu\n",nunknowns); */
    struct FTparam* ftp = ft_param_alloc(dim,fapp,NULL,ranks);
    ft_param_create_from_lin_ls(ftp,ndata,x,y,1e-5);
    struct RegressOpts* ropts = regress_opts_create(dim,AIO,FTLS);
    struct c3Opt * optimizer = c3opt_create(LBFGS);
    c3opt_set_verbose(optimizer,0);
    c3opt_set_gtol(optimizer,1e-8);
    c3opt_set_relftol(optimizer,1e-8);

    struct FunctionTrain * ft_final = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\n\t  Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-3);
    
    ft_param_free(ftp);            ftp       = NULL;
    regress_opts_free(ropts);      ropts     = NULL;
    c3opt_free(optimizer);         optimizer = NULL;
    bounding_box_free(bds);        bds       = NULL;
    function_train_free(a);        a         = NULL;
    function_train_free(ft_final); ft_final  = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;

}

void Test_LS_AIO3(CuTest * tc)
{
    printf("\nLS_AIO3: Testing AIO regression on a randomly generated low rank function\n");
    printf("\t  Dimensions: 5\n");
    printf("\t  Ranks:      [1 2 3 2 8 1]\n");
    printf("\t  LPOLY order: 5\n");
    printf("\t  nunknowns:   229\n");
    printf("\t  ndata:       2290\n");


    srand(seed);
    
    size_t dim = 5;

    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[6] = {1,2,3,2,8,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 5;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t nunknown = 1;
    for (size_t ii = 0; ii < dim; ii++){
        nunknown += ranks[ii]*ranks[ii+1]*(maxorder+1);
    }

    /* printf("nunknown =%zu\n",nunknown); */
    size_t ndata = nunknown*10;

    /* printf("\t Ndata: %zu\n",ndata); */
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

    struct FTparam* ftp = ft_param_alloc(dim,fapp,NULL,ranks);
    ft_param_create_from_lin_ls(ftp,ndata,x,y,1e-2);
    struct RegressOpts* ropts = regress_opts_create(dim,AIO,FTLS);
    struct c3Opt* optimizer = c3opt_create(BFGS);
    c3opt_set_verbose(optimizer,0);
    c3opt_set_gtol(optimizer,1e-6);
    c3opt_set_maxiter(optimizer,2000);


    struct FunctionTrain * ft_final = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\n\t  Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,2e-3);
    
    ft_param_free(ftp);      ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;
    c3opt_free(optimizer);
    bounding_box_free(bds); bds       = NULL;
    function_train_free(a); a         = NULL;
    function_train_free(ft_final); ft_final = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
}


void Test_LS_AIO_new(CuTest * tc)
{
    srand(seed);
    printf("\nLS_AIO_new: Testing AIO regression on a randomly generated low rank function \n");
    printf("\t  Dimensions: 5\n");
    printf("\t  Ranks:      [1 3 2 4 2 1]\n");
    printf("\t  LPOLY order: 3\n");
    printf("\t  nunknowns:   108\n");
    printf("\t  ndata:       1000\n");

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
    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){ nunknowns += (maxorder+1)*ranks[ii]*ranks[ii+1];}
    
    double * param_space = calloc_double(nunknowns);
    double * true_params = calloc_double(nunknowns);
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
    struct RegressOpts* ropts = regress_opts_create(dim,AIO,FTLS);
    struct c3Opt * optimizer = c3opt_create(BFGS);

    
    /* // Some tests */
    ft_param_update_params(ftp,true_params);
    double val = ft_param_eval_objective_aio(ftp,ropts,NULL,ndata,x,y,NULL);
    double * check_param = calloc_double(nunknowns);
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

    struct FunctionTrain * ft = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);
    double diff = function_train_relnorm2diff(ft,a);
    printf("\n\t  Relative Error from low level interface = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-4);

    struct FTRegress * reg = ft_regress_alloc(dim,fapp,ranks);
    ft_regress_set_alg_and_obj(reg,AIO,FTLS);
    ft_regress_update_params(reg,param_space);
    struct FunctionTrain * ft2 = ft_regress_run(reg,optimizer,ndata,x,y);

    diff = function_train_relnorm2diff(ft2,a);
    printf("\t  Relative Error from higher level interface = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-4);
    
    ft_regress_free(reg);     reg = NULL;
    function_train_free(ft2); ft2 = NULL;

    c3opt_free(optimizer); optimizer = NULL;
    
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




static double lin_func(double * x){
    double w[5] = {0.2, -0.2, 0.4, 0.3, -0.1};

    double out = 0.0;
    for (size_t ii = 0; ii < 5; ii++){
        out += w[ii]*x[ii];
    }
    return out;
}

void Test_LS_AIO_ftparam_create_from_lin_ls(CuTest * tc)
{
    srand(seed);
    printf("\nTesting Function: ftparam_create_from_lin_ls \n");
    printf("\t  Dimensions:  5\n");
    printf("\t  Ranks:       [1 3 2 4 2 1]\n");
    printf("\t  LPOLY order: 9\n");
    printf("\t  ndata:       10\n");

    size_t dim = 5;
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 9;
    size_t ranks[6] = {1,3,2,4,2,1};
    
    // create data
    size_t ndata = 10;

    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        y[ii] = lin_func(x+ii*dim);

    }

    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        nunknowns += (maxorder+1)*ranks[ii]*ranks[ii+1];
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,NULL,ranks);
    ft_param_create_from_lin_ls(ftp,ndata,x,y,0.0);
    struct FunctionTrain * ft = ft_param_get_ft(ftp);


    double diff = 0.0;
    double norm = 0.0;
    for (size_t ii = 0; ii < 1000; ii++){
        double pt[5];
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }
        double v1 = lin_func(pt);
        double v2 = function_train_eval(ft,pt);
        diff += pow(v1-v2,2);
        norm += pow(v1,2);
    }
    printf("\n\t  Difference = %G, norm = %G, rat = %G\n",diff,norm,diff/norm);
    CuAssertDblEquals(tc,0.0,diff/norm,1e-15);
    
    ft_param_free(ftp);       ftp  = NULL;

    free(x); x = NULL;
    free(y); y = NULL;

    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}

void Test_LS_AIO_ftparam_create_from_lin_ls_kernel(CuTest * tc)
{
    srand(seed);
    printf("\nTesting Function: ftparam_create_from_lin_ls_kernel \n");
    printf("\t  Dimensions: 5\n");
    printf("\t  Ranks:      [1 3 2 4 2 1]\n");
    printf("\t  Nkernels:   10\n");
    printf("\t  ndata:      500\n");

    size_t dim = 5;
    double lb = -1.0;
    double ub = 1.0;
    size_t ranks[6] = {1,3,2,4,2,1};
    
    // create data
    size_t ndata = 500;

    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        y[ii] = lin_func(x+ii*dim);
    }


    // Initialize Approximation Structure
    // Initialize Approximation Structure
    size_t nparams = 10;
    double * centers = linspace(lb,ub,nparams);
    double scale = 1.0;
    double width = pow(nparams,-0.2)*2.0/12.0;
    width *= 20;
    /* printf("\t width = %G\n",width); */

    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,centers,scale,width);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(KERNEL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    /* size_t nunknowns = 0; */
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        /* nunknowns += (maxorder+1)*ranks[ii]*ranks[ii+1]; */
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,NULL,ranks);
    ft_param_create_from_lin_ls(ftp,ndata,x,y,1e-10);

    struct FunctionTrain * ft = ft_param_get_ft(ftp);

    double diff = 0.0;
    double norm = 0.0;
    for (size_t ii = 0; ii < 1000; ii++){
        double pt[5];
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }
        double v1 = lin_func(pt);
        double v2 = function_train_eval(ft,pt);
        diff += pow(v1-v2,2);
        norm += pow(v1,2);
    }
    printf("\n\t  Difference = %G, norm = %G, rat = %G\n",diff,norm,diff/norm);
    CuAssertDblEquals(tc,0.0,diff/norm,1e-6);
    
    /* c3opt_free(optimizer); optimizer = NULL; */
    

    ft_param_free(ftp);       ftp  = NULL;

    free(x); x = NULL;
    free(y); y = NULL;

    free(centers); centers = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}

void Test_LS_AIO_ftparam_update_restricted_ranks(CuTest * tc)
{
    srand(seed);
    printf("\nTesting Function: ftparam_update_restricted_ranks \n");
    printf("\t  Dimensions: 5\n");
    printf("\t  ndata:      500\n");
    
    size_t dim = 5;
    double lb = -1.0;
    double ub = 1.0;
    size_t ranks[6] = {1,3,3,3,3,1};
    
    // create data
    size_t ndata = 500;

    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        y[ii] = lin_func(x+ii*dim);
    }

    // Initialize Approximation Structure
    // Initialize Approximation Structure
    size_t nparams = 10;
    double * centers = linspace(lb,ub,nparams);
    double scale = 1.0;
    double width = pow(nparams,-0.2)*2.0/12.0;
    width *= 20;
    /* printf("\t width = %G\n",width); */

    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,centers,scale,width);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(KERNEL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);

        nunknowns += 2*nparams + 2*nparams + nparams; // just updating last rank!
    }
    
    struct FTparam* ftp = ft_param_alloc(dim,fapp,NULL,ranks);
    size_t ntotparams = ft_param_get_nparams(ftp);
    /* printf("\t ntotparams = %zu\n",ntotparams); */
    double * old_tot_params = calloc_double(ntotparams);
    ft_param_create_from_lin_ls(ftp,ndata,x,y,1e-10);


    struct FunctionTrain * ft = ft_param_get_ft(ftp);
    function_train_get_params(ft,old_tot_params);

    double * new_params = calloc_double(nunknowns);
    size_t rank_start[5] = {2, 2, 2, 2, 2};
    /* printf("\t nunknowns = %zu\n",nunknowns); */
    for (size_t ii = 0; ii < nunknowns; ii++){
        new_params[ii] = randn();
    }
    
    ft_param_update_restricted_ranks(ftp,new_params,rank_start);
    ft = ft_param_get_ft(ftp);

    double * new_tot_params = calloc_double(ntotparams);
    function_train_get_params(ft,new_tot_params);
    
    size_t onparam = 0;
    size_t on_new_param = 0;
    for (size_t kk = 0; kk < dim; kk++){
        for (size_t jj = 0; jj < ranks[kk+1]; jj++){
            for (size_t ii = 0; ii < ranks[kk]; ii++){
                for (size_t ll = 0; ll < nparams; ll++){
                    if ( (ii < 2) && (jj < 2)){
                        CuAssertDblEquals(tc,old_tot_params[onparam],new_tot_params[onparam],1e-15);
                    }
                    else{
                        CuAssertDblEquals(tc,new_params[on_new_param],new_tot_params[onparam],1e-15);
                        on_new_param+=1;
                    }
                    onparam+=1;
                }
            }
        }
    }
    
    free(centers); centers = NULL;
    free(old_tot_params); old_tot_params = NULL;
    free(new_tot_params); new_tot_params = NULL;
    free(new_params);     new_params = NULL;

    ft_param_free(ftp);       ftp  = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}

//  a little more than a linear function
static double lin_func_plus(double * x){
    double w[5] = {0.2, -0.2, 0.4, 0.3, -0.1};

    double out = 0.0;
    for (size_t ii = 0; ii < 5; ii++){
        out += w[ii]*x[ii];
    }

    out += x[0]*x[1];
    
    return out;
}

void Test_LS_AIO_ftparam_restricted_ranks_opt(CuTest * tc)
{
    srand(seed);
    printf("\nTesting Function: ftparam_update_restricted_ranks with optimization \n");

    size_t dim = 5;
    double lb = -1.0;
    double ub = 1.0;
    size_t maxr = 5;
    size_t ranks[6] = {1,maxr,maxr,maxr,maxr,1};
    
    // create data
    size_t ndata = 100;
    printf("\t ndata = %zu\n",ndata);
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        y[ii] = lin_func_plus(x+ii*dim);
    }

    // Initialize Approximation Structure
    // Initialize Approximation Structure
    size_t maxorder = 5;
    size_t nparams = maxorder+1;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,nparams);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    size_t nunknowns = (maxr-2)*nparams + (maxr-2)*nparams;
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);

        if ((ii > 1) && (ii < dim)){
            nunknowns += (maxr-2)*2*nparams + (maxr-2)*2*nparams +
                        (maxr-2)*(maxr-2)*nparams; // just updating ranks after last ones!
        }
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,NULL,ranks);
    size_t ntotparams = ft_param_get_nparams(ftp);
    printf("\t ntotparams = %zu\n",ntotparams);
    double * old_tot_params = calloc_double(ntotparams);
    ft_param_create_from_lin_ls(ftp,ndata,x,y,1e-10);

    struct FunctionTrain * ft = ft_param_get_ft(ftp);
    function_train_get_params(ft,old_tot_params);

    struct RegressOpts* ropts = regress_opts_create(dim,AIO,FTLS);
    size_t rank_start[5] = {2, 2, 2, 2, 2};
    for (size_t ii = 0; ii < dim; ii++){
        regress_opts_set_restrict_rank(ropts,ii,rank_start[ii]);
    }

    printf("\t nunknowns = %zu\n",nunknowns);
    double * new_params = calloc_double(nunknowns);
    for (size_t ii = 0; ii < nunknowns; ii++){
        new_params[ii] = 1e-3;
    }
    ft_param_update_restricted_ranks(ftp,new_params,rank_start);

    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_verbose(optimizer,0);

    struct FunctionTrain * ftfinal = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);
    double * new_tot_params = calloc_double(ntotparams);
    function_train_get_params(ftfinal,new_tot_params);
    
    size_t onparam = 0;
    for (size_t kk = 0; kk < dim; kk++){
        for (size_t jj = 0; jj < ranks[kk+1]; jj++){
            for (size_t ii = 0; ii < ranks[kk]; ii++){
                for (size_t ll = 0; ll < nparams; ll++){
                    if ( (ii < 2) && (jj < 2)){
                        CuAssertDblEquals(tc,old_tot_params[onparam],new_tot_params[onparam],1e-15);
                    }
                    /* else{ */
                    /*     CuAssertDblEquals(tc,new_params[on_new_param],new_tot_params[onparam],1e-15); */
                    /*     on_new_param+=1; */
                    /* } */
                    onparam+=1;
                }
            }
        }
    }

    printf("\t start ranks: "); iprint_sz(dim+1,function_train_get_ranks(ftfinal));
    struct FunctionTrain * ftr1 = function_train_round(ftfinal,1e-8,fapp);
    printf("\t rounded ranks 1e-8: "); iprint_sz(dim+1,function_train_get_ranks(ftr1));
    struct FunctionTrain * ftr2 = function_train_round(ftfinal,5e-5,fapp);
    printf("\t rounded ranks 1e-5: "); iprint_sz(dim+1,function_train_get_ranks(ftr2));

    
    size_t nparam = function_train_get_nparams(ftr2);
    double * newnew = calloc_double(nparam);
    function_train_get_params(ftr2,newnew);
    struct FTparam * ftp2 =  ft_param_alloc(dim,fapp,newnew,function_train_get_ranks(ftr2));
    struct RegressOpts * ropts2 = regress_opts_create(dim,AIO,FTLS);
    struct FunctionTrain * ftf2 = c3_regression_run(ftp2,ropts2,optimizer,ndata,x,y);
    
    double diff = 0.0;
    double diff2 = 0.0;
    double diffr1 = 0.0;
    double diffr2 = 0.0;
    double norm = 0.0;
    for (size_t ii = 0; ii < 10000; ii++){
        double pt[5];
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }
        double v1 = lin_func_plus(pt);
        double v2 = function_train_eval(ftfinal,pt);
        double v3 = function_train_eval(ftr1,pt);
        double v4 = function_train_eval(ftr2,pt);
        double v5 = function_train_eval(ftf2,pt);
        
        diff += pow(v1-v2,2);
        diffr1 += pow(v1-v3,2);
        diffr2 += pow(v1-v4,2);
        diff2 += pow(v1-v5,2);
        norm += pow(v1,2);
    }
    printf("\t Original error  = %G, norm  = %G, rat = %G\n",diff,norm,diff/norm);
    printf("\t Rounded  error  = %G, norm  = %G, rat = %G\n",diffr1,norm,diffr1/norm);
    printf("\t Rounded looser  = %G, norm  = %G, rat = %G\n",diffr2,norm,diffr2/norm);
    printf("\t Optimize from rounded error = %G, norm = %G, rat = %G\n",diff2,norm,diff2/norm);
    CuAssertDblEquals(tc,0.0,diff2/norm,1e-6);

    free(new_params); new_params = NULL;
    free(newnew); newnew = NULL;
    regress_opts_free(ropts2); ropts2 = NULL;
    ft_param_free(ftp2); ftp2 = NULL;
    function_train_free(ftf2); ftf2 = NULL;
    function_train_free(ftr2); ftr2 = NULL;    
    function_train_free(ftr1); ftr1 = NULL;
    function_train_free(ftfinal); ftfinal = NULL;
    
    free(old_tot_params); old_tot_params = NULL;
    free(new_tot_params); new_tot_params = NULL;

    c3opt_free(optimizer); optimizer = NULL;
    regress_opts_free(ropts); ropts = NULL;

    ft_param_free(ftp);       ftp  = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}

/* void Test_LS_c3approx_interface(CuTest * tc) */
/* { */
/*     printf("Testing Function: c3approx_interface\n"); */
/*     /\* srand(seed); *\/ */
    
/*     size_t dim = 5; */
/*     size_t ranks[6] = {1,2,2,2,2,1}; */
/*     size_t maxrank = 3; */
/*     double lb = -1.0; */
/*     double ub = 1.0; */
/*     size_t maxorder = 3; */

/*     struct BoundingBox * bds = bounding_box_init(dim,lb,ub); */
/*     struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder); */
    
/*     // create data */
/*     size_t ndata = 1000;//dim * 8 * 8 * (maxorder+1); */
/*     double * x = calloc_double(ndata*dim); */
/*     double * y = calloc_double(ndata); */

/*     for (size_t ii = 0 ; ii < ndata; ii++){ */
/*         for (size_t jj = 0; jj < dim; jj++){ */
/*             x[ii*dim+jj] = randu()*(ub-lb) + lb; */
/*         } */
/*         // no noise! */
/*         y[ii] = function_train_eval(a,x+ii*dim); */
/*     } */


/*     // Initialize Approximation Structure */
/*     struct OpeOpts * opts = ope_opts_alloc(LEGENDRE); */
/*     ope_opts_set_lb(opts,lb); */
/*     ope_opts_set_ub(opts,ub); */
/*     struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts); */
/*     struct C3Approx * c3a = c3approx_create(REGRESS,dim); */
/*     for (size_t ii = 0; ii < dim; ii++){ */
/*         c3approx_set_approx_opts_dim(c3a,ii,qmopts); */
/*     } */

/*     // Set regression parameters */
/*     c3approx_set_regress_type(c3a,AIO); */
/*     c3approx_set_regress_start_ranks(c3a,maxrank); */
/*     c3approx_set_regress_num_param_per_func(c3a,maxorder+1); */
/*     c3approx_init_regress(c3a); */

/*     // Perform regression */
/*     struct FunctionTrain * ft_final = c3approx_do_regress(c3a,ndata,x,1,y,1,FTLS); */
    
/*     /\* printf("done!\n"); *\/ */
    
/*     double diff = function_train_relnorm2diff(ft_final,a); */
/*     printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff); */
/*     CuAssertDblEquals(tc,0.0,diff,1e-3); */

/*     function_train_free(ft_final); ft_final = NULL; */
/*     bounding_box_free(bds); bds       = NULL; */
/*     function_train_free(a); a         = NULL; */
/*     free(x); x = NULL; */
/*     free(y); y = NULL; */

/*     c3approx_destroy(c3a); */
/*     one_approx_opts_free_deep(&qmopts); */
/* } */


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
    printf("\nTesting Function: cross validation on x_1^2+x_2^2 + ... + x_5^2\n");
    srand(seed);
    printf("\t  Dimensions:  5\n");
    printf("\t  Start ranks: [1 2 2 2 2 1]\n");
    printf("\t  Nunknowns:   48\n");
    printf("\t  ndata:       40\n");
    
    size_t dim = 5;
    size_t ranks[6] = {1,2,2,2,2,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 2;

    // create data
    size_t ndata = 40;
    /* size_t ndata = 80; */
    /* printf("\t ndata = %zu\n",ndata); */
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
            /* printf("%G ",x[ii*dim+jj]); */
        }
        /* printf("\n"); */
        // no noise!
        y[ii] = funccv(x+ii*dim);
    }
    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){
        nunknowns += (maxorder+1)*ranks[ii]*ranks[ii+1];
    }
    /* printf("\t nunknowns = %zu\n",nunknowns); */

    /* dprint2d(ndata,dim,x); */
    // Regression

    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
    }

    struct FTRegress * ftr = ft_regress_alloc(dim,fapp,ranks);
    ft_regress_set_alg_and_obj(ftr,AIO,FTLS);

    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_maxiter(optimizer,1000);
    c3opt_set_verbose(optimizer,0);


    size_t kfold = ndata-1;
    /* size_t kfold = 10; */
    int cvverbose = 0;
    /* printf("\t kfold = %zu\n",kfold); */
    struct CrossValidate * cv = cross_validate_init(ndata,dim,x,y,kfold,cvverbose);

    double err = cross_validate_run(cv,ftr,optimizer);
    printf("\t CV Error estimate = %G\n",err);
    double err2 = cross_validate_run(cv,ftr,optimizer);
    printf("\t CV Error estimate = %G\n",err2);
    CuAssertDblEquals(tc,err,err2,1e-5);

    // increase order
    /* printf("\n \t Increasing order and rank\n"); */
    for (size_t ii = 0; ii < dim; ii++){
        ranks[ii] = ranks[ii]+2;
    }
    ranks[0] = 1;
    ope_opts_set_nparams(opts,maxorder+2);
    ft_regress_reset_param(ftr,fapp,ranks);
    double errb = cross_validate_run(cv,ftr,optimizer);
    /* printf("\t  CV Error estimate = %G\n",errb); */

    CuAssertIntEquals(tc,err<errb,1);

    /* // set options for parameters */
    size_t norder_ops = 6;
    size_t order_ops[6] = {1,2,3,4,5,6};
    size_t nranks = 4;
    size_t rank_ops[4] ={4,3,2,1};
    /* /\* size_t nmiters = 3; *\/ */
    /* /\* size_t miter_ops[3]={500,1000,2000}; *\/ */

    struct CVOptGrid * cvgrid = cv_opt_grid_init(3); // 1 more than normal see if it works
    cv_opt_grid_set_verbose(cvgrid,0);
    cv_opt_grid_add_param(cvgrid,"num_param",norder_ops,order_ops);
    cv_opt_grid_add_param(cvgrid,"rank",nranks,rank_ops);
    cross_validate_grid_opt(cv,cvgrid,ftr,optimizer);
    cv_opt_grid_free(cvgrid);
    
    /* c3opt_set_verbose(optimizer,0); */
    struct FunctionTrain * ft = ft_regress_run(ftr,optimizer,ndata,x,y);
    /* struct FunctionTrain * ft = NULL; */

    // check to make sure options are set to optimal ones
    size_t nparams_per_func;
    function_train_core_get_nparams(ft,0,&nparams_per_func);
    CuAssertIntEquals(tc,3,nparams_per_func);
    /* size_t * ranks_check = function_train_get_ranks(ft); */
    /* iprint_sz(dim+1,ranks_check); */
    for (size_t jj = 1; jj < dim; jj++){
        /* CuAssertIntEquals(tc,2,ranks_check[jj]); // regression testin to match previous */
        function_train_core_get_nparams(ft,jj,&nparams_per_func);
        CuAssertIntEquals(tc,3,nparams_per_func); // this is just for regression testing (to match prev code)
        
    }

    size_t ntest = 1000;
    double norm = 0.0;
    double err_ap = 0.0;
    double pt[5];
    for (size_t ii = 0; ii < ntest; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }

        double val = funccv(pt);
        double val2 = function_train_eval(ft,pt);
        double diff = val-val2;
        err_ap += diff*diff;
        norm += val*val;
    }

    printf("\n\t  abs error = %G, norm = %G, ratio = %G \n",err_ap/(double)ntest,norm/(double)ntest,err_ap/norm);
    
    cross_validate_free(cv); cv = NULL;
    
    free(x); x = NULL;
    free(y); y = NULL;
    c3opt_free(optimizer);
    function_train_free(ft); ft = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    ft_regress_free(ftr); ftr = NULL;
}


void Test_function_train_param_grad_sqnorm(CuTest * tc)
{
    printf("\nTesting Function: function_train_param_grad_sqnorm \n");
    srand(seed);
    size_t dim = 4;
    double weights[4] = {0.5,0.5,0.5,0.5};
    
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
    

    double val_alternate = function_train_param_grad_sqnorm(a,weights,NULL);
    double val = function_train_param_grad_sqnorm(a,weights,grad);

    /* printf("val_alternate = %3.15G\n",val_alternate); */

    double val_should = 0.0;
    for (size_t ii = 0; ii < dim; ii++){
        for (size_t jj = 0; jj < ranks[ii]*ranks[ii+1]; jj++){
            val_should +=weights[ii]*generic_function_inner(a->cores[ii]->funcs[jj],a->cores[ii]->funcs[jj]);
        }
        /* printf("val_should = %3.15G\n",val_should); */
    }
    /* printf("val_should = %3.15G\n",val_should); */
    /* printf("diff = %3.15G\n",(val_should-val_alternate)/val_should); */
    CuAssertDblEquals(tc,0.0,(val_should-val_alternate)/val_should,1e-15);
    CuAssertDblEquals(tc,0.0,(val_should-val)/val_should,1e-15);
    /* printf("val_shoul = %G\n",val_should); */
    /* return; */



    size_t running = 0;
    size_t notused;
    for (size_t zz = 0; zz < dim; zz++){
        double h = 1e-8;
        /* printf("%zu\n",zz); */
        size_t nparam_core = function_train_core_get_nparams(a,zz,&notused);
        for (size_t jj = 0; jj < nparam_core; jj++){
            /* printf("jj = %zu\n",jj); */
            guess[running+jj] += h;
            function_train_core_update_params(a,zz,nparam_core,guess + running);
            /* printf("here?!\n"); */
            double val2 = function_train_param_grad_sqnorm(a,weights,NULL);
            /* printf("val2 = %3.15G\n",val2); */
            double fd = (val2-val)/h;
            /* printf("\t (%3.5G,%3.5G)\n",fd,grad[running+jj]); */
            CuAssertDblEquals(tc,fd,grad[running+jj],1e-5);
            guess[running+jj] -= h;
            function_train_core_update_params(a,zz,nparam_core,guess + running);
        }
        running += nparam_core;
        
    }

    free(guess); guess = NULL;
    free(grad); grad = NULL;
    
    bounding_box_free(bds); bds = NULL;
    function_train_free(a); a   = NULL;
}

void Test_SPARSELS_AIO(CuTest * tc)
{
    srand(seed);
    printf("\nTesting Function: regress_sparse_ls (5 dimensional, max rank = 3, max order = 3) \n");

    size_t dim = 5;

    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[6] = {1,2,3,2,3,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t nunknown = 0;
    for (size_t ii = 0; ii < dim; ii++){
        nunknown += ranks[ii]*ranks[ii+1]*(maxorder+1);
    }
    printf("\t  Number of degrees of freedom in truth = %zu\n",nunknown);
    size_t ndata = 200;
    printf("\t  Number of data = %zu\n",ndata);
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
    size_t rank_approx = 5;
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(dim * 25 * (maxorder+1));
    size_t onparam=0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);

        for (size_t jj = 0; jj < rank_approx; jj++){
            for (size_t kk = 0; kk < rank_approx; kk++){
                for (size_t ll = 0; ll < maxorder+1; ll++){
                    param_space[onparam] += randu()*2.0-1.0;
                    onparam++;
                }
            }
        }
    }
    size_t ranks_approx[6] = {1,rank_approx,rank_approx,rank_approx,rank_approx,1};
    // create data
    nunknown = 0;
    for (size_t ii = 0; ii < dim; ii++){
        nunknown += ranks_approx[ii]*ranks_approx[ii+1]*(maxorder+1);
    }
    printf("\t  Number of unknowns = %zu\n",nunknown);
    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks_approx);
    struct RegressOpts* ropts = regress_opts_create(dim,AIO,FTLS_SPARSEL2);
    regress_opts_set_regularization_weight(ropts,7e-3);
    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_verbose(optimizer,0);
    /* c3opt_set_gtol(optimizer,0); */
    /* c3opt_set_relftol(optimizer,1e-14); */

    struct FunctionTrain * ft_final = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);
    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\n\t  Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,2e-1);
    
    ft_param_free(ftp);      ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;
    free(param_space); param_space = NULL;
    bounding_box_free(bds); bds       = NULL;
    c3opt_free(optimizer); optimizer = NULL;
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
    printf("\nTesting Function: regress_sparse_ls with cross validation (5 dimensional, max rank = 3, max order = 3) \n");

    size_t dim = 5;

    size_t ranks[6] = {1,2,3,2,3,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 200;
    printf("\t ndata = %zu\n",ndata);
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
    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        nunknowns += (maxorder+1)*ranks[ii]*ranks[ii+1];
    }

    printf("\t nunknowns = %zu\n",nunknowns);

    struct FTRegress * ftr = ft_regress_alloc(dim,fapp,ranks);
    ft_regress_set_alg_and_obj(ftr,AIO,FTLS_SPARSEL2);

    struct c3Opt * optimizer = c3opt_create(BFGS);
    /* c3opt_set_verbose(optimizer,1); */
    c3opt_set_maxiter(optimizer,1000);
    /* // set options for parameters */
    size_t kfold = 5;
    int cvverbose = 0;
    struct CrossValidate * cv = cross_validate_init(ndata,dim,x,y,kfold,cvverbose);
    
    size_t nweight = 9;
    double weight_ops[9]={ 1e-11, 1e-10, 1e-9,
                           1e-8, 1e-7, 1e-6,
                           1e-5, 1e-4, 1e-3};
    
    struct CVOptGrid * cvgrid = cv_opt_grid_init(1);
    cv_opt_grid_add_param(cvgrid,"reg_weight",nweight,weight_ops);
    cv_opt_grid_set_verbose(cvgrid,1);

    cross_validate_grid_opt(cv,cvgrid,ftr,optimizer);
    
    double opt_weight = ft_regress_get_regularization_weight(ftr);
    printf("\t (should match best) opt weight = %G\n",opt_weight);

    c3opt_set_maxiter(optimizer,10000);
    struct FunctionTrain * ft = ft_regress_run(ftr,optimizer,ndata,x,y);

    double diff = function_train_relnorm2diff(ft,a);
    printf("\n\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-3);

    ft_regress_free(ftr); ftr = NULL;
    cross_validate_free(cv); cv = NULL;
    cv_opt_grid_free(cvgrid); cvgrid = NULL;
    c3opt_free(optimizer); optimizer = NULL;
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
    printf("\nTesting Function: cross validation with sparse regularization on rank 2 function\n");
    srand(seed);
    
    size_t dim = 5;
    double lb = -1.0;
    double ub = 1.0;
    size_t ranks[6] = {1,1,1,1,1,1};
    // create data
    size_t ndata = 40;//dim * 8 * 8 * (maxorder+1);
    printf("\t ndata = %zu\n",ndata);
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = funccv(x+ii*dim);
    }


    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
    }

    struct FTRegress * ftr = ft_regress_alloc(dim,fapp,ranks);
    ft_regress_set_alg_and_obj(ftr,AIO,FTLS_SPARSEL2);

    struct c3Opt * optimizer = c3opt_create(BFGS);
    
    // set options for parameters
    size_t norder_ops = 6;
    size_t order_ops[6] = {1,2,3,4,5,6};
    size_t nranks = 4;
    size_t rank_ops[4] ={1,2,3,4};
    size_t nweight = 8;
    double weight_ops[8]={1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4};

    size_t kfold = 10;
    int cvverbose = 0;
    printf("\t kfold = %zu\n",kfold);
    struct CrossValidate * cv = cross_validate_init(ndata,dim,x,y,kfold,cvverbose);

    struct CVOptGrid * cvgrid = cv_opt_grid_init(3);
    cv_opt_grid_add_param(cvgrid,"num_param",norder_ops,order_ops);
    cv_opt_grid_add_param(cvgrid,"rank",nranks,rank_ops);
    cv_opt_grid_add_param(cvgrid,"reg_weight",nweight,weight_ops);
    cross_validate_grid_opt(cv,cvgrid,ftr,optimizer);
    cv_opt_grid_free(cvgrid);
    
    struct FunctionTrain * ft = ft_regress_run(ftr,optimizer,ndata,x,y);
    /* struct FunctionTrain * ft = NULL; */

    // check to make sure options are set to optimal ones
    size_t nparams_per_func;
    function_train_core_get_nparams(ft,0,&nparams_per_func);
    CuAssertIntEquals(tc,3,nparams_per_func);
    /* size_t * ranks_opt = function_train_get_ranks(ft); */
    /* iprint_sz(dim+1,ranks_opt); */
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

    printf("\n\t Relative error = %G\n",err3/norm);
    CuAssertDblEquals(tc,0.0,err3/norm,1e-6);
    cross_validate_free(cv); cv = NULL;
    
    free(x); x = NULL;
    free(y); y = NULL;

    function_train_free(ft); ft = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    ft_regress_free(ftr); ftr = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    CuAssertIntEquals(tc,0,0);
}

double ff_reg(double * x)
{

    return sin(x[0]*2.0) + x[1] + x[2] + x[3]*x[4] + x[3]*x[2]*x[2];
           
}

void Test_LS_AIO_kernel(CuTest * tc)
{
    srand(seed);
    printf("\nTesting Function: least squares regression with kernel basis \n");

    size_t dim = 5;
    double lb = -1.0;
    double ub = 1.0;
    /* size_t maxorder = 2; */
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    /* size_t ranks[3] = {1,3,1}; */
    /* size_t ranks[6] = {1,2,3,5,5,1}; */
    size_t ranks[6] = {1,7,5,4,3,1};
    
    // create data
    size_t ndata = 100;
    /* size_t ndata = 500;     */
    printf("\t ndata = %zu\n",ndata);
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
    size_t nparams = 40;
    double * centers = linspace(lb,ub,nparams);
    double scale = 1.0;
    double width = pow(nparams,-0.2)*2.0/12.0;
    width *= 20;
    /* printf("\t width = %G\n",width); */

    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,centers,scale,width);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(KERNEL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);

    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        nunknowns += ranks[ii]*ranks[ii+1]*nparams;
    }
    
    printf("\t nunknowns = %zu\n",nunknowns);

    struct FTparam* ftp = ft_param_alloc(dim,fapp,NULL,ranks);
    ft_param_create_from_lin_ls(ftp,ndata,x,y,1e-10);
    size_t * npercore = ft_param_get_nparams_per_core(ftp);
    for (size_t jj = 0; jj < dim; jj++){
        CuAssertIntEquals(tc,ranks[jj]*ranks[jj+1]*nparams,npercore[jj]);
                          
    }
    
    struct RegressOpts* ropts = regress_opts_create(dim,AIO,FTLS);

    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_verbose(optimizer,0);
    c3opt_set_gtol(optimizer,1e-5);
    /* c3opt_ls_set_maxiter(optimizer,50); */
    c3opt_set_maxiter(optimizer,2000);
    

    double resid = 0.0;
    struct FunctionTrain * ft = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);

    
    for (size_t ii = 0; ii < ndata; ii++){
        double eval = function_train_eval(ft,x+ii*dim);
        /* printf("x = ");dprint(dim,x+ii*dim); */
        /* printf(" y = %G, ft = %G\n",y[ii],eval); */
        resid += pow(y[ii]-eval,2);
    }
    resid /= (double) ndata;
    printf("\t resid = %G\n",resid);
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
    printf("\n\t Error = %G, norm = %G, rat = %G\n",diff,norm,diff/norm);
    CuAssertDblEquals(tc,0.0,diff/norm,1e-1);
                                         
    free(centers); centers = NULL;
    function_train_free(ft);  ft = NULL;
    ft_param_free(ftp);       ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;
    
    c3opt_free(optimizer); optimizer = NULL;

    
    free(x); x = NULL;
    free(y); y = NULL;

    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}

void Test_LS_AIO_kernel_nonlin(CuTest * tc)
{
    srand(seed);
    printf("\nTesting Function: least squares regression with kernel basis and moving centers\n");

    size_t dim = 5;
    double lb = -1.0;
    double ub = 1.0;
    size_t ranks[6] = {1,7,5,4,3,1};
    
    // create data
    size_t ndata = 100;
    /* size_t ndata = 500; */
    printf("\t ndata = %zu\n",ndata);
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = ff_reg(x+ii*dim);
        y[ii] += 0.001*randn();
    }
    

    // Initialize Approximation Structure
    size_t nparams = 5;
    double * centers = linspace(lb,ub,nparams);
    double scale = 1.0;
    double width = pow(nparams,-0.2)*2.0/12.0;
    width *= 10;

    /* printf("\t width = %G\n",width); */

    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,centers,scale,width);
    kernel_approx_opts_set_center_adapt(opts,1);
    
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(KERNEL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);

    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        nunknowns += ranks[ii]*ranks[ii+1]*nparams*2;
    }

    /* double * params = calloc_double(nunknowns); */
    /* for (size_t ii = 0; ii < nunknowns; ii++){ */
    /*     params[ii] = randu()*2.0-1.0; */
    /* } */
    printf("\t nunknowns = %zu\n",nunknowns);

    struct FTparam* ftp = ft_param_alloc(dim,fapp,NULL,ranks);
    ft_param_create_from_lin_ls(ftp,ndata,x,y,1e-8);
    size_t * npercore = ft_param_get_nparams_per_core(ftp);
    for (size_t jj = 0; jj < dim; jj++){
        CuAssertIntEquals(tc,ranks[jj]*ranks[jj+1]*nparams*2,npercore[jj]);
                          
    }
    
    struct RegressOpts* ropts = regress_opts_create(dim,AIO,FTLS);

    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_verbose(optimizer,0);
    c3opt_set_gtol(optimizer,1e-5);
    c3opt_set_relftol(optimizer,1e-20);
    c3opt_set_absxtol(optimizer,0);
    c3opt_ls_set_maxiter(optimizer,50);
    c3opt_set_maxiter(optimizer,2000);
    

    double resid = 0.0;
    struct FunctionTrain * ft = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);

    
    for (size_t ii = 0; ii < ndata; ii++){
        double eval = function_train_eval(ft,x+ii*dim);
        resid += pow(y[ii]-eval,2);
    }
    resid /= (double) ndata;
    printf("\t resid = %G\n",resid);
    /* CuAssertDblEquals(tc,0,resid,1e-3); */

    double diff = 0.0;
    double norm = 0.0;
    for (size_t ii = 0; ii < 1000; ii++){
        double pt[5];
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }
        double v1 = ff_reg(pt);
        double v2 = function_train_eval(ft,pt);

        diff += pow(v1-v2,2);
        norm += pow(v1,2);

    }
    printf("\n\t Error = %G, norm = %G, rat = %G\n",diff,norm,diff/norm);
    CuAssertDblEquals(tc,0.0,diff/norm,1e-1);
                                         
    free(centers); centers = NULL;
    function_train_free(ft);  ft = NULL;
    ft_param_free(ftp);       ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;
    
    c3opt_free(optimizer); optimizer = NULL;

    
    free(x); x = NULL;
    free(y); y = NULL;

    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}

double fff_reg(double * x)
{
    return sin(x[0]*2.0) + cos(x[1] + x[2] + x[3]*x[4]*x[1]) + x[3]*x[2]*x[2] + pow(x[4]*x[0],4);
}

void Test_LS_AIO_rounding(CuTest * tc)
{
    srand(seed);
    printf("\nTesting Function: least squares regression with rounded result \n");

    size_t dim = 5;
    double lb = -1.0;
    double ub = 1.0;
    /* size_t maxorder = 2; */
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    /* size_t ranks[3] = {1,3,1}; */
    /* size_t ranks[6] = {1,2,3,5,5,1}; */
    size_t ranks[6] = {1,10,10,10,10,1};
    
    // create data
    size_t ndata = 1500;
    /* size_t ndata = 500;     */
    printf("\t ndata = %zu\n",ndata);
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
        y[ii] = fff_reg(x+ii*dim);
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
    size_t nparams = 6;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,nparams);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);

    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        nunknowns += ranks[ii]*ranks[ii+1]*nparams;
    }
    
    printf("\t nunknowns = %zu\n",nunknowns);

    struct FTparam* ftp = ft_param_alloc(dim,fapp,NULL,ranks);
    ft_param_create_from_lin_ls(ftp,ndata,x,y,1e-10);
    size_t * npercore = ft_param_get_nparams_per_core(ftp);
    for (size_t jj = 0; jj < dim; jj++){
        CuAssertIntEquals(tc,ranks[jj]*ranks[jj+1]*nparams,npercore[jj]);
                          
    }
    
    struct RegressOpts* ropts = regress_opts_create(dim,AIO,FTLS);

    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_verbose(optimizer,0);
    c3opt_set_gtol(optimizer,1e-4);
    /* c3opt_ls_set_maxiter(optimizer,50); */
    c3opt_set_maxiter(optimizer,2000);
    
    double resid = 0.0;
    struct FunctionTrain * ft = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);
    struct FunctionTrain * ftrounded = function_train_round(ft,1e-10,fapp);
    size_t * rounded_ranks = ftrounded->ranks;
    printf("\t rounded ranks = "); iprint_sz(dim+1,rounded_ranks);

    size_t nparams_rounded = function_train_get_nparams(ftrounded);
    printf("\t nrounded params = %zu\n", nparams_rounded);
    double * new_params = calloc_double(nparams_rounded);
    function_train_get_params(ftrounded,new_params);
    struct FTparam * ftp_new = ft_param_alloc(dim,fapp,new_params,rounded_ranks);
    c3opt_set_gtol(optimizer,1e-6);
    struct FunctionTrain * ftnew = c3_regression_run(ftp_new,ropts,optimizer,ndata,x,y);
    
    for (size_t ii = 0; ii < ndata; ii++){
        /* double eval = function_train_eval(ft,x+ii*dim); */
        /* double eval = function_train_eval(ftrounded,x+ii*dim); */
        double eval = function_train_eval(ftnew,x+ii*dim);
        
        /* printf("x = ");dprint(dim,x+ii*dim); */
        /* printf(" y = %G, ft = %G\n",y[ii],eval); */
        resid += pow(y[ii]-eval,2);
    }
    resid /= (double) ndata;
    printf("\t resid = %G\n",resid);
    /* CuAssertDblEquals(tc,0,resid,1e-3); */

    double diff = 0.0;
    double norm = 0.0;
    for (size_t ii = 0; ii < 1000; ii++){
        double pt[5];
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }
        double v1 = fff_reg(pt);
        /* v1 = (2*v1-maxy-miny)/(maxy-miny); */
        double v2 = function_train_eval(ft,pt);
        /* double v2 = function_train_eval(ftrounded,pt); */
        /* double v2 = function_train_eval(ftnew,pt); */

        /* dprint(5,pt); */
        /* printf(" v1 = %G, v2 = %G\n",v1,v2); */
        /* v2 *= (maxy-miny); */
        /* v2 += maxy + miny; */
        /* v2 /= 2.0; */
        diff += pow(v1-v2,2);

        norm += pow(v1,2);

        /* (2.0*y[ii]-maxy-miny)/(maxy-miny); */

    }
    printf("\t Error = %G, norm = %G, rat = %G\n",diff,norm,diff/norm);
    CuAssertDblEquals(tc,0.0,diff/norm,1e-1);
                                         
    function_train_free(ft);  ft = NULL;
    ft_param_free(ftp);       ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    free(x); x = NULL;
    free(y); y = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    function_train_free(ftrounded); ftrounded = NULL;
    free(new_params);
    ft_param_free(ftp_new); ftp_new = NULL;
    function_train_free(ftnew); ftnew = NULL;
}

/* static double fff_reg2(double * x) */
/* { */
/*     double out = 0.0; */
/*     for (size_t ii = 0; ii < 5; ii++){ */
/*         out += x[ii]; */
/*     } */

/*     return sin(out); */
/* } */

static double ff_reg2(double * x)
{
    double out = 0.0;
    for (size_t ii = 0; ii < 10; ii++){
        out += x[ii];
    }

    return sin(out);
}

void Test_LS_AIO_rankadapt(CuTest * tc)
{
    srand(seed);
    printf("\nTesting Function: least squares regression with rank adaptation \n");

    size_t dim = 5;
    double lb = 0.0;
    double ub = 1.0;
    /* size_t maxorder = 2; */
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    /* size_t ranks[3] = {1,3,1}; */
    /* size_t ranks[6] = {1,2,3,5,5,1}; */
    /* size_t ranks[6] = {1,5,5,5,5,1}; */
    size_t ranks[6] = {1,2,2,2,2,1};
    
    // create data
    size_t ndata = 150;
    /* size_t ndata = 500;     */
    printf("\t ndata = %zu\n",ndata);
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = fff_reg(x+ii*dim);
        /* y[ii] = fff_reg2(x+ii*dim); */
        /* y[ii] += 0.1*randn(); */
    }
    
    // Initialize Approximation Structure
    size_t nparams = 5;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,nparams);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        nunknowns += ranks[ii]*ranks[ii+1]*nparams;
    }
    
    printf("\t nunknowns = %zu\n",nunknowns);

    struct FTRegress * ftr = ft_regress_alloc(dim,fapp,ranks);
    ft_regress_set_alg_and_obj(ftr,AIO,FTLS);
    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_verbose(optimizer,0);
    /* c3opt_set_gtol(optimizer,1e-3); */
    /* /\* c3opt_ls_set_maxiter(optimizer,50); *\/ */
    /* c3opt_set_maxiter(optimizer,2000); */

    ft_regress_set_adapt(ftr,1);
    ft_regress_set_roundtol(ftr,1e-5);
    ft_regress_set_maxrank(ftr,10);
    ft_regress_set_kickrank(ftr,2);
    struct FunctionTrain * ft = ft_regress_run(ftr,optimizer,ndata,x,y);

    double resid = 0.0;
    for (size_t ii = 0; ii < ndata; ii++){
        double eval = function_train_eval(ft,x+ii*dim);
        resid += pow(y[ii]-eval,2);
    }
    resid /= (double) ndata;
    printf("\t resid = %G\n",resid);
    /* CuAssertDblEquals(tc,0,resid,1e-3); */

    double diff = 0.0;
    double norm = 0.0;
    for (size_t ii = 0; ii < 10000; ii++){
        double pt[5];
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }
        /* double v1 = fff_reg2(pt); */
        double v1 = fff_reg(pt);
        double v2 = function_train_eval(ft,pt);
        diff += pow(v1-v2,2);
        norm += pow(v1,2);
    }
    printf("\t Difference = %G, norm = %G, rat = %G\n",diff,norm,diff/norm);
    CuAssertDblEquals(tc,0.0,diff/norm,1e-1);
                                         
    function_train_free(ft);  ft = NULL;
    ft_regress_free(ftr); ftr = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    free(x); x = NULL;
    free(y); y = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}

void Test_LS_AIO_rankadapt_kernel(CuTest * tc)
{
    srand(seed);
    printf("\nTesting Function: least squares regression with kernels and rank adaptation \n");

    size_t dim = 5;
    double lb = 0.0;
    double ub = 1.0;
    /* size_t maxorder = 2; */
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    /* size_t ranks[3] = {1,3,1}; */
    /* size_t ranks[6] = {1,2,3,5,5,1}; */
    /* size_t ranks[6] = {1,5,5,5,5,1}; */
    size_t ranks[6] = {1,2,2,2,2,1};
    
    // create data
    size_t ndata = 150;
    /* size_t ndata = 500;     */
    printf("\t ndata = %zu\n",ndata);
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = fff_reg(x+ii*dim);
        /* y[ii] = fff_reg2(x+ii*dim); */
        /* y[ii] += 0.1*randn(); */
    }
    
    // Initialize Approximation Structure
    size_t nparams = 5;
    double * centers = linspace(lb,ub,nparams);
    double scale = 1.0;
    double width = pow(nparams,-0.2)*2.0/12.0;
    width *= 20;
    /* width *= 10; */
    printf("\t width = %G\n",width);
    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,centers,scale,width);
    size_t nunknown=0;
    for (size_t ii = 0; ii < dim; ii++){ nunknown += (nparams*ranks[ii]*ranks[ii+1]);}
    printf("\t nunknown=%zu\n",nunknown);
    
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(KERNEL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        nunknowns += ranks[ii]*ranks[ii+1]*nparams;
    }
    
    printf("\t nunknowns = %zu\n",nunknowns);

    struct FTRegress * ftr = ft_regress_alloc(dim,fapp,ranks);
    ft_regress_set_alg_and_obj(ftr,AIO,FTLS);
    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_verbose(optimizer,0);
    /* c3opt_set_gtol(optimizer,1e-3); */
    /* /\* c3opt_ls_set_maxiter(optimizer,50); *\/ */
    /* c3opt_set_maxiter(optimizer,2000); */

    ft_regress_set_adapt(ftr,1);
    ft_regress_set_roundtol(ftr,1e-5);
    ft_regress_set_maxrank(ftr,5);
    ft_regress_set_kickrank(ftr,2);
    struct FunctionTrain * ft = ft_regress_run(ftr,optimizer,ndata,x,y);

    double resid = 0.0;
    for (size_t ii = 0; ii < ndata; ii++){
        double eval = function_train_eval(ft,x+ii*dim);
        resid += pow(y[ii]-eval,2);
    }
    resid /= (double) ndata;
    printf("\t resid = %G\n",resid);
    /* CuAssertDblEquals(tc,0,resid,1e-3); */

    double diff = 0.0;
    double norm = 0.0;
    for (size_t ii = 0; ii < 10000; ii++){
        double pt[5];
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }
        /* double v1 = fff_reg2(pt); */
        double v1 = fff_reg(pt);
        double v2 = function_train_eval(ft,pt);
        diff += pow(v1-v2,2);
        norm += pow(v1,2);
    }
    printf("\t Difference = %G, norm = %G, rat = %G\n",diff,norm,diff/norm);
    CuAssertDblEquals(tc,0.0,diff/norm,1e-1);
                                         
    function_train_free(ft);  ft = NULL;
    ft_regress_free(ftr); ftr = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    free(x); x = NULL;
    free(y); y = NULL;
    free(centers); centers = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}





void Test_LS_AIO_kernel2(CuTest * tc)
{
    srand(seed);
    printf("\nTesting Function: least squares regression with kernel basis on sin of sum 10d rank 2\n");

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
    size_t ndata = 100;
    printf("\t ndata = %zu\n",ndata);
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
    printf("\t width = %G\n",width);
    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,centers,scale,width);
    size_t nunknown=0;
    for (size_t ii = 0; ii < dim; ii++){ nunknown += (nparams*ranks[ii]*ranks[ii+1]);}
    printf("\t nunknown=%zu\n",nunknown);
    
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(KERNEL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(nunknown);
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
    size_t * npercore = ft_param_get_nparams_per_core(ftp);
    for (size_t jj = 0; jj < dim; jj++){
        CuAssertIntEquals(tc,ranks[jj]*ranks[jj+1]*nparams,npercore[jj]);
                          
    }
    struct RegressOpts* ropts = regress_opts_create(dim,AIO,FTLS);
    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_verbose(optimizer,0);
    c3opt_set_gtol(optimizer,1e-10);
    
    double resid = 0.0;
    struct FunctionTrain * ft = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);
    for (size_t ii = 0; ii < ndata; ii++){
        double eval = function_train_eval(ft,x+ii*dim);
        /* printf("x = ");dprint(dim,x+ii*dim); */
        /* printf(" y = %G, ft = %G\n",y[ii],eval); */
        resid += pow(y[ii]-eval,2);
    }
    resid /= (double) ndata;
    printf("\t resid = %G\n",resid);
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
    printf("\t Error = %G, norm = %G, rat = %G\n",diff,norm,diff/norm);
    CuAssertDblEquals(tc,0.0,diff/norm,1e-2);
                                         
    free(centers);
    function_train_free(ft);  ft = NULL;
    free(param_space);        param_space = NULL;
    ft_param_free(ftp);       ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;

    c3opt_free(optimizer); optimizer = NULL;

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
    printf("\n Testing Function: least squares regression with kernel basis (on rosen) \n");

    size_t dim = 2;
    double lb = -2.0;
    double ub = 2.0;
    size_t ranks[3] = {1,4,1};
    
    // create data
    size_t ndata = 50;
    printf("\t ndata = %zu\n",ndata);
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
    double width = pow(nparams,-0.2)*(ub-lb)/12.0;
    width *= 20;
    /* width *= 10; */
    printf("\t width = %G\n",width);

    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){ nunknowns += (nparams * ranks[ii]*ranks[ii+1]);}
    printf("\t nunknowns = %zu\n",nunknowns);
    
    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,centers,scale,width);

    struct OneApproxOpts * qmopts = one_approx_opts_alloc(KERNEL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(nunknowns);

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
    size_t * npercore = ft_param_get_nparams_per_core(ftp);
    for (size_t jj = 0; jj < dim; jj++){
        CuAssertIntEquals(tc,ranks[jj]*ranks[jj+1]*nparams,npercore[jj]);
                          
    }
    
    struct RegressOpts* ropts = regress_opts_create(dim,AIO,FTLS_SPARSEL2);
    regress_opts_set_regularization_weight(ropts,1e-9); // great for AIO

    struct c3Opt * optimizer = c3opt_create(BFGS);

    double resid = 0.0;
    struct FunctionTrain * ft = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);
    for (size_t ii = 0; ii < ndata; ii++){
        double eval = function_train_eval(ft,x+ii*dim);
        /* printf("x = ");dprint(dim,x+ii*dim); */
        /* printf(" y = %G, ft = %G\n",y[ii],eval); */
        resid += pow(y[ii]-eval,2);
    }
    resid /= (double) ndata;
    printf("\t resid = %G\n",resid);
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
    printf("\t Error = %G, norm = %G, rat = %G\n",diff,norm,diff/norm);
    CuAssertDblEquals(tc,0.0,diff/norm,1e-2);
                                         
    free(centers);
    function_train_free(ft);  ft = NULL;
    free(param_space);        param_space = NULL;
    ft_param_free(ftp);       ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;

    c3opt_free(optimizer); optimizer = NULL;
    free(x); x = NULL;
    free(y); y = NULL;

    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}


void Test_LS_AIO3_sgd(CuTest * tc)
{
    printf("\nLS_AIO3_sgd: Testing AIO regression on a randomly generated low rank function with SGD\n");
    printf("\t  Dimensions: 5\n");
    printf("\t  Ranks:      [1 2 3 2 8 1]\n");
    printf("\t  LPOLY order: 10\n");
    printf("\t  nunknowns:   419\n");
    printf("\t  ndata:       838\n");

    srand(seed);
    
    size_t dim = 5;

    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[6] = {1,2,2,2,2,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 5;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t nunknown = 1;
    for (size_t ii = 0; ii < dim; ii++){
        nunknown += ranks[ii]*ranks[ii+1]*(maxorder+1);
    }
    printf("nunknown = %zu\n",nunknown);
    size_t ndata = 10*nunknown;
    /* size_t ndata = 2; */
    /* printf("\t Ndata: %zu\n",ndata); */
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
        y[ii] = function_train_eval(a,x+ii*dim);
        if (y[ii] > maxy){
            maxy = y[ii];
        }
        else if (y[ii] < miny){
            miny = y[ii];
        }
        /* y[ii] += randn(); */
    }

    for (size_t ii = 0; ii < ndata; ii++){
        y[ii] = (2.0*y[ii] - maxy - miny)/(maxy-miny);
    }
    dprint(ndata,y);

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

    struct FTparam* ftp = ft_param_alloc(dim,fapp,NULL,ranks);
    ft_param_create_from_lin_ls(ftp,ndata,x,y,1e-4);
    struct RegressOpts* ropts = regress_opts_create(dim,AIO,FTLS);
    regress_opts_set_stoch_obj(ropts,1);
    struct c3Opt * optimizer = c3opt_create(SGD);
    c3opt_set_verbose(optimizer,1);
    c3opt_set_sgd_nsamples(optimizer,ndata);
    c3opt_set_sgd_learn_rate(optimizer,1e-4);
    c3opt_set_maxiter(optimizer,6000);

    /* struct c3Opt* optimizer = c3opt_create(BFGS); */
    /* c3opt_set_verbose(optimizer,0); */
    /* c3opt_set_gtol(optimizer,1e-8); */

    struct FunctionTrain * ft_final = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);
    double num = 0.0;
    double den = 0.0;
    double pt[5];
    for (size_t ii = 0; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randu()*(ub-lb) + lb;
        }
        double f1 = function_train_eval(ft_final,pt)*(maxy-miny)/2 + (maxy+miny)/2;
        double f2 = function_train_eval(a,pt);
        double diff = f1- f2;
        double diff2 = diff*diff;
        den += f2 * f2;
        num += diff2;
    }

    printf("num = %G\n",num);
    printf("den = %G\n",den);
    
    /* double diff = function_train_relnorm2diff(ft_final,a); */
    /* printf("\n\t  Relative Error: ||f - f_approx||/||f|| = %G\n",diff); */
    /* CuAssertDblEquals(tc,0.0,diff,1e-3); */

    /* c3opt_set_sgd_learn_rate(optimizer,1.0); */
    /* struct FunctionTrain * ft_final2 = c3_regression_run(ftp,ropts,optimizer,ndata,x,y); */
    /* diff = function_train_relnorm2diff(ft_final2,a); */
    /* printf("\n\t  Relative Error: ||f - f_approx||/||f|| = %G\n",diff); */
    /* CuAssertDblEquals(tc,0.0,diff,1e-3); */

    ft_param_free(ftp);      ftp  = NULL;
    regress_opts_free(ropts); ropts = NULL;
    c3opt_free(optimizer);
    bounding_box_free(bds); bds       = NULL;
    function_train_free(a); a         = NULL;
    function_train_free(ft_final); ft_final = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
}

/* void Test_LS_AIO_new_sgd(CuTest * tc) */
/* { */
/*     srand(seed); */
/*     printf("\nLS_AIO_new_sgd: Testing AIO regression on a randomly generated low rank function \n"); */
/*     printf("                  with stochastic gradient descent\n"); */
/*     printf("\t  Dimensions: 5\n"); */
/*     printf("\t  Ranks:      [1 3 2 4 2 1]\n"); */
/*     printf("\t  LPOLY order: 3\n"); */
/*     printf("\t  nunknowns:   108\n"); */
/*     printf("\t  ndata:       1000\n"); */

/*     size_t dim = 5; */
/*     double lb = -1.0; */
/*     double ub = 1.0; */
/*     size_t maxorder = 3; */
/*     /\* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; *\/ */
/*     /\* size_t ranks[3] = {1,3,1}; *\/ */
/*     size_t ranks[6] = {1,3,2,4,2,1}; */
/*     struct BoundingBox * bds = bounding_box_init(dim,lb,ub); */
/*     struct FunctionTrain * a = */
/*         function_train_poly_randu(LEGENDRE,bds,ranks,maxorder); */
    
/*     // create data */
/*     size_t ndata = 1000; */
/*     double * x = calloc_double(ndata*dim); */
/*     double * y = calloc_double(ndata); */

/*     // // add noise */
/*     for (size_t ii = 0 ; ii < ndata; ii++){ */
/*         for (size_t jj = 0; jj < dim; jj++){ */
/*             x[ii*dim+jj] = randu()*(ub-lb) + lb; */
/*         } */
/*         y[ii] = function_train_eval(a,x+ii*dim); */
/*         if (ii == 351){ */
/*             printf("y[351] = %G\n",y[ii]); */
/*         } */
/*     } */


/*     // Initialize Approximation Structure */
/*     struct OpeOpts * opts = ope_opts_alloc(LEGENDRE); */
/*     ope_opts_set_lb(opts,lb); */
/*     ope_opts_set_ub(opts,ub); */
/*     ope_opts_set_nparams(opts,maxorder+1); */
/*     struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts); */
/*     struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim); */
/*     size_t nunknowns = 0; */
/*     for (size_t ii = 0; ii < dim; ii++){ nunknowns += (maxorder+1)*ranks[ii]*ranks[ii+1];} */
    
/*     for (size_t ii = 0; ii < dim; ii++){ */
/*         multi_approx_opts_set_dim(fapp,ii,qmopts); */
/*     } */

/*     struct c3Opt * optimizer = c3opt_create(SGD); */
/*     c3opt_set_verbose(optimizer,0); */
/*     c3opt_set_sgd_nsamples(optimizer,ndata); */
/*     c3opt_set_maxiter(optimizer,500); */
    
/*     struct FTRegress * reg = ft_regress_alloc(dim,fapp,ranks); */
/*     ft_regress_set_alg_and_obj(reg,AIO,FTLS); */
/*     ft_regress_set_stoch_obj(reg,1); */
/*     struct FunctionTrain * ft2 = ft_regress_run(reg,optimizer,ndata,x,y); */

/*     double diff = function_train_relnorm2diff(ft2,a); */
/*     double diffabs = function_train_norm2diff(ft2,a); */
/*     printf("\t  Relative Error from higher level interface = %G\n",diff); */
/*     printf("\t  AbsoluteError from higher level interface = %G\n",diffabs); */


/*     double test_error = 0; */
/*     double pt[5]; */
/*     size_t N = 100000; */
/*     for (size_t ii = 0; ii < N; ii++){ */
/*         for (size_t jj = 0; jj < 5; jj++){ */
/*             pt[jj] = 2.0*randu()-1.0; */
/*         } */
/*         double v1 = function_train_eval(ft2,pt); */
/*         double v2 = function_train_eval(a,pt); */
/*         test_error += (v1-v2)*(v1-v2); */
/*     } */
/*     test_error /= (double) N; */
/*     printf("\t  rmse = %G\n",sqrt(test_error*pow(2,5)));//\*pow(2,5)); */
/*     printf("\t  mse = %G\n",test_error); */

    
/*     CuAssertDblEquals(tc,0.0,diff,1e-3); */
    
/*     ft_regress_free(reg);     reg = NULL; */
/*     function_train_free(ft2); ft2 = NULL; */

/*     c3opt_free(optimizer); optimizer = NULL; */
/*     bounding_box_free(bds);   bds  = NULL; */
/*     function_train_free(a);   a    = NULL; */

/*     free(x); x = NULL; */
/*     free(y); y = NULL; */

/*     one_approx_opts_free_deep(&qmopts); */
/*     multi_approx_opts_free(fapp); */

    
/* } */


CuSuite * CLinalgRegressGetSuite()
{
    CuSuite * suite = CuSuiteNew();

    /* /\* next 3 are good *\/ */
    /* SUITE_ADD_TEST(suite, Test_LS_ALS); */
    /* SUITE_ADD_TEST(suite, Test_LS_ALS2); */
    /* SUITE_ADD_TEST(suite, Test_LS_ALS_SPARSE2); */

    /* /\* next 5 are good *\/ */
    /* SUITE_ADD_TEST(suite, Test_function_train_param_grad_eval); */
    /* SUITE_ADD_TEST(suite, Test_function_train_param_grad_eval_simple); */
    /* SUITE_ADD_TEST(suite, Test_function_train_core_param_grad_eval1); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO2); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO3); */

    /* /\* next 4 are good *\/ */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_new); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_ftparam_create_from_lin_ls); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_ftparam_create_from_lin_ls_kernel); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_ftparam_update_restricted_ranks); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_ftparam_restricted_ranks_opt); */
    /* SUITE_ADD_TEST(suite, Test_LS_cross_validation); */
    
    /* /\* SUITE_ADD_TEST(suite, Test_LS_c3approx_interface); *\/ */

    /* /\* Next 2 are good *\/ */
    /* SUITE_ADD_TEST(suite, Test_function_train_param_grad_sqnorm); */
    /* SUITE_ADD_TEST(suite, Test_SPARSELS_AIO); */

    /* /\* next 2 are good *\/ */
    /* SUITE_ADD_TEST(suite, Test_SPARSELS_AIOCV); */
    /* SUITE_ADD_TEST(suite, Test_SPARSELS_cross_validation); */

    /* /\* Next 3 are good *\/ */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_kernel); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_kernel_nonlin); */

    
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_rounding); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_rankadapt); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_rankadapt_kernel); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_kernel2); */
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_kernel3); */

    SUITE_ADD_TEST(suite, Test_LS_AIO3_sgd);
    /* SUITE_ADD_TEST(suite, Test_LS_AIO_new_sgd); */
        
    return suite;
}


