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
    size_t ranks[6] = {1,2,8,3,2,1};
    /* size_t ranks[6] = {1,2,2,2,2,1}; */
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 600;
    /* size_t ndata = 200; */
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
    /* c3opt_set_relftol(optimizer,1e-12); */
    c3opt_set_verbose(optimizer,0);
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
    /* struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks); */
    struct FTparam* ftp = ft_param_alloc(dim,fapp,NULL,ranks);
    ft_param_create_from_lin_ls(ftp,ndata,x,y,1e-2);

    struct RegressOpts* als_opts = regress_opts_create(dim,ALS,FTLS_SPARSEL2);
    regress_opts_set_verbose(als_opts,0);
    regress_opts_set_max_als_sweep(als_opts,100);
    regress_opts_set_als_conv_tol(als_opts,1e-3);
    regress_opts_set_regularization_weight(als_opts,regweight);

    struct FunctionTrain * ft_final = c3_regression_run(ftp,als_opts,optimizer,ndata,x,y);

    double diff = function_train_relnorm2diff(ft_final,a);
    printf("\n\t Relative Error after first round of ALS: ||f - f_approx||/||f|| = %G\n",diff);
    function_train_free(ft_final);  ft_final    = NULL;
    
    struct RegressOpts * aio_opts = regress_opts_create(dim,AIO,FTLS_SPARSEL2);
    regress_opts_set_verbose(aio_opts,1);
    regress_opts_set_regularization_weight(aio_opts,regweight);

    struct FunctionTrain * ft_final2 = c3_regression_run(ftp,aio_opts,optimizer,ndata,x,y);

    double diff2 = function_train_relnorm2diff(ft_final2,a);
    printf("\t Relative Error after continuing with AIO: ||f - f_approx||/||f|| = %G\n",diff2);
    regress_opts_free(aio_opts);    aio_opts    = NULL;
    function_train_free(ft_final2); ft_final2   = NULL;

    
    regress_opts_set_verbose(als_opts,0);
    regress_opts_set_regularization_weight(als_opts,regweight);

    struct FunctionTrain * ft_final3 = c3_regression_run(ftp,als_opts,optimizer,ndata,x,y);
    double diff3 = function_train_relnorm2diff(ft_final3,a);
    printf("\t Relative Error after second round of ALS: ||f - f_approx||/||f|| = %G\n",diff3);
    CuAssertDblEquals(tc,0.0,diff3,1e-3);
    
    ft_param_free(ftp);             ftp         = NULL;

    regress_opts_free(als_opts);    als_opts    = NULL;
    free(param_space);              param_space = NULL;
    bounding_box_free(bds);         bds         = NULL;
    function_train_free(a);         a           = NULL;

    
    function_train_free(ft_final3); ft_final3   = NULL;


    c3opt_free(optimizer); optimizer = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
}

void Test_ft_param_core_gradeval(CuTest * tc)
{
    printf("\nTesting Function: ft_param_core_gradeval \n");
    printf("\t  Dimensions: 4\n");
    printf("\t  Ranks:      [1 2 3 8 1]\n");
    printf("\t  LPOLY order: 10\n");

    srand(seed);

    size_t dim = 4;
    size_t ranks[5] = {1,2,3,8,1};
    size_t maxorder = 10; 
    size_t nparam = ranks[1]*ranks[0] + ranks[2]*ranks[1] +
                    ranks[3]*ranks[2] + ranks[4]*ranks[3];
    nparam *= (maxorder+1);
    double * param = calloc_double(nparam);
    for (size_t ii = 0; ii < nparam; ii++){
        param[ii] = randn();
    }

    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.5);
    ope_opts_set_ub(opts,2.3);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fapp,qmopts);

    size_t N = 2;
    double pt[8] = {0.2,-1.2,2.1,0.9,
                    0.4, 0.1, 0.3, -1.2};
    struct FTparam * ftp = ft_param_alloc(dim,fapp,param,ranks);
    struct SLMemManager * slm = sl_mem_manager_alloc(dim,N,nparam,LINEAR_ST);
    sl_mem_manager_check_structure(slm,ftp,pt);

    double * grad = calloc_double(nparam);
    double core_evals[100];


    for (size_t ii = 0; ii < N; ii++){
        process_sweep_right_left(ftp,ftp->dim-1,pt[ii*ftp->dim + ftp->dim-1],core_evals,
                                 NULL,slm->running_rl[ftp->dim-1][ii]);
    }
    for (size_t zz = ftp->dim-2; zz > 0; zz--){
         for (size_t ii = 0; ii < N; ii++){
             process_sweep_right_left(ftp,zz,pt[ii * ftp->dim + zz],core_evals,
                                      slm->running_rl[zz+1][ii],slm->running_rl[zz][ii]);
         }   
    }

    size_t core = 0;
    size_t onparam = 0;
    double h = 1e-7;
    for (size_t ii = 0; ii < N; ii++){
        double eval_true = function_train_eval(ftp->ft,pt + ii * dim);
        double eval = ft_param_core_gradeval(ftp,core,pt[core + ii * ftp->dim],grad,NULL,
                                             slm->running_rl[core+1][ii],slm->running_eval);

        for (size_t ll = 0; ll < ftp->nparams_per_core[core]; ll++){
            param[onparam] += h;
            ft_param_update_params(ftp,param);
            double eval2 = function_train_eval(ftp->ft,pt + ii * dim);
            double fd = (eval2 - eval_true)/h;
            CuAssertDblEquals(tc,fd,grad[ll],1e-5);
            param[onparam] -= h;
            ft_param_update_params(ftp,param);
            onparam++;
        }
        onparam -= ftp->nparams_per_core[core];
        
        CuAssertDblEquals(tc,eval_true,eval,1e-11);
    }
    for (size_t ii = 0; ii < N; ii++){
        process_sweep_left_right(ftp,core,pt[ii*ftp->dim+core],core_evals,NULL,slm->running_lr[core][ii]);
    }
    onparam += ftp->nparams_per_core[core];
    for (core = 1; core < ftp->dim-1; core ++){
        for (size_t ii = 0; ii < N; ii++){
            double eval_true = function_train_eval(ftp->ft,pt + ii*dim);
            double eval = ft_param_core_gradeval(ftp,core,pt[core + ii * ftp->dim],grad,
                                                 slm->running_lr[core-1][ii], slm->running_rl[core+1][ii],
                                                 slm->running_eval);
            CuAssertDblEquals(tc,eval_true,eval,1e-11);

            for (size_t ll = 0; ll < ftp->nparams_per_core[core]; ll++){
                param[onparam] += h;
                ft_param_update_params(ftp,param);
                double eval2 = function_train_eval(ftp->ft,pt + ii * dim);
                double fd = (eval2 - eval_true)/h;
                CuAssertDblEquals(tc,fd,grad[ll],1e-5);
                param[onparam] -= h;
                ft_param_update_params(ftp,param);
                onparam++;
            }
            onparam -= ftp->nparams_per_core[core];
        }
        onparam += ftp->nparams_per_core[core];
        
        for (size_t ii = 0; ii < N; ii++){
            process_sweep_left_right(ftp,core,pt[ii*ftp->dim+core],core_evals,slm->running_lr[core-1][ii],
                                     slm->running_lr[core][ii]);
        }
    }

    core = ftp->dim-1;
    for (size_t ii = 0; ii < N; ii++){
        double eval_true = function_train_eval(ftp->ft,pt + ii*dim);
        double eval = ft_param_core_gradeval(ftp,core,pt[core + ii * ftp->dim],grad,
                                             slm->running_lr[core-1][ii], NULL,
                                             slm->running_eval);
        CuAssertDblEquals(tc,eval_true,eval,1e-11);

        for (size_t ll = 0; ll < ftp->nparams_per_core[core]; ll++){
            param[onparam] += h;
            ft_param_update_params(ftp,param);
            double eval2 = function_train_eval(ftp->ft,pt + ii * dim);
            double fd = (eval2 - eval_true)/h;
            CuAssertDblEquals(tc,fd,grad[ll],1e-5);
            param[onparam] -= h;
            ft_param_update_params(ftp,param);
            onparam++;
        }
        onparam -= ftp->nparams_per_core[core];
    }

    free(param);              param = NULL;
    free(grad);               grad = NULL;
    ft_param_free(ftp);       ftp  = NULL;
    sl_mem_manager_free(slm); slm  = NULL;
    multi_approx_opts_free(fapp); fapp = NULL;
    one_approx_opts_free_deep(&qmopts); qmopts = NULL;
}

void Test_ft_param_gradeval(CuTest * tc)
{
    printf("\nTesting Function: ft_param_gradeval \n");
    printf("\t  Dimensions: 4\n");
    printf("\t  Ranks:      [1 2 3 8 1]\n");
    printf("\t  LPOLY order: 10\n");

    srand(seed);
    size_t dim = 4;
    size_t ranks[5] = {1,2,3,8,1};
    size_t maxorder = 10; 
    size_t nparam = ranks[1]*ranks[0] + ranks[2]*ranks[1] +
                    ranks[3]*ranks[2] + ranks[4]*ranks[3];
    nparam *= (maxorder+1);
    double * param = calloc_double(nparam);
    for (size_t ii = 0; ii < nparam; ii++){
        param[ii] = randn();
    }

    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.5);
    ope_opts_set_ub(opts,2.3);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fapp,qmopts);

    double pt[4] = {0.2,-1.2,2.1,0.9};
    struct FTparam * ftp = ft_param_alloc(dim,fapp,param,ranks);
    struct SLMemManager * slm = sl_mem_manager_alloc(dim,1,nparam,LINEAR_ST);
    sl_mem_manager_check_structure(slm,ftp,pt);

    double * grad = calloc_double(nparam);
    double eval_true = function_train_eval(ftp->ft,pt);
    double eval = ft_param_gradeval(ftp,pt,grad,slm->lin_structure_vals,
                                    slm->running_grad,slm->running_eval);

    CuAssertDblEquals(tc,eval_true,eval,1e-11);
    /* dprint(nparam,grad); */
    double h = 1e-7;
    for (size_t zz = 0; zz < nparam; zz++){
        size_t ii = nparam-zz-1;
        /* if (ii <= 351){ */
        /*     printf("ii = %zu\n",ii); */
        /* } */
        param[ii] += h;
        ft_param_update_params(ftp,param);
        double ehere = function_train_eval(ftp->ft,pt);
        double fd  = (ehere-eval_true)/h;

        CuAssertDblEquals(tc,fd,grad[ii],1e-5);
        param[ii] -=h;
        ft_param_update_params(ftp,param);
    }
    
    free(grad);
    multi_approx_opts_free(fapp);
    one_approx_opts_free_deep(&qmopts);
    free(param);
    ft_param_free(ftp);
    sl_mem_manager_free(slm);
}

void Test_ft_param_eval_lin(CuTest * tc)
{
    printf("\nTesting Function: ft_param_eval_lin \n");
    printf("\t  Dimensions: 4\n");
    printf("\t  Ranks:      [1 2 3 8 1]\n");
    printf("\t  LPOLY order: 10\n");

    srand(seed);
    size_t dim = 4;
    size_t ranks[5] = {1,2,3,8,1};
    size_t maxorder = 10; 
    size_t nparam = ranks[1]*ranks[0] + ranks[2]*ranks[1] +
                    ranks[3]*ranks[2] + ranks[4]*ranks[3];
    nparam *= (maxorder+1);
    double * param = calloc_double(nparam);
    for (size_t ii = 0; ii < nparam; ii++){
        param[ii] = randn();
    }

    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.5);
    ope_opts_set_ub(opts,2.3);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fapp,qmopts);

    double pt[4] = {0.2,-1.2,2.1,0.9};
    struct FTparam * ftp = ft_param_alloc(dim,fapp,param,ranks);
    struct SLMemManager * slm = sl_mem_manager_alloc(dim,1,nparam,LINEAR_ST);

    sl_mem_manager_check_structure(slm,ftp,pt);

    double eval_true = function_train_eval(ftp->ft,pt);
    double eval_lin = ft_param_eval_lin(ftp,slm->lin_structure_vals,slm->running_eval);

    CuAssertDblEquals(tc,eval_true,eval_lin,1e-14);
    
    multi_approx_opts_free(fapp);
    one_approx_opts_free_deep(&qmopts);
    free(param);
    ft_param_free(ftp);
    sl_mem_manager_free(slm);
}

void Test_ft_param_gradeval_lin(CuTest * tc)
{
    printf("\n Testing Function: ft_param_gradeval_lin \n");
    printf("\t  Dimensions: 4\n");
    printf("\t  Ranks:      [1 2 3 8 1]\n");
    printf("\t  LPOLY order: 10\n");

    srand(seed);
    size_t dim = 4;
    size_t ranks[5] = {1,2,3,8,1};
    size_t maxorder = 10; 
    size_t nparam = ranks[1]*ranks[0] + ranks[2]*ranks[1] +
                    ranks[3]*ranks[2] + ranks[4]*ranks[3];
    nparam *= (maxorder+1);
    double * param = calloc_double(nparam);
    for (size_t ii = 0; ii < nparam; ii++){
        param[ii] = randn();
    }

    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.5);
    ope_opts_set_ub(opts,2.3);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fapp,qmopts);

    double pt[4] = {0.2,-1.2,2.1,0.9};
    struct FTparam * ftp = ft_param_alloc(dim,fapp,param,ranks);
    struct SLMemManager * slm = sl_mem_manager_alloc(dim,1,nparam,LINEAR_ST);
    sl_mem_manager_check_structure(slm,ftp,pt);

    double eval_true = function_train_eval(ftp->ft,pt);

    /* printf("nparam = %zu\n",nparam); */
    /* printf("eval_true = %G\n",eval_true); */
    double * grad = calloc_double(nparam);
    double eval_should = ft_param_gradeval_lin(ftp,slm->lin_structure_vals,
                                               grad,slm->running_eval,
                                               slm->running_grad);
    CuAssertDblEquals(tc,eval_true,eval_should,1e-10);
    /* dprint(nparam,grad); */
    double h = 1e-7;
    for (size_t zz = 0; zz < nparam; zz++){
        size_t ii = nparam-zz-1;
        /* if (ii <= 351){ */
        /*     printf("ii = %zu\n",ii); */
        /* } */
        param[ii] += h;
        ft_param_update_params(ftp,param);
        double ehere = function_train_eval(ftp->ft,pt);
        double fd  = (ehere-eval_true)/h;

        CuAssertDblEquals(tc,fd,grad[ii],1e-5);
        param[ii] -=h;
        ft_param_update_params(ftp,param);
    }

    free(grad); grad = NULL;
    multi_approx_opts_free(fapp);
    one_approx_opts_free_deep(&qmopts);
    free(param);
    ft_param_free(ftp);
    sl_mem_manager_free(slm);
    

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
    printf("\t  ndata:       350\n");

    size_t dim = 5;

    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[6] = {1,2,3,2,3,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 350;
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
    ft_param_create_from_lin_ls(ftp,ndata,x,y,1e-3);
    struct RegressOpts* ropts = regress_opts_create(dim,AIO,FTLS);
    struct c3Opt * optimizer = c3opt_create(BFGS);
    /* c3opt_set_verbose(optimizer,1); */
    /* c3opt_set_absxtol(optimizer,1e-20); */
    /* c3opt_set_relftol(optimizer,1e-20); */
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
    /* ft_param_update_params(ftp,true_params); */
    /* double val = ft_param_eval_objective_aio(ftp,ropts,NULL,ndata,x,y,NULL); */
    /* double * check_param = calloc_double(nunknowns); */
    /* running=0; */
    /* /\* printf("\n\n\n"); *\/ */
    /* for (size_t ii = 0; ii < dim; ii++){ */
    /*     /\* printf("ii = %zu\n",ii); *\/ */
    /*     incr = function_train_core_get_params(ft_param_get_ft(ftp),ii, */
    /*                                           check_param); */
    /*     for (size_t jj = 0; jj < incr; jj++){ */
    /*         /\* printf("%zu,%G,%G\n",running+jj,true_params[running+jj],check_param[jj]); *\/ */
    /*         CuAssertDblEquals(tc,true_params[running+jj],check_param[jj],1e-20); */
    /*     } */
        
    /*     running+= incr; */
    /* } */
    /* free(check_param); check_param = NULL; */
    /* CuAssertDblEquals(tc,0.0,val,1e-20); */
    /* ft_param_update_params(ftp,param_space); */

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

void Test_LS_AIO_new_weighted(CuTest * tc)
{
    srand(seed);
    printf("\nLS_AIO_new: Testing AIO regression on a randomly generated low rank function with weighted first sample \n");
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
    double * w = calloc_double(ndata);
    
    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        // no noise!
        y[ii] = function_train_eval(a,x+ii*dim);
        /* y[ii] += randn(); */

        if (ii > 0){
            w[ii] = 1.0;
        }
        else{
            w[ii] = 1e10;
        }
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
    regress_opts_set_sample_weights(ropts, w);
    struct c3Opt * optimizer = c3opt_create(BFGS);
    

    struct FunctionTrain * ft = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);
    double diff = function_train_relnorm2diff(ft,a);
    printf("\n\t  Relative Error from low level interface = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-4);

    double eval0 = function_train_eval(ft, x);
    double diff0 = fabs(y[0] - eval0);
    double eval1 = function_train_eval(ft, x + dim);
    double diff1 = fabs(y[1] - eval1);
    printf("\t Error on heavily weighted sample = %3.5G\n", diff0);
    printf("\t Error on next sample = %3.5G\n", diff1);
    CuAssertDblEquals(tc, 0.0, diff0, 1e-9);
    CuAssertIntEquals(tc, 1, diff0 < diff1);

    struct FTRegress * reg = ft_regress_alloc(dim,fapp,ranks);
    ft_regress_set_alg_and_obj(reg,AIO,FTLS);
    ft_regress_update_params(reg,param_space);
    ft_regress_set_sample_weights(reg, w);
    struct FunctionTrain * ft2 = ft_regress_run(reg,optimizer,ndata,x,y);

    diff = function_train_relnorm2diff(ft2,a);
    printf("\t  Relative Error from higher level interface = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-4);

    eval0 = function_train_eval(ft, x);
    diff0 = fabs(y[0] - eval0);
    eval1 = function_train_eval(ft, x + dim);
    diff1 = fabs(y[1] - eval1);
    printf("\t Error on heavily weighted sample = %3.5G\n", diff0);
    printf("\t Error on next sample = %3.5G\n", diff1);
    CuAssertDblEquals(tc, 0.0, diff0, 1e-10);
    CuAssertIntEquals(tc, 1, diff0 < diff1);

    
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
    free(w); w = NULL;
    
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


static double funccv(double * x)
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
    size_t ndata = 300;
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
    int cvverbose = 2;
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
    size_t nparams = 10;
    double * centers = linspace(lb,ub,nparams);
    double scale = 1.0;
    double width = pow(nparams,-0.2)*2.0/12.0;
    width *= 10;
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

    struct c3Opt * optimizer = c3opt_create(SGD);
    c3opt_set_sgd_nsamples(optimizer,ndata);
    c3opt_set_maxiter(optimizer,500);

    
    /* struct c3Opt * optimizer = c3opt_create(BFGS); */
    /* c3opt_set_maxiter(optimizer,50); */

    c3opt_set_verbose(optimizer,0);
    c3opt_set_gtol(optimizer,1e-5);

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

    struct c3Opt * optimizer = c3opt_create(SGD);
    c3opt_set_sgd_nsamples(optimizer,ndata);
    c3opt_set_maxiter(optimizer,500);
    
    /* struct c3Opt * optimizer = c3opt_create(BFGS); */
    /* c3opt_ls_set_maxiter(optimizer,50); */
    
    c3opt_set_verbose(optimizer,0);
    c3opt_set_gtol(optimizer,1e-5);
    c3opt_set_relftol(optimizer,1e-20);
    c3opt_set_absxtol(optimizer,0);

    

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
    size_t ndata = 150;
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


void Test_kristoffel(CuTest * tc)
{
    srand(seed);
    printf("\nkristoffel: Testing kristoffel_code \n");

    size_t dim = 5;
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    /* size_t ranks[3] = {1,3,1}; */
    size_t ranks[6] = {1,3,2,4,2,1};
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);
        
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    struct OrthPolyExpansion * poly = orth_poly_expansion_init_from_opts(opts,maxorder+1);
    double x = 0.5;

    double kristoffel_weight = 0.0;
    for (size_t ii = 0; ii < maxorder+1; ii++){
        poly->coeff[ii] = 1.0;
        double v1 = orth_poly_expansion_eval(poly,x);
        kristoffel_weight += v1*v1;
        poly->coeff[ii] = 0.0;
    }
    kristoffel_weight = sqrt(kristoffel_weight / (double) (maxorder+1));

    
    double pt[5];    
    for (size_t ii = 0; ii < dim; ii++){
        pt[ii] = x;
    }

    function_train_activate_kristoffel(a);

    double ak = function_train_get_kristoffel_weight(a,pt);

    /* printf("ft weight = %G\n",ak); */
    /* printf("uni weight = %G, uni^5 =%G\n", kristoffel_weight,pow(kristoffel_weight,5)); */
    CuAssertDblEquals(tc,pow(kristoffel_weight,5),ak,1e-14);
    function_train_free(a); a = NULL;
    orth_poly_expansion_free(poly); poly = NULL;
    bounding_box_free(bds); bds = NULL;
    ope_opts_free(opts); opts = NULL;


}

void Test_LS_AIO_kristoffel(CuTest * tc)
{
    srand(seed);
    printf("\nLS_AIO_kristoffel: Testing AIO regression on a randomly generated low rank function with kristoffel weighting \n");
    printf("\t  Dimensions: 3\n");
    printf("\t  Ranks:      [1 1 2 1]\n");
    printf("\t  LPOLY order: 20\n");
    printf("\t  ndata:       1000\n");

    size_t dim = 3;
    double lb = -1.0;
    double ub = 1.0;
    /* size_t maxorder = 3; */
    size_t maxorder = 20;
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    /* size_t ranks[3] = {1,3,1}; */
    /* size_t ranks[6] = {1,1,1,1,1,1}; */
    size_t ranks[4] = {1,1,2,1};
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a =
        function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);
    
    // create data
    size_t ndata = 1000;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    double * xmc = calloc_double(ndata*dim);
    double * ymc = calloc_double(ndata*dim);
    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = cos(randu()*3.14159);
            xmc[ii*dim+jj] =  randu()*(ub-lb) + lb; 
        }
        // no noise!
        y[ii] = function_train_eval(a,x+ii*dim);
        ymc[ii] = function_train_eval(a,xmc+ii*dim);
        /* dprint(dim, x+ii*dim); */
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
        multi_approx_opts_set_dim(fapp,ii,qmopts);
    }
    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_verbose(optimizer,0);
    /* c3opt_set_relftol(optimizer,1e-20); */
    /* c3opt_set_absxtol(optimizer,1e-20); */
    /* c3opt_set_gtol(optimizer,1e-20); */
    
    struct FTRegress * reg2 = ft_regress_alloc(dim,fapp,ranks);
    ft_regress_set_alg_and_obj(reg2,AIO,FTLS);
    struct FunctionTrain * ft3 = ft_regress_run(reg2,optimizer,ndata,xmc,ymc);

    double diff2 = function_train_relnorm2diff(ft3,a);
    printf("\t  Relative Error for monte carlo sampling = %G\n",diff2);


    c3opt_set_verbose(optimizer,0);    
    struct FTRegress * reg = ft_regress_alloc(dim,fapp,ranks);
    ft_regress_set_alg_and_obj(reg,AIO,FTLS);
    ft_regress_set_kristoffel(reg,1);
    struct FunctionTrain * ft2 = ft_regress_run(reg,optimizer,ndata,x,y);

    double diff = function_train_relnorm2diff(ft2,a);
    printf("\t  Relative Error for kristoffel sampling = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff,1e-4);

    CuAssertIntEquals(tc,1,diff < diff2);

    
    ft_regress_free(reg);     reg = NULL;
    function_train_free(ft2); ft2 = NULL;

    ft_regress_free(reg2);    reg2 = NULL;
    function_train_free(ft3); ft3 = NULL;

    c3opt_free(optimizer); optimizer = NULL;
    
    bounding_box_free(bds);   bds  = NULL;
    function_train_free(a);   a    = NULL;

    free(x); x = NULL;
    free(y); y = NULL;

    free(xmc); xmc = NULL;
    free(ymc); ymc = NULL;

    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}


double fsgd2(const double * x)
{
    return  x[1]*x[0] + x[0] + x[1] + x[1]*x[0]*x[0] + x[1]*x[1]*x[1] + sin(x[4]*x[1]) + x[3]*cos(2.0*x[2]); /* + 1.0 / (x[2] + 1.1); */
}


void Test_LS_AIO3_sgd(CuTest * tc)
{
    srand(seed);
    printf("\nLS_AIO_new_sgd: Testing AIO regression on a somewhat complex function \n");
    printf("\t  Dimensions: 5\n");
    printf("\t  Ranks:      [1 8 8 9 7 1]\n");
    printf("\t  LPOLY order: 4\n");
    printf("\t  nunknowns:   1070\n");
    printf("\t  ndata:       1000\n");

    size_t dim = 5;
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 4;
    size_t ranks[6] = {1,8,8,9,7,1};


    // create data
    size_t ndata = 1000;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        y[ii] = fsgd2(x+ii*dim);
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
    /* printf("nunknowns = %zu\n",nunknowns); */
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
    }

    int sgd = 1; // SGD gets way better errors than BFGS after 100 iterations
    struct c3Opt * optimizer = NULL;
    struct FTRegress * reg = NULL;
    if (sgd == 1){
        
        optimizer = c3opt_create(SGD);
        /* c3opt_set_verbose(optimizer,1); */
        c3opt_set_sgd_nsamples(optimizer,ndata);
        c3opt_set_sgd_learn_rate(optimizer,1e-3);
        c3opt_set_absxtol(optimizer,1e-20);
        c3opt_set_maxiter(optimizer,100);
    
        reg = ft_regress_alloc(dim,fapp,ranks);
        ft_regress_set_alg_and_obj(reg,AIO,FTLS);
    }
    else{
        optimizer = c3opt_create(BFGS);
        c3opt_set_verbose(optimizer,1);
        /* c3opt_set_sgd_nsamples(optimizer,ndata); */
        /* c3opt_set_sgd_learn_rate(optimizer,5e-2); */
        c3opt_set_maxiter(optimizer,100);
    
        reg = ft_regress_alloc(dim,fapp,ranks);
        ft_regress_set_alg_and_obj(reg,AIO,FTLS);
    }
    
    struct FunctionTrain * ft2 = ft_regress_run(reg,optimizer,ndata,x,y);

    /* double * params = calloc_double(nunknowns); */
    /* function_train_get_params(ft2,params); */
    /* dprint(nunknowns,params); */
    /* free(params); params = NULL; */


    double test_error = 0;
    double pt[5];
    size_t N = 100000;
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < 5; jj++){
            pt[jj] = 2.0*randu()-1.0;
        }
        double v1 = function_train_eval(ft2,pt);
        double v2 = fsgd2(pt);
        test_error += (v1-v2)*(v1-v2);
    }
    test_error /= (double) N;
    printf("\t  rmse = %G\n",sqrt(test_error));
    printf("\t  mse = %G\n",test_error);
    CuAssertDblEquals(tc,0.0,test_error,1e-3);
    
    ft_regress_free(reg);     reg = NULL;
    function_train_free(ft2); ft2 = NULL;

    c3opt_free(optimizer); optimizer = NULL;

    free(x); x = NULL;
    free(y); y = NULL;

    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);

    
}

double fsgd(const double * x)
{
    return  x[1]*x[0] + x[0] + x[1] + x[1]*x[0]*x[0] + x[1]*x[1]*x[1];
}


void Test_LS_AIO_new_sgd(CuTest * tc)
{
    srand(seed);
    printf("\nLS_AIO_new_sgd: Testing AIO regression on a simple function\n");
    printf("\t  Dimensions: 2\n");
    printf("\t  Ranks:      [1 4 1]\n");
    printf("\t  LPOLY order: 3\n");
    printf("\t  nunknowns:   32\n");
    printf("\t  ndata:       100\n");

    size_t dim = 2;
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[3] = {1,4,1};
    /* size_t ranks[6] = {1,3,2,4,2,1}; */

    // create data
    size_t ndata = 100;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        y[ii] = fsgd(x+ii*dim);
    }
    /* printf("y = "); */
    /* dprint(ndata,y); */


    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    size_t nunknowns = 0;
    for (size_t ii = 0; ii < dim; ii++){ nunknowns += (maxorder+1)*ranks[ii]*ranks[ii+1];}
    
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
    }

    int sgd = 1;
    struct c3Opt * optimizer = NULL;
    struct FTRegress * reg = NULL;
    if (sgd == 1){
        
        optimizer = c3opt_create(SGD);
        /* c3opt_set_verbose(optimizer,1); */
        c3opt_set_sgd_nsamples(optimizer,ndata);
        c3opt_set_absxtol(optimizer,1e-30);
        c3opt_set_sgd_learn_rate(optimizer,1e-3);
        c3opt_set_maxiter(optimizer,2000);
    
        reg = ft_regress_alloc(dim,fapp,ranks);
        ft_regress_set_alg_and_obj(reg,AIO,FTLS);
    }
    else{
        optimizer = c3opt_create(BFGS);
        /* c3opt_set_verbose(optimizer,1); */
        /* c3opt_set_sgd_nsamples(optimizer,ndata); */
        /* c3opt_set_sgd_learn_rate(optimizer,5e-2); */
        c3opt_set_maxiter(optimizer,1000);
    
        reg = ft_regress_alloc(dim,fapp,ranks);
        ft_regress_set_alg_and_obj(reg,AIO,FTLS);
    }
    
    struct FunctionTrain * ft2 = ft_regress_run(reg,optimizer,ndata,x,y);

    /* double * params = calloc_double(nunknowns); */
    /* function_train_get_params(ft2,params); */
    /* dprint(nunknowns,params); */


    double test_error = 0;
    double pt[5];
    size_t N = 100000;
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < 5; jj++){
            pt[jj] = 2.0*randu()-1.0;
        }
        double v1 = function_train_eval(ft2,pt);
        double v2 = fsgd(pt);
        test_error += (v1-v2)*(v1-v2);
    }
    test_error /= (double) N;
    printf("\t  rmse = %G\n",sqrt(test_error));
    printf("\t  mse = %G\n",test_error);
    CuAssertDblEquals(tc,0.0,test_error,1e-6);
    
    ft_regress_free(reg);     reg = NULL;
    function_train_free(ft2); ft2 = NULL;

    c3opt_free(optimizer); optimizer = NULL;

    free(x); x = NULL;
    free(y); y = NULL;

    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
}


static double hess_fd_ij(struct FTparam * ftp, double * param, double * x,
                         double h,size_t ii, size_t jj)
{
    size_t nparam = ftp->nparams;
    double * y = calloc_double(nparam);
    memmove(y,param,nparam*sizeof(double));

    /* printf("here %zu\n",nparam); */
    
    y[ii] += h;
    y[jj] += h;
    ft_param_update_params(ftp,y);
    double v1 = function_train_eval(ftp->ft,x);
    /* printf("v1 = %G ",v1); */
    y[ii] -= h;
    y[jj] -= h;


    y[ii] += h;
    y[jj] -= h;
    ft_param_update_params(ftp,y);
    double v2 = function_train_eval(ftp->ft,x);
    /* printf("v2 = %G ",v2); */
    y[ii] -= h;
    y[jj] += h;

    y[ii] -= h;
    y[jj] += h;
    ft_param_update_params(ftp,y);
    double v3 = function_train_eval(ftp->ft,x);
    /* printf("v3 = %G ",v3); */
    y[ii] += h;
    y[jj] -= h;


    y[ii] -= h;
    y[jj] -= h;
    ft_param_update_params(ftp,y);
    double v4 = function_train_eval(ftp->ft,x);
    y[ii] += h;
    y[jj] += h;

    double num = v1 - v2 - v3 + v4;
    double den = 4.0 * h * h;

    double res = num / den;
    ft_param_update_params(ftp,param);
    free(y); y = NULL;
    return res;

}

void Test_ft_param_hess_vec(CuTest * tc)
{
    srand(seed);
    printf("\nft_param_hess_vec: Testing ft_param_hess_vec (1)\n");
    printf("\t  Dimensions: 2\n");
    printf("\t  Ranks:      [1 2 1]\n");
    printf("\t  LPOLY order: 3\n");

    size_t dim = 2;
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    size_t ranks[3] = {1,2,1};

    // create data
    size_t ndata = 1;
    double * x = calloc_double(ndata*dim);
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
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
    
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
    }

    /* printf("nunknowns = %zu\n",nunknowns); */
    double param[16];
    
    // first core / first function
    param[0] = 1.0;
    param[1] = 2.0;
    param[2] = 3.0;
    param[3] = 4.0;

    // first core / second function
    param[4] = 1.5;
    param[5] = 2.5;
    param[6] = 3.5;
    param[7] = 4.5;

    // second core / first function
    param[8] = -0.3;
    param[9] = -0.6;
    param[10] = -0.9;
    param[11] = -1.2;

    // second core / second function
    param[12] = 0.50;
    param[13] = 0.75;
    param[14] = 1.00;
    param[15] = 1.25;

    struct FTparam* ftp = ft_param_alloc(dim,fapp,param,ranks);
    /* printf("nunknowns = %zu\n",ftp->nparams); */
    /* iprint_sz(4,ftp->nparams_per_uni); */

    double vec[16] = {1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                      0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

    double vec_out[16] = {1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                          0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    

    double h = 1e-5;
    for (size_t ii = 0; ii < 16; ii++){
        vec[ii] = 1.0;
        /* printf("what!\n"); */
        ft_param_hessvec(ftp,x,vec,vec_out);
        /* printf("ok!\n"); */
        for (size_t jj = 0; jj < 16; jj++){

            /* printf("%3.5G ",vec_out[jj]); */
            double should = hess_fd_ij(ftp,param,x,h,ii,jj);
            /* printf("%3.5G %3.5G\n",vec_out[jj],should); */
            double err = should - vec_out[jj];
            CuAssertDblEquals(tc,0.0,err,1e-5);
        }
        /* printf("\n"); */
        vec[ii] = 0.0;
    }

    /* printf("\n\n"); */
    /* for (size_t ii = 0; ii < 16; ii++){ */
    /*     for (size_t jj = 0; jj < 16; jj++){ */
    /*         double should = hess_fd_ij(ftp,param,x,h,ii,jj); */
    /*         printf("%3.5G ",should); */
    /*     } */
    /*     printf("\n"); */
    /* } */
    /* dprint(16,vec_out); */

    
    free(x); x = NULL;

    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    ft_param_free(ftp);
}


void Test_ft_param_hess_vec2(CuTest * tc)
{
    srand(seed);
    printf("\nft_param_hess_vec: Testing ft_param_hess_vec (2)\n");
    printf("\t  Dimensions: 6\n");
    printf("\t  Ranks:      [1 3 4 2 8 5 1]\n");
    printf("\t  LPOLY order: 3\n");

    size_t dim = 6;
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 3;
    size_t ranks[7] = {1,3,4,2,8,5,1};

    // create data
    size_t ndata = 1;
    double * x = calloc_double(ndata*dim);
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
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
    
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
    }

    printf("\t  nunknowns = %zu\n",nunknowns);
    double * param = calloc_double(nunknowns);
    for (size_t ii = 0; ii < nunknowns; ii++){
        param[ii] = randn();
    }

    struct FTparam* ftp = ft_param_alloc(dim,fapp,param,ranks);
    /* printf("nunknowns = %zu\n",ftp->nparams); */
    /* iprint_sz(4,ftp->nparams_per_uni); */

    double * vec = calloc_double(nunknowns);
    double * vec_out = calloc_double(nunknowns);

    double h = 1e-4;
    for (size_t ii = 0; ii < nunknowns; ii++){
        vec[ii] = 1.0;
        /* printf("what!\n"); */
        ft_param_hessvec(ftp,x,vec,vec_out);
        /* printf("ok!\n"); */
        for (size_t jj = 0; jj < nunknowns; jj++){

            double should = hess_fd_ij(ftp,param,x,h,ii,jj);
            /* printf("%3.5G ",vec_out[jj]); */

            double err = (should - vec_out[jj]);
            /* printf("%3.5G %3.5G %3.5G\n",vec_out[jj],should,err); */
            CuAssertDblEquals(tc,0.0,err,1e-3);
        }
        /* printf("\n"); */
        vec[ii] = 0.0;
    }

    free(vec); vec = NULL;
    free(vec_out); vec_out = NULL;
    free(x); x = NULL;
    free(param); param = NULL;
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    ft_param_free(ftp);
}





CuSuite * CLinalgRegressGetSuite()
{
    CuSuite * suite = CuSuiteNew();

    /* next 3 are good */
    SUITE_ADD_TEST(suite, Test_LS_ALS);
    SUITE_ADD_TEST(suite, Test_LS_ALS2);
    SUITE_ADD_TEST(suite, Test_LS_ALS_SPARSE2);

    /* next 5 are good */
    SUITE_ADD_TEST(suite, Test_ft_param_core_gradeval);
    SUITE_ADD_TEST(suite, Test_ft_param_gradeval);
    SUITE_ADD_TEST(suite, Test_ft_param_eval_lin);
    SUITE_ADD_TEST(suite, Test_ft_param_gradeval_lin);
    SUITE_ADD_TEST(suite, Test_LS_AIO);
    SUITE_ADD_TEST(suite, Test_LS_AIO2);
    SUITE_ADD_TEST(suite, Test_LS_AIO3);


    /* next 4 are good */
    SUITE_ADD_TEST(suite, Test_LS_AIO_new);
    SUITE_ADD_TEST(suite, Test_LS_AIO_new_weighted);
    SUITE_ADD_TEST(suite, Test_LS_AIO_ftparam_create_from_lin_ls);
    SUITE_ADD_TEST(suite, Test_LS_AIO_ftparam_create_from_lin_ls_kernel);
    /* SUITE_ADD_TEST(suite, Test_LS_cross_validation); */
    

    /* Next 2 are good */
    SUITE_ADD_TEST(suite, Test_function_train_param_grad_sqnorm);
    SUITE_ADD_TEST(suite, Test_SPARSELS_AIO);

    /* next 2 are good */
    SUITE_ADD_TEST(suite, Test_SPARSELS_AIOCV);
    SUITE_ADD_TEST(suite, Test_SPARSELS_cross_validation);

    /* /\* Next 3 are good *\/ */
    SUITE_ADD_TEST(suite, Test_LS_AIO_kernel);
    SUITE_ADD_TEST(suite, Test_LS_AIO_kernel_nonlin);
    
    SUITE_ADD_TEST(suite, Test_LS_AIO_rounding);
    SUITE_ADD_TEST(suite, Test_LS_AIO_rankadapt);
    SUITE_ADD_TEST(suite, Test_LS_AIO_rankadapt_kernel);
    SUITE_ADD_TEST(suite, Test_LS_AIO_kernel2);
    SUITE_ADD_TEST(suite, Test_LS_AIO_kernel3);

    SUITE_ADD_TEST(suite, Test_kristoffel);
    SUITE_ADD_TEST(suite, Test_LS_AIO_kristoffel);

    SUITE_ADD_TEST(suite, Test_LS_AIO3_sgd);
    SUITE_ADD_TEST(suite, Test_LS_AIO_new_sgd);

    SUITE_ADD_TEST(suite,Test_ft_param_hess_vec);
    SUITE_ADD_TEST(suite,Test_ft_param_hess_vec2);

    return suite;
}


