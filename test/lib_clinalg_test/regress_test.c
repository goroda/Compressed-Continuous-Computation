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

void Test_LS_ALS_grad(CuTest * tc)
{
    printf("Testing Function: regress_core_LS \n");

    size_t dim = 4;    
    size_t core = 1;
    
    size_t ranks[5] = {1,2,2,2,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 100;
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

    /* printf("data outside = \n"); */
    /* dprint2d_col(dim,ndata,x); */
    
    struct RegressALS * als = regress_als_alloc(dim);
    regress_als_add_data(als,ndata,x,y);
    regress_als_prep_memory(als,a);
    regress_als_set_core(als,core);

    size_t totparam = 0;
    size_t maxparam = 0;
    function_train_core_get_nparams(a,core,&totparam,&maxparam);
    CuAssertIntEquals(tc,(maxorder+1)*2*2,totparam);
    CuAssertIntEquals(tc,maxorder+1,maxparam);
    double * guess = calloc_double(totparam);

    double space1[1000];
    double space2[1000];
    double grad[100];
    double pre[100];
    double cur[100];
    double post[100];
    double core_eval[100];
    double core_eval2[100];
    double qm_grad[1000];
    double qm_eval[1000];

    double h = 1e-8;
    for (size_t ii = 0; ii < ndata; ii++){
        for (size_t zz = 0; zz < totparam; zz++){
            guess[zz] = randn();
        }
    	/* printf("guess = \n"); dprint(totparam,guess); */
        function_train_update_core_params(a,core,totparam,guess);
    	/* print_qmarray(a->cores[core],0,NULL); */
        CuAssertIntEquals(tc,dim,a->dim);
        for (size_t zz = 1; zz < dim; zz++){
            CuAssertIntEquals(tc,2,a->ranks[zz]);
        }

        double val = function_train_core_param_grad_eval(a,x+ii*dim,core,totparam,space1,space2,
                                                         grad,pre,cur,post);
        double val2 = function_train_eval(a,x+ii*dim);
        CuAssertDblEquals(tc,val2,val,1e-13);
        CuAssertIntEquals(tc,4,a->dim);

        qmarray_eval(a->cores[core],x[ii*dim+core],core_eval);
        CuAssertIntEquals(tc,4,a->dim);

        qmarray_param_grad_eval(a->cores[core],x[ii*dim+core],qm_eval,qm_grad,space2);
        CuAssertIntEquals(tc,4,a->dim);

    	double * param_grad = calloc_double(maxorder+1);
    	/* printf("x = %G\n",x[ii*dim+core]); */
    	size_t onparam = 0;
        for (size_t zz = 0; zz < 4; zz++){
            /* print_generic_function(a->cores[core]->funcs[zz],0,NULL); */
            generic_function_param_grad_eval(a->cores[core]->funcs[zz],1,x+ii*dim+core,param_grad);
            CuAssertDblEquals(tc,core_eval[zz],cur[zz],1e-14);
            CuAssertDblEquals(tc,core_eval[zz],qm_eval[zz],1e-14);
            /* dprint(maxorder+1,param_grad); */
	  
            for (size_t qq = 0; qq < maxorder+1; qq++){
                /* printf("pred grad with respect to first param\n"); */
                /* dprint2d_col(2,2,qm_grad + onparam*4); */
                CuAssertDblEquals(tc,qm_grad[onparam*4+zz],param_grad[qq],1e-10);
                onparam++;
            }
        }
        CuAssertIntEquals(tc,4,a->dim);
    	free(param_grad); param_grad = NULL;

        // check derivatives
        for (size_t jj = 0; jj < totparam; jj++){
    	    /* printf("jj=%zu\n",jj); */
            guess[jj] = guess[jj]-h;
            function_train_update_core_params(a,core,totparam,guess);
            qmarray_eval(a->cores[core],x[ii*dim+core],core_eval2);

    	    /* printf("grad think = \n"); */
    	    /* dprint2d_col(2,2,qm_grad + jj*4); */
            // test derivative of the core
    	    double fd_diff[4];
            for (size_t zz = 0; zz < 4; zz++){
                double v1 = core_eval[zz];
                double v2 = core_eval2[zz];
                fd_diff[zz] = (v1-v2)/h;
                CuAssertDblEquals(tc,fd_diff[zz],space1[jj*4+zz],1e-5);
                CuAssertDblEquals(tc,fd_diff[zz],qm_grad[jj*4+zz],1e-5);
            }
    	    /* printf("FD is = \n"); */
    	    /* dprint2d_col(2,2,fd_diff); */

            // test derivative of the function evaluation
            double val3 = function_train_eval(a,x+ii*dim);
            double fv_diff = (val2 - val3)/h;
            CuAssertDblEquals(tc,fv_diff,grad[jj],1e-5);

            guess[jj] = guess[jj]+h;
            function_train_update_core_params(a,core,totparam,guess);
        }
    }
    /* printf("Great!\n");     */
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,totparam);
    c3opt_set_verbose(optimizer,0);
    c3opt_add_objective(optimizer,regress_core_LS,als);
    for (size_t zz = 0; zz < totparam; zz++){
        guess[zz] = 1.0;
    }

    // check derivative
    double * deriv_diff = calloc_double(totparam);
    double gerr = c3opt_check_deriv_each(optimizer,guess,1e-8,deriv_diff);
    for (size_t ii = 0; ii < totparam; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3);
    }
    /* printf("gerr = %G\n",gerr); */
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

    double val;
    int res = c3opt_minimize(optimizer,guess,&val);
    CuAssertIntEquals(tc,1,res>-1);

    // minimum should be zero because there is no noise 
    // in the data;
    CuAssertDblEquals(tc,0.0,val,1e-10);
    
    bounding_box_free(bds); bds       = NULL;
    function_train_free(a); a         = NULL;
    c3opt_free(optimizer);  optimizer = NULL;
    regress_als_free(als);  als       = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    free(guess); guess = NULL;
}

CuSuite * CLinalgRegressGetSuite()
{
    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_LS_ALS_grad);
    return suite;
}
