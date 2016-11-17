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
    size_t ndata = 5000;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
        }
        y[ii] = function_train_eval(a,x+ii*dim);
        y[ii] += randn()*0.1;
    }


    
    struct RegressALS * als = regress_als_alloc(dim);
    regress_als_add_data(als,ndata,x,y);
    regress_als_prep_memory(als,a);
    regress_als_set_core(als,core);


    size_t totparam = 0;
    size_t maxparam = 0;
    function_train_core_get_nparams(a,core,&totparam,&maxparam);
    CuAssertIntEquals(tc,(maxorder+1)*dim,totparam);
    CuAssertIntEquals(tc,maxorder+1,maxparam);

    printf("Great!\n");
    double * guess = calloc_double((maxorder+1)*dim);
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,totparam);
    c3opt_set_verbose(optimizer,0);
    c3opt_add_objective(optimizer,regress_core_LS,als);

    // check derivative
    double * deriv_diff = calloc_double(totparam);
    double gerr = c3opt_check_deriv_each(optimizer,guess,1e-8,deriv_diff);
    for (size_t ii = 0; ii < totparam; ii++){
        printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]);
        CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3);
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;

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
