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

void Test_c3opt_lbfgs_storage(CuTest * tc)
{
    printf("Testing Function: storage functions surrounding lbfgs\n");

    double x[2] = {0.5, 0.3};
    double xn[2] = {1.0, 0.3};
    double g[2] = {0.4, 0.2};
    double gn[2] = {2.0, 3.0};
    size_t m = 5;

    struct c3opt_lbfgs_list * list = c3opt_lbfgs_list_alloc(m,2);

    double rhoinv[10];
    for (size_t ii = 0; ii < 10; ii++){
        c3opt_lbfgs_list_insert(list,ii,xn,x,gn,g);
        double rhoshould = 0.0;
        for (size_t jj = 0; jj < 2; jj++){
            rhoshould += (xn[jj]-x[jj])* (gn[jj]-g[jj]);
        }
        rhoinv[ii] = rhoshould;
        /* printf("rhoinvshould = %G\n", rhoshould); */

        x[0] = xn[0]; x[1] = xn[1];
        g[0] = gn[0]; g[1] = gn[1];
        xn[0] = x[0]+0.4*randu()*3; xn[1] = x[1]+0.3*randu()*3;
        gn[0] = g[0]-0.2*randu()*3; gn[1] = g[1]+0.7*randu()*3;
    }

    size_t iter;
    double s[2];
    double y[2];
    double rhoi;

    for (size_t jj = 0; jj < 10; jj++)
    {
        c3opt_lbfgs_list_step(list,&iter,s,y,&rhoi);
        CuAssertIntEquals(tc,9-jj%m,iter);
        /* printf("jj=%zu, iter=%zu, check=%zu\n",jj,iter,9-jj%m); */
    }

    // NOTE THIS WORKS BECAUSE PREVIOUSLY STEPPED 2M
    for (size_t jj = 0; jj < 10; jj++)
    {
        c3opt_lbfgs_list_step_back(list,&iter,s,y,&rhoi);
        CuAssertIntEquals(tc,jj%m+m,iter);
        /* printf("jj=%zu, iter=%zu, check=%zu\n",jj,iter,jj%m+m); */
    }

    for (size_t jj = 9; jj > 9-m; jj--)
    {
        c3opt_lbfgs_list_step(list,&iter,s,y,&rhoi);
        /* printf("rho=%G, rhoi=%G\n",rhoinv[jj],rhoi); */
        CuAssertDblEquals(tc,rhoinv[jj],rhoi,1e-15);
    }
    
    /* printf("\n\n\n"); */
    /* c3opt_lbfgs_list_print(list,stdout,3,5); */
    /* printf("\n\n\n"); */
    c3opt_lbfgs_list_free(list); list = NULL;
}

CuSuite * LBFGSGetSuite(){
    //printf("----------------------------\n");

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_c3opt_lbfgs_storage);
    return suite;
}
