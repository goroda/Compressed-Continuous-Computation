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

#include "CuTest.h"
#include "array.h"
#include "tensortrain.h"
#include "tt_multilinalg.h"
#include "tt_integrate.h"
#include "candecomp.h"
#include "convert_decomp.h"
#include "cross.h"

void Test_ttscale(CuTest * tc)
{
    printf("Testing function: cd2tt and ttscal\n");

    size_t elem[3];
    size_t ii,jj,kk;
    size_t N = 10;
    double * x = linspace(0.0,1.0,N);
    double * y = linspace(0.0,1.0,N);
    double * z = linspace(0.0,1.0,N);

    struct mat **cores = malloc(sizeof(struct mat)*3);
    cores[0] = v2m(N,x,0);
    cores[1] = v2m(N,y,0);
    cores[2] = v2m(N,z,0);
     
    struct candecomp cd;
    cd.dim = 3;
    cd.rank = 1;
    cd.cores = cores;
    
    struct tt * t = cd2tt(&cd);
    CuAssertIntEquals(tc, 3, t->dim);
    CuAssertIntEquals(tc, 10, t->nvals[0]);
    CuAssertIntEquals(tc, 10, t->nvals[1]);
    CuAssertIntEquals(tc, 10, t->nvals[2]);
    CuAssertIntEquals(tc, 1, t->ranks[0]);
    CuAssertIntEquals(tc, 1, t->ranks[1]);
    CuAssertIntEquals(tc, 1, t->ranks[2]);
    CuAssertIntEquals(tc, 1, t->ranks[3]);
    
    for (ii = 0; ii < 10; ii++){
        elem[0] = ii;
        for (jj = 0; jj < 10; jj++){
            elem[1] = jj;
            for (kk =0; kk < 10; kk++){
                elem[2] = kk;
                CuAssertDblEquals(tc, x[ii]*y[jj]*z[kk], ttelem(t,elem), 1e-14);
            }
        }
    }

    double scale = -2.0;
    ttscal(t,scale);
    for (ii = 0; ii < 10; ii++){
        elem[0] = ii;
        for (jj = 0; jj < 10; jj++){
            elem[1] = jj;
            for (kk =0; kk < 10; kk++){
                elem[2] = kk;
                CuAssertDblEquals(tc, scale*x[ii]*y[jj]*z[kk], ttelem(t,elem), 1e-14);
            }
        }
    }


    free(x);
    free(y);
    free(z);
    freemat(cores[0]);
    freemat(cores[1]);
    freemat(cores[2]);
    free(cores);

    freett(t);
}

void Test_ttadd(CuTest * tc)
{
    printf("Testing function: ttadd\n");
    // assume cd2tt works
    
    size_t elem[3];
    size_t ii,jj,kk;
    size_t N = 10;
    double * x = linspace(0.0,1.0,N);
    double * y = linspace(0.0,1.0,N);
    double * z = linspace(0.0,1.0,N);

    struct mat **cores = malloc(sizeof(struct mat)*3);
    cores[0] = v2m(N,x,0);
    cores[1] = v2m(N,y,0);
    cores[2] = v2m(N,z,0);
     
    struct candecomp cd;
    cd.dim = 3;
    cd.rank = 1;
    cd.cores = cores;
    
    struct tt * t = cd2tt(&cd);
    CuAssertIntEquals(tc, 3, t->dim);
    CuAssertIntEquals(tc, 10, t->nvals[0]);
    CuAssertIntEquals(tc, 10, t->nvals[1]);
    CuAssertIntEquals(tc, 10, t->nvals[2]);
    CuAssertIntEquals(tc, 1, t->ranks[0]);
    CuAssertIntEquals(tc, 1, t->ranks[1]);
    CuAssertIntEquals(tc, 1, t->ranks[2]);
    CuAssertIntEquals(tc, 1, t->ranks[3]);

    //int shape[3];
    //shape[0] = N;
    //shape[1] = N;
    //shape[2] = N;
   // struct tt * ones = tt_ones(3, shape);
    struct tt * ones = cd2tt(&cd);
    ttscal(ones,-1.0);
    struct tt * ttuse = ttadd(t, ones);

    for (ii = 0; ii < 10; ii++){
        elem[0] = ii;
        for (jj = 0; jj < 10; jj++){
            elem[1] = jj;
            for (kk =0; kk < 10; kk++){
                elem[2] = kk;
                CuAssertDblEquals(tc, 0.0, ttelem(ttuse,elem), 1e-14);
            }
        }
    }
    
   // printf("here!\n");
    double norm = tt_norm(ttuse);
   // printf("there!\n");
    CuAssertDblEquals(tc, 0.0, norm, 1e-14);

    free(x);
    free(y);
    free(z);
    freemat(cores[0]);
    freemat(cores[1]);
    freemat(cores[2]);
    free(cores);

    freett(ones);
    freett(ttuse);
    freett(t);
}

void Test_tt_dot(CuTest * tc)
{
    printf("Testing function: tt_dot 2d\n");
    // assume cd2tt works
    
    size_t N = 10;
    size_t nvals[2] = {N,N};
    double * x = linspace(0.0,1.0,N);
    double * y = linspace(0.0,1.0,N);
    
    struct tt * a = tt_x(x, 0, 2, nvals);
    struct tt * b = tt_x(y, 1, 2, nvals);
     
    size_t ii, jj;
    double dot_s = 0.0;
    size_t elem[2];
    for (ii = 0; ii < N; ii++){
        elem[0] = ii;
        for (jj = 0; jj < N; jj++){
            elem[1] = jj;
            dot_s += ttelem(a,elem) * ttelem(b,elem);
        }
    }
    double dot = tt_dot(a,b);
    
    CuAssertDblEquals(tc, dot_s, dot, 1e-14);

    freett(a);
    freett(b);
    free(x);
    free(y);
}

void Test_tt_norm(CuTest * tc)
{
    printf("Testing function: tt_norm and tt_copy\n");
    // assume cd2tt works
    
    size_t ii,jj,kk;
    size_t N = 10;
    double * x = linspace(0.0,1.0,N);
    double * y = linspace(0.0,1.0,N);
    double * z = linspace(0.0,1.0,N);

    struct mat **cores = malloc(sizeof(struct mat)*3);
    cores[0] = v2m(N,x,0);
    cores[1] = v2m(N,y,0);
    cores[2] = v2m(N,z,0);
     
    struct candecomp cd;
    cd.dim = 3;
    cd.rank = 1;
    cd.cores = cores;
    
    struct tt * t = cd2tt(&cd);
    CuAssertIntEquals(tc, 3, t->dim);
    CuAssertIntEquals(tc, 10, t->nvals[0]);
    CuAssertIntEquals(tc, 10, t->nvals[1]);
    CuAssertIntEquals(tc, 10, t->nvals[2]);
    CuAssertIntEquals(tc, 1, t->ranks[0]);
    CuAssertIntEquals(tc, 1, t->ranks[1]);
    CuAssertIntEquals(tc, 1, t->ranks[2]);
    CuAssertIntEquals(tc, 1, t->ranks[3]);

    double norm_is = tt_norm(t);
    double norm_s = 0.0;
    for (ii = 0; ii < 10; ii++){
        for (jj = 0; jj < 10; jj++){
            for (kk =0; kk < 10; kk++){
                norm_s += pow(x[ii] * y[jj] * z[kk], 2);
            }
        }
    }

    norm_s = sqrt(norm_s);
    CuAssertDblEquals(tc, norm_s, norm_is,1e-14);
    struct tt * cp = copy_tt(t);

    CuAssertIntEquals(tc, 3, cp->dim);
    CuAssertIntEquals(tc, 10, cp->nvals[0]);
    CuAssertIntEquals(tc, 10, cp->nvals[1]);
    CuAssertIntEquals(tc, 10, cp->nvals[2]);
    CuAssertIntEquals(tc, 1, cp->ranks[0]);
    CuAssertIntEquals(tc, 1, cp->ranks[1]);
    CuAssertIntEquals(tc, 1, cp->ranks[2]);
    CuAssertIntEquals(tc, 1, cp->ranks[3]);


    double norm_cp = tt_norm(cp);
    CuAssertDblEquals(tc, norm_s, norm_cp,1e-14);

    free(x);
    free(y);
    free(z);
    freemat(cores[0]);
    freemat(cores[1]);
    freemat(cores[2]);
    free(cores);
    
    freett(cp);
    freett(t);
}

void Test_full_to_tt(CuTest * tc)
{
    printf("Testing function: full_to_tt and tt_round \n");

    size_t ii,jj,kk;
    size_t dim = 3;
    size_t nvals[3];
    nvals[0] = 5; nvals[1]=4; nvals[2]=11;
    struct tensor * t;
    init_tensor(&t, dim, nvals);
    for (ii = 0; ii < nvals[0]*nvals[1]*nvals[2]; ii++){
        t->vals[ii] = randu();
    }
        
    double epsilon = 1e-15;
    struct tt * a = full_to_tt(t, epsilon);
    CuAssertIntEquals(tc, 3, a->dim);
    CuAssertIntEquals(tc, nvals[0], a->nvals[0]);
    CuAssertIntEquals(tc, nvals[1], a->nvals[1]);
    CuAssertIntEquals(tc, nvals[2], a->nvals[2]);

    double t_elem;
    double tta_elem;
    size_t elem[3];
    elem[0] = 2;
    elem[1] = 1;
    elem[2] = 0;
    double err = 0.0;
    for (ii = 0; ii < nvals[0]; ii++){
        for (jj = 0; jj < nvals[1]; jj++){
            for (kk = 0; kk < nvals[2]; kk++){
                t_elem = t->vals[ii + jj * t->nvals[0] + 
                    kk * t->nvals[1] * t->nvals[0]]; // column major
                elem[0] = ii; elem[1] = jj; elem[2] = kk;
                tta_elem = ttelem(a, elem);
                err += pow(tta_elem-t_elem,2);
                //printf("diff = %3.2f : ", tta_elem-t_elem); iprint(3,elem);
            }
        }
    }
    err = sqrt(err);
    CuAssertDblEquals(tc, 0.0, err,1e-10);

    struct tt * rounded = tt_round(a, epsilon);
    err = 0.0;
    for (ii = 0; ii < nvals[0]; ii++){
        for (jj = 0; jj < nvals[1]; jj++){
            for (kk = 0; kk < nvals[2]; kk++){
                t_elem = t->vals[ii + jj * t->nvals[0] + 
                    kk * t->nvals[1] * t->nvals[0]]; // column major
                elem[0] = ii; elem[1] = jj; elem[2] = kk;
                tta_elem = ttelem(rounded, elem);
                err += pow(tta_elem-t_elem,2);
            }
        }
    }
    freett(rounded);
    err = sqrt(err);
    CuAssertDblEquals(tc, 0.0, err,1e-10);

    free_tensor(&t);
    freett(a);
}

CuSuite * TTGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_ttscale);
    SUITE_ADD_TEST(suite, Test_ttadd);
    SUITE_ADD_TEST(suite, Test_tt_dot);
    SUITE_ADD_TEST(suite, Test_tt_norm);
    SUITE_ADD_TEST(suite, Test_full_to_tt);
    return suite;
}

void Test_tensor_elem_wrap(CuTest * tc)
{
    printf("Testing Functions: tensor_elem and wrap_full_tensor_for_cross \n");
    // 2d function x + y => ranks = 2;
    
    size_t ii,jj;
    double x[11] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    double y[6] = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };
    double vals[66];
    for (ii = 0; ii < 6; ii++){
        for (jj = 0; jj < 11; jj++){
            vals[jj + ii * 11] = y[ii] + x[jj];
        }
    }
    size_t nvals[2] = {11, 6};
   // size_t ranks[3] = {1, 2, 1};
    struct tensor t;
    t.nvals = nvals;
    t.vals = vals;
    t.dim = 2;

    size_t elem[2];
    double outy[6];
    for (ii = 0; ii < t.nvals[0]; ii++){
        elem[0] = ii; //elem[1] = jj;
        wrap_full_tensor_for_cross(nvals[1], elem, 1, outy, (void *)(&t));
        for (jj = 0; jj < t.nvals[1]; jj++){
            elem[1] = jj;
            CuAssertDblEquals(tc, t.vals[jj * nvals[0] + ii], tensor_elem(&t,elem), 1e-14);
            CuAssertDblEquals(tc, t.vals[jj * nvals[0] + ii], outy[jj], 1e-14);
        }
    }
}

void Test_tt_cross2d(CuTest * tc)
{
    printf("Testing Functions: tt_cross for 2d tensor x + y\n");
    size_t ii,jj;
    double x[11] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    double y[6] = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };
    double vals[66];
    for (ii = 0; ii < 6; ii++){
        for (jj = 0; jj < 11; jj++){
            vals[jj + ii * 11] = y[ii] + x[jj];
        }
    }
    size_t nvals[2] = {11, 6};
    size_t ranks[3] = {1, 3, 1};
    struct tensor t;
    t.nvals = nvals;
    t.vals = vals;
    t.dim = 2;
    
    struct tt_cross_opts  * opts = init_cross_opts(2, ranks, t.nvals);
    opts->epsilon = 1e-5;

    for (ii = 0; ii < opts->dim-1; ii++){
        create_naive_index_set(opts->right[ii], t.nvals + ii);
        create_naive_index_set(opts->left[ii], t.nvals); // this might be wrong
    }

    struct tt * ttb = tt_cross(&wrap_full_tensor_for_cross, opts, (void *) (&t));
    CuAssertIntEquals(tc, 2, ttb->dim);
    CuAssertIntEquals(tc, ranks[0], ttb->ranks[0]);
    CuAssertIntEquals(tc, ranks[1], ttb->ranks[1]);
    CuAssertIntEquals(tc, ranks[2], ttb->ranks[2]);
    CuAssertIntEquals(tc, nvals[0], ttb->nvals[0]);
    CuAssertIntEquals(tc, nvals[1], ttb->nvals[1]);

    size_t elem[2];
    double tt_elem;
    double tens_elem;
    for (ii = 0; ii < t.nvals[0]; ii++){
        elem[0] = ii;
        for (jj = 0; jj < t.nvals[1]; jj++){
            elem[1] = jj;
            tens_elem = tensor_elem(&t,elem);
            tt_elem = ttelem(ttb,elem);
            CuAssertDblEquals(tc, tens_elem, tt_elem, 1e-14);
        }
    }

    free_cross_opts(&opts);
    freett(ttb);
}

void Test_tt_cross3d(CuTest * tc)
{
    printf("Testing Functions: tt_cross for 3d tensor x + y + z\n");
    size_t ii,jj,kk;
    double x[11] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    double y[6] = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };
    double z[5] = {0.0, 0.3, 0.6, 0.9, 1.2 };
    double vals[66*5];
    for (ii = 0; ii < 6; ii++){ //y
        for (jj = 0; jj < 11; jj++){ // x
            for (kk = 0; kk < 5; kk++){ // z
                vals[jj + ii * 11 + kk * 66] = y[ii] + x[jj] + z[kk];
            }
        }
    }
    size_t nvals[3] = {11,6,5};
    size_t ranks[4] = {1,3,3,1};
    struct tensor t;
    t.nvals = nvals;
    t.vals = vals;
    t.dim = 3;
    
    struct tt_cross_opts  * opts = init_cross_opts(3, ranks, t.nvals);

    opts->epsilon = 1e-5;

    for (ii = 0; ii < opts->dim-1; ii++){
        create_naive_index_set(opts->right[ii], t.nvals + ii);
        create_naive_index_set(opts->left[ii], t.nvals);
    }
    struct tt * ttb = tt_cross(&wrap_full_tensor_for_cross, opts, (void *) (&t));

    CuAssertIntEquals(tc, 3, ttb->dim);
    CuAssertIntEquals(tc, ranks[0], ttb->ranks[0]);
    CuAssertIntEquals(tc, ranks[1], ttb->ranks[1]);
    CuAssertIntEquals(tc, ranks[2], ttb->ranks[2]);
    CuAssertIntEquals(tc, ranks[3], ttb->ranks[3]);
    CuAssertIntEquals(tc, nvals[0], ttb->nvals[0]);
    CuAssertIntEquals(tc, nvals[1], ttb->nvals[1]);
    CuAssertIntEquals(tc, nvals[2], ttb->nvals[2]);
    size_t elem[3];
    double tt_elem;
    double tens_elem;
    for (ii = 0; ii < t.nvals[0]; ii++){
        elem[0] = ii; 
        for (jj = 0; jj < t.nvals[1]; jj++){
            elem[1] = jj;
            for (kk = 0; kk < t.nvals[2]; kk++){
                elem[2] = kk;
                tens_elem = tensor_elem(&t,elem);
                tt_elem = ttelem(ttb,elem);
                CuAssertDblEquals(tc, tens_elem, tt_elem, 1e-10);
            }
        }
    }
    
    free_cross_opts(&opts);
    freett(ttb);
}

void Test_tt_cross_adapt(CuTest * tc)
{
    printf("Testing Functions: tt_cross_adapt for 3d tensor x + y + z starting too low ranks\n");
    size_t ii,jj,kk;
    double x[11] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    double y[6] = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };
    double z[5] = {0.0, 0.3, 0.6, 0.9, 1.2 };
    double vals[66*5];
    for (ii = 0; ii < 6; ii++){ //y
        for (jj = 0; jj < 11; jj++){ // x
            for (kk = 0; kk < 5; kk++){ // z
                vals[jj + ii * 11 + kk * 66] = y[ii] + x[jj] + z[kk];
            }
        }
    }
    size_t nvals[3] = {11,6,5};
    size_t ranks[4] = {1,1,1,1};
    struct tensor t;
    t.nvals = nvals;
    t.vals = vals;
    t.dim = 3;
    
    struct tt_cross_opts  * opts = init_cross_opts_with_naive_set(3, ranks, t.nvals);
    opts->epsilon = 1e-5;
    opts->verbose=1;
    size_t kickrank = 3;
    double epsilon = 1e-10;
    struct tt * ttb = tt_cross_adapt(&wrap_full_tensor_for_cross, opts, 
                        kickrank, 10, epsilon, (void *) (&t));

    CuAssertIntEquals(tc, 3, ttb->dim);
    CuAssertIntEquals(tc, 1, ttb->ranks[0]);
    CuAssertIntEquals(tc, 2, ttb->ranks[1]);
    CuAssertIntEquals(tc, 2, ttb->ranks[2]);
    CuAssertIntEquals(tc, 1, ttb->ranks[3]);
    CuAssertIntEquals(tc, nvals[0], ttb->nvals[0]);
    CuAssertIntEquals(tc, nvals[1], ttb->nvals[1]);
    CuAssertIntEquals(tc, nvals[2], ttb->nvals[2]);

    size_t elem[3];
    double tt_elem;
    double tens_elem;
    for (ii = 0; ii < t.nvals[0]; ii++){
        elem[0] = ii; 
        for (jj = 0; jj < t.nvals[1]; jj++){
            elem[1] = jj;
            for (kk = 0; kk < t.nvals[2]; kk++){
                elem[2] = kk;
                tens_elem = tensor_elem(&t,elem);
                tt_elem = ttelem(ttb,elem);
                CuAssertDblEquals(tc, tens_elem, tt_elem, 1e-10);
            }
        }
    }
    
    free_cross_opts(&opts);
    freett(ttb);
}

CuSuite * CrossGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_tensor_elem_wrap);
    SUITE_ADD_TEST(suite, Test_tt_cross2d);
    SUITE_ADD_TEST(suite, Test_tt_cross3d);
    SUITE_ADD_TEST(suite, Test_tt_cross_adapt);
    return suite;
}

void RunAllTests(void) {
    
    printf("Running Test Suite: lib_tensordecomp\n");

    CuString * output = CuStringNew();
    CuSuite * suite = CuSuiteNew();
    
    CuSuite * t = TTGetSuite();
    CuSuite * cross = CrossGetSuite();

    CuSuiteAddSuite(suite, t);
    CuSuiteAddSuite(suite, cross);

    CuSuiteRun(suite);
    CuSuiteSummary(suite, output);
    CuSuiteDetails(suite, output);
    printf("%s \n", output->buffer);
    
    CuSuiteDelete(t);
    CuSuiteDelete(cross);
    CuStringDelete(output);
    free(suite);
}

int main(void) {
    RunAllTests();
}
