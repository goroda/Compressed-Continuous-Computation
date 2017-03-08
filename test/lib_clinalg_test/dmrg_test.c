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

#include "array.h"

#include "CuTest.h"
#include "testfunctions.h"

#include "lib_funcs.h"
#include "lib_linalg.h"
#include "lib_clinalg.h"
#include "c3_interface.h"

void Test_rightorth(CuTest * tc)
{
    printf("Testing Function: function_train_orthor\n");
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcCheck2,NULL);
    // set function monitor

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-10.0);
    ope_opts_set_ub(opts,10.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 5;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(-1.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);

    struct FunctionTrain * fcopy = function_train_copy(ft);
    struct MultiApproxOpts * fopts = c3approx_get_approx_args(c3a);
    struct FunctionTrain * ao = function_train_orthor(ft,fopts);
    
    size_t ii,jj,kk;
    for (ii = 1; ii < dim; ii++){
        double * intmat = qmaqmat_integrate(ao->cores[ii],ao->cores[ii]);
        for (jj = 0; jj < ao->cores[ii]->nrows; jj++){
            for (kk = 0; kk < ao->cores[ii]->nrows; kk++){
                if (jj == kk){
                    CuAssertDblEquals(tc,1.0,intmat[jj*ao->cores[ii]->nrows+kk],1e-14);
                }
                else{
                    CuAssertDblEquals(tc,0.0,intmat[jj*ao->cores[ii]->nrows+kk],1e-14);
                }
            }
        }
        free(intmat); intmat = NULL;
    }

    double diff = function_train_relnorm2diff(ao,fcopy);
    //printf("\nfinal diff = %G\n",diff*diff);
    CuAssertDblEquals(tc,0.0,diff*diff,1e-14);

    function_train_free(ft); ft = NULL;
    function_train_free(fcopy); fcopy = NULL;
    function_train_free(ao); ao = NULL;
    fwrap_destroy(fw); fw = NULL;
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,start);
}

/* double funcCheck3(double * x, void * args){ */
/*     assert (args == NULL); */
/*     double out = pow(x[0] * x[1],2) + x[1]*sin(x[0]); */
/*     return out; */
/* } */

void Test_dmrg_prod(CuTest * tc)
{

    printf("Testing Function: dmrg_prod\n");
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcCheck2,NULL);
    // set function monitor

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.0);
    ope_opts_set_ub(opts,1.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 5;
    double ** startp = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        startp[ii] = linspace(-1.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,startp);
    struct FunctionTrain * a = c3approx_do_cross(c3a,fw,1);
    struct MultiApproxOpts * fopts = c3approx_get_approx_args(c3a);
    struct FunctionTrain * ft = function_train_product(a,a);
    struct FunctionTrain * fcopy = function_train_copy(ft);
    struct FunctionTrain * rounded = function_train_round(ft,1e-12,fopts);

    struct FunctionTrain * start = function_train_constant(1.0,fopts);
    struct FunctionTrain * finish = dmrg_product(start,a,a,1e-10,10,1e-12,0,fopts);

    double diff = function_train_relnorm2diff(finish,fcopy);
    CuAssertDblEquals(tc,0.0,diff*diff,1e-14);
    

    function_train_free(a); a = NULL;
    function_train_free(ft); ft = NULL;
    function_train_free(fcopy); fcopy = NULL;
    function_train_free(start); start = NULL;
    function_train_free(finish); finish = NULL;
    function_train_free(rounded); rounded = NULL;
    fwrap_destroy(fw);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,startp);
}

CuSuite * CLinalgDMRGGetSuite()
{
    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_rightorth);
    SUITE_ADD_TEST(suite,Test_dmrg_prod);
    return suite;
}

void Test_diffusion_midleft(CuTest * tc)
{
    printf("Testing Function: dmrg_diffusion_midleft \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;
    
    size_t r11 = 2;
    size_t r12 = 4;

    size_t r21 = 7;
    size_t r22 = 3;

    size_t r = 8;
    double diff;

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.0);
    ope_opts_set_ub(opts,1.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    
    struct Qmarray * a = qmarray_poly_randu(LEGENDRE,r11,r12,
                                            maxorder,lb,ub);
    struct Qmarray * da = qmarray_deriv(a);

    struct Qmarray * f = qmarray_poly_randu(LEGENDRE,r21,r22,
                                            maxorder,lb,ub);
    struct Qmarray * df = qmarray_deriv(f);
    struct Qmarray * ddf = qmarray_deriv(df);
 
    double * mat = drandu(r*r11*r21*2);

    struct Qmarray * af = qmarray_kron(a,f);
    struct Qmarray * af2a = qmarray_kron(da,df);
    struct Qmarray * af2b = qmarray_kron(a,ddf);

    qmarray_axpy(1.0,af2a, af2b);

    struct Qmarray * zer = qmarray_zeros(af->nrows,
                                         af->ncols,qmopts);
    struct Qmarray * c1 = qmarray_stackv(af,af2b);
    struct Qmarray * c2 = qmarray_stackv(zer,af);
    struct Qmarray * comb = qmarray_stackh(c1,c2);

    struct Qmarray * shouldbe = mqma(mat,comb,r);

    struct Qmarray * is = qmarray_alloc(r,2*r12 * r22);
    dmrg_diffusion_midleft(da,a,ddf,df,f,mat,r,is);
    
    size_t ii;
    for (ii = 0; ii < is->nrows * is->ncols; ii++){
        diff = generic_function_norm2diff(is->funcs[ii],shouldbe->funcs[ii]);
        CuAssertDblEquals(tc,0.0,diff*diff,1e-14);
        //printf("diff = %G\n",diff);
    }
    diff = qmarray_norm2diff(is,shouldbe);
    //printf("diff = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff*diff,1e-14);
    
    qmarray_free(a); a = NULL;
    qmarray_free(da); da = NULL;
    qmarray_free(f); f = NULL;
    qmarray_free(af); af = NULL;
    qmarray_free(af2a); af2a = NULL;
    qmarray_free(af2b); af2b = NULL;
    qmarray_free(zer); zer = NULL;
    qmarray_free(df); df = NULL;
    qmarray_free(ddf); ddf = NULL;
    qmarray_free(is); is = NULL;
    qmarray_free(c1); c1 = NULL;
    qmarray_free(c2); c2 = NULL;
    qmarray_free(comb); comb = NULL;
    qmarray_free(shouldbe); shouldbe = NULL;
    free(mat); mat = NULL;

    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
}

void Test_diffusion_lastleft(CuTest * tc)
{
    printf("Testing Function: dmrg_diffusion_lastleft \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;
    
    size_t r11 = 9;
    size_t r12 = 4;

    size_t r21 = 7;
    size_t r22 = 4;

    size_t r = 8;
    double diff;
    
    struct Qmarray * a = qmarray_poly_randu(LEGENDRE,r11,r12,
                                            maxorder,lb,ub);
    struct Qmarray * da = qmarray_deriv(a);

    struct Qmarray * f = qmarray_poly_randu(LEGENDRE,r21,r22,
                                            maxorder,lb,ub);
    struct Qmarray * df = qmarray_deriv(f);
    struct Qmarray * ddf = qmarray_deriv(df);
 
    double * mat = drandu(r*r11*r21*2);

    struct Qmarray * af = qmarray_kron(a,f);
    struct Qmarray * af2a = qmarray_kron(da,df);
    struct Qmarray * af2b = qmarray_kron(a,ddf);
    
    qmarray_axpy(1.0,af2a, af2b);
    struct Qmarray * comb = qmarray_stackv(af,af2b);
    struct Qmarray * shouldbe = mqma(mat,comb,r);
    
    struct Qmarray * is = qmarray_alloc(r,r22*r22);
    dmrg_diffusion_lastleft(da,a,ddf,df,f,mat,r,is);
    
    size_t ii;
    for (ii = 0; ii < is->nrows * is->ncols; ii++){
        diff = generic_function_norm2diff(is->funcs[ii],shouldbe->funcs[ii]);
        CuAssertDblEquals(tc,0.0,diff*diff,1e-14);
        //printf("diff = %G\n",diff);
    }
    diff = qmarray_norm2diff(is,shouldbe);
    //printf("diff = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff*diff,1e-14);
    
    qmarray_free(is); is = NULL;
    qmarray_free(a); a = NULL;
    qmarray_free(da); da = NULL;
    qmarray_free(f); f = NULL;
    qmarray_free(df); df = NULL;
    qmarray_free(ddf); ddf = NULL;
    qmarray_free(af); af = NULL;
    qmarray_free(af2a); af2a = NULL;
    qmarray_free(af2b); af2b = NULL;
    qmarray_free(comb); comb = NULL;
    qmarray_free(shouldbe); shouldbe = NULL;
    free(mat); mat = NULL;
}

void Test_diffusion_midright(CuTest * tc)
{
    printf("Testing Function: dmrg_diffusion_midright \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;
    
    size_t r11 = 2;
    size_t r12 = 4;

    size_t r21 = 7;
    size_t r22 = 3;

    size_t r = 8;
    double diff;
    
    struct Qmarray * a = qmarray_poly_randu(LEGENDRE,r11,r12,maxorder,lb,ub);
    struct Qmarray * da = qmarray_deriv(a);

    struct Qmarray * f = qmarray_poly_randu(LEGENDRE,r21,r22,maxorder,lb,ub);
    struct Qmarray * df = qmarray_deriv(f);
    struct Qmarray * ddf = qmarray_deriv(df);
 
    double * mat = drandu(r*r12*r22*2);

    struct Qmarray * af = qmarray_kron(a,f);
    struct Qmarray * af2a = qmarray_kron(da,df);
    struct Qmarray * af2b = qmarray_kron(a,ddf);
    qmarray_axpy(1.0,af2a, af2b);

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.0);
    ope_opts_set_ub(opts,1.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct Qmarray * zer = qmarray_zeros(af->nrows,af->ncols,qmopts);
    struct Qmarray * c1 = qmarray_stackv(af,af2b);
    struct Qmarray * c2 = qmarray_stackv(zer,af);
    struct Qmarray * comb = qmarray_stackh(c1,c2);

    struct Qmarray * shouldbe = qmam(comb,mat,r);

    struct Qmarray * is = qmarray_alloc(2*r11 * r21,r);
    dmrg_diffusion_midright(da,a,ddf,df,f,mat,r,is);
    
    size_t ii;
    for (ii = 0; ii < is->nrows * is->ncols; ii++){
        diff = generic_function_norm2diff(is->funcs[ii],shouldbe->funcs[ii]);
        CuAssertDblEquals(tc,0.0,diff*diff,1e-14);
        //printf("diff = %G\n",diff);
    }
    diff = qmarray_norm2diff(is,shouldbe);
    //printf("diff = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff*diff,1e-14);
    
    qmarray_free(a); a = NULL;
    qmarray_free(da); da = NULL;
    qmarray_free(f); f = NULL;
    qmarray_free(af); af = NULL;
    qmarray_free(af2a); af2a = NULL;
    qmarray_free(af2b); af2b = NULL;
    qmarray_free(df); df = NULL;
    qmarray_free(ddf); ddf = NULL;
    qmarray_free(is); is = NULL;
    qmarray_free(comb); comb = NULL;
    qmarray_free(c1); c1 = NULL;
    qmarray_free(c2); c2 = NULL;
    qmarray_free(zer); zer = NULL;
    qmarray_free(shouldbe); shouldbe = NULL;
    free(mat); mat = NULL;
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
}

void Test_diffusion_firstright(CuTest * tc)
{
    printf("Testing Function: dmrg_diffusion_firstright \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;
    
    size_t r11 = 7;
    size_t r12 = 4;

    size_t r21 = 7;
    size_t r22 = 3;

    size_t r = 6;
    double diff;
    
    struct Qmarray * a = qmarray_poly_randu(LEGENDRE,r11,r12,maxorder,lb,ub);
    struct Qmarray * da = qmarray_deriv(a);

    struct Qmarray * f = qmarray_poly_randu(LEGENDRE,r21,r22,maxorder,lb,ub);
    struct Qmarray * df = qmarray_deriv(f);
    struct Qmarray * ddf = qmarray_deriv(df);
 
    double * mat = drandu(r*r12*r22*2);

    struct Qmarray * af = qmarray_kron(a,f);
    struct Qmarray * af2a = qmarray_kron(da,df);
    struct Qmarray * af2b = qmarray_kron(a,ddf);
    
    qmarray_axpy(1.0,af2a, af2b);
    struct Qmarray * comb = qmarray_stackh(af2b,af);
    struct Qmarray * shouldbe = qmam(comb,mat,r);

    struct Qmarray * is = qmarray_alloc(r11 * r21,r);
    dmrg_diffusion_firstright(da,a,ddf,df,f,mat,r,is);
    
    size_t ii;
    for (ii = 0; ii < is->nrows * is->ncols; ii++){
        diff = generic_function_norm2diff(is->funcs[ii],shouldbe->funcs[ii]);
        CuAssertDblEquals(tc,0.0,diff*diff,1e-14);
        //printf("diff = %G\n",diff);
    }
    diff = qmarray_norm2diff(is,shouldbe);
    //printf("diff = %G\n",diff);
    CuAssertDblEquals(tc,0.0,diff*diff,1e-14);
    
    qmarray_free(a); a = NULL;
    qmarray_free(da); da = NULL;
    qmarray_free(f); f = NULL;
    qmarray_free(af); af = NULL;
    qmarray_free(af2a); af2a = NULL;
    qmarray_free(af2b); af2b = NULL;
    qmarray_free(df); df = NULL;
    qmarray_free(ddf); ddf = NULL;
    qmarray_free(is); is = NULL;
    qmarray_free(comb); comb = NULL;
    qmarray_free(shouldbe); shouldbe = NULL;
    free(mat); mat = NULL;
}

void Test_diffusion_op_struct(CuTest * tc)
{
    
    printf("Testing Function: diffusion operator structure (not really a function)\n");

    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcCheck2,NULL);
    // set function monitor

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.0);
    ope_opts_set_ub(opts,1.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 2;
    double ** startp = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        startp[ii] = linspace(-1.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,startp);
    struct FunctionTrain * f = c3approx_do_cross(c3a,fw,0);

    size_t ranks[5] = {1,2,2,2,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a =
        function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);
    
    struct FT1DArray * fgrad = function_train_gradient(f);
    struct FT1DArray * temp1 = ft1d_array_alloc(dim);
    size_t ii;
    for (ii = 0; ii < dim; ii++){
        temp1->ft[ii] = function_train_product(a,fgrad->ft[ii]);
    }

    struct FunctionTrain * ftsum = function_train_copy(temp1->ft[0]);
    qmarray_free(ftsum->cores[0]); ftsum->cores[0] = NULL;
    ftsum->cores[0] = qmarray_deriv(temp1->ft[0]->cores[0]);

    struct FunctionTrain * is = function_train_alloc(dim);
    is->ranks[0] = 1;
    is->ranks[dim] = 1;

    struct Qmarray * da[4];
    struct Qmarray * df[4];
    struct Qmarray * ddf[4];
    da[0] = qmarray_deriv(a->cores[0]);
    df[0] = qmarray_deriv(f->cores[0]);
    ddf[0] = qmarray_deriv(df[0]);
    
    struct Qmarray * l1 = qmarray_kron(a->cores[0],ddf[0]);
    struct Qmarray * l2 = qmarray_kron(da[0],df[0]);
    qmarray_axpy(1.0,l1,l2);
    struct Qmarray * l3 = qmarray_kron(a->cores[0],f->cores[0]);
    is->cores[0] = qmarray_stackh(l2,l3);
    is->ranks[1] = is->cores[0]->ncols;
    qmarray_free(l1); l1 = NULL;
    qmarray_free(l2); l2 = NULL;
    qmarray_free(l3); l3 = NULL;

    for (ii = 1; ii < dim; ii++){
        struct FunctionTrain * t = function_train_copy(temp1->ft[ii]);
        qmarray_free(t->cores[ii]); t->cores[ii] = NULL;
        t->cores[ii] = qmarray_deriv(temp1->ft[ii]->cores[ii]);

        struct FunctionTrain * t2 =
            function_train_sum(ftsum,t);

        function_train_free(ftsum); ftsum = NULL;
        ftsum = function_train_copy(t2);
        function_train_free(t2); t2 = NULL;
        function_train_free(t); t = NULL;

        da[ii] = qmarray_deriv(a->cores[ii]);
        df[ii] = qmarray_deriv(f->cores[ii]);
        ddf[ii] = qmarray_deriv(df[ii]);
    
        if (ii < dim-1){
            struct Qmarray * addf = qmarray_kron(a->cores[ii],ddf[ii]);
            struct Qmarray * dadf = qmarray_kron(da[ii],df[ii]);
            qmarray_axpy(1.0,addf,dadf);
            struct Qmarray * af = qmarray_kron(a->cores[ii],f->cores[ii]);
            struct Qmarray * zer = qmarray_zeros(af->nrows,af->ncols,qmopts);
            struct Qmarray * l3l1 = qmarray_stackv(af,dadf);
            struct Qmarray * zl3 = qmarray_stackv(zer,af);
            is->cores[ii] = qmarray_stackh(l3l1,zl3);
            is->ranks[ii+1] = is->cores[ii]->ncols;
            qmarray_free(addf); addf = NULL;
            qmarray_free(dadf); dadf = NULL;
            qmarray_free(af); af = NULL;
            qmarray_free(zer); zer = NULL;
            qmarray_free(l3l1); l3l1 = NULL;
            qmarray_free(zl3); zl3 = NULL;
        }
    }
    ii = dim-1;
    l1 = qmarray_kron(a->cores[ii],ddf[ii]);
    l2 = qmarray_kron(da[ii],df[ii]);
    qmarray_axpy(1.0,l1,l2);
    l3 = qmarray_kron(a->cores[ii],f->cores[ii]);
    is->cores[ii] = qmarray_stackv(l3,l2);
    is->ranks[ii+1] = is->cores[ii]->ncols;
    qmarray_free(l1); l1 = NULL;
    qmarray_free(l2); l2 = NULL;
    qmarray_free(l3); l3 = NULL;

    struct FunctionTrain * shouldbe = function_train_copy(ftsum);
    
    double diff = function_train_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff*diff,1e-10);

    for (ii = 0; ii  < dim; ii++){
        qmarray_free(da[ii]);
        qmarray_free(df[ii]);
        qmarray_free(ddf[ii]);
    }
    
    function_train_free(shouldbe); shouldbe = NULL;
    function_train_free(is); is = NULL;
    bounding_box_free(bds); bds = NULL;
    function_train_free(f); f = NULL;
    function_train_free(a); a = NULL;
    ft1d_array_free(fgrad); fgrad = NULL;
    ft1d_array_free(temp1); temp1 = NULL;
    function_train_free(ftsum); ftsum = NULL;

    fwrap_destroy(fw);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,startp);
}

void Test_diffusion_dmrg(CuTest * tc)
{
    
    printf("Testing Function: dmrg_diffusion\n");
    size_t dim = 4;    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,funcCheck2,NULL);
    // set function monitor

    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.0);
    ope_opts_set_ub(opts,1.0);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    
    int verbose = 0;
    size_t init_rank = 2;
    double ** startp = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        startp[ii] = linspace(-1.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,startp);
    struct FunctionTrain * f = c3approx_do_cross(c3a,fw,0);

    
    size_t ranks[5] = {1,2,2,2,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,
                                                         maxorder);

    struct MultiApproxOpts * fopts = c3approx_get_approx_args(c3a);
    struct FunctionTrain * is = dmrg_diffusion(a,f,1e-5,5,1e-10,0,fopts);
    struct FunctionTrain * shouldbe = exact_diffusion(a,f,fopts);
    
    double diff = function_train_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff*diff,1e-10);

    function_train_free(shouldbe); shouldbe = NULL;
    function_train_free(is); is = NULL;
    bounding_box_free(bds); bds = NULL;
    function_train_free(f); f = NULL;
    function_train_free(a); a = NULL;

    fwrap_destroy(fw);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim,startp);
}

CuSuite * CLinalgDiffusionGetSuite()
{
    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_diffusion_midleft);
    SUITE_ADD_TEST(suite, Test_diffusion_lastleft);
    SUITE_ADD_TEST(suite, Test_diffusion_midright);
    SUITE_ADD_TEST(suite, Test_diffusion_firstright);
    SUITE_ADD_TEST(suite, Test_diffusion_op_struct);
    SUITE_ADD_TEST(suite, Test_diffusion_dmrg);
    return suite;
}
