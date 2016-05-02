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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "array.h"
#include "CuTest.h"

#include "lib_funcs.h"
#include "lib_clinalg.h"
#include "lib_linalg.h"


void Test_function_train_initsum(CuTest * tc){

    printf("Testing function: function_train_initsum \n");

    double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};
    
    enum poly_type ptype = LEGENDRE;
    struct FtApproxArgs * ftargs = 
            ft_approx_args_createpoly(4, &ptype, NULL);
    struct BoundingBox * bounds = bounding_box_init_std(4);
    
    struct FunctionTrain * ft = function_train_initsum(4, funcs, args, bounds, 
                                    ftargs);
        
    double pt[4];
    double val, tval; 
    
    size_t N = 20;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;

    size_t ii,jj,kk,ll;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    tval = funcs[0](pt[0],args[0]) + 
                            funcs[1](pt[1],args[1]) + 
                            funcs[2](pt[2],args[2]) + 
                            funcs[3](pt[3],args[3]);
                    val = function_train_eval(ft,pt);
                    den += pow(tval,2.0);
                    err += pow(tval-val,2.0);
                }
            }
        }
    }
    err = err/den;

    CuAssertDblEquals(tc,0.0,err,1e-15);
    //printf("err = %G\n",err);
    
    free(xtest);
    function_train_free(ft);
    bounding_box_free(bounds);
    ft_approx_args_free(ftargs);
}   

void Test_function_train_linear(CuTest * tc)
{
    printf("Testing Function: function_train_linear \n");
    
    struct BoundingBox * bounds = bounding_box_init_std(3);

    double coeffs[3] = {1.0, 2.0, 3.0};
    enum function_class fc = POLYNOMIAL;
    enum poly_type ptype = LEGENDRE;
    struct FunctionTrain * f =function_train_linear(fc,&ptype,3,
                                                    bounds, coeffs,NULL);
    
    double pt[3] = { -0.1, 0.4, 0.2 };
    double eval = function_train_eval(f,pt);
    CuAssertDblEquals(tc, 1.3, eval, 1e-14);
    
    pt[0] = 0.8; pt[1] = -0.2; pt[2] = 0.3;
    eval = function_train_eval(f,pt);
    CuAssertDblEquals(tc, 1.3, eval, 1e-14);

    pt[0] = -0.8; pt[1] = 1.0; pt[2] = -0.01;
    eval = function_train_eval(f,pt);
    CuAssertDblEquals(tc, 1.17, eval, 1e-14);
    
    bounding_box_free(bounds);
    function_train_free(f);
}

void Test_function_train_quadratic(CuTest * tc)
{
    printf("Testing Function: function_train_quadratic (1/2)\n");
    size_t dim = 3;
    double lb = -3.12;
    double ub = 2.21;
    struct BoundingBox * bounds = bounding_box_init(dim,lb,ub);
    double * quad = calloc_double(dim * dim);
    double * coeff = calloc_double(dim);
    size_t ii,jj,kk;
    for (ii = 0; ii < dim; ii++){
        coeff[ii] = randu();
        for (jj = 0; jj < dim; jj++){
            quad[ii*dim+jj] = randu();
        }
    }
    enum function_class fc = POLYNOMIAL;
    enum poly_type ptype = LEGENDRE;
    struct FunctionTrain * f = function_train_quadratic(fc,&ptype,dim,
                                                        bounds, quad,
                                                        coeff,NULL);
    size_t N = 10;
    double * xtest = linspace(lb,ub,N);
    double * pt = calloc_double(dim);
    size_t ll, mm;
    double should, is;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N;  jj++){
            for (kk = 0; kk < N; kk++){
                pt[0] = xtest[ii]; pt[1] = xtest[jj]; pt[2] = xtest[kk];
                should = 0.0;
                for (ll = 0; ll< dim; ll++){
                    for (mm = 0; mm < dim; mm++){
                        should += (pt[ll]-coeff[ll])*quad[mm*dim+ll]*(pt[mm]-coeff[mm]);
                    }
                }
                //printf("should=%G\n",should);
                is = function_train_eval(f,pt);
                //printf("is=%G\n",is);
                CuAssertDblEquals(tc,should,is,1e-12);
            }
        }
    }
    
    free(xtest);
    free(pt);
    free(quad);
    free(coeff);
    bounding_box_free(bounds);
    function_train_free(f);
}

void Test_function_train_quadratic2(CuTest * tc)
{
    printf("Testing Function: function_train_quadratic (2/2)\n");
    size_t dim = 4;
    double lb = -1.32;
    double ub = 6.0;
    struct BoundingBox * bounds = bounding_box_init(dim,lb,ub);
    double * quad = calloc_double(dim * dim);
    double * coeff = calloc_double(dim);
    size_t ii,jj,kk,zz;
    for (ii = 0; ii < dim; ii++){
        coeff[ii] = randu();
        for (jj = 0; jj < dim; jj++){
            quad[ii*dim+jj] = randu();
        }
    }
    enum function_class fc = POLYNOMIAL;
    enum poly_type ptype = LEGENDRE;
    struct FunctionTrain * f = function_train_quadratic(fc,&ptype,dim,
                                                        bounds, quad,coeff,NULL);
    size_t N = 10;
    double * xtest = linspace(lb,ub,N);
    double * pt = calloc_double(dim);
    size_t ll, mm;
    double should, is;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N;  jj++){
            for (kk = 0; kk < N; kk++){
                for (zz = 0; zz < N; zz++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; pt[2] = xtest[kk]; pt[3] = xtest[zz];
                    should = 0.0;
                    for (ll = 0; ll< dim; ll++){
                        for (mm = 0; mm < dim; mm++){
                            should += (pt[ll]-coeff[ll])*quad[mm*dim+ll]*(pt[mm]-coeff[mm]);
                        }
                    }
                    //printf("should=%G\n",should);
                    is = function_train_eval(f,pt);
                    //printf("is=%G\n",is);
                    CuAssertDblEquals(tc,should,is,1e-12);
                }
            }
        }
    }
    
    free(xtest);
    free(pt);
    free(quad);
    free(coeff);
    bounding_box_free(bounds);
    function_train_free(f);
}


void Test_function_train_sum_function_train_round(CuTest * tc)
{
    printf("Testing Function: function_train_sum and ft_round \n");
    
    struct BoundingBox * bounds = bounding_box_init_std(3);

    enum function_class fc = POLYNOMIAL;
    enum poly_type ptype = LEGENDRE;
    double coeffs[3] = {1.0, 2.0, 3.0};
    struct FunctionTrain * a =
        function_train_linear(fc,&ptype,3,bounds,coeffs,NULL);

    double coeffsb[3] = {1.5, -0.2, 3.310};
    struct FunctionTrain * b = 
        function_train_linear(fc,&ptype,3,bounds,coeffsb,NULL);
    
    struct FunctionTrain * c = function_train_sum(a,b);
    CuAssertIntEquals(tc,1,c->ranks[0]);
    CuAssertIntEquals(tc,4,c->ranks[1]);
    CuAssertIntEquals(tc,4,c->ranks[2]);
    CuAssertIntEquals(tc,1,c->ranks[3]);


    double pt[3];
    double eval, evals;
    
    pt[0] = -0.1; pt[1] = 0.4; pt[2]=0.2; 
    eval = function_train_eval(c,pt);
    evals = -0.1*(1.0 + 1.5) + 0.4*(2.0-0.2) + 0.2*(3.0 + 3.31);
    CuAssertDblEquals(tc, evals, eval, 1e-14);
    
    pt[0] = 0.8; pt[1] = -0.2; pt[2] = 0.3;
    evals = 0.8*(1.0 + 1.5) - 0.2*(2.0-0.2) + 0.3*(3.0 + 3.31);
    eval = function_train_eval(c,pt);
    CuAssertDblEquals(tc, evals, eval, 1e-14);

    pt[0] = -0.8; pt[1] = 1.0; pt[2] = -0.01;
    evals = -0.8*(1.0 + 1.5) + 1.0*(2.0-0.2) - 0.01*(3.0 + 3.31);
    eval = function_train_eval(c,pt);
    CuAssertDblEquals(tc, evals, eval,1e-14);
    
    struct FunctionTrain * d = function_train_round(c, 1e-10);
    CuAssertIntEquals(tc,1,d->ranks[0]);
    CuAssertIntEquals(tc,2,d->ranks[1]);
    CuAssertIntEquals(tc,2,d->ranks[2]);
    CuAssertIntEquals(tc,1,d->ranks[3]);

    pt[0] = -0.1; pt[1] = 0.4; pt[2]=0.2; 
    eval = function_train_eval(d,pt);
    evals = -0.1*(1.0 + 1.5) + 0.4*(2.0-0.2) + 0.2*(3.0 + 3.31);
    CuAssertDblEquals(tc, evals, eval, 1e-14);
    
    pt[0] = 0.8; pt[1] = -0.2; pt[2] = 0.3;
    evals = 0.8*(1.0 + 1.5) - 0.2*(2.0-0.2) + 0.3*(3.0 + 3.31);
    eval = function_train_eval(d,pt);
    CuAssertDblEquals(tc, evals, eval, 1e-14);

    pt[0] = -0.8; pt[1] = 1.0; pt[2] = -0.01;
    evals = -0.8*(1.0 + 1.5) + 1.0*(2.0-0.2) - 0.01*(3.0 + 3.31);
    eval = function_train_eval(d,pt);
    CuAssertDblEquals(tc, evals, eval,1e-14);
    

    bounding_box_free(bounds);
    function_train_free(a);
    function_train_free(b);
    function_train_free(c);
    function_train_free(d);
}

void Test_function_train_scale(CuTest * tc)
{
    printf("Testing Function: function_train_scale \n");
    double (*funcs [4])(double, void *) = {func, func2, func3, func4};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};
    
    struct BoundingBox * bounds = bounding_box_init_std(4);
    struct FunctionTrain * ft = function_train_initsum(4, funcs, args, bounds,NULL); 

    double pt[4];
    double val, tval;
    double scale = 4.0;
    function_train_scale(ft,scale);
    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;

    size_t ii,jj,kk,ll;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    tval = func(pt[0],&c) + func2(pt[1],&c2) + 
                            func3(pt[2],&c3) + func4(pt[3],&c4);
                    tval = tval * scale;
                    val = function_train_eval(ft,pt);
                    den += pow(tval,2.0);
                    err += pow(tval-val,2.0);
                }
            }
        }
    }
    err = err/den;

    CuAssertDblEquals(tc,0.0,err,1e-15);
    
    free(xtest);
    bounding_box_free(bounds);
    function_train_free(ft);
}

void Test_function_train_product(CuTest * tc)
{
    printf("Testing Function: function_train_product \n");
    double (*funcs [4])(double, void *) = {func, func2, func3, func4};
    double (*funcs2 [4])(double, void *) = {func2, func5, func4, func6};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};
    struct counter c5; c5.N = 0;
    struct counter c6; c6.N = 0;
    struct counter c7; c7.N = 0;
    struct counter c8; c8.N = 0;
    void * args2[4] = {&c5, &c6, &c7, &c8};
    
    struct BoundingBox * bounds = bounding_box_init_std(4);
    struct FunctionTrain * ft = function_train_initsum(4, funcs, args, bounds,NULL); 
    struct FunctionTrain * gt = function_train_initsum(4, funcs2, args2, bounds,NULL); 
    struct FunctionTrain * ft2 =  function_train_product(ft,gt);

    double pt[4];
    double val, tval1,tval2; 
    
    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;

    size_t ii,jj,kk,ll;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    tval1 =  function_train_eval(ft,pt);
                    tval2 =  function_train_eval(gt,pt);
                    val = function_train_eval(ft2,pt);
                    den += pow(tval1*tval2,2.0);
                    err += pow(tval1*tval2-val,2.0);
                }
            }
        }
    }
    err = err/den;

    CuAssertDblEquals(tc,0.0,err,1e-15);
    
    free(xtest);
    bounding_box_free(bounds);
    function_train_free(ft);
    function_train_free(gt);
    function_train_free(ft2);
}

void Test_function_train_integrate(CuTest * tc)
{
    printf("Testing Function: function_train_integrate \n");
    double (*funcs [4])(double, void *) = {&func, &func2, &func3, &func4};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};
    
    struct BoundingBox * bounds = bounding_box_init_std(4);
    bounds->lb[0] = 0;
    bounds->lb[1] = -1.0;
    bounds->lb[2] = -5.0;
    bounds->lb[3] = -5.0;
    struct FunctionTrain * ft = function_train_initsum(4, funcs, args, bounds,NULL); 
    double out =  function_train_integrate(ft);
    
    double shouldbe = 110376.0/5.0;
    double rel_error = pow(out-shouldbe,2)/fabs(shouldbe);
    CuAssertDblEquals(tc, 0.0 ,rel_error,1e-15);

    bounding_box_free(bounds);
    function_train_free(ft);
}

void Test_function_train_inner(CuTest * tc)
{
    printf("Testing Function: function_train_inner \n");
    double (*funcs [4])(double, void *) = {func, func2, func3, func4};
    double (*funcs2 [4])(double, void *) = {func6, func5, func4, func3};
    struct counter c; c.N = 0;
    struct counter c2; c2.N = 0;
    struct counter c3; c3.N = 0;
    struct counter c4; c4.N = 0;
    void * args[4] = {&c, &c2, &c3, &c4};
    struct counter c5; c5.N = 0;
    struct counter c6; c6.N = 0;
    struct counter c7; c7.N = 0;
    struct counter c8; c8.N = 0;
    void * args2[4] = {&c5, &c6, &c7, &c8};
    
    struct BoundingBox * bounds = bounding_box_init_std(4);
    struct FunctionTrain * ft = function_train_initsum(4,funcs,args,bounds,NULL); 
    struct FunctionTrain * gt = function_train_initsum(4,funcs2,args2,bounds,NULL); 
    struct FunctionTrain * ft2 =  function_train_product(gt,ft);
    
    double int1 = function_train_integrate(ft2);
    double int2 = function_train_inner(gt,ft);
    
    double relerr = pow(int1-int2,2)/pow(int1,2);
    CuAssertDblEquals(tc,0.0,relerr,1e-13);
    
    bounding_box_free(bounds);
    function_train_free(ft);
    function_train_free(ft2);
    function_train_free(gt);
}

// 2 dimensional function
double funcnda(double x, void * args){
    assert ( args == NULL);
    return x;
}
double funcndb(double x, void * args){
    assert ( args == NULL);
    return pow(x,2);
}
double funcndc(double x, void * args){
    assert ( args == NULL);
    return exp(x);
}
double funcndd(double x, void * args){
    assert ( args == NULL);
    return sin(x);
}

// two dimensions
double funcnd1(double * x, void * args){
    
    assert (args == NULL);
    double out = funcnda(x[0],NULL) + funcndb(x[1],NULL);
    return out;
}

double funcnd2(double * x, void * args){

    assert (args == NULL);
    
    double out;
    out = x[0] + x[1] + x[2] + x[3];
    return out;
}

void Test_ftapprox_cross(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross  (1/4)\n");
    double (*funcs [2])(double, void *) = {&funcnda, &funcndb};
    void * args[2] = {NULL, NULL};
    
    struct BoundingBox * bounds = bounding_box_init_std(2);
    struct FunctionTrain * ftref = function_train_initsum(2,funcs,args,bounds,NULL); 
       
    size_t dim = 2;
    size_t rank[3] = {1, 3, 1};
    double * yr[2];
    yr[1] = calloc_double(3);
    yr[1][0] = -1.0;
    yr[1][1] =  0.0;
    yr[1][2] =  1.0;
    yr[0] = calloc_double(3);

    struct BoundingBox * bds = bounding_box_init_std(dim);
    
    enum poly_type ptype = LEGENDRE;
    struct FtApproxArgs * fapp = 
        ft_approx_args_createpoly(dim,&ptype,NULL);
    
    struct FtCrossArgs fca;
    fca.dim = 2;
    fca.ranks = rank;
    fca.epsilon = 1e-5;
    fca.maxiter = 10;
    fca.verbose = 0;
    fca.optargs = NULL;

    struct CrossIndex * isl[2];
    struct CrossIndex * isr[2];
    cross_index_array_initialize(dim,isl,1,0,NULL,NULL);
    cross_index_array_initialize(dim,isr,0,1,rank,yr);

    //struct IndexSet ** isr = index_set_array_rnested(dim, rank, yr);
    //struct IndexSet ** isl = index_set_array_lnested(dim, rank, yr);
    
    //print_index_set_array(2,isr);
    //print_index_set_array(2,isl);

    //printf("start cross approximation\n");
    struct FunctionTrain * ft = ftapprox_cross(funcnd1,NULL,bds,ftref,
                                               isl, isr, &fca,fapp);
    //printf("ended cross approximation\n");
    
    //print_index_set_array(2,isr);
    //print_index_set_array(2,isl);

    size_t N = 20;
    double * xtest = linspace(-1,1,N);
    size_t ii,jj;
    double err = 0.0;
    double den = 0.0;
    double pt[2];
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            pt[0] = xtest[ii];
            pt[1] = xtest[jj];
            den += pow(funcnd1(pt,NULL),2);
            err += pow(funcnd1(pt,NULL) - function_train_eval(ft,pt),2);
        }
    }
    
    err /= den;
    CuAssertDblEquals(tc,0.0,err,1e-15);

    cross_index_free(isl[1]);
    cross_index_free(isr[0]);
//    index_set_array_free(dim,isr);
//    index_set_array_free(dim,isl);
    //
    bounding_box_free(bds);
    bounding_box_free(bounds);
    ft_approx_args_free(fapp);
    function_train_free(ft);
    function_train_free(ftref);
    free(yr[0]);
    free(yr[1]);
    free(xtest);
}


void Test_ftapprox_cross2(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross  (2/4)\n");
     
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-1.0,1.0);
    size_t ii,jj,kk,ll;

    struct FunctionMonitor * fm = function_monitor_initnd(funcnd2,NULL,dim,1000);

    //struct FunctionTrain * ft = 
    //    function_train_cross(funcnd2,NULL,dim,lb,ub,NULL);
    struct FunctionTrain * ft = 
        function_train_cross(function_monitor_eval,fm,bds,NULL,NULL,NULL);
    function_monitor_free(fm);

    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;
    double pt[4];
    
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    den += pow(funcnd2(pt,NULL),2.0);
                    err += pow(funcnd2(pt,NULL)-function_train_eval(ft,pt),2.0);
                    //printf("err=%G\n",err);
                }
            }
        }
    }
    err = err/den;
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc,0.0,err,1e-10);
    //CuAssertDblEquals(tc,0.0,0.0,1e-15);

    bounding_box_free(bds);
    function_train_free(ft);
    free(xtest);
}



double disc2d(double * xy, void * args)
{
    assert (args == NULL);
     
    double x = xy[0];
    double y = xy[1];
    double out = 0.0;
    if ((x > 0.5) || (y > 0.4)){
        out = 0.0;    
    }
    else{
        out = exp(5.0 * x + 5.0 * y);
        //out = x+y;
    }
    return out;
}

void Test_ftapprox_cross3(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross  (3/4)\n");
    size_t dim = 2;
    
    struct BoundingBox * bds = bounding_box_init(2,0.0,1.0); 
    
    double coeffs[2] = {0.5, 0.5};
    size_t ranks[3] = {1, 2, 1};

    enum function_class fc = POLYNOMIAL;
    enum poly_type ptype = LEGENDRE;
    struct FunctionTrain * ftref = 
        function_train_linear(fc,&ptype,dim, bds, coeffs,NULL);
            
    struct FunctionMonitor * fm = 
            function_monitor_initnd(disc2d,NULL,dim,1000*dim);
            
    double * yr[2];
    yr[1] = calloc_double(2);
    yr[1][0] = 0.3;
    yr[1][1] =  0.0;
    yr[0] = calloc_double(2);

    struct CrossIndex * isl[2];
    struct CrossIndex * isr[2];
    cross_index_array_initialize(dim,isl,1,0,NULL,NULL);
    cross_index_array_initialize(dim,isr,0,1,ranks,yr);

    //struct IndexSet ** isr = index_set_array_rnested(dim, ranks, yr);
//    struct IndexSet ** isl = index_set_array_lnested(dim, ranks, yr);

    struct FtCrossArgs fca;
    fca.epsilon = 1e-6;
    fca.maxiter = 5;
    fca.verbose = 0;
    fca.dim = dim;
    fca.ranks = ranks;
    fca.optargs = NULL;

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-6;
    aopts.minsize = 1e-8;
    aopts.nregions = 5;
    aopts.pts = NULL;

    struct FtApproxArgs * fapp = 
       ft_approx_args_createpwpoly(dim,&ptype,&aopts);

    struct FunctionTrain * ft = ftapprox_cross(function_monitor_eval, fm,
                                    bds, ftref, isl, isr, &fca,fapp);

//    printf("X Nodes \n");
//    print_cross_index(isl[1]);

//    printf("Y Nodes are \n");
//    print_cross_index(isr[0]);
    
    free(yr[0]);
    free(yr[1]);
    ft_approx_args_free(fapp);

    cross_index_free(isr[0]);
    cross_index_free(isl[1]);
//    index_set_array_free(dim,isr);
//    index_set_array_free(dim,isl);
            
    double v1, v2;
    size_t ii,jj;
    size_t N1 = 40;
    size_t N2 = 40;
    double * xtest = linspace(0.0,1.0,N1);
    double * ytest = linspace(0.0,1.0,N2);

    double out1=0.0;
    double den=0.0;
    double pt[2];
    for (ii = 0; ii < N1; ii++){
        for (jj = 0; jj < N2; jj++){
            pt[0] = xtest[ii]; pt[1] = ytest[jj];
            v1 = disc2d(pt, NULL);
            v2 = function_train_eval(ft,pt);
            den += pow(v1,2.0);
            out1 += pow(v1-v2,2.0);
            //printf("f(%G,%G) = %G, pred = %G\n",pt[0],pt[1],v1,v2);
        }
    }

    double err = sqrt(out1/den);
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc,0.0,err,1e-10);
    //CuAssertDblEquals(tc,0.0,0.0,1e-15);
    //
    bounding_box_free(bds);
    function_train_free(ftref);
    function_train_free(ft);
    function_monitor_free(fm);
    free(xtest);
    free(ytest);
}

double funcH4(double * x, void * args){
    assert (args == NULL);
    //double out = 2.0*x[0] + x[1]*pow(x[2],4) + x[3]*pow(x[0],2);
    double out = 2.0*x[0] + x[1]*pow(x[2],4) +  x[3]*pow(x[0],2);
    return out;
}

void Test_ftapprox_cross4(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross  (4/4)\n");
     
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-1.0,1.0);
    size_t ii,jj,kk,ll;

    struct FunctionMonitor * fm = function_monitor_initnd(funcH4,NULL,dim,1000);

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-3;
    aopts.minsize = 1e-2;
    aopts.nregions = 4;
    aopts.pts = NULL;

    enum poly_type ptype = LEGENDRE;
    struct FtApproxArgs * fapp = 
       ft_approx_args_createpwpoly(dim,&ptype,&aopts);


    //struct FunctionTrain * ft = 
    //    function_train_cross(funcnd2,NULL,dim,lb,ub,NULL);
    struct FunctionTrain * ft = 
        function_train_cross(function_monitor_eval,fm,bds,NULL,NULL,fapp);
    function_monitor_free(fm);
    ft_approx_args_free(fapp);
    

    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;
    double pt[4];
    
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    den += pow(funcH4(pt,NULL),2.0);
                    err += pow(funcH4(pt,NULL)-function_train_eval(ft,pt),2.0);
                    //printf("err=%G\n",err);
                }
            }
        }
    }
    err = err/den;
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc,0.0,err,1e-10);
    //CuAssertDblEquals(tc,0.0,0.0,1e-15);

    bounding_box_free(bds);
    function_train_free(ft);
    free(xtest);
}

void Test_function_train_eval_co_peruturb(CuTest * tc)
{
    printf("Testing Function: function_train_eval_co_perturb \n");
    
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-1.0,1.0);

    struct FunctionMonitor * fm = function_monitor_initnd(funcnd2,NULL,dim,1000);

    struct FunctionTrain * ft = 
        function_train_cross(function_monitor_eval,fm,bds,NULL,NULL,NULL);
    function_monitor_free(fm);

    double pt[4] = {0.5, 0.2 ,0.3, 0.8};
    double pert[8] = { 0.3, 0.6, 0.1, 0.9, 0.4, 0.6, -0.2, -0.4};
    double evals[8];
    double val = function_train_eval_co_perturb(ft,pt,pert,evals);

    double valshould = funcnd2(pt,NULL);
    CuAssertDblEquals(tc,valshould,val,1e-13);
    
    double evals_should[8];
    double pt2[4] = {pt[0],pt[1],pt[2],pt[3]};
    
    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        pt2[ii] = pert[2*ii];
        evals_should[2*ii] = funcnd2(pt2,NULL);
        CuAssertDblEquals(tc,evals_should[2*ii],evals[2*ii],1e-13);
        
        pt2[ii] = pert[2*ii+1];
        evals_should[2*ii+1] = funcnd2(pt2,NULL);;
        CuAssertDblEquals(tc,evals_should[2*ii+1],evals[2*ii+1],1e-13);

        pt2[ii] = pt[ii];
    }

    bounding_box_free(bds);
    function_train_free(ft);

}

double funch1(double * x, void * arg)
{
    (void)(arg);
    double out = x[0]+x[1]+x[2]+x[3];
    return out;
}

void Test_ftapprox_cross_hermite1(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross for hermite (1) \n");
     
    size_t dim = 4;
    size_t ii,jj,kk,ll;

    struct FunctionTrain * ft = NULL;
    ft = function_train_cross_ub(funch1,NULL,dim,NULL,NULL,NULL,NULL);

    size_t N = 10;
    double * xtest = linspace(-2.0,2.0,N);
    double err = 0.0;
    double den = 0.0;
    double pt[4];
    
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    den += pow(funch1(pt,NULL),2.0);
                    err += pow(funch1(pt,NULL)-function_train_eval(ft,pt),2.0);
                    //printf("err=%G\n",err);
                }
            }
        }
    }
    err = err/den;
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc,0.0,err,1e-10);
    //CuAssertDblEquals(tc,0.0,0.0,1e-15);

    // make sure serialization works
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double diff = function_train_relnorm2diff(ft,ftd);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;
    
    function_train_free(ft);
    free(xtest);
}

double funch2(double * x, void * arg)
{
    (void)(arg);
    double out = x[0]*x[1] + x[2]*x[3] +  x[1]*pow(x[2],5) + 
        pow(x[1],8)*pow(x[3],2);
//    double out = x[0]*x[1] + x[2]*x[3] + x[0]*exp(-x[2]*x[3]);
//    double out = x[0]*x[1] + x[2]*x[3] + x[0]*sin(x[1]*x[2]);
    /* printf("x = "); dprint(4,x); */
    /* printf("out = %G\n",out); */
    return out;
}

void Test_ftapprox_cross_hermite2(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross for hermite (2) \n");
     
    size_t dim = 4;

    struct FunctionTrain * ft = NULL;
    ft = function_train_cross_ub(funch2,NULL,dim,NULL,NULL,NULL,NULL);

//    size_t N = 10;
//    double * xtest = linspace(-2.0,2.0,N);
    double err = 0.0;
    double den = 0.0;
    double pt[4];
    
    size_t nsamples = 100000;
    for (size_t ii = 0; ii < nsamples; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            pt[jj] = randn();
        }
        double eval = funch2(pt,NULL);
        double diff = eval-function_train_eval(ft,pt);
        den += pow(eval,2.0);
        err += pow(diff,2);
//        printf("pt = "); dprint(dim,pt);
//        printf("eval = %G, diff=%G\n",eval,diff);
//        if (fabs(diff) > 1e-1){
//            exit(1);
//        }
    }
    err = err/den;
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc,0.0,err,1e-3);
    /* err = 0.0; */
    /* den = 0.0; */
    /* for (ii = 0; ii < N; ii++){ */
    /*     for (jj = 0; jj < N; jj++){ */
    /*         for (kk = 0; kk < N; kk++){ */
    /*             for (ll = 0; ll < N; ll++){ */
    /*                 pt[0] = xtest[ii]; pt[1] = xtest[jj];  */
    /*                 pt[2] = xtest[kk]; pt[3] = xtest[ll]; */
    /*                 den += pow(funch2(pt,NULL),2.0); */
    /*                 err += pow(funch2(pt,NULL)-function_train_eval(ft,pt),2.0); */
    /*                 //printf("err=%G\n",err/den); */
    /*             } */
    /*         } */
    /*     } */
    /* } */
    /* err = err/den; */
    /* //printf("err=%G\n",err); */
    /* CuAssertDblEquals(tc,0.0,err,1e-10); */
    /* //CuAssertDblEquals(tc,0.0,0.0,1e-15); */

    // make sure serialization works
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double diff = function_train_relnorm2diff(ft,ftd);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;
    
    function_train_free(ft);
    //  free(xtest);
}

void Test_ftapprox_cross_linelm1(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross for linelm (1) \n");
     
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-1.0,1.0);
    size_t ii,jj,kk,ll;

    struct FunctionMonitor * fm = function_monitor_initnd(funcnd2,NULL,
                                                          dim,1000);
    
    struct FtApproxArgs * fapp = ft_approx_args_create_le(dim,NULL);
    struct FunctionTrain * ft = NULL;
    ft = function_train_cross(function_monitor_eval,fm,bds,
                              NULL,NULL,fapp);
    function_monitor_free(fm);

    //printf("finished !\n");
    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;
    double pt[4];
    
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    den += pow(funcnd2(pt,NULL),2.0);
                    err += pow(funcnd2(pt,NULL)-function_train_eval(ft,pt),2.0);
                    //printf("err=%G\n",err);
                }
            }
        }
    }
    err = err/den;
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc,0.0,err,1e-10);
    //CuAssertDblEquals(tc,0.0,0.0,1e-15);

    // make sure serialization works
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double diff = function_train_relnorm2diff(ft,ftd);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;
    
    bounding_box_free(bds);
    function_train_free(ft);
    ft_approx_args_free(fapp);
    free(xtest);
}

void Test_ftapprox_cross_linelm2(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross for linelm (2) \n");
     
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-1.0,1.0);
    size_t ii,jj,kk,ll;

    struct FunctionMonitor * fm = function_monitor_initnd(funcnd2,NULL,
                                                          dim,1000);

    double delta = 1e-2;
    double hmin = 1e-2;
    struct LinElemExpAopts * opts = NULL;
    opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,delta,hmin);
    struct FtApproxArgs * fapp = ft_approx_args_create_le(dim,opts);
//    printf("start cross\n");
    struct FunctionTrain * ft = NULL;
    ft = function_train_cross(function_monitor_eval,fm,bds,
                              NULL,NULL,fapp);
    function_monitor_free(fm);

    //printf("finished !\n");
    size_t N = 10;
    double * xtest = linspace(-1.0,1.0,N);
    double err = 0.0;
    double den = 0.0;
    double pt[4];
    
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                for (ll = 0; ll < N; ll++){
                    pt[0] = xtest[ii]; pt[1] = xtest[jj]; 
                    pt[2] = xtest[kk]; pt[3] = xtest[ll];
                    den += pow(funcnd2(pt,NULL),2.0);
                    err += pow(funcnd2(pt,NULL)-function_train_eval(ft,pt),2.0);
                    //printf("err=%G\n",err);
                }
            }
        }
    }
    err = err/den;
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc,0.0,err,1e-10);
    //CuAssertDblEquals(tc,0.0,0.0,1e-15);

    // make sure serialization works
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double diff = function_train_relnorm2diff(ft,ftd);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;
    
    bounding_box_free(bds);
    function_train_free(ft);
    lin_elem_exp_aopts_free(opts);
    ft_approx_args_free(fapp);
    free(xtest);
}

double func_not_all(double * x, void * args)
{
    (void)(args);
    double out = x[1] + x[4];
    //printf("out = %G\n",out);
    return out;
}

void Test_ftapprox_cross_linelm3(CuTest * tc)
{
    printf("Testing Function: ftapprox_cross for linelm (3) \n");
     
    size_t dim = 6;
    struct BoundingBox * bds = bounding_box_init(dim,-1.0,1.0);

    struct FunctionMonitor * fm = 
        function_monitor_initnd(func_not_all,NULL,
                                dim,1000);
    
    struct FtApproxArgs * fapp = ft_approx_args_create_le(dim,NULL);
    struct FunctionTrain * ft = NULL;
    ft = function_train_cross(function_monitor_eval,fm,bds,
                              NULL,NULL,fapp);
    function_monitor_free(fm);

    //printf("ranks = "); iprint_sz(dim+1,ft->ranks);
    //for (size_t ii = 0; ii < dim;ii++){
    //    print_qmarray(ft->cores[ii],0,NULL);
    //}

    //printf("finished !\n");
    double pt[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double eval = function_train_eval(ft,pt);
    CuAssertDblEquals(tc,0.0,eval,1e-14);
    //printf("eval = %G\n",eval);
    
    // make sure serialization works
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double diff = function_train_relnorm2diff(ft,ftd);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    function_train_free(ftd); ftd = NULL;
    free(text); text = NULL;
    
    bounding_box_free(bds);
    function_train_free(ft);
    ft_approx_args_free(fapp);
}

double sin10d(double * x, void * args){
    
    assert ( args == NULL );

    size_t ii;
    double out = 0.0;
    for (ii = 0; ii < 10; ii++){
        out += x[ii];
    }
    out = sin(out);
    
    return out;
}

void Test_sin10dint(CuTest * tc)
{
    printf("Testing Function: integration of sin10d AND (de)serialization\n");
       
    size_t dim = 10;
    size_t rank[11] = {1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1};

    
    struct BoundingBox * bds = bounding_box_init(dim,0.0,1.0);
    
    enum poly_type ptype = LEGENDRE;
    struct FtApproxArgs * fapp = ft_approx_args_createpoly(dim,&ptype,NULL);
    
    struct FtCrossArgs fca;
    fca.dim = dim;
    fca.ranks = rank;
    fca.epsilon = 1e-5;
    fca.maxiter = 10;
    fca.verbose = 0;
    fca.optargs = NULL;

    double ** yr = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        yr[ii] = linspace(0.0,1.0,2);
    }

    struct CrossIndex * isl[10];
    struct CrossIndex * isr[10];
    cross_index_array_initialize(dim,isl,1,0,NULL,NULL);
    cross_index_array_initialize(dim,isr,0,1,rank,yr);

    //print_cross_index(isr[0]);
    //print_cross_index(isr[1]);
    //exit(1); 
    
    //print_index_set_array(dim,isr);
    //print_index_set_array(dim,isl);

    double coeffs[10] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 1.0};
    enum function_class fc = POLYNOMIAL;
    struct FunctionTrain * ftref = 
        function_train_linear(fc,&ptype,dim, bds, coeffs,NULL);


    struct FunctionTrain * ft = ftapprox_cross(sin10d,NULL,bds,ftref,
                                               isl, isr, &fca,fapp);
    
    unsigned char * text = NULL;
    size_t size;
    function_train_serialize(NULL,ft,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    function_train_serialize(text,ft,NULL);

    struct FunctionTrain * ftd = NULL;
    //printf("derserializing ft\n");
    function_train_deserialize(text, &ftd);

    double intval = function_train_integrate(ftd);
    
    double should = -0.62993525905472629935874873250680615583558172687;

    double relerr = fabs(intval-should)/fabs(should);
    //printf("Relative error of integrating 10 dimensional sin = %G\n",relerr);
    CuAssertDblEquals(tc,0.0,relerr,1e-12);

    for (size_t ii = 0; ii < dim; ii++){
        cross_index_free(isr[ii]); isr[ii] = NULL;
        cross_index_free(isl[ii]); isl[ii] = NULL;
    }
//    index_set_array_free(dim,isr);
//    index_set_array_free(dim,isl);
    
    free_dd(dim,yr);
    //
    bounding_box_free(bds);
    ft_approx_args_free(fapp);
    free(text);
    function_train_free(ft);
    function_train_free(ftd);
    function_train_free(ftref);
}

double sin100d(double * x, void * args){
    
    assert ( args == NULL );

    size_t ii;
    double out = 0.0;
    for (ii = 0; ii < 100; ii++){
        out += x[ii];
    }
    out = sin(out);
    
    return out;
}

void Test_sin100dint(CuTest * tc)
{
    printf("Testing Function: integration of sin100d\n");
       
    size_t dim = 100;
    struct BoundingBox * bds = bounding_box_init(dim,0.0,1.0);

    struct FunctionMonitor * fm = 
            function_monitor_initnd(sin100d,NULL,dim,10000);
    struct FunctionTrain * ft = 
        function_train_cross(function_monitor_eval,fm,bds,NULL,NULL,NULL);
    //struct FunctionTrain * ft = function_train_cross(sin100d,NULL,dim,lb,ub,NULL);
    function_monitor_free(fm);

    double intval = function_train_integrate(ft);
    
    double should = -0.00392679526107635150777939525615131307695379649361;

    double relerr = fabs(intval-should)/fabs(should);
    printf("Relative error of integrating 100 dimensional sin = %G\n",relerr);
    CuAssertDblEquals(tc,0.0,relerr,1e-10);

    function_train_free(ft);
    bounding_box_free(bds);
}

double sin1000d(double * x, void * args){
    
    assert ( args == NULL );

    size_t ii;
    double out = 0.0;
    for (ii = 0; ii < 1000; ii++){
        out += x[ii];
    }
    out = sin(out);
    
    return out;
}

/* void Test_sin1000dint(CuTest * tc) */
/* { */
/*     printf("Testing Function: integration of sin1000d\n"); */
       
/*     size_t dim = 1000; */
/*     size_t rank[1001]; */
/*     double yr[1000]; */
/*     double coeffs[1000]; */
    
/*     struct BoundingBox * bds = bounding_box_init(dim,0.0,1.0); */
/*     size_t ii; */
/*     for (ii = 0; ii < dim; ii++){ */
/*         rank[ii] = 2; */
/*         yr[ii] = 0.0; */
/*         coeffs[ii] = 1.0/ (double) dim; */
/*     } */
/*     rank[0] = 1; */
/*     rank[dim] = 1; */
    
/*     enum poly_type ptype = LEGENDRE; */
/*     struct FtApproxArgs * fapp = ft_approx_args_createpoly(dim,&ptype,NULL); */
    
/*     struct FtCrossArgs fca; */
/*     fca.dim = dim; */
/*     fca.ranks = rank; */
/*     fca.epsilon = 1e-5; */
/*     fca.maxiter = 10; */
/*     fca.verbose = 1; */
/*     fca.optargs = NULL; */

/*     struct IndexSet ** isr = index_set_array_rnested(dim, rank, yr); */
/*     struct IndexSet ** isl = index_set_array_lnested(dim, rank, yr); */
    
/*     //print_index_set_array(dim,isr); */
/*     //print_index_set_array(dim,isl); */

/*     struct FunctionTrain * ftref =function_train_linear(dim, bds, coeffs,NULL); */
/*     struct FunctionTrain * ft = ftapprox_cross(sin1000d,NULL,bds,ftref, */
/*                                     isl, isr, &fca,fapp); */

/*     double intval = function_train_integrate(ft); */
/*     double should = -2.6375125156875276773939642726964969819689605535e-19; */

/*     double relerr = fabs(intval-should)/fabs(should); */
/*     printf("Relative error of integrating 1000 dimensional sin = %G\n",relerr); */
/*     CuAssertDblEquals(tc,0.0,relerr,1e-10); */


/*     index_set_array_free(dim,isr); */
/*     index_set_array_free(dim,isl); */
/*     // */
/*     bounding_box_free(bds); */
/*     free(fapp); */
/*     function_train_free(ft); */
/*     function_train_free(ftref); */
/* } */

CuSuite * CLinalgFuncTrainGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_function_train_initsum);
    SUITE_ADD_TEST(suite, Test_function_train_linear);
    SUITE_ADD_TEST(suite, Test_function_train_quadratic);
    SUITE_ADD_TEST(suite, Test_function_train_quadratic2);
    SUITE_ADD_TEST(suite, Test_function_train_sum_function_train_round);
    SUITE_ADD_TEST(suite, Test_function_train_scale);
    SUITE_ADD_TEST(suite, Test_function_train_product);
    SUITE_ADD_TEST(suite, Test_function_train_integrate);
    SUITE_ADD_TEST(suite, Test_function_train_inner);
    
    SUITE_ADD_TEST(suite, Test_ftapprox_cross);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross2);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross3);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross4);
    SUITE_ADD_TEST(suite, Test_function_train_eval_co_peruturb);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_hermite1);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_hermite2);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_linelm1);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_linelm2);
    SUITE_ADD_TEST(suite, Test_ftapprox_cross_linelm3);
    SUITE_ADD_TEST(suite, Test_sin10dint);

    //SUITE_ADD_TEST(suite, Test_sin100dint);
    //SUITE_ADD_TEST(suite, Test_sin1000dint);
    return suite;
}

void Test_CrossIndexing(CuTest * tc)
{
   printf("Testing Function: general cross indexing functions (uncomment print statements for visual test)\n");
   size_t d = 1;
   struct CrossIndex * ci = cross_index_alloc(d);
   size_t N = 10;
   double * pts = linspace(-2.0,2.0,N);
   for (size_t ii = 0; ii < N; ii++){
       cross_index_add_index(ci,d,&(pts[ii]));
   }

   CuAssertIntEquals(tc,N,ci->n);
//   print_cross_index(ci);
   
   size_t N2 = 7;
   double * pts2 = linspace(-1.5,1.5,N);
   size_t Ntot = 14;
   int newfirst = 1;
   struct CrossIndex * ci2 = cross_index_create_nested(newfirst,0,Ntot,N2,pts2,ci);
   CuAssertIntEquals(tc,Ntot,ci2->n);
//   print_cross_index(ci2);

   struct CrossIndex * ci3 = cross_index_create_nested(newfirst,1,Ntot,N2,pts2,ci2);
   CuAssertIntEquals(tc,Ntot,ci3->n);
//   printf("\n\n\nci3\n");
//   print_cross_index(ci3);

   newfirst = 0;
   struct CrossIndex * ci4 = cross_index_create_nested(newfirst,1,Ntot,N2,pts2,ci2);
   CuAssertIntEquals(tc,Ntot,ci4->n);
//   printf("\n\n\nci4\n");
//   print_cross_index(ci4);

   size_t ind[5] = {1, 3, 0, 3, 2};
   double nx[5] = {0.2, -0.8, 0.3, -1.0, 0.2};
   struct CrossIndex * ci5 = cross_index_create_nested_ind(0,5,ind,nx,ci4);
   CuAssertIntEquals(tc,5,ci5->n);
//   print_cross_index(ci5);

   double ** vals = cross_index_merge_wspace(ci3,ci4);
//   printf("merged\n");
   for (size_t ii = 0; ii < Ntot*Ntot; ii++){
//       dprint(7,vals[ii]);
       free(vals[ii]); vals[ii] = NULL;
   }
   free(vals);
   
   cross_index_free(ci); ci = NULL;
   cross_index_free(ci2); ci2 = NULL;
   cross_index_free(ci3); ci3 = NULL;
   cross_index_free(ci4); ci4 = NULL;
   cross_index_free(ci5); ci5 = NULL;
   free(pts); pts = NULL;
   free(pts2); pts2 = NULL;

}

CuSuite * CLinalgCrossIndGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_CrossIndexing);

    return suite;
 }


double funcGrad(double * x, void * args){
    assert (args == NULL);
    double out = x[0] * x[1] + x[2] * x[3];
    return out;
}

void Test_ftapprox_grad(CuTest * tc)
{
    printf("Testing Function: function_train_gradient\n");
     
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-10.0,10.0);

    struct FunctionTrain * ft = 
        function_train_cross(funcGrad,NULL,bds,NULL,NULL,NULL);
    struct FT1DArray * ftg = function_train_gradient(ft);

    double pt[4] = {2.0, -3.1456, 1.0, 0.0};
    double * grad = ft1d_array_eval(ftg,pt);

    CuAssertDblEquals(tc, pt[1], grad[0],1e-12);
    CuAssertDblEquals(tc, pt[0], grad[1],1e-12);
    CuAssertDblEquals(tc, pt[3], grad[2],1e-12);
    CuAssertDblEquals(tc, pt[2], grad[3],1e-12);
    
    free(grad); grad = NULL;
    function_train_free(ft); ft = NULL; 
    ft1d_array_free(ftg); ftg = NULL;
    bounding_box_free(bds); bds = NULL;
}

void Test_ft1d_array_serialize(CuTest * tc)
{
    printf("Testing Function: ft1d_array_(de)serialize\n");
     
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-10.0,10.0);

    struct FunctionTrain * ft = 
        function_train_cross(funcGrad,NULL,bds,NULL,NULL,NULL);
    struct FT1DArray * ftgg = function_train_gradient(ft);


    unsigned char * text = NULL;
    size_t size;
    ft1d_array_serialize(NULL,ftgg,&size);
    //printf("Number of bytes = %zu\n", size);
    text = malloc(size * sizeof(unsigned char));
    ft1d_array_serialize(text,ftgg,NULL);

    struct FT1DArray * ftg = NULL;
    //printf("derserializing ft\n");
    ft1d_array_deserialize(text, &ftg);

    double pt[4] = {2.0, -3.1456, 1.0, 0.0};
    double * grad = ft1d_array_eval(ftg,pt);

    CuAssertDblEquals(tc, pt[1], grad[0],1e-12);
    CuAssertDblEquals(tc, pt[0], grad[1],1e-12);
    CuAssertDblEquals(tc, pt[3], grad[2],1e-12);
    CuAssertDblEquals(tc, pt[2], grad[3],1e-12);
    
    free(grad); grad = NULL;
    function_train_free(ft); ft = NULL; 
    ft1d_array_free(ftgg); ftgg = NULL;
    ft1d_array_free(ftg); ftg = NULL;
    bounding_box_free(bds); bds = NULL;
    free(text);
}


double funcHess(double * x, void * args){
    assert (args == NULL);
    //double out = 2.0*x[0] + x[1]*pow(x[2],4) + x[3]*pow(x[0],2);
    double out =  x[0] + pow(x[0],2)*x[2] +  x[1] * pow(x[2],4) ;// + x[3]*pow(x[0],2);
    return out;
}

void Test_ftapprox_hess(CuTest * tc)
{
    printf("Testing Function: function_train_hessian\n");
     
    size_t dim = 3;
    struct BoundingBox * bds = bounding_box_init(dim,-2.0,2.0);
    size_t ii,jj,kk;//,ll;
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-4;
    aopts.minsize = 1e-2;
    aopts.nregions = 5;
    aopts.pts = NULL;

    enum poly_type ptype = LEGENDRE;
    struct FtApproxArgs * fapp = 
       ft_approx_args_createpwpoly(dim,&ptype,&aopts);


    struct FunctionTrain * ft = 
        function_train_cross(funcHess,NULL,bds,NULL,NULL,fapp);
    ft_approx_args_free(fapp);
    //printf("ranks are\n");
    //iprint_sz(dim+1,ft->ranks);
    size_t N = 10;
    double * xtest = linspace(bds->lb[0],bds->ub[0],N);
    double err = 0.0;
    double den = 0.0;
    double ptt[3];
    
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            for (kk = 0; kk < N; kk++){
                //for (ll = 0; ll < N; ll++){
                    ptt[0] = xtest[ii]; ptt[1] = xtest[jj]; 
                    ptt[2] = xtest[kk]; //ptt[3] = xtest[ll];
                    den += pow(funcHess(ptt,NULL),2.0);
                    err += pow(funcHess(ptt,NULL) - 
                                    function_train_eval(ft,ptt),2.0);
                    //printf("err=%G\n",err);
               // }
            }
        }
    }

    //printf("ft ranks = \n");
    //iprint_sz(dim+1,ft->ranks);
    err = sqrt(err/den);
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc,0.0,err,1e-12);
    free(xtest); xtest = NULL;

    struct FT1DArray * fth = function_train_hessian(ft);
    double pt[3] = {1.8, -1.0, 1.0};//, 0.5};
    double * hess = ft1d_array_eval(fth,pt);
    
    //dprint2d_col(3,3,hess);
    CuAssertDblEquals(tc, 2.0*pt[2], hess[0],1e-6);
    CuAssertDblEquals(tc, 0.0, hess[1],1e-6);
    CuAssertDblEquals(tc, 2.0*pt[0], hess[2],1e-6);
    CuAssertDblEquals(tc, hess[1], hess[3],1e-8);
    CuAssertDblEquals(tc, 0.0, hess[4], 1e-6);
    CuAssertDblEquals(tc, 4.0*pow(pt[2],3), hess[5], 1e-6);
    CuAssertDblEquals(tc, hess[2], hess[6], 1e-6);
    CuAssertDblEquals(tc, hess[5], hess[7], 1e-6);
    CuAssertDblEquals(tc, 12.0*pt[1]*pow(pt[2],2), hess[8],1e-6);
    
    free(hess); hess = NULL;
    function_train_free(ft); ft = NULL; 
    ft1d_array_free(fth); fth = NULL;
    bounding_box_free(bds);bds = NULL;
}

CuSuite * CLinalgFuncTrainArrayGetSuite()
{
    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_ftapprox_grad);
    SUITE_ADD_TEST(suite,Test_ft1d_array_serialize);
    SUITE_ADD_TEST(suite, Test_ftapprox_hess);
    return suite;
}

double funcCheck2(double * x, void * args){
    assert (args == NULL);
    double out = pow(x[0] * x[1],2) + x[2] * x[3]  + x[1]*sin(x[3]);
    return out;
}

void Test_rightorth(CuTest * tc)
{
    printf("Testing Function: function_train_orthor\n");
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-10.0,10.0);

    struct FunctionTrain * ft = 
        function_train_cross(funcCheck2,NULL,bds,NULL,NULL,NULL);
    
    struct FunctionTrain * fcopy = function_train_copy(ft);
    struct FunctionTrain * ao = function_train_orthor(ft);
    
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
    bounding_box_free(bds); bds = NULL;
}

double funcCheck3(double * x, void * args){
    assert (args == NULL);
    double out = pow(x[0] * x[1],2) + x[1]*sin(x[0]);
    return out;
}

void Test_fast_mat_kron(CuTest * tc)
{

    printf("Testing Function: fast_mat_kron \n");
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;
    size_t r11 = 5;
    size_t r12 = 6;
    size_t r21 = 7;
    size_t r22 = 8;
    size_t k = 5;
    double diff; 
    
    struct Qmarray * mat1 = qmarray_poly_randu(LEGENDRE,r11,r12,
                                               maxorder,lb,ub);
    struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,
                                               maxorder,lb,ub);

    double * mat = drandu(k*r11*r21);

    struct Qmarray * mat3 = qmarray_kron(mat1,mat2);
    struct Qmarray * shouldbe = mqma(mat,mat3,k);
    struct Qmarray * is = qmarray_mat_kron(k,mat,mat1,mat2);

    diff = qmarray_norm2diff(shouldbe,is);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    free(mat); mat = NULL;
    qmarray_free(mat1); mat1 = NULL;
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(mat3); mat3 = NULL;
    qmarray_free(shouldbe); shouldbe = NULL;
    qmarray_free(is); is = NULL;
}

void Test_fast_kron_mat(CuTest * tc)
{

    printf("Testing Function: fast_kron_mat \n");
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;
    size_t r11 = 3;
    size_t r12 = 4;
    size_t r21 = 5;
    size_t r22 = 6;
    size_t k = 2;
    double diff; 
    
    struct Qmarray * mat1 = 
        qmarray_poly_randu(LEGENDRE,r11,r12,maxorder,lb,ub);
    struct Qmarray * mat2 = 
        qmarray_poly_randu(LEGENDRE,r21,r22,maxorder,lb,ub);

    double * mat = drandu(k*r12*r22);

    struct Qmarray * mat3 = qmarray_kron(mat1,mat2);
    struct Qmarray * shouldbe = qmam(mat3,mat,k);
    struct Qmarray * is = qmarray_kron_mat(k,mat,mat1,mat2);

    diff = qmarray_norm2diff(shouldbe,is);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    free(mat); mat = NULL;
    qmarray_free(mat1); mat1 = NULL;
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(mat3); mat3 = NULL;
    qmarray_free(shouldbe); shouldbe = NULL;
    qmarray_free(is); is = NULL;
}

void Test_block_kron_mat1(CuTest * tc)
{
    printf("Testing Function: block_kron_mat (1/6) \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;

    size_t nblocks = 5;
    size_t rl1[5] = {3, 6, 9, 1, 6};
    size_t rl2[5] = {2, 4, 2, 5, 3};
    size_t sum_rl1 = 0;
    size_t sum_rl2 = 0;
    struct Qmarray * mat1[5];
    

    size_t r21 = 7;
    size_t r22 = 3;
    size_t k = 8;
    double diff; 
    

    size_t ii;
    for (ii = 0; ii < nblocks; ii++){
        sum_rl1 += rl1[ii];
        sum_rl2 += rl2[ii];
        mat1[ii] = qmarray_poly_randu(LEGENDRE,rl1[ii],rl2[ii],
                                      maxorder,lb,ub);
    }
    struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,
                                               maxorder,lb,ub);
    double * mat = drandu(k*sum_rl1*r21);

    struct Qmarray * is = qmarray_alloc(k, sum_rl2* r22);
    qmarray_block_kron_mat('D',1,nblocks,mat1,mat2,k,mat,is);


    struct Qmarray * big1 = qmarray_blockdiag(mat1[0],mat1[1]);
    struct Qmarray * big2 = qmarray_blockdiag(big1,mat1[2]);
    struct Qmarray * big3 = qmarray_blockdiag(big2,mat1[3]);
    struct Qmarray * big4 = qmarray_blockdiag(big3,mat1[4]);

    struct Qmarray * mid = qmarray_kron(big4, mat2);
    struct Qmarray * shouldbe = mqma(mat,mid,k);

    struct Qmarray * is2 = qmarray_mat_kron(k,mat,big4,mat2);
    diff = qmarray_norm2diff(shouldbe,is2);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    diff = qmarray_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff,1e-10);

    qmarray_free(big1);
    qmarray_free(big2);
    qmarray_free(big3);
    qmarray_free(big4);
    qmarray_free(mid);
    qmarray_free(shouldbe);
    qmarray_free(is2);

    free(mat); mat = NULL;
    for (ii = 0; ii < nblocks; ii++){
        qmarray_free(mat1[ii]); mat1[ii] = NULL;
    }
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(is); is = NULL;

    //qmarray_free(mat3); mat3 = NULL;
    //qmarray_free(shouldbe); shouldbe = NULL;
}

void Test_block_kron_mat2(CuTest * tc)
{
    printf("Testing Function: block_kron_mat (2/6) \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;

    size_t nblocks = 5;
    size_t rl1[5] = {4, 4, 4, 4, 4};
    size_t rl2[5] = {2, 4, 2, 5, 3};
    size_t sum_rl1 = 4;
    size_t sum_rl2 = 0;
    struct Qmarray * mat1[5];
    

    size_t r21 = 7;
    size_t r22 = 3;
    size_t k = 8;
    double diff; 
    

    size_t ii;
    for (ii = 0; ii < nblocks; ii++){
        sum_rl2 += rl2[ii];
        mat1[ii] = qmarray_poly_randu(LEGENDRE,rl1[ii],rl2[ii],maxorder,lb,ub);
    }
    struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,
                                               maxorder,lb,ub);
    double * mat = drandu(k*sum_rl1*r21);

    struct Qmarray * is = qmarray_alloc(k, sum_rl2* r22);
    qmarray_block_kron_mat('R',1,nblocks,mat1,mat2,k,mat,is);


    struct Qmarray * big1 = qmarray_stackh(mat1[0],mat1[1]);
    struct Qmarray * big2 = qmarray_stackh(big1,mat1[2]);
    struct Qmarray * big3 = qmarray_stackh(big2,mat1[3]);
    struct Qmarray * big4 = qmarray_stackh(big3,mat1[4]);

    struct Qmarray * mid = qmarray_kron(big4, mat2);
    struct Qmarray * shouldbe = mqma(mat,mid,k);

    //struct Qmarray * is2 = qmarray_mat_kron(k,mat,big4,mat2);
    //diff = qmarray_norm2diff(shouldbe,is2);
    //CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    diff = qmarray_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff,1e-10);

    qmarray_free(big1);
    qmarray_free(big2);
    qmarray_free(big3);
    qmarray_free(big4);
    qmarray_free(mid);
    qmarray_free(shouldbe);
    //qmarray_free(is2);

    free(mat); mat = NULL;
    for (ii = 0; ii < nblocks; ii++){
        qmarray_free(mat1[ii]); mat1[ii] = NULL;
    }
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(is); is = NULL;

    //qmarray_free(mat3); mat3 = NULL;
    //qmarray_free(shouldbe); shouldbe = NULL;

}

void Test_block_kron_mat3(CuTest * tc)
{
    printf("Testing Function: block_kron_mat (3/6) \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;

    size_t nblocks = 5;
    size_t rl1[5] = {3, 6, 9, 1, 6};
    size_t rl2[5] = {4, 4, 4, 4, 4};
    size_t sum_rl1 = 0;
    size_t sum_rl2 = 4;
    struct Qmarray * mat1[5];
    

    size_t r21 = 7;
    size_t r22 = 3;
    size_t k = 8;
    double diff; 
    
    size_t ii;
    for (ii = 0; ii < nblocks; ii++){
        sum_rl1 += rl1[ii];
        mat1[ii] = qmarray_poly_randu(LEGENDRE,rl1[ii],rl2[ii],
                                      maxorder,lb,ub);
    }
    struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,
                                               maxorder,lb,ub);
    double * mat = drandu(k*sum_rl1*r21);

    struct Qmarray * is = qmarray_alloc(k, sum_rl2 * r22);
    qmarray_block_kron_mat('C',1,nblocks,mat1,mat2,k,mat,is);
    
    struct Qmarray * big1 = qmarray_stackv(mat1[0],mat1[1]);
    struct Qmarray * big2 = qmarray_stackv(big1,mat1[2]);
    struct Qmarray * big3 = qmarray_stackv(big2,mat1[3]);
    struct Qmarray * big4 = qmarray_stackv(big3,mat1[4]);

    struct Qmarray * mid = qmarray_kron(big4, mat2);
    struct Qmarray * shouldbe = mqma(mat,mid,k);

    diff = qmarray_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff,1e-10);

    qmarray_free(big1);
    qmarray_free(big2);
    qmarray_free(big3);
    qmarray_free(big4);
    qmarray_free(mid);
    qmarray_free(shouldbe);

    free(mat); mat = NULL;
    for (ii = 0; ii < nblocks; ii++){
        qmarray_free(mat1[ii]); mat1[ii] = NULL;
    }
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(is); is = NULL;
}

void Test_block_kron_mat4(CuTest * tc)
{
    printf("Testing Function: block_kron_mat (4/6) \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;

    size_t nblocks = 5;
    size_t rl1[5] = {3, 6, 9, 1, 6};
    size_t rl2[5] = {2, 4, 6, 5, 3};
    size_t sum_rl1 = 0;
    size_t sum_rl2 = 0;
    struct Qmarray * mat1[5];
    

    size_t r21 = 7;
    size_t r22 = 3;
    size_t k = 8;
    double diff; 
    
    size_t ii;
    for (ii = 0; ii < nblocks; ii++){
        sum_rl1 += rl1[ii];
        sum_rl2 += rl2[ii];
        mat1[ii] = qmarray_poly_randu(LEGENDRE,rl1[ii],rl2[ii],
                                      maxorder,lb,ub);
    }
    struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,
                                               maxorder,lb,ub);
    double * mat = drandu(k*sum_rl2*r22);

    struct Qmarray * is = qmarray_alloc(sum_rl1 * r21,k);
    qmarray_block_kron_mat('D',0,nblocks,mat1,mat2,k,mat,is);


    struct Qmarray * big1 = qmarray_blockdiag(mat1[0],mat1[1]);
    struct Qmarray * big2 = qmarray_blockdiag(big1,mat1[2]);
    struct Qmarray * big3 = qmarray_blockdiag(big2,mat1[3]);
    struct Qmarray * big4 = qmarray_blockdiag(big3,mat1[4]);

    struct Qmarray * mid = qmarray_kron(big4, mat2);
    struct Qmarray * shouldbe = qmam(mid,mat,k);

    struct Qmarray * is2 = qmarray_kron_mat(k,mat,big4,mat2);
    diff = qmarray_norm2diff(shouldbe,is2);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    diff = qmarray_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff,1e-10);

    qmarray_free(big1);
    qmarray_free(big2);
    qmarray_free(big3);
    qmarray_free(big4);
    qmarray_free(mid);
    qmarray_free(shouldbe);
    qmarray_free(is2);

    free(mat); mat = NULL;
    for (ii = 0; ii < nblocks; ii++){
        qmarray_free(mat1[ii]); mat1[ii] = NULL;
    }
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(is); is = NULL;

    //qmarray_free(mat3); mat3 = NULL;
    //qmarray_free(shouldbe); shouldbe = NULL;

}


void Test_block_kron_mat5(CuTest * tc)
{
    printf("Testing Function: block_kron_mat (5/6) \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;

    size_t nblocks = 5;
    size_t rl1[5] = {2, 2, 2, 2, 2};
    size_t rl2[5] = {2, 4, 2, 5, 3};
    size_t sum_rl1 = 2;
    size_t sum_rl2 = 0;
    struct Qmarray * mat1[5];
    

    size_t r21 = 7;
    size_t r22 = 3;
    size_t k = 8;
    double diff; 
    
    size_t ii;
    for (ii = 0; ii < nblocks; ii++){
        sum_rl2 += rl2[ii];
        mat1[ii] = qmarray_poly_randu(LEGENDRE,rl1[ii],rl2[ii],
                                      maxorder,lb,ub);
    }
    struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,
                                               maxorder,lb,ub);
    double * mat = drandu(k*sum_rl2*r22);

    struct Qmarray * is = qmarray_alloc(sum_rl1 * r21,k);
    qmarray_block_kron_mat('R',0,nblocks,mat1,mat2,k,mat,is);

    struct Qmarray * big1 = qmarray_stackh(mat1[0],mat1[1]);
    struct Qmarray * big2 = qmarray_stackh(big1,mat1[2]);
    struct Qmarray * big3 = qmarray_stackh(big2,mat1[3]);
    struct Qmarray * big4 = qmarray_stackh(big3,mat1[4]);

    struct Qmarray * mid = qmarray_kron(big4, mat2);
    struct Qmarray * shouldbe = qmam(mid,mat,k);

    struct Qmarray * is2 = qmarray_kron_mat(k,mat,big4,mat2);
    diff = qmarray_norm2diff(shouldbe,is2);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    diff = qmarray_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff,1e-10);

    qmarray_free(big1);
    qmarray_free(big2);
    qmarray_free(big3);
    qmarray_free(big4);
    qmarray_free(mid);
    qmarray_free(shouldbe);
    qmarray_free(is2);

    free(mat); mat = NULL;
    for (ii = 0; ii < nblocks; ii++){
        qmarray_free(mat1[ii]); mat1[ii] = NULL;
    }
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(is); is = NULL;

    //qmarray_free(mat3); mat3 = NULL;
    //qmarray_free(shouldbe); shouldbe = NULL;

}

void Test_block_kron_mat6(CuTest * tc)
{
    printf("Testing Function: block_kron_mat (6/6) \n");

    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 10;

    size_t nblocks = 5;
    size_t rl1[5] = {3, 6, 9, 1, 6};
    size_t rl2[5] = {2, 2, 2, 2, 2};
    size_t sum_rl1 = 0;
    size_t sum_rl2 = 2;
    struct Qmarray * mat1[5];
    

    size_t r21 = 7;
    size_t r22 = 3;
    size_t k = 8;
    double diff; 
    
    size_t ii;
    for (ii = 0; ii < nblocks; ii++){
        sum_rl1 += rl1[ii];
        mat1[ii] = qmarray_poly_randu(LEGENDRE,rl1[ii],rl2[ii],
                                      maxorder,lb,ub);
    }
    struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,
                                               maxorder,lb,ub);
    double * mat = drandu(k*sum_rl2*r22);

    struct Qmarray * is = qmarray_alloc(sum_rl1 * r21,k);
    qmarray_block_kron_mat('C',0,nblocks,mat1,mat2,k,mat,is);


    struct Qmarray * big1 = qmarray_stackv(mat1[0],mat1[1]);
    struct Qmarray * big2 = qmarray_stackv(big1,mat1[2]);
    struct Qmarray * big3 = qmarray_stackv(big2,mat1[3]);
    struct Qmarray * big4 = qmarray_stackv(big3,mat1[4]);

    struct Qmarray * mid = qmarray_kron(big4, mat2);
    struct Qmarray * shouldbe = qmam(mid,mat,k);

    struct Qmarray * is2 = qmarray_kron_mat(k,mat,big4,mat2);
    diff = qmarray_norm2diff(shouldbe,is2);
    CuAssertDblEquals(tc,0.0,diff,1e-10);
    
    diff = qmarray_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff,1e-10);

    qmarray_free(big1);
    qmarray_free(big2);
    qmarray_free(big3);
    qmarray_free(big4);
    qmarray_free(mid);
    qmarray_free(shouldbe);
    qmarray_free(is2);

    free(mat); mat = NULL;
    for (ii = 0; ii < nblocks; ii++){
        qmarray_free(mat1[ii]); mat1[ii] = NULL;
    }
    qmarray_free(mat2); mat2 = NULL;
    qmarray_free(is); is = NULL;

    //qmarray_free(mat3); mat3 = NULL;
    //qmarray_free(shouldbe); shouldbe = NULL;

}

void Test_dmrg_prod(CuTest * tc)
{

    printf("Testing Function: dmrg_prod\n");
    size_t dim = 4;
    struct BoundingBox * bds = bounding_box_init(dim,-1.0,1.0);
    struct FunctionTrain * a = function_train_cross(funcCheck2,NULL,bds,NULL,NULL,NULL);
    struct FunctionTrain * ft = function_train_product(a,a);

    struct FunctionTrain * fcopy = function_train_copy(ft);
    struct FunctionTrain * rounded = function_train_round(ft,1e-12);

    enum function_class fc = POLYNOMIAL;
    enum poly_type ptype = LEGENDRE;
    struct FunctionTrain * start = 
        function_train_constant(fc,&ptype,dim,1.0,bds,NULL);
    struct FunctionTrain * finish = dmrg_product(start,a,a,1e-10,10,1e-12,0);

    double diff = function_train_relnorm2diff(finish,fcopy);
    CuAssertDblEquals(tc,0.0,diff*diff,1e-14);
    
    bounding_box_free(bds); bds = NULL;
    function_train_free(a); a = NULL;
    function_train_free(ft); ft = NULL;
    function_train_free(fcopy); fcopy = NULL;
    function_train_free(start); start = NULL;
    function_train_free(finish); finish = NULL;
    function_train_free(rounded); rounded = NULL;
}

CuSuite * CLinalgDMRGGetSuite()
{
    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_rightorth);
    SUITE_ADD_TEST(suite,Test_fast_mat_kron);
    SUITE_ADD_TEST(suite,Test_fast_kron_mat);
    SUITE_ADD_TEST(suite,Test_block_kron_mat1);
    SUITE_ADD_TEST(suite,Test_block_kron_mat2);
    SUITE_ADD_TEST(suite,Test_block_kron_mat3);
    SUITE_ADD_TEST(suite,Test_block_kron_mat4);
    SUITE_ADD_TEST(suite,Test_block_kron_mat5);
    SUITE_ADD_TEST(suite,Test_block_kron_mat6);
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

    struct Qmarray * zer = qmarray_zeros(LEGENDRE,af->nrows,
                                         af->ncols,lb,ub);
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

    struct Qmarray * zer = qmarray_zeros(LEGENDRE,af->nrows,af->ncols,lb,ub);
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
    size_t ranks[5] = {1,2,2,2,1};
    size_t maxorder = 10;
    double lb = -1.0;
    double ub = 1.0;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);

    struct FunctionTrain * f = 
        function_train_cross(funcCheck2,NULL,bds,NULL,NULL,NULL);
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
            struct Qmarray * zer = qmarray_zeros(LEGENDRE,af->nrows,af->ncols,
                                                 lb,ub);
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
}

void Test_diffusion_dmrg(CuTest * tc)
{
    
    printf("Testing Function: dmrg_diffusion\n");
    size_t dim = 4;
    size_t ranks[5] = {1,2,2,2,1};
    size_t maxorder = 10;
    double lb = -1.0;
    double ub = 1.0;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);

    struct FunctionTrain * f = function_train_cross(funcCheck2,NULL,bds,NULL,NULL,NULL);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,
                                                         maxorder);

    struct FunctionTrain * is = dmrg_diffusion(a,f,1e-5,5,1e-10,0);
    struct FunctionTrain * shouldbe = exact_diffusion(a,f);
    
    double diff = function_train_norm2diff(is,shouldbe);
    CuAssertDblEquals(tc,0.0,diff*diff,1e-10);

    function_train_free(shouldbe); shouldbe = NULL;
    function_train_free(is); is = NULL;
    bounding_box_free(bds); bds = NULL;
    function_train_free(f); f = NULL;
    function_train_free(a); a = NULL;
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


