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
#include <string.h>
#include <assert.h>
#include <float.h>

#include "CuTest.h"
#include "testfunctions.h"

#include "array.h"
#include "lib_linalg.h"

#include "lib_funcs.h"


void Test_gauss_eval(CuTest * tc){
    
    printf("Testing functions: gauss_kernel_eval(deriv) \n");

    double scale = 1.2;
    double width = 0.2;
    double center = -0.3;

    double h = 1e-8;
    double x = 0.2;
    double xh = x+h;
    double valh = gauss_kernel_eval(scale,width*width,center,xh);
    double x2h = x-h;
    double val2h = gauss_kernel_eval(scale,width*width,center,x2h);

    double numerical_deriv = (valh-val2h)/(2.0*h);
    double analytical_deriv = gauss_kernel_deriv(scale,width*width,center,x);
    CuAssertDblEquals(tc,numerical_deriv,analytical_deriv,1e-5);
}

void Test_gauss_integrate(CuTest * tc){
    
    printf("Testing functions: gauss_kernel_integrate\n");

    double scale = 1.2;
    double width = 0.2;
    double center = -0.3;

    double lb = -1;
    double ub = 1;
    size_t N = 100;
    double * x = linspace(-1,1,N);
    double numerical_int = 0.0;
    for (size_t ii = 0; ii < N-1; ii++){
        numerical_int += (x[ii+1]-x[ii])*gauss_kernel_eval(scale,width*width,center,x[ii]);
    }
    free(x); x = NULL;
    
    double analytical_int = gauss_kernel_integrate(scale,width*width,center,lb,ub);
    /* printf("%3.15G,%3.15G\n",numerical_int,analytical_int); */
    CuAssertDblEquals(tc,numerical_int,analytical_int,1e-5);
}

void Test_gauss_inner(CuTest * tc){
    
    printf("Testing functions: gauss_kernel_inner\n");

    double s1 = 1.2;
    double w1 = 0.2;
    double c1 = -0.3;
    double s2 = 0.3;
    double w2 = 0.5;
    double c2 = -0.1;

    double lb = -10;
    double ub = 10;
    size_t N = 10000;
    double * x = linspace(lb,ub,N);
    double numerical_int = 0.0;
    for (size_t ii = 0; ii < N-1; ii++){
        double v1 = gauss_kernel_eval(s1,w1*w1,c1,x[ii]);
        double v2 = gauss_kernel_eval(s2,w2*w2,c2,x[ii]);
        numerical_int += (x[ii+1]-x[ii])*v1*v2;
    }
    free(x); x= NULL;
    double analytical_int = gauss_kernel_inner(s1,w1*w1,c1,s2,w2*w2,c2,-DBL_MAX,DBL_MAX);
    /* printf("%3.15G,%3.15G\n",numerical_int,analytical_int); */
    CuAssertDblEquals(tc,numerical_int,analytical_int,1e-5);
}


void Test_kernel_expansion_mem(CuTest * tc)
{

    printf("Testing functions: kernel allocation and deallocation memory \n");
    
    double scale = 1.2;
    double width = 0.2;
    size_t N = 20;
    double * centers = linspace(-1,1,N);

    struct KernelExpansion * ke = kernel_expansion_alloc(1);
    CuAssertIntEquals(tc,1,ke!=NULL);
    for (size_t ii = 0; ii < N; ii++){
        struct Kernel * kern = kernel_gaussian(scale,width*width,centers[ii]);
        kernel_expansion_add_kernel(ke,0.0,kern);
        kernel_free(kern); kern = NULL;
    }
    
    kernel_expansion_free(ke); ke = NULL;

    free(centers); centers = NULL;
}

void Test_kernel_expansion_copy(CuTest * tc)
{

    printf("Testing functions: kernel_expansion_copy \n");
    
    double scale = 1.2;
    double width = 0.2;
    size_t N = 20;
    double * centers = linspace(-1,1,N);

    struct KernelExpansion * ke = kernel_expansion_alloc(1);
    for (size_t ii = 0; ii < N; ii++){
        struct Kernel * kern = kernel_gaussian(scale,width*width,centers[ii]);
        kernel_expansion_add_kernel(ke,randu(),kern);
        kernel_free(kern); kern = NULL;
    }

    struct KernelExpansion * ke2 = kernel_expansion_copy(ke);

    double * x = linspace(-1,1,200);
    for (size_t ii = 0; ii < 200; ii++){
        double val1 = kernel_expansion_eval(ke,x[ii]);
        double val2 = kernel_expansion_eval(ke2,x[ii]);
        CuAssertDblEquals(tc,val1,val2,1e-15);
    }

    free(x); x = NULL;
    
    kernel_expansion_free(ke2); ke2 = NULL;
    kernel_expansion_free(ke); ke = NULL;

    free(centers); centers = NULL;
}

void Test_serialize_kernel_expansion(CuTest * tc){
    
    printf("Testing functions: serialize_kernel_expansion \n");
    double scale = 1.2;
    double width = 0.2;
    size_t N = 18;
    double * centers = linspace(-1,1,N);

    struct KernelExpansion * ke = kernel_expansion_alloc(1);
    for (size_t ii = 0; ii < N; ii++){
        struct Kernel * kern = kernel_gaussian(scale,width*width,centers[ii]);
        kernel_expansion_add_kernel(ke,randu(),kern);
        kernel_free(kern); kern = NULL;
    }

    unsigned char * text = NULL;
    size_t size_to_be;
    serialize_kernel_expansion(text, ke, &size_to_be);
    text = malloc(size_to_be * sizeof(char));

    serialize_kernel_expansion(text, ke, NULL);

    struct KernelExpansion * k2 = NULL;
    deserialize_kernel_expansion(text, &k2);
    free(text); text = NULL;
    
    CuAssertIntEquals(tc,N,kernel_expansion_get_nkernels(k2));
    
    double * x = linspace(-1,1,200);
    for (size_t ii = 0; ii < 200; ii++){
        double val1 = kernel_expansion_eval(ke,x[ii]);
        double val2 = kernel_expansion_eval(k2,x[ii]);
        /* printf("%G, %G\n",val1,val2); */
        CuAssertDblEquals(tc,val1,val2,1e-15);
    }
    free(x); x = NULL;
    
    kernel_expansion_free(k2); k2 = NULL;
    kernel_expansion_free(ke); ke = NULL;

    free(centers); centers = NULL;
}

void Test_serialize_generic_function_kernel (CuTest * tc){
    
    printf("Testing functions: (de)serializing generic_function with kernels\n");
    double scale = 1.2;
    double width = 0.2;
    size_t N = 18;
    double lb = -1.0, ub = 2.0;
    double * centers = linspace(lb,ub,N);

    struct KernelExpansion * ke = kernel_expansion_alloc(1);
    for (size_t ii = 0; ii < N; ii++){
        struct Kernel * kern = kernel_gaussian(scale,width*width,centers[ii]);
        kernel_expansion_add_kernel(ke,randu(),kern);
        kernel_free(kern); kern = NULL;
    }
    kernel_expansion_set_bounds(ke,lb,ub);
    
    struct GenericFunction * pl = generic_function_alloc(1,KERNEL);
    pl->f = ke;
     
    unsigned char * text = NULL;
    size_t size_to_be;
    serialize_generic_function(text, pl, &size_to_be);
    text = malloc(size_to_be * sizeof(char));

    serialize_generic_function(text, pl, NULL);
    
    struct GenericFunction * pt = NULL;
    deserialize_generic_function(text, &pt);

    double * xtest = linspace(lb,ub,1000);
    size_t ii;
    double err = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(pl,xtest[ii]) -
                   generic_function_1d_eval(pt,xtest[ii]),2);
    }
    err = sqrt(err);
    CuAssertDblEquals(tc, 0.0, err, 1e-15);
    free(xtest);
    free(text);

    generic_function_free(pl);
    generic_function_free(pt);
    free(centers); centers = NULL;
}

void Test_kernel_expansion_create_with_params_and_grad(CuTest * tc){
    
    printf("Testing functions: kernel_expansion_create_with_params and grad\n");

    double scale = 1.2;
    double width = 0.2;
    size_t N = 18;
    double * centers = linspace(-1,1,N);
    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(N,centers,scale,width);

    double * params = calloc_double(N);
    for (size_t ii = 0; ii < N; ii++) params[ii]=0.1*randn();

    struct KernelExpansion * ke = kernel_expansion_create_with_params(opts,N,params);

    double grad[36];
    double xloc[2] = {0.4, 0.8};
    int res = kernel_expansion_param_grad_eval(ke,2,xloc,grad);
    CuAssertIntEquals(tc,0,res);


    // numerical derivative
    struct KernelExpansion * ke1 = NULL;
    struct KernelExpansion * ke2 = NULL;

    size_t dim = N;
    double * x1 = calloc_double(dim);
    double * x2 = calloc_double(dim);
    for (size_t ii = 0; ii < dim; ii++){
        x1[ii] = params[ii];
        x2[ii] = params[ii];
    }
    
    double diff = 0.0;
    double v1,v2;
    double norm = 0.0;
    double eps = 1e-8;
    for (size_t zz = 0; zz < 2; zz++){
        /* size_t zz = 0; */
        for (size_t ii = 0; ii < dim; ii++){
            x1[ii] += eps;
            x2[ii] -= eps;
            ke1 = kernel_expansion_create_with_params(opts,dim,x1);
            v1 = kernel_expansion_eval(ke1,xloc[zz]);

            ke2 = kernel_expansion_create_with_params(opts,dim,x2);
            v2 = kernel_expansion_eval(ke2,xloc[zz]);

            double diff_iter = pow( (v1-v2)/2.0/eps - grad[zz*dim+ii], 2 );
            /* printf("current diff = %G\n",diff_iter); */
            /* printf("\t norm = %G\n",grad[zz*dim+ii]); */
            diff += diff_iter;
            norm += pow( (v1-v2)/2.0/eps,2);
        
            x1[ii] -= eps;
            x2[ii] += eps;

            kernel_expansion_free(ke1); ke1 = NULL;
            kernel_expansion_free(ke2); ke2 = NULL;
        }
        if (norm > 1){
            diff /= norm;
        }
        CuAssertDblEquals(tc,0.0,diff,1e-7);
    }
    free(x1); x1 = NULL;
    free(x2); x2 = NULL;

    kernel_approx_opts_free(opts); opts = NULL;
    free(centers); centers = NULL;
    free(params); params = NULL;
    kernel_expansion_free(ke); ke = NULL;
}

void Test_kernel_expansion_integrate(CuTest * tc)
{

    printf("Testing functions: kernel_expansion_integrate \n");
    
    double scale = 1.2;
    double width = 0.2;
    size_t N = 20;
    double * centers = linspace(-1,1,N);

    struct KernelExpansion * ke = kernel_expansion_alloc(1);
    for (size_t ii = 0; ii < N; ii++){
        struct Kernel * kern = kernel_gaussian(scale,width*width,centers[ii]);
        kernel_expansion_add_kernel(ke,randu(),kern);
        kernel_free(kern); kern = NULL;
    }


    size_t nint = 10000;
    double * x = linspace(-20,20,nint);
    double num_int = 0.0;
    for (size_t ii = 0; ii < nint-1; ii++){
        double val1 = kernel_expansion_eval(ke,x[ii]);
        num_int += val1*(x[ii+1]-x[ii]);
    }
    free(x); x = NULL;

    double ana_int = kernel_expansion_integrate(ke);

    CuAssertDblEquals(tc,num_int,ana_int,1e-5);

    kernel_expansion_free(ke); ke = NULL;
    free(centers); centers = NULL;
}

void Test_kernel_expansion_inner(CuTest * tc)
{

    printf("Testing functions: kernel_expansion_inner\n");
    
    double s1 = 1.2;
    double w1 = 0.2;
    size_t N1 = 20;
    double * c1 = linspace(-1,1,N1);
    struct KernelExpansion * ke = kernel_expansion_alloc(1);
    for (size_t ii = 0; ii < N1; ii++){
        struct Kernel * kern = kernel_gaussian(s1,w1*w1,c1[ii]);
        kernel_expansion_add_kernel(ke,randu(),kern);
        kernel_free(kern); kern = NULL;
    }

    double s2 = 3.2;
    double w2 = 0.8;
    size_t N2 = 14;
    double * c2 = linspace(-3,0.2,N2);
    struct KernelExpansion * ke2 = kernel_expansion_alloc(1);
    for (size_t ii = 0; ii < N2; ii++){
        struct Kernel * kern = kernel_gaussian(s2,w2*w2,c2[ii]);
        kernel_expansion_add_kernel(ke2,randu(),kern);
        kernel_free(kern); kern = NULL;
    }

    double ana_int = kernel_expansion_inner(ke,ke2);

    size_t nint = 500000;
    double * x = linspace(-20,20,nint);
    double num_int = 0.0;
    for (size_t ii = 0; ii < nint-1; ii++){
        double val1 = kernel_expansion_eval(ke,x[ii]);
        double val2 = kernel_expansion_eval(ke2,x[ii]);
        num_int += val1*val2*(x[ii+1]-x[ii]);
    }
    free(x); x = NULL;

    /* printf("%3.15G,%3.15G,%3.15G\n",num_int,ana_int,num_int-ana_int); */
    CuAssertDblEquals(tc,num_int,ana_int,1e-13);
    
    kernel_expansion_free(ke); ke = NULL;
    kernel_expansion_free(ke2); ke2 = NULL;

    free(c1); c1 = NULL;
    free(c2); c2 = NULL;
}

void Test_kernel_expansion_orth_basis(CuTest * tc)
{

    printf("Testing functions: kernel_expansion_orth_basis\n");
        
    double scale = 1.2;

    size_t N = 30;
    double width = pow((double) N, -0.2) * 1.0/12.0 * 2;
    /* width *= 0.1; */
    /* printf("width = %G\n",width); */
    double * centers = linspace(-1,1,N);
    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(N,centers,scale,width);
    kernel_approx_opts_set_lb(opts,-10.0);
    kernel_approx_opts_set_ub(opts,10.0);
    
    size_t north = 30;    
    struct KernelExpansion * ke[200];
    for (size_t ii = 0; ii < north; ii++){
        ke[ii] = NULL;
    }

    kernel_expansion_orth_basis(north,ke,opts);
    for (size_t ii = 0; ii < north; ii++){
        /* printf("\n"); */
        /* printf("ii = %zu\n",ii); */
        for (size_t jj = 0; jj < north; jj++){
            double val = kernel_expansion_inner(ke[ii],ke[jj]);
            if (ii == jj){
                /* printf("should be 1: %3.15G\n",val); */
                CuAssertDblEquals(tc,1.0,val,1e-12);
            }
            else{
                /* printf("should be 0: %3.15G\n",val); */
                CuAssertDblEquals(tc,0.0,val,1e-13);
            }
        }
    }
    
    
    for (size_t ii = 0; ii < north; ii++){
        kernel_expansion_free(ke[ii]); ke[ii] = NULL;
    }

    kernel_approx_opts_free(opts); opts = NULL;
    free(centers); centers = NULL;
}

static void regress_func(size_t N, const double * x, double * out)
{
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = -1.0 * sin(2.0*x[ii]);
    }
}

void Test_kernel_LSregress(CuTest * tc){
    
    printf("Testing functions: least squares regression with kernel expansion\n");

    // create data
    size_t ndata = 200;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise
    double * params = calloc_double(ndata);
    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
        params[ii] = y[ii];
    }


    
    size_t nparams = ndata;
    double scale = 0.1;
    double width = pow(ndata,-0.2)/12.0;
    width *= 5.0;
    /* printf("width = %G\n",width); */
    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,x,scale,width);
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    c3opt_set_maxiter(optimizer,2000);
    c3opt_set_gtol(optimizer,1e-6);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,LS,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,KERNEL,opts);
    regress_1d_opts_set_initial_parameters(regopts,params);

    // check derivative
    c3opt_add_objective(optimizer,param_LSregress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    for (size_t ii = 0; ii < nparams; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        /* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); */
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
    
    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    /* CuAssertIntEquals(tc,1,info>-1); */

    /* print_generic_function(gf,5,NULL); */
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-2);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    kernel_approx_opts_free(opts); opts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
    free(params); params = NULL;
}

void Test_kernel_LSregress2(CuTest * tc){
    
    printf("Testing functions: least squares regression with kernel expansion (2)\n");

    // create data
    size_t ndata = 200;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise

    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }

    size_t nparams = 40;
    double * params = linspace(-1.0,1.0,nparams);


    double scale = 0.1;
    double width = pow(nparams,-0.2)/12.0;
    width *= 5.0;
    /* printf("width = %G\n",width); */
    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,params,scale,width);
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    c3opt_set_maxiter(optimizer,2000);
    c3opt_set_gtol(optimizer,1e-6);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,LS,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,KERNEL,opts);
    regress_1d_opts_set_initial_parameters(regopts,params);

    // check derivative
    c3opt_add_objective(optimizer,param_LSregress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    for (size_t ii = 0; ii < nparams; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        /* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); */
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
    
    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    /* CuAssertIntEquals(tc,1,info>-1); */

    /* print_generic_function(gf,5,NULL); */
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-2);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    kernel_approx_opts_free(opts); opts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
    free(params); params = NULL;
}

void Test_kernel_LSregress_with_centers(CuTest * tc){
    
    printf("Testing functions: least squares regression with kernel expansion and moving centers\n");

    // create data
    size_t ndata = 200;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise

    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }

    size_t nparams = 2*ndata;
    double * params = calloc_double(2*ndata);
    memmove(params,y,ndata*sizeof(double));
    memmove(params+ndata,x,ndata*sizeof(double));

    double scale = 0.1;
    double width = pow(ndata,-0.2)/12.0;
    width *= 5.0;
    /* printf("width = %G\n",width); */
    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams/2,x,scale,width);
    kernel_approx_opts_set_center_adapt(opts,1);
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    c3opt_set_maxiter(optimizer,2000);
    c3opt_set_gtol(optimizer,1e-6);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,LS,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,KERNEL,opts);
    regress_1d_opts_set_initial_parameters(regopts,params);

    // check derivative
    c3opt_add_objective(optimizer,param_LSregress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    for (size_t ii = 0; ii < nparams; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        /* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); */
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
    
    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    /* CuAssertIntEquals(tc,1,info>-1); */

    /* print_generic_function(gf,5,NULL); */
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-2);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    kernel_approx_opts_free(opts); opts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
    free(params); params = NULL;
}


void Test_kernel_LSregress_with_centers2(CuTest * tc){
    
    printf("Testing functions: least squares regression with kernel expansion and moving centers (2)\n");

    // create data
    size_t ndata = 200;
    double * x = linspace(-1,1,ndata);
    double * y = calloc_double(ndata);
    regress_func(ndata,x,y);
    // // add noise

    for (size_t ii =0 ; ii < ndata; ii++){
        y[ii] += randn()*0.01;
    }

    size_t nparams = 40;
    double * params = calloc_double(nparams);
    for (size_t ii = 0; ii < nparams; ii++){
        params[ii] = randu()*2.0-1.0;
    }

    double scale = 0.1;
    double width = pow(nparams,-0.2)/12.0;
    width *= 5.0;
    /* printf("width = %G\n",width); */
    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams/2,params+nparams/2,scale,width);
    kernel_approx_opts_set_center_adapt(opts,1);
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,nparams);
    c3opt_set_verbose(optimizer,0);
    c3opt_set_maxiter(optimizer,2000);
    c3opt_set_gtol(optimizer,1e-6);
    
    struct Regress1DOpts * regopts = regress_1d_opts_create(PARAMETRIC,LS,ndata,x,y);
    regress_1d_opts_set_parametric_form(regopts,KERNEL,opts);
    regress_1d_opts_set_initial_parameters(regopts,params);

    // check derivative
    c3opt_add_objective(optimizer,param_LSregress_cost,regopts);
    double * deriv_diff = calloc_double(nparams);
    double gerr = c3opt_check_deriv_each(optimizer,params,1e-8,deriv_diff);
    for (size_t ii = 0; ii < nparams; ii++){
        /* printf("ii = %zu, diff=%G\n",ii,deriv_diff[ii]); */
        /* CuAssertDblEquals(tc,0.0,deriv_diff[ii],1e-3); */
    }
    CuAssertDblEquals(tc,0.0,gerr,1e-3);
    free(deriv_diff); deriv_diff = NULL;
    
    int info;
    struct GenericFunction * gf = generic_function_regress1d(regopts,optimizer,&info);
    /* CuAssertIntEquals(tc,1,info>-1); */

    /* print_generic_function(gf,5,NULL); */
    
    double * xtest = linspace(-1.0,1.0,1000);
    double * vals = calloc_double(1000);
    regress_func(1000,xtest,vals);
    size_t ii;
    double err = 0.0;
    double norm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(generic_function_1d_eval(gf,xtest[ii]) - vals[ii],2);
        norm += vals[ii]*vals[ii];
    }
    err = sqrt(err);
    norm = sqrt(norm);
    double rat = err/norm;
    printf("\t error = %G, norm=%G, rat=%G\n",err,norm,rat);
    CuAssertDblEquals(tc, 0.0, rat, 1e-2);
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;

    free(x); x = NULL;
    free(y); y = NULL;
    kernel_approx_opts_free(opts); opts = NULL;
    regress_1d_opts_destroy(regopts); regopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    generic_function_free(gf); gf = NULL;;
    free(params); params = NULL;
}

void Test_kernel_linear(CuTest * tc){

    printf("Testing function: kernel_expansion_linear\n");


    double lb = -1.0;
    double ub = 1.0;
    size_t nparams = 10;
    double * x = linspace(lb,ub,nparams);
    double scale = 1.0;
    double width = pow(nparams,-0.2)/12.0 * (ub-lb);
    width *= 20.0;
    /* printf("width = %G\n",width); */
    struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,x,scale,width);

    double a = 2.0, offset=3.0;
    
    struct KernelExpansion * ke = kernel_expansion_linear(a,offset,opts);
    
    size_t N = 100;
    double * pts = linspace(lb,ub,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = kernel_expansion_eval(ke,pts[ii]);
        double eval2 = a*pts[ii] + offset;
        double diff= fabs(eval1-eval2);
        /* printf("eval1 = %G, eval2=%G, diff=%G\n",eval1,eval2,diff); */
        CuAssertDblEquals(tc, 0.0, diff, 1e-5);
    }
    free(pts); pts = NULL;

    /* print_kernel_expansion(ke,5,NULL); */
    
    kernel_expansion_free(ke); ke = NULL;
    free(x); x = NULL;
    kernel_approx_opts_free(opts); opts = NULL;
}


CuSuite * KernGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_gauss_eval);
    SUITE_ADD_TEST(suite, Test_gauss_integrate);
    SUITE_ADD_TEST(suite, Test_gauss_inner);
    SUITE_ADD_TEST(suite, Test_kernel_expansion_mem);
    SUITE_ADD_TEST(suite, Test_kernel_expansion_copy);
    SUITE_ADD_TEST(suite, Test_serialize_kernel_expansion);
    SUITE_ADD_TEST(suite, Test_serialize_generic_function_kernel);
    SUITE_ADD_TEST(suite, Test_kernel_expansion_create_with_params_and_grad);
    SUITE_ADD_TEST(suite, Test_kernel_expansion_integrate);
    SUITE_ADD_TEST(suite, Test_kernel_expansion_inner);
    SUITE_ADD_TEST(suite, Test_kernel_expansion_orth_basis);
    SUITE_ADD_TEST(suite, Test_kernel_LSregress);
    SUITE_ADD_TEST(suite, Test_kernel_LSregress2);
    SUITE_ADD_TEST(suite, Test_kernel_LSregress_with_centers);
    SUITE_ADD_TEST(suite, Test_kernel_LSregress_with_centers2);
    SUITE_ADD_TEST(suite, Test_kernel_linear);
    return suite;
}
