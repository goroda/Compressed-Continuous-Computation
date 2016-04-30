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
#include <string.h>
#include <assert.h>
#include <float.h>

#include "array.h"
#include "lib_linalg.h"
#include "CuTest.h"

#include "lib_funcs.h"

void Test_lin_elem_exp_approx(CuTest * tc){

    printf("Testing function: lin_elem_exp_init\n");
    size_t N = 50;
    double * x = linspace(-1.0,1.0,N);
    double f[50];
    
    struct counter c;
    c.N = 0;

    for (size_t ii = 0; ii < N; ii++){
        f[ii] = func(x[ii],&c);
    }

//    printf("go\n");
    struct LinElemExp * fa = lin_elem_exp_init(N,x,f);

    double * xtest = linspace(-1,1,1000);
    size_t ii;
    double err = 0.0;
    double errNorm = 0.0;
//    printf("went\n");
    for (ii = 0; ii < 1000; ii++){

        double eval1 = lin_elem_exp_eval(fa,xtest[ii]);
        double evalt = func(xtest[ii],&c);
        //printf("ii = %zu, err=%G\n",ii,evalt-eval1);
        err += pow(eval1-evalt,2);
        errNorm += pow(evalt,2);
    }
    err = sqrt(err) / errNorm;
//    printf("err = %G\n",err);
    CuAssertDblEquals(tc, 0.0, err, 1e-3);

    free(x);
    lin_elem_exp_free(fa);
    free(xtest);
}

double fleadapt(double x, void * args)
{
    struct counter * c = args;
    c->N = c->N + 1;
    double out = 1.0 / (pow(x - 0.3,2) + 0.01);
    out += 1.0/(pow(x-0.9,2) + 0.04);
    return out;
}

void Test_lin_elem_exp_adapt(CuTest * tc){

    printf("Testing function: lin_elem_adapt\n");

    struct counter c;
    c.N = 0;
    
    double xl = 0.0;
    double fl = fleadapt(xl,&c);
    double xr = 3.0;
    double fr = fleadapt(xr,&c);
    
    double delta = 5e-2;
    double hmin = 1e-3;
    
    struct LinElemXY * xy = NULL;
    lin_elem_adapt(fleadapt,&c,xl,fl,xr,fr,delta,hmin,&xy);
    struct LinElemXY * temp = xy;
    size_t count = 0;
//    printf("lets go!\n");
    while (temp != NULL){
        /* double x = lin_elem_xy_get_x(temp); */
        /* double y = lin_elem_xy_get_y(temp); */
//        printf("(x,y) = (%G,%G)\n",x,y);
        count ++;
        temp = lin_elem_xy_next(temp);
    }
    /* printf("count1 = %zu\n",count); */
    /* printf("count2 = %zu\n",c.N); */
    
    CuAssertIntEquals(tc,count,c.N);
    lin_elem_xy_free(xy); xy = NULL;
}

void Test_lin_elem_exp_approx_adapt(CuTest * tc){

    printf("Testing function: lin_elem_exp_approx (1) \n");
    
    struct counter c;
    c.N = 0;


    double lb = -1.0;
    double ub = 1.0;
    double delta = 1e-2;
    double hmin = 1e-1;
    struct LinElemExpAopts * opts = NULL;
    opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,delta,hmin);
    
    struct LinElemExp * fa = lin_elem_exp_approx(func,&c,lb,ub,opts);

    CuAssertIntEquals(tc,c.N,fa->num_nodes);
//    printf("N = %d\n",c.N);    
    double * xtest = linspace(lb,ub,1000);
    size_t ii;
    double err = 0.0;
    double errNorm = 0.0;
    for (ii = 0; ii < 1000; ii++){

        double eval1 = lin_elem_exp_eval(fa,xtest[ii]);
        double evalt = func(xtest[ii],&c);
        err += pow(eval1-evalt,2);
        errNorm += pow(evalt,2);
    }
    err = err/errNorm;
//    printf("err = %G\n",err);
    CuAssertDblEquals(tc, 0.0, err, 1e-3);

    lin_elem_exp_aopts_free(opts);
    lin_elem_exp_free(fa);
    free(xtest);
}

void Test_lin_elem_exp_approx_adapt2(CuTest * tc){

    printf("Testing function: lin_elem_exp_approx (2) \n");
    
    struct counter c;
    c.N = 0;


    double lb = -1.0;
    double ub = 1.0;
    double delta = 1e-2;
    double hmin = 1e-2;
    struct LinElemExpAopts * opts = NULL;
    opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,delta,hmin);
    
    struct LinElemExp * fa = lin_elem_exp_approx(fleadapt,&c,lb,ub,opts);

    CuAssertIntEquals(tc,c.N,fa->num_nodes);
//    printf("N = %d\n",c.N);    
    double * xtest = linspace(lb,ub,1000);
    size_t ii;
    double err = 0.0;
    double errNorm = 0.0;
    for (ii = 0; ii < 1000; ii++){

        double eval1 = lin_elem_exp_eval(fa,xtest[ii]);
        double evalt = fleadapt(xtest[ii],&c);
        err += pow(eval1-evalt,2);
        errNorm += pow(evalt,2);
    }
    err = err/errNorm;
//    printf("err = %G\n",err);
    CuAssertDblEquals(tc, 0.0, err, 1e-3);

    lin_elem_exp_aopts_free(opts);
    lin_elem_exp_free(fa);
    free(xtest);
}

void Test_lin_elem_exp_prod(CuTest * tc){

    printf("Testing function: lin_elem_exp_prod \n");
    double lb = -3.0;
    double ub = 2.0;

    double delta = 1e-2;
    double hmin = 1e-2;
    struct LinElemExpAopts * opts = NULL;
    opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,delta,hmin);

    struct counter c;
    c.N = 0;
    struct LinElemExp * f1 = lin_elem_exp_approx(func2,&c,lb,ub,opts);

    struct counter c2;
    c2.N = 0;
    struct LinElemExp * f2 = lin_elem_exp_approx(func3,&c2,lb,ub,opts);

    struct LinElemExp * f3 = lin_elem_exp_prod(f1,f2,NULL);
    
    //print_orth_poly_expansion(cpoly3,0,NULL);
    
    size_t N = 100;
    double * pts = linspace(lb,ub,N); 
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = lin_elem_exp_eval(f3,pts[ii]);
        double eval2 = lin_elem_exp_eval(f1,pts[ii]) * 
                       lin_elem_exp_eval(f2,pts[ii]);
        double diff= fabs(eval1-eval2);
        CuAssertDblEquals(tc, 0.0, diff, 1e-2);
    }

    free(pts); pts = NULL;
    lin_elem_exp_free(f1);
    lin_elem_exp_free(f2);
    lin_elem_exp_free(f3);
    lin_elem_exp_aopts_free(opts);
}

void Test_lin_elem_exp_derivative(CuTest * tc){

    printf("Testing function: lin_elem_exp_deriv  on (a,b)\n");
    double lb = -2.0;
    double ub = -1.0;

    double delta = 1e-2;
    double hmin = 1e-3;
    struct LinElemExpAopts * opts = NULL;
    opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,delta,hmin);

    struct counter c;
    c.N = 0;
    struct LinElemExp * f1 = lin_elem_exp_approx(func,&c,lb,ub,opts);
    struct LinElemExp * der = lin_elem_exp_deriv(f1);

    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    size_t ii;
    double err = 0.0;
    double errNorm = 0.0;
    for (ii = 0; ii < N; ii++){
        err += pow(lin_elem_exp_eval(der,xtest[ii])-funcderiv(xtest[ii], NULL),2);
        errNorm += pow(funcderiv(xtest[ii],NULL),2);
        //printf("pt= %G err = %G \n",xtest[ii], err);
    }
    //printf("num polys adapted=%zu\n",cpoly->num_poly);
    err = err / errNorm;
//    printf("err = %G\n",err);
    CuAssertDblEquals(tc, 0.0, err, 1e-3);
    
    lin_elem_exp_free(f1);
    lin_elem_exp_free(der);
    free(xtest);
    lin_elem_exp_aopts_free(opts);
}

void Test_lin_elem_exp_integrate(CuTest * tc){

    printf("Testing function: lin_elem_exp_integrate\n");
    double lb = -2;
    double ub = 3.0;

    size_t N = 1000;
    double * x = linspace(lb,ub,N);
    double f[1000];
    
    struct counter c;
    c.N = 0;

    for (size_t ii = 0; ii < N; ii++){
        f[ii] = func2(x[ii],&c);
    }

    struct LinElemExp * fa = lin_elem_exp_init(N,x,f);
    
    double intshould = (pow(ub,3) - pow(lb,3))/3;
    double intis = lin_elem_exp_integrate(fa);
    CuAssertDblEquals(tc, intshould, intis, 1e-4);

    lin_elem_exp_free(fa);
    free(x);
}

void Test_lin_elem_exp_inner(CuTest * tc){

    printf("Testing function: lin_elem_exp_inner (1) \n");
    double lb = -2.0;
    double ub = 3.0;

    size_t N = 1000;
    double * x = linspace(lb,ub,N);
    double f[1000];
    double g[1000];
    
    struct counter c;
    c.N = 0;

    for (size_t ii = 0; ii < N; ii++){
        f[ii] = func2(x[ii],&c);
        g[ii] = func3(x[ii],&c);
    }

    struct LinElemExp * fa = lin_elem_exp_init(N,x,f);
    struct LinElemExp * fb = lin_elem_exp_init(N,x,g);
        
    double intshould = (pow(ub,6) - pow(lb,6))/3;
    double intis = lin_elem_exp_inner(fa,fb);
//    printf("int should true true true! = %3.15G, intis=%3.15G\n",
//           intshould,intis);
    double diff = fabs(intshould-intis)/fabs(intshould);
    // printf("diff = %G\n",diff);
    CuAssertDblEquals(tc, 0.0, diff, 1e-5);

    lin_elem_exp_free(fa);
    lin_elem_exp_free(fb);
    free(x);
}

void Test_lin_elem_exp_inner2(CuTest * tc){

    printf("Testing function: lin_elem_exp_inner (2)\n");
    double lb = -2.0;
    double ub = 3.0;

    size_t N1 = 10;
    size_t N2 = 20;

    double * p1 = linspace(lb, 0.5,N1);
    double * p2 = linspace(0.0,ub,N2);
    double f[1000];
    double g[1000];

    //printf("p1 = "); dprint(N1,p1);
    //printf("p2 = "); dprint(N2,p2);

    struct counter c;
    c.N = 0;

    for (size_t ii = 0; ii < N1; ii++){
        f[ii] = func2(p1[ii],&c);
    }
    for (size_t ii = 0; ii < N2; ii++){
        g[ii] = func3(p2[ii],&c);        
    }
    // printf("g values are = "); dprint(N2,g);

    struct LinElemExp * fa = lin_elem_exp_init(N1,p1,f);
    struct LinElemExp * fb = lin_elem_exp_init(N2,p2,g);
        
    double intis = lin_elem_exp_inner(fa,fb);
    double intis2 = lin_elem_exp_inner(fb,fa);

    size_t ntest = 10000000;
    double * xtest = linspace(lb,ub,ntest);
    double integral = 0.0;
    for (size_t ii = 0; ii < ntest; ii++){
        integral += (lin_elem_exp_eval(fa,xtest[ii]) * 
                     lin_elem_exp_eval(fb,xtest[ii]));
    }
    integral /= (double) ntest;
    integral *= (ub - lb);
    
    double intshould = integral;
    //printf("intshould=%3.15G, intis=%3.15G\n",intshould,intis);
    double diff = fabs(intshould-intis)/fabs(intshould);
    //printf("diff = %G\n",diff);
    CuAssertDblEquals(tc, 0.0, diff, 1e-6);
    CuAssertDblEquals(tc,intis,intis2,1e-15);

    lin_elem_exp_free(fa);
    lin_elem_exp_free(fb);
    free(p1);
    free(p2);
    free(xtest);
}

void Test_lin_elem_exp_norm(CuTest * tc){
    
    printf("Testing function: lin_elem_exp_norm\n");
    double lb = -2.0;
    double ub = 3.0;

    size_t N = 1000;
    double * x = linspace(lb,ub,N);
    double f[1000];
    
    struct counter c = {0};

    for (size_t ii = 0; ii < N; ii++){
        f[ii] = func2(x[ii],&c);
    }

    struct LinElemExp * fa = lin_elem_exp_init(N,x,f);

    double intshould = (pow(ub,5) - pow(lb,5))/5;
    double intis = lin_elem_exp_norm(fa);
    double diff = fabs(sqrt(intshould) - intis)/fabs(sqrt(intshould));
    // printf("diff = %G\n",diff);
    CuAssertDblEquals(tc, 0.0, diff, 1e-6);

    free(x); x = NULL;
    lin_elem_exp_free(fa);
}

void Test_lin_elem_exp_axpy(CuTest * tc){

    printf("Testing function: lin_elem_exp_axpy (1) \n");
    double lb = -2.0;
    double ub = 1.0;

    size_t N1 = 100;
    size_t N2 = 100;

    struct counter c1 = {0};
    struct counter c2 = {0};
    double * x1 = linspace(lb,ub,N1);
    double * x2 = linspace(lb,ub,N2);
    double f1[1000];
    double f2[1000];
    for (size_t ii = 0; ii < N1; ii++){
        f1[ii] = func3(x1[ii],&c1);
    }
    for (size_t ii = 0; ii < N2; ii++){
        f2[ii] = func2(x2[ii],&c2);
    }
    struct LinElemExp * le1 = lin_elem_exp_init(N1,x1,f1);
    struct LinElemExp * le2 = lin_elem_exp_init(N2,x2,f2);
    struct LinElemExp * le3 = lin_elem_exp_copy(le2);

    int success = lin_elem_exp_axpy(2.0,le1,le3);
    CuAssertIntEquals(tc,0,success);
    //print_orth_poly_expansion(cpoly3,0,NULL);
    
    size_t N = 200;
    double * pts = linspace(lb-0.5,ub+0.5,N); 
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = lin_elem_exp_eval(le3,pts[ii]);
        double eval2 = 2.0 * lin_elem_exp_eval(le1,pts[ii]) +
            lin_elem_exp_eval(le2,pts[ii]);
        double diff= fabs(eval1-eval2);
        //printf("x = %G, diff = %G\n",pts[ii],diff);
        CuAssertDblEquals(tc, 0.0, diff, 4e-15);
    }

    free(pts); pts = NULL;
    free(x1); x1 = NULL;
    free(x2); x2 = NULL;
    lin_elem_exp_free(le1);
    lin_elem_exp_free(le2);
    lin_elem_exp_free(le3);
}

void Test_lin_elem_exp_axpy2(CuTest * tc){

    printf("Testing function: lin_elem_exp_axpy (2) \n");
    double lb = -2.0;
    double ub = 1.0;

    size_t N1 = 302;
    size_t N2 = 20;

    struct counter c1 = {0};
    struct counter c2 = {0};
    double * x1 = linspace(lb,0.2,N1);
    double * x2 = linspace(-0.15,ub,N2);
    double f1[1000];
    double f2[1000];
    for (size_t ii = 0; ii < N1; ii++){
        f1[ii] = func3(x1[ii],&c1);
    }
    for (size_t ii = 0; ii < N2; ii++){
        f2[ii] = func2(x2[ii],&c2);
    }
    struct LinElemExp * le1 = lin_elem_exp_init(N1,x1,f1);
    struct LinElemExp * le2 = lin_elem_exp_init(N2,x2,f2);
    struct LinElemExp * le3 = lin_elem_exp_copy(le2);

    int success = lin_elem_exp_axpy(2.0,le1,le3);
    //printf("done!\n");
    CuAssertIntEquals(tc,0,success);
    //print_orth_poly_expansion(cpoly3,0,NULL);
    
    size_t N = 200;
    double * pts = linspace(lb-0.5,ub+0.5,N); 
    size_t ii;
    for (ii = 0; ii < N; ii++){
        double eval1 = lin_elem_exp_eval(le3,pts[ii]);
        double eval2 = 2.0 * lin_elem_exp_eval(le1,pts[ii]) +
            lin_elem_exp_eval(le2,pts[ii]);
        double diff= fabs(eval1-eval2);
        //printf("diff = %G\n",diff);
        CuAssertDblEquals(tc, 0.0, diff, 4e-15);
    }

    free(pts); pts = NULL;
    free(x1); x1 = NULL;
    free(x2); x2 = NULL;
    lin_elem_exp_free(le1);
    lin_elem_exp_free(le2);
    lin_elem_exp_free(le3);
}

void Test_lin_elem_exp_constant(CuTest * tc)
{
    printf("Testing function: lin_elem_exp_constant\n");
    double lb = -2.0;
    double ub = 0.2;
    struct LinElemExp * f = lin_elem_exp_constant(2.0,lb,ub,NULL);

    double * xtest = linspace(lb,ub,1000);
    for (size_t ii = 0; ii < 1000; ii++){
        double val = lin_elem_exp_eval(f,xtest[ii]);
        CuAssertDblEquals(tc,2.0,val,1e-15);
    }
    free(xtest);
    lin_elem_exp_free(f);
}

void Test_lin_elem_exp_flipsign(CuTest * tc)
{
    printf("Testing function: lin_elem_exp_flip_sign\n");
    double lb = -2.0;
    double ub = 0.2;
    struct LinElemExp * f = lin_elem_exp_constant(0.3,lb,ub,NULL);
    lin_elem_exp_flip_sign(f);
    double * xtest = linspace(lb,ub,1000);
    for (size_t ii = 0; ii < 1000; ii++){
        double val = lin_elem_exp_eval(f,xtest[ii]);
        CuAssertDblEquals(tc,-0.3,val,1e-15);
    }
    free(xtest);
    lin_elem_exp_free(f);
}

void Test_lin_elem_exp_scale(CuTest * tc)
{
    printf("Testing function: lin_elem_exp_scale\n");
    double lb = -2.0;
    double ub = 0.2;
    struct LinElemExp * f = lin_elem_exp_constant(0.3,lb,ub,NULL);
    lin_elem_exp_scale(0.3, f);
    double * xtest = linspace(lb,ub,1000);
    for (size_t ii = 0; ii < 1000; ii++){
        double val = lin_elem_exp_eval(f,xtest[ii]);
        CuAssertDblEquals(tc,0.09,val,1e-15);
    }
    free(xtest);
    lin_elem_exp_free(f);
}

void Test_lin_elem_exp_orth_basis(CuTest * tc)
{
    printf("Testing function: lin_elem_exp_orth_basis\n");
    double lb = -2.0;
    double ub = 0.2;
    size_t N = 100;
    double * x = linspace(lb,ub,N);
    double * coeff = calloc_double(N);

    struct LinElemExp * f[100];
    for (size_t ii = 0; ii < N; ii++){
        f[ii] = lin_elem_exp_init(N,x,coeff);
    }

    lin_elem_exp_orth_basis(N,f);
    for (size_t ii = 0; ii < N; ii++){
        for (size_t jj = 0; jj < N; jj++){
            double val = lin_elem_exp_inner(f[ii],f[jj]);
            if (ii == jj){
                CuAssertDblEquals(tc,1.0,val,1e-15);
            }
            else{
                CuAssertDblEquals(tc,0.0,val,1e-15);
            }
        }
    }

    for (size_t ii = 0; ii < N; ii++){
        lin_elem_exp_free(f[ii]);        
    }
    free(x); x = NULL;
    free(coeff); coeff = NULL;
}

void Test_lin_elem_exp_serialize(CuTest * tc){
    
    printf("Testing functions: (de)serializing lin_elem_exp \n");
    
    double lb = -1.0;
    double ub = 2.0;

    struct counter c1 = {0};
    size_t N1 = 10;
    double * x1 = linspace(lb,ub,N1);
    double f1[1000];
    for (size_t ii = 0; ii < N1; ii++){
        f1[ii] = func3(x1[ii],&c1);
    }

    struct LinElemExp * pl = lin_elem_exp_init(N1,x1,f1);
    free(x1); x1 = NULL;
    //print_lin_elem_exp(pl,4,NULL,stdout);
      
    unsigned char * text = NULL;
    size_t size_to_be;
    serialize_lin_elem_exp(text, pl, &size_to_be);
    text = malloc(size_to_be * sizeof(unsigned char));
    serialize_lin_elem_exp(text, pl, NULL);
     
    //printf("text=\n%s\n",text);
    struct LinElemExp * pt = NULL;
    deserialize_lin_elem_exp(text, &pt);

    //printf("pt num nodes = %zu\n",pt->num_nodes);
    //print_lin_elem_exp(pt,0,NULL,stdout);            

    double * xtest = linspace(lb,ub,1000);
    size_t ii;
    double err = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(lin_elem_exp_eval(pl,xtest[ii]) -
                   lin_elem_exp_eval(pt,xtest[ii]),2);
    }
    err = sqrt(err);
    CuAssertDblEquals(tc, 0.0, err, 1e-15);

    free(xtest);
    free(text);
    lin_elem_exp_free(pl);
    lin_elem_exp_free(pt);
}

CuSuite * LelmGetSuite(){

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_approx);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_adapt);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_approx_adapt);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_approx_adapt2);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_prod);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_derivative);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_integrate);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_inner);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_inner2);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_norm);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_axpy);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_axpy2);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_constant);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_flipsign);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_scale);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_orth_basis);
    SUITE_ADD_TEST(suite, Test_lin_elem_exp_serialize);
    return suite;
}

void Test_Linked_List(CuTest * tc){
   
    printf("Testing functions: Linked_List \n");

    double x[5] = {0.0, 1.0, 2.0, 0.5, 0.3};
    double val = 2.0;

    size_t sv1 = 5 * sizeof(double) + sizeof(char);
    char * v1 = malloc(sv1);
    memmove(v1, x, 5 * sizeof(double));
    v1[sv1-1] = '\0';

    //printf("v1[0]=%c\n",v1[0]);

    size_t sv2 = sizeof(double) + sizeof(char);
    char * v2 = malloc(sv2);
    memmove(v2, &val, sizeof(double));
    v2[sv2-1] = '\0';

    struct Cpair * pl = cpair_create(v1,v2);
    struct PairList * ll = NULL;

    pair_push(&ll,pl);

   // print_pair_list(ll);
   // printf("============\n");
    pair_list_delete(&ll);

    //print_pair_list(ll);
    //printf("============\n");
    
    CuAssertIntEquals(tc,1,1);

    free(v1);
    free(v2);
    cpair_free(pl);
}

CuSuite * LinkedListGetSuite(){
    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_Linked_List);
    return suite;
}

double pw_lin(double x){
    
    return 2.0 * x + -0.2;
}
void Test_pw_linear(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_linear \n");
    
    double lb = -2.0;
    double ub = 0.7;
    
    struct PiecewisePoly * pw = 
            piecewise_poly_linear(2.0,-0.2,LEGENDRE,lb,ub);
    
    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double terr;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        terr = fabs(pw_lin(xtest[ii]) - piecewise_poly_eval(pw,xtest[ii]));
        err+= terr;
    }

    CuAssertDblEquals(tc, 0.0, err, 1e-13);

    free(xtest);
    xtest = NULL;
    piecewise_poly_free(pw);
    pw = NULL;

}

double pw_quad(double x){
    
    return 1e-10 * x * x + 3.2 * x + -0.2;
}
void Test_pw_quad(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_quad \n");
    
    double lb = -2.0;
    double ub = 0.7;
    
    struct PiecewisePoly * pw = 
            piecewise_poly_quadratic(1e-10,3.2,-0.2,LEGENDRE,lb,ub);
    
    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double terr;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        terr = fabs(pw_quad(xtest[ii]) - piecewise_poly_eval(pw,xtest[ii]));
        err+= terr;
    }

    CuAssertDblEquals(tc, 0.0, err, 1e-12);

    free(xtest);
    xtest = NULL;
    piecewise_poly_free(pw);
    pw = NULL;

}

void Test_pw_approx(CuTest * tc){

    printf("Testing function: piecewise_poly_approx1 (1/1);\n");
    struct counter c;
    c.N = 0;

    struct PiecewisePoly * p = NULL;
    p = piecewise_poly_approx1(func,&c,-1.0,1.0,NULL);

    double * xtest = linspace(-1,1,1000);
    size_t ii;
    double err = 0.0;
    double errNorm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(piecewise_poly_eval(p,xtest[ii]) - func(xtest[ii],&c),2);
        errNorm += pow(func(xtest[ii],&c),2);
    }
    err = sqrt(err / errNorm);
    //printf("err = %G\n",err);
    CuAssertDblEquals(tc, 0.0, err, 1e-8);
    piecewise_poly_free(p);
    free(xtest);
}

void Test_pw_approx_nonnormal(CuTest * tc){

    printf("Testing function: piecewise_poly_approx on (a,b)\n");

    double lb = -3.0;
    double ub = 2.0;
    struct counter c;
    c.N = 0;

    size_t N = 15;
    double * pts = linspace(lb,ub,N); //

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.nregions = N-1;
    aopts.pts = pts;
    aopts.other = NULL;

    
    struct PiecewisePoly * p = NULL;
    p = piecewise_poly_approx1(func,&c,lb,ub,&aopts);
    
    double lb1 = piecewise_poly_lb(p->branches[0]);
    double ub1 = piecewise_poly_ub(p->branches[0]);
    CuAssertDblEquals(tc,pts[0],lb1,1e-14);
    CuAssertDblEquals(tc,pts[1],ub1,1e-14);


    double * xtest = linspace(lb,ub,1000);
    size_t ii;
    double err = 0.0;
    double errNorm = 0.0;
    for (ii = 0; ii < 1000; ii++){
        err += pow(piecewise_poly_eval(p,xtest[ii]) - func(xtest[ii],&c),2);
        errNorm += pow(func(xtest[ii],&c),2);
    }
    err = sqrt(err / errNorm);
    //printf("err = %G\n",err);
    CuAssertDblEquals(tc, 0.0, err, 1e-9);
    
    piecewise_poly_free(p);
    free(xtest);
    free(pts); pts = NULL;
}

void Test_pw_approx1_adapt(CuTest * tc){

    printf("Testing function:  pw_approx1_adapt\n");

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.minsize = 1e-10;
    aopts.coeff_check = 2;
    aopts.epsilon=1e-8;
    aopts.nregions = 5;
    aopts.pts = NULL;

    struct counter c;
    c.N = 0;
    struct PiecewisePoly * p = 
        piecewise_poly_approx1_adapt(func, &c, -1.0,1.0, &aopts);
    
    size_t nbounds;
    double * bounds = NULL;
    piecewise_poly_boundaries(p,&nbounds,&bounds,NULL);
    //dprint(nbounds,bounds);
    free(bounds);

    double * xtest = linspace(-1,1,100);
    size_t ii;
    double err = 0.0;
    double errNorm = 0.0;
    for (ii = 0; ii < 100; ii++){
        double diff = piecewise_poly_eval(p,xtest[ii]) - func(xtest[ii],&c);
        //printf("pt=%G, diff=%G\n",xtest[ii],diff);
        err += pow(diff,2);
        errNorm += pow(func(xtest[ii],&c),2);
    }
    //printf("num polys adapted=%zu\n",cpoly->num_poly);
    err = sqrt(err / errNorm);
    //printf("err = %G\n",err);
    CuAssertDblEquals(tc, 0.0, err, 1e-14);
    piecewise_poly_free(p); p = NULL;
    free(xtest);
}

void Test_pw_approx_adapt_weird(CuTest * tc){

    printf("Testing function: piecewise_poly_approx1_adapt on (a,b)\n");
    double lb = -2;
    double ub = -1.0;

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.epsilon = 1e-10;
    aopts.minsize = 1e-5;
    aopts.coeff_check = 2;
    aopts.nregions = 5;
    aopts.pts = NULL;

    struct counter c;
    c.N = 0;
    struct PiecewisePoly * p = 
        piecewise_poly_approx1_adapt(func, &c, lb,ub, &aopts);
    
    double * xtest = linspace(lb,ub,1000);
    size_t ii;
    double err = 0.0;
    double errNorm = 0.0;
    double diff;
    for (ii = 0; ii < 1000; ii++){
        diff = piecewise_poly_eval(p,xtest[ii]) - func(xtest[ii],&c);
        err += pow(diff,2);
        errNorm += pow(func(xtest[ii],&c),2);
    }
    //printf("num polys adapted=%zu\n",cpoly->num_poly);
    err = sqrt(err / errNorm);
    //printf("error = %G\n",err);
    CuAssertDblEquals(tc, 0.0, err, 1e-9);
    piecewise_poly_free(p);
    free(xtest);
}

double pw_disc(double x, void * args){
    
    assert ( args == NULL );
    double split = 0.0;
    if (x > split){
        return sin(x);
    }
    else{
        return pow(x,2) + 2.0 * x + 1.0;
    }
}

void Test_pw_approx1(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_approx1 on discontinuous function (1/2) \n");
    
    double lb = -5.0;
    double ub = 1.0;

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.minsize = 1e-2;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-3;
    aopts.nregions = 5;
    aopts.pts = NULL;


    struct PiecewisePoly * p = 
            piecewise_poly_approx1_adapt(pw_disc, NULL, lb, ub, &aopts);

    size_t nbounds;
    double * bounds = NULL;
    piecewise_poly_boundaries(p,&nbounds,&bounds,NULL);
    //printf("Number of regions = %zu\n",nbounds-1);
    //dprint(nbounds,bounds);
    free(bounds);

    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double errNorm = 0.0;
    double diff;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        diff = piecewise_poly_eval(p,xtest[ii]) - pw_disc(xtest[ii],NULL);
        err += pow(diff,2);
        errNorm += pow(pw_disc(xtest[ii],NULL),2);

        //printf("x=%G, terr=%G\n",xtest[ii],terr);
    }
    err = sqrt(err / errNorm);
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc, 0.0, err, 1e-14);
    piecewise_poly_free(p); p = NULL;
    free(xtest); xtest = NULL;
}

void Test_pw_flatten(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_flatten \n");
    
    double lb = -5.0;
    double ub = 1.0;

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.minsize = 1e-2;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-3;
    aopts.nregions = 5;
    aopts.pts = NULL;


    struct PiecewisePoly * p = 
            piecewise_poly_approx1_adapt(pw_disc, NULL, lb, ub, &aopts);
    
    size_t nregions = piecewise_poly_nregions(p);
    int isflat = piecewise_poly_isflat(p);
    CuAssertIntEquals(tc,0,isflat);
    piecewise_poly_flatten(p);
    CuAssertIntEquals(tc,nregions,p->nbranches);
    isflat = piecewise_poly_isflat(p);
    CuAssertIntEquals(tc,1,isflat);
    
    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double errNorm = 0.0;
    double diff;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        diff = piecewise_poly_eval(p,xtest[ii]) - pw_disc(xtest[ii],NULL);
        err += pow(diff,2);
        errNorm += pow(pw_disc(xtest[ii],NULL),2);

        //printf("x=%G, terr=%G\n",xtest[ii],terr);
    }
    err = sqrt(err / errNorm);
    //printf("err=%G\n",err);
    CuAssertDblEquals(tc, 0.0, err, 1e-14);
    piecewise_poly_free(p); p = NULL;
    free(xtest); xtest = NULL;
}

double pw_disc2(double x, void * args){
    
    assert ( args == NULL );
    double split = 0.2;
    if (x < split){
        return sin(x);
    }
    else{
        return pow(x,2) + 2.0 * x;
    }
}

void Test_pw_integrate(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_integrate (1/2) \n");
    
    double lb = -2.0;
    double ub = 1.0;

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-3;
    aopts.minsize = 1e-5;
    aopts.nregions = 5;
    aopts.pts = NULL;
    
    double sol ;
    if ( ub > 0.2 ) {
        sol = pow(ub,3)/3.0 + pow(ub,2) -  pow(0.2,3)/3.0 - pow(0.2,2) +
                ( -cos(0.2) - (-cos(lb)));
    }
    else{
        sol = -cos(ub) - (-cos(lb));
    }
    struct PiecewisePoly * p2 = 
            piecewise_poly_approx1_adapt(pw_disc2, NULL, lb, ub, &aopts);
    double ints = piecewise_poly_integrate(p2);

    CuAssertDblEquals(tc, sol, ints, 1e-6);
    piecewise_poly_free(p2);
    p2 = NULL;
}

void Test_pw_integrate2(CuTest * tc){

    printf("Testing function: piecewise_poly_integrate (2/2)\n");
    double lb = -2;
    double ub = 3.0;

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-3;
    aopts.minsize = 1e-5;
    aopts.nregions = 5;
    aopts.pts = NULL;
    
    struct counter c;
    c.N = 0;
    struct PiecewisePoly * p2 = 
            piecewise_poly_approx1_adapt(func2, &c, lb, ub, &aopts);
    double ints = piecewise_poly_integrate(p2);
    double intshould = (pow(ub,3) - pow(lb,3))/3;
    CuAssertDblEquals(tc, intshould, ints, 1e-13);
    piecewise_poly_free(p2); p2 = NULL;
}

void Test_pw_inner(CuTest * tc){

    printf("Testing function: piecewise_poly_inner\n");
    double lb = -2;
    double ub = 3.0;

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-8;
    aopts.minsize = 1e-5;
    aopts.nregions = 5;
    aopts.pts = NULL;
    
    struct counter c;
    c.N = 0;
    struct PiecewisePoly * cpoly = 
        piecewise_poly_approx1_adapt(func2,&c,lb,ub,&aopts);
    
    struct counter c2;
    c2.N = 0;
    struct PiecewisePoly * cpoly2 =
        piecewise_poly_approx1_adapt(func3,&c2,lb,ub,&aopts);
    
    double intshould = (pow(ub,6) - pow(lb,6))/3;
    double intis = piecewise_poly_inner(cpoly,cpoly2);
    CuAssertDblEquals(tc, intshould, intis, 1e-10);
    piecewise_poly_free(cpoly);
    piecewise_poly_free(cpoly2);
}
void Test_pw_norm(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_norm (1/2)\n");
    
    double lb = -2.0;
    double ub = 0.7;
    
    double sol = sqrt(1.19185 + 0.718717);
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-3;
    aopts.minsize = 1e-5;
    aopts.nregions = 5;
    aopts.pts = NULL;
    
    struct PiecewisePoly * pw = 
            piecewise_poly_approx1_adapt(pw_disc2, NULL, lb, ub,&aopts);
    
    double ints = piecewise_poly_norm(pw);

    CuAssertDblEquals(tc, sol, ints, 1e-5);

    piecewise_poly_free(pw);
    pw = NULL;

}

void Test_pw_norm2(CuTest * tc){
    
    printf("Testing function: piecewise_poly_norm (2/2)\n");
    double lb = -2;
    double ub = 3.0;

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-3;
    aopts.minsize = 1e-5;
    aopts.nregions = 5;
    aopts.pts = NULL;
    
    struct counter c;
    c.N = 0;
    struct PiecewisePoly * pw = 
            piecewise_poly_approx1_adapt(func2, &c, lb, ub,&aopts);
    
    double intshould = (pow(ub,5) - pow(lb,5))/5;
    double intis = piecewise_poly_norm(pw);
    CuAssertDblEquals(tc, sqrt(intshould), intis, 1e-10);
    piecewise_poly_free(pw);
}

void Test_pw_daxpby(CuTest * tc){

    printf("Testing functions: piecewise_poly_daxpby (1/2)\n");

    double lb = -2.0;
    double ub = 0.7;
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-10;
    aopts.minsize = 1e-5;
    aopts.nregions = 5;
    aopts.pts = NULL;
    
    struct PiecewisePoly * a = 
            piecewise_poly_approx1_adapt(pw_disc2, NULL, lb,ub,&aopts);

    struct PiecewisePoly * b = 
            piecewise_poly_approx1_adapt(pw_disc, NULL, lb,ub,&aopts);
    
    struct PiecewisePoly * c = 
            piecewise_poly_daxpby(0.4,a,0.5,b);
    
    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double errden = 0.0;
    double err = 0.0;
    double diff,val;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        val = (0.4*pw_disc2(xtest[ii],NULL) + 0.5*pw_disc(xtest[ii],NULL));
        diff= piecewise_poly_eval(c,xtest[ii]) - val;

        //val = pw_disc2(xtest[ii],NULL);
        //diff = piecewise_poly_eval(a,xtest[ii]) - val;

        //val = pw_disc(xtest[ii],NULL);
        //diff = piecewise_poly_eval(b,xtest[ii]) - val;

        err+= pow(diff,2.0);
        errden += pow(val,2.0);
        //printf("(x,terr)=(%G,%G)\n", xtest[ii],diff);
    }
    err = sqrt(err/errden);
    //printf("err = %G\n",err);
    CuAssertDblEquals(tc, 0.0, err, 1e-12);

    free(xtest);
    xtest = NULL;
    piecewise_poly_free(a); a = NULL;
    piecewise_poly_free(b); b = NULL;
    piecewise_poly_free(c); c = NULL;
}

double pw_exp(double x, void * args){
    assert (args == NULL);

    if (x < -0.2){
        return 0.0;
    }
    else{
        return (exp(5.0 * x));
    }
}

void Test_pw_daxpby2(CuTest * tc){

    printf("Testing functions: piecewise_poly_daxpby (2/2)\n");

    double lb = -1.0;
    double ub = 1.0;
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-10;
    aopts.minsize = 1e-5;
    aopts.nregions = 5;
    aopts.pts = NULL;
    
    struct PiecewisePoly * a = 
            piecewise_poly_approx1_adapt(pw_disc2, NULL, lb, ub,&aopts);

    struct PiecewisePoly * b = 
            piecewise_poly_approx1_adapt(pw_exp, NULL, lb, ub, &aopts);
    
    //printf("got a and b\n");
    /*
    size_t sb; double * nodesb = NULL;
    piecewise_poly_boundaries(b,&sb,&nodesb,NULL);
    //printf("number of merged points %zu\n",sb);
    //dprint(sb,nodesb);
    free(nodesb); nodesb=NULL;
    */

    struct PiecewisePoly * c = 
            piecewise_poly_daxpby(0.5,a,0.5,b);
    
    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double terr;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        terr = fabs(piecewise_poly_eval(c,xtest[ii]) -
                (0.5*pw_disc2(xtest[ii],NULL) + 0.5*pw_exp(xtest[ii],NULL)));
        err+= terr;
        //printf("(x,terr)=(%G,%G)\n", xtest[ii],terr);
    }
    //printf("err=%3.15G\n",err/N);
    CuAssertDblEquals(tc, 0.0, err/N, 1e-10);

    free(xtest);
    xtest = NULL;
    piecewise_poly_free(a); a = NULL;
    piecewise_poly_free(b); b = NULL;
    piecewise_poly_free(c); c = NULL;
}

void Test_pw_derivative(CuTest * tc){

    printf("Testing function: piecewise_poly_deriv  on (a,b)\n");
    double lb = -2.0;
    double ub = -1.0;

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-13;
    aopts.minsize = 1e-5;
    aopts.nregions = 5;
    aopts.pts = NULL;
    
    struct counter c;
    c.N = 0;
    struct PiecewisePoly * cpoly = 
        piecewise_poly_approx1_adapt(func, &c, lb,ub, &aopts);
    struct PiecewisePoly * der = piecewise_poly_deriv(cpoly); 

    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    size_t ii;
    double err = 0.0;
    double errNorm = 0.0;
    double diff;
    for (ii = 0; ii < N; ii++){
        diff = piecewise_poly_eval(der,xtest[ii]) - funcderiv(xtest[ii], NULL);
        err += pow(diff,2);
        errNorm += pow(funcderiv(xtest[ii],NULL),2);

        //printf("pt= %G err = %G \n",xtest[ii], err);
    }
    //printf("num polys adapted=%zu\n",cpoly->num_poly);
    err = sqrt(err) / errNorm;
    //printf("err = %G\n",err);
    CuAssertDblEquals(tc, 0.0, err, 1e-12);
    piecewise_poly_free(cpoly); cpoly = NULL;
    piecewise_poly_free(der); der = NULL;
    free(xtest); xtest = NULL;
}

void Test_pw_real_roots(CuTest * tc){
    
    printf("Testing function: piecewise_poly_real_roots \n");
    
    double lb = -3.0;
    double ub = 2.0;

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-8;
    aopts.minsize = 1e-5;
    aopts.nregions = 5;
    aopts.pts = NULL;
    
    struct PiecewisePoly * pl = 
            piecewise_poly_approx1_adapt(func6, NULL,lb,ub,&aopts);

    size_t nroots;
    double * roots = piecewise_poly_real_roots(pl, &nroots);
    
    //printf("roots are: (double roots in piecewise_poly)\n");
    //dprint(nroots, roots);
    
    CuAssertIntEquals(tc, 1, 1);
    /*
    CuAssertIntEquals(tc, 5, nroots);
    CuAssertDblEquals(tc, -3.0, roots[0], 1e-9);
    CuAssertDblEquals(tc, 0.0, roots[1], 1e-9);
    CuAssertDblEquals(tc, 1.0, roots[2], 1e-5);
    CuAssertDblEquals(tc, 1.0, roots[3], 1e-5);
    CuAssertDblEquals(tc, 2.0, roots[4], 1e-9);
    */
    free(roots);
    piecewise_poly_free(pl);
}

void Test_maxmin_pw(CuTest * tc){
    
    printf("Testing functions: absmax, max and min of pw \n");
    
    double lb = -1.0;
    double ub = 2.0;
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-8;
    aopts.minsize = 1e-5;
    aopts.nregions = 5;
    aopts.pts = NULL;
    
    struct PiecewisePoly * pl = 
            piecewise_poly_approx1_adapt(func7, NULL,lb,ub,&aopts);

    double loc;
    double max = piecewise_poly_max(pl, &loc);
    double min = piecewise_poly_min(pl, &loc);
    double absmax = piecewise_poly_absmax(pl, &loc,NULL);

    
    CuAssertDblEquals(tc, 1.0, max, 1e-10);
    CuAssertDblEquals(tc, -1.0, min, 1e-10);
    CuAssertDblEquals(tc, 1.0, absmax, 1e-10);

    piecewise_poly_free(pl);
}


void Test_pw_serialize(CuTest * tc){
   
    printf("Testing functions: (de)serialize_piecewise_poly (and approx2) \n");
    
    double lb = -2.0;
    double ub = 0.7;
    
    struct PiecewisePoly * pw = 
            piecewise_poly_approx1(pw_disc, NULL, lb, ub, NULL);
    
    //printf("approximated \n");
    size_t size;
    serialize_piecewise_poly(NULL,pw,&size);
    //printf("size=%zu \n",size);
    unsigned char * text = malloc(size);
    serialize_piecewise_poly(text,pw,NULL);
    
    struct PiecewisePoly * pw2 = NULL;
    deserialize_piecewise_poly(text,&pw2);

    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double terr;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        terr = fabs(piecewise_poly_eval(pw2,xtest[ii]) -
                     piecewise_poly_eval(pw,xtest[ii]));
        err+= terr;
    }

    CuAssertDblEquals(tc, 0.0, err, 1e-12);

    free(xtest);
    xtest = NULL;
    free(text); text = NULL;
    piecewise_poly_free(pw); 
    piecewise_poly_free(pw2);
    pw = NULL;
    pw2 = NULL;

}

void Test_poly_match(CuTest * tc){

    printf("Testing functions: piecewise_poly_match \n");

    double lb = -2.0;
    double ub = 0.7;

    size_t Na, Nb;
    double * nodesa = NULL;
    double * nodesb = NULL;

    struct PiecewisePoly * a = 
            piecewise_poly_approx1(pw_disc2, NULL, lb, ub, NULL);

    size_t npa = piecewise_poly_nregions(a);
    piecewise_poly_boundaries(a,&Na, &nodesa, NULL);
    CuAssertIntEquals(tc, npa, Na-1);
    CuAssertDblEquals(tc,-2.0,nodesa[0],1e-15);
    CuAssertDblEquals(tc,0.7,nodesa[Na-1],1e-15);

    struct PiecewisePoly * b = 
            piecewise_poly_approx1(pw_disc, NULL, lb, ub, NULL);
    
    printf("got both\n");
    size_t npb = piecewise_poly_nregions(b);
    piecewise_poly_boundaries(b,&Nb, &nodesb, NULL);
    printf("got boundaries\n");
    CuAssertIntEquals(tc, npb, Nb-1);
    CuAssertDblEquals(tc,-2.0,nodesb[0],1e-15);
    CuAssertDblEquals(tc,0.7,nodesb[Nb-1],1e-15);

    struct PiecewisePoly * aa = NULL;
    struct PiecewisePoly * bb = NULL;
    printf("matching\n");
    piecewise_poly_match(a,&aa,b,&bb);
    printf("matched\n");

    size_t npaa = piecewise_poly_nregions(aa);
    size_t npbb = piecewise_poly_nregions(bb);
    CuAssertIntEquals(tc,npaa,npbb);

    size_t Naa, Nbb;
    double * nodesaa = NULL;
    double * nodesbb = NULL;
    
    piecewise_poly_boundaries(aa,&Naa, &nodesaa, NULL);
    CuAssertDblEquals(tc,-2.0,nodesaa[0],1e-15);
    CuAssertDblEquals(tc,0.7,nodesaa[Naa-1],1e-15);

    piecewise_poly_boundaries(bb,&Nbb, &nodesbb, NULL);
    CuAssertDblEquals(tc,-2.0,nodesbb[0],1e-15);
    CuAssertDblEquals(tc,0.7,nodesbb[Nbb-1],1e-15);
    
    CuAssertIntEquals(tc,Naa,Nbb);
    size_t ii; 
    for (ii = 0; ii < Naa; ii++){
        CuAssertDblEquals(tc,nodesaa[ii],nodesbb[ii],1e-15);
    }

    free(nodesa);
    free(nodesb);
    free(nodesaa);
    free(nodesbb);
    piecewise_poly_free(a);
    piecewise_poly_free(b);
    piecewise_poly_free(aa);
    piecewise_poly_free(bb);
    //dprint(Naa, nodesa);
    //dprint(Nbb, nodesb);

}


CuSuite * PiecewisePolyGetSuite(){
    CuSuite * suite = CuSuiteNew();
    
    SUITE_ADD_TEST(suite, Test_pw_linear);
    SUITE_ADD_TEST(suite, Test_pw_quad);
    SUITE_ADD_TEST(suite, Test_pw_approx);
    SUITE_ADD_TEST(suite, Test_pw_approx_nonnormal);
    SUITE_ADD_TEST(suite, Test_pw_approx1_adapt);
    SUITE_ADD_TEST(suite, Test_pw_approx_adapt_weird);
    SUITE_ADD_TEST(suite, Test_pw_approx1);
    SUITE_ADD_TEST(suite, Test_pw_flatten);
    SUITE_ADD_TEST(suite, Test_pw_integrate);
    SUITE_ADD_TEST(suite, Test_pw_integrate2);
    SUITE_ADD_TEST(suite, Test_pw_inner);
    SUITE_ADD_TEST(suite, Test_pw_norm);
    SUITE_ADD_TEST(suite, Test_pw_norm2);
    SUITE_ADD_TEST(suite, Test_pw_daxpby);
    SUITE_ADD_TEST(suite, Test_pw_daxpby2);
    SUITE_ADD_TEST(suite, Test_pw_derivative);
    SUITE_ADD_TEST(suite, Test_pw_real_roots);
    SUITE_ADD_TEST(suite, Test_maxmin_pw);
    SUITE_ADD_TEST(suite, Test_pw_serialize);
    //SUITE_ADD_TEST(suite, Test_poly_match);

    //SUITE_ADD_TEST(suite, Test_minmod_disc_exists);
    //SUITE_ADD_TEST(suite, Test_locate_jumps);
    //SUITE_ADD_TEST(suite, Test_locate_jumps2);
    //SUITE_ADD_TEST(suite, Test_pw_approx1pa);
    //SUITE_ADD_TEST(suite, Test_pw_approx12);
    //SUITE_ADD_TEST(suite, Test_pw_approx12pa);
    //SUITE_ADD_TEST(suite, Test_pw_trim);
    return suite;
}

double pap1(double x, void * args)
{
    assert (args == NULL);
	
    return 5.0 * exp(5.0*x);// + randn();
}
void Test_pap1(CuTest * tc){

    printf("Testing function: approx (1/1) \n");
	
    double lb = -5.0;
    double ub = 5.0;
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 1;
    aopts.epsilon = 1e-5;
    aopts.minsize = 1e-2;
    aopts.nregions = 4;
    aopts.pts = NULL;

    struct PiecewisePoly * cpoly = 
        piecewise_poly_approx1_adapt(pap1, NULL, lb,ub, &aopts);

    size_t nbounds;
    double * bounds = NULL;
    piecewise_poly_boundaries(cpoly,&nbounds,&bounds,NULL);
   // printf("nregions = %zu \n",nbounds-1);
    //dprint(nbounds,bounds);
    free(bounds);

    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    size_t ii;
    double err = 0.0;
    double errNorm = 0.0;
    double diff,val;
    for (ii = 0; ii < N; ii++){
        val = pap1(xtest[ii],NULL);
        diff = piecewise_poly_eval(cpoly,xtest[ii]) - val;
        err += pow(diff,2);
        errNorm += pow(val,2);
        //printf("x=%G,diff=%G\n",xtest[ii],diff/val);
    }
    err = err / errNorm;
    //printf("error = %G\n",err);
    CuAssertDblEquals(tc, 0.0, err, 1e-10);
    piecewise_poly_free(cpoly);
    free(xtest);
}


CuSuite * PolyApproxSuite(){
    CuSuite * suite = CuSuiteNew();
    
    SUITE_ADD_TEST(suite, Test_pap1);
    return suite;   
}



//old stuff
/*


void Test_minmod_disc_exists(CuTest * tc)
{
    printf("Testing functions: minmod_disc_exists \n");

    size_t N = 20;
    double * xtest = linspace(-4.0,1.0,N);
    double * vals = calloc_double(N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        vals[ii] = pw_disc(xtest[ii],NULL);
    }
    size_t minm = 2;
    size_t maxm = 5;
    
    double x;
    int disc;
    //double jumpval;
    for (ii = 0; ii < N-1; ii++){
        x = (xtest[ii]+xtest[ii+1])/2.0;
        disc = minmod_disc_exists(x,xtest,vals,N,minm,maxm);
        //jumpval = minmod_eval(x,xtest,vals,N,minm,maxm);
        //printf("x,disc,jumpval = %G,%d,%G\n",x,disc,jumpval);
        if ( (xtest[ii] < 0.0) && (xtest[ii+1]) > 0.0){
            CuAssertIntEquals(tc,1,disc);
            break;
        }
        //else{
        //    CuAssertIntEquals(tc,0,disc);
        //}
    }
    free(xtest); xtest = NULL;
    free(vals); vals = NULL;
}

void Test_locate_jumps(CuTest * tc)
{
    printf("Testing functions: locate_jumps (1/2) \n");
    
    double lb = -4.0;
    double ub = 1.0;
    double tol = DBL_EPSILON/1000.0;
    size_t nsplit = 10;

    double * edges = NULL;
    size_t nEdge = 0;
    
    locate_jumps(pw_disc,NULL,lb,ub,nsplit,tol,&edges,&nEdge);
    //printf("number of edges = %zu\n",nEdge);
    //printf("Edges = \n");
    //size_t ii = 0;
    //for (ii = 0; ii < nEdge; ii++){
    //    printf("%G ", edges[ii]);
    //}
    //printf("\n");
    CuAssertIntEquals(tc,1,1);
    free(edges); edges = NULL;
}

double pw_multi_disc(double x, void * args){
    
    assert ( args == NULL );
    double split1 = 0.0;
    double split2 = 0.5;
    if (x < split1){
        return pow(x,2) + 2.0 * x + 1.0;
    }
    else if (x < split2){
        return sin(x);
    }
    else{
        return exp(x);
    }
}

void Test_locate_jumps2(CuTest * tc)
{
    printf("Testing functions: locate_jumps (2/2)\n");
    
    double lb = -4.0;
    double ub = 1.0;
    double tol = 1e-7;
    size_t nsplit = 10;

    double * edges = NULL;
    size_t nEdge = 0;
    
    locate_jumps(pw_multi_disc,NULL,lb,ub,nsplit,tol,&edges,&nEdge);
    //printf("number of edges = %zu\n",nEdge);
    //printf("Edges = \n");
    //size_t ii = 0;
    //for (ii = 0; ii < nEdge; ii++){
    //    printf("%G ", edges[ii]);
    //}
    //printf("\n");
    free(edges); edges = NULL;
    CuAssertIntEquals(tc,1,1);
    free(edges);
}


void Test_pw_approx1pa(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_approx2 (1/2) \n");
    
    double lb = -5.0;
    double ub = 1.0;
    
    struct PiecewisePoly * pw = 
            piecewise_poly_approx1(pw_disc, NULL, lb, ub, NULL);

    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double terr;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        terr = fabs(pw_disc(xtest[ii],NULL) -
                            piecewise_poly_eval(pw,xtest[ii]));
        err += terr;
        //printf("x=%G, terr=%G\n",xtest[ii],terr);
    }
    CuAssertDblEquals(tc, 0.0, err, 1e-9);
    free(xtest); xtest=NULL;
    piecewise_poly_free(pw);
    pw = NULL;
}

void Test_pw_approx12(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_approx1 (2/2) \n");
    
    double lb = -1.0;
    double ub = 1.0;
    
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 6;
    aopts.minsize = 1e-3;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-10;

    struct PiecewisePoly * p2 = 
            piecewise_poly_approx1(pw_disc, NULL, lb, ub, &aopts);

    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double terr;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        terr = fabs(pw_disc(xtest[ii],NULL) -
                            piecewise_poly_eval(p2,xtest[ii]));
        err += terr;
       // printf("terr=%G\n",terr);
    }

    CuAssertDblEquals(tc, 0.0, err, 1e-12);

    free(xtest);
    xtest = NULL;
    piecewise_poly_free(p2);
    p2 = NULL;
}

void Test_pw_approx12pa(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_approx2 (2/2) \n");
    
    double lb = -1.0;
    double ub = 1.0;
    
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 10;
    aopts.minsize = 1e-5;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-10;

    struct PiecewisePoly * pw = 
            piecewise_poly_approx1(pw_disc, NULL, lb, ub, &aopts);
    size_t N = 100;
    double * xtest = linspace(lb,ub,N);
    double err = 0.0;
    double terr;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        terr = fabs(pw_disc(xtest[ii],NULL) -
                            piecewise_poly_eval(pw,xtest[ii]));
        err += terr;
       // printf("terr=%G\n",terr);
    }

    CuAssertDblEquals(tc, 0.0, err, 1e-12);

    free(xtest);
    xtest = NULL;
    piecewise_poly_free(pw);
    pw = NULL;
}
*/

/*
void Test_pw_trim(CuTest * tc){
   
    printf("Testing functions: piecewise_poly_trim \n");
    
    double lb = -1.0;
    double ub = 1.0;
    
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 10;
    aopts.minsize = 1e-7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-10;

    struct PiecewisePoly * pw = 
            piecewise_poly_approx1(pw_disc, NULL, lb, ub, &aopts);
    //printf("got approximation \n");
    size_t M;
    double * nodes = NULL;
    piecewise_poly_boundaries(pw,&M,&nodes,NULL);

    double new_lb = nodes[1];
    struct OrthPolyExpansion * temp = piecewise_poly_trim_left(&pw);

    double new_lb_check = piecewise_poly_lb(pw);
    CuAssertDblEquals(tc,new_lb,new_lb_check,1e-15);

    orth_poly_expansion_free(temp);
    temp=NULL;

    //printf("number of pieces is %zu\n",M);
    //printf("nodes are =");
    //dprint(M,nodes);
    free(nodes); nodes = NULL;

    piecewise_poly_free(pw);
    pw = NULL;

}
*/

