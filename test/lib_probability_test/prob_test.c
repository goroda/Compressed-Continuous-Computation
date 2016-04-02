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
#include "lib_clinalg.h"
#include "lib_probability.h"

#include "CuTest.h"

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884197169399375105820974944
#endif 

void Test_stdnorm(CuTest * tc)
{
    printf("Testing Function: probability_density_standard_normal \n");
    size_t dim = 10;
    struct ProbabilityDensity * pdf = probability_density_standard_normal(dim);
    CuAssertIntEquals(tc,1,pdf!=NULL);
    double * x = calloc_double(dim);
    
    double pdfval = probability_density_eval(pdf,x);
    
    double shouldbe = pow(2.0*M_PI,-(double) dim/2.0);
    CuAssertDblEquals(tc,shouldbe,pdfval,1e-13);
    
    double * mean = probability_density_mean(pdf);
    double * cov = probability_density_cov(pdf);
    double * var = probability_density_var(pdf);

    size_t ii,jj;
    for (ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,0.0,mean[ii],1e-4);
        CuAssertDblEquals(tc,1.0,cov[ii*dim+ii],1e-4);
        CuAssertDblEquals(tc,1.0,var[ii],1e-4);
        for (jj = 0; jj < dim; jj++){
            if (jj != ii){
                CuAssertDblEquals(tc,0.0,cov[jj*dim+ii],1e-11);
            }
        }
    }
    probability_density_free(pdf);
    free(x);
    free(mean);
    free(cov);
    free(var);
}

void Test_mvn1d(CuTest * tc)
{
    printf("Testing Function: probability_density_mvn for 1d \n");
    size_t dim = 1;
    double m[1] = {1.0};
    double c[1] = {0.25};

    struct ProbabilityDensity * pdf = probability_density_mvn(dim,m,c);
    double x[1] = {1.0};
    double pdfval = probability_density_eval(pdf,x);
        
    double shouldbe = 1.0 / sqrt(2.0 * M_PI * c[0]);
    shouldbe = shouldbe * exp(-0.5 * pow(x[0]-m[0],2.0) / c[0] );
    
    //printf("detinv=%G\n",pdf->lt->detinv);
    //double trans = 2.0*x[0]-2.0;
    //printf("trans=%G\n",trans);
    //double s2 = 1.0 / sqrt(2.0 * M_PI) * exp(-0.5 * trans * trans) * 2.0;
    //printf("s2=%G\n",s2);
    
    
    CuAssertDblEquals(tc,sqrt(c[0]),pdf->lt->A[0],1e-13);
    CuAssertDblEquals(tc,m[0],pdf->lt->b[0],1e-13);
    /*
    printf("A = %G\n ",pdf->lt->A[0]);
    printf("b = %G\n ",pdf->lt->b[0]);
    printf("Ainv = %G\n ",pdf->lt->Ainv[0]);
    printf("binv = %G\n ",pdf->lt->binv[0]);
    */
    CuAssertDblEquals(tc,shouldbe,pdfval,1e-13);

    double * mean = probability_density_mean(pdf);
    double * cov = probability_density_cov(pdf);

    double * var = probability_density_var(pdf);
    size_t ii;
    for (ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,m[ii],mean[ii],1e-4);
        CuAssertDblEquals(tc,c[ii*dim+ii],cov[ii*dim+ii],1e-4);
        CuAssertDblEquals(tc,c[ii*dim+ii],var[ii],1e-4);
    }

    probability_density_free(pdf);
    free(mean);
    free(cov);
    free(var);
}


void Test_mvn(CuTest * tc)
{
    printf("Testing Function: probability_density_mvn \n");
    size_t dim = 2;
    double m[2] = {1.0, 2.0};
    double c[4] = {1.0, 0.5, 0.5, 1.0};

    struct ProbabilityDensity * pdf = probability_density_mvn(dim,m,c);
    double x[2] = {1.0,0.0};
    double pdfval = probability_density_eval(pdf,x);
    double rho = c[2] / (sqrt(c[0]*c[3]));

    double z = pow(x[0]-m[0],2.0)/c[0] - 
              2.0 * rho * (x[0]-m[0])*(x[1]-m[1])/(sqrt(c[0]*c[3])) + 
             pow(x[1] - m[1],2)/c[3];
    //printf("z=%G\n",z);
    double shouldbe = 1.0 / 2.0 / M_PI / sqrt(c[0]*c[3]) / sqrt(1.0-pow(rho,2.0));
    shouldbe *= exp(-0.5 * z / (1.0-pow(rho,2.0)));
    
    //printf("Ainv = \n");
    //dprint2d_col(2,2,pdf->lt->Ainv);

    //printf("Ainv det = %G\n",pdf->lt->detinv);
    CuAssertDblEquals(tc,shouldbe,pdfval,1e-6);

    double * mean = probability_density_mean(pdf);
    double * cov = probability_density_cov(pdf);
    /*
    printf("mean=\n");
    dprint(dim,mean);
    printf("cov=\n");
    dprint2d_col(2,2,cov);
    */
    double * var = probability_density_var(pdf);
    size_t ii,jj; 
    for (ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,m[ii],mean[ii],1e-4);
        CuAssertDblEquals(tc,c[ii*dim+ii],cov[ii*dim+ii],1e-4);
        CuAssertDblEquals(tc,1.0,var[ii],1e-4);
        for (jj = 0; jj < dim; jj++){
            if (jj != ii){
                CuAssertDblEquals(tc,c[jj*dim+ii],cov[jj*dim+ii],1e-4);
            }
        }
    }

    probability_density_free(pdf);
    free(mean);
    free(cov);
    free(var);
}

void Test_lt_ser(CuTest * tc)
{
    printf("Testing Function: (de)serialize linear_transform \n");
    size_t dim = 2;
    double m[2] = {1.0, 2.0};
    double c[4] = {1.0, 0.5, 0.5, 1.0};

    struct ProbabilityDensity * pdf = probability_density_mvn(dim,m,c);
    unsigned char * ser = NULL;
    size_t totsize;
    linear_transform_serialize(NULL,pdf->lt,&totsize);
    ser = malloc(totsize);
    linear_transform_serialize(ser,pdf->lt,NULL);
    
    struct LinearTransform * lt = NULL;
    linear_transform_deserialize(ser, &lt);

    double diffA = norm2diff(lt->A,pdf->lt->A, 4);
    double diffb = norm2diff(lt->b,pdf->lt->b, 2);
    CuAssertDblEquals(tc,0.0,diffA,1e-15);
    CuAssertDblEquals(tc,0.0,diffb,1e-15);
    linear_transform_free(lt); lt = NULL;
    free(ser); ser = NULL;

    linear_transform_invert(pdf->lt);

    linear_transform_serialize(NULL,pdf->lt,&totsize);
    ser = malloc(totsize);
    linear_transform_serialize(ser,pdf->lt,NULL);
    
    linear_transform_deserialize(ser, &lt);

    diffA = norm2diff(lt->A,pdf->lt->A, 4);
    diffb = norm2diff(lt->b,pdf->lt->b, 2);
    CuAssertDblEquals(tc,0.0,diffA,1e-15);
    CuAssertDblEquals(tc,0.0,diffb,1e-15);
        
    diffA = norm2diff(lt->Ainv,pdf->lt->Ainv, 4);
    diffb = norm2diff(lt->binv,pdf->lt->binv, 2);
    CuAssertDblEquals(tc,0.0,diffA,1e-15);
    CuAssertDblEquals(tc,0.0,diffb,1e-15);

    linear_transform_free(lt); lt = NULL;
    free(ser); ser = NULL;

    probability_density_free(pdf); pdf = NULL;
}

void Test_pdf_ser(CuTest * tc)
{
    printf("Testing Function: (de)serialize probability_density \n");
    size_t dim = 2;
    double m[2] = {1.0, 2.0};
    double c[4] = {1.0, 0.5, 0.5, 1.0};

    struct ProbabilityDensity * pdf = probability_density_mvn(dim,m,c);

    unsigned char * ser = NULL;
    size_t totsize; 
    probability_density_serialize(NULL,pdf,&totsize);
    ser = malloc(totsize);
    probability_density_serialize(ser,pdf,NULL);
    

    struct ProbabilityDensity * pdf2 = NULL;
    probability_density_deserialize(ser,&pdf2);

    double diffA = function_train_norm2diff(pdf->pdf,pdf2->pdf);

    CuAssertDblEquals(tc,0.0,diffA,1e-15);

    free(ser); ser = NULL;
    probability_density_free(pdf); pdf = NULL;
    probability_density_free(pdf2); pdf2 = NULL;
}

double * gradProb(double x[2], void * args){
    double * prec = args;
    double * out = calloc_double(2);
    //printf("pt = \n");
    //dprint(2,x);
    out[0] = (prec[0] * x[0] + prec[2] * x[1]);
    //printf("here out[0] = %G\n",out[0]);
    out[1] = (prec[1] * x[0] + prec[3] * x[1]);
    return out;
}

double * hessProb(double x[2], void * args){
    
    assert (x != NULL);
    double * prec = args;
    double * out = calloc_double(4);
    out[0] = prec[0];
    out[1] = prec[1];
    out[2] = prec[2];
    out[3] = prec[3];

    return out;
}

double * fd(struct ProbabilityDensity * pdf, double * x)
{
    double * out = calloc_double(2);
    double h = 1e-8;

    double ptf[2] = {x[0] + h, x[1]};
    double ptb[2] = {x[0] - h, x[1]};

    double val1 = log(probability_density_eval(pdf, ptf));
    double val2 = log(probability_density_eval(pdf, ptb));
    out[0] = (val1-val2)/(2.0 * h);

    double ptf2[2] = {x[0], x[1] + h};
    double ptb2[2] = {x[0], x[1] - h};
    
    double val3 = log(probability_density_eval(pdf, ptf2));
    double val4 = log(probability_density_eval(pdf, ptb2));
    out[1] = (val3-val4)/(2.0 * h);

    return out;
}

void Test_log_gradient_eval(CuTest * tc)
{
    printf("Testing Function: probability_density_log_gradient_eval \n");
    size_t dim = 2;
    double m[2] = {0.0, 0.0};
    double c[4] = {1.0, 0.5, 0.5, 1.0};
    
    struct ProbabilityDensity * pdf = probability_density_mvn(dim,m,c);
    double prec[4];
    pinv(dim,dim,dim,c,prec,0.0);
    
    size_t N = 10;
    double * xtest = linspace(-0.5,0.5,N);
    size_t ii,jj;
    double pt[2];
    for (ii = 0; ii < N; ii++){
        pt[0] = xtest[ii];
        for (jj = 0; jj < N; jj++){
            pt[1] = xtest[jj];

            double * gradShould = gradProb(pt,prec); 
            //double * gradShould = fd(pdf,pt);
            double * gradIs = probability_density_log_gradient_eval(pdf,pt);
            CuAssertDblEquals(tc, -gradShould[0], gradIs[0], 1e-12);
            CuAssertDblEquals(tc, -gradShould[1], gradIs[1], 1e-12);
            free(gradShould); gradShould = NULL;
            free(gradIs); gradIs = NULL;
        }
    }
    probability_density_free(pdf); pdf = NULL;
    free(xtest); xtest = NULL;
}
 
void Test_log_hessian_eval(CuTest * tc)
{
    printf("Testing Function: probability_density_log_hessian_eval \n");
    size_t dim = 2;
    double m[2] = {0.2, -0.2};
    double c[4] = {1.0, 0.5, 0.5, 1.0};
    //double c[4] = {1.0, 0.0, 0.0, 1.0};
    
    struct ProbabilityDensity * pdf = probability_density_mvn(dim,m,c);
    double prec[4];
    pinv(dim,dim,dim,c,prec,0.0);
    

    size_t N = 5;
    double * xtest = linspace(-0.5,0.5,N);
    size_t ii,jj;
    double pt[2];
    for (ii = 0; ii < N; ii++){
        pt[0] = xtest[ii];
        for (jj = 0; jj < N; jj++){
            pt[1] = xtest[jj];

            // this is actually constant everywhere
            double * hessShould = hessProb(pt,prec);  
           // printf("\n");
            //printf("new hess\n");
            //dprint2d_col(2,2,hessShould);

            double * hessIs = probability_density_log_hessian_eval(pdf,pt);

            //printf("my hess\n");
            //dprint2d_col(2,2,hessIs);

            CuAssertDblEquals(tc, -hessShould[0], hessIs[0], 1e-10);
            CuAssertDblEquals(tc, -hessShould[1], hessIs[1], 1e-10);
            CuAssertDblEquals(tc, -hessShould[2], hessIs[2], 1e-10);
            CuAssertDblEquals(tc, -hessShould[3], hessIs[3], 1e-10);

            free(hessShould); hessShould = NULL;
            free(hessIs); hessIs = NULL;
            //printf("\n");
        }
    }
    probability_density_free(pdf); pdf = NULL;
    free(xtest); xtest = NULL;
}


void Test_laplace(CuTest * tc)
{
    printf("Testing Function: probability_density_laplace \n");
    size_t dim = 2;
    double m[2] = {0.0, 0.0};
    double c[4] = {1.0, 0.5, 0.5, 1.0};
    double c2[4] = {1.0, 0.5, 0.5, 1.0};
    
    double start[2] = {0.5, 1.2};

    double prec[4];
    pinv(dim,dim,dim,c,prec,0.0);

    struct ProbabilityDensity * pdf = 
        probability_density_laplace(gradProb,hessProb,prec,dim,start);

    double * mean = probability_density_mean(pdf);
    double * cov = probability_density_cov(pdf);
    double * var = probability_density_var(pdf);
    size_t ii,jj; 
    for (ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,m[ii],mean[ii],1e-8);
        CuAssertDblEquals(tc,c2[ii*dim+ii],cov[ii*dim+ii],1e-4);
        CuAssertDblEquals(tc,1.0,var[ii],1e-4);
        for (jj = 0; jj < dim; jj++){
            if (jj != ii){
                CuAssertDblEquals(tc,c2[jj*dim+ii],cov[jj*dim+ii],1e-4);
            }
        }
    }

    
    probability_density_free(pdf); pdf = NULL;
    free(mean); mean = NULL;
    free(cov); cov = NULL;
    free(var); var = NULL;
}

double poisson_should_be10d(double * x){
    size_t ii; 
    double out = 1.0;
    for (ii = 0; ii < 10; ii++){
        out *= exp(1.0*x[ii] - 1.0/10.0*exp(x[ii]));
    }
    return out;
}

void Test_poisson_like(CuTest * tc)
{
    printf("Testing Function: poisson_like \n");
    size_t dim = 10;
    double likeparam[1] = {1.0 / (double) 10.0};
    double * count = darray_val(dim,1.0);
    struct Likelihood * like = 
    likelihood_alloc(dim,count,1,likeparam,dim,-10.0,10.0,
                    POISSON_LIKE);
        
    //size_t N = 20;
    double * xtest = darray_val(dim, 0.5);
    double val = function_train_eval(like->like,xtest);
    double shouldbe = poisson_should_be10d(xtest);
    CuAssertDblEquals(tc,shouldbe,val,1e-11);

    free(xtest); xtest = NULL;
    free(count); count = NULL;
    likelihood_free(like); like = NULL;

}

double evalLike(size_t N, double * coeff, double * points, double * data, 
                double noise)
{
    
    size_t ii;
    double detSig = pow(noise * noise,-0.5);
    double pre = pow(2.0 * M_PI, -0.5) * detSig;
    //printf("log pre = %G\n", log(pre));
    double out = 1.0;
    for (ii = 0; ii < N; ii++){
        double val1 = coeff[0] + coeff[1] * points[ii] - data[ii];
        double inexp = -0.5 * pow(noise,-2.0) * pow(val1,2);
        double thisdata = pre * exp(inexp);
        //double thisdata = exp(inexp);
        out *= thisdata;
    }
    return out;
}

void Test_linear_regression(CuTest * tc)
{
    printf("Testing Function: likelihood_linear_regression\n");
    //true: y = 2x + 4
    size_t dim = 1;
    size_t N = 5;
    double covariates[5] = {-2.0, -0.3, 0.0, 1.0, 2.6};
    //double data[5] = {0.0, 2.0, 4.0, 6.0, 8.0};
    double data[5] = {0.01, 2.2, 3.7, 6.8, 7.0};
    double noise = 1e-1;
    struct BoundingBox * bds = bounding_box_init(dim+1,-10.0,10.0);

    double m[2] = {0.0, 0.0};
    double c[4] = {4.0, 0.0, 0.0, 4.0};
    
    struct ProbabilityDensity * prior = probability_density_mvn(dim+1,m,c);
    struct Likelihood * like = likelihood_linear_regression(dim, N, data,
                                    covariates, noise, bds);
    
    struct BayesRule br;
    br.prior = prior;
    br.like = like;
    
    //printf("get laplace!\n");
    //struct ProbabilityDensity * post2 = NULL; //bayes_rule_compute(&br);
    
    double datasum = data[0]+data[1]+data[2]+data[3]+data[4];
    double dps = data[0]*covariates[0] + data[1]*covariates[1] + 
                 data[2]*covariates[2] + data[3]*covariates[3] +
                 data[4]*covariates[4];
    double xty[2] = {datasum, dps};
    
    double cprod = covariates[0]*covariates[0] + covariates[1]*covariates[1] + 
                 covariates[2]*covariates[2] + covariates[3]*covariates[3] +
                 covariates[4]*covariates[4];
    
    double csum = covariates[0] + covariates[1] + covariates[2] + 
                  covariates[3] + covariates[4];

    double xtx[4] = { 5.0, csum,
                      csum, cprod};
    
    double ss[4] = { 0.25*pow(noise,2) + xtx[0], csum,
                       csum, 0.25 * pow(noise,2) + xtx[3]};

    double sinv[4];// = {1.0/ss[0], 0.0, 0.0, 1.0/ss[3]};
    pinv(dim+1,dim+1,dim+1,ss,sinv,0.0);


    double ms[2] = { sinv[0] * xty[0] + sinv[1] * xty[1], 
                     sinv[2] * xty[0] + sinv[3] * xty[1]};
                     

    double cinv[4] = {
            xtx[0] * pow(noise,-2) + 0.25, xtx[1] * pow(noise,-2.0),
            xtx[2] * pow(noise,-2.0), xtx[3] * pow(noise,-2) + 0.25};

    double cshould[4];
    pinv(dim+1,dim+1,dim+1,cinv,cshould,0.0);
        
    /*
    printf("cov = \n");
    printf( "%G %G \n %G %G \n",cshould[0],cshould[2],cshould[1],cshould[3]);

    printf("mean should = \n");
    dprint(2,ms);
    */
        
    double likeeval = function_train_eval(br.like->like,ms) + br.like->logextra;
    double likeshould =  log(evalLike(N,ms,covariates,data,noise));

    CuAssertDblEquals(tc,likeshould,likeeval,1e-10);
                
    //printf("prior eval at mean = %G\n",probability_density_eval(br.prior,ms));

    //struct ProbabilityDensity * post2 = bayes_rule_compute(&br);
    //struct ProbabilityDensity * post2 = NULL;
    struct ProbabilityDensity * post = bayes_rule_laplace(&br);

    double * mean = probability_density_mean(post);
    double * cov = probability_density_cov(post);

    //double * mean2 = probability_density_mean(post2);
    //double * cov2 = probability_density_cov(post2);
     
    double diffmean = norm2diff(ms,mean,2);
    double diffcov = norm2diff(cshould,cov,4);
    CuAssertDblEquals(tc,0.0,diffmean,1e-12);
    CuAssertDblEquals(tc,0.0,diffcov,1e-12);
    free(mean); mean = NULL;
    free(cov); cov = NULL;
    probability_density_free(post); post = NULL;


    
    /*
    printf("mean2 = \n");
    dprint(2,mean2);

    printf("cov2 = \n");
    printf( "%G %G \n %G %G \n",cov2[0],cov2[2],cov2[1],cov2[3]);
    */

    //diffmean = norm2diff(ms,mean2,2);
    //diffcov = norm2diff(cshould,cov2,4);
    //CuAssertDblEquals(tc,0.0,diffmean,1e-12);
    //CuAssertDblEquals(tc,0.0,diffcov,1e-12);

    bounding_box_free(bds); bds = NULL;
    likelihood_free(like); like = NULL;
    probability_density_free(prior); prior = NULL;
    
  
    //probability_density_free(post2); post2 = NULL;
    //free(mean2); mean2 = NULL;
    //free(cov2); cov2 = NULL;
}


void Test_cdf_normal(CuTest * tc)
{
    printf("Testing Function: cdf_normal\n");
    
    double mean = 2.0;
    double std = 2.0;
    
    int n = 1;
    double val1 = cdf_normal(mean,std,mean+n*std); 
    double val2 = cdf_normal(mean,std,mean-n*std);
    CuAssertDblEquals(tc,0.682689492137,val1-val2,1e-8);

    n = 2;
    val1 = cdf_normal(mean,std,mean+n*std); 
    val2 = cdf_normal(mean,std,mean-n*std);
    CuAssertDblEquals(tc,0.954499736104,val1-val2,1e-8);


    n = 3;
    val1 = cdf_normal(mean,std,mean+n*std); 
    val2 = cdf_normal(mean,std,mean-n*std);
    CuAssertDblEquals(tc,0.997300203937,val1-val2,1e-8);

    n = 4;
    val1 = cdf_normal(mean,std,mean+n*std); 
    val2 = cdf_normal(mean,std,mean-n*std);
    CuAssertDblEquals(tc,0.999936657516,val1-val2,1e-8);

    n = 5;
    val1 = cdf_normal(mean,std,mean+n*std); 
    val2 = cdf_normal(mean,std,mean-n*std);
    CuAssertDblEquals(tc,0.999999426697,val1-val2,1e-8);

    n = 6;
    val1 = cdf_normal(mean,std,mean+n*std); 
    val2 = cdf_normal(mean,std,mean-n*std);
    CuAssertDblEquals(tc,0.999999998027,val1-val2,1e-8);
}

void Test_icdf_normal(CuTest * tc)
{
    printf("Testing Function: icdf_normal\n");
    
    double mean = 0.6;
    double std = 8.0;

    double x = 0.3;
    double val = cdf_normal(mean,std,x);
    double ix = icdf_normal(mean,std,val);
    //CuAssertDblEquals(tc,x,ix,1e-12);


    x = mean + 4.0 * std;
    val = cdf_normal(mean,std,x);
    ix = icdf_normal(mean,std,val);
    //printf("val=%G\n",val);
    //printf("diff=%G\n",x-ix);
    CuAssertDblEquals(tc,x,ix,1e-10);


    x = mean - 4.0 * std;
    val = cdf_normal(mean,std,x);
    ix = icdf_normal(mean,std,val);
    //printf("val=%G\n",val);
    //printf("diff=%G\n",x-ix);
    CuAssertDblEquals(tc,x,ix,1e-10);
    
}


CuSuite * ProbGetSuite(){
    //printf("----------------------------\n");

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_stdnorm);
    SUITE_ADD_TEST(suite, Test_mvn1d); 
    SUITE_ADD_TEST(suite, Test_mvn);
    SUITE_ADD_TEST(suite, Test_lt_ser);
    SUITE_ADD_TEST(suite, Test_pdf_ser);
    /* SUITE_ADD_TEST(suite, Test_log_gradient_eval); */
    /* SUITE_ADD_TEST(suite, Test_log_hessian_eval); */
    SUITE_ADD_TEST(suite, Test_laplace);
    /* SUITE_ADD_TEST(suite, Test_poisson_like); */
    /* SUITE_ADD_TEST(suite, Test_linear_regression);  */
    SUITE_ADD_TEST(suite, Test_cdf_normal);
    SUITE_ADD_TEST(suite, Test_icdf_normal);

    return suite;
}
