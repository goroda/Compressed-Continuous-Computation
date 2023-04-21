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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "lib_linalg.h"
#include "lib_probability.h"
#include "probability.h"
#include "c3_interface.h"

#include "CuTest.h"

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884197169399375105820974944
#endif 

static int rosenbrock(size_t N, const double * x, double * out, void * args)
{
    (void)(args);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = 100.0 * pow( x[ii*2 + 1] - pow(x[ii*2],2), 2) +
            pow(1.0-x[ii*2],2);
    }
    return 0;
}

void Test_sobol(CuTest * tc)
{
    printf("Testing Function: function_train_sobol_sensitivities \n");

    size_t dim = 2;
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,rosenbrock,NULL);
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-2);
    ope_opts_set_ub(opts,2);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);

    int verbose = 0;
    size_t init_rank = 5;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(-2.0,2.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);

    struct C3SobolSensitivity * sobol = c3_sobol_sensitivity_calculate(ft,dim);

    /* c3_sobol_sensitivity_print(sobol); */
    /* printf("Main effects: "); dprint(dim, main_effects); */
    /* printf("Total effects: "); dprint(dim, total_effects); */
    /* printf("Interaction effects: \n"); */
    /* dprint2d_col(2,2,interact_effects); */


    double var = c3_sobol_sensitivity_get_variance(sobol);

    CuAssertDblEquals(tc,7.0363551328e-01,
                      c3_sobol_sensitivity_get_total(sobol,0),
                      1e-9);
    CuAssertDblEquals(tc,5.0253108617e-01,
                      c3_sobol_sensitivity_get_total(sobol,1),
                      1e-9);
    
    size_t main_effect = 0;
    CuAssertDblEquals(tc,4.9746891383e-01,
                      c3_sobol_sensitivity_get_interaction(
                          sobol,1,&main_effect)/var,1e-9);
    main_effect = 1;
    CuAssertDblEquals(tc,2.9636448672e-01,
                      c3_sobol_sensitivity_get_interaction(
                          sobol,1,&main_effect)/var,1e-9);


    size_t main_effects[2] = {0,1};
    CuAssertDblEquals(tc,2.0616659946e-01,
                      c3_sobol_sensitivity_get_interaction(
                          sobol,2,main_effects)/var,1e-9);


    c3_sobol_sensitivity_free(sobol);
    function_train_free(ft);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim, start);
    fwrap_destroy(fw);
}

static int gfunction(size_t N, const double * x, double * out, void * args)
{
    (void)(args);
    double a[6] = {0.0, 0.5, 3.0, 9.0, 99.0, 99.9};
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = 1.0;
        for (size_t jj = 0; jj < 6; jj++){
            double temp = (sqrt(pow(4.0*x[ii*6+jj] - 2,2)) + a[jj]) / (1 + a[jj]);
            out[ii] *= temp;
        }
    }
    return 0;
}

void Test_sobol2(CuTest * tc)
{
    printf("Testing Function: function_train_sobol_sensitivities (2) \n");

    size_t dim = 6;
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,gfunction,NULL);
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,0);
    ope_opts_set_ub(opts,1);
    ope_opts_set_maxnum(opts,50);
    ope_opts_set_tol(opts,1e-12);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);

    int verbose = 0;
    size_t init_rank = 2;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(0.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    c3approx_set_cross_tol(c3a,1e-14);
    c3approx_set_round_tol(c3a,1e-14);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);

    size_t max_order = dim;
    struct C3SobolSensitivity * sobol =
        c3_sobol_sensitivity_calculate(ft,max_order);
    /* c3_sobol_sensitivity_print(sobol); */

    printf("\n");

    double a[6] = {0.0,0.5,3.0,9.0,99.0,99.0};
    double main_should[6];
    for (size_t ii = 0; ii < 6; ii++){
        main_should[ii] = 1.0/3.0 * pow(1.0 + a[ii],-2);
    }

    /* printf("main should: "); dprint(6,main_should); */
    
    // check first order effects;
    for (size_t ii = 0; ii < dim; ii++){
        CuAssertDblEquals(tc,main_should[ii],
                          c3_sobol_sensitivity_get_interaction(sobol,1,&ii),1e-2);
    }

    // check second order effects;
    size_t vars[2];
    for (size_t ii = 0; ii < dim; ii++){
        for (size_t jj = ii+1; jj < dim; jj++){
            vars[0] = ii;
            vars[1] = jj;
            /* printf("vari (%zu,%zu) = %G\n",ii,jj, */
            /*        c3_sobol_sensitivity_get_interaction(sobol,2,vars)); */
            /* printf("-> should = %G\n", */
            /*        main_should[ii]*main_should[jj]); */

            CuAssertDblEquals(tc,main_should[ii]*main_should[jj],
                              c3_sobol_sensitivity_get_interaction(
                                  sobol,2,vars),1e-2);
        }
    }


    // check third order effects;
    size_t vars3[3];
    for (size_t ii = 0; ii < dim; ii++){
        for (size_t jj = ii+1; jj < dim; jj++){
            for (size_t kk = jj+1; kk < dim; kk++){
                vars3[0] = ii;
                vars3[1] = jj;
                vars3[2] = kk;
                /* printf("vari (%zu,%zu,%zu) = %G\n",ii,jj,kk, */
                /*    c3_sobol_sensitivity_get_interaction(sobol,3,vars3)); */
                /* printf("-> should = %G\n", */
                /*        main_should[ii]*main_should[jj]*main_should[kk]); */
                CuAssertDblEquals(tc,
                                  main_should[ii]*main_should[jj]*main_should[kk],
                                  c3_sobol_sensitivity_get_interaction(
                                      sobol,3,vars3),1e-2);
            }
        }
    }

    // check fourth order effects
    size_t vars4[4];
    for (size_t ii = 0; ii < dim; ii++){
        for (size_t jj = ii+1; jj < dim; jj++){
            for (size_t kk = jj+1; kk < dim; kk++){
                for (size_t ll = kk+1; ll < dim; ll++){
                    vars4[0] = ii;
                    vars4[1] = jj;
                    vars4[2] = kk;
                    vars4[3] = ll;
                    /* printf("vari (%zu,%zu,%zu,%zu) = %G\n",ii,jj,kk,ll, */
                    /*        c3_sobol_sensitivity_get_interaction( */
                    /*            sobol,4,vars4)); */
                    /* printf("-> should = %G\n", */
                    /*        main_should[ii]*main_should[jj]* */
                    /*        main_should[kk]*main_should[ll]); */
                    CuAssertDblEquals(tc,
                                      main_should[ii]*main_should[jj]*
                                      main_should[kk]*main_should[ll],
                                      c3_sobol_sensitivity_get_interaction(
                                          sobol,4,vars4),1e-2);
                }
            }
        }
    }

    // check fifth order effects
    size_t vars5[5];
    for (size_t ii = 0; ii < dim; ii++){
        for (size_t jj = ii+1; jj < dim; jj++){
            for (size_t kk = jj+1; kk < dim; kk++){
                for (size_t ll = kk+1; ll < dim; ll++){
                    for (size_t zz = ll+1; zz < dim; zz++){
                        vars5[0] = ii;
                        vars5[1] = jj;
                        vars5[2] = kk;
                        vars5[3] = ll;
                        vars5[4] = zz;
                        /* printf("vari (%zu,%zu,%zu,%zu,%zu) = %G\n", */
                        /*        ii,jj,kk,ll,zz, */
                        /*        c3_sobol_sensitivity_get_interaction( */
                        /*            sobol,5,vars5)); */
                        /* printf("-> should = %G\n", */
                        /*        main_should[ii]*main_should[jj]* */
                        /*        main_should[kk]*main_should[ll]* */
                        /*        main_should[zz]); */
                        CuAssertDblEquals(tc,
                                          main_should[ii]*main_should[jj]*
                                          main_should[kk]*main_should[ll]*
                                          main_should[zz],
                                          c3_sobol_sensitivity_get_interaction(
                                              sobol,5,vars5),1e-2);
                    }
                }
            }
        }
    }

    // check sixth order effects
    size_t vars6[6];
    for (size_t ii = 0; ii < dim; ii++){
        for (size_t jj = ii+1; jj < dim; jj++){
            for (size_t kk = jj+1; kk < dim; kk++){
                for (size_t ll = kk+1; ll < dim; ll++){
                    for (size_t zz = ll+1; zz < dim; zz++){
                        for (size_t qq = zz+1; qq < dim; qq++){
                            vars6[0] = ii;
                            vars6[1] = jj;
                            vars6[2] = kk;
                            vars6[3] = ll;
                            vars6[4] = zz;
                            vars6[5] = qq;
                            /* printf("vari (%zu,%zu,%zu,%zu,%zu) = %G\n", */
                            /*        ii,jj,kk,ll,zz, */
                            /*        c3_sobol_sensitivity_get_interaction( */
                            /*            sobol,6,vars6)); */
                            /* printf("-> should = %G\n", */
                            /*        main_should[ii]*main_should[jj]* */
                            /*        main_should[kk]*main_should[ll]* */
                            /*        main_should[zz]*main_should[qq]); */
                            CuAssertDblEquals(tc,
                                              main_should[ii]*main_should[jj]*
                                              main_should[kk]*main_should[ll]*
                                              main_should[zz]*main_should[qq],
                                              c3_sobol_sensitivity_get_interaction(
                                                  sobol,6,vars6),1e-2);
                        }
                    }
                }
            }
        }
    }
    
    /* printf("vari (0,1) = %G\n",c3_sobol_sensitivity_get_interaction(sobol,2,interact2)); */

    c3_sobol_sensitivity_free(sobol);    
    function_train_free(ft);
    c3approx_destroy(c3a);
    one_approx_opts_free_deep(&qmopts);
    free_dd(dim, start);
    fwrap_destroy(fw);
}



/* void Test_stdnorm(CuTest * tc) */
/* { */
/*     printf("Testing Function: probability_density_standard_normal \n"); */
/*     size_t dim = 10; */
/*     struct ProbabilityDensity * pdf = probability_density_standard_normal(dim); */
/*     CuAssertIntEquals(tc,1,pdf!=NULL); */
/*     double * x = calloc_double(dim); */
/*     size_t nrand = 10; */

/*     double pdfval = probability_density_eval(pdf,x); */
/*     double shouldbe = pow(2.0*M_PI,-(double) dim/2.0); */
/*     CuAssertDblEquals(tc,shouldbe,pdfval,1e-13); */

/*     for (size_t ii = 0; ii < nrand; ii++){ */
/*         for (size_t jj = 0; jj < dim;jj++){ */
/*             x[jj] = randn(); */
/*         } */
/*         pdfval = probability_density_eval(pdf,x); */
/*         shouldbe = pow(2.0*M_PI,-(double) dim/2.0); */
/*         CuAssertDblEquals(tc,shouldbe,pdfval,1e-3); */
/*     } */
    
/*     double * mean = probability_density_mean(pdf); */
/*     double * cov = probability_density_cov(pdf); */
/*     double * var = probability_density_var(pdf); */

/* //    dprint(10,var); */
/*     size_t ii,jj; */
/*     for (ii = 0; ii < dim; ii++){ */
/*         CuAssertDblEquals(tc,0.0,mean[ii],1e-4); */
/*         CuAssertDblEquals(tc,1.0,cov[ii*dim+ii],1e-4); */
/*         CuAssertDblEquals(tc,1.0,var[ii],1e-4); */
/*         for (jj = 0; jj < dim; jj++){ */
/*             if (jj != ii){ */
/*                 CuAssertDblEquals(tc,0.0,cov[jj*dim+ii],1e-11); */
/*             } */
/*         } */
/*     } */
/*     probability_density_free(pdf); */
/*     free(x); */
/*     free(mean); */
/*     free(cov); */
/*     free(var); */
/* } */

/* void Test_mvn1d(CuTest * tc) */
/* { */
/*     printf("Testing Function: probability_density_mvn for 1d \n"); */
/*     size_t dim = 1; */
/*     double m[1] = {1.0}; */
/*     double c[1] = {0.25}; */

/*     struct ProbabilityDensity * pdf = probability_density_mvn(dim,m,c); */
/*     double x[1] = {1.0}; */
/*     double pdfval = probability_density_eval(pdf,x); */
        
/*     double shouldbe = 1.0 / sqrt(2.0 * M_PI * c[0]); */
/*     shouldbe = shouldbe * exp(-0.5 * pow(x[0]-m[0],2.0) / c[0] ); */
    
/*     //printf("detinv=%G\n",pdf->lt->detinv); */
/*     //double trans = 2.0*x[0]-2.0; */
/*     //printf("trans=%G\n",trans); */
/*     //double s2 = 1.0 / sqrt(2.0 * M_PI) * exp(-0.5 * trans * trans) * 2.0; */
/*     //printf("s2=%G\n",s2); */
    
    
/*     CuAssertDblEquals(tc,sqrt(c[0]),pdf->lt->A[0],1e-13); */
/*     CuAssertDblEquals(tc,m[0],pdf->lt->b[0],1e-13); */
/*     /\* */
/*     printf("A = %G\n ",pdf->lt->A[0]); */
/*     printf("b = %G\n ",pdf->lt->b[0]); */
/*     printf("Ainv = %G\n ",pdf->lt->Ainv[0]); */
/*     printf("binv = %G\n ",pdf->lt->binv[0]); */
/*     *\/ */
/*     CuAssertDblEquals(tc,shouldbe,pdfval,1e-13); */

/*     double * mean = probability_density_mean(pdf); */
/*     double * cov = probability_density_cov(pdf); */

/*     double * var = probability_density_var(pdf); */
/*     size_t ii; */
/*     for (ii = 0; ii < dim; ii++){ */
/*         CuAssertDblEquals(tc,m[ii],mean[ii],1e-4); */
/*         CuAssertDblEquals(tc,c[ii*dim+ii],cov[ii*dim+ii],1e-4); */
/*         CuAssertDblEquals(tc,c[ii*dim+ii],var[ii],1e-4); */
/*     } */

/*     probability_density_free(pdf); */
/*     free(mean); */
/*     free(cov); */
/*     free(var); */
/* } */


/* void Test_mvn(CuTest * tc) */
/* { */
/*     printf("Testing Function: probability_density_mvn \n"); */
/*     size_t dim = 2; */
/*     double m[2] = {1.0, 2.0}; */
/*     double c[4] = {1.0, 0.5, 0.5, 1.0}; */

/*     struct ProbabilityDensity * pdf = probability_density_mvn(dim,m,c); */
/*     double x[2] = {1.0,0.0}; */
/*     double pdfval = probability_density_eval(pdf,x); */
/*     double rho = c[2] / (sqrt(c[0]*c[3])); */

/*     double z = pow(x[0]-m[0],2.0)/c[0] -  */
/*               2.0 * rho * (x[0]-m[0])*(x[1]-m[1])/(sqrt(c[0]*c[3])) +  */
/*              pow(x[1] - m[1],2)/c[3]; */
/*     //printf("z=%G\n",z); */
/*     double shouldbe = 1.0 / 2.0 / M_PI / sqrt(c[0]*c[3]) / sqrt(1.0-pow(rho,2.0)); */
/*     shouldbe *= exp(-0.5 * z / (1.0-pow(rho,2.0))); */
    
/*     //printf("Ainv = \n"); */
/*     //dprint2d_col(2,2,pdf->lt->Ainv); */

/*     //printf("Ainv det = %G\n",pdf->lt->detinv); */
/*     CuAssertDblEquals(tc,shouldbe,pdfval,1e-6); */

/*     double * mean = probability_density_mean(pdf); */
/*     double * cov = probability_density_cov(pdf); */
/*     /\* */
/*     printf("mean=\n"); */
/*     dprint(dim,mean); */
/*     printf("cov=\n"); */
/*     dprint2d_col(2,2,cov); */
/*     *\/ */
/*     double * var = probability_density_var(pdf); */
/*     size_t ii,jj;  */
/*     for (ii = 0; ii < dim; ii++){ */
/*         CuAssertDblEquals(tc,m[ii],mean[ii],1e-4); */
/*         CuAssertDblEquals(tc,c[ii*dim+ii],cov[ii*dim+ii],1e-4); */
/*         CuAssertDblEquals(tc,1.0,var[ii],1e-4); */
/*         for (jj = 0; jj < dim; jj++){ */
/*             if (jj != ii){ */
/*                 CuAssertDblEquals(tc,c[jj*dim+ii],cov[jj*dim+ii],1e-4); */
/*             } */
/*         } */
/*     } */

/*     probability_density_free(pdf); */
/*     free(mean); */
/*     free(cov); */
/*     free(var); */
/* } */

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
            CuAssertDblEquals(tc, -gradShould[0], gradIs[0], 1e-4);
            CuAssertDblEquals(tc, -gradShould[1], gradIs[1], 1e-4);
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

            CuAssertDblEquals(tc, -hessShould[0], hessIs[0], 1e-2);
            CuAssertDblEquals(tc, -hessShould[1], hessIs[1], 1e-2);
            CuAssertDblEquals(tc, -hessShould[2], hessIs[2], 1e-2);
            CuAssertDblEquals(tc, -hessShould[3], hessIs[3], 1e-2);

            free(hessShould); hessShould = NULL;
            free(hessIs); hessIs = NULL;
            //printf("\n");
        }
    }
    probability_density_free(pdf); pdf = NULL;
    free(xtest); xtest = NULL;
}

double c3opt_lap_test(size_t dx, const double * x, double * grad, void * args)
{
    (void)(dx);
    double * prec = args;
    double t1 = -(prec[0] * x[0] + prec[2] * x[1]);
    double t2 = -(prec[1] * x[0] + prec[3] * x[1]); 
    if (grad != NULL){
        grad[0] = t1;
        grad[1] = t2;
    }
    /* double inexp = exp(0.5 * (x[0]*t1 + x[1]*t2)); */
    /* double det = sqrt(prec[0]*prec[3] - prec[1]*prec[2]); */
    /* return -det*inexp; */
    double out = -0.5 * (x[0]*t1 + x[1]*t2);
    return out;
}


/* void Test_laplace(CuTest * tc) */
/* { */
/*     printf("Testing Function: probability_density_laplace \n"); */
/*     size_t dim = 2; */
/*     double m[2] = {0.0, 0.0}; */
/*     double c[4] = {1.0, 0.5, 0.5, 1.0}; */
/*     double c2[4] = {1.0, 0.5, 0.5, 1.0}; */
    
/*     double start[2] = {0.5, 1.2}; */

/*     double prec[4]; */
/*     pinv(dim,dim,dim,c,prec,0.0); */

/*     struct ProbabilityDensity * pdf =  */
/* //        probability_density_laplace(gradProb,hessProb,prec,dim,start); */
/*         probability_density_laplace(c3opt_lap_test,hessProb,prec,dim,start); */

/*     double * mean = probability_density_mean(pdf); */
/*     double * cov = probability_density_cov(pdf); */
/*     double * var = probability_density_var(pdf); */
/*     size_t ii,jj;  */
/*     for (ii = 0; ii < dim; ii++){ */
/*         CuAssertDblEquals(tc,m[ii],mean[ii],1e-4); */
/*         CuAssertDblEquals(tc,c2[ii*dim+ii],cov[ii*dim+ii],1e-4); */
/*         CuAssertDblEquals(tc,1.0,var[ii],1e-4); */
/*         for (jj = 0; jj < dim; jj++){ */
/*             if (jj != ii){ */
/*                 CuAssertDblEquals(tc,c2[jj*dim+ii],cov[jj*dim+ii],1e-4); */
/*             } */
/*         } */
/*     } */
    
/*     probability_density_free(pdf); pdf = NULL; */
/*     free(mean); mean = NULL; */
/*     free(cov); cov = NULL; */
/*     free(var); var = NULL; */
/* } */

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

/* void Test_likelihood_linear_regression(CuTest * tc) */
/* { */
/*     printf("Testing Function: likelihood_linear_regression \n"); */
/*     //true: y = 2x + 4 */
/*     size_t dim = 1; */
/*     size_t N = 5; */
/*     double covariates[5] = {-2.0, -0.3, 0.0, 1.0, 2.6}; */
/*     //double data[5] = {0.0, 2.0, 4.0, 6.0, 8.0}; */
/*     double data[5] = {0.01, 2.2, 3.7, 6.8, 7.0}; */
/*     double noise = 1e-1; */
/*     struct BoundingBox * bds = bounding_box_init(dim+1,-10.0,10.0); */

/*     struct Likelihood * like = NULL; */
/*     like = likelihood_linear_regression(dim, N, data, */
/*                                         covariates, noise, bds); */

/* //    printf("done with construction\n"); */
/*     double ms[2] = {4.0, 2.0}; */
/*     double likeeval = function_train_eval(like->like,ms) + like->logextra; */
/*     double likeshould = log(evalLike(N,ms,covariates,data,noise)); */

/*     double diff = fabs(likeshould-likeeval)/fabs(likeshould); */
/*     CuAssertDblEquals(tc,0,diff,1e-3); */
    
/*     likelihood_free(like); */
/*     bounding_box_free(bds); */
/* } */


/* void Test_linear_regression(CuTest * tc) */
/* { */
/*     printf("Testing Function: linear_regression (1)\n"); */
/*     //true: y = 2x + 4 */
/*     size_t dim = 1; */
/*     size_t N = 5; */
/*     double covariates[5] = {-2.0, -0.3, 0.0, 1.0, 2.6}; */
/*     //double data[5] = {0.0, 2.0, 4.0, 6.0, 8.0}; */
/*     double data[5] = {0.01, 2.2, 3.7, 6.8, 7.0}; */
/*     double noise = 1e-1; */
/*     struct BoundingBox * bds = bounding_box_init(dim+1,-10.0,10.0); */

/*     double m[2] = {0.0, 0.0}; */
/*     double c[4] = {4.0, 0.0, 0.0, 4.0}; */
    
/*     struct ProbabilityDensity * prior = probability_density_mvn(dim+1,m,c); */
/*     struct Likelihood * like = likelihood_linear_regression(dim, N, data, */
/*                                     covariates, noise, bds); */
    
/*     struct BayesRule br; */
/*     br.prior = prior; */
/*     br.like = like; */
    
/*     //printf("get laplace!\n"); */
/*     //struct ProbabilityDensity * post2 = NULL; //bayes_rule_compute(&br); */
    
/*     double datasum = data[0]+data[1]+data[2]+data[3]+data[4]; */
/*     double dps = data[0]*covariates[0] + data[1]*covariates[1] +  */
/*                  data[2]*covariates[2] + data[3]*covariates[3] + */
/*                  data[4]*covariates[4]; */
/*     double xty[2] = {datasum, dps}; */
    
/*     double cprod = covariates[0]*covariates[0] + covariates[1]*covariates[1] +  */
/*                  covariates[2]*covariates[2] + covariates[3]*covariates[3] + */
/*                  covariates[4]*covariates[4]; */
    
/*     double csum = covariates[0] + covariates[1] + covariates[2] +  */
/*                   covariates[3] + covariates[4]; */

/*     double xtx[4] = { 5.0, csum, */
/*                       csum, cprod}; */
    
/*     double ss[4] = { 0.25*pow(noise,2) + xtx[0], csum, */
/*                        csum, 0.25 * pow(noise,2) + xtx[3]}; */

/*     double sinv[4];// = {1.0/ss[0], 0.0, 0.0, 1.0/ss[3]}; */
/*     pinv(dim+1,dim+1,dim+1,ss,sinv,0.0); */


/*     double ms[2] = { sinv[0] * xty[0] + sinv[1] * xty[1],  */
/*                      sinv[2] * xty[0] + sinv[3] * xty[1]}; */
                     

/*     double cinv[4] = { */
/*             xtx[0] * pow(noise,-2) + 0.25, xtx[1] * pow(noise,-2.0), */
/*             xtx[2] * pow(noise,-2.0), xtx[3] * pow(noise,-2) + 0.25}; */

/*     double cshould[4]; */
/*     pinv(dim+1,dim+1,dim+1,cinv,cshould,0.0); */
        
/*     /\* */
/*     printf("cov = \n"); */
/*     printf( "%G %G \n %G %G \n",cshould[0],cshould[2],cshould[1],cshould[3]); */

/*     printf("mean should = \n"); */
/*     dprint(2,ms); */
/*     *\/ */
        
/*     double likeeval = function_train_eval(br.like->like,ms) + br.like->logextra; */
/*     double likeshould =  log(evalLike(N,ms,covariates,data,noise)); */

/*     double diff = fabs(likeshould-likeeval)/fabs(likeshould); */
/*     CuAssertDblEquals(tc,0.0,diff,1e-3); */
                
/*     //printf("prior eval at mean = %G\n",probability_density_eval(br.prior,ms)); */

   
/*     //struct ProbabilityDensity * post2 = NULL; */
/*     printf("bayes_rule_laplace\n"); */
/*     struct ProbabilityDensity * post = bayes_rule_laplace(&br); */

/*     double * mean = probability_density_mean(post); */
/*     double * cov = probability_density_cov(post); */

/*     printf("mean = \n"); */
/*     dprint(2,mean); */

/*     printf("cov = \n"); */
/*     printf( "%G %G \n %G %G \n",cov[0],cov[2],cov[1],cov[3]); */

     
/*     double diffmean = norm2diff(ms,mean,2); */
/*     double diffcov = norm2diff(cshould,cov,4); */
/*     CuAssertDblEquals(tc,0.0,diffmean,1e-4); */
/*     CuAssertDblEquals(tc,0.0,diffcov,1e-4); */
/*     free(mean); mean = NULL; */
/*     free(cov); cov = NULL; */
/*     probability_density_free(post); post = NULL; */

   
/*     struct ProbabilityDensity * post2 = bayes_rule_compute(&br);  */
/*     double * mean2 = probability_density_mean(post2); */
/*     double * cov2 = probability_density_cov(post2); */

/*     printf("mean2 = \n"); */
/*     dprint(2,mean2); */

/*     printf("cov2 = \n"); */
/*     printf( "%G %G \n %G %G \n",cov2[0],cov2[2],cov2[1],cov2[3]); */

/*     diffmean = norm2diff(ms,mean2,2); */
/*     diffcov = norm2diff(cshould,cov2,4); */
/*     CuAssertDblEquals(tc,0.0,diffmean,1e-4); */
/*     CuAssertDblEquals(tc,0.0,diffcov,1e-4); */

/*     bounding_box_free(bds); bds = NULL; */
/*     likelihood_free(like); like = NULL; */
/*     probability_density_free(prior); prior = NULL; */
    
  
/*     probability_density_free(post2); post2 = NULL; */
/*     free(mean2); mean2 = NULL; */
/*     free(cov2); cov2 = NULL; */
/* } */

/* void Test_linear_regression2(CuTest * tc) */
/* { */
/*     printf("Testing Function: likelihood_linear_regression (2)\n"); */


/*     size_t dim = 2; */
/*     size_t N = 5; */
/*     double * covariates = calloc_double(dim*N); */
/*     double * data = calloc_double(N); */
/*     double noise = 1e-1; */
/*     // y = 0.2 + x - y */
/*     printf("covariates are \n"); */
/*     for (size_t ii = 0; ii < N; ii++){ */
/*         covariates[ii*dim+1] = randn(); */
/*         covariates[ii*dim+2] = randn(); */
/*         data[ii] = 0.2 + covariates[ii*dim+1] - covariates[ii*dim+2] + noise*randn(); */
/*         dprint(2,covariates + ii*dim); */
/*     } */

/*     struct BoundingBox * bds = bounding_box_init(dim+1,-10.0,10.0); */

/*     double m[3] = {0.0, 0.0, 0.0}; */
/*     double c[9] = {4.0, 0.0, 0.0, */
/*                    0.0, 4.0, 0.0, */
/*                    0.0, 0.0, 4.0}; */

    
/*     struct ProbabilityDensity * prior = probability_density_mvn(dim+1,m,c); */

/*     printf("compute like\n"); */
/*     struct Likelihood * like = likelihood_linear_regression(dim, N, data, */
/*                                     covariates, noise, bds); */
    
/*     struct BayesRule br; */
/*     br.prior = prior; */
/*     br.like = like; */

/*     printf("do laplace\n"); */
/*     struct ProbabilityDensity * post = bayes_rule_laplace(&br); */

/*     double * mean = probability_density_mean(post); */
/*     double * cov = probability_density_cov(post); */

/*     printf("mean = \n"); */
/*     dprint(3,mean); */

/*     printf("cov = \n"); */
/*     dprint2d_col(3,3,cov); */

     
/*     /\* double diffmean = norm2diff(ms,mean,2); *\/ */
/*     /\* double diffcov = norm2diff(cshould,cov,4); *\/ */
/*     /\* CuAssertDblEquals(tc,0.0,diffmean,1e-4); *\/ */
/*     /\* CuAssertDblEquals(tc,0.0,diffcov,1e-4); *\/ */
/*     /\* free(mean); mean = NULL; *\/ */
/*     /\* free(cov); cov = NULL; *\/ */
/*     /\* probability_density_free(post); post = NULL; *\/ */

   
/*     /\* struct ProbabilityDensity * post2 = bayes_rule_compute(&br);  *\/ */
/*     /\* double * mean2 = probability_density_mean(post2); *\/ */
/*     /\* double * cov2 = probability_density_cov(post2); *\/ */

/*     /\* printf("mean2 = \n"); *\/ */
/*     /\* dprint(2,mean2); *\/ */

/*     /\* printf("cov2 = \n"); *\/ */
/*     /\* printf( "%G %G \n %G %G \n",cov2[0],cov2[2],cov2[1],cov2[3]); *\/ */

/*     /\* diffmean = norm2diff(ms,mean2,2); *\/ */
/*     /\* diffcov = norm2diff(cshould,cov2,4); *\/ */
/*     /\* CuAssertDblEquals(tc,0.0,diffmean,1e-4); *\/ */
/*     /\* CuAssertDblEquals(tc,0.0,diffcov,1e-4); *\/ */

/*     /\* bounding_box_free(bds); bds = NULL; *\/ */
/*     /\* likelihood_free(like); like = NULL; *\/ */
/*     /\* probability_density_free(prior); prior = NULL; *\/ */
    
  
/*     /\* probability_density_free(post2); post2 = NULL; *\/ */
/*     /\* free(mean2); mean2 = NULL; *\/ */
/*     /\* free(cov2); cov2 = NULL; *\/ */
/* } */


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
    SUITE_ADD_TEST(suite, Test_sobol);
    SUITE_ADD_TEST(suite, Test_sobol2);
    
    /* SUITE_ADD_TEST(suite, Test_stdnorm); */
    /* SUITE_ADD_TEST(suite, Test_mvn1d); */
    /* SUITE_ADD_TEST(suite, Test_mvn); */
    /* SUITE_ADD_TEST(suite, Test_lt_ser); */
    /* SUITE_ADD_TEST(suite, Test_pdf_ser); */
    /* SUITE_ADD_TEST(suite, Test_log_gradient_eval); */
    /* SUITE_ADD_TEST(suite, Test_log_hessian_eval); */
    /* SUITE_ADD_TEST(suite, Test_laplace); */
    /* SUITE_ADD_TEST(suite, Test_likelihood_linear_regression); */
    /* SUITE_ADD_TEST(suite, Test_linear_regression); */
    /* SUITE_ADD_TEST(suite, Test_linear_regression2); */
    /* SUITE_ADD_TEST(suite, Test_cdf_normal); */
    /* SUITE_ADD_TEST(suite, Test_icdf_normal); */
    /* /\* SUITE_ADD_TEST(suite, Test_poisson_like); *\/ */

    return suite;
}

double poisson_should_be10d(double * x){
    size_t ii; 
    double out = 1.0;
    for (ii = 0; ii < 10; ii++){
        out *= exp(1.0*x[ii] - 1.0/10.0*exp(x[ii]));
    }
    return out;
}

/* void Test_poisson_like(CuTest * tc) */
/* { */
/*     printf("Testing Function: poisson_like \n"); */
/*     size_t dim = 10; */
/*     double likeparam[1] = {1.0 / (double) 10.0}; */
/*     double * count = darray_val(dim,1.0); */
/*     struct Likelihood * like =  */
/*     likelihood_alloc(dim,count,1,likeparam,dim,-10.0,10.0, */
/*                     POISSON_LIKE); */
        
/*     //size_t N = 20; */
/*     double * xtest = darray_val(dim, 0.5); */
/*     double val = function_train_eval(like->like,xtest); */
/*     double shouldbe = poisson_should_be10d(xtest); */
/*     CuAssertDblEquals(tc,shouldbe,val,1e-11); */

/*     free(xtest); xtest = NULL; */
/*     free(count); count = NULL; */
/*     likelihood_free(like); like = NULL; */

/* } */
