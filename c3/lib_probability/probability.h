// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016-2017 Sandia Corporation
// Copyright (c) 2017 NTESS, LLC.

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



#ifndef PROBABILITY_H
#define PROBABILITY_H

#include "lib_clinalg.h"

struct C3SobolSensitivity;
struct C3SobolSensitivity *
c3_sobol_sensitivity_calculate(const struct FunctionTrain *,size_t);
void c3_sobol_sensitivity_free(struct C3SobolSensitivity *);
double c3_sobol_sensitivity_get_interaction(const struct C3SobolSensitivity *,
                                            size_t leni, const size_t * xi);
double c3_sobol_sensitivity_get_total(const struct C3SobolSensitivity *, size_t);
double c3_sobol_sensitivity_get_main(const struct C3SobolSensitivity *, size_t);
double c3_sobol_sensitivity_get_variance(const struct C3SobolSensitivity *);
void c3_sobol_sensitivity_print(const struct C3SobolSensitivity *);
void c3_sobol_sensitivity_apply_external(const struct C3SobolSensitivity *,
                                         void (*)(double, size_t, size_t*,void*),
                                         void *);


enum lt_matrix_type {LT, SYMMETRIC, GEN};
enum pdf_type {GAUSSIAN, GENERAL};
enum likelihood_type {GAUSSIAN_LIKE, POISSON_LIKE, GENERIC_LIKE};

// y = Ax + b
struct LinearTransform
{
    double * A;
    double * b;
    double det;
    
    //transformation: x  = Ainv * y - binv
    double * Ainv;
    double * binv;
    double detinv;

    size_t dimin;
    size_t dimout;

    enum lt_matrix_type mt; // matrix type 
    enum lt_matrix_type mti; //inverse matrix type
};

struct ProbabilityDensity
{
    struct FunctionTrain * pdf; // pdf(y)
    // transformation from y to x. total pdf is on x. 
    struct LinearTransform * lt; 

    // = 0 no transform; 
    // = 1 linear transformation
    int transform; 
    
    enum pdf_type type;
    void * extra;
};

struct LinearTransform * 
linear_transform_alloc(size_t,size_t,double *, double *,double);
struct LinearTransform * linear_transform_copy(struct LinearTransform *);
void linear_transform_free(struct LinearTransform *);
double * linear_transform_apply(size_t, size_t, double *, double *, double *);
void linear_transform_invert(struct LinearTransform *);
unsigned char * 
    linear_transform_serialize(unsigned char *, struct LinearTransform *, 
                               size_t *);
unsigned char * 
    linear_transform_deserialize(unsigned char *, struct LinearTransform **);

struct ProbabilityDensity * probability_density_alloc(void);
void probability_density_free(struct ProbabilityDensity *);
unsigned char * 
probability_density_serialize(unsigned char *, struct ProbabilityDensity *,
                                size_t *);

unsigned char *
probability_density_deserialize(unsigned char *, struct ProbabilityDensity **);
int probability_density_save(struct ProbabilityDensity *, char *);
struct ProbabilityDensity * probability_density_load(char *);

double probability_density_eval(struct ProbabilityDensity *, double *);
double * probability_density_log_gradient_eval(
                struct ProbabilityDensity *, 
                double *);
double * probability_density_log_hessian_eval(
                struct ProbabilityDensity *, 
                double *);

/* double * probability_density_mean(struct ProbabilityDensity *); */
/* double * probability_density_cov(struct ProbabilityDensity *); */
/* double * probability_density_var(struct ProbabilityDensity *); */

double * probability_density_lb_base(struct ProbabilityDensity *);
double * probability_density_ub_base(struct ProbabilityDensity *);

struct ProbabilityDensity * probability_density_standard_normal(size_t);
struct ProbabilityDensity * 
probability_density_mvn(size_t, double *, double *);
double * probability_density_sample(struct ProbabilityDensity *);

struct ProbabilityDensity * 
/* probability_density_laplace(double *(*)(double *, void *),  */
/*                             double *(*)(double *, void *), */
/*                             void *, size_t, double *); */
probability_density_laplace(double (*)(size_t,const double *,double*,void *),
                            double *(*)(double *, void *),
                            void *,size_t, double *);

struct Likelihood{

    size_t datadim;
    size_t paramdim; 
    size_t inputdim;
    enum likelihood_type type;
    struct FunctionTrain * like;
    int loglike; // 0 or 1
    double logextra; // 0 if loglike 0, otherwise additional 
};

/* struct Likelihood * */
/* likelihood_alloc(size_t, double *, size_t,  */
/*                  double *, size_t,  */
/*                  double, double, enum likelihood_type); */

/* struct Likelihood * likelihood_gaussian(int, size_t, */
/*     size_t, double *, size_t, double *, */
/*     size_t, struct FT1DArray *); */

/* struct Likelihood * likelihood_linear_regression(size_t, size_t,  */
/*     double *, double *, double, struct BoundingBox *); */

void likelihood_free(struct Likelihood *);

struct BayesRule{
    
    struct Likelihood * like;
    struct ProbabilityDensity * prior;
};

/* double * bayes_rule_gradient(double *, void *); */
/* double * bayes_rule_log_gradient_negative(double *, void *); */
/* double * bayes_rule_log_hessian(double *, void *); */
/* double * bayes_rule_log_hessian_negative(double *, void *); */

/* struct ProbabilityDensity * bayes_rule_laplace(struct BayesRule *); */
/* struct ProbabilityDensity * bayes_rule_compute(struct BayesRule *); */

#endif
