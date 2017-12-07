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




#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <stddef.h>
#include <stdio.h>

enum c3opt_ls_alg {BACKTRACK, STRONGWOLFE, WEAKWOLFE};

enum c3opt_alg {BFGS, LBFGS, BATCHGRAD, BRUTEFORCE, SGD};
enum c3opt_return {
    C3OPT_LS_PARAM_INVALID=-4,
    C3OPT_LS_MAXITER_REACHED=-3,
    C3OPT_X_BOUND_VIOLATED=-2,
    C3OPT_MAXITER_REACHED=-1,
    C3OPT_SUCCESS=0,
    C3OPT_FTOL_REACHED=1,
    C3OPT_XTOL_REACHED=2,
    C3OPT_GTOL_REACHED=3
};

struct c3Opt;
struct c3Opt * c3opt_create(enum c3opt_alg);
void c3opt_set_nvars(struct c3Opt *, size_t);
struct c3Opt * c3opt_alloc(enum c3opt_alg, size_t);
struct c3Opt * c3opt_copy(struct c3Opt *);
void c3opt_free(struct c3Opt *);

void c3opt_set_sgd_learn_rate(struct c3Opt *, double);
void c3opt_set_sgd_nsamples(struct c3Opt *, size_t);
size_t c3opt_get_sgd_nsamples(const struct c3Opt *);
void c3opt_set_nvectors_store(struct c3Opt *, size_t);
size_t c3opt_get_nvectors_store(const struct c3Opt *);
void c3opt_set_lbfgs_scale(struct c3Opt *, int);
int c3opt_get_lbfgs_scale(struct c3Opt *);

int c3opt_is_bruteforce(const struct c3Opt *);
int c3opt_is_sgd(const struct c3Opt *);
void c3opt_add_lb(struct c3Opt *, double *);
void c3opt_add_ub(struct c3Opt *, double *);
void c3opt_add_objective(struct c3Opt *,
                         double (*)(size_t, const double *, double *,void *),
                         void *);
void c3opt_add_objective_stoch(
    struct c3Opt *,
    double(*)(size_t,size_t,const double *,double *,void *),
    void *);
void c3opt_set_verbose(struct c3Opt *, int);
void c3opt_set_maxiter(struct c3Opt *, size_t);
size_t c3opt_get_maxiter(const struct c3Opt *);
void c3opt_set_absxtol(struct c3Opt *, double);

size_t c3opt_get_niters(struct c3Opt *);
double c3opt_get_stored_function(const struct c3Opt *, size_t);
size_t c3opt_get_nevals(struct c3Opt *);
size_t c3opt_get_ngvals(struct c3Opt *);

void c3opt_set_relftol(struct c3Opt *, double);
void c3opt_set_gtol(struct c3Opt *, double);

void c3opt_set_storage_options(struct c3Opt *, int, int, int);
int c3opt_get_store_func(const struct c3Opt *);

void c3opt_print_stored_values(struct c3Opt *, FILE *, int, int);
    
int c3opt_ls_get_initial(const struct c3Opt *);

void c3opt_ls_set_alg(struct c3Opt *, enum c3opt_ls_alg);
enum c3opt_ls_alg c3opt_ls_get_alg(const struct c3Opt *);

void   c3opt_ls_set_alpha(struct c3Opt *,double);
double c3opt_ls_get_alpha(struct c3Opt *);

void   c3opt_ls_set_beta(struct c3Opt *,double);
double c3opt_ls_get_beta(struct c3Opt *);

void   c3opt_ls_set_maxiter(struct c3Opt *,size_t);
size_t c3opt_ls_get_maxiter(struct c3Opt *);

void c3opt_ls_set_alg(struct c3Opt *, enum c3opt_ls_alg);



double c3opt_ls_wolfe_bisect(struct c3Opt *, double *, double,
                             double *, double *,
                             double *, double *, int *);
double c3opt_ls_strong_wolfe(struct c3Opt *, double *, double,
                             double *, double *,
                             double *, double *, int *);


void c3opt_set_brute_force_vals(struct c3Opt *, size_t, double *);

int c3opt_minimize(struct c3Opt *, double *, double *);

double * c3opt_get_lb(struct c3Opt *);
double * c3opt_get_ub(struct c3Opt *);
double c3opt_eval(struct c3Opt *, const double *, double *);
double c3opt_eval_stoch(struct c3Opt *, size_t, const double *, double *);
double c3opt_check_deriv(struct c3Opt *, const double *, double);
double c3opt_check_deriv_each(struct c3Opt *, const double *, double, double *);


// LBFGS STUFF
struct c3opt_lbfgs_list;
struct c3opt_lbfgs_list * c3opt_lbfgs_list_alloc(size_t,size_t);
void c3opt_lbfgs_list_insert(struct c3opt_lbfgs_list *, size_t,
                             const double *, const double *,
                             const double *, const double *);
void c3opt_lbfgs_reset_step(struct c3opt_lbfgs_list *);
void c3opt_lbfgs_list_step(struct c3opt_lbfgs_list *, size_t *, double **, double **, double *);
void c3opt_lbfgs_list_step_back(struct c3opt_lbfgs_list *, size_t *, double **, double **, double *);
void c3opt_lbfgs_list_print(struct c3opt_lbfgs_list *, FILE *, int, int);
void c3opt_lbfgs_list_free(struct c3opt_lbfgs_list *);
/* int c3Opt_lbfgs(struct c3Opt * opt, */
/*                 double * x, double * fval, double * grad) */

void
newton(double **, size_t, double, double,
        double * (*)(double *, void *),
        double * (*)(double *, void *), void *);

double backtrack_line_search(size_t, double *, double, double *, 
                             double *,double *, double *,
                             double, 
                             double, 
                             double (*)(double *, void *), 
                             void *, 
                             size_t, int *);

double backtrack_line_search_bc(size_t, double *, 
                                double *, double *, double, 
                                double *, double *,
                                double *, double *, double, 
                                double, 
                                double (*)(double *, void *), 
                                void *, 
                                size_t,int *);


int gradient_descent(size_t, double *, double *, double *,
                     double *,
                     double (*)(double *,void*),void *,
                     int (*)(double *,double*,void*), void *,
                     double,size_t, size_t, double, double,int);

int box_pg_descent(size_t, double *, double *,
                   double *, double *, double *,
                   double *, double (*)(double *,void*),void *,
                   int (*g)(double *,double*,void*), void *,
                   double,size_t, size_t,
                   double, double, int);
int box_damp_newton(size_t, double *, double *,
                   double *, double *, double *,double*,
                   double *, double (*)(double *,void*),void *,
                   int (*)(double *,double*,void*), void *,
                   int (*)(double *,double*,void*), void *,
                   double,size_t, size_t,
                   double, double, int);
int box_damp_bfgs(size_t d, double * lb, double * ub,
                  double * x, double * fval, double * grad,
                  double * invhess,
                  double * space,
                  double (*f)(double*,void*),void * fargs,
                  int (*g)(double*,double*,void*),
                  double tol,size_t maxiter,size_t maxsubiter,
                  double alpha, double beta, int verbose);
#endif

