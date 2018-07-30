// Copyright (c) 2018, University of Michigan

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


/** \file online.c
 * Provides routines for online learning
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

/* #include "lib_linalg.h" */
#include "online.h"

struct PP
{
    struct FTparam * ftp;
    struct RegressOpts * opts;
};


struct StochasticUpdater * stochastic_updater_alloc(enum SU_ALGS alg)
{
    struct StochasticUpdater * su = malloc(sizeof(struct StochasticUpdater));
    if (su == NULL){
        fprintf(stderr, "Failure To allocate Stochastic Updater\n");
        exit(1);
    }
    su->alg = alg;

    su->mem = NULL;
    su->obj = NULL;
    su->mem1 = NULL;
    su->mem2 = NULL;

    su->aux_args = NULL;
    su->aux_obj = NULL;
    return su;
}

void stochastic_updater_free(struct StochasticUpdater * su)
{
    if (su != NULL){
        sl_mem_manager_free(su->mem); su->mem = NULL;
        objective_function_free(&(su->obj)); su->obj = NULL;
        free(su->mem1); su->mem1 = NULL;
        free(su->mem2); su->mem2 = NULL;
        free(su->aux_args); su->aux_args = NULL;
        free(su->aux_obj); su->aux_obj = NULL;
        free(su); su = NULL;
    }
}

void stochastic_updater_reset(struct StochasticUpdater * su)
{
    if (su != NULL){
        free(su->mem1); su->mem1 = NULL;
        free(su->mem2); su->mem2 = NULL;
        su->mem1 = calloc_double(su->nparams);
        su->mem2 = calloc_double(su->nparams);
    }
}


static int
online_learning_interface(size_t nparam, const double * param,
                          size_t N, size_t * ind,
                          struct SLMemManager * mem,
                          const struct Data * data,
                          double ** evals, double ** grads, void * args)
{
    /* printf("in the learning interface\n"); */
    (void) nparam;
    struct PP * mem_opts = args;
    struct FTparam      * ftp   = mem_opts->ftp;
    /* struct RegressOpts  * ropts = mem_opts->opts;  */
    const double * x = data_get_subset_ref(data,N,ind);

    /* printf("update params = \n"); */
    /* printf("ftp dim = %zu\n", ftp->dim); */
    /* printf("ropts dim = %zu\n", ropts->dim); */
    /* printf("ftp nparam = %zu\n", ftp->nparams); */

    ft_param_update_params(ftp, param);

    /* printf("cool!\n"); */
    if (grads == NULL)
    {
        for (size_t ii = 0; ii < N; ii++){
            mem->evals->vals[ii] = function_train_eval(ftp->ft,x + ii * ftp->dim);
        }
        *evals = mem->evals->vals;
    }
    else{
        for (size_t ii = 0; ii < N; ii++){
            mem->evals->vals[ii] =
                ft_param_gradeval(ftp, x + ii * ftp->dim,
                                  mem->grad->vals + ii*ftp->nparams,
                                  mem->lin_structure_vals,
                                  mem->running_eval,
                                  mem->running_grad);
        }
        *evals = mem->evals->vals;
        *grads = mem->grad->vals;
    }
    
    return 0;
}

int
setup_least_squares_online_learning(
    struct StochasticUpdater * su,
    double eta,
    struct FTparam * ftp,
    struct RegressOpts * ropts)
{



    /* printf("in setup\n"); */
    size_t ndata = 1;
    struct SLMemManager * mem = sl_mem_manager_alloc(ftp->dim, ndata, ftp->nparams, NONE_ST);
    sl_mem_manager_check_structure(mem, ftp, NULL);


    struct PP * ls_args = malloc(sizeof(struct PP));
    ls_args->ftp = ftp;
    ls_args->opts = ropts;

    struct LeastSquaresArgs * ls = malloc(sizeof(struct LeastSquaresArgs));
    /* printf("\t nparams in here = %zu\n", ftp->nparams); */
    /* printf("\t dim in here = %zu\n", ftp->dim); */
    /* printf("\t dim2 in here = %zu\n", ropts->dim); */
    ls->mapping = online_learning_interface;
    ls->args = ls_args;

    struct ObjectiveFunction * obj = NULL;
    objective_function_add(&obj, 1.0, c3_objective_function_least_squares, ls);
    
    su->mem = mem;
    su->obj = obj;
    su->nparams = ftp->nparams;

    su->eta = eta;
    su->mem1 = calloc_double(ftp->nparams);
    su->mem2 = calloc_double(ftp->nparams);

    su->aux_obj = ls;
    su->aux_args = ls_args;

    
    return 0;
}

/***********************************************************//**
    Perform an update of the parameters

    \param[in]     su    - stochastic updater
    \param[in,out] param - parameters
    \param[in,out] grad  - gradient
    \param[in]     data  - data point

***************************************************************/
double stochastic_update_step(const struct StochasticUpdater * su,
                              double * param,
                              double * grad,
                              const struct Data * data)
{
    /* printf("in stoch update step\n"); */
    struct ObjectiveFunction * obj  = su->obj;
    /* printf("\t get mem manager\n"); */
    struct SLMemManager      * mem  = su->mem;
    /* printf("\t got it\n"); */
    size_t nparam = su->nparams;
    /* printf("\t nparam = %zu\n", nparam); */
    /* printf("\t objective eval = "); */
    double eval = objective_eval_data(nparam, param, grad, data, obj, mem);
    /* printf("\t done!\n"); */

    /* printf("grad = "); dprint(nparam, grad); */
    /* printf("param pre = "); dprint(nparam, param); */
    /* for (size_t ii = 0; ii < 10; ii++){ */
    /*     size_t ind_update = (size_t) rand() % nparam; */
    /*     param[ind_update] -= su->eta * grad[ind_update]; */
    /* } */


    double alpha = 0.7;
    double epsilon = 1e-8;
    for (size_t ii = 0; ii < nparam; ii++){
        if (su->alg == SU_SGD){
            param[ii] -= su->eta * grad[ii];
        }
        else if (su->alg == SU_MOMENTUM){
            // momentum 
            su->mem1[ii] = alpha * su->mem1[ii] - su->eta * grad[ii];
            param[ii] += su->mem1[ii];
        }
        else if (su->alg == SU_ADAGRAD){
            su->mem1[ii] += grad[ii]*grad[ii];
            param[ii] -= su->eta / sqrt(su->mem1[ii]) * grad[ii];
        }
        else if (su->alg == SU_ADADELTA){

            su->mem1[ii] = su->eta * su->mem1[ii] + (1 - su->eta) * grad[ii]*grad[ii];
            double delta =  - sqrt(su->mem2[ii] + epsilon) / sqrt(su->mem1[ii] + epsilon) * grad[ii];
            su->mem2[ii] = su->eta * su->mem2[ii] + (1 - su->eta) * delta * delta;
            param[ii] += delta;
        }
        else{
            fprintf(stderr, "Stochastic Updater Algorithm Type doesnt exist\n");
            exit(1);
        }

    }
    /* printf("param post = "); dprint(nparam, param); */

    return eval;
}
    
