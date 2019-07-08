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


/** \file objective_functions.c
 * Provides routines for building and optimizing objective functions
 */

#include <stdlib.h>
#include <assert.h>

#include "lib_optimization.h"
#include "objective_functions.h"
#include "superlearn_util.h"
#include "superlearn.h"

#include "regress.h"

struct ObjectiveFunction
{
    double weight;
    double (*func)(size_t nparam, const double * param, double * grad, size_t N, size_t * ind,struct Data * data,
                   struct SLMemManager *, void * args);
    void * arg;
    struct ObjectiveFunction * next;
};

struct ObjectiveFunction * objective_function_alloc(
    double weight,
    double (*func)(size_t nparam, const double * param, double * grad, size_t N, size_t * ind,struct Data * data,
                   struct SLMemManager *, void * args),
    void * arg)
{
    struct ObjectiveFunction * obj = malloc(sizeof(struct ObjectiveFunction));
    if (obj == NULL){
        fprintf(stderr,"Failure to allocate objective function\n");
        exit(1);
    }
    obj->weight = weight;
    obj->func = func;
    obj->arg = arg;
    obj->next = NULL;
    return obj;
}

void objective_function_free(struct ObjectiveFunction ** obj)
{
    if ((*obj) != NULL){
        struct ObjectiveFunction * temp = (*obj)->next;
        free(*obj); *obj = NULL;
        objective_function_free(&temp);
    }
}

void objective_function_add(struct ObjectiveFunction ** obj, double weight,
                            double (*func)(size_t nparam, const double * param, double * grad,
                                           size_t N, size_t * ind,struct Data * data,
                                           struct SLMemManager *, void * args),
                            void * arg)
{
    struct ObjectiveFunction * new_obj = objective_function_alloc(weight,func,arg);
    if (*obj == NULL)
    {
        *obj = new_obj;
    }
    else{
        struct ObjectiveFunction * temp = *obj;
        while (temp->next != NULL){
            temp = temp->next;
        }
        temp->next = new_obj;
    }
}


double objective_eval(size_t nparam, const double * param, double * grad,
                      size_t N, size_t * ind,
                      void * obj_void)
{
    struct SLP               * slp  = obj_void;
    struct ObjectiveFunction * obj  = slp->objective_function;
    struct Data              * data = slp->data;
    struct SLMemManager      * mem  = slp->mem;

    if (grad != NULL){
        for (size_t ii = 0; ii < nparam; ii++){
            grad[ii] = 0.0;
        }
    }
    double out = 0.0;
    while (obj != NULL){
        out += obj->weight * obj->func(nparam,param,grad,N,ind,data,mem,obj->arg);
        obj = obj->next;
    }
    
    /* printf("OUT = %G\n",out); */
    return out;
}

// Note this is also defined in regress.c
struct PP
{
    struct FTparam * ftp;
    struct RegressOpts * opts;
};

double c3_objective_function_least_squares(size_t nparam, const double * param, double * grad,
                                           size_t Ndata, size_t * data_index, struct Data * data,
                                           struct SLMemManager * mem, void * args)
{

    struct LeastSquaresArgs * ls = args;
    struct PP * pp =  ls->args;
    struct RegressOpts * opts = pp->opts;
    
    double * evals = NULL;
    double * grads = NULL; 

    double out = 0.0;
    double dy;
    int res = 0;
    if (grad == NULL){
        res = ls->mapping(nparam,param,Ndata,data_index,mem,data,&evals,NULL,ls->args);
        assert (res == 0);
        if (data_index == NULL){
            for (size_t ii = 0; ii < Ndata; ii++){
                dy = data_subtract_from_y(data,ii,evals[ii]);
                if (opts->sample_weights == NULL){
                    out += dy * dy;
                }
                else{
                    out += dy * dy * opts->sample_weights[ii];
                }
                
            }
        }
        else{
            for (size_t ii = 0; ii < Ndata; ii++){
                dy = data_subtract_from_y(data,data_index[ii],evals[ii]);

                if (opts->sample_weights == NULL){
                    out += dy * dy;
                }
                else{
                    out += dy * dy * opts->sample_weights[data_index[ii]];
                }
            }
        }
    }
    else{
        res = ls->mapping(nparam,param,Ndata,data_index,mem,data,&evals,&grads,ls->args);
        assert (res == 0);
        if (data_index == NULL){
            for (size_t ii = 0; ii < Ndata; ii++){
                dy = data_subtract_from_y(data,ii,evals[ii]);

                if (opts->sample_weights == NULL){
                    out += dy * dy;
                    for (size_t jj = 0; jj < nparam; jj++){
                        grad[jj] -= 2.0 * dy * grads[ii*nparam+jj];
                    }
                }
                else{
                    out += dy * dy * opts->sample_weights[ii];
                    for (size_t jj = 0; jj < nparam; jj++){
                        grad[jj] -= 2.0 * dy * grads[ii*nparam+jj] * opts->sample_weights[ii];
                    }
                }
            }
        }
        else{
            for (size_t ii = 0; ii < Ndata; ii++){
                dy = data_subtract_from_y(data,data_index[ii],evals[ii]);
                if (opts->sample_weights == NULL){
                    out += dy * dy;
                    for (size_t jj = 0; jj < nparam; jj++){
                        grad[jj] -= 2.0 * dy * grads[ii*nparam+jj];
                    }
                }
                else{
                    out += dy * dy * opts->sample_weights[data_index[ii]];
                    for (size_t jj = 0; jj < nparam; jj++){
                        grad[jj] -= 2.0 * dy * grads[ii*nparam+jj] *
                            opts->sample_weights[data_index[ii]];
                    }
                }
            }
        }
        
        for (size_t jj = 0; jj < nparam; jj++){
            grad[jj] /= (double)Ndata;
        }
    }

    out /= (double)Ndata;
    
    return out;
}

