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

#include "lib_linalg.h"
#include "lib_optimization.h"
#include "regress.h"
#include "objective_functions.h"

/** \struct StochasticUpdater
 * \brief Interface to online learning
 * \var StochasticUpdater::eta
 * Learning Rate
 */ 
struct StochasticUpdater
{
    double eta;
    struct SLMemManager * mem;
    struct ObjectiveFunction * obj;
};


struct PP
{
    struct FTparam * ftp;
    struct RegressOpts * opts;
};

int setup_least_squares_online_learning(struct StochasticUpdater * su, struct FTparam * ftp, struct RegressOpts * ropts)
{
    struct SLMemManager * mem = sl_mem_manager_alloc(ftp->dim, 1, ftp->nparams, NONE_ST);
    struct LeastSquaresArgs * ls = malloc(sizeof(struct LeastSquaresArgs));

    struct PP ls_args;
    ls_args.ftp = ftp;
    ls_args.opts = ropts;

    ls->mapping = ft_param_learning_interface;
    ls->args = &ls_args;

    struct ObjectiveFunction * obj = NULL;
    objective_function_add(&obj, 1.0, c3_objective_function_least_squares, ls);
    
    su->mem = mem;
    su->obj = obj;
    
    return 0;
}
    

/***********************************************************//**
    Perform an update of the parameters

    \param[in] su         - stochastic updater
    \param[in] nparam     - number of parameters
    \param[in,out] param  - parameters
    \param[in,out] grad   - gradient
    \param[in] data       - data point
    \param[in] obj        - objective function

***************************************************************/
void stochastic_update_step(const struct StochasticUpdater * su,
                            size_t nparam,
                            double * param,
                            double * grad,
                            const struct Data * data,
                            const struct ObjectiveFunction * obj
                            struct SLMemManager * mem)
{

    if (grad != NULL){
        for (size_t ii = 0; ii < nparam; ii++){
            grad[ii] = 0.0;
        }
    }

    double out = 0.0;
    while (obj != NULL){
        out += obj->weight * obj->func(nparam, param, grad, 1, 0, data, mem, obj->arg);
        obj = obj->next;
    }

    for (size_t ii = 0; ii < nparam; ii++){
        param[ii] -= su->eta * grad[ii];
    }
}
    
