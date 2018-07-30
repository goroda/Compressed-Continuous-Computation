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


/** \file online.h
 * header files for online learning
 */


#include <stdlib.h>
#include "regress.h"
#include "objective_functions.h"


enum SU_ALGS {SU_SGD, SU_MOMENTUM, SU_ADAGRAD, SU_ADADELTA};

/** \struct StochasticUpdater
 * \brief Interface to online learning
 * \var StochasticUpdater::eta
 * Learning Rate
 */
struct StochasticUpdater
{
    enum SU_ALGS alg;
    
    double eta;
    size_t nparams;
    struct SLMemManager * mem;
    struct ObjectiveFunction * obj;

    double * mem1;
    double * mem2;

    void * aux_args;
    void * aux_obj;
    
};


struct StochasticUpdater * stochastic_updater_alloc(enum SU_ALGS);
void stochastic_updater_free(struct StochasticUpdater *);
void stochastic_updater_reset(struct StochasticUpdater *);

int
setup_least_squares_online_learning(
    struct StochasticUpdater * su,
    double eta,
    struct FTparam * ftp,
    struct RegressOpts * ropts);


double
stochastic_update_step(const struct StochasticUpdater *,
                       double *,
                       double *,
                       const struct Data *);
