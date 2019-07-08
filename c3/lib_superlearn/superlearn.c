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


/** \file superlearn.c
 * Provides routines for supervised learning and interfaces to optimizers
 */

#include "superlearn.h"
#include "objective_functions.h"


double slp_get_minimum(struct SLP * slp)
{
    return slp->obj_min;
}

/***********************************************************//**
    C3OPt optimizer interface for batch descent
***************************************************************/
double objective_batch(size_t nparam, const double * param, double * grad, void * arg)
{
    struct SLP * slp = arg;
    size_t N = data_get_N(slp->data);
    return objective_eval(nparam,param,grad,N,NULL,arg);
}

/***********************************************************//**
    C3OPt optimizer interface for stochastic gradient descent
***************************************************************/
double objective_stoch(size_t nparam, size_t ind, const double * param, double * grad, void * arg)
{
    return objective_eval(nparam,param,grad,1,&ind,arg);
}


/***********************************************************//**
   Setup C3OPT optimizer and minimizeo objective                                                         
***************************************************************/
int objective_minimize(struct SLP * slp, struct c3Opt * optimizer,
                       size_t nparam, double * guess, double *val)

{
    c3opt_set_nvars(optimizer,nparam);
    if (c3opt_is_sgd(optimizer)){
        c3opt_add_objective_stoch(optimizer,objective_stoch,slp);
    }
    else{
        c3opt_add_objective(optimizer,objective_batch,slp);
    }
    int res = c3opt_minimize(optimizer,guess,val);
    return res;
}


/***********************************************************//**
    Solve a supervised learning problem                                                            

    \param[in,out] slp   - supervised learning problem
    \param[in,out] guess - initial/final parameters

    \return evaluation
***************************************************************/
int slp_solve(struct SLP * slp, double * guess)
{

    int res = objective_minimize(slp,slp->optimizer,slp->nparams,guess,&(slp->obj_min));
    if (res < -1){
        fprintf(stderr,"Warning: optimizer exited with code %d\n",res);
    }
    return 0;
}



