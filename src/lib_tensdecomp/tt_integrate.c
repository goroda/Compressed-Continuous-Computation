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








#include "tt_integrate.h"
#include "array.h"
#include "tensor.h"
#include "linalg.h"


//////////////////////////////////////////////////////////////
double tt_integrate(const struct tt *integrand, double ** weights)
{
    double out = 0.0;
    size_t ii;
    
    double * temp1;
    double * temp2 = NULL;
    temp1 = tensor_sum2(integrand->cores[0], weights[0]);
    double * vals;
    for (ii = 1; ii < integrand->dim; ii++){
        vals =tensor_sum2(integrand->cores[ii], weights[ii]);

        if (temp2 == NULL){
            temp2 = calloc_double(integrand->ranks[ii+1]);
            cblas_dgemv(CblasColMajor, CblasTrans, integrand->ranks[ii], integrand->ranks[ii+1], 
                         1.0, vals, integrand->ranks[ii], temp1, 1, 0.0, temp2, 1); 
            //temp2 = vecmat(temp1, gamma[ii]);
            free(temp1);
            temp1=NULL;
        }
        else {
            temp1 = calloc_double(integrand->ranks[ii+1]);

            cblas_dgemv(CblasColMajor, CblasTrans, integrand->ranks[ii], integrand->ranks[ii+1], 
                         1.0, vals, integrand->ranks[ii], temp2, 1, 0.0, temp1, 1); 
            //temp1 = vecmat(temp2, gamma[ii]);
            free(temp2);
            temp2=NULL;
        }
        free(vals);
    }
    if (temp1 == NULL) {
        out = temp2[0];
        free(temp2);
    }
    else {
        out = temp1[0];
        free(temp1);
    }

    return out;
}


