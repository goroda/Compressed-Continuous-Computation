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








#include <stdlib.h>
#include "tensor.h"
#include "candecomp.h"

/********************************************************//**
    Function cdelem

    Purpose: Get an element of a tensor in canonical form

    Parameters:
        - cd - tensor in canonical decomp
        - elem - element to get (dim,)

    Returns: 
        Nothing. void function
***********************************************************/
double cdelem(const struct candecomp * cd, const size_t * elem)
{
    double out = 0.0;
    double temp;
    size_t ii, jj;
    for (ii = 0; ii < cd->rank; ii++){
        temp = 1.0;
        for (jj = 0; jj < cd->dim; jj++){
            temp *= cd->cores[jj]->vals[elem[jj]*cd->rank + ii];
        }
        out += temp;
    }
    return out;
}

/********************************************************//**
    Function cd2tt_corem

    Purpose: convert a canonical decomp core to a TT core
             (only valid for core 0 < i < dim-1

    Parameters:
        - cdcore - canonical decomp core

    Returns: core - TT core
***********************************************************/
struct tensor * cd2tt_corem(struct mat * cdcore){
    struct tensor * core;
    size_t nvals[3];
    nvals[0] = cdcore->ncols;
    nvals[1] = cdcore->nrows;
    nvals[2] = cdcore->ncols;
    init_tensor(&core, 3, nvals);
    
    size_t ii,jj;
    for (ii = 0; ii < nvals[1]; ii++){
        for (jj = 0; jj < cdcore->ncols; jj++){
            core->vals[jj + ii*nvals[0] + jj * nvals[0] * nvals[1]] = 
                    cdcore->vals[ii*cdcore->ncols + jj];
        }
    }
    return core;
}

/********************************************************//**
    Function cd2tt_corel

    Purpose: convert a canonical decomp core to a TT core
             (only valid for core 0)

    Parameters:
        - cdcore - canonical decomp core

    Returns: core - TT core
***********************************************************/
struct tensor * cd2tt_corel(struct mat * cdcore){
    struct tensor * core;
    size_t nvals[3];
    nvals[0] = 1; 
    nvals[1] = cdcore->nrows;
    nvals[2] = cdcore->ncols;
    init_tensor(&core, 3, nvals);
    
    size_t ii,jj;
    for (ii = 0; ii < nvals[1]; ii++){
        for (jj = 0; jj < nvals[2]; jj++){
            core->vals[0 + ii*nvals[0] + jj * nvals[0] * nvals[1]] = 
                    cdcore->vals[ii*cdcore->ncols + jj];
        }
    }
    return core;
}

/********************************************************//**
    Function cd2tt_corer

    Purpose: convert a canonical decomp core to a TT core
             (only valid for core dim-1)

    Parameters:
        - cdcore - canonical decomp core

    Returns: core - TT core

***********************************************************/
struct tensor * cd2tt_corer(struct mat * cdcore)
{
    struct tensor * core;
    size_t nvals[3];
    nvals[0] = cdcore->ncols;
    nvals[1] = cdcore->nrows;
    nvals[2] = 1;
    init_tensor(&core, 3, nvals);
    size_t ii,jj;
    for (ii = 0; ii < nvals[1]; ii++){
        for (jj = 0; jj < nvals[0]; jj++){
            core->vals[jj + ii*nvals[0]] = cdcore->vals[ii*cdcore->ncols + jj];
        }
    }
    return core;
}

