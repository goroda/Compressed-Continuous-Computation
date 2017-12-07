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
#include "array.h"
#include "convert_decomp.h"

/********************************************************//**
    Function cd2tt

    Purpose: convert a tensor in canonical form to TT

    Parameters:
        - cd - tensor in candecomp form

    Returns: tens - tensor in TT form
***********************************************************/
struct tt * cd2tt(struct candecomp * cd)
{   
    size_t ii;
    size_t * nvals = calloc_size_t(cd->dim);
    for (ii = 0; ii < cd->dim; ii++) { nvals[ii] = cd->cores[ii]->nrows; }

    struct tt * tens;
    init_tt_alloc(&tens, cd->dim, nvals);

    tens->cores[0] = cd2tt_corel(cd->cores[0]);
    tens->ranks[0] = 1;
    for (ii = 1; ii < tens->dim-1; ii++){
        tens->cores[ii] = cd2tt_corem(cd->cores[ii]);
        tens->ranks[ii] = tens->cores[ii]->nvals[0];
    }
    tens->cores[tens->dim-1] = cd2tt_corer(cd->cores[tens->dim-1]);
    tens->ranks[tens->dim-1] = tens->cores[tens->dim-1]->nvals[0];
    tens->ranks[tens->dim] = 1;

    free(nvals);
    return tens;
}
