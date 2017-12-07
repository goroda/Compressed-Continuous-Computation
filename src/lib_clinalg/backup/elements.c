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





/** \file elements.c
 * Provides routines for initializing / using elements of continuous linear algebra
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "stringmanip.h"
#include "array.h"
#include "linalg.h"
#include "elements.h"
#include "algs.h"

// helper functions for function_train_initsum2
struct wrap_spec_func
{
    double (*f)(double, size_t, void *);
    size_t which;
    void * args;
};
double eval_spec_func(double x, void * args)
{
    struct wrap_spec_func * spf = args;
    return spf->f(x,spf->which,spf->args);
}

/* /\**********************************************************\//\** */
/*     Extract a copy of the first nkeep columns of a qmarray */

/*     \param a [in] - qmarray from which to extract columns */
/*     \param nkeep [in] : number of columns to extract */

/*     \return qm - qmarray */
/* **************************************************************\/ */
/* struct Qmarray * qmarray_extract_ncols(struct Qmarray * a, size_t nkeep) */
/* { */
    
/*     struct Qmarray * qm = qmarray_alloc(a->nrows,nkeep); */
/*     size_t ii,jj; */
/*     for (ii = 0; ii < nkeep; ii++){ */
/*         for (jj = 0; jj < a->nrows; jj++){ */
/*             qm->funcs[ii*a->nrows+jj] = generic_function_copy(a->funcs[ii*a->nrows+jj]); */
/*         } */
/*     } */
/*     return qm; */
/* } */

/////////////////////////////////////////////////////////
// qm_array



////////////////////////////////////////////////////////////////////
// function_train 
//


/////////////////////////////////////////////////////////
// Utilities
//
