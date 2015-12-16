// Copyright (c) 2014-2015, Massachusetts Institute of Technology
//
// This file is part of the Compressed Continuous Computation (C3) toolbox
// Author: Alex A. Gorodetsky 
// Contact: goroda@mit.edu

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

/** \file diffusion.c
 * Provides routines for applying diffusion operator
 */

#include <stdlib.h>
#include <assert.h>

#include "lib_clinalg.h"
#include "array.h"

void dmrg_diffusion_midleft(struct Qmarray * dA, struct Qmarray * A,
                     struct Qmarray * ddF, struct Qmarray * dF, struct Qmarray * F,
                     double * mat, size_t r, struct Qmarray * out)
{
    assert (out != NULL);
    
    size_t sgfa = r * F->ncols * A->nrows;
    size_t offset = r * A->nrows * F->nrows;

    struct GenericFunction ** gfa1 =
        generic_function_array_alloc(sgfa);

    generic_function_kronh(1,r,A->nrows,F->nrows,F->ncols,mat,F->funcs,gfa1);


    struct GenericFunction ** gfa2 = generic_function_array_alloc(sgfa);
    generic_function_kronh(1,r,A->nrows,F->nrows,F->ncols,mat + offset,dF->funcs,gfa2);

    struct GenericFunction ** gfa3 = generic_function_array_alloc(sgfa);
    generic_function_kronh(1,r,A->nrows,F->nrows,F->ncols,mat + offset,ddF->funcs,gfa3);

    generic_function_kronh2(1,r,F->ncols,
        A->nrows, A->ncols, A->funcs, gfa1, out->funcs);

    size_t width = r * A->ncols * F->ncols;

    generic_function_kronh2(1,r,F->ncols,
        A->nrows, A->ncols, dA->funcs, gfa2, out->funcs + width);
    
    struct GenericFunction ** temp = generic_function_array_alloc(width);
    generic_function_kronh2(1,r,F->ncols,
        A->nrows, A->ncols, A->funcs, gfa3, temp);

    size_t ii;
    for (ii = 0; ii < width; ii++){
        generic_function_axpy(1.0,temp[ii],out->funcs[width+ii]);
    }

    generic_function_array_free(gfa1,sgfa); gfa1 = NULL;
    generic_function_array_free(gfa2,sgfa); gfa2 = NULL;
    generic_function_array_free(gfa3,sgfa); gfa3 = NULL;
    generic_function_array_free(temp,width); temp = NULL;
}


void dmrg_diffusion_lastleft(struct Qmarray * dA, struct Qmarray * A,
                     struct Qmarray * ddF, struct Qmarray * dF, struct Qmarray * F,
                     double * mat, size_t r, struct Qmarray * out)
{

    assert (out != NULL );
    assert (A->ncols == F->ncols);

    size_t sgfa = r * F->ncols * A->nrows;
    size_t offset = r * A->nrows * F->nrows;

    struct GenericFunction ** gfa1 = generic_function_array_alloc(sgfa);

    generic_function_kronh(1,r,A->nrows,F->nrows,F->ncols,mat,F->funcs,gfa1);

    struct GenericFunction ** gfa2 = generic_function_array_alloc(sgfa);
    generic_function_kronh(1,r,A->nrows,F->nrows,F->ncols,mat+offset,dF->funcs,gfa2);

    struct GenericFunction ** gfa3 = generic_function_array_alloc(sgfa);
    generic_function_kronh(1,r,A->nrows,F->nrows,F->ncols,mat+offset,ddF->funcs,gfa3);
 
    generic_function_kronh2(1,r,F->ncols, A->nrows, A->ncols, A->funcs, gfa1, out->funcs);

    size_t width = r * A->ncols * F->ncols;
    struct GenericFunction ** temp = generic_function_array_alloc(width);
    generic_function_kronh2(1,r,F->ncols, A->nrows, A->ncols, dA->funcs, gfa2, temp);
    struct GenericFunction ** temp2 = generic_function_array_alloc(width);
    generic_function_kronh2(1,r,F->ncols, A->nrows, A->ncols, A->funcs, gfa3, temp2);

    size_t ii;
    for (ii = 0; ii < width; ii++){
        generic_function_axpy(1.0,temp[ii],out->funcs[ii]);
        generic_function_axpy(1.0,temp2[ii],out->funcs[ii]);

    }

    generic_function_array_free(gfa1,sgfa); gfa1 = NULL;
    generic_function_array_free(gfa2,sgfa); gfa2 = NULL;
    generic_function_array_free(gfa3,sgfa); gfa3 = NULL;
    generic_function_array_free(temp,width); temp = NULL;
    generic_function_array_free(temp2,width); temp2 = NULL;
    
}

void dmrg_diffusion_midright(struct Qmarray * dA, struct Qmarray * A,
                     struct Qmarray * ddF, struct Qmarray * dF, struct Qmarray * F,
                     double * mat, size_t r, struct Qmarray * out)
{
    size_t sgfa = r * F->nrows * A->ncols;
    struct GenericFunction ** gfa1 = generic_function_array_alloc(sgfa);
    struct GenericFunction ** gfa2 = generic_function_array_alloc(sgfa);
    struct GenericFunction ** gfa3 = generic_function_array_alloc(sgfa);
    
    size_t ms = r * A->ncols * F->ncols;
    double * mat1 = calloc_double(ms);
    double * mat2 = calloc_double(ms);
    size_t ii,jj,kk;
    for (ii = 0; ii < r; ii++){
        for (jj = 0; jj < A->ncols; jj++){
            for (kk = 0; kk < F->ncols; kk++){
                mat1[kk + jj*F->ncols + ii * A->ncols*F->ncols] =
                    mat[kk + jj*F->ncols + ii * (2*A->ncols*F->ncols)];
                mat2[kk + jj*F->ncols + ii * A->ncols*F->ncols] =
                    mat[kk + (jj+A->ncols)*F->ncols + ii * (2*A->ncols*F->ncols)];
            }
        }
    }

    generic_function_kronh(0,r,A->ncols,F->nrows,F->ncols,mat1,F->funcs,gfa1);
    generic_function_kronh(0,r,A->ncols,F->nrows,F->ncols,mat2,dF->funcs,gfa2);
    generic_function_kronh(0,r,A->ncols,F->nrows,F->ncols,mat2,ddF->funcs,gfa3);

    size_t s2 = r * A->nrows * F->nrows;
    struct GenericFunction ** t1 = generic_function_array_alloc(s2);
    struct GenericFunction ** t2 = generic_function_array_alloc(s2);
    struct GenericFunction ** t3 = generic_function_array_alloc(s2);
 
    generic_function_kronh2(0,r,F->nrows,A->nrows,A->ncols,A->funcs,gfa1,t1);
    generic_function_kronh2(0,r,F->nrows,A->nrows,A->ncols,dA->funcs,gfa2,t2);
    generic_function_kronh2(0,r,F->nrows,A->nrows,A->ncols,A->funcs,gfa3,t3);

    for (ii = 0; ii < r; ii++){
        for (jj = 0; jj < A->nrows; jj++){
            for (kk = 0; kk < F->nrows; kk++){
                out->funcs[kk + jj*F->nrows + ii*2*A->nrows*F->nrows ]  = 
                    t1[kk + jj*F->nrows + ii*F->nrows*A->nrows];
                out->funcs[kk + (jj+A->nrows)*F->nrows + ii*2*A->nrows*F->nrows ]  = 
                    t2[kk + jj*F->nrows + ii*F->nrows*A->nrows];
                generic_function_axpy(1.0,
                    t3[kk + jj*F->nrows + ii*F->nrows*A->nrows],
                out->funcs[kk + (jj+A->nrows)*F->nrows + ii*2*A->nrows*F->nrows]);

            }
        }
    }

    generic_function_array_free(gfa1,sgfa); gfa1 = NULL;
    generic_function_array_free(gfa2,sgfa); gfa2 = NULL;
    generic_function_array_free(gfa3,sgfa); gfa3 = NULL;
    free(t1); t1 = NULL;
    free(t2); t2 = NULL;
    generic_function_array_free(t3,s2); t3 = NULL;
    free(mat1); mat1 = NULL;
    free(mat2); mat2 = NULL;
}

void dmrg_diffusion_firstright(struct Qmarray * dA, struct Qmarray * A,
                     struct Qmarray * ddF, struct Qmarray * dF, struct Qmarray * F,
                     double * mat, size_t r, struct Qmarray * out)
{
    
    assert (A->nrows == F->nrows);
    assert (out != NULL);

    struct Qmarray * temp = qmarray_alloc(2*A->nrows*F->nrows,r);
    dmrg_diffusion_midright(dA,A,ddF,dF,F,mat,r,temp);

    size_t ii,jj,kk;
    for (ii = 0; ii < r; ii++){
        for (jj = 0; jj < A->nrows; jj++){
            for (kk = 0; kk < F->nrows; kk++){
                out->funcs[kk + jj*F->nrows + ii*F->nrows*A->nrows] = 
                    generic_function_copy(
                        temp->funcs[kk + jj*F->nrows + ii*2*F->nrows*A->nrows]);
                generic_function_axpy(1.0,
                    temp->funcs[kk + (jj+A->nrows)*F->nrows + ii*2*F->nrows*A->nrows],
                    out->funcs[kk + jj*F->nrows + ii*F->nrows*A->nrows]);

            }
        }
    }

    qmarray_free(temp); temp = NULL;

}
