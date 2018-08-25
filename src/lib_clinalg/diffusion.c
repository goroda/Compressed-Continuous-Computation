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





/** \file diffusion.c
 * Provides routines for applying diffusion operator
 */

#include <stdlib.h>
#include <assert.h>

#include "diffusion.h"

/// @private
void dmrg_diffusion_midleft(struct Qmarray * dA, struct Qmarray * A,
                 struct Qmarray * ddF, struct Qmarray * dF, struct Qmarray * F,
                 double * mat, size_t r, struct Qmarray * out)
{
    assert (out != NULL);
    
    size_t sgfa = r * F->ncols * A->nrows;
    size_t offset = r * A->nrows * F->nrows;
    size_t width = r * A->ncols * F->ncols;

    struct GenericFunction ** gfa1 = generic_function_array_alloc(sgfa);
    struct GenericFunction ** gfa2 = generic_function_array_alloc(sgfa);
    struct GenericFunction ** gfa3 = generic_function_array_alloc(sgfa);
    struct GenericFunction ** gfa4 = generic_function_array_alloc(sgfa);

    struct GenericFunction ** temp1 = generic_function_array_alloc(width);
    struct GenericFunction ** temp2 = generic_function_array_alloc(width);

    generic_function_kronh(1,r,A->nrows,F->nrows,F->ncols,mat,F->funcs,gfa1);
    generic_function_kronh(1,r,A->nrows,F->nrows,F->ncols,mat+offset,dF->funcs,gfa2);
    generic_function_kronh(1,r,A->nrows,F->nrows,F->ncols,mat+offset,ddF->funcs,gfa3);
    generic_function_kronh(1,r,A->nrows,F->nrows,F->ncols,mat+offset,F->funcs,gfa4);

    generic_function_kronh2(1,r,F->ncols,A->nrows,A->ncols,A->funcs,gfa1,out->funcs);
    generic_function_kronh2(1,r,F->ncols,A->nrows,A->ncols,dA->funcs,gfa2,temp1);
    generic_function_kronh2(1,r,F->ncols,A->nrows,A->ncols,A->funcs,gfa3,temp2);

    generic_function_kronh2(1,r,F->ncols,A->nrows,A->ncols,A->funcs,gfa4,out->funcs+width);
    size_t ii;
    for (ii = 0; ii < width; ii++){
        generic_function_axpy(1.0,temp1[ii],out->funcs[ii]);
        generic_function_axpy(1.0,temp2[ii],out->funcs[ii]);
    }

    generic_function_array_free(gfa1,sgfa); gfa1 = NULL;
    generic_function_array_free(gfa2,sgfa); gfa2 = NULL;
    generic_function_array_free(gfa3,sgfa); gfa3 = NULL;
    generic_function_array_free(gfa4,sgfa); gfa4 = NULL;
    generic_function_array_free(temp1,width); temp1 = NULL;
    generic_function_array_free(temp2,width); temp2 = NULL;
}

/// @private
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
    struct GenericFunction ** gfa4 = generic_function_array_alloc(sgfa);
    
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
    generic_function_kronh(0,r,A->ncols,F->nrows,F->ncols,mat1,dF->funcs,gfa2);
    generic_function_kronh(0,r,A->ncols,F->nrows,F->ncols,mat1,ddF->funcs,gfa3);
    generic_function_kronh(0,r,A->ncols,F->nrows,F->ncols,mat2,F->funcs,gfa4);

    size_t s2 = r * A->nrows * F->nrows;
    struct GenericFunction ** t1 = generic_function_array_alloc(s2);
    struct GenericFunction ** t2 = generic_function_array_alloc(s2);
    struct GenericFunction ** t3 = generic_function_array_alloc(s2);
    struct GenericFunction ** t4 = generic_function_array_alloc(s2);
 
    generic_function_kronh2(0,r,F->nrows,A->nrows,A->ncols,A->funcs,gfa1,t1);
    generic_function_kronh2(0,r,F->nrows,A->nrows,A->ncols,dA->funcs,gfa2,t2);
    generic_function_kronh2(0,r,F->nrows,A->nrows,A->ncols,A->funcs,gfa3,t3);
    generic_function_kronh2(0,r,F->nrows,A->nrows,A->ncols,A->funcs,gfa4,t4);

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
                generic_function_axpy(1.0,
                    t4[kk + jj*F->nrows + ii*F->nrows*A->nrows],
                    out->funcs[kk + (jj+A->nrows)*F->nrows + ii*2*A->nrows*F->nrows]);

            }
        }
    }

    generic_function_array_free(gfa1,sgfa); gfa1 = NULL;
    generic_function_array_free(gfa2,sgfa); gfa2 = NULL;
    generic_function_array_free(gfa3,sgfa); gfa3 = NULL;
    generic_function_array_free(gfa4,sgfa); gfa4 = NULL;
    free(t1); t1 = NULL;
    free(t2); t2 = NULL;
    generic_function_array_free(t3,s2); t3 = NULL;
    generic_function_array_free(t4,s2); t4 = NULL;
    free(mat1); mat1 = NULL;
    free(mat2); mat2 = NULL;
}

/// @private
void dmrg_diffusion_firstright(struct Qmarray * dA, struct Qmarray * A,
                     struct Qmarray * ddF, struct Qmarray * dF, struct Qmarray * F,
                     double * mat, size_t r, struct Qmarray * out)
{
    
    assert (A->nrows == F->nrows);
    assert (out != NULL);

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

    size_t sgfa = r * F->nrows * A->ncols;
    struct GenericFunction ** gfa1 = generic_function_array_alloc(sgfa);
    struct GenericFunction ** gfa2 = generic_function_array_alloc(sgfa);
    struct GenericFunction ** gfa3 = generic_function_array_alloc(sgfa);

    generic_function_kronh(0,r,A->ncols,F->nrows,F->ncols,mat2,F->funcs,gfa1);
    generic_function_kronh(0,r,A->ncols,F->nrows,F->ncols,mat1,dF->funcs,gfa2);
    generic_function_kronh(0,r,A->ncols,F->nrows,F->ncols,mat1,ddF->funcs,gfa3);

    size_t s2 = r * A->nrows * F->nrows;
    struct GenericFunction ** t2 = generic_function_array_alloc(s2);
    struct GenericFunction ** t3 = generic_function_array_alloc(s2);
 
    generic_function_kronh2(0,r,F->nrows,A->nrows,A->ncols,A->funcs,gfa1,out->funcs);
    generic_function_kronh2(0,r,F->nrows,A->nrows,A->ncols,dA->funcs,gfa2,t2);
    generic_function_kronh2(0,r,F->nrows,A->nrows,A->ncols,A->funcs,gfa3,t3);

    for (ii = 0; ii < r; ii++){
        for (jj = 0; jj < A->nrows; jj++){
            for (kk = 0; kk < F->nrows; kk++){
                generic_function_axpy(1.0,
                    t2[kk + jj*F->nrows + ii*F->nrows*A->nrows],
                    out->funcs[kk + jj*F->nrows + ii*A->nrows*F->nrows]);
                generic_function_axpy(1.0,
                    t3[kk + jj*F->nrows + ii*F->nrows*A->nrows],
                    out->funcs[kk + jj*F->nrows + ii*A->nrows*F->nrows]);
            }
        }
    }

    generic_function_array_free(gfa1,sgfa); gfa1 = NULL;
    generic_function_array_free(gfa2,sgfa); gfa2 = NULL;
    generic_function_array_free(gfa3,sgfa); gfa3 = NULL;
    generic_function_array_free(t2,s2); t2 = NULL;
    generic_function_array_free(t3,s2); t3 = NULL;
    free(mat1); mat1 = NULL;
    free(mat2); mat2 = NULL;

}

/// @private
void dmrg_diffusion_support(char type, size_t core, size_t r, double * mat,
                        struct Qmarray ** out, void * args)
{
    assert (*out == NULL);
    struct DmDiff * dd = args;
    size_t dim = dd->A->dim;
    //printf("in \n");
    if (type == 'L'){
        if (core == 0){
            struct Qmarray * t1 = qmarray_kron(dd->A->cores[0], dd->F->cores[0]);
            struct Qmarray * t2 = qmarray_kron(dd->dA[0],dd->dF[0]);
            struct Qmarray * t3 = qmarray_kron(dd->A->cores[0],dd->ddF[0]);
            qmarray_axpy(1.0,t3,t2);
            /* printf("size A->cores[0] = (%zu,%zu)\n",dd->A->cores[0]->nrows,dd->A->cores[0]->ncols); */
            /* printf("size t2 = (%zu,%zu)\n",t2->nrows,t2->ncols); */
            *out = qmarray_stackh(t2,t1);
            qmarray_free(t1); t1 = NULL;
            qmarray_free(t2); t2 = NULL;
            qmarray_free(t3); t3 = NULL;
        }
        else if (core < dim-1){
            size_t ra = dd->A->cores[core]->ncols;
            size_t rf = dd->F->cores[core]->ncols;
            *out = qmarray_alloc(r,2*ra*rf);
            dmrg_diffusion_midleft(dd->dA[core],
                                   dd->A->cores[core],
                                   dd->ddF[core],
                                   dd->dF[core],
                                   dd->F->cores[core],mat,r,*out);
        }
        else { //core dim-1
            size_t ra = dd->A->cores[core]->ncols;
            size_t rf = dd->F->cores[core]->ncols;
            *out = qmarray_alloc(r,ra*rf);
            dmrg_diffusion_lastleft(dd->dA[core],
                                    dd->A->cores[core],
                                    dd->ddF[core],
                                    dd->dF[core],
                                    dd->F->cores[core],
                                    mat,r,*out);
        }
    }
    else if (type == 'R'){
        if (core == dim-1){
            struct Qmarray * t1 = qmarray_kron(dd->A->cores[core],dd->F->cores[core]);
            struct Qmarray * t2 = qmarray_kron(dd->dA[core],dd->dF[core]);
            struct Qmarray * t3 = qmarray_kron(dd->A->cores[core],dd->ddF[core]);
            qmarray_axpy(1.0,t3,t2);
            *out = qmarray_stackv(t1,t2);
            qmarray_free(t1); t1 = NULL;
            qmarray_free(t2); t2 = NULL;
            qmarray_free(t3); t3 = NULL;
        }
        else if (core > 0){
            size_t ra = dd->A->cores[core]->nrows;
            size_t rf = dd->F->cores[core]->nrows;
            *out = qmarray_alloc(2*ra*rf,r);
            dmrg_diffusion_midright(dd->dA[core],
                                    dd->A->cores[core],
                                    dd->ddF[core],
                                    dd->dF[core],
                                    dd->F->cores[core],
                                    mat,r,*out);
        }
        else{ // core == 0
            size_t ra = dd->A->cores[core]->nrows;
            size_t rf = dd->F->cores[core]->nrows;
            *out = qmarray_alloc(ra*rf,r);
            dmrg_diffusion_firstright(dd->dA[core],
                                      dd->A->cores[core],
                                      dd->ddF[core],
                                      dd->dF[core],
                                      dd->F->cores[core],mat,r,*out);
        }
    }
}

/***********************************************************//**
    Compute \f[ z(x) = \nabla \cdot \left[ a(x) \nabla f(x) \right] \f] using ALS+DMRG

    \param[in] a          - scalar coefficients
    \param[in] f          - input to diffusion operator
    \param[in] delta      - threshold to stop iterating
    \param[in] max_sweeps - maximum number of left-right-left sweeps 
    \param[in] epsilon    - SVD tolerance for rank determination
    \param[in] verbose    - verbosity level 0 or >0
    \param[in] opts       - approximation options

    \return na - approximate application of diffusion operator
***************************************************************/
struct FunctionTrain * dmrg_diffusion(
    struct FunctionTrain * a,
    struct FunctionTrain * f,
    double delta, size_t max_sweeps, double epsilon, int verbose,
    struct MultiApproxOpts * opts)
{
    
    size_t dim = f->dim;

    struct DmDiff dd;

    dd.A = a;
    dd.F = f;

    dd.dA = malloc(dim * sizeof(struct Qmarray *));
    dd.dF = malloc(dim * sizeof(struct Qmarray *));
    dd.ddF = malloc(dim * sizeof(struct Qmarray *));
    
    if (dd.dA == NULL || dd.dF == NULL || dd.ddF == NULL){
        fprintf(stderr, "Could not allocate memory in dmrg_diffusion\n");
        return NULL;
    }

    size_t ii;
    for (ii = 0; ii < dim; ii++){
        dd.dA[ii] = qmarray_deriv(a->cores[ii]);
        dd.dF[ii] = qmarray_deriv(f->cores[ii]);
        dd.ddF[ii] = qmarray_deriv(dd.dF[ii]);
    }

    struct FunctionTrain * guess = function_train_copy(f);
    struct FunctionTrain * na = dmrg_approx(guess,dmrg_diffusion_support,&dd,
                                            delta,max_sweeps,epsilon,verbose,opts);

    for (ii = 0; ii < dim; ii++){
        qmarray_free(dd.dA[ii]); dd.dA[ii] = NULL;
        qmarray_free(dd.dF[ii]); dd.dF[ii] = NULL;
        qmarray_free(dd.ddF[ii]); dd.ddF[ii] = NULL;
    }

    free(dd.dA); dd.dA = NULL;
    free(dd.dF); dd.dF = NULL;
    free(dd.ddF); dd.ddF = NULL;

    function_train_free(guess); guess = NULL;

    return na;
}

/***********************************************************//**
    Compute \f[ z(x) = \nabla \cdot \left[ a(x) \nabla f(x) \right] \f] exactly

    \param[in] a    - scalar coefficients
    \param[in] f    - input to diffusion operator
    \param[in] opts - approxmation options

    \return out - exact application of diffusion operator 
    
    \note 
        Result is not rounded. Might want to perform rounding afterwards
***************************************************************/
struct FunctionTrain * exact_diffusion(
    struct FunctionTrain * a, struct FunctionTrain * f, struct MultiApproxOpts * opts)
{
    size_t dim = a->dim;
    struct FunctionTrain * out = function_train_alloc(dim);
    out->ranks[0] = 1;
    out->ranks[dim] = 1;

    struct Qmarray ** da = malloc(dim * sizeof(struct Qmarray *));
    struct Qmarray ** df = malloc(dim * sizeof(struct Qmarray *));
    struct Qmarray ** ddf = malloc(dim * sizeof(struct Qmarray *));
    
    if (da == NULL || df == NULL || ddf == NULL){
        fprintf(stderr, "Could not allocate memory in dmrg_diffusion\n");
        return NULL;
    }

    size_t ii;
    for (ii = 0; ii < dim; ii++){
        da[ii] = qmarray_deriv(a->cores[ii]);
        df[ii] = qmarray_deriv(f->cores[ii]);
        ddf[ii] = qmarray_deriv(df[ii]);
    }

    struct Qmarray * l1 = qmarray_kron(a->cores[0],ddf[0]);
    struct Qmarray * l2 = qmarray_kron(da[0],df[0]);
    qmarray_axpy(1.0,l1,l2);
    struct Qmarray * l3 = qmarray_kron(a->cores[0],f->cores[0]);
    out->cores[0] = qmarray_stackh(l2,l3);
    out->ranks[1] = out->cores[0]->ncols;
    qmarray_free(l1); l1 = NULL;
    qmarray_free(l2); l2 = NULL;
    qmarray_free(l3); l3 = NULL;

    struct OneApproxOpts * o = NULL; 
    for (ii = 1; ii < dim-1; ii++){
        o = multi_approx_opts_get_aopts(opts,ii);
        struct Qmarray * addf = qmarray_kron(a->cores[ii],ddf[ii]);
        struct Qmarray * dadf = qmarray_kron(da[ii],df[ii]);
        qmarray_axpy(1.0,addf,dadf);
        struct Qmarray * af = qmarray_kron(a->cores[ii],f->cores[ii]);
        struct Qmarray * zer = qmarray_zeros(af->nrows,af->ncols,o);
        struct Qmarray * l3l1 = qmarray_stackv(af,dadf);
        struct Qmarray * zl3 = qmarray_stackv(zer,af);
        out->cores[ii] = qmarray_stackh(l3l1,zl3);
        out->ranks[ii+1] = out->cores[ii]->ncols;
        qmarray_free(addf); addf = NULL;
        qmarray_free(dadf); dadf = NULL;
        qmarray_free(af); af = NULL;
        qmarray_free(zer); zer = NULL;
        qmarray_free(l3l1); l3l1 = NULL;
        qmarray_free(zl3); zl3 = NULL;
    }
    ii = dim-1;
    l1 = qmarray_kron(a->cores[ii],ddf[ii]);
    l2 = qmarray_kron(da[ii],df[ii]);
    qmarray_axpy(1.0,l1,l2);
    l3 = qmarray_kron(a->cores[ii],f->cores[ii]);
    out->cores[ii] = qmarray_stackv(l3,l2);
    out->ranks[ii+1] = out->cores[ii]->ncols;
    qmarray_free(l1); l1 = NULL;
    qmarray_free(l2); l2 = NULL;
    qmarray_free(l3); l3 = NULL;

    for (ii = 0; ii < dim; ii++){
        qmarray_free(da[ii]); da[ii] = NULL;
        qmarray_free(df[ii]); df[ii] = NULL;
        qmarray_free(ddf[ii]); ddf[ii] = NULL;
    }
    free(da); da = NULL;
    free(df); df = NULL;
    free(ddf); ddf = NULL;

    return out;
}


/***********************************************************//**
    Compute \f[ z(x) = \nabla \cdot \left[ \nabla f(x) \right] \f] exactly

    \param[in] fin  - input to diffusion operator
    \param[in] opts - approximation options

    \return out - exact application of laplace operator 
    
    \note 
    Result is not rounded. Might want to perform rounding afterwards
***************************************************************/
struct FunctionTrain * exact_laplace(struct FunctionTrain * fin,
                                     struct MultiApproxOpts * opts)
{
    struct FunctionTrain * f = function_train_copy(fin);
    size_t dim = f->dim;
    struct FunctionTrain * out = function_train_alloc(dim);
    out->ranks[0] = f->ranks[0];
    out->ranks[dim] = f->ranks[dim];

    struct Qmarray ** ddf = malloc(dim * sizeof(struct Qmarray *));
    
    if (ddf == NULL){
        fprintf(stderr,
                "Could not allocate memory in exact_laplace_periodic\n");
        return NULL;
    }

    size_t ii;
    for (ii = 0; ii < dim; ii++){
        ddf[ii] = qmarray_dderiv(f->cores[ii]);
    }

    out->cores[0] = qmarray_stackh(ddf[0], f->cores[0]);
    out->ranks[1] = out->cores[0]->ncols;

    struct OneApproxOpts * o = NULL; 
    for (ii = 1; ii < dim-1; ii++){
        o = multi_approx_opts_get_aopts(opts,ii);
        // dimensions of blocks
        size_t nrows = f->cores[ii]->nrows;
        size_t ncols = f->cores[ii]->ncols;
        struct Qmarray * zer = qmarray_zeros(nrows,ncols,o);
        struct Qmarray * first_col = qmarray_stackv(f->cores[ii], ddf[ii]);
        struct Qmarray * second_col = qmarray_stackv(zer, f->cores[ii]);

        out->cores[ii] = qmarray_stackh(first_col, second_col);
        out->ranks[ii+1] = out->cores[ii]->ncols;

        qmarray_free(zer); zer = NULL;
        qmarray_free(first_col); first_col = NULL;
        qmarray_free(second_col); second_col = NULL;

    }
    ii = dim-1;
    out->cores[ii] = qmarray_stackv(f->cores[ii],ddf[ii]);
    out->ranks[ii+1] = out->cores[ii]->ncols;

    for (ii = 0; ii < dim; ii++){
        qmarray_free(ddf[ii]); ddf[ii] = NULL;
    }
    free(ddf); ddf = NULL;
    return out;
}



/***********************************************************//**
    Compute \f[ z(x) = \nabla \cdot \left[ \nabla f(x) \right] \f] exactly
    enforce boundary conditions

    \param[in] f    - input to diffusion operator
    \param[in] opts - approximation options

    \return out - exact application of diffusion operator 
    
    \note 
        Result is not rounded. Might want to perform rounding afterwards
***************************************************************/
struct FunctionTrain *
exact_laplace_periodic(struct FunctionTrain * f,
                       struct MultiApproxOpts * opts)
{
    size_t dim = f->dim;
    struct FunctionTrain * out = function_train_alloc(dim);
    out->ranks[0] = f->ranks[0];
    out->ranks[dim] = f->ranks[dim];

    struct Qmarray ** ddf = malloc(dim * sizeof(struct Qmarray *));
    
    if (ddf == NULL){
        fprintf(stderr,
                "Could not allocate memory in exact_laplace_periodic\n");
        return NULL;
    }

    size_t ii;
    for (ii = 0; ii < dim; ii++){
        ddf[ii] = qmarray_dderiv_periodic(f->cores[ii]);
    }

    out->cores[0] = qmarray_stackh(ddf[0], f->cores[0]);
    out->ranks[1] = out->cores[0]->ncols;

    struct OneApproxOpts * o = NULL; 
    for (ii = 1; ii < dim-1; ii++){
        o = multi_approx_opts_get_aopts(opts,ii);
        // dimensions of blocks
        size_t nrows = f->cores[ii]->nrows;
        size_t ncols = f->cores[ii]->ncols;
        struct Qmarray * zer = qmarray_zeros(nrows,ncols,o);
        struct Qmarray * first_col = qmarray_stackv(f->cores[ii], ddf[ii]);
        struct Qmarray * second_col = qmarray_stackv(zer, f->cores[ii]);

        out->cores[ii] = qmarray_stackh(first_col, second_col);
        out->ranks[ii+1] = out->cores[ii]->ncols;

        qmarray_free(zer); zer = NULL;
        qmarray_free(first_col); first_col = NULL;
        qmarray_free(second_col); second_col = NULL;

    }
    ii = dim-1;
    out->cores[ii] = qmarray_stackv(f->cores[ii],ddf[ii]);
    out->ranks[ii+1] = out->cores[ii]->ncols;

    for (ii = 0; ii < dim; ii++){
        qmarray_free(ddf[ii]); ddf[ii] = NULL;
    }
    free(ddf); ddf = NULL;

    return out;
}

/***********************************************************//**
    Compute \f[ z(x) = \nabla \cdot \left[ \nabla f(x) \right] \f] 
    Takes an operator that computes the laplacian of each univariate generic function

    \param[in] f    - input to diffusion operator
    \param[in] op   - (dim, ) array of operators that compute the second derivative in each dimension
    \param[in] opts - approximation options

    \return out - exact application of laplace operator 
    
    \note 
    Result is not rounded. Might want to perform rounding afterwards
***************************************************************/
struct FunctionTrain *
exact_laplace_op(struct FunctionTrain * f,
                 struct Operator ** op,
                 struct MultiApproxOpts * opts)
{
    if (op == NULL){
        return exact_laplace(f, opts);
    }
    size_t dim = f->dim;
    struct FunctionTrain * out = function_train_alloc(dim);
    out->ranks[0] = f->ranks[0];
    out->ranks[dim] = f->ranks[dim];

    struct Qmarray ** ddf = malloc(dim * sizeof(struct Qmarray *));
    
    if (ddf == NULL){
        fprintf(stderr,
                "Could not allocate memory in exact_laplace_periodic\n");
        return NULL;
    }

    size_t ii;
    for (ii = 0; ii < dim; ii++){
        ddf[ii] = qmarray_operate_elements(f->cores[ii], op[ii]);
    }

    out->cores[0] = qmarray_stackh(ddf[0], f->cores[0]);
    out->ranks[1] = out->cores[0]->ncols;

    struct OneApproxOpts * o = NULL; 
    for (ii = 1; ii < dim-1; ii++){
        o = multi_approx_opts_get_aopts(opts,ii);
        // dimensions of blocks
        size_t nrows = f->cores[ii]->nrows;
        size_t ncols = f->cores[ii]->ncols;
        struct Qmarray * zer = qmarray_zeros(nrows,ncols,o);
        struct Qmarray * first_col = qmarray_stackv(f->cores[ii], ddf[ii]);
        struct Qmarray * second_col = qmarray_stackv(zer, f->cores[ii]);

        out->cores[ii] = qmarray_stackh(first_col, second_col);
        out->ranks[ii+1] = out->cores[ii]->ncols;

        qmarray_free(zer); zer = NULL;
        qmarray_free(first_col); first_col = NULL;
        qmarray_free(second_col); second_col = NULL;

    }
    ii = dim-1;
    out->cores[ii] = qmarray_stackv(f->cores[ii],ddf[ii]);
    out->ranks[ii+1] = out->cores[ii]->ncols;

    for (ii = 0; ii < dim; ii++){
        qmarray_free(ddf[ii]); ddf[ii] = NULL;
    }
    free(ddf); ddf = NULL;
    return out;
}





