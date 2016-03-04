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

/** \file qmarray_qr.c
 * Provides routines for fast computation of qr decomposition of a quasimatrix array
 */

#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>

#include "elements.h"
#include "algs.h"

#include "lib_funcs.h"
#include "array.h"

#define ZERO 1e2*DBL_EPSILON

int qmarray_qr(struct Qmarray * A, struct Qmarray ** Q, double ** R)
{

    size_t ii,kk,ll;

    double lb = generic_function_get_lower_bound(A->funcs[0]);
    double ub = generic_function_get_upper_bound(A->funcs[0]);
    size_t nrows = A->nrows;
    size_t ncols = A->ncols;
    enum poly_type ptype = LEGENDRE;
    if ( (*Q) == NULL){
        *Q = qmarray_orth1d_columns(POLYNOMIAL,&ptype,nrows,ncols,
                                    lb,ub); 
    }

   // print_qmarray(*Q,0,NULL);
    
    if ((*R) == NULL){
        *R = calloc_double(ncols*ncols);
    }
    
    struct Qmarray * V = qmarray_alloc(nrows,ncols);
    for (ii = 0; ii < nrows * ncols; ii++){
        V->funcs[ii] = generic_function_alloc_base(1);    
    }

    double s, rho, alpha, sigma,v;
    for (ii = 0; ii < ncols; ii++){
//        printf("ii = %zu \n", ii);

        rho = generic_function_array_norm(nrows,1,A->funcs+ii*nrows);
        (*R)[ii*ncols+ii] = rho;
        alpha = generic_function_inner_sum(nrows,1,
                                           (*Q)->funcs+ii*nrows,1,
                                           A->funcs+ii*nrows);
        
        //printf("rho = %G\n alpha=%G\n",rho,alpha);
        if (fabs(alpha) < ZERO){
            alpha = 0.0;
            s = 0.0;
        }
        else{
            s = -alpha / fabs(alpha);    
            if (s < 0.0){
                generic_function_array_flip_sign(nrows,1,
                                                 (*Q)->funcs+ii*nrows);
            }
        }

//        printf("s = %G\n",s);

        for (kk = 0; kk < nrows; kk++){
            generic_function_weighted_sum_pa(
                rho,(*Q)->funcs[ii*nrows+kk],-1.0,
                A->funcs[ii*nrows+kk],
                &(V->funcs[ii*nrows+kk]));
        }

        //improve orthogonality
        for (size_t zz = 0; zz < ii; zz++){
            v = generic_function_inner_sum(nrows,1,
                                           V->funcs+ii*nrows,
                                           1, (*Q)->funcs+zz*nrows);
            generic_function_array_axpy(nrows,-v,(*Q)->funcs+zz*nrows,
                                    V->funcs+ii*nrows);
        }

        sigma = 0.0;
        for (kk = 0; kk < nrows; kk++){
            sigma += generic_function_inner(V->funcs[ii*nrows+kk],
                                            V->funcs[ii*nrows+kk]);
        }

        sigma = sqrt(sigma);
        // weird that this line is needed for stabilization
        // Need to add reorthoganlization
        //sigma = sigma + 1e-16;
        if (fabs(sigma) < ZERO){
            //printf("less than zero sigma = %G,Zero=%G\n",sigma,ZERO);
            //V=E
            for (kk = 0; kk < nrows; kk++){
                generic_function_free(V->funcs[ii*nrows+kk]);
                V->funcs[ii*nrows+kk] = NULL;
                V->funcs[ii*nrows+kk] = generic_function_copy(
                    (*Q)->funcs[ii*nrows+kk]);
                //generic_function_copy_pa((*Q)->funcs[ii*nrows+kk],V->funcs[ii*nrows+kk]);
            }
        }
        else{
            generic_function_array_scale(1.0/sigma,V->funcs+ii*nrows,
                                         nrows);
        }
        
        //printf("start inner loop\n");
        double ev = generic_function_inner_sum(nrows,1,
                                               (*Q)->funcs+ii*nrows,1,
                                               V->funcs+ii*nrows);
        for (kk = ii+1; kk < ncols; kk++){
            double temp = 
                generic_function_inner_sum(nrows,1,
                                           V->funcs+ii*nrows,1,
                                           A->funcs+kk*nrows);
            (*R)[kk*ncols+ii]=
                generic_function_inner_sum(nrows,1,
                                           (*Q)->funcs+ii*nrows,1,
                                           A->funcs+kk*nrows);
            (*R)[kk*ncols+ii] += (-2.0 * ev * temp);
            for (ll = 0; ll < nrows; ll++){
                int success =
                    generic_function_axpy(-2.0*temp,
                                          V->funcs[ii*nrows+ll],
                                          A->funcs[kk*nrows+ll]);
                if (success == 1){
                    return 1;
                }
                //assert (success == 0);
                success = generic_function_axpy(
                    -(*R)[kk*ncols+ii],
                    (*Q)->funcs[ii*nrows+ll],
                    A->funcs[kk*nrows+ll]);

                if (success == 1){
                    return 1;
                }
//                assert (success == 0);
            }
        }
    }
    
    // form Q
    size_t counter = 0;
    ii = ncols-1;
    size_t jj;
    while (counter < ncols){
        
        for (jj = ii; jj < ncols; jj++){
            double val = generic_function_inner_sum(nrows,1,
                    (*Q)->funcs+jj*nrows, 1, V->funcs+ii*nrows);

            for (ll = 0; ll < nrows; ll++){
                generic_function_axpy(-2.0*val,V->funcs[ii*nrows+ll],
                                        (*Q)->funcs[jj*nrows+ll]);

            }
        }

        ii = ii - 1;
        counter += 1;
    }
    
    qmarray_free(V); V = NULL;
    return 0;
}

int qmarray_lq(struct Qmarray * A, struct Qmarray ** Q, double ** L)
{

    size_t ii,kk,ll;

    double lb = generic_function_get_lower_bound(A->funcs[0]);
    double ub = generic_function_get_upper_bound(A->funcs[0]);
    size_t nrows = A->nrows;
    size_t ncols = A->ncols;
    enum poly_type ptype = LEGENDRE;

    if ( (*Q) == NULL){
        *Q = qmarray_orth1d_rows(POLYNOMIAL,&ptype,nrows,ncols, lb, ub); 
    }

    if ((*L) == NULL){
        *L = calloc_double(nrows*nrows);
    }
    
    struct Qmarray * V = qmarray_alloc(nrows,ncols);
    for (ii = 0; ii < nrows * ncols; ii++){
        V->funcs[ii] = generic_function_alloc_base(1);    
    }

    double s, rho, alpha, sigma, v;
    for (ii = 0; ii < nrows; ii++){
        //printf("ii = %zu \n", ii);

        rho = generic_function_array_norm(ncols,nrows,A->funcs+ii);
        (*L)[ii*nrows+ii] = rho;
        alpha = generic_function_inner_sum(ncols,nrows,(*Q)->funcs+ii,nrows, A->funcs+ii);
        
        //printf("alpha=%G\n",alpha);
        if (fabs(alpha) < ZERO){
            alpha = 0.0;
            s = 0.0;
        }
        else{
            s = -alpha / fabs(alpha);    
            if (s < 0.0){
                generic_function_array_flip_sign(ncols,nrows,(*Q)->funcs+ii);
            }
        }
        
        for (kk = 0; kk < ncols; kk++){
            generic_function_weighted_sum_pa(
                rho,(*Q)->funcs[kk*nrows+ii],-1.0,
                A->funcs[kk*nrows+ii],
                &(V->funcs[kk*nrows+ii]));
        }

        //improve orthogonality
        for (size_t zz = 0; zz < ii; zz++){
            v = generic_function_inner_sum(ncols,nrows,
                                           V->funcs+ii,
                                           nrows, (*Q)->funcs+zz);
            for (size_t ll = 0; ll < ncols; ll++){
                generic_function_axpy(-v,(*Q)->funcs[ll*nrows+zz],
                                      V->funcs[ll*nrows+ii]);
            }
        }

        sigma = 0.0;
        for (kk = 0; kk < ncols;kk++){
            sigma += generic_function_inner(V->funcs[kk*nrows+ii],
                                            V->funcs[kk*nrows+ii]);
        }

        sigma = sqrt(sigma);
        // weird that this line is needed for stabilization
        // Need to add reorthoganlization
        //sigma = sigma + ZERO*1e-3;
        if (fabs(sigma) < ZERO){
            //printf("less than zero sigma = %G,Zero=%G\n",sigma,ZERO);
            //V=E
            for (kk = 0; kk < ncols; kk++){
                generic_function_free(V->funcs[kk*nrows+ii]);
                V->funcs[kk*nrows+ii] = NULL;
                V->funcs[kk*nrows+ii] = generic_function_copy((*Q)->funcs[kk*nrows+ii]);
                //generic_function_copy_pa((*Q)->funcs[ii*nrows+kk],V->funcs[ii*nrows+kk]);
            }
        }
        else{
            for (kk = 0; kk < ncols; kk++){
                generic_function_scale(1.0/sigma,V->funcs[kk*nrows+ii]);
            }
        }
        
        //printf("start inner loop\n");
        double ev = generic_function_inner_sum(ncols,nrows,(*Q)->funcs+ii,nrows,V->funcs+ii);
        for (kk = ii+1; kk < nrows; kk++){
            double temp = generic_function_inner_sum(ncols,nrows,V->funcs+ii,nrows,A->funcs+kk);
            (*L)[ii*nrows+kk]=generic_function_inner_sum(ncols,nrows,(*Q)->funcs+ii,nrows,A->funcs+kk);
            (*L)[ii*nrows+kk] += (-2.0 * ev * temp);
            for (ll = 0; ll < ncols; ll++){
                int success = generic_function_axpy(-2.0*temp,V->funcs[ll*nrows+ii],A->funcs[ll*nrows+kk]);
                assert (success == 0);
                success = generic_function_axpy(-(*L)[ii*nrows+kk],
                        (*Q)->funcs[ll*nrows+ii],A->funcs[ll*nrows+kk]);
                assert (success == 0);
            }
        }
    }
    
    // form Q
    size_t counter = 0;
    ii = nrows-1;
    size_t jj;
    while (counter < nrows){
        
        for (jj = ii; jj < nrows; jj++){
            double val = generic_function_inner_sum(ncols,nrows,
                    (*Q)->funcs+jj, nrows, V->funcs+ii);

            for (ll = 0; ll < ncols; ll++){
                generic_function_axpy(-2.0*val,V->funcs[ll*nrows+ii],
                                        (*Q)->funcs[ll*nrows+jj]);

            }
        }

        ii = ii - 1;
        counter += 1;
    }
    
    qmarray_free(V); V = NULL;
    return 0;
}

int qmarray_qr_gs(struct Qmarray * A, double ** R)
{
    size_t nrows = A->nrows;
    size_t ncols = A->ncols;
    if ((*R) == NULL){
        *R = calloc_double(ncols*ncols);
    }
    
    for (size_t ii = 0; ii < ncols; ii++ ){
        (*R)[ii*ncols+ii] = generic_function_array_norm(nrows,1,A->funcs + ii*nrows);
        if ((*R)[ii*ncols+ii] > ZERO){
            generic_function_array_scale(1.0/(*R)[ii*ncols+ii],A->funcs+ii*nrows,nrows);
            for (size_t jj = ii+1; jj < ncols; jj++){
                (*R)[jj*ncols+ii] = generic_function_inner_sum(nrows,1,A->funcs+ii*nrows,
                                                               1, A->funcs + jj*nrows);
                generic_function_array_axpy(nrows,-(*R)[jj*ncols+ii],A->funcs+ii*nrows,A->funcs+jj*nrows);
            }
        }
        else{
            printf("warning!!\n");
            printf("norm = %G\n",(*R)[ii*ncols+ii]);
            assert(1 == 0);
        }
    }
    return 0;
}


int qmarray_lq_gs(struct Qmarray * A, double ** R)
{
    size_t nrows = A->nrows;
    size_t ncols = A->ncols;
    if ((*R) == NULL){
        *R = calloc_double(nrows*nrows);
    }
    
    for (size_t ii = 0; ii < nrows; ii++ ){
        (*R)[ii*nrows+ii] = generic_function_array_norm(ncols,nrows,A->funcs + ii);
        if ((*R)[ii*nrows+ii] > ZERO){
            for (size_t jj = 0; jj < ncols; jj++){
                generic_function_scale(1.0/(*R)[ii*nrows+ii],A->funcs[jj*nrows+ii]);                
            }
            for (size_t jj = ii+1; jj < nrows; jj++){
                (*R)[ii*nrows+jj] = generic_function_inner_sum(ncols,nrows,A->funcs+ii,
                                                               nrows, A->funcs + jj);
                for (size_t kk = 0; kk < ncols; kk++){
                    generic_function_axpy(-(*R)[ii*nrows+jj],
                                          A->funcs[kk*nrows+ii],
                                          A->funcs[kk*nrows+jj]);
                }
            }
        }
        else{
            printf("norm = %G\n",(*R)[ii*nrows+ii]);
            assert (1 == 0);
        }
    }
    return 0;
}
