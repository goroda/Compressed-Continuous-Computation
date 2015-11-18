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

/** \file dmrg.c
 * Provides routines for dmrg based algorithms for the FT
 */

#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "array.h"
#include "lib_linalg.h"
#include "lib_clinalg.h"

struct QR * qr_reduced(struct Qmarray * a, int type)
{
    //printf("word type=%d\n",type);
    struct Qmarray * ac = qmarray_copy(a);
    struct QR * qr = NULL;
    if ( NULL == (qr = malloc(sizeof(struct QR)))){
        fprintf(stderr, "failed to allocate memory for QR decomposition.\n");
        exit(1);
    }
    
    if (type == 0){
        //printf("word \n");
        qr->right = 0;
        qr->mr = a->nrows;
        
        //printf("lets go\n");
        double * L = calloc_double(a->nrows * a->nrows);
        struct Qmarray * Q = qmarray_householder_simple("LQ",ac,L);
        //size_t ii = 0;
        //for (ii = 1; ii < a->nrows; ii++){
        //    if (fabs(L[ii*a->nrows+ii]) < 1e-14){
        //        break;
        //    }
       // }
        //printf("got LQ\n");
        //if (ii == a->nrows){
            qr->mat = L;
            qr->Q = Q;
            qr->mc = a->nrows;
       // }
       /*
        else{
            qr->mc = ii;
            qr->mat = calloc_double(qr->mr * qr->mc);
            qr->Q = qmarray_alloc(qr->mc,a->ncols);
            size_t jj,kk;
            for (jj = 0; jj < qr->mc; jj++){
                for (kk = 0; kk < qr->mr; kk++){
                    qr->mat[kk+jj*qr->mr] = L[kk + jj*a->nrows];
                }
                for (kk = 0; kk < a->ncols; kk++){
                    qr->Q->funcs[jj+kk*qr->mc] = generic_function_copy(Q->funcs[jj+kk*a->nrows]);
                }
            }
            qmarray_free(Q); Q = NULL;
            free(L); L = NULL;
        }
        */
    }
    else if (type == 1){
        qr->right = 1;
        qr->mc = a->ncols;

        double * R = calloc_double(qr->mc * qr->mc);
        struct Qmarray * Q = qmarray_householder_simple("QR",ac,R);
        //size_t ii = 0;
        //for (ii = 1; ii < qr->mc; ii++){
        //    if (fabs(R[ii*qr->mc+ii]) < 1e-14){
        //        break;
        //    }
        //}
        //if (ii == qr->mc){
            qr->mat = R;
            qr->Q = Q;
            qr->mr = qr->mc;
       // }
        /*
        else{
            qr->mr = ii;
            qr->mat = calloc_double(qr->mr * qr->mc);
            qr->Q = qmarray_alloc(a->nrows, qr->mr);
            size_t jj,kk;
            for (jj = 0; jj < qr->mc; jj++){
                for (kk = 0; kk < qr->mr; kk++){
                    qr->mat[kk+jj*qr->mr] = R[kk + jj*a->ncols];
                }
                for (kk = 0; kk < a->nrows; kk++){
                    qr->Q->funcs[kk + jj*a->nrows] =
                        generic_function_copy(Q->funcs[kk+jj*a->nrows]);
                }
            }
            qmarray_free(Q); Q = NULL;
            free(R); R = NULL;

        }
        */
    }
    else{
        fprintf(stderr, "Can't take reduced qr decomposition of type %d\n",type);
        exit(1);
    }
    qmarray_free(ac); ac = NULL;
    return qr;

}

void qr_free(struct QR * qr)
{
    if (qr != NULL){
        free(qr->mat); qr->mat = NULL;
        qmarray_free(qr->Q); qr->Q = NULL;
        free(qr); qr = NULL;
    }
}

struct QR ** qr_array_alloc(size_t n)
{
    struct QR ** qr = NULL;
    if ( NULL == (qr = malloc(n*sizeof(struct QR *)))){
        fprintf(stderr, "failed to allocate memory for QR array.\n");
        exit(1);
    }
    size_t ii;
    for (ii = 0; ii < n; ii++){
        qr[ii] = NULL;
    }
    return qr;
}


void qr_array_free(struct QR ** qr, size_t n)
{
    if (qr != NULL){
        size_t ii;
        for (ii = 0; ii < n; ii++){
            qr_free(qr[ii]); qr[ii] = NULL;
        }
        free(qr); qr = NULL;
    }
}


struct QR *
dmrg_update_right2(struct Qmarray * nextleft, struct QR * prev, struct Qmarray * z)
{

    double * val = qmaqmat_integrate(prev->Q,z);
    struct Qmarray * temp = qmam(nextleft,val,z->nrows);
    struct QR * lq = qr_reduced(temp,0);
    qmarray_free(temp); temp = NULL;
    free(val); val = NULL;
    return lq;
}

struct QR ** 
dmrg_update_right2_all(struct FunctionTrain * y, struct FunctionTrain * z)
{
    size_t dim = y->dim;
    struct QR ** right = NULL;
    if ( NULL == (right = malloc((dim-1)*sizeof(struct QR *)))){
        fprintf(stderr, "failed to allocate memory for dmrg all right QR decompositions.\n");
        exit(1);
    }
    right[dim-2] = qr_reduced(y->cores[dim-1],0);
    int ii;
    for (ii = dim-3; ii > -1; ii--){
        //printf("on core %d\n",ii);
        right[ii] = dmrg_update_right2(y->cores[ii+1], 
                                        right[ii+1],
                                        z->cores[ii+2]);
    }
    return right;
}

/***********************************************************//**
    Update \f$ \Psi \f$ for the dmrg equations

    \param psikp [in] - \f$ \Psi_{k+1} \f$
    \param left [in] - left core for update
    \param right [in] - right core for update

    \return val - \f$ \Psi_k \f$

    \note 
        Computing \f$ \Psi_k = \int left(x) \Psi_{k+1} right^T(x) dx \f$
***************************************************************/
double * 
dmrg_update_right(double * psikp, struct Qmarray * left, struct Qmarray * right)
{
    //printf("update right (%zu, %zu)\n",left->nrows,right->nrows);
    
    double * val = calloc_double(left->nrows * right->nrows);
    size_t nrows = left->nrows;
    size_t ncols = right->nrows;
    size_t ii,jj,kk,ll;
    for (ii = 0; ii < ncols; ii++){
        for (jj = 0; jj < nrows; jj++){
            for (kk = 0; kk < left->ncols; kk++){
                for (ll = 0; ll < right->ncols; ll++){
                    val[ii*nrows+jj] += psikp[ll*left->ncols+kk]*
                            generic_function_inner(
                                    left->funcs[kk*left->nrows+ jj],
                                    right->funcs[ll*right->nrows+ ii]);

                }
            }
        }
    }
    return val;
}

/***********************************************************//**
    Generate all \f$ \Psi_{i} \f$ for the dmrg for \f$ i = 0,\ldots,d-2\f$

    \param a [in] - left
    \param b [in] - right
    \param mats [inout] - allocated space for $d-2$ doubles representing the matrices

    \note 
        Computing \f$ \Psi_k = \int left(x) \Psi_{k+1} right^T(x) dx \f$
***************************************************************/
void dmrg_update_all_right(struct FunctionTrain * a, 
                           struct FunctionTrain * b, 
                           double ** mats)
{
    size_t ii;
    if (mats[a->dim-2] != NULL){
        free(mats[a->dim-2]); mats[a->dim-2] = NULL;
    }
    mats[a->dim-2] = calloc_double(1);
    mats[a->dim-2][0] = 1.0;
    for (ii = a->dim-3; ii > 0; ii--){
        free(mats[ii]); 
        mats[ii] = NULL;
        mats[ii] = dmrg_update_right(mats[ii+1],a->cores[ii+2],b->cores[ii+2]);
    }
    free(mats[0]); 
    mats[0] = NULL;
    mats[0] = dmrg_update_right(mats[1],a->cores[2],b->cores[2]);
}

/***********************************************************//**
    Update \f$ \Phi \f$ for the dmrg equations

    \param phik [in] - \f$ \Phi_{k} \f$
    \param left [in] - left core for update
    \param right [in] - right core for update

    \return val -  \f$ \Phi_{k+1} \f$

    \note 
        Computing \f$ \Phi_{k+1} = \int left(x)^T \Phi_{k} right(x) dx \f$
***************************************************************/
double * dmrg_update_left(double * phik, struct Qmarray * left, struct Qmarray * right)
{
    //printf("start update left\n");
    //
    
    size_t nrows = left->ncols;
    size_t ncols = right->ncols;
    double * val = calloc_double(nrows * ncols);
    size_t ii,jj,kk,ll;
    for (ii = 0; ii < ncols; ii++){
        for (jj = 0; jj < nrows; jj++){
            for (kk = 0; kk < left->nrows; kk++)
                for (ll = 0; ll < right->nrows; ll++)
                    val[ii*nrows+jj] += phik[ll*left->nrows+kk] * 
                                    generic_function_inner(
                                    left->funcs[jj*left->nrows+kk],
                                    right->funcs[ii*right->nrows+ll]);
        }
    }
    
    /*
    struct Qmarray * temp = mqma(phik, right, left->nrows);
    double * val = qmatqma_integrate(left,temp);
    qmarray_free(temp); temp = NULL;
    */

    return val;
}

/***********************************************************//**
    Perform a left-right dmrg sweep

    \param z [in] - initial guess
    \param y [in] - desired ft
    \param phi [inout] - left multipliers
    \param psi [in] - right multiplies
    \param epsilon [in] - splitting tolerance

    \return na - a new approximation
***************************************************************/
struct FunctionTrain * 
dmrg_sweep_lr2(struct FunctionTrain * z, struct FunctionTrain * y, 
            struct QR ** phil, struct QR ** psir, double epsilon)
{
    double * RL = NULL;
    size_t dim = z->dim;
    struct FunctionTrain * na = function_train_alloc(dim);
    na->ranks[0] = 1;
    na->ranks[na->dim] = 1;
    
    
    if (phil[0] == NULL){
        phil[0] = qr_reduced(y->cores[0],1);
    }
    
    size_t nrows = phil[0]->mr;
    size_t nmult = phil[0]->mc;
    size_t ncols = psir[0]->mc;

    RL = calloc_double(nrows * ncols);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,nrows,ncols,
                nmult,1.0,phil[0]->mat,nrows,psir[0]->mat,nmult,0.0,RL,nrows);

    double * u = NULL;
    double * vt = NULL;
    double * s = NULL;
    size_t rank = truncated_svd(nrows,ncols,nrows,RL,&u,&s,&vt,epsilon);
    na->ranks[1] = rank;
    na->cores[0] = qmam(phil[0]->Q,u,rank);
    
    size_t ii;
    for (ii = 1; ii < dim-1; ii++){
        double * newphi = calloc_double(rank * phil[ii-1]->mc);
        cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,rank,nmult,
                    nrows,1.0,u,nrows,phil[ii-1]->mat,nrows,0.0,newphi,rank);
        struct Qmarray * temp = mqma(newphi,y->cores[ii],rank);

        qr_free(phil[ii]); phil[ii] = NULL;
        phil[ii] = qr_reduced(temp,1);
        
        free(RL); RL = NULL;
        free(newphi); newphi = NULL;
        free(u); u = NULL;
        free(vt); vt = NULL;
        free(s); s = NULL;
        qmarray_free(temp); temp = NULL;

        nrows = phil[ii]->mr;
        nmult = phil[ii]->mc;
        ncols = psir[ii]->mc;

        RL = calloc_double(nrows * ncols);
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,nrows,ncols,
                    nmult,1.0,phil[ii]->mat,nrows,psir[ii]->mat,nmult,0.0,RL,nrows);

        rank = truncated_svd(nrows,ncols,nrows,RL,&u,&s,&vt,epsilon);
        na->ranks[ii+1] = rank;
        na->cores[ii] = qmam(phil[ii]->Q,u,rank);
    }
    
    size_t kk,jj;
    for (kk = 0; kk < ncols; kk++){
        for (jj = 0; jj < rank; jj++){
            vt[kk*rank+jj] = vt[kk*rank+jj]*s[jj];
        }
    }
    
    
    na->cores[dim-1] = mqma(vt,psir[dim-2]->Q,rank);

    free(RL); RL = NULL;
    free(u); u = NULL;
    free(vt); vt = NULL;
    free(s); s = NULL;

    return na;
}

/***********************************************************//**
    Perform a left-right dmrg sweep

    \param a [in] - initial guess
    \param b [in] - desired ft
    \param phi [inout] - left multipliers
    \param psi [in] - right multiplies
    \param epsilon [in] - splitting tolerance

    \return na - a new approximation
***************************************************************/
struct FunctionTrain * 
dmrg_sweep_lr(struct FunctionTrain * a, struct FunctionTrain * b, 
                double ** phi, double ** psi, double epsilon)
{
    double * L = NULL;
    double * R = NULL;
    double * RL = NULL;
    struct Qmarray * tempr = NULL;
    struct Qmarray * templ = NULL;
    struct FunctionTrain * na = function_train_alloc(a->dim);
    na->ranks[0] = 1;
    na->ranks[na->dim] = 1;
    
    if (phi[0] == NULL){
        phi[0] = calloc_double(1);
        phi[0][0] = 1;
    }
    size_t ii,jj,kk;
    size_t dimtemp, lsize,rsize;
    lsize = 1;
    for (ii = 0; ii < a->dim-1; ii++){
        dimtemp = b->cores[ii]->ncols;
        //printf("\n\n\n lets sweep right boys! %zu dimtemp=%zu \n", ii,dimtemp);
        if (ii == a->dim-2){
            rsize = 1;
        }
        else{
            rsize = a->cores[ii+1]->ncols;
        }

        // Deal with ii+1 first ... 
        //printf("ii+1\n");
        struct Qmarray * newcorer = qmam(b->cores[ii+1],psi[ii],rsize);
        //print_qmarray(newcorer,3,NULL);
        L = calloc_double(dimtemp*dimtemp);
        templ = qmarray_householder_simple("LQ",newcorer,L);
    
        //printf("newcorer (%zu,%zu)\n",newcorer->nrows,newcorer->ncols);
        //printf("templ (%zu,%zu)\n",templ->nrows,templ->ncols);
        //printf("L = \n");
        //dprint2d_col(dimtemp,dimtemp,L);
        // Deal with ii 
        //printf("ii\n");
        struct Qmarray * newcorel = mqma(phi[ii],b->cores[ii],lsize);
        R = calloc_double(dimtemp*dimtemp);
        tempr = qmarray_householder_simple("QR",newcorel,R);

        //printf("R = \n");
        //dprint2d_col(dimtemp,dimtemp,R);

        // do RL
        RL = calloc_double(dimtemp*dimtemp);
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,dimtemp,dimtemp,
                    dimtemp,1.0,R,dimtemp,L,dimtemp,0.0,RL,dimtemp);
        
        //printf("RL = \n");
        //dprint2d_col(dimtemp,dimtemp,RL);

        // Take svd of RL to find new ranks and split the cores
        double * u = NULL;
        double * vt = NULL;
        double * s = NULL;
        size_t rank = truncated_svd(dimtemp,dimtemp,dimtemp,RL,&u,&s,&vt,epsilon);
        na->ranks[ii+1] = rank;
        na->cores[ii] = qmam(tempr,u,rank);
        if ((ii+1) == a->dim-1){
            for (kk = 0; kk < dimtemp; kk++){
                for (jj = 0; jj < rank; jj++){
                    vt[kk*rank+jj] = vt[kk*rank+jj]*s[jj];
                }
            }
            //printf("last core updated \n");
            na->cores[ii+1] = mqma(vt,templ,rank);
        }
        
        //printf("new rank = %zu\n",rank);
        //printf("roo! %zu \n", ii);
        // now create new phi matrix
        if (ii < a->dim-2){
            free(phi[ii+1]); phi[ii+1] = NULL;
            phi[ii+1] = dmrg_update_left(phi[ii],na->cores[ii],b->cores[ii]);
        }

        lsize = a->cores[ii]->ncols;
        free(u); u = NULL;
        free(s); s = NULL;
        free(vt); vt = NULL;
        free(L); L = NULL;
        free(R); R = NULL;
        free(RL); RL = NULL;
        qmarray_free(newcorer); newcorer = NULL;
        qmarray_free(newcorel); newcorel = NULL;
        qmarray_free(templ); templ = NULL;
        qmarray_free(tempr); tempr = NULL;
    }

    return na;
}

/***********************************************************//**
    Perform a right-left dmrg sweep

    \param a [in] - initial guess
    \param b [in] - desired ft
    \param phi [in] - left multipliers
    \param psi [inout] - right multiplies
    \param epsilon [in] - splitting tolerance

    \return na - a new approximation
***************************************************************/
struct FunctionTrain * 
dmrg_sweep_rl(struct FunctionTrain * a, struct FunctionTrain * b, 
                double ** phi, double ** psi, double epsilon)
{
    double * L = NULL;
    double * R = NULL;
    double * RL = NULL;
    struct Qmarray * tempr = NULL;
    struct Qmarray * templ = NULL;
    struct FunctionTrain * na = function_train_alloc(a->dim);
    na->ranks[0] = 1;
    na->ranks[na->dim] = 1;
    
    if (psi[a->dim-2] == NULL){
        psi[a->dim-2] = calloc_double(1);
        psi[a->dim-2][0] = 1.0;
    }
    int ii;
    size_t jj,kk;
    size_t dimtemp, lsize,rsize;
    rsize = 1;
    for (ii = a->dim-2; ii > -1; ii--){
        //printf("lets sweep boys! %d \n", ii);

        dimtemp = b->cores[ii]->ncols;

        //printf("lets sweep boys! %d dimtemp=%zu \n", ii,dimtemp);
        if (ii == 0){
            lsize = 1;
        }
        else{
            lsize = a->cores[ii-1]->ncols;
        }

        // Deal with ii+1 first ... 
        struct Qmarray * newcorer = qmam(b->cores[ii+1],psi[ii],rsize);
        L = calloc_double(dimtemp*dimtemp);
        templ = qmarray_householder_simple("LQ",newcorer,L);
        //printf("L = \n");
        //dprint2d_col(dimtemp,dimtemp,L);

        // Deal with ii 
        struct Qmarray * newcorel = mqma(phi[ii],b->cores[ii],lsize);
        R = calloc_double(dimtemp*dimtemp);
        tempr = qmarray_householder_simple("QR",newcorel,R);
        //printf("R = \n");
        //dprint2d_col(dimtemp,dimtemp,R);

        // do RL
        RL = calloc_double(dimtemp*dimtemp);
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,dimtemp,dimtemp,
                    dimtemp,1.0,R,dimtemp,L,dimtemp,0.0,RL,dimtemp);
        
        //printf("RL = \n");
        //dprint2d_col(dimtemp,dimtemp,RL);
        // Take svd of RL to find new ranks and split the cores
        double * u = NULL;
        double * vt = NULL;
        double * s = NULL;
        size_t rank = truncated_svd(dimtemp,dimtemp,dimtemp,RL,&u,&s,&vt,epsilon);
        na->ranks[ii+1] = rank;
        na->cores[ii+1] = mqma(vt,templ,rank);

        //double * tcheck = calloc_double(rank*rank);
        //cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,rank,rank,dimtemp,1.0,vt,rank,vt,rank,0.0,tcheck,rank);

        //printf("rank = %zu\n",rank);
       // printf("cores[%d] shape = (%zu,%zu)\n",ii+1,
       //                     na->cores[ii+1]->nrows,na->cores[ii+1]->ncols);
        if (ii == 0){
            for (jj = 0; jj < rank; jj++){
                for (kk = 0; kk < dimtemp; kk++){
                    u[jj*dimtemp+kk] = u[jj*dimtemp+kk]*s[jj];
                }
            }
            //printf("first core updated\n");
            na->cores[ii] = qmam(tempr,u,rank);
          //  printf("cores[%d] shape = (%zu,%zu)\n",ii,
          //                  na->cores[ii]->nrows,na->cores[ii]->ncols);
        }
        
        //printf("roo! %zu \n", ii);
        // now create new psi matrix
        if (ii > 0){
            free(psi[ii-1]); psi[ii-1] = NULL;
            psi[ii-1] = dmrg_update_right(psi[ii],b->cores[ii+1],na->cores[ii+1]);
            //psi[ii-1] = dmrg_update_right(psi[ii],na->cores[ii+1],b->cores[ii+1]);
        }

        rsize = a->cores[ii+1]->nrows;
    
        free(u); u = NULL;
        free(s); s = NULL;
        free(vt); vt = NULL;
        free(L); L = NULL;
        free(R); R = NULL;
        free(RL); RL = NULL;
        qmarray_free(newcorer); newcorer = NULL;
        qmarray_free(newcorel); newcorel = NULL;
        qmarray_free(templ); templ = NULL;
        qmarray_free(tempr); tempr = NULL;
    }
    return na;
}

/***********************************************************//**
    Perform a left-right-left dmrg sweep

    \param a [in] - initial guess
    \param b [in] - desired ft
    \param phi [in] - left multipliers
    \param psi [inout] - right multiplies
    \param epsilon [in] - splitting tolerance

    \return na - a new approximation
***************************************************************/
struct FunctionTrain * 
dmrg_sweep_lrl(struct FunctionTrain * a, struct FunctionTrain * b, 
                double ** phi, double ** psi, double epsilon)
{

    struct FunctionTrain * temp = dmrg_sweep_lr(a,b,phi,psi,epsilon);
    struct FunctionTrain * na = dmrg_sweep_rl(temp,b,phi,psi,epsilon);
    function_train_free(temp); temp = NULL;
    return na;
}

/***********************************************************//**
    Find an approximation of an FT with another FT through dmrg sweeps

    \param a [inout] - initial guess (destroyed);
    \param b [in] - desired ft
***************************************************************/
struct FunctionTrain * dmrg_approx(struct FunctionTrain * a, struct FunctionTrain * b,
            double delta, size_t max_sweeps, int verbose, double epsilon)
{
    size_t dim = a->dim;
    struct FunctionTrain * ao = function_train_orthor(a);
    double ** phi = malloc((dim-1)*sizeof(double));
    double ** psi = malloc((dim-1)*sizeof(double));
    size_t ii;
    for (ii = 0; ii < dim-1; ii++){
        phi[ii] = NULL;
        psi[ii] = NULL;
    }
   // printf("okydokie\n");
    dmrg_update_all_right(b,ao,psi);
    //printf("radachokie\n");
    
    if (verbose == 2){
        printf("verbose works\n");
        printf("number of sweeps = %zu\n",max_sweeps);
        printf("Starting ranks = ");
        iprint_sz(dim+1,ao->ranks);
    }
    struct FunctionTrain * na = NULL;
    for (ii = 0; ii < max_sweeps; ii++){
        if (verbose > 0){
            printf("On dmrg_approx iteration (%zu/%zu)\n",ii,max_sweeps-1);
        }
        function_train_free(na); na = NULL;
        na = dmrg_sweep_lrl(ao,b,phi,psi,epsilon);

        //double diff = function_train_norm2diff(na,b);
        double diff = 10;
        function_train_free(ao); ao = NULL;
        //printf("diff is %G\n",diff);
        if (diff < delta){
            printf("diff is small\n");
        }
        else{
        //    ao = function_train_copy(na);
        }
        ao = function_train_copy(na);
    }
    function_train_free(ao); ao = NULL;
    for (ii = 0; ii < dim-1; ii++){
        free(phi[ii]); phi[ii] = NULL;
        free(psi[ii]); psi[ii] = NULL;
    }
    free(phi); phi = NULL;
    free(psi); psi = NULL;

    return na;
}
