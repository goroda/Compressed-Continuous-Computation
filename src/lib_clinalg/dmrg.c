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

/** \file algs.c
 * Provides routines for dmrg based algorithms for the FT
 */
#include <stdlib.h>
#include <assert.h>

#include "array.h"
#include  "lib_linalg.h"
#include "lib_clinalg.h"

/***********************************************************//**
    Update \f$ \Psi \f$ for the dmrg equations

    \param psikp [in] - \f$ \Psi_{k+1} \f$
    \param left [in] - left core for update
    \param right [in] - right core for update

    \return val -  \f$ \Psi_k \f$

    \note
       \f$ \Psi_k = \int left(x) \Psi_{k+1} right^T(x) dx \f$
***************************************************************/
double * 
dmrg_update_right(double * psikp, struct Qmarray * left, struct Qmarray * right)
{
    //printf("update right\n");
    struct Qmarray * temp = qmam(left,psikp,right->ncols);
    //printf("got temp \n");
    struct Qmarray * temp2 = qmaqmat(temp,right);
    //printf("got temp2 \n");
    //print
    double * val = qmarray_integrate(temp2);

    qmarray_free(temp); temp = NULL;
    qmarray_free(temp2); temp2 = NULL;
    return val;
}

/***********************************************************//**
    Generate all \f$ \Psi_{i} \f$ for the dmrg for \f$ i = 0,\ldots,d-2\f$

    \param a [in] - left
    \param b [in] - right
    \param mats [inout] - allocated space for $d-2$ doubles representing the matrices

    \note
       \f$ \Psi_k = \int left(x) \Psi_{k+1} right^T(x) dx \f$
***************************************************************/
void dmrg_update_all_right(struct FunctionTrain * a, 
                           struct FunctionTrain * b, 
                           double ** mats)
{
    size_t ii;
    mats[a->dim-2] = calloc_double(1);
    mats[a->dim-2][0] = 1.0;
    //printf("ok\n");
    for (ii = a->dim-3; ii > 0; ii--){
        //printf("update ii = %zu \n",ii);
        //print_qmarray(a->cores[ii+2],0,NULL);
        //print_qmarray(b->cores[ii+2],0,NULL);
        //printf("go \n");
        mats[ii] = dmrg_update_right(mats[ii+1],a->cores[ii+2],b->cores[ii+2]);
    }
    //printf("finish\n");
    mats[0] = dmrg_update_right(mats[1],a->cores[2],b->cores[2]);
}

/***********************************************************//**
    Update \f$ \Phi \f$ for the dmrg equations

    \param phik [in] - \f$ \Phi_{k} \f$
    \param left [in] - left core for update
    \param right [in] - right core for update

    \return val -  \f$ \Phi_{k+1} \f$

    \note
       \f$ \Phi_{k+1} = \int left(x)^T \Psi_{k} right(x) dx \f$
***************************************************************/
double * dmrg_update_left(double * phik, struct Qmarray * left, struct Qmarray * right)
{
    //printf("start update left\n");
    struct Qmarray * temp = mqma(phik, right, left->nrows);
    //printf("\t\t\t ttttttttttt\n");
    //print_qmarray(temp,0,NULL);
    //printf("pow\n\n\n\n\n\n\n");
    //printf("left = \n");
    //print_qmarray(left,0,NULL);
    //printf("\n\n\n lasdkaldaskld \n\n\n");
    struct Qmarray * temp2 = qmatqma(left,temp);
    //print_qmarray(temp2,0,NULL);
    //printf("\n\n\n\npowpow\n");
    double * val = qmarray_integrate(temp2);
    //printf("wow \n");

    qmarray_free(temp); temp = NULL;
    qmarray_free(temp2); temp2 = NULL;
    return val;
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
        printf("\n\n\n lets sweep right boys! %zu dimtemp=%zu \n", ii,dimtemp);
        if (ii == a->dim-2){
            rsize = 1;
        }
        else{
            rsize = a->cores[ii+1]->ncols;
        }

        // Deal with ii+1 first ... 
        //printf("ii+1\n");
        struct Qmarray * newcorer = qmam(b->cores[ii+1],psi[ii],rsize);
        L = calloc_double(dimtemp*dimtemp);
        templ = qmarray_householder_simple("LQ",newcorer,L);

        printf("L = \n");
        dprint2d_col(dimtemp,dimtemp,L);
        // Deal with ii 
        //printf("ii\n");
        struct Qmarray * newcorel = mqma(phi[ii],b->cores[ii],lsize);
        R = calloc_double(dimtemp*dimtemp);
        tempr = qmarray_householder_simple("QR",newcorel,R);

        printf("R = \n");
        dprint2d_col(dimtemp,dimtemp,R);

        // do RL
        RL = calloc_double(dimtemp*dimtemp);
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,dimtemp,dimtemp,
                    dimtemp,1.0,R,dimtemp,L,dimtemp,0.0,RL,dimtemp);
        
        printf("RL = \n");
        dprint2d_col(dimtemp,dimtemp,RL);

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
            na->cores[ii+1] = mqma(vt,templ,rank);
        }
        
        //printf("new rank = %zu\n",rank);
        //printf("roo! %zu \n", ii);
        // now create new psi matrix
        if (ii < a->dim-2){
            free(phi[ii+1]); phi[ii+1] = NULL;
            //printf("phi=\n");
            //dprint2d_col(na->cores[ii]->nrows,b->cores[ii]->nrows,phi[ii]);
            //printf("new core\n");
            //print_qmarray(na->cores[ii],0,NULL);
            //printf("new core shape = (%zu,%zu)\n",na->cores[ii]->nrows, na->cores[ii]->ncols);
            //printf("b core shape = (%zu,%zu)\n",b->cores[ii]->nrows, b->cores[ii]->ncols);
            //printf("b core\n");
            //print_qmarray(b->cores[ii],0,NULL);
            //phi[ii+1] = dmrg_update_left(phi[ii],na->cores[ii],b->cores[ii]);
            phi[ii+1] = dmrg_update_left(phi[ii],b->cores[ii],na->cores[ii]);
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
        psi[a->dim-2][0] = 1;
    }
    int ii;
    size_t jj,kk;
    size_t dimtemp, lsize,rsize;
    rsize = 1;
    for (ii = a->dim-2; ii > -1; ii--){
        //printf("lets sweep boys! %d \n", ii);
        dimtemp = b->cores[ii]->ncols;

        printf("\n\n\nlets sweep boys! %d dimtemp=%zu \n", ii,dimtemp);
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
        printf("L = \n");
        dprint2d_col(dimtemp,dimtemp,L);

        // Deal with ii 
        struct Qmarray * newcorel = mqma(phi[ii],b->cores[ii],lsize);
        R = calloc_double(dimtemp*dimtemp);
        tempr = qmarray_householder_simple("QR",newcorel,R);
        printf("R = \n");
        dprint2d_col(dimtemp,dimtemp,R);

        // do RL
        RL = calloc_double(dimtemp*dimtemp);
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,dimtemp,dimtemp,
                    dimtemp,1.0,R,dimtemp,L,dimtemp,0.0,RL,dimtemp);
        
        printf("RL = \n");
        dprint2d_col(dimtemp,dimtemp,RL);
        // Take svd of RL to find new ranks and split the cores
        double * u = NULL;
        double * vt = NULL;
        double * s = NULL;
        size_t rank = truncated_svd(dimtemp,dimtemp,dimtemp,RL,&u,&s,&vt,epsilon);
        na->ranks[ii+1] = rank;
        na->cores[ii+1] = mqma(vt,templ,rank);
        printf("rank = %zu\n",rank);
       // printf("cores[%d] shape = (%zu,%zu)\n",ii+1,
       //                     na->cores[ii+1]->nrows,na->cores[ii+1]->ncols);
        if (ii == 0){
            for (jj = 0; jj < rank; jj++){
                for (kk = 0; kk < dimtemp; kk++){
                    u[jj*dimtemp+kk] = u[jj*dimtemp+kk]*s[jj];
                }
            }
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
    Find an approximation of an FT with another FT through dmrg sweeps

    \param a [inout] - initial guess and final solution
    \param b [in] - desired ft
***************************************************************/
void dmrg_approx(struct FunctionTrain ** a, struct FunctionTrain * b,
            double delta, size_t max_sweeps, int verbose, double epsilon)
{
    size_t dim = (*a)->dim;
    struct FunctionTrain * ao = function_train_orthor(*a);
    double ** phi = malloc((dim-1)*sizeof(double));
    double ** psi = malloc((dim-1)*sizeof(double));
    size_t ii;
    for (ii = 0; ii < dim-1; ii++){
        phi[ii] = NULL;
        psi[ii] = NULL;
    }
    printf("okydokie\n");
    dmrg_update_all_right(b,ao,psi);
    //dmrg_update_all_right(ao,b,psi);
    printf("radachokie\n");
    
    struct FunctionTrain * temp = NULL;
    double diff;
    for (ii = 0; ii < max_sweeps; ii++){
        printf("On dmrg_approx iteration (%zu/%zu)\n",ii,max_sweeps-1);
        function_train_free(*a); *a = NULL;
        printf("left right\n");
        temp = dmrg_sweep_lr(ao,b,phi,psi,epsilon);
        //*a = function_train_copy(temp);
        //break;
        printf("temp ranks = ");
        iprint_sz(dim+1,temp->ranks);
        printf("right left \n");
        *a = dmrg_sweep_rl(temp,b,phi,psi,epsilon);
     
        printf("new ranks = ");
        iprint_sz(dim+1,(*a)->ranks);
        diff = function_train_norm2diff(*a,ao);
        if (verbose > 0){
            printf("\t diff = %G\n",diff);
        }
        function_train_free(ao); ao = NULL;
        function_train_free(temp); temp = NULL;
        //if (diff < delta){
        if (1 == 2){
            break;
        }
        else{
            ao = function_train_copy(*a);
        }
    }

    function_train_free(ao); ao = NULL;
    for (ii = 0; ii < dim-1; ii++){
        free(phi[ii]); phi[ii] = NULL;
        free(psi[ii]); psi[ii] = NULL;
    }
    free(phi); phi = NULL;
    free(psi); psi = NULL;


}
