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





/** \file dmrg.c
 * Provides routines for dmrg based algorithms for the FT
 */

#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "dmrg.h"
#include "lib_linalg.h"

/***********************************************************//**
    Perform and store a reduced QR or LQ decompositions

    \param[in] a    - qmarray to decompose
    \param[in] type - 1 then QR, 0 then LQ
    \param[in] o    - approximation options

    \return qr - QR structure

    \note
        Doesnt actually do reduced QR because it seems to break the algorithm
***************************************************************/
struct QR * qr_reduced(const struct Qmarray * a, int type, struct OneApproxOpts * o)
{
    struct Qmarray * ac = qmarray_copy(a);
    struct QR * qr = NULL;
    if ( NULL == (qr = malloc(sizeof(struct QR)))){
        fprintf(stderr, "failed to allocate memory for QR decomposition.\n");
        exit(1);
    }
    qr->mat = NULL;
    qr->Q = NULL;
    int success = 0;
    if (type == 0){
        qr->right = 0;
        qr->mr = a->nrows;
        qr->mc = a->nrows;
        success = qmarray_lq(ac,&(qr->Q),&(qr->mat),o);
        assert (success == 0);
    }
    else if (type == 1){
        qr->right = 1;
        qr->mc = a->ncols;
        qr->mr = a->ncols;
        success = qmarray_qr(ac,&(qr->Q),&(qr->mat),o);
        assert (success == 0);
    }
    else{
        fprintf(stderr, "Can't take reduced qr decomposition of type %d\n",type);
        exit(1);
    }
    qmarray_free(ac); ac = NULL;
    return qr;

}

/***********************************************************//**
    Free memory allocated to QR decomposition

    \param[in,out] qr - QR structure to free
***************************************************************/
void qr_free(struct QR * qr)
{
    if (qr != NULL){
        free(qr->mat); qr->mat = NULL;
        qmarray_free(qr->Q); qr->Q = NULL;
        free(qr); qr = NULL;
    }
}

/***********************************************************//**
    Allocate memory for an array of QR structures
    
    \param[in] n - size of array

    \return qr - array of QR structure (each of which is set to NULL)
***************************************************************/
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


/***********************************************************//**
    Free memory allocated to array of QR structures

    \param[in,out] qr - array to free
    \param[in]     n  - number of QR structures

***************************************************************/
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


/***********************************************************//**
    Update right side component of supercore for FT product

    \param[in] a    - core 1
    \param[in] b    - core 2
    \param[in] prev - previous right side to update
    \param[in] z    - right multiplier
    \param[in] o    - approximation options
    
    \return lq - new right side component
***************************************************************/
struct QR * dmrg_super_rprod(struct Qmarray * a, struct Qmarray * b,
                             struct QR * prev, struct Qmarray * z,
                             struct OneApproxOpts * o)
{

    double * val = qmaqmat_integrate(prev->Q,z);
    struct Qmarray * temp = qmarray_kron_mat(z->nrows,val,a,b);
    //struct Qmarray * temp = qmam(nextleft,val,z->nrows);
    struct QR * lq = qr_reduced(temp,0,o);
    qmarray_free(temp); temp = NULL;
    free(val); val = NULL;
    return lq;
}

/***********************************************************//**
    Update right side component of supercore
    
    \param[in] dim  - core which we are considering
    \param[in] f    - specialized function to multiply core by matrix
    \param[in] args - arguments to f
    \param[in] prev - previous right side to update
    \param[in] z    - right multiplier
    \param[in] o    - approximation options
    
    \return lq - new right side component
***************************************************************/
struct QR *
dmrg_super_r(size_t dim,
             void (*f)(char,size_t,size_t,double *,struct Qmarray **,void *),
             void * args, struct QR * prev, struct Qmarray * z,struct OneApproxOpts *o)
{

    double * val = qmaqmat_integrate(prev->Q,z);

    struct Qmarray * temp = NULL;
    f('R',dim,z->nrows,val,&temp,args);

    struct QR * lq = qr_reduced(temp,0,o);
    qmarray_free(temp); temp = NULL;
    free(val); val = NULL;
    return lq;
}

/***********************************************************//**
    Update all right side components of supercore 

    \param[in] f    - specialized function to multiply core by matrix
    \param[in] args - arguments to f
    \param[in] z    - guess
    \param[in] opts - approximation options
    
    \return lq - new right side component
***************************************************************/
struct QR ** 
dmrg_super_r_all(
      void (*f)(char,size_t,size_t,double *,struct Qmarray **,void *),
      void * args, struct FunctionTrain * z, struct MultiApproxOpts * opts)
{
    struct OneApproxOpts * o = NULL;
    size_t dim = function_train_get_dim(z);
    struct QR ** right = NULL;
    if ( NULL == (right = malloc((dim-1)*sizeof(struct QR *)))){
        fprintf(stderr, "failed to allocate memory for dmrg all right QR decompositions.\n");
        exit(1);
    }
    struct Qmarray * temp = NULL;
    f('R',dim-1,1,NULL,&temp,args);
    o = multi_approx_opts_get_aopts(opts,dim-1);
    right[dim-2] = qr_reduced(temp,0,o);
    qmarray_free(temp);

    int ii;
    for (ii = dim-3; ii > -1; ii--){
        o = multi_approx_opts_get_aopts(opts,(size_t)ii+1);
        right[ii] = dmrg_super_r((size_t)ii+1,f,args,right[ii+1],z->cores[ii+2],o);
    }
    return right;
}


/***********************************************************//**
    Perform a left-right dmrg sweep 

    \param[in]     z       - initial guess
    \param[in]     f       - specialized function to multiply core by matrix
    \param[in]     args    - arguments to f
    \param[in,out] phil    - left multipliers
    \param[in]     psir    - right multiplies
    \param[in]     epsilon - splitting tolerance
    \param[in]     opts    - approximation options

    \return na - a new approximation
***************************************************************/
struct FunctionTrain * 
dmrg_sweep_lr(struct FunctionTrain * z, 
              void (*f)(char,size_t,size_t, double *, struct Qmarray **, void *),
              void * args, struct QR ** phil, struct QR ** psir, double epsilon,
              struct MultiApproxOpts * opts)
{
    double * RL = NULL;
    size_t dim = z->dim;
    struct FunctionTrain * na = function_train_alloc(dim);
    struct OneApproxOpts * o = NULL;
    na->ranks[0] = 1;
    na->ranks[na->dim] = 1;
    
    if (phil[0] == NULL){
        struct Qmarray * temp0 = NULL;
        f('L',0,1,NULL,&temp0,args);
        /* printf("temp0 size(%zu,%zu) \n",temp0->nrows,temp0->ncols); */
        o = multi_approx_opts_get_aopts(opts,0);
        phil[0] = qr_reduced(temp0,1,o);
        qmarray_free(temp0); temp0 = NULL;
    }
    /* exit(1); */
    
    size_t nrows = phil[0]->mr;
    size_t nmult = phil[0]->mc;
    size_t ncols = psir[0]->mc;

    RL = calloc_double(nrows * ncols);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,nrows,ncols,
                nmult,1.0,phil[0]->mat,nrows,psir[0]->mat,nmult,0.0,RL,nrows);

    double * u = NULL;
    double * vt = NULL;
    double * s = NULL;
    /* printf("Left-Right sweep\n"); */
    /* printf("(nrows,ncols)=(%zu,%zu), epsilon=%G\n",nrows,ncols,epsilon); */
    size_t rank = truncated_svd(nrows,ncols,nrows,RL,&u,&s,&vt,epsilon);
    /* printf("rank=%zu\n",rank); */
    na->ranks[1] = rank;
    na->cores[0] = qmam(phil[0]->Q,u,rank);
    

    size_t ii;
    for (ii = 1; ii < dim-1; ii++){
        /* printf("ii = %zu\n",ii); */
        double * newphi = calloc_double(rank * phil[ii-1]->mc);
        cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,rank,nmult,
                    nrows,1.0,u,nrows,phil[ii-1]->mat,nrows,0.0,newphi,rank);
        //struct Qmarray * temp = mqma(newphi,y->cores[ii],rank);
        //struct Qmarray * temp = qmarray_mat_kron(rank,newphi,a->cores[ii],b->cores[ii]);
        struct Qmarray * temp = NULL;
        f('L',ii,rank,newphi,&temp,args);

        qr_free(phil[ii]); phil[ii] = NULL;
        o = multi_approx_opts_get_aopts(opts,ii);
        phil[ii] = qr_reduced(temp,1,o);
        
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

        /* printf("(nrows,ncols)=(%zu,%zu), epsilon=%G\n",nrows,ncols,epsilon); */
        rank = truncated_svd(nrows,ncols,nrows,RL,&u,&s,&vt,epsilon);
        /* dprint(nrows,s); */
        /* printf("rank=%zu\n",rank); */
        na->ranks[ii+1] = rank;
        na->cores[ii] = qmam(phil[ii]->Q,u,rank);
    }
    
    /* exit(1); */
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
    Perform a right-left dmrg sweep as part of ft-product

    \param[in]     z       - initial guess
    \param[in]     f       - specialized function to multiply core by matrix
    \param[in]     args    - arguments to f
    \param[in,out] phil    - left multipliers
    \param[in]     psir    - right multiplies
    \param[in]     epsilon - splitting tolerance
    \param[in]     opts    - approximation options

    \return na - a new approximation
***************************************************************/
struct FunctionTrain * 
dmrg_sweep_rl(struct FunctionTrain * z, 
              void (*f)(char,size_t,size_t,double *,struct Qmarray **,void *),
              void * args, struct QR ** phil, struct QR ** psir, double epsilon,
              struct MultiApproxOpts * opts)
{
    double * RL = NULL;
    size_t dim = z->dim;
    struct FunctionTrain * na = function_train_alloc(dim);
    na->ranks[0] = 1;
    na->ranks[na->dim] = 1;
    struct OneApproxOpts * o = NULL;
    if (psir[dim-2] == NULL){
        struct Qmarray * temp0 = NULL;
        f('R',dim-1,1,NULL,&temp0,args);
        //qmarray_kron(a->cores[dim-1],b->cores[dim-1]);
        o = multi_approx_opts_get_aopts(opts,dim-1);
        psir[dim-2] = qr_reduced(temp0,0,o);
        qmarray_free(temp0); temp0 = NULL;
    }
    
    size_t nrows = phil[dim-2]->mr;
    size_t nmult = phil[dim-2]->mc;
    size_t ncols = psir[dim-2]->mc;

    RL = calloc_double(nrows * ncols);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,nrows,ncols,
                nmult,1.0,phil[dim-2]->mat,nrows,psir[dim-2]->mat,nmult,0.0,RL,nrows);

    double * u = NULL;
    double * vt = NULL;
    double * s = NULL;
    /* printf("Right-Left sweep\n"); */
    /* printf("(nrows,ncols)=(%zu,%zu), epsilon=%G\n",nrows,ncols,epsilon); */
    size_t rank = truncated_svd(nrows,ncols,nrows,RL,&u,&s,&vt,epsilon);
    /* printf("rank=%zu\n",rank); */
    na->ranks[dim-1] = rank;
    na->cores[dim-1] = mqma(vt,psir[dim-2]->Q,rank); 
    

    int ii;
    for (ii = dim-3; ii > -1; ii--){
        double * newpsi = calloc_double( psir[ii+1]->mr * rank);
        //
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,
                    psir[ii+1]->mr,rank, psir[ii+1]->mc,
                    1.0,psir[ii+1]->mat,psir[ii+1]->mr,vt,rank,
                    0.0,newpsi,psir[ii+1]->mr);

        struct Qmarray * temp = NULL;
        // qmarray_kron_mat(rank,newpsi,a->cores[ii+1],b->cores[ii+1]);
        f('R',(size_t)ii+1,rank,newpsi,&temp,args);

        qr_free(psir[ii]); psir[ii] = NULL;
        o = multi_approx_opts_get_aopts(opts,(size_t)ii+1);
        psir[ii] = qr_reduced(temp,0,o);
        
        free(RL); RL = NULL;
        free(newpsi); newpsi = NULL;
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

        /* printf("(nrows,ncols)=(%zu,%zu), epsilon=%G\n",nrows,ncols,epsilon); */
        rank = truncated_svd(nrows,ncols,nrows,RL,&u,&s,&vt,epsilon);
        /* printf("rank=%zu\n",rank); */
        na->ranks[ii+1] = rank;
        na->cores[ii+1] = mqma(vt,psir[ii]->Q,rank); 
    }
    
    size_t kk,jj;
    for (jj = 0; jj < rank; jj++){
        for (kk = 0; kk < nrows; kk++){
            u[jj*nrows+kk] = u[jj*nrows+kk]*s[jj];
        }
    }
    
    na->cores[0] = qmam(phil[0]->Q,u,rank);

    /* exit(1); */

    free(RL); RL = NULL;
    free(u); u = NULL;
    free(vt); vt = NULL;
    free(s); s = NULL;

    return na;
}

/***********************************************************//**
    Perform a left-right-left dmrg sweep for FT product

    \param[in]     z       - initial guess
    \param[in]     f       - specialized function to multiply core by matrix
    \param[in]     args    - arguments to f
    \param[in,out] phil    - left multipliers
    \param[in,out] psir    - right multiplies
    \param[in,out] epsilon - splitting tolerance
    \param[in]     opts    - approximation options

    \return na - a new approximation
***************************************************************/
struct FunctionTrain * 
dmrg_sweep_lrl(struct FunctionTrain * z,
            void (*f)(char,size_t,size_t,double *,struct Qmarray **,void *),
               void * args, struct QR ** phil, struct QR ** psir, double epsilon,
               struct MultiApproxOpts * opts)
{

    struct FunctionTrain * temp = dmrg_sweep_lr(z,f,args,phil,psir,epsilon,opts);
    struct FunctionTrain * na = dmrg_sweep_rl(temp,f,args,phil,psir,epsilon,opts);
    function_train_free(temp); temp = NULL;
    return na;
}

/***********************************************************//**
    Compute ALS+DMRG

    \param[in,out] z          - initial guess (destroyed);
    \param[in]     f          - specialized function to multiply core by matrix
    \param[in]     args       - arguments to f
    \param[in]     delta      - threshold to stop iterating
    \param[in]     max_sweeps - maximum number of left-right-left sweeps 
    \param[in]     epsilon    - SVD tolerance for rank determination
    \param[in]     verbose    - verbosity level 0 or >0
    \param[in]     opts       - approximation options

    \returns function train
***************************************************************/
struct FunctionTrain *
dmrg_approx(struct FunctionTrain * z,
            void (*f)(char,size_t,size_t,double *,struct Qmarray **,void *),
            void * args, double delta, size_t max_sweeps, 
            double epsilon, int verbose, struct MultiApproxOpts * opts)
{
    size_t dim = z->dim;
    struct FunctionTrain * na = function_train_orthor(z,opts);

    struct QR ** psir = dmrg_super_r_all(f,args,na,opts);
    struct QR ** phil = qr_array_alloc(dim-1);

    /* printf("starting ranks\n"); */
    /* iprint_sz(dim+1,function_train_get_ranks(na)); */

    size_t ii;
    double diff;
    for (ii = 0; ii < max_sweeps; ii++){
        if (verbose>0){
            printf("On Sweep (%zu/%zu) \n",ii+1,max_sweeps);
        }
        struct FunctionTrain * check = dmrg_sweep_lrl(na,f,args,phil,psir,epsilon,opts);
        diff = function_train_relnorm2diff(check,na);
        function_train_free(na); na = NULL;
        na = function_train_copy(check);
        function_train_free(check); check = NULL;
        
        if (verbose > 0){
            printf("\t The relative error between approximations is %G\n",diff);
        }
        if (diff < delta){
            break;
        }
    }

    qr_array_free(psir,dim-1);
    qr_array_free(phil,dim-1);
    return na;
}
