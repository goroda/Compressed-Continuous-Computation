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





/** \file quasimatrix.c
 * Provides algorithms for quasimatrix manipulation
 */

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "stringmanip.h"
#include "array.h"
#include "linalg.h"

#include "quasimatrix.h"

#define ZEROTHRESH 1e2*DBL_EPSILON

/** \struct Quasimatrix
 *  \brief Defines a vector-valued function
 *  \var Quasimatrix::n
 *  number of functions
 *  \var Quasimatrix::funcs
 *  array of functions
 */
struct Quasimatrix {
    size_t n;
    struct GenericFunction ** funcs;
};

/*********************************************************//**
    Allocate space for a quasimatrix

    \param[in] n - number of columns of quasimatrix

    \return quasimatrix
*************************************************************/
struct Quasimatrix * 
quasimatrix_alloc(size_t n){
    
    struct Quasimatrix * qm;
    if ( NULL == (qm = malloc(sizeof(struct Quasimatrix)))){
        fprintf(stderr, "failed to allocate memory for quasimatrix.\n");
        exit(1);
    }
    qm->n = n;
    if ( NULL == (qm->funcs = malloc(n * sizeof(struct GenericFunction *)))){
        fprintf(stderr, "failed to allocate memory for quasimatrix.\n");
        exit(1);
    }
    size_t ii;
    for (ii = 0; ii < n; ii++){
        qm->funcs[ii] = NULL;
    }
    return qm;
}

/*********************************************************//**
    Free functions
*************************************************************/
void quasimatrix_free_funcs(struct Quasimatrix * qm)
{
    assert (qm != NULL);
    if (qm->funcs != NULL){
        for (size_t ii = 0; ii < qm->n ;ii++){
            generic_function_free(qm->funcs[ii]); qm->funcs[ii] = NULL;
        }
        free(qm->funcs); qm->funcs = NULL;
    }
}

/*********************************************************//**
    Free memory allocated to quasimatrix

    \param[in,out] qm 
*************************************************************/
void quasimatrix_free(struct Quasimatrix * qm){
    
    if (qm != NULL){
        quasimatrix_free_funcs(qm);
        free(qm);qm=NULL;
    }
}

/*********************************************************//**
    Get size
*************************************************************/
size_t quasimatrix_get_size(const struct Quasimatrix * q)
{
    assert(q != NULL);
    return q->n;
}

/*********************************************************//**
    Set size
*************************************************************/
void quasimatrix_set_size(struct Quasimatrix * q,size_t n)
{
    assert(q != NULL);
    q->n = n;
}

/*********************************************************//**
    Get function
*************************************************************/
struct GenericFunction *
quasimatrix_get_func(const struct Quasimatrix * q, size_t ind)
{
    assert(q != NULL);
    assert(ind < q->n);
    return q->funcs[ind];
}

/*********************************************************//**
   set function by copying
*************************************************************/
void
quasimatrix_set_func(struct Quasimatrix * q, const struct GenericFunction * gf,
                     size_t ind)
{
    assert(q != NULL);
    assert(ind < q->n);
    generic_function_free(q->funcs[ind]);
    q->funcs[ind] = generic_function_copy(gf);
}

/*********************************************************//**
    Get all functions
*************************************************************/
struct GenericFunction **
quasimatrix_get_funcs(const struct Quasimatrix * q)
{
    assert(q != NULL);
    return q->funcs;
}

/*********************************************************//**
    Set all functions
*************************************************************/
void
quasimatrix_set_funcs(struct Quasimatrix * q, struct GenericFunction ** gf)
{
    assert(q != NULL);
    quasimatrix_free_funcs(q);
    q->funcs = gf;
}

/*********************************************************//**
    Get reference to all functions
*************************************************************/
void
quasimatrix_get_funcs_ref(const struct Quasimatrix * q,struct GenericFunction *** gf)
{
    assert(q != NULL);
    *gf = q->funcs;
}

/*********************************************************//**
    Create a quasimatrix by approximating 1d functions

    \param[in]     n     - number of columns of quasimatrix
    \param[in,out] fw    - wrapped functions
    \param[in]     fc    - function class of each column
    \param[in]     aopts - approximation options

    \return quasimatrix
    
    // same approximation arguments in each dimension
*************************************************************/
struct Quasimatrix * 
quasimatrix_approx1d(size_t n, struct Fwrap * fw,
                     enum function_class * fc,
                     void * aopts)

{
    struct Quasimatrix * qm = quasimatrix_alloc(n);
    size_t ii;
    for (ii = 0; ii < n; ii++){
        /* printf("ii=%zu\n",ii); */
        fwrap_set_which_eval(fw,ii);
        qm->funcs[ii] = generic_function_approximate1d(fc[ii],aopts,fw);
        /* printf("integral = %G\n",generic_function_integral(qm->funcs[ii])); */
    }
    return qm;
}

/*********************************************************
    Create a quasimatrix from a fiber_cuts array

    \param[in] n        - number of columns of quasimatrix
    \param[in] f        - function
    \param[in] fcut     - array of fiber cuts
    \param[in] fc       - function class of each column
    \param[in] sub_type - sub_type of each column
    \param[in] lb       - lower bound of inputs to functions
    \param[in] ub       - upper bound of inputs to functions
    \param[in] aopts    - approximation options

    \return quasimatrix
*************************************************************/
/* struct Quasimatrix *  */
/* quasimatrix_approx_from_fiber_cuts(size_t n,  */
/*                                    double (*f)(double,void *), */
/*                                    struct FiberCut ** fcut,  */
/*                                    enum function_class fc, */
/*                                    void * sub_type,double lb, */
/*                                    double ub, void * aopts) */
/* { */
/*     struct Quasimatrix * qm = quasimatrix_alloc(n); */
/*     size_t ii; */
/*     for (ii = 0; ii < n; ii++){ */
/*         qm->funcs[ii] = */
/*             generic_function_approximate1d(f, fcut[ii],  */
/*                                            fc, sub_type, */
/*                                            lb, ub, aopts); */
/*     } */
/*     return qm; */
/* } */

/*********************************************************//**
    Copy a quasimatrix

    \param[in] qm - quasimatrix to copy

    \return copied quasimatrix
*************************************************************/
struct Quasimatrix *
quasimatrix_copy(const struct Quasimatrix * qm)
{
    struct Quasimatrix * qmo = quasimatrix_alloc(qm->n);
    size_t ii;
    for (ii = 0; ii < qm->n; ii++){
        qmo->funcs[ii] = generic_function_copy(qm->funcs[ii]);
    }

    return qmo;
}

/*********************************************************//**
    Serialize a quasimatrix

    \param[in,out] ser       - stream to serialize to
    \param[in]     qm        - quasimatrix
    \param[in,out] totSizeIn - if NULL then serialize, 
                               if not NULL then return size;

    \return ptr - ser shifted by number of bytes
*************************************************************/
unsigned char * 
quasimatrix_serialize(unsigned char * ser,
                      const struct Quasimatrix * qm, 
                      size_t *totSizeIn)
{
    // n -> func -> func-> ... -> func
    if (totSizeIn != NULL){
        size_t ii;
        size_t totSize = sizeof(size_t);
        for (ii = 0; ii < qm->n; ii++){
            size_t size_temp = 0 ;
            serialize_generic_function(NULL,qm->funcs[ii],&size_temp);
            totSize += size_temp;
        
        }
        *totSizeIn = totSize;
        return ser;
    }
    
    unsigned char * ptr = ser;
    ptr = serialize_size_t(ptr, qm->n);
    for (size_t ii = 0; ii < qm->n; ii++){
        ptr = serialize_generic_function(ptr, qm->funcs[ii],NULL);
    }
    return ptr;
}

/*********************************************************//**
    Deserialize a quasimatrix

    \param[in]     ser - serialized quasimatrix
    \param[in,out] qm  - quasimatrix

    \return shifted ser after deserialization
*************************************************************/
unsigned char *
quasimatrix_deserialize(unsigned char * ser,
                        struct Quasimatrix ** qm)
{
    unsigned char * ptr = ser;

    size_t n;
    ptr = deserialize_size_t(ptr, &n);
    *qm = quasimatrix_alloc(n);

    size_t ii;
    for (ii = 0; ii < n; ii++){
        ptr = deserialize_generic_function(ptr,
                                           &((*qm)->funcs[ii]));
    }
    
    return ptr;
}

/*********************************************************//**
    Generate a quasimatrix with orthonormal columns

    \param[in] n    - number of columns
    \param[in] fc   - function class
    \param[in] opts - options


    \return quasimatrix with orthonormal columns
*************************************************************/
struct Quasimatrix *
quasimatrix_orth1d(size_t n, enum function_class fc,
                   void * opts)
{
    struct Quasimatrix * qm = quasimatrix_alloc(n);
    generic_function_array_orth(n,qm->funcs,fc,opts);
    return qm;
}

/*********************************************************//**
    Find the absolute maximum element of a 
    quasimatrix of 1d functions

    \param[in]     qm      - quasimatrix
    \param[in,out] absloc  - location (row) of maximum
    \param[in,out] absval  - value of maximum
    \param[in]     optargs - optimization arguments

    \return col - column of maximum element
*************************************************************/
size_t 
quasimatrix_absmax(struct Quasimatrix * qm, 
                   double * absloc, double * absval,
                   void * optargs)
{   
    size_t col = 0;
    size_t ii;
    double temp_loc;
    double temp_max;
    *absval = generic_function_absmax(qm->funcs[0],absloc,
                                      optargs);
    for (ii = 1; ii < qm->n; ii++){
        temp_max = generic_function_absmax(qm->funcs[ii],
                                           &temp_loc,optargs);
        if (temp_max > *absval){
            col = ii;
            *absval = temp_max;
            *absloc = temp_loc;
        }
    }
    return col;
}



///////////////////////////////////////////
// Linear Algebra
///////////////////////////////////////////

/*********************************************************//**
    Inner product of integral of quasimatrices
    returns sum_i=1^n int A->funcs[ii], B->funcs[ii] dx

    \param[in] A - quasimatrix
    \param[in] B - quasimatrix

    \return mag : inner product
*************************************************************/
double quasimatrix_inner(const struct Quasimatrix * A,
                         const struct Quasimatrix * B)
{
    assert (A != NULL);
    assert (B != NULL);
    assert(A->n == B->n);
    assert (A->funcs != NULL);
    assert (B->funcs != NULL);

    size_t ii;
    double int1;
    double mag = 0.0;
    //printf("here! %zu,%zu\n",A1->n,A2->n);
    for (ii = 0; ii < A->n; ii++){
        int1 = generic_function_inner(A->funcs[ii],B->funcs[ii]);
        mag += int1;
    }
    //printf("mag = %3.2f \n", mag);
    
    return mag;
}

/*********************************************************//**
    Quasimatrix - vector multiplication

    \param[in] Q - quasimatrix
    \param[in] v - array

    \return f -  generic function
*************************************************************/
struct GenericFunction *
qmv(const struct Quasimatrix * Q, const double * v)
{
    struct GenericFunction * f =
        generic_function_lin_comb(Q->n,Q->funcs,v);
    return f;
}

/*********************************************************//**
    Quasimatrix - matrix multiplication

    \param[in] Q - quasimatrix
    \param[in] R - matrix  (fortran order)
    \param[in] b - number of columns of R

    \return quasimatrix
*************************************************************/
struct Quasimatrix *
qmm(const struct Quasimatrix * Q, const double * R, size_t b)
{
    struct Quasimatrix * B = quasimatrix_alloc(b);
    size_t ii;
    for (ii = 0; ii < b; ii++){
        B->funcs[ii] =
            generic_function_lin_comb(Q->n,Q->funcs,
                                      R + ii*Q->n);
    }
    return B;
}

/*********************************************************//**
    Quasimatrix - transpose matrix multiplication

    \param[in]     Q - quasimatrix
    \param[in,out] R - matrix  (fortran order)
    \param[in]     b - number of rows of R

    \return quasimatrix
*************************************************************/
struct Quasimatrix *
qmmt(const struct Quasimatrix * Q, const double * R, size_t b)
{
    double * Rt = calloc_double(Q->n * b);
    size_t ii, jj;
    for (ii = 0; ii < b; ii++){
        for (jj = 0; jj < Q->n; jj++){
            Rt[ii * Q->n + jj] = R[jj * b + ii];
        }
    }
    struct Quasimatrix * B = qmm(Q,Rt, b);
    free(Rt);
    return B;
}

/*********************************************************//**
    Compute ax + by where x, y are quasimatrices and a,b are scalars

    \param[in] a - scale for x
    \param[in] x - quasimatrix
    \param[in] b - scale for y
    \param[in] y - quasimatrix

    \return quasimatrix result
*************************************************************/
struct Quasimatrix *
quasimatrix_daxpby(double a, const struct Quasimatrix * x,
                   double b, const struct Quasimatrix * y)
{
    struct Quasimatrix * qm = NULL;
    if ((x == NULL) && (y == NULL)){
        return qm;
    }
    else if ((x == NULL) && (y != NULL)){
        //printf("here y not null \n");
        qm = quasimatrix_alloc(y->n);
        free(qm->funcs); qm->funcs = NULL;
        qm->funcs = generic_function_array_daxpby(y->n, b, 1, y->funcs, 0.0,1,NULL);
    }
    else if ((x != NULL) && (y == NULL)){
        //printf("here x not null \n");
        qm = quasimatrix_alloc(x->n);
        free(qm->funcs); qm->funcs = NULL;
        qm->funcs = generic_function_array_daxpby(x->n, a, 1, x->funcs, 0.0,1,NULL);
        //printf("out of x not null \n");
    }
    else{
        //printf("here in qm daxpby\n");
        qm = quasimatrix_alloc(x->n);
        free(qm->funcs); qm->funcs = NULL;
        qm->funcs = generic_function_array_daxpby(x->n,a,1,x->funcs,b,1,y->funcs);
    }

    return qm;
}


/*********************************************************//**
    Compute the householder triangularization of a 
    quasimatrix. (Trefethan 2010) whose columns consist of
    one dimensional functions

    \param[in,out] A    - quasimatrix to triangularize
    \param[in,out] R    - upper triangluar matrix
    \param[in]     opts - options for approximation

    \return quasimatrix denoting the Q term
    
    \note
        For now I have only implemented this for polynomial 
        function class and legendre subtype
*************************************************************/
struct Quasimatrix *
quasimatrix_householder_simple(struct Quasimatrix * A,
                               double * R, void * opts)
{
    
    size_t ncols = A->n;

    // generate two quasimatrices needed for householder
    enum function_class fc = generic_function_get_fc(A->funcs[0]);
    struct Quasimatrix * Q = quasimatrix_orth1d(ncols,fc,opts);
    struct Quasimatrix * V = quasimatrix_alloc(ncols);

    int out = 0;
    out = quasimatrix_householder(A,Q,V,R);
    
    assert(out == 0);
    out = quasimatrix_qhouse(Q,V);
    assert(out == 0);

    quasimatrix_free(V);
    return  Q;
}


/*********************************************************//**
    Compute the householder triangularization of a quasimatrix. (Trefethan 2010)

    \param[in,out] A - quasimatrix to triangularize(destroyed)
    \param[in,out] E - quasimatrix with orthonormal columns
    \param[in,out] V - allocated space for a quasimatrix, 
                       stores reflectors in the end
    \param[in,out] R - allocated space upper triangular factor

    \return info (if != 0 then something is wrong)
*************************************************************/
int 
quasimatrix_householder(struct Quasimatrix * A,
                        struct Quasimatrix * E, 
                        struct Quasimatrix * V, double * R)
{
    size_t ii, jj;
    struct GenericFunction * e;
    struct GenericFunction * x;
    struct GenericFunction * v;
    struct GenericFunction * v2;
    struct GenericFunction * atemp;
    double rho, sigma, alpha;
    double temp1;
    for (ii = 0; ii < A->n; ii++){
        /* printf("ii = %zu\n",ii); */
        e = E->funcs[ii];    
        x = A->funcs[ii];
        rho = generic_function_norm(x);

        R[ii * A->n + ii] = rho;

        alpha = generic_function_inner(e,x);

        if (alpha >= ZEROTHRESH){
            generic_function_flip_sign(e);
        }

        /* printf("compute v, rho = %G\n",rho); */
        /* printf("integral of x = %G\n",generic_function_integral(x)); */
        /* printf("norm of e = %G\n",generic_function_norm(e)); */
        v = generic_function_daxpby(rho,e,-1.0,x);
        /* printf("got v\n"); */
        // skip step improve orthogonality
        // improve orthogonality

        sigma = generic_function_norm(v);

        if (fabs(sigma) <= ZEROTHRESH){
            V->funcs[ii] = generic_function_copy(e); 
            generic_function_free(v);
        }
        else {
            V->funcs[ii] = generic_function_daxpby(1.0/sigma, v, 0.0, NULL);
            generic_function_free(v);
        }
        v2 = V->funcs[ii];
        
        for (jj = ii+1; jj < A->n; jj++){
            /* printf("\t jj = %zu\n",jj); */
            temp1 = generic_function_inner(v2,A->funcs[jj]);
            if (fabs(temp1) < ZEROTHRESH){
                temp1 = 0.0;
            }

            atemp = generic_function_daxpby(1.0,
                                            A->funcs[jj],
                                            -2.0 * temp1, v2);
            R[jj * A->n + ii] = generic_function_inner(e,
                                                       atemp);

            generic_function_free(A->funcs[jj]);

            A->funcs[jj] =
                generic_function_daxpby(1.0, atemp,
                                        -R[jj * A->n + ii],e);
            generic_function_free(atemp);
        }
        
    }
    return 0;
}

/*********************************************************//**
    Compute the Q matrix from householder reflectors (Trefethan 2010)

    \param[in,out] Q - quasimatrix E obtained after
                       househodler_qm (destroyed 
    \param[in,out] V - Householder functions, 
                       obtained from quasimatrix_householder

    \return info (if != 0 then something is wrong)
*************************************************************/
int quasimatrix_qhouse(struct Quasimatrix * Q,
                       struct Quasimatrix * V)
{
    
    int info = 0;
    size_t ii, jj;
    
    
    
    struct GenericFunction * v;
    struct GenericFunction * q;
    double temp1;
    size_t counter = 0;
    ii = Q->n-1;
    while (counter < Q->n){
        v = V->funcs[ii];
        for (jj = ii; jj < Q->n; jj++){
            temp1 = generic_function_inner(v,Q->funcs[jj]);
            q = generic_function_daxpby(1.0, Q->funcs[jj],
                                        -2.0 * temp1, v);

            generic_function_free(Q->funcs[jj]);
            Q->funcs[jj] = generic_function_copy(q);
            generic_function_free(q);
        }
        ii = ii - 1;
        counter = counter + 1;
    }

    return info;
}

/*********************************************************//**
    Compute the LU decomposition of a quasimatrix of one 
    dimensional functions (Townsend 2015)

    \param[in]     A          - quasimatrix to decompose
    \param[in,out] L          - quasimatrix representing L factor
    \param[in,out] u          - allocated space for U factor
    \param[in,out] p          - pivots 
    \param[in]     approxargs - approximation arguments for each column
    \param[in]     optargs    - optimization arguments

    \return inno

    \note
        info = 
            - 0 converges
            - <0 low rank ( rank = A->n + info )
*************************************************************/
int quasimatrix_lu1d(struct Quasimatrix * A,
                     struct Quasimatrix * L, double * u,
                     double * p, void * approxargs,
                     void * optargs)
{
    int info = 0;
    
    size_t ii,kk;
    double val, amloc;
    struct GenericFunction * temp;
    
    for (kk = 0; kk < A->n; kk++){

        generic_function_absmax(A->funcs[kk], &amloc,optargs);
        p[kk] = amloc;

        val = generic_function_1d_eval(A->funcs[kk], amloc);
        
        if (fabs(val) < 2.0 * ZEROTHRESH){
            L->funcs[kk] = 
                generic_function_constant(1.0,POLYNOMIAL,approxargs);
        }
        else{
            L->funcs[kk] =
                generic_function_daxpby(1.0/val,A->funcs[kk],
                                        0.0, NULL);
        }
        for (ii = 0; ii < A->n; ii++){

            u[ii*A->n+kk] =
                generic_function_1d_eval(A->funcs[ii], amloc);
            temp = generic_function_daxpby(1.0, A->funcs[ii], 
                                           -u[ii*A->n+kk],
                                           L->funcs[kk]);
            generic_function_free(A->funcs[ii]);
            A->funcs[ii] = generic_function_daxpby(1.0,
                                                   temp,
                                                   0.0, NULL);
            generic_function_free(temp); temp = NULL;
        }
    }
    
    // check if matrix is full rank
    for (kk = 0; kk < A->n; kk++){
        if (fabs(u[kk*A->n + kk]) < ZEROTHRESH){
            info --;
        }
    }
    
    return info;
}

/*********************************************************//**
    Perform a greedy maximum volume procedure to find the 
    maximum volume submatrix of quasimatrix 

    \param[in]     A          - quasimatrix 
    \param[in,out] Asinv      - submatrix inv
    \param[in,out] p          - pivots 
    \param[in]     approxargs - approximation arguments for each column
    \param[in]     optargs    - optimization arguments


    \return info 
    
    \note
        info =
          -  0 converges
          -  < 0 no invertible submatrix
          -  >0 rank may be too high
        naive implementation without rank 1 updates
*************************************************************/
int quasimatrix_maxvol1d(struct Quasimatrix * A,
                         double * Asinv, double * p,
                         void * approxargs, void * optargs)
{
    int info = 0;
    double delta = 0.01;
    size_t r = A->n;

    struct Quasimatrix * L = quasimatrix_alloc(r);
    double * U = calloc_double(r * r);
    
    double *eye = calloc_double(r * r);
    size_t ii;
    for (ii = 0; ii < r; ii++){
        eye[ii*r + ii] = 1.0;
    }
    struct Quasimatrix * Acopy = qmm(A,eye,r);

    info = quasimatrix_lu1d(Acopy,L,U,p,approxargs,optargs);
    //printf("pivot immediate \n");
    //dprint(A->n, p);

    if (info != 0){
        printf("Couldn't find an invertible submatrix for maxvol %d\n",info);
        //printf("Couldn't Error from quasimatrix_lu1d \n");
        quasimatrix_free(L);
        quasimatrix_free(Acopy);
        free(U);
        free(eye);
        return info;
    }
    
    size_t jj;
    for (ii = 0; ii < r; ii++){
        for (jj = 0; jj < r; jj++){
            Asinv[jj * r + ii] = 
                    generic_function_1d_eval(A->funcs[jj],p[ii]);
        }
    }

    int * ipiv2 = calloc(r, sizeof(int));
    size_t lwork = r*r;
    double * work = calloc(lwork, sizeof(double));
    int info2;
    
    dgetrf_((int*)&r,(int*)&r, Asinv, (int*)&r, ipiv2, &info2); 
    dgetri_((int*)&r, Asinv, (int*)&r, ipiv2, work, (int*)&lwork, &info2); //invert
        
    struct Quasimatrix * B = qmm(A,Asinv,r);
    size_t maxcol;
    double maxloc, maxval;
    maxcol = quasimatrix_absmax(B,&maxloc,&maxval,NULL);

    while (maxval > (1.0 + delta)){
        //printf("maxval = %3.4f\n", maxval);
        p[maxcol] = maxloc;
        for (ii = 0; ii < r; ii++){
            for (jj = 0; jj < r; jj++){
                Asinv[jj * r + ii] = 
                    generic_function_1d_eval(A->funcs[jj],p[ii]);
            }
        }

        dgetrf_((int*)&r,(int*)&r, Asinv, (int*)&r, ipiv2, &info2);
        //invert
        dgetri_((int*)&r, Asinv, (int*)&r, ipiv2, work, (int*)&lwork, &info2); 
        quasimatrix_free(B);
        B = qmm(A,Asinv,r);
        maxcol = quasimatrix_absmax(B,&maxloc,&maxval,NULL);
    }
    //printf("maxval = %3.4f\n", maxval);

    free(work);
    free(ipiv2);
    quasimatrix_free(B);
    quasimatrix_free(L);
    quasimatrix_free(Acopy);
    free(U);
    free(eye);
    return info;
}

/*********************************************************//**
    Compute rank of a quasimatrix

    \param[in] A    - quasimatrix 
    \param[in] opts - options for QR decomposition

    \return rank 
*************************************************************/
size_t quasimatrix_rank(const struct Quasimatrix * A, void * opts)
{
    size_t rank = A->n;
    size_t ncols = A->n;
    /* enum poly_type ptype = LEGENDRE; */
    // generate two quasimatrices needed for householder
    //
    /* double lb = generic_function_get_lower_bound(A->funcs[0]); */
    /* double ub = generic_function_get_upper_bound(A->funcs[0]); */
    
    struct Quasimatrix * Q = quasimatrix_orth1d(ncols,POLYNOMIAL, opts);

    struct Quasimatrix * V = quasimatrix_alloc(ncols);
    double * R = calloc_double(rank * rank);

    double * eye = calloc_double(rank * rank);
    size_t ii;
    for (ii = 0; ii < rank; ii++){
        eye[ii * rank + ii] = 1.0;
    }
    struct Quasimatrix * qm = quasimatrix_copy(A);
    int out = 0;
    out = quasimatrix_householder(qm,Q,V,R);

    assert(out == 0);
    /* dprint2d_col(qm->n,qm->n,R); */
    for (ii = 0; ii < qm->n; ii++){
        if (fabs(R[ii * qm->n + ii]) < 10.0*ZEROTHRESH){
            rank = rank - 1;
        }
    }
    
    quasimatrix_free(Q);
    quasimatrix_free(V);
    quasimatrix_free(qm);
    free(eye);
    free(R);
    return rank;
}

/*********************************************************//**
    Compute norm of a quasimatrix

    \param[in] A - quasimatrix

    \return mag - norm of the quasimatrix
*************************************************************/
double quasimatrix_norm(const struct Quasimatrix * A)
{
    double mag = sqrt(quasimatrix_inner(A,A));
    return mag;
}

//////////////////////////////////////////////
// Local functions
/* static void quasimatrix_flip_sign(struct Quasimatrix * x) */
/* { */
/*     size_t ii; */
/*     for (ii = 0; ii < x->n; ii++){ */
/*         generic_function_flip_sign(x->funcs[ii]); */
/*     } */
/* } */

void quasimatrix_print(const struct Quasimatrix * qm,FILE *fp,
                       size_t prec, void * args)
{
    (void)(fp);
    printf("Quasimatrix consists of %zu columns\n",qm->n);
    printf("=========================================\n");
    size_t ii;
    for (ii = 0; ii < qm->n; ii++){
        print_generic_function(qm->funcs[ii],prec,args, fp);
        printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    }
}
