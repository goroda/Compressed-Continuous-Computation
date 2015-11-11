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
 * Provides routines for performing continuous linear algebra and working with function
 * trains
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>

#include "lib_funcs.h"
#include "algs.h"
#include "array.h"
#include "linalg.h"

//#define ZEROTHRESH 1e0*DBL_EPSILON
//#define ZEROTHRESH  1e-14
#define ZEROTHRESH 1e-20

#ifndef VQMALU
    #define VQMALU 0
#endif

#ifndef VQMAMAXVOL
    #define VQMAMAXVOL 0
#endif

#ifndef VQMAHOUSEHOLDER
    #define VQMAHOUSEHOLDER 0
#endif

#ifndef VPREPCORE
    #define VPREPCORE 0
#endif

#ifndef VFTCROSS
    #define VFTCROSS 0
#endif


/***********************************************************//**
    Quasimatrix - vector multiplication

    \param Q [in] - quasimatrix
    \param v [in] - array

    \return f -  generic function
***************************************************************/
struct GenericFunction *
qmv(struct Quasimatrix * Q, double * v)
{
    struct GenericFunction * f = generic_function_lin_comb(Q->n, Q->funcs,v);
    return f;
}

/***********************************************************//**
    Quasimatrix - matrix multiplication

    \param Q [in] - quasimatrix
    \param R [in] - matrix  (fortran order)
    \param b [in] - number of columns of R

    \return B - quasimatrix
***************************************************************/
struct Quasimatrix *
qmm(struct Quasimatrix * Q, double * R, size_t b)
{
    struct Quasimatrix * B = quasimatrix_alloc(b);
    size_t ii;
    for (ii = 0; ii < b; ii++){
        B->funcs[ii] = generic_function_lin_comb(Q->n, Q->funcs, R + ii*Q->n);
    }
    return B;
}

/***********************************************************//**
    Quasimatrix - transpose matrix multiplication

    \param Q [in] - quasimatrix
    \param R [in] - matrix  (fortran order)
    \param b [in] - number of rows of R

    \return B - quasimatrix
***************************************************************/
struct Quasimatrix *
qmmt(struct Quasimatrix * Q, double * R, size_t b)
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

/***********************************************************//**
    Compute ax + by where x, y are quasimatrices and a,b are scalars

    \param a [in] - scale for x
    \param x [in] - quasimatrix
    \param b [in] - scale for y
    \param y [in] - quasimatrix

    \return qm - quasimatrix result
***************************************************************/
struct Quasimatrix * quasimatrix_daxpby(double a, struct Quasimatrix * x,
                                         double b, struct Quasimatrix * y)
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

// local function
void quasimatrix_flip_sign(struct Quasimatrix * x)
{
    size_t ii;
    for (ii = 0; ii < x->n ;ii++){
        generic_function_flip_sign(x->funcs[ii]);
    }
}

/***********************************************************//**
    Compute the householder triangularization of a 
    quasimatrix. (Trefethan 2010) whose columns consist of
    one dimensional functions

    \param A [inout] - quasimatrix to triangularize
    \param R [inout] - upper triangluar matrix

    \return E - quasimatrix denoting the Q term
    
    \note
        For now I have only implemented this for polynomial 
        function class and legendre subtype
***************************************************************/
struct Quasimatrix *
quasimatrix_householder_simple(struct Quasimatrix * A, double * R)
{
    
    double lb = generic_function_get_lower_bound(A->funcs[0]);
    double ub = generic_function_get_upper_bound(A->funcs[0]);

    size_t ncols = A->n;
    enum poly_type ptype = LEGENDRE;
    // generate two quasimatrices needed for householder
    struct Quasimatrix * Q = quasimatrix_orth1d(POLYNOMIAL, 
                    (void *) (&ptype), ncols, lb, ub); 
    struct Quasimatrix * V = quasimatrix_alloc(ncols);

    int out = 0;
    out = quasimatrix_householder(A,Q,V,R);

    /*
    printf("Vqm[0] = \n");
    print_generic_function(V->funcs[0],0,NULL);
    printf("Vqm[1] = \n");
    print_generic_function(V->funcs[1],0,NULL);
    if (V->n > 2){
        printf("Vqm[2] = \n");
        print_generic_function(V->funcs[2],0,NULL);
    }
    */

    assert(out == 0);
    out = quasimatrix_qhouse(Q,V);
    assert(out == 0);

    quasimatrix_free(V);
    return  Q;
}

/***********************************************************//**
    Compute the householder triangularization of a quasimatrix. (Trefethan 2010)

    \param A [inout] - quasimatrix to triangularize (destroyed 
    \param E [inout] - quasimatrix with orthonormal columns
    \param V [inout] - allocated space for a quasimatrix, stores reflectors in the end
    \param R [inout] - allocated space upper triangular factor

    \return info (if != 0 then something is wrong)
***************************************************************/
int 
quasimatrix_householder(struct Quasimatrix * A, struct Quasimatrix * E, 
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
        printf(" On iter (%zu / %zu) \n", ii,A->n);
        e = E->funcs[ii];    
        x = A->funcs[ii];
        rho = generic_function_norm(x);
        printf(" \t ... rho = %3.15G\n",rho);

        /*
        printf("X+==================================\n");
        print_generic_function(x,3,NULL);
        printf("E+==================================\n");
        print_generic_function(e,3,NULL);
        printf("K+==================================\n");
        */
        //printf(" \t .... norm of e = %3.5f\n", generic_function_norm(e));
        R[ii * A->n + ii] = rho;
        //printf("pre alpha------------------------------------\n");
        alpha = generic_function_inner(e,x);
        printf(" \t ... alpha = %3.15G\n",alpha);
        if (alpha >= ZEROTHRESH){
            //printf("flip sign\n");
            generic_function_flip_sign(e);
        }

        //printf("epostalpha = \n");
        //print_generic_function(e,3,NULL);

        //printf("apostalpha = \n");
        //print_generic_function(x,3,NULL);
        v = generic_function_daxpby(rho,e,-1.0,x);
    
        // skip step improve orthogonality
        // improve orthogonality

        //print_generic_function(v,3,NULL);
        sigma = generic_function_norm(v);
        printf(" \t ... sigma = %3.15G\n",sigma);

        //printf("v2pre = \n");
        //print_generic_function(v,3,NULL);
        if (fabs(sigma) <= ZEROTHRESH){
            V->funcs[ii] = generic_function_copy(e); 
            generic_function_free(v);
        }
        else {
            V->funcs[ii] = generic_function_daxpby(1.0/sigma, v, 0.0, NULL);
            generic_function_free(v);
        }
        v2 = V->funcs[ii];
        
        //printf("v2 = \n");
        //print_generic_function(v2,3,NULL);

        for (jj = ii+1; jj < A->n; jj++){
            printf("\t On sub-iter %zu\n", jj);
            //printf("Aqm = \n");
            //print_generic_function(A->funcs[jj],3,NULL);
            temp1 = generic_function_inner(v2,A->funcs[jj]);
            //printf("\t\t temp1 pre = %3.15G DBL_EPS=%G\n", temp1,DBL_EPSILON);
            if (fabs(temp1) < ZEROTHRESH){
                temp1 = 0.0;
            }
            printf("\t\t temp1= %3.15G\n", temp1);

            atemp = generic_function_daxpby(1.0, A->funcs[jj], -2.0 * temp1, v2);
            R[jj * A->n + ii] = generic_function_inner(e, atemp);
            printf("\t\t e^T atemp= %3.15G\n", R[jj*A->n+ii]);
            generic_function_free(A->funcs[jj]);

            A->funcs[jj] = generic_function_daxpby(1.0, atemp, -R[jj * A->n + ii], e);
            generic_function_free(atemp);
        }
        
    }
    //printf("R = \n");
    //dprint2d_col(A->n,A->n,R);
    return 0;
}

/***********************************************************//**
    Compute the Q matrix from householder reflectors (Trefethan 2010)

    \param Q [inout] - quasimatrix E obtained afterm househodler_qm (destroyed 
    \param V [inout] - Householder functions, obtained from quasimatrix_householder

    \return info (if != 0 then something is wrong)
***************************************************************/
int quasimatrix_qhouse(struct Quasimatrix * Q, struct Quasimatrix * V)
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
            q = generic_function_daxpby(1.0, Q->funcs[jj], -2.0 * temp1, v);

            generic_function_free(Q->funcs[jj]);
            Q->funcs[jj] = generic_function_copy(q);
            generic_function_free(q);
        }
        ii = ii - 1;
        counter = counter + 1;
    }

    return info;
}

/***********************************************************//**
    Compute the LU decomposition of a quasimatrix of one dimensional functions (Townsend 2015)

    \param A [in] - quasimatrix to decompose
    \param L [inout] - quasimatrix representing L factor
    \param u [inout] - allocated space for U factor
    \param p [inout] - pivots 

    \return info

    \note
        info = 
            - 0 converges
            - <0 low rank ( rank = A->n + info )
***************************************************************/
int quasimatrix_lu1d(struct Quasimatrix * A, struct Quasimatrix * L, double * u,
            double * p)
{
    int info = 0;
    
    size_t ii,kk;
    double val, amloc;
    struct GenericFunction * temp;
    double lb = generic_function_get_lower_bound(A->funcs[0]);
    double ub = generic_function_get_upper_bound(A->funcs[0]);
    for (kk = 0; kk < A->n; kk++){
        //printf("kk=%zu\n",kk);
        //printf("getting max of function:\n");
        //print_generic_function(A->funcs[kk],0,NULL);
        generic_function_absmax(A->funcs[kk], &amloc);
        p[kk] = amloc;
        //printf("location of max =%3.5f\n",amloc);

        val = generic_function_1d_eval(A->funcs[kk], amloc);
        
        if (fabs(val) < 2.0 * ZEROTHRESH){
            enum poly_type ptype = LEGENDRE;
            L->funcs[kk] = 
                generic_function_constant(1.0,POLYNOMIAL,&ptype,lb,ub,NULL);
        }
        else{
            L->funcs[kk] =
                generic_function_daxpby(1.0/val, A->funcs[kk], 0.0, NULL);
        }
        for (ii = 0; ii < A->n; ii++){
            //printf("evaluation = %3.2f \n", 
            //        generic_function_1d_eval(A->funcs[ii], amloc));
            u[ii*A->n+kk] = generic_function_1d_eval(A->funcs[ii], amloc);
            temp = generic_function_daxpby(1.0, A->funcs[ii], 
                                            -u[ii*A->n+kk], L->funcs[kk]);
            generic_function_free(A->funcs[ii]);
            A->funcs[ii] = generic_function_daxpby(1.0, temp, 0.0, NULL);
            generic_function_free(temp);
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

/***********************************************************//**
    Perform a greedy maximum volume procedure to find the maximum volume submatrix of quasimatrix 

    \param A [in] - quasimatrix 
    \param Asinv [inout] - submatrix inv
    \param p [inout] - pivots 

    \return info 
    
    \note
        info =
          -  0 converges
          -  < 0 no invertible submatrix
          -  >0 rank may be too high
        naive implementation without rank 1 updates
***************************************************************/
int quasimatrix_maxvol1d(struct Quasimatrix * A, double * Asinv, double * p)
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

    info =  quasimatrix_lu1d(Acopy,L,U,p);
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
    
    dgetrf_(&r,&r, Asinv, &r, ipiv2, &info2); 
    dgetri_(&r, Asinv, &r, ipiv2, work, &lwork, &info2); //invert
        
    struct Quasimatrix * B = qmm(A,Asinv,r);
    size_t maxcol;
    double maxloc, maxval;
    maxcol = quasimatrix_absmax(B,&maxloc,&maxval);

    while (maxval > (1.0 + delta)){
        //printf("maxval = %3.4f\n", maxval);
        p[maxcol] = maxloc;
        for (ii = 0; ii < r; ii++){
            for (jj = 0; jj < r; jj++){
                Asinv[jj * r + ii] = 
                    generic_function_1d_eval(A->funcs[jj],p[ii]);
            }
        }

        dgetrf_(&r,&r, Asinv, &r, ipiv2, &info2); 
        dgetri_(&r, Asinv, &r, ipiv2, work, &lwork, &info2); //invert
        quasimatrix_free(B);
        B = qmm(A,Asinv,r);
        maxcol = quasimatrix_absmax(B,&maxloc,&maxval);
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

/***********************************************************//**
    Check the convergence of cross2d

    \param s1 [in] - skeleton decomp 1
    \param s2 [in] - skeleton decomp 2

    \return dist - relative distance between the two approximations
                \f$ ||1 - 2||^2/ ||1||^2 \f$
***************************************************************/
double check_cross_2d_convergence(struct SkeletonDecomp * s1,
                                  struct SkeletonDecomp * s2)
{
    double * eye1 = calloc_double(s1->r * s1->r);
    double * eye2 = calloc_double(s2->r * s2->r);
    
    size_t ii;
    for (ii = 0; ii < s1->r; ii++){
        eye1[ii * s1->r + ii] = 1.0;
    }
    for (ii = 0; ii < s2->r; ii++){
        eye2[ii * s2->r + ii] = 1.0;
    }

    struct Quasimatrix * t1 = qmm(s1->xqm,eye1,s1->r);
    struct Quasimatrix * t2 = qmm(s2->xqm,eye2,s2->r);

    double den = skeleton_decomp_inner_help(t1,s1->yqm,t1,s1->yqm);
    //printf("den = %3.5f\n",den);
    double num = skeleton_decomp_inner_help(t2,s2->yqm,t2,s2->yqm) + den - 
                    2.0 * skeleton_decomp_inner_help(t1,s1->yqm,t2,s2->yqm);
    
    double dist = num/den;

    free(eye1);
    free(eye2);
    quasimatrix_free(t1);
    quasimatrix_free(t2);
    //double dist = den;
    return dist;
}

/***********************************************************//**
    Cross approximation of a two dimensional function f(x,y)

    \param f [in] - function
    \param args [in] - function arguments
    \param bounds (IN): bounds on input space
    \param skd_init [inout] - initial skeleton decomposition, changed in func
    \param pivx [inout] - x values for skeleton
    \param pivy [inout] - y values for skeleton
    \param cargs [in] - algorithm parameters

    \return skd - skeleton decomposition
***************************************************************/
struct SkeletonDecomp *
cross_approx_2d(double (*f)(double, double, void *), void * args, 
                struct BoundingBox * bounds, struct SkeletonDecomp ** skd_init,
                double * pivx, double * pivy, struct Cross2dargs * cargs)
{
    int info;
    struct SkeletonDecomp * skd = skeleton_decomp_alloc(cargs->r);
    
    size_t r = cargs->r;
    double * T = calloc_double(r * r);
    double * eye = calloc_double(r * r);
    struct Quasimatrix * Q;
    struct Quasimatrix * tempqm;
    struct FiberCut ** fx;  
    struct FiberCut ** fy;

    size_t ii;
    for (ii = 0; ii < r; ii++){
        eye[ii*r+ii] = 1.0;
        skd->skeleton[ii*r+ii] = 1.0;
    }

    int done = 0;
    size_t iter = 0;
    double reldist;
    while (done == 0){
        if (cargs->verbose > 0)
            printf("cross iter=%zu \n",iter);

        // functions of y
        quasimatrix_free(skd->yqm);
        fy = fiber_cut_2darray(f,args,1, r, pivx);
        tempqm = quasimatrix_approx_from_fiber_cuts(
                r, fiber_cut_eval2d, fy, cargs->fclass[1], cargs->sub_type[1],
                bounds->lb[1],bounds->ub[1], cargs->approx_args[1]);
        fiber_cut_array_free(r, fy);

        Q = quasimatrix_householder_simple(tempqm,T);
        //print_quasimatrix(Q,1,NULL);

        info = quasimatrix_maxvol1d(Q,T,pivy);
        //printf("New y pivot = ");
        //dprint(r,pivy);

        skd->yqm = qmm(Q,T,r); 
        quasimatrix_free(tempqm);
        quasimatrix_free(Q);

        // functions of x
        quasimatrix_free(skd->xqm);
        fx = fiber_cut_2darray(f,args,0, r, pivy);
        skd->xqm = quasimatrix_approx_from_fiber_cuts(
                r, fiber_cut_eval2d, fx, cargs->fclass[0], cargs->sub_type[0],
                bounds->lb[0],bounds->ub[0], cargs->approx_args[0]);
        fiber_cut_array_free(r, fx);
        
        // check convergence here
        reldist = check_cross_2d_convergence(skd,*skd_init);

        if (cargs->verbose > 0)
            printf("reldist = %3.5f\n",reldist);
        if (reldist < cargs->delta) {
        //if (iter == 2) {
            done = 1;
        }
        else{     
            skeleton_decomp_free(*skd_init);
            *skd_init = skeleton_decomp_copy(skd);
                
            Q = quasimatrix_householder_simple(skd->xqm,T);
            quasimatrix_free(skd->xqm);
            skd->xqm = qmm(Q,eye,skd->r);
            //print_quasimatrix(skd->xqm,0,NULL);
            info = quasimatrix_maxvol1d(Q,T,pivx);

            //printf("New x pivot = ");
            //dprint(r,pivx);
            if ( info < 0){
                printf("Warning: rank may be too high");
                //fprintf(stderr, "maxvol failed in cross approximation.\n");
                //exit(1);
            }
            quasimatrix_free(Q);
            
        }
        /*
        printf("pivot (x;y) out = \n");                                             
        dprint(2,pivx);                                                             
        dprint(2,pivy);                                                             
        printf("i am out!\n");   
        */

        iter++;
    }
    
    free(T);
    free(eye);
    return skd;
}

////////////////////////////////////////////////////////////////////////////.

/***********************************************************//**
    Compute rank of a quasimatrix

    \param A [in] - quasimatrix 

    \return rank - rank of the quasimatrix
***************************************************************/
size_t quasimatrix_rank(struct Quasimatrix * A){
    size_t rank = A->n;

    size_t ncols = A->n;
    enum poly_type ptype = LEGENDRE;
    // generate two quasimatrices needed for householder
    //
    double lb = generic_function_get_lower_bound(A->funcs[0]);
    double ub = generic_function_get_upper_bound(A->funcs[0]);
    
    struct Quasimatrix * Q = quasimatrix_orth1d(POLYNOMIAL, 
                    (void *) (&ptype), ncols, lb, ub); 
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
    
    //printf("R=\n");
    //dprint2d_col(Q->n,Q->n,R);
    for (ii = 0; ii < qm->n; ii++){
        //printf("R[ii*Qm->n+ii]=%G for ii=%zu\n",R[ii*qm->n+ii],ii);
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

/***********************************************************//**
    Inner product between two functions represented in skeleton format by their quasimatrices

    \param A1 [in] - x quasimatrix  of function 1
    \param B1 [in] - y quasimatrix  of function 1
    \param A2 [in] - x quasimatrix  of function 2
    \param B2 [in] - y quasimatrix  of function 2

    \return mag : \f$ int (A1B1^T)(A2B2^T) dx dy \f$
***************************************************************/
double skeleton_decomp_inner_help(struct Quasimatrix * A1, struct Quasimatrix * B1,
            struct Quasimatrix * A2, struct Quasimatrix * B2)
{
    size_t ii, jj;
    double int1, int2;
    double mag = 0.0;
    //printf("here! %zu,%zu\n",A1->n,A2->n);
    for (ii = 0; ii < A1->n; ii++){
        for (jj = 0; jj < A2->n; jj++){
            int1 = generic_function_inner(A1->funcs[ii],A2->funcs[jj]);
            //printf("int1=%3.2f\n",int1);
            int2 = generic_function_inner(B1->funcs[ii],B2->funcs[jj]);
            //printf("int2=%3.2f\n",int2);
            mag += int1 * int2;
            //printf("(int1,int2) = (%3.5f,%3.5f)\n",int1,int2);
        }
    }
    //printf("mag = %3.2f \n", mag);
    
    return mag;
}


/***********************************************************//**
    Inner product of integral of quasimatrices
    returns sum_i=1^n int A->funcs[ii], B->funcs[ii] dx

    \param A [in] - quasimatrix
    \param B [in] - quasimatrix

    \return mag : inner product
***************************************************************/
double quasimatrix_inner(struct Quasimatrix * A, struct Quasimatrix * B)
{
    assert(A->n == B->n);

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

/***********************************************************//**
    Compute norm of a quasimatrix

    \param A [in] - quasimatrix

    \return mag - norm of the quasimatrix
***************************************************************/
double quasimatrix_norm(struct Quasimatrix * A)
{
    double mag = sqrt(quasimatrix_inner(A,A));
    return mag;
}


//////////////////////////////////////////////////////////////////

struct GenericFunction **
qmav_base(struct Qmarray * Q, double *v)
{
    struct GenericFunction ** fout = NULL;
    if ( NULL == 
        (fout = malloc(Q->nrows*sizeof(struct GenericFunction *)))){
        fprintf(stderr, "failed to allocate memory in qmav.\n");
        exit(1);
    }

    struct GenericFunction ** f = NULL;
    if ( NULL == (f = malloc(Q->ncols*sizeof(struct GenericFunction *)))){
        fprintf(stderr, "failed to allocate memory in qmav.\n");
        exit(1);
    }
    size_t ii, jj;
    for (ii = 0; ii < Q->nrows; ii++){
        for (jj = 0; jj < Q->ncols; jj++){
            f[jj] = Q->funcs[jj*Q->nrows+ii];
        }
        fout[ii] =  generic_function_lin_comb(Q->ncols,f,v);
    }

    return fout;
}

/***********************************************************//**
    Quasimatrix array - vector multiplication

    \param Q [in] - qmarray
    \param v [in] - array

    \return q - quasimatrix
***************************************************************/
struct Quasimatrix *
qmav(struct Qmarray * Q, double * v)
{
    struct Quasimatrix * q = quasimatrix_alloc(Q->nrows);
    free(q->funcs);
    q->funcs = qmav_base(Q,v);
    return q;
}

/***********************************************************//**
    Quasimatrix array - matrix multiplication

    \param Q [in] - quasimatrix array
    \param R [in] - matrix  (fortran order)
    \param b [in] - number of columns of R

    \return B - qmarray
***************************************************************/
struct Qmarray *
qmam(struct Qmarray * Q, double * R, size_t b)
{
    size_t nrows = Q->nrows;
    struct Qmarray * B = qmarray_alloc(nrows,b);
    size_t ii,jj;
    for (jj = 0; jj < nrows; jj++){
        for (ii = 0; ii < b; ii++){
            B->funcs[ii*nrows+jj] = // Q[jj,:]B[:,ii]
                generic_function_lin_comb2(Q->ncols, Q->nrows, Q->funcs + jj, 
                                            1, R + ii*Q->ncols);
        }
    }
    return B;
}

/***********************************************************//**
    Transpose Quasimatrix array - matrix multiplication

    \param Q [in] - quasimatrix array
    \param R [in] - matrix  (fortran order)
    \param b [in] - number of columns of R

    \return B - qmarray
***************************************************************/
struct Qmarray *
qmatm(struct Qmarray * Q, double * R, size_t b)
{
    struct Qmarray * B = qmarray_alloc(Q->ncols,b);
    size_t ii,jj;
    for (jj = 0; jj < B->nrows; jj++){
        for (ii = 0; ii < b; ii++){
            B->funcs[ii*B->nrows+jj] = // Q[:,jj]^T B[:,ii]
                generic_function_lin_comb2(Q->nrows, 1, 
                                            Q->funcs + jj * Q->nrows, 
                                            1, R + ii*Q->nrows);
        }
    }
    return B;
}


/***********************************************************//**
    Matrix - Quasimatrix array multiplication

    \param R [in] - matrix  (fortran order)
    \param Q [in] - quasimatrix array
    \param b [in] - number of rows of R

    \return B - qmarray (b x Q->ncols)
***************************************************************/
struct Qmarray *
mqma(double * R, struct Qmarray * Q, size_t b)
{
    struct Qmarray * B = qmarray_alloc(b, Q->ncols);
    size_t ii,jj;
    for (jj = 0; jj < Q->ncols; jj++){
        for (ii = 0; ii < b; ii++){
            B->funcs[jj*b+ii] = // R[ii,:]Q[:,jj]
                generic_function_lin_comb2(Q->nrows, 1, Q->funcs + jj*Q->nrows, 
                                            b, R + ii);
        }
    }
    return B;
}

/***********************************************************//**
    Qmarray - Qmarray mutliplication

    \param a [in] - qmarray 1
    \param b [in] - qmarray 2
 
    \return c - qmarray 
***************************************************************/
struct Qmarray *
qmaqma(struct Qmarray * a, struct Qmarray * b)
{
    struct Qmarray * c = qmarray_alloc(a->nrows, b->ncols);
    size_t ii,jj;
    for (jj = 0; jj < b->ncols; jj++){
        for (ii = 0; ii < a->nrows; ii++){
            // a[ii,:]b[:,jj]
            c->funcs[jj*c->nrows+ii] =  generic_function_sum_prod(a->ncols, a->nrows, 
                                a->funcs + ii, 1, b->funcs + jj*b->nrows);
        }
    }
    return c;
}

/***********************************************************//**
    Transpose qmarray - qmarray mutliplication

    \param a [in] - qmarray 1
    \param b [in] - qmarray 2

    \return c - qmarray : c = a^T b
***************************************************************/
struct Qmarray *
qmatqma(struct Qmarray * a, struct Qmarray * b)
{
    struct Qmarray * c = qmarray_alloc(a->ncols, b->ncols);
    size_t ii,jj;
    for (jj = 0; jj < b->ncols; jj++){
        for (ii = 0; ii < a->ncols; ii++){
            //printf("computing c[%zu,%zu] \n",ii,jj);
            // c[jj*a->ncols+ii] = a[:,ii]^T b[:,jj]
            c->funcs[jj*c->nrows+ii] =  generic_function_sum_prod(b->nrows, 1, 
                    a->funcs + ii*a->nrows, 1, b->funcs + jj*b->nrows);
        }
    }
    return c;
}

/***********************************************************//**
    qmarray - transpose qmarray mutliplication

    \param a [in] - qmarray 1
    \param b [in] - qmarray 2

    \return c - qmarray : c = a b^T
***************************************************************/
struct Qmarray *
qmaqmat(struct Qmarray * a, struct Qmarray * b)
{
    struct Qmarray * c = qmarray_alloc(a->nrows, b->nrows);
    size_t ii,jj;
    for (jj = 0; jj < b->nrows; jj++){
        for (ii = 0; ii < a->nrows; ii++){
            // c[jj*c->ncols+ii] = a[ii,:]^T b[jj,:]^T
            c->funcs[jj*c->nrows+ii] =  generic_function_sum_prod(a->ncols, a->nrows, 
                    a->funcs + ii, b->nrows, b->funcs + jj);
        }
    }
    return c;
}


/***********************************************************//**
    Transpose qmarray - transpose qmarray mutliplication

    \param a [in] - qmarray 1
    \param b [in] - qmarray 2

    \return c - qmarray 
***************************************************************/
struct Qmarray *
qmatqmat(struct Qmarray * a, struct Qmarray * b)
{
    struct Qmarray * c = qmarray_alloc(a->ncols, b->nrows);
    size_t ii,jj;
    for (jj = 0; jj < b->nrows; jj++){
        for (ii = 0; ii < a->ncols; ii++){
            // c[jj*a->ncols+ii] = a[:,ii]^T b[jj,:]
            c->funcs[ii*c->nrows+jj] =  generic_function_sum_prod(b->ncols, 1, 
                    a->funcs + ii*a->nrows, b->nrows, b->funcs + jj);
        }
    }
    return c;
}

/***********************************************************//**
    Integral of Transpose qmarray - qmarray mutliplication

    \param a [in] - qmarray 1
    \param b [in] - qmarray 2

    \return c - array of integrals of  \f$ c = \int a(x)^T b(x) dx \f$
***************************************************************/
double *
qmatqma_integrate(struct Qmarray * a, struct Qmarray * b)
{
    double * c = calloc_double(a->ncols*b->ncols);
    size_t nrows = a->ncols;
    size_t ncols = b->ncols;
    size_t ii,jj;
    for (jj = 0; jj < ncols; jj++){
        for (ii = 0; ii < nrows; ii++){
            // c[jj*nrows+ii] = a[:,ii]^T b[jj,:]
            c[jj*nrows+ii] =  generic_function_inner_sum(b->nrows, 1, 
                    a->funcs + ii*a->nrows, 1, b->funcs + jj*b->nrows);
        }
    }
    return c;
}

/***********************************************************//**
    Integral of qmarray - transpose qmarray mutliplication

    \param a [in] - qmarray 1
    \param b [in] - qmarray 2

    \return c - array of integrals of  \f$ c = \int a(x) b(x)^T dx \f$
***************************************************************/
double *
qmaqmat_integrate(struct Qmarray * a, struct Qmarray * b)
{
    double * c = calloc_double(a->nrows*b->nrows);
    size_t nrows = a->nrows;
    size_t ncols = b->nrows;
    size_t ii,jj;
    for (jj = 0; jj < ncols; jj++){
        for (ii = 0; ii < nrows; ii++){
            // c[jj*nrows+ii] = a[:,ii]^T b[jj,:]
            c[jj*nrows+ii] =  generic_function_inner_sum(a->ncols, a->nrows, 
                    a->funcs + ii, b->nrows, b->funcs + jj);
        }
    }
    return c;
}



/***********************************************************//**
    Integral of Transpose qmarray - transpose qmarray mutliplication

    \param a [in] - qmarray 1
    \param b [in] - qmarray 2

    \return c - array of integrals
***************************************************************/
double *
qmatqmat_integrate(struct Qmarray * a, struct Qmarray * b)
{
    double * c = calloc_double(a->ncols*b->nrows);
    size_t ii,jj;
    for (jj = 0; jj < b->nrows; jj++){
        for (ii = 0; ii < a->ncols; ii++){
            // c[jj*a->ncols+ii] = a[:,ii]^T b[jj,:]
            //c[ii*b->nrows+jj] =  generic_function_sum_prod_integrate(b->ncols, 1, 
            //        a->funcs + ii*a->nrows, b->nrows, b->funcs + jj);

            c[ii*b->nrows+jj] =  generic_function_inner_sum(b->ncols, 1, 
                    a->funcs + ii*a->nrows, b->nrows, b->funcs + jj);
        }
    }
    return c;
}


/***********************************************************//**
    Kronecker product between two qmarrays

    \param a [in] - qmarray 1
    \param b [in] - qmarray 2

    \return c -  qmarray kron(a,b)
***************************************************************/
struct Qmarray * qmarray_kron(struct Qmarray * a, struct Qmarray * b)
{
    struct Qmarray * c = qmarray_alloc(a->nrows * b->nrows, 
                                       a->ncols * b->ncols);
    size_t ii,jj,kk,ll,column,row, totrows;

    totrows = a->nrows * b->nrows;

    for (ii = 0; ii < a->ncols; ii++){
        for (jj = 0; jj < a->nrows; jj++){
            for (kk = 0; kk < b->ncols; kk++){
                for (ll = 0; ll < b->nrows; ll++){
                    column = ii*b->ncols+kk;
                    row = jj*b->nrows + ll;
                    c->funcs[column*totrows + row] = generic_function_prod(
                        a->funcs[ii*a->nrows+jj],b->funcs[kk*b->nrows+ll]);
                }
            }
        }
    }
    return c;
}

/***********************************************************//**
    Kronecker product between two qmarrays

    \param a [in] - qmarray 1
    \param b [in] - qmarray 2

    \return c -  qmarray kron(a,b)
***************************************************************/
double * qmarray_kron_integrate(struct Qmarray * a, struct Qmarray * b)
{
    double * c = calloc_double(a->nrows*b->nrows * a->ncols*b->ncols);

    size_t ii,jj,kk,ll,column,row, totrows;

    totrows = a->nrows * b->nrows;

    for (ii = 0; ii < a->ncols; ii++){
        for (jj = 0; jj < a->nrows; jj++){
            for (kk = 0; kk < b->ncols; kk++){
                for (ll = 0; ll < b->nrows; ll++){
                    column = ii*b->ncols+kk;
                    row = jj*b->nrows + ll;
                    c[column*totrows + row] = generic_function_inner(
                        a->funcs[ii*a->nrows+jj],b->funcs[kk*b->nrows+ll]);
                }
            }
        }
    }
    return c;
}


/***********************************************************//**
    If a is vector, b and c are quasimatrices then computes
             a^T kron(b,c)

    \param a [in] - vector,array (b->ncols * c->ncols);
    \param b [in] - qmarray 1
    \param c [in] - qmarray 2

    \return d - qmarray  b->ncols x (c->ncols)
***************************************************************/
struct Qmarray *
qmarray_vec_kron(double * a, struct Qmarray * b, struct Qmarray * c)
{
    struct Qmarray * d = NULL;

    struct Qmarray * temp = qmatm(b,a,c->nrows); // b->ncols * c->ncols
    //printf("got qmatm\n");
    //d = qmaqma(temp,c);
    d = qmatqmat(c,temp);
    qmarray_free(temp); temp = NULL;
    return d;
}

/***********************************************************//**
    If a is vector, b and c are quasimatrices then computes
            \f$ \int a^T kron(b(x),c(x)) dx  \f$

    \param a [in] - vector,array (b->ncols * c->ncols);
    \param b [in] - qmarray 1
    \param c [in] - qmarray 2

    \return d - qmarray  b->ncols x (c->ncols)
***************************************************************/
double *
qmarray_vec_kron_integrate(double * a, struct Qmarray * b, struct Qmarray * c)
{
    double * d = NULL;

    struct Qmarray * temp = qmatm(b,a,c->nrows); // b->ncols * c->ncols
    //printf("got qmatm\n");
    //d = qmaqma(temp,c);
    d = qmatqmat_integrate(c,temp);
    qmarray_free(temp); temp = NULL;
    return d;
}


/***********************************************************//**
    Integrate all the elements of a qmarray

    \param a [in] - qmarray to integrate

    \return out - array containing integral of every function in a
***************************************************************/
double * qmarray_integrate(struct Qmarray * a)
{
    
    double * out = calloc_double(a->nrows*a->ncols);
    size_t ii, jj;
    for (ii = 0; ii < a->ncols; ii++){
        for (jj = 0; jj < a->nrows; jj++){
            out[ii*a->nrows + jj] = generic_function_integral(a->funcs[ii*a->nrows+jj]);
        }
    }
    
    return out;
}
/***********************************************************//**
    Two norm difference between two functions

    \param a [in] - first qmarray
    \param b [in] - second qmarray

    \return out - difference in 2 norm
***************************************************************/
double qmarray_norm2diff(struct Qmarray * a, struct Qmarray * b)
{
    struct GenericFunction ** temp = generic_function_array_daxpby(
            a->ncols*a->nrows, 1.0,1,a->funcs,-1.0,1,b->funcs);
    double out = 0.0;
    size_t ii;
    for (ii = 0; ii < a->ncols * a->nrows; ii++){
        out += generic_function_inner(temp[ii],temp[ii]);
    }
    if (out < 0){
        fprintf(stderr,"Inner product between two qmarrays cannot be negative %G\n",out);
        exit(1);
        return 0.0;
    }
    generic_function_array_free(a->ncols*a->nrows,temp);
    //free(temp);
    temp = NULL;
    return sqrt(out);
}

struct Zeros {
    double * zeros;
    size_t N;
};

double eval_zpoly(double x, void * args){

    struct Zeros * z = args;
    double out = 1.0;
    size_t ii;
    for (ii = 0; ii < z->N; ii++){
        out *= (x - z->zeros[ii]);
    }
    return out;
}
void create_any_L(struct GenericFunction ** L, size_t nrows, 
            size_t upto,size_t * piv, double * px, double lb,double ub)
{
    
    //create an arbitrary quasimatrix array that has zeros at piv[:upto-1],px[:upto-1]
    // and one at piv[upto],piv[upto] less than one every else
    size_t ii,jj;
    size_t * count = calloc_size_t(nrows);
    double ** zeros = malloc( nrows * sizeof(double *));
    enum poly_type ptype = LEGENDRE;
    assert (zeros != NULL);
    for (ii = 0; ii < nrows; ii++){
        zeros[ii] = calloc_double(upto);
        for (jj = 0; jj < upto; jj++){
            if (piv[jj] == ii){
                zeros[ii][count[ii]] = px[jj];
                count[ii]++;
            }
        }
        if (count[ii] == 0){
            L[ii] = generic_function_constant(1.0,POLYNOMIAL,&ptype,lb,ub,NULL);
        }
        else{
            struct Zeros zz;
            zz.zeros = zeros[ii];
            zz.N = count[ii];
            L[ii] = generic_function_approximate1d(eval_zpoly, &zz,
                            POLYNOMIAL,&ptype,lb,ub,NULL);
        }
        free(zeros[ii]); zeros[ii] = NULL;
    }
    
    free(zeros); zeros = NULL;
    free(count); count = NULL;

    //this line is critical!!
    size_t amind;
    double xval;
    if (VQMALU){
        printf("inside of creatin g any L \n");
        printf("nrows = %zu \n",nrows);
        for (ii = 0; ii < nrows; ii++){
            print_generic_function(L[ii],3,NULL);
        }
    }

    double val = generic_function_array_absmax(nrows, 1, L,&amind, &xval);
    if (VQMALU){
        printf("got new val = %G\n", val);
    }

    px[upto] = xval;
    piv[upto] = amind;
    assert (val > ZEROTHRESH);
    generic_function_array_scale(1.0/val,L,nrows);
}

/***********************************************************//**
    Compute the LU decomposition of a quasimatrix array of 1d functioins

    \param A [in] - qmarray to decompose
    \param L [inout] - qmarray representing L factor
    \param u [inout] - allocated space for U factor
    \param piv [inout] - row of pivots 
    \param px [inout] - x values of pivots 

    \return info = 0 full rank <0 low rank ( rank = A->n + info )
***************************************************************/
int qmarray_lu1d(struct Qmarray * A, struct Qmarray * L, double * u,
            size_t * piv, double * px)
{
    int info = 0;
    
    size_t ii,kk;
    double val, amloc;
    size_t amind;
    struct GenericFunction ** temp = NULL;
    
    double lb = generic_function_get_lower_bound(A->funcs[0]);
    double ub = generic_function_get_upper_bound(A->funcs[0]);
    for (kk = 0; kk < A->ncols; kk++){

        if (VQMALU){
            printf("\n\n\n\n\n");
            printf("lu kk=%zu out of %zu \n",kk,A->ncols-1);
            printf("norm=%G\n",generic_function_norm(A->funcs[kk*A->nrows]));
        }
        //printf("A->nrows=%zu\n",A->nrows);
        //print_generic_function(A->funcs[kk*A->nrows],0,NULL);
        //print_qmarray(A,3,NULL);
        //printf("before absmax\n");
        //this line is critical!!
        val = generic_function_array_absmax(A->nrows, 1, A->funcs + kk*A->nrows, 
                                                    &amind, &amloc);
        //printf("after absmax\n");
        piv[kk] = amind;
        px[kk] = amloc;
        //printf("absmax\n");
        if (VQMALU){
            printf("locmax=%3.15G\n",amloc);
            printf("amindmax=%zu\n",amind);
            printf("val of max =%3.15G\n",val);
            print_generic_function(A->funcs[kk*A->nrows+amind],0,NULL);
        }

        val = generic_function_1d_eval(A->funcs[kk*A->nrows+amind], amloc);
        //printf("val = %G\n", val);
        //assert(fabs(val) > ZEROTHRESH); // dont deal with zero functions yet
        if (fabs(val) <= ZEROTHRESH) {
            // THIS IS STILL INCORRECT BECAUSE A CONSTANT L DOES NOT SATISFY THE REQUIRED CONDITIONS
            if (VQMALU){
                printf("creating any L\n");
            }
            create_any_L(L->funcs+kk*L->nrows,L->nrows,kk,piv,px,lb,ub);
            amind = piv[kk];
            amloc = px[kk];

            if (VQMALU){
                printf("done creating any L\n");
            }
            //print_generic_function(L->funcs[kk*L->nrows],0,NULL);
            //print_generic_function(L->funcs[kk*L->nrows+1],0,NULL);
            //printf("in here\n");
            //val = 0.0;
        }
        else{
            generic_function_array_daxpby2(A->nrows,1.0/val, 1, 
                                    A->funcs + kk*A->nrows, 
                                    0.0, 1, NULL, 1, L->funcs + kk * L->nrows);
        }
        //printf("k start here\n");
        for (ii = 0; ii < A->ncols; ii++){
            if (VQMALU){
                printf(" in lu qmarray ii=%zu/%zu \n",ii,A->ncols);
            }
            if (ii == kk){
                u[ii*A->ncols+kk] = val;
            }
            else{
                u[ii*A->ncols+kk] = generic_function_1d_eval(
                            A->funcs[ii*A->nrows + amind], amloc);
            }
            if (VQMALU){
                printf("u=%3.15G\n",u[ii*A->ncols+kk]);
            }
            /*
            print_generic_function(A->funcs[ii*A->nrows],3,NULL);
            print_generic_function(L->funcs[kk*L->nrows],3,NULL);
            */
            //printf("compute temp\n");
            temp = generic_function_array_daxpby(A->nrows, 1.0, 1,
                            A->funcs + ii*A->nrows, 
                            -u[ii*A->ncols+kk], 1, L->funcs+ kk*L->nrows);
            if (VQMALU){
                print_generic_function(A->funcs[ii*A->nrows],0,NULL);
                printf("norm pre=%G\n",generic_function_norm(A->funcs[ii*A->nrows]));
            }
            //printf("got this daxpby\n");
            qmarray_set_column_gf(A,ii,temp);
            if (VQMALU)
                printf("norm post=%G\n",generic_function_norm(A->funcs[ii*A->nrows]));
            generic_function_array_free(A->nrows,temp);
        }
    }
    //printf("done?\n");
    // check if matrix is full rank
    for (kk = 0; kk < A->ncols; kk++){
        if (fabs(u[kk*A->ncols + kk]) < ZEROTHRESH){
            info --;
        }
    }
    
    return info;
}


void remove_duplicates(size_t dim, size_t ** pivi, double ** pivx, double lb, double ub)
{
    
    size_t ii,jj;
    size_t * piv = *pivi;
    double * xiv = *pivx;
    
    //printf("startindices\n");
    //dprint(dim,xiv);
    int done = 0;
    while (done == 0){
        done = 1;
        //printf("start again\n");
        for (ii = 0; ii < dim; ii++){
            for (jj = ii+1; jj < dim; jj++){
                if (piv[ii] == piv[jj]){
                    double diff = fabs(xiv[ii] - xiv[jj]);
                    double difflb = fabs(xiv[jj] - lb);
                    //double diffub = fabs(xiv[jj] - ub);
        
                    if (diff < ZEROTHRESH){
                        //printf("difflb=%G\n",difflb);
                        if (difflb > ZEROTHRESH){
                            xiv[jj] = (xiv[jj] + lb)/2.0;
                        }
                        else{
                        //    printf("use upper bound=%G\n",ub);
                            xiv[jj] = (xiv[jj] + ub)/2.0;
                        }
                        done = 0;
                        //printf("\n ii=%zu, old xiv[%zu]=%G\n",ii,jj,xiv[jj]);
                        //xiv[jj] = 0.12345;
                        //printf("new xiv[%zu]=%G\n",jj,xiv[jj]);
                        break;
                    }
                }
            }
            if (done == 0){
                break;
            }
        }
        //printf("indices\n");
        //dprint(dim,xiv);
        //done = 1;
    }
}


/***********************************************************//**
    Perform a greedy maximum volume procedure to find the 
    maximum volume submatrix of a qmarray

    \param A [in] - qmarray
    \param Asinv [inout] - submatrix inv
    \param pivi [inout] - row pivots 
    \param pivx [inout] - x value in row pivots 

    \return info = 
            0 converges,
            < 0 no invertible submatrix,
            >0 rank may be too high,
    
    \note
        naive implementation without rank 1 updates
***************************************************************/
int qmarray_maxvol1d(struct Qmarray * A, double * Asinv, size_t * pivi, 
        double * pivx)
{
    //printf("in maxvolqmarray!\n");

    int info = 0;
    double delta = 0.01;
    size_t r = A->ncols;

    struct Qmarray * L = qmarray_alloc(A->nrows, r);
    double * U = calloc_double(r * r);
    
    struct Qmarray * Acopy = qmarray_copy(A);
    
    if (VQMAMAXVOL){
        printf("luqmarray \n");
        size_t ll;
        for (ll = 0; ll < A->nrows * A->ncols; ll++){
            printf("%G\n", generic_function_norm(Acopy->funcs[ll]));
        }
    }

    //print_qmarray(Acopy,0,NULL);
    info =  qmarray_lu1d(Acopy,L,U,pivi, pivx);
    if (VQMAMAXVOL){
        printf("pivot immediate \n");
        iprint_sz(A->ncols, pivi);
        //pivx[A->ncols-1] = 0.12345;
        dprint(A->ncols,pivx);
    }
    if (info > 0){
        //printf("Couldn't find an invertible submatrix for maxvol %d\n",info);
        printf("Error in input %d of qmarray_lu1d\n",info);
        //printf("Couldn't Error from quasimatrix_lu1d \n");
        qmarray_free(L);
        qmarray_free(Acopy);
        free(U);
        return info;
    }
    
    size_t ii,jj;
    for (ii = 0; ii < r; ii++){
        for (jj = 0; jj < r; jj++){
            U[jj * r + ii] = generic_function_1d_eval( 
                    A->funcs[jj*A->nrows + pivi[ii]], pivx[ii]);
        }
    }
    if (VQMAMAXVOL){
        printf("built U\nn");
    }

    int * ipiv2 = calloc(r, sizeof(int));
    size_t lwork = r*r;
    double * work = calloc(lwork, sizeof(double));
    //int info2;
    
    pinv(r,r,r,U,Asinv,ZEROTHRESH);
    /*
    dgetrf_(&r,&r, Asinv, &r, ipiv2, &info2); 
    printf("info2a=%d\n",info2);
    dgetri_(&r, Asinv, &r, ipiv2, work, &lwork, &info2); //invert
    printf("info2b=%d\n",info2);
    */
        
    if (VQMAMAXVOL){
        printf("took pseudoinverse \nn");
    }
    struct Qmarray * B = qmam(A,Asinv,r);

    if (VQMAMAXVOL){
        printf("did qmam \n");
    }
    size_t maxrow, maxcol;
    double maxx, maxval, maxval2;
    
    //printf("B size = (%zu,%zu)\n",B->nrows, B->ncols);
    // BLAH 2

    if (VQMAMAXVOL){
        printf("do absmax1d\n");
    }
    qmarray_absmax1d(B, &maxx, &maxrow, &maxcol, &maxval);
    if (VQMAMAXVOL){
        printf("maxrow=%zu maxcol=%zu maxval=%G\n",maxrow, maxcol, maxval);
    }
    size_t maxiter = 10;
    size_t iter =0;
    while (maxval > (1.0 + delta)){
        pivi[maxcol] = maxrow;
        pivx[maxcol] = maxx;
        
        //printf(" update Asinv A size = (%zu,%zu) \n",A->nrows,A->ncols);
        for (ii = 0; ii < r; ii++){
            //printf("pivx[%zu]=%3.5f pivi=%zu \n",ii,pivx[ii],pivi[ii]);
            for (jj = 0; jj < r; jj++){
                //printf("jj=%zu\n",jj);
                //print_generic_function(A->funcs[jj*A->nrows+pivi[ii]],0,NULL);
                U[jj * r + ii] = generic_function_1d_eval(
                        A->funcs[jj*A->nrows + pivi[ii]], pivx[ii]);
                //printf("got jj \n");
            }
        }
        
        //printf(" before dgetrf \n");

        pinv(r,r,r,U,Asinv,ZEROTHRESH);
        //dgetrf_(&r,&r, Asinv, &r, ipiv2, &info2); 
        //dgetri_(&r, Asinv, &r, ipiv2, work, &lwork, &info2); //invert
        qmarray_free(B); B = NULL;
        B = qmam(A,Asinv,r);
        qmarray_absmax1d(B, &maxx, &maxrow, &maxcol, &maxval2);

        if (fabs(maxval2-maxval)/fabs(maxval) < 1e-2){
            break;
        }
        if (iter > maxiter){
            break;
        }
        maxval = maxval2;
        //printf("maxval=%G\n",maxval);
        iter++;
    }
    //printf("done\n");
    //if ( r < 0){
    if ( r > 1){
        double lb = generic_function_get_lower_bound(A->funcs[0]);
        double ub = generic_function_get_upper_bound(A->funcs[0]);
        remove_duplicates(r, &pivi, &pivx,lb,ub);
        for (ii = 0; ii < r; ii++){
            for (jj = 0; jj < r; jj++){
                U[jj * r + ii] = generic_function_1d_eval(
                        A->funcs[jj*A->nrows + pivi[ii]], pivx[ii]);
            }
        }
        pinv(r,r,r,U,Asinv,ZEROTHRESH);
    }
    //printf("done with maxval = %3.4f\n", maxval);

    free(work);
    free(ipiv2);
    qmarray_free(B);
    qmarray_free(L);
    qmarray_free(Acopy);
    free(U);
    return info;
}

/***********************************************************//**
    Compute the householder triangularization of a quasimatrix array. 

    \param A [inout] - qmarray to triangularize (overwritten)
    \param E [inout] - qmarray with orthonormal columns
    \param V [inout] - allocated space for a qmarray (will store reflectors)
    \param R [inout] - allocated space upper triangular factor

    \return info - (if != 0 then something is wrong)
***************************************************************/
int 
qmarray_householder(struct Qmarray * A, struct Qmarray * E, 
        struct Qmarray * V, double * R)
{
    //printf("\n\n\n\n\n\n\n\n\nSTARTING QMARRAY_HOUSEHOLDER\n");
    size_t ii, jj;
    struct Quasimatrix * v = NULL;
    struct Quasimatrix * v2 = NULL;
    struct Quasimatrix * atemp = NULL;
    double rho, sigma, alpha;
    double temp1;
    
    //printf("ORTHONORMAL QMARRAY IS \n");
    //print_qmarray(E,3,NULL);
    //printf("qm array\n");
    for (ii = 0; ii < A->ncols; ii++){
        if (VQMAHOUSEHOLDER){
            printf(" On iter (%zu / %zu) \n", ii, A->ncols);
        }
        //printf("function is\n");
        //print_generic_function(A->funcs[ii*A->nrows],3,NULL);

        rho = generic_function_array_norm(A->nrows,1,A->funcs+ii*A->nrows);
        
        if (rho < ZEROTHRESH){
           rho = 0.0;
        }

        if (VQMAHOUSEHOLDER){
            printf(" \t ... rho = %3.15G ZEROTHRESH=%3.15G\n",
                            rho,ZEROTHRESH);
        }

        //printf(" new E = \n");
        //print_generic_function(E->funcs[ii*E->nrows],3,NULL);
        R[ii * A->ncols + ii] = rho;
        alpha = generic_function_inner_sum(A->nrows,1,E->funcs+ii*E->nrows,
                                            1, A->funcs+ii*A->nrows);
        
        if (fabs(alpha) < ZEROTHRESH)
            alpha=0.0;

        if (VQMAHOUSEHOLDER){
            printf(" \t ... alpha = %3.15G\n",alpha);
        }

        if (alpha >= ZEROTHRESH){
            //printf("flip sign\n");
            generic_function_array_flip_sign(E->nrows,1,E->funcs+ii*E->nrows);
        }
        v = quasimatrix_alloc(A->nrows);
        free(v->funcs);v->funcs=NULL;
        //printf("array daxpby nrows = %zu\n",A->nrows);
        //printf("epostalpha_qma = \n");
        //print_generic_function(E->funcs[ii*E->nrows],3,NULL);
        //printf("Apostalpha_qma = \n");
        //print_generic_function(A->funcs[ii*A->nrows],3,NULL);
        v->funcs = generic_function_array_daxpby(A->nrows,rho,1,
                        E->funcs+ii*E->nrows,-1.0, 1, A->funcs+ii*A->nrows);

        //printf("v update = \n");
        //print_generic_function(v->funcs[0],3,NULL);

        //printf("QMARRAY MOD IS \n");
        //print_qmarray(E,0,NULL);
        // improve orthogonality
        //*
        if (ii > 1){
            struct Quasimatrix * tempv = NULL;
            for (jj = 0; jj < ii-1; jj++){
                //printf("jj=%zu (max is %zu) \n",jj, ii-1);
                tempv = quasimatrix_copy(v);
                //printf("compute temp\n");
                //print_quasimatrix(tempv, 0, NULL);
                temp1 =  generic_function_inner_sum(A->nrows,1,tempv->funcs,1,
                                                E->funcs + jj*E->nrows);
                //printf("temp1= %G\n",temp1);
                generic_function_array_free(A->nrows, v->funcs);
                v->funcs = generic_function_array_daxpby(A->nrows,1.0,1,tempv->funcs,
                                                -temp1, 1, E->funcs+jj*E->nrows);
                //printf("k ok= %G\n",temp1);
                quasimatrix_free(tempv);
            }
        }
        //*/

        //printf("compute sigma\n");
        //print_generic_function(v->funcs[0],3,NULL);
        sigma = generic_function_array_norm(v->n, 1, v->funcs);
        if (sigma < ZEROTHRESH){
            sigma = 0.0;
        }

        if (VQMAHOUSEHOLDER){
            printf(" \t ... sigma = %3.15G ZEROTHRESH=%3.15G\n",
                            sigma,ZEROTHRESH);
        }
        
        //printf("v2preqma = \n");
        //print_generic_function(v->funcs[0],3,NULL);
        double ttol = ZEROTHRESH;
        if (fabs(sigma) <= ttol){
            //printf("here sigma too small!\n");
            v2 = quasimatrix_alloc(A->nrows);
            free(v2->funcs);v2->funcs=NULL;
            v2->funcs = generic_function_array_daxpby(A->nrows,1.0, 1, 
                            E->funcs+ii*E->nrows, 0.0, 1, NULL); //just a copy
        }
        else {
            //printf("quasi daxpby\n");
            //printf("sigma = %G\n",sigma);
            v2 = quasimatrix_daxpby(1.0/sigma, v, 0.0, NULL);
            //printf("quasi got it daxpby\n");
        }
        quasimatrix_free(v);
        qmarray_set_column(V,ii,v2);
        
        //printf("v2qma = \n");
        //print_generic_function(v2->funcs[0],3,NULL);
        //printf("go into cols \n");
        for (jj = ii+1; jj < A->ncols; jj++){

            if (VQMAHOUSEHOLDER){
                //printf("\t On sub-iter %zu\n", jj);
            }
            //printf("Aqma = \n");
            //print_generic_function(A->funcs[jj*A->nrows],3,NULL);
            temp1 = generic_function_inner_sum(A->nrows,1,v2->funcs,1,
                                        A->funcs+jj*A->nrows);

            //printf("\t\t temp1 pre = %3.15G DBL_EPS=%G\n", temp1,DBL_EPSILON);
            if (fabs(temp1) < ZEROTHRESH){
                temp1 = 0.0;
            }

            atemp = quasimatrix_alloc(A->nrows);
            free(atemp->funcs);
            atemp->funcs = generic_function_array_daxpby(A->nrows, 1.0, 1, 
                                A->funcs+jj*A->nrows, -2.0 * temp1, 1, 
                                v2->funcs);
            
            //printf("atemp = \n");
            //print_generic_function(atemp->funcs[0], 0, NULL);
            R[jj * A->ncols + ii] = generic_function_inner_sum(E->nrows, 1,
                        E->funcs + ii*E->nrows, 1, atemp->funcs);


            //double aftemp = 
            //    generic_function_array_norm(A->nrows,1,atemp->funcs);

            //double eftemp = 
            //    generic_function_array_norm(E->nrows,1,E->funcs+ii*E->nrows);

            if (VQMAHOUSEHOLDER){
                //printf("\t \t ... temp1 = %3.15G\n",temp1);
                //printf("\t\t e^T atemp= %3.15G\n", R[jj*A->ncols+ii]);
            }

            v = quasimatrix_alloc(A->nrows);
            free(v->funcs);v->funcs=NULL;
            v->funcs = generic_function_array_daxpby(A->nrows, 1.0, 1, 
                atemp->funcs, -R[jj * A->ncols + ii], 1,
                E->funcs + ii*E->nrows);
            


            if (VQMAHOUSEHOLDER){
                //double temprho = 
                //    generic_function_array_norm(A->nrows,1,v->funcs);
                //printf("new v func norm = %3.15G\n",temprho);
	        }

            qmarray_set_column(A,jj,v);  // overwrites column jj

            quasimatrix_free(atemp); atemp=NULL;
            quasimatrix_free(v); v = NULL;
        }

        quasimatrix_free(v2);
    }
    
    //printf("qmarray v after = \n");
    //print_qmarray(V,0,NULL);

    //printf("qmarray E after = \n");
    //print_qmarray(E,0,NULL);
    //printf("\n\n\n\n\n\n\n\n\n Ending QMARRAY_HOUSEHOLDER\n");
    return 0;
}

/***********************************************************//**
    Compute the Q for the QR factor from householder reflectors
    for a Quasimatrix Array

    \param Q [inout] - qmarray E obtained afterm qmarray_householder
                       becomes overwritten
    \param V [inout] - Householder functions, obtained from qmarray_householder

    \return info - if != 0 then something is wrong)
***************************************************************/
int qmarray_qhouse(struct Qmarray * Q, struct Qmarray * V)
{
    
    int info = 0;
    size_t ii, jj;

    struct Quasimatrix * q2 = NULL;
    double temp1;
    size_t counter = 0;
    ii = Q->ncols-1;

    /*
    printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
    printf("************Q beforer*************\n");
    print_qmarray(Q, 0, NULL);
    printf("************V beforer*************\n");
    print_qmarray(V, 0, NULL);
    */
    while (counter < Q->ncols){
        //printf("INCREMENT COUNTER \n");
        for (jj = ii; jj < Q->ncols; jj++){

            temp1 = generic_function_inner_sum(V->nrows,1,V->funcs + ii*V->nrows,
                                        1, Q->funcs + jj*Q->nrows);
            //printf("temp1=%G\n",temp1);    
            q2 = quasimatrix_alloc(Q->nrows);
            free(q2->funcs); q2->funcs = NULL;
            q2->funcs = generic_function_array_daxpby(Q->nrows, 1.0, 1, 
                        Q->funcs + jj * Q->nrows, -2.0 * temp1, 1, 
                        V->funcs + ii * V->nrows);

            qmarray_set_column(Q,jj, q2);
            //printf("Q new = \n");
            //print_generic_function(Q->funcs[jj],1,NULL);
            quasimatrix_free(q2); q2 = NULL;
        }
        /*
        printf(" *-========================== Q temp ============== \n");
        print_qmarray(Q, 0, NULL);
        */
        ii = ii - 1;
        counter = counter + 1;
    }
    //printf("************Q after*************\n");
    //print_qmarray(Q, 0, NULL);
    //printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");

    return info;
}

/***********************************************************//**
    Compute the householder triangularization of a 
    quasimatrix array. (LQ= A), Q orthonormal, 
    L lower triangular

    \param A [inout] - qmarray to triangularize (destroyed 
    \param E [inout] - qmarray with orthonormal rows
    \param V [inout] - allocated space for a qmarray (will store reflectors)
    \param L [inout] - allocated space lower triangular factor

    \return info (if != 0 then something is wrong)
***************************************************************/
int 
qmarray_householder_rows(struct Qmarray * A, struct Qmarray * E, 
        struct Qmarray * V, double * L)
{
    size_t ii, jj;
    struct Quasimatrix * e = NULL;
    struct Quasimatrix * x = NULL;
    struct Quasimatrix * v = NULL;//quasimatrix_alloc(A->ncols);
    struct Quasimatrix * v2 = NULL;
    struct Quasimatrix * atemp = NULL;
    double rho, sigma, alpha;
    double temp1;



    for (ii = 0; ii < A->nrows; ii++){
        if (VQMAHOUSEHOLDER)
            printf(" On iter %zu\n", ii);

        rho = generic_function_array_norm(A->ncols,A->nrows,A->funcs+ii);

        if (rho < ZEROTHRESH){
            rho = 0.0;
        }

        if (VQMAHOUSEHOLDER){
            printf(" \t ... rho = %3.15G ZEROTHRESH=%3.15G\n",
                                rho,ZEROTHRESH);
        }

        L[ii * A->nrows + ii] = rho;
        alpha = generic_function_inner_sum(A->ncols,E->nrows,E->funcs+ii,
                                            A->nrows, A->funcs+ii);
        if (fabs(alpha) < ZEROTHRESH){
           alpha = 0.0;
        }

        if (VQMAHOUSEHOLDER)
            printf(" \t ... alpha = %3.15G\n",alpha);

        if (alpha >= ZEROTHRESH){
            generic_function_array_flip_sign(E->ncols,E->nrows,E->funcs+ii);
        }

        v = quasimatrix_alloc(A->ncols);
        free(v->funcs);v->funcs=NULL;

        v->funcs = generic_function_array_daxpby(A->ncols, rho, E->nrows, 
                    E->funcs+ii, -1.0, A->nrows, A->funcs+ii);
    
        // improve orthogonality
        //*
        if (ii > 1){
            struct Quasimatrix * tempv = NULL;
            for (jj = 0; jj < ii-1; jj++){
                tempv = quasimatrix_copy(v);
                temp1 =  generic_function_inner_sum(A->ncols,1,tempv->funcs,E->nrows,
                                                E->funcs + jj);
                generic_function_array_free(A->ncols, v->funcs);
                v->funcs = generic_function_array_daxpby(A->ncols,1.0,1,tempv->funcs,
                                        -temp1, E->nrows, E->funcs+jj);
                quasimatrix_free(tempv);
            }
        }
        //*/
        sigma = generic_function_array_norm(v->n, 1, v->funcs);
        if (sigma < ZEROTHRESH){
            sigma = 0.0;
        }
        //
        double ttol = ZEROTHRESH;

        if (VQMAHOUSEHOLDER)
            printf(" \t ... sigma = %G ttol=%G \n",sigma,ttol);
        if (fabs(sigma) <= ttol){
            //printf("HERE SIGMA TOO SMALL\n");
            v2 = quasimatrix_alloc(A->ncols);
            free(v2->funcs);v2->funcs=NULL;
            v2->funcs = generic_function_array_daxpby(E->ncols,1.0, E->nrows, 
                            E->funcs+ii, 0.0, 1, NULL); //just a copy
        }
        else {
            v2 = quasimatrix_daxpby(1.0/sigma, v, 0.0, NULL);
        }
        quasimatrix_free(v);
        qmarray_set_row(V,ii,v2);

        for (jj = ii+1; jj < A->nrows; jj++){

            if (VQMAHOUSEHOLDER){
                //printf("\t On sub-iter %zu\n", jj);
		}

            temp1 = generic_function_inner_sum(A->ncols,1,v2->funcs,
                                            A->nrows, A->funcs+jj);

            if (fabs(temp1) < ZEROTHRESH){
                temp1 = 0.0;
            }
            //if (VQMAHOUSEHOLDER)
            //    printf("\t\t temp1= %3.15G\n", temp1);

            atemp = quasimatrix_alloc(A->ncols);
            free(atemp->funcs);
            atemp->funcs = generic_function_array_daxpby(A->ncols, 1.0, 
                                A->nrows, A->funcs+jj, -2.0 * temp1, 1, 
                                v2->funcs);

            L[ii * A->nrows + jj] = generic_function_inner_sum(E->ncols, 
                        E->nrows, E->funcs + ii, 1, atemp->funcs);

            //if (VQMAHOUSEHOLDER)
            //    printf("\t\t e^T atemp= %3.15G\n", L[ii*A->nrows+jj]);

            v = quasimatrix_alloc(A->ncols);
            free(v->funcs);v->funcs=NULL;
            v->funcs = generic_function_array_daxpby(A->ncols, 1.0, 1,
                atemp->funcs, -L[ii * A->nrows + jj], E->nrows, E->funcs + ii);
            
            qmarray_set_row(A,jj,v);  // overwrites column jj

            quasimatrix_free(atemp); atemp=NULL;
            quasimatrix_free(v); v = NULL;
        }

        quasimatrix_free(x); 
        quasimatrix_free(e);
        quasimatrix_free(v2);
    }
    return 0;
}

/***********************************************************//**
    Compute the Q matrix from householder reflector for LQ decomposition

    \param Q [inout] - qmarray E obtained afterm qmarray_householder_rows (overwritten)
    \param V [inout] - Householder functions, obtained from qmarray_householder

    \return info - (if != 0 then something is wrong)
***************************************************************/
int qmarray_qhouse_rows(struct Qmarray * Q, struct Qmarray * V)
{
    int info = 0;
    size_t ii, jj;

    struct Quasimatrix * q2 = NULL;
    double temp1;
    size_t counter = 0;
    ii = Q->nrows-1;
    while (counter < Q->nrows){
        for (jj = ii; jj < Q->nrows; jj++){
            temp1 = generic_function_inner_sum(V->ncols, V->nrows, V->funcs + ii,
                                        Q->nrows, Q->funcs + jj);
        
            q2 = quasimatrix_alloc(Q->ncols);
            free(q2->funcs); q2->funcs = NULL;
            q2->funcs = generic_function_array_daxpby(Q->ncols, 1.0, Q->nrows, 
                        Q->funcs + jj, -2.0 * temp1, V->nrows, 
                        V->funcs + ii);

            qmarray_set_row(Q,jj, q2);
            quasimatrix_free(q2); q2 = NULL;
        }
        ii = ii - 1;
        counter = counter + 1;
    }

    return info;
}


/***********************************************************//**
    Compute the householder triangularization of a 
    qmarray. whose elements consist of
    one dimensional functions (simple interface)

    \param dir [in] - type either "QR" or "LQ"
    \param A [inout] - qmarray to triangularize (destroyed in call)
    \param R [inout] - allocated space upper triangular factor

    \return E - quasimatrix denoting the Q term

    \note 
        For now I have only implemented this for polynomial function class 
            and legendre subtype
***************************************************************/
struct Qmarray *
qmarray_householder_simple(char * dir, struct Qmarray * A, double * R)
{
    
    double lb = generic_function_get_lower_bound(A->funcs[0]);
    double ub = generic_function_get_upper_bound(A->funcs[0]);
    

    size_t ncols = A->ncols;
    enum poly_type ptype = LEGENDRE;
    // generate two quasimatrices needed for householder
   
   
    struct Qmarray * Q = NULL;
    if (strcmp(dir,"QR") == 0){
        Q = qmarray_orth1d_columns(POLYNOMIAL, 
                        &ptype, A->nrows, ncols, lb, ub); 

        struct Qmarray * V = qmarray_alloc(A->nrows,ncols);
        
        //printf("here\n");
        int out = 0;

        
        //printf("orth=\n");
        //print_qmarray(Q, 0, NULL);
        out = qmarray_householder(A,Q,V,R);
        assert(out == 0);
        //printf("R=\n");
        //dprint2d_col(A->ncols, A->ncols,R);
        
        //printf("there!\n");
        /*
        printf("V=\n");
        print_qmarray(V, 0, NULL);
        printf("orth after =\n");
        print_qmarray(Q, 0, NULL);
        */
        out = qmarray_qhouse(Q,V);
        assert(out == 0);

        qmarray_free(V);
    }
    else if (strcmp(dir, "LQ") == 0){
        Q = qmarray_orth1d_rows(POLYNOMIAL, 
                        &ptype, A->nrows, ncols, lb, ub); 

        struct Qmarray * V = qmarray_alloc(A->nrows,ncols);
        
        //printf("here\n");
        int out = 0;
        out = qmarray_householder_rows(A,Q,V,R);
        
        //printf("there!\n");
        assert(out == 0);
        out = qmarray_qhouse_rows(Q,V);
        assert(out == 0);

        qmarray_free(V);

    }
    else{
        fprintf(stderr, "No clear QR/LQ decomposition for type=%s\n",dir);
        exit(1);
    }
    return  Q;
}

/***********************************************************//**
    Compute the svd of a quasimatrix array Udiag(lam)vt = A

    \param A [inout] - qmarray to get SVD (destroyed)
    \param U [inout] - qmarray with orthonormal columns
    \param lam [inout] - singular values
    \param vt [inout] - matrix containing right singular vectors

    \return info - if not == 0 then error
***************************************************************/
int qmarray_svd(struct Qmarray * A, struct Qmarray ** U, double * lam, 
            double * vt)
{

    int info = 0;

    double * R = calloc_double(A->ncols * A->ncols);
    struct Qmarray * temp = qmarray_householder_simple("QR",A,R);

    double * u = calloc_double(A->ncols * A->ncols);
    svd(A->ncols, A->ncols, A->ncols, R, u,lam,vt);
    
    qmarray_free(*U);
    *U = qmam(temp,u,A->ncols);
    
    qmarray_free(temp);
    free(u);
    free(R);
    return info;
}

/***********************************************************//**
    Compute the reduced svd of a quasimatrix array 
    Udiag(lam)vt = A  with every singular value greater than delta

    \param A [inout] - qmarray to get SVD (destroyed)
    \param U [inout] - qmarray with orthonormal columns
    \param lam [inout] - singular values 
    \param vt [inout] - matrix containing right singular vectors
    \param delta [inout] - threshold

    \return rank - rank of the qmarray
    
    \note
        *U*, *lam*, and *vt* are allocated in this function
***************************************************************/
size_t qmarray_truncated_svd(struct Qmarray * A, struct Qmarray ** U, 
            double ** lam, double ** vt, double delta)
{

    //int info = 0;
    double * R = calloc_double(A->ncols * A->ncols);
    struct Qmarray * temp = qmarray_householder_simple("QR",A,R);

    double * u = NULL;
    size_t rank = truncated_svd(A->ncols, A->ncols, A->ncols, R, &u, lam, vt, 
                                delta);
    
    qmarray_free(*U);
    *U = qmam(temp,u,rank);
    
    qmarray_free(temp);
    free(u);
    free(R);
    return rank;
}

/***********************************************************//**
    Compute the location of the (abs) maximum element in aqmarray consisting of 1 dimensional functions

    \param qma [in] - qmarray 1
    \param xmax [inout] - x value at maximum
    \param rowmax [inout] - row of maximum
    \param colmax [inout] - column of maximum
    \param maxval [inout] - absolute maximum value
***************************************************************/
void qmarray_absmax1d(struct Qmarray * qma, double * xmax, size_t * rowmax,
                       size_t * colmax, double * maxval)
{
    size_t combrowcol;
    *maxval = generic_function_array_absmax(qma->nrows * qma->ncols,
                            1, qma->funcs, &combrowcol, xmax);
    
    size_t nsubs = 0;
    //printf("combrowcol = %zu \n", combrowcol);
    while ((combrowcol+1) > qma->nrows){
        nsubs++;
        combrowcol -= qma->nrows;
    }

    *rowmax = combrowcol;
    *colmax = nsubs;
    //printf("rowmax, colmaxx = (%zu,%zu)\n", *rowmax, *colmax);
}

/***********************************************************//**
    Transpose a qmarray

    \param a [in] - qmarray 

    \return b - \f$ b(x) = a(x)^T \f$
***************************************************************/
struct Qmarray * 
qmarray_transpose(struct Qmarray * a)
{
    
    struct Qmarray * b = qmarray_alloc(a->ncols, a->nrows);
    size_t ii, jj;
    for (ii = 0; ii < a->nrows; ii++){
        for (jj = 0; jj < a->ncols; jj++){
            b->funcs[ii*a->ncols+jj] = 
                generic_function_copy(a->funcs[jj*a->nrows+ii]);
        }
    }
    return b;
}

/***********************************************************//**
    Stack two qmarrays horizontally

    \param a [in] - qmarray 1
    \param b [in] - qmarray 2

    \return c - \f$ [a b] \f$
***************************************************************/
struct Qmarray * qmarray_stackh(struct Qmarray * a, struct Qmarray * b)
{
    assert (a->nrows == b->nrows);
    struct Qmarray * c = qmarray_alloc(a->nrows, a->ncols + b->ncols);
    size_t ii;
    for (ii = 0; ii < a->nrows * a->ncols; ii++){
        c->funcs[ii] = generic_function_copy(a->funcs[ii]);
    }
    for (ii = 0; ii < b->nrows * b->ncols; ii++){
        c->funcs[ii+a->nrows*a->ncols] = generic_function_copy(b->funcs[ii]);
    }
    return c;
}

/***********************************************************//**
    Stack two qmarrays vertically

    \param a [in] - qmarray 1
    \param b [in] - qmarray 2

    \return  c : \f$ [a; b] \f$
***************************************************************/
struct Qmarray * qmarray_stackv(struct Qmarray * a, struct Qmarray * b)
{
    assert (a->ncols == b->ncols);
    struct Qmarray * c = qmarray_alloc(a->nrows+b->nrows, a->ncols);
    size_t ii, jj;
    for (jj = 0; jj < a->ncols; jj++){
        for (ii = 0; ii < a->nrows; ii++){
            c->funcs[jj*c->nrows+ii] = 
                            generic_function_copy(a->funcs[jj*a->nrows+ii]);
        }
        for (ii = 0; ii < b->nrows; ii++){
            c->funcs[jj*c->nrows+ii+a->nrows] = 
                            generic_function_copy(b->funcs[jj*b->nrows+ii]);
        }
    }

    return c;
}

/***********************************************************//**
    Block diagonal qmarray from two qmarrays

    \param a [in] - qmarray 1
    \param b [in] - qmarray 2

    \return c - \f$ [a 0 ;0 b] \f$
***************************************************************/
struct Qmarray * qmarray_blockdiag(struct Qmarray * a, struct Qmarray * b)
{
    struct Qmarray * c = qmarray_alloc(a->nrows+b->nrows, a->ncols+b->ncols);
    size_t ii, jj;
    for (jj = 0; jj < a->ncols; jj++){
        for (ii = 0; ii < a->nrows; ii++){
            c->funcs[jj*c->nrows+ii] = 
                            generic_function_copy(a->funcs[jj*a->nrows+ii]);
        }
        for (ii = 0; ii < b->nrows; ii++){
            c->funcs[jj*c->nrows+ii+a->nrows] = generic_function_daxpby(0.0,
                        a->funcs[0],0.0,NULL);
        }
    }
    for (jj = a->ncols; jj < c->ncols; jj++){
        for (ii = 0; ii < a->nrows; ii++){
            c->funcs[jj*c->nrows+ii] = generic_function_daxpby(0.0,a->funcs[0],0.0,NULL);

        }
        for (ii = 0; ii < b->nrows; ii++){
            c->funcs[jj*c->nrows+ii+a->nrows] = 
                generic_function_copy(b->funcs[(jj-a->ncols)*b->nrows+ii]);
        }
    }

    return c;
}

/***********************************************************//**
    Compute the derivative of every function in the qmarray

    \param a [in] - qmarray

    \return b - qmarray of derivatives
***************************************************************/
struct Qmarray * qmarray_deriv(struct Qmarray * a)
{

    struct Qmarray * b = qmarray_alloc(a->nrows, a->ncols);
    size_t ii;
    for (ii = 0; ii < a->nrows*a->ncols; ii++){
        //printf("function %zu is \n",ii);
        //print_generic_function(a->funcs[ii],3,NULL);
        b->funcs[ii] = generic_function_deriv(a->funcs[ii]);
        //printf("got its deriv\n");
    }
    return b;
}


/********************************************************//**
    Rounding (thresholding) of quasimatrix array

    \param qma [inout] - quasimatrix
    \param epsilon [in] - threshold

***********************************************************/
void qmarray_roundt(struct Qmarray ** qma, double epsilon)
{
    size_t ii;
    for (ii = 0; ii < (*qma)->nrows * (*qma)->ncols; ii++){
        generic_function_roundt(&((*qma)->funcs[ii]),epsilon);
    }
}


////////////////////////////////////////////////////////////////////////////
// function_train

/***********************************************************//**
    Evaluate a function train

    \param ft [in] - function train
    \param x [in] - location at which to evaluate

    \return val - value of the function train
***************************************************************/
double function_train_eval(struct FunctionTrain * ft, double * x)
{
    size_t dim = ft->dim;
    size_t ii = 0;
    double * t1 = generic_function_1darray_eval(
            ft->cores[ii]->nrows * ft->cores[ii]->ncols, 
            ft->cores[ii]->funcs, x[ii]);

    double * t2 = NULL;
    double * t3 = NULL;
    for (ii = 1; ii < dim; ii++){
        t2 = generic_function_1darray_eval(
            ft->cores[ii]->nrows * ft->cores[ii]->ncols, 
            ft->cores[ii]->funcs, x[ii]);
            
        if (ii%2 == 1){
            // previous times new core
            t3 = calloc_double(ft->ranks[ii+1]);
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, 1,
                ft->ranks[ii+1], ft->ranks[ii], 1.0, t1, 1, t2,
                ft->ranks[ii], 0.0, t3, 1);
            free(t2); t2 = NULL;
            free(t1); t1 = NULL;
        }
        else {
            t1 = calloc_double(ft->ranks[ii+1]);
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, 1,
                ft->ranks[ii+1], ft->ranks[ii], 1.0, t3, 1, t2,
                ft->ranks[ii], 0.0, t1, 1);
            free(t2); t2 = NULL;
            free(t3); t3 = NULL;
        }
    }
    
    double out = 0.123456789;
    if (t1 == NULL){
        out = t3[0];
        free(t3); t3 = NULL;
    }
    else if ( t3 == NULL){
        out = t1[0];
        free(t1); t1 = NULL;
    }
    
    return out;
}

/********************************************************//**
    Right orthogonalize the cores (except the first one) of the function train

    \param a [inout] - FT (overwritten)
    
    \return ftrl - new ft with orthogonalized cores
***********************************************************/
struct FunctionTrain * function_train_orthor(struct FunctionTrain * a)
{
    //right left sweep
    struct FunctionTrain * ftrl = function_train_alloc(a->dim);
    double * L = NULL; 
    struct Qmarray * temp = NULL;
    size_t ii = 1;
    size_t core = a->dim-ii;  
    L = calloc_double(a->cores[core]->nrows * a->cores[core]->nrows);
    memmove(ftrl->ranks,a->ranks,(a->dim+1)*sizeof(size_t));


    // update last core
    ftrl->cores[core] = qmarray_householder_simple("LQ",a->cores[core],L);
    for (ii = 2; ii < a->dim; ii++){
        core = a->dim-ii;
        //printf("on core %zu\n",core);
        temp = qmam(a->cores[core],L,ftrl->ranks[core+1]);
        free(L); L = NULL;
        L = calloc_double(ftrl->ranks[core]*ftrl->ranks[core]);
        ftrl->cores[core] = qmarray_householder_simple("LQ",temp,L);
        qmarray_free(temp); temp = NULL;
        
    }
    ftrl->cores[0] = qmam(a->cores[0],L,ftrl->ranks[1]);
    free(L); L = NULL;
    return ftrl;
}


/********************************************************//**
    Rounding of a function train

    \param a [inout] - FT (overwritten)
    \param epsilon [in] - threshold

    \return ft - rounded function train
***********************************************************/
struct FunctionTrain * 
function_train_round(struct FunctionTrain * a, double epsilon)
{
    size_t ii, core;
    struct Qmarray * temp = NULL;

    double delta = function_train_norm2(a);
    delta = delta * epsilon / sqrt(a->dim-1);
    //double delta = epsilon;
   
    struct FunctionTrain * ftrl = function_train_orthor(a);

    struct FunctionTrain * ft = function_train_alloc(a->dim);
    //struct FunctionTrain * ft = function_train_copy(ftrl);
    double * vt = NULL;
    double * s = NULL;
    double * sdiag = NULL;
    double * svt = NULL;
    size_t rank;
    size_t sizer;
    core = 0;
    ft->ranks[core] = 1;
    sizer = ftrl->cores[core]->ncols;
    
    //*
    //printf("rank before = %zu\n",ftrl->ranks[core+1]);
    rank = qmarray_truncated_svd(ftrl->cores[core], &(ft->cores[core]),
                                        &s, &vt, delta);
    //printf("rankdone\n");
    //printf("rank after = %zu\n",rank);
    ft->ranks[core+1] = rank;
    sdiag = diag(rank,s);
    svt = calloc_double(rank * sizer);
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, rank, 
                sizer, rank, 1.0, sdiag, rank, vt, rank, 0.0, 
                svt, rank);
    //printf("rank\n");
    temp = mqma(svt,ftrl->cores[core+1],rank);
    //*/

    free(vt); vt = NULL;
    free(s); s = NULL;
    free(sdiag); sdiag = NULL;
    free(svt); svt = NULL;
    for (core = 1; core < a->dim-1; core++){
        rank = qmarray_truncated_svd(temp, &(ft->cores[core]), &s, &vt, delta);
        qmarray_free(temp); temp = NULL;

        ft->ranks[core+1] = rank;
        sdiag = diag(rank,s);
        svt = calloc_double(rank * a->ranks[core+1]);
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, rank, 
                    a->ranks[core+1], rank, 1.0, sdiag, rank, vt, rank, 0.0, 
                    svt, rank);
        temp = mqma(svt,ftrl->cores[core+1],rank);
        free(vt); vt = NULL;
        free(s); s = NULL;
        free(sdiag); sdiag = NULL;
        free(svt); svt = NULL;
    }
    core = a->dim-1;
    ft->cores[core] = temp;
    ft->ranks[a->dim] = 1;
    
    function_train_free(ftrl);
    for (ii = 0; ii < ft->dim; ii++){
        qmarray_roundt(&ft->cores[ii], epsilon);
    }
    return ft;
}

/********************************************************//**
    Addition of two functions in FT format

    \param a [in] - FT 1
    \param b [in] - FT 2

    \return ft - function representing a+b
***********************************************************/
struct FunctionTrain * function_train_sum(struct FunctionTrain * a,
                                struct FunctionTrain * b)
{
    struct FunctionTrain * ft = function_train_alloc(a->dim);
    
    // first core
    ft->cores[0] = qmarray_stackh(a->cores[0], b->cores[0]);
    ft->ranks[0] = a->ranks[0];
    // middle cores
    size_t ii;
    for (ii = 1; ii < ft->dim-1; ii++){
    
        ft->cores[ii] = qmarray_blockdiag(a->cores[ii], b->cores[ii]);
        ft->ranks[ii] = a->ranks[ii] + b->ranks[ii];
    }
    ft->ranks[ft->dim-1] = a->ranks[a->dim-1] + b->ranks[b->dim-1];
    // last core
    ft->cores[ft->dim-1] = qmarray_stackv(a->cores[a->dim-1],
                                b->cores[b->dim-1]);
    ft->ranks[ft->dim] = a->ranks[ft->dim];
    
    return ft;
}

/********************************************************//**
    af(x) + b

    \param a [in] - scaling factor
    \param b [in] - offset
    \param f [in] - object to scale
    \param epsilon [in] - rounding tolerance

    \return ft - function representing a+b
***********************************************************/
struct FunctionTrain * function_train_afpb(double a, double b,
                        struct FunctionTrain * f, double epsilon)
{
    struct BoundingBox * bds = function_train_bds(f);

    struct FunctionTrain * off = function_train_constant(f->dim,b,bds,NULL);
    struct FunctionTrain * af = function_train_copy(f);
    function_train_scale(af,a);

    struct FunctionTrain * temp = function_train_sum(af,off);
    
    
    //printf("round it \n");
    //printf("temp ranks \n");
    //iprint_sz(temp->dim+1,temp->ranks);

    struct FunctionTrain * ft = function_train_round(temp,epsilon);
    //printf("got it \n");
        
    function_train_free(off); off = NULL;
    function_train_free(af); af = NULL;
    function_train_free(temp); temp = NULL;
    bounding_box_free(bds); bds = NULL;
    
    return ft;
}


/********************************************************//**
    Scale a function train representation

    \param x [inout] - Function train
    \param a [in] - scaling factor
***********************************************************/
void function_train_scale(struct FunctionTrain * x, double a)
{
    struct GenericFunction ** temp = generic_function_array_daxpby(
        x->cores[0]->nrows*x->cores[0]->ncols,a,1,x->cores[0]->funcs,0.0,1,NULL);
    generic_function_array_free(x->cores[0]->nrows * x->cores[0]->ncols, 
                                        x->cores[0]->funcs);
    x->cores[0]->funcs = temp;
}

/********************************************************//**
    Product of two functions in function train form

    \param a [in] - Function train 1
    \param b [in] - Function train 2

    \return ft - \f$ ft(x) = a(x)b(x) \f$
***********************************************************/
struct FunctionTrain * 
function_train_product(struct FunctionTrain * a, struct FunctionTrain * b)
{
    struct FunctionTrain * ft = function_train_alloc(a->dim);

    size_t ii;
    for (ii = 0; ii < ft->dim; ii++){
        ft->ranks[ii] = a->ranks[ii] * b->ranks[ii];
        ft->cores[ii] = qmarray_kron(a->cores[ii], b->cores[ii]);
    }

    ft->ranks[ft->dim] = a->ranks[ft->dim] * b->ranks[ft->dim];
    return ft;
}

/********************************************************//**
    Integrate a function in function train format

    \param ft [in] - Function train 1

    \return val - \f$ val = int a dx \f$
***********************************************************/
double 
function_train_integrate(struct FunctionTrain * ft)
{
    size_t dim = ft->dim;
    size_t ii = 0;
    double * t1 = qmarray_integrate(ft->cores[ii]);

    double * t2 = NULL;
    double * t3 = NULL;
    for (ii = 1; ii < dim; ii++){
        t2 = qmarray_integrate(ft->cores[ii]);
        if (ii%2 == 1){
            // previous times new core
            t3 = calloc_double(ft->ranks[ii+1]);
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, 1,
                ft->ranks[ii+1], ft->ranks[ii], 1.0, t1, 1, t2,
                ft->ranks[ii], 0.0, t3, 1);
            free(t2); t2 = NULL;
            free(t1); t1 = NULL;
        }
        else {
            t1 = calloc_double(ft->ranks[ii+1]);
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, 1,
                ft->ranks[ii+1], ft->ranks[ii], 1.0, t3, 1, t2,
                ft->ranks[ii], 0.0, t1, 1);
            free(t2); t2 = NULL;
            free(t3); t3 = NULL;
        }
    }
    
    double out = 0.123456789;
    if (t1 == NULL){
        out = t3[0];
        free(t3); t3 = NULL;
    }
    else if ( t3 == NULL){
        out = t1[0];
        free(t1); t1 = NULL;
    }
    
    return out;
}

/********************************************************//**
    Inner product between two functions in FT form

    \param a [in] - Function train 1
    \param b [in] - Function train 2

    \return out - int a(x)b(x) dx

    \note
        This is a slow version, I cant seem to get fast version to be accurate
***********************************************************/
double function_train_inner(struct FunctionTrain * a, struct FunctionTrain * b)
{
    double out = 0.123456789;
    size_t ii;
    //struct Qmarray * c1 = qmarray_kron(b->cores[0],a->cores[0]);
   // double * temp = generic_function_integral_array(c1->nrows*c1->ncols,1,c1->funcs);
   // qmarray_free(c1); c1 = NULL;

    ///*
    double * temp = qmarray_kron_integrate(a->cores[0],b->cores[0]);

    //*/
    //printf("temp = ");
    //dprint(b->cores[0]->ncols*a->cores[0]->ncols, temp);
    //*/
        
    /*
    printf("in function_train_inner\n");
    printf(" a ranks = ");
    iprint_sz(a->dim+1,a->ranks);
    printf(" b ranks = ");
    iprint_sz(b->dim+1,b->ranks);
    */
    double * temp2 = NULL;

    //size_t ii;
    for (ii = 1; ii < a->dim; ii++){
        //printf("ii=%zu/%zu\n",ii,a->dim);
        //print_qmarray(a->cores[ii],3,NULL);
        //printf("c1 nr = %zu, nc =%zu\n",a->cores[ii]->nrows,a->cores[ii]->ncols);
       
        temp2 = qmarray_kron_integrate(a->cores[ii],b->cores[ii]);
        //printf("got kron integrate\n");
        size_t nrows = a->cores[ii]->nrows * b->cores[ii]->nrows;
        size_t ncols = a->cores[ii]->ncols * b->cores[ii]->ncols;
        double * temp3 = calloc_double(ncols);
        cblas_dgemv(CblasColMajor,CblasTrans,nrows,ncols,1.0,temp2,nrows,temp,1,0.0,temp3,1);
        //printf("did dgemv\n");
        free(temp); temp = NULL;
        free(temp2); temp2 = NULL;
        temp = calloc_double(ncols);
        memmove(temp,temp3,ncols*sizeof(double));
        free(temp3); temp3 = NULL;

        //*
        //temp2 = qmarray_vec_kron_integrate(temp, a->cores[ii],b->cores[ii]);
        //size_t stemp = a->cores[ii]->ncols * b->cores[ii]->ncols;
        //free(temp);temp=NULL;
        //temp = calloc_double(stemp);
        //memmove(temp, temp2,stemp*sizeof(double));
        //*/
        
        free(temp2); temp2 = NULL;
    }
    
    out = temp[0];
    free(temp); temp=NULL;

    return out;
}

/********************************************************//**
    Compute the L2 norm of a function in FT format

    \param a [in] - Function train 

    \return val [out] - sqrt( int a^2(x) dx )
***********************************************************/
double function_train_norm2(struct FunctionTrain * a)
{
    printf("in norm2\n");
    double out = function_train_inner(a,a);
    if (out < 0.0){
        fprintf(stderr, "inner product of FT with itself should not be neg %G \n",out);
        exit(1);
    }
    return sqrt(fabs(out));
}

/********************************************************//**
    Compute the L2 norm of the difference between two functions

    \param a [in] -Function train 
    \param b [in] - function train 2

    \return val - \f$ \sqrt( \int (a(x)-b(x))^2 dx ) \f$
***********************************************************/
double function_train_norm2diff(struct FunctionTrain * a, struct FunctionTrain * b)
{   
    
    struct FunctionTrain * c = function_train_copy(b);
    function_train_scale(c,-1.0);
    struct FunctionTrain * d = function_train_sum(a,c);
    printf("in function_train_norm2diff\n");
    double val = function_train_norm2(d);
    function_train_free(c);
    function_train_free(d);
    return val;
}

/********************************************************//**
    Compute the gradient of a function train 

    \param ft [in] - Function train 

    \return ftg - gradient

***********************************************************/
struct FT1DArray * function_train_gradient(struct FunctionTrain * ft)
{
    struct FT1DArray * ftg = ft1d_array_alloc(ft->dim);
    size_t ii;
    for (ii = 0; ii < ft->dim; ii++){
        //printf("**********\n\n\n**********\n");
        //printf("ii = %zu\n",ii);
        ftg->ft[ii] = function_train_copy(ft);
        qmarray_free(ftg->ft[ii]->cores[ii]);
        ftg->ft[ii]->cores[ii] = NULL;
        //printf("get deriv ii = %zu\n",ii);
        //print_qmarray(ft->cores[ii],1,NULL);
        ftg->ft[ii]->cores[ii] = qmarray_deriv(ft->cores[ii]);
        //printf("got deriv ii = %zu\n",ii);
        //print_qmarray(ftg->ft[ii]->cores[ii],1,NULL);
    }

    return ftg;
}

/********************************************************//**
    Compute the Jacobian of a Function Train 1darray

    \param fta [in] - Function train array

    \return jac - jacobian

***********************************************************/
struct FT1DArray * ft1d_array_jacobian(struct FT1DArray * fta)
{
    struct FT1DArray * jac = ft1d_array_alloc(fta->size * fta->ft[0]->dim);
    size_t ii,jj;
    for (ii = 0; ii < fta->ft[0]->dim; ii++){
        for (jj = 0; jj < fta->size; jj++){
            jac->ft[ii*fta->size+jj] = function_train_copy(fta->ft[jj]);
            qmarray_free(jac->ft[ii*fta->size+jj]->cores[ii]);
            jac->ft[ii*fta->size+jj]->cores[ii] = NULL;
            jac->ft[ii*fta->size+jj]->cores[ii] = 
                qmarray_deriv(fta->ft[jj]->cores[ii]);
            
        }
    }
    return jac;
}

/********************************************************//**
    Compute the hessian of a function train 

    \param fta [in] - Function train 

    \return fth - hessian of a function train

***********************************************************/
struct FT1DArray * function_train_hessian(struct FunctionTrain * fta)
{
    struct FT1DArray * ftg = function_train_gradient(fta);

    struct FT1DArray * fth = ft1d_array_jacobian(ftg);
        
    ft1d_array_free(ftg); ftg = NULL;
    return fth;
}

/********************************************************//**
    Scale a function train array

    \param fta [inout] - Function train Array
    \param n [in] - number of elements in the array to scale
    \param inc [in] - increment between function trains to scale in the array
    \param scale [in] - value by which to scale

***********************************************************/
void ft1d_array_scale(struct FT1DArray * fta, size_t n, size_t inc, double scale)
{
    size_t ii;
    for (ii = 0; ii < n; ii++){
        function_train_scale(fta->ft[ii*inc],scale);
    }
}

/********************************************************//**
    Evaluate a function train 1darray

    \param fta [in] - Function train array to evaluate
    \param x [in] - location at which to obtain evaluations

    \return out evaluation
***********************************************************/
double * ft1d_array_eval(struct FT1DArray * fta, double * x)
{
    double * out = calloc_double(fta->size);
    size_t ii; 
    for (ii = 0; ii < fta->size; ii++){
        out[ii] = function_train_eval(fta->ft[ii], x);
    }
    return out;
}

/********************************************************//**
    Multiply together and sum the elements of two function train arrays
    \f[ 
        out(x) = \sum_{i=1}^{N} coeff[i] f_i(x)  g_i(x) 
    \f]
    
    \param N [in] - number of function trains in each array
    \param coeff [in] - coefficients to multiply each element
    \param f [in] - first array
    \param g [in] - second array
    \param epsilon [in] - rounding accuracy

    \return out - function train
***********************************************************/
struct FunctionTrain * 
ft1d_array_sum_prod(size_t N, double * coeff, 
               struct FT1DArray * f, struct FT1DArray * g, 
               double epsilon)
{
    
    struct FunctionTrain * ft1 = NULL;
    struct FunctionTrain * ft2 = NULL;
    struct FunctionTrain * out = NULL;
    struct FunctionTrain * temp = NULL;
    
    temp = function_train_product(f->ft[0],g->ft[0]);
    function_train_scale(temp,coeff[0]);
    out = function_train_round(temp,epsilon); 
    function_train_free(temp); temp = NULL;
    size_t ii;
    for (ii = 1; ii < N; ii++){
        temp =  function_train_product(f->ft[ii],g->ft[ii]);
        function_train_scale(temp,coeff[ii]);

        ft2 = function_train_round(temp,epsilon);
        ft1 = function_train_sum(out, ft2);
    
        function_train_free(temp); temp = NULL;
        function_train_free(ft2); ft2 = NULL;
        function_train_free(out); out = NULL;

        out = function_train_round(ft1,epsilon);

        function_train_free(ft1); ft1 = NULL;
    }

    return out;
}

// utility function for function_train_cross (not in header file)
struct Qmarray *
prepCore(size_t ii, size_t nrows, double(*f)(double *, void *), void * args,
        struct BoundingBox * bd,
        struct IndexSet ** left_ind, struct IndexSet ** right_ind, 
        struct FtCrossArgs * cargs, struct FtApproxArgs * fta, int t)
{

    enum function_class fc;
    void * sub_type = NULL;
    void * approx_args = NULL;
    struct FiberCut ** fcut = NULL;  
    struct Qmarray * temp = NULL;
    double ** vals = NULL;
    size_t ncols, ncuts;
    size_t dim = bd->dim;
    
    //printf("here!|n");
    fc = ft_approx_args_getfc(fta,ii);
    sub_type = ft_approx_args_getst(fta,ii);
    approx_args = ft_approx_args_getaopts(fta, ii);
    
    //printf("t=%d\n",t);
    if (t == 1){
        ncuts = nrows;
        ncols = 1;
        vals = index_set_merge_fill_end(left_ind[ii],right_ind[ii-1]->inds); 
        fcut = fiber_cut_ndarray(f,args, dim, ii, ncuts, vals);
    }
    else if (t == -1){
        ncuts = cargs->ranks[ii+1];
        ncols = ncuts;
        vals = index_set_merge_fill_beg(left_ind[ii+1]->inds,right_ind[ii]); 
        fcut = fiber_cut_ndarray(f,args, dim, ii, ncuts, vals);
    }
    else{
        vals = index_set_merge(left_ind[ii],right_ind[ii], &ncuts); 
        fcut = fiber_cut_ndarray(f,args, dim, ii, ncuts, vals);
        ncols = ncuts / nrows;
    }
    if (VPREPCORE)
        printf("compute from fibercuts\n");
    temp = qmarray_from_fiber_cuts(nrows, ncols,
                    fiber_cut_eval, fcut, fc, sub_type,
                    bd->lb[ii],bd->ub[ii], approx_args);

    //print_qmarray(temp,0,NULL);
    if (VPREPCORE)
        printf("computed!\n");

    free_dd(ncuts,vals); vals = NULL;
    fiber_cut_array_free(ncuts, fcut); fcut = NULL;
    
    
    return temp;
}

void update_lindex(size_t ii,size_t rank, struct IndexSet ** left_ind, 
                    size_t * pivind, double * pivx)
{
    size_t jj,kk;
    for (jj = 0; jj < rank; jj++){
        left_ind[ii+1]->inds[jj][ii] = pivx[jj];
        for (kk = 0; kk < ii; kk++){
            left_ind[ii+1]->inds[jj][kk] = 
                left_ind[ii]->inds[pivind[jj]][kk];
        }
    }
}
void update_rindex(size_t ii, size_t oncore, size_t rank, 
                    struct IndexSet ** right_ind,
                    size_t * pivind, double * pivx)
{
    size_t jj, kk;
    for (jj = 0; jj < rank; jj++){
        right_ind[ii-1]->inds[jj][0] = pivx[jj];
        for (kk = 1; kk < oncore; kk++){
            right_ind[ii-1]->inds[jj][kk] = 
                        right_ind[ii]->inds[pivind[jj]][kk-1];
        }
    }
}

/***********************************************************//**
    Cross approximation of a of a dim-dimensional function

    \param f [in] - function
    \param args [in] - function arguments
    \param bd [in] - bounds on input space
    \param ftref [inout] - initial ftrain decomposition, changed in func
    \param left_ind [inout] - left indices (first element should be NULL)
    \param right_ind [inout] - right indices (last element should be NULL)
    \param cargs [in] - algorithm parameters
    \param apargs [in] - approximation arguments

    \return ft - function train decomposition of $f$

    \note
       both left and right indices are nested
***************************************************************/
struct FunctionTrain *
ftapprox_cross(double (*f)(double *, void *), void * args, 
    struct BoundingBox * bd, struct FunctionTrain * ftref, 
    struct IndexSet ** left_ind, struct IndexSet ** right_ind, 
    struct FtCrossArgs * cargs, struct FtApproxArgs * apargs)
{
    size_t dim = bd->dim;
    int info;
    size_t nrows, ii, oncore;
    struct Qmarray * temp = NULL;
    struct Qmarray * Q = NULL;
    struct Qmarray * Qt = NULL;
    double * R = NULL;
    size_t * pivind = NULL;
    double * pivx = NULL;
    double diff, den;
    
    struct FunctionTrain * ft = function_train_alloc(dim);
    memmove(ft->ranks, cargs->ranks, (dim+1)*sizeof(size_t));
    
    struct FunctionTrain * fti = function_train_copy(ftref);

    int done = 0;
    size_t iter = 0;
    while (done == 0){
        if (cargs->verbose > 0)
            printf("cross iter=%zu \n",iter);
        
        // left right sweep;
        nrows = 1; 
        for (ii = 0; ii < dim-1; ii++){
            if (cargs->verbose > 1){
                printf(" ............. on left-right sweep (%zu/%zu)\n",ii,dim-1);
            }
            //printf("ii=%zu\n",ii);
            pivind = calloc_size_t(ft->ranks[ii+1]);
            pivx = calloc_double(ft->ranks[ii+1]);
            
            if (VFTCROSS){
                printf( "prepCore \n");
                //printf( "left index set = \n");
                //print_index_set_array(dim,left_ind);
                //printf( "right index set = \n");
                //print_index_set_array(dim,right_ind);
            }
            temp = prepCore(ii,nrows,f,args,bd,left_ind,right_ind,cargs,apargs,0);

            if (VFTCROSS){
                printf ("got it \n");
                //print_qmarray(temp,0,NULL);
                struct Qmarray * tempp = qmarray_copy(temp);
                R = calloc_double(temp->ncols * temp->ncols);
                Q = qmarray_householder_simple("QR", temp,R);
                printf("R=\n");
                dprint2d_col(temp->ncols, temp->ncols, R);
                
                struct Qmarray * mult = qmam(Q,R,temp->ncols);
                //print_qmarray(Q,3,NULL);
                double difftemp = qmarray_norm2diff(mult,tempp);
                printf("difftemp = %3.15G\n",difftemp);
                qmarray_free(tempp);
                qmarray_free(mult);
            }
            else{
                R = calloc_double(temp->ncols * temp->ncols);
                Q = qmarray_householder_simple("QR", temp,R);
            }

        
            info = qmarray_maxvol1d(Q,R,pivind, pivx);

            if (VFTCROSS){
                printf( " got info=%d\n",info);
                printf("indices and pivots\n");
                iprint_sz(ft->ranks[ii+1],pivind);
                dprint(ft->ranks[ii+1],pivx);
            }

            if (info < 0){
                fprintf(stderr, "no invertible submatrix in maxvol in cross\n");
            }
            if (info > 0){
                fprintf(stderr, " error in qmarray_maxvol1d \n");
                exit(1);
            }

            update_lindex(ii,ft->ranks[ii+1], left_ind, pivind, pivx);
            
            qmarray_free(ft->cores[ii]); ft->cores[ii]=NULL;
            ft->cores[ii] = qmam(Q,R, temp->ncols);
            nrows = left_ind[ii+1]->rank;

            qmarray_free(temp); temp = NULL;
            qmarray_free(Q); Q = NULL;
            free(pivind); pivind =NULL;
            free(pivx); pivx = NULL;
            free(R); R=NULL;

        }
        ii = dim-1;
        if (cargs->verbose > 1){
            printf(" ............. on left-right sweep (%zu/%zu)\n",ii,dim-1);
        }
        qmarray_free(ft->cores[ii]); ft->cores[ii] = NULL;
        ft->cores[ii] = prepCore(ii,cargs->ranks[ii],f,args,bd,
                                    left_ind,right_ind, cargs, apargs,1);
        if (VFTCROSS){
            printf ("got it \n");
            //print_qmarray(ft->cores[ii],0,NULL);
            printf("integral = %G\n",function_train_integrate(ft));
            struct FunctionTrain * tprod = function_train_product(ft,ft);
            printf("prod integral = %G\n",function_train_integrate(tprod));
            printf("norm2 = %G\n",function_train_norm2(ft));
            print_qmarray(tprod->cores[0],0,NULL);
            //print_qmarray(tprod->cores[1],0,NULL);
            function_train_free(tprod);

        }
        den = function_train_norm2(ft);
        diff = function_train_norm2diff(ft,fti);
        if (den > ZEROTHRESH){
            diff /= den;
        }

        if (cargs->verbose > 0){
            printf("...... New FT norm L/R Sweep = %E\n",den);
            printf("...... Error L/R Sweep = %E\n",diff);
        }
        
        if (diff < cargs->epsilon){
            done = 1;
            break;
        }
        
        function_train_free(fti); fti=NULL;
        //printf("copy \n");
        fti = function_train_copy(ft);
        //printf("copied \n");
        //printf("copy diff= %G\n", function_train_norm2diff(ft,fti));
        //print_index_set_array(dim,left_ind);

        ///////////////////////////////////////////////////////
        // right-left sweep
        for (oncore = 1; oncore < dim; oncore++){
            
            ii = dim-oncore;

            if (cargs->verbose > 1){
                printf(" ............. on right_left sweep (%zu/%zu)\n",ii,dim-1);
            }

            nrows = ft->ranks[ii]; 
            
            //printf("prep core\n");
            temp = prepCore(ii,nrows,f,args,bd,left_ind,right_ind,cargs,apargs,0);
            //printf("prepped core\n");

            R = calloc_double(temp->nrows * temp->nrows);
            Q = qmarray_householder_simple("LQ", temp,R);

            Qt = qmarray_transpose(Q);

            pivind = calloc_size_t(ft->ranks[ii]);
            pivx = calloc_double(ft->ranks[ii]);
            info = qmarray_maxvol1d(Qt,R,pivind, pivx);
            
            if (VFTCROSS){
                printf("got info=%d\n",info);
                printf("indices and pivots\n");
                iprint_sz(ft->ranks[ii],pivind);
                dprint(ft->ranks[ii],pivx);
            }

            //printf("got maxvol\n");
            if (info < 0){
                fprintf(stderr, "noinvertible submatrix in maxvol in rl cross");
            }

            if (info > 0){
                fprintf(stderr, " error in qmarray_maxvol1d \n");
                exit(1);
            }
            qmarray_free(Q); Q = NULL;

            //printf("pivx \n");
            //dprint(ft->ranks[ii], pivx);

            Q = qmam(Qt,R, temp->nrows);

            qmarray_free(ft->cores[ii]); ft->cores[ii] = NULL;
            ft->cores[ii] = qmarray_transpose(Q);
            
            update_rindex(ii, oncore, ft->ranks[ii], right_ind, pivind, pivx);

            qmarray_free(temp); temp = NULL;
            qmarray_free(Q); Q = NULL;
            qmarray_free(Qt); Qt = NULL;
            free(pivind);
            free(pivx);
            free(R); R=NULL;

        }

        ii = 0;
        qmarray_free(ft->cores[ii]); ft->cores[ii] = NULL;

        if (cargs->verbose > 1)
            printf(" ............. on right_left sweep (%zu/%zu)\n",ii,dim-1);
        ft->cores[ii] = prepCore(ii,1,f,args,bd,left_ind,right_ind,cargs,apargs,-1);
        if (cargs->verbose > 1)
            printf(" ............. done with right left sweep\n");

        den = function_train_norm2(ft);
        diff = function_train_norm2diff(ft,fti);
        if (den > ZEROTHRESH){
            diff /= den;
        }

        if (cargs->verbose > 0){
            printf("...... New FT norm R/L Sweep = %E\n",den);
            printf("...... Error R/L Sweep = %E\n",diff);
        }

        if (diff < cargs->epsilon){
            done = 1;
            break;
        }

        function_train_free(fti); fti=NULL;
        fti = function_train_copy(ft);

        iter++;
        if (iter  == cargs->maxiter){
            done = 1;
            break;
        }
    }

    function_train_free(fti); fti=NULL;
    return ft;
}

/***********************************************************//**
    An interface for cross approximation of a function

    \param f [in] - function
    \param args [in] - function arguments
    \param bds [in] - bounding box 
    \param xstart [in] - location for first fibers (if null then middle of domain)
    \param fca [in] - cross approximation args, if NULL then default exists
    \param apargs [in] - function approximation arguments (if null then defaults)

    \return ft - function train decomposition of f

    \note
        Nested indices both left and right
***************************************************************/
struct FunctionTrain *
function_train_cross(double (*f)(double *, void *), void * args, 
                struct BoundingBox * bds,
                double * xstart,
                struct FtCrossArgs * fca,
                struct FtApproxArgs * apargs)
{   
    size_t dim = bds->dim;
    
    size_t * init_ranks = NULL;
    double * init_x = NULL;
    struct FtCrossArgs * fcause = NULL;
    struct FtCrossArgs temp;
    struct FtApproxArgs * fapp = NULL;
    

    double * init_coeff = darray_val(dim,1.0/ (double) dim);
    struct FunctionTrain * ftref = 
            function_train_constant(dim, 1.0, bds, NULL);
    
    size_t ii;
    if ( xstart == NULL) {
        init_x = calloc_double(dim);
        for (ii = 0; ii < dim; ii++){
            init_x[ii] = (bds->lb[ii]+bds->ub[ii])/2.0;
        }
    }
    else{
        init_x = xstart;
    }

    
    if (fca != NULL){
        fcause = fca;
    }
    else{
        size_t init_rank = 5;
        init_ranks = calloc_size_t(dim+1);
        for (ii = 0; ii < dim ;ii++){
            init_ranks[ii] = init_rank;
        }
        init_ranks[0] = 1;
        init_ranks[dim] = 1;
        

        temp.dim = dim;
        temp.ranks = init_ranks;
        temp.epsilon = 1e-5;
        temp.maxiter = 5;
        temp.verbose = 2;
        
        temp.epsround = 1e-10;
        temp.kickrank = 10;
        temp.maxiteradapt = 5;
        fcause = &temp;
    }
    
    if (apargs != NULL){
        fapp = apargs;
    }
    else{
        enum poly_type ptype = LEGENDRE;
        fapp = ft_approx_args_createpoly(dim,&ptype,NULL);
    }

    //printf("ranks 500 = %zu \n",fcause->ranks[500]);
    struct IndexSet ** isr = index_set_array_rnested(dim, fcause->ranks, init_x);
    struct IndexSet ** isl = index_set_array_lnested(dim, fcause->ranks, init_x);

    struct FunctionTrain * ft  = NULL;
    ft = ftapprox_cross_rankadapt(f,args,bds,ftref,isl,isr,init_x,
                                        fcause,fapp);
    
    /*
    index_set_array_free(dim, isr); isr = NULL;
    index_set_array_free(dim, isl); isl = NULL;
    isr = index_set_array_rnested(dim, ft->ranks, init_x);
    isl = index_set_array_lnested(dim, ft->ranks, init_x);
    memmove(fcause->ranks,ft->ranks,(dim+1)*sizeof(size_t));
    function_train_free(ft); ft = NULL;
    ft = ftapprox_cross(f,args, bds, dim, ftref, isl, isr, fcause);
    */
    if (xstart == NULL){
        free(init_x); init_x = NULL;
    }
    if (apargs == NULL){
        free(fapp); fapp = NULL;
    }
    if (fca == NULL){
        free(init_ranks); init_ranks = NULL;
    }

    function_train_free(ftref); ftref = NULL;
    index_set_array_free(dim, isr); isr = NULL;
    index_set_array_free(dim, isl); isl = NULL;
    free(init_coeff); init_coeff = NULL;
    //function_train_free(ft); ft = NULL;

    return ft;
}

/***********************************************************//**
    Cross approximation of a of a dim-dimensional function with rank adaptation

    \param f [in] - function
    \param args [in] - function arguments
    \param bds [in] - bounds on input space
    \param ftref [inout] - initial ftrain decomposition, changed in func
    \param isl [inout] - left indices (first element should be NULL)
    \param isr [inout] - right indices (last element should be NULL)
    \param xhelp [in] - values helpful to create new index sets if *fca* is NULL
    \param fca [in] - algorithm parameters, if NULL then default paramaters used
    \param apargs [in] - function approximation args 

    \return ft - function train decomposition of $f$

    \note
       both left and right indices are nested
***************************************************************/
struct FunctionTrain *
ftapprox_cross_rankadapt( double (*f)(double *, void *),
                void * args,
                struct BoundingBox * bds, 
                struct FunctionTrain * ftref, 
                struct IndexSet ** isl, 
                struct IndexSet ** isr,
                double * xhelp,
                struct FtCrossArgs * fca,
                struct FtApproxArgs * apargs)
                
{
    size_t dim = bds->dim;
    double eps = fca->epsround;
    size_t kickrank = fca->kickrank;

    struct FunctionTrain * ft = NULL;
    ft = ftapprox_cross(f, args,bds,ftref, isl, isr, fca,apargs);
    //return ft;
    if (fca->verbose > 0){
        printf("done with first cross... rounding\n");
    }
    size_t * ranks_found = calloc_size_t(dim+1);
    memmove(ranks_found,fca->ranks,(dim+1)*sizeof(size_t));
    
    struct FunctionTrain * ftc = function_train_copy(ft);
    struct FunctionTrain * ftr = function_train_round(ft, eps);
    //struct FunctionTrain * ftr = function_train_copy(ft);
    //return ftr; 
    //printf("DOOONNTT FORGET MEEE HEERREEEE \n");
    int adapt = 0;
    size_t ii;
    for (ii = 1; ii < dim; ii++){
        //printf("%zu == %zu\n",ranks_found[ii],ftr->ranks[ii]);
        if (ranks_found[ii] == ftr->ranks[ii]){
            adapt = 1;
            fca->ranks[ii] = ranks_found[ii] + kickrank;
            ranks_found[ii] = fca->ranks[ii];
        }
    }

    //printf("adapt here!\n");

    struct IndexSet ** isln = NULL;
    struct IndexSet ** isrn = NULL;
    size_t iter = 0;
    while ( (adapt == 1) && (fca->maxiteradapt>0))
    {
        adapt = 0;
        if (fca->verbose > 0){
            printf("adapting \n");
            printf("Increasing rank\n");
            iprint_sz(ft->dim+1,fca->ranks);
        }
        
        // need a better way to boost indices
        isrn = index_set_array_rnested(dim, fca->ranks, xhelp);
        isln = index_set_array_lnested(dim, fca->ranks, xhelp);

        function_train_free(ft); ft = NULL;
        ft = ftapprox_cross(f, args, bds, ftc, isln, isrn, fca,apargs);
         
        //printf("Done with adapted one\n");
        index_set_array_free(dim,isrn);
        index_set_array_free(dim,isln);

        function_train_free(ftc); ftc = NULL;
        function_train_free(ftr); ftr = NULL;

        //printf("copying\n");
        function_train_free(ftc); ftc = NULL;
        ftc = function_train_copy(ft);
        //printf("rounding\n");
        function_train_free(ftr); ftr = NULL;
        //ftr = function_train_copy(ft);//, eps);
        ftr = function_train_round(ft, eps);
        //printf("done rounding\n");
        for (ii = 1; ii < dim; ii++){
            if (ranks_found[ii] == ftr->ranks[ii]){
                adapt = 1;
                fca->ranks[ii] = ranks_found[ii] + kickrank;
                ranks_found[ii] = fca->ranks[ii];
            }
        }

        iter++;
        if (iter == fca->maxiteradapt) {
            adapt = 0;
        }
        //adapt = 0;

    }
    function_train_free(ft); ft = NULL;
    function_train_free(ftc); ftc = NULL;
    free(ranks_found);
    return ftr;
}

//////////////////////////////////////////////////////////////////////
// Blas type interface 1
//

/***********************************************************//**
    Computes 
    \f[
        y \leftarrow \texttt{round}(a x + y, epsilon)
    \f]

    \param a [in] - scaling factor
    \param x [in] - first function train
    \param y [inout] - second function train
    \param epsilon - rounding accuracy (0 for exact)
***************************************************************/
void c3axpy(double a, struct FunctionTrain * x, struct FunctionTrain ** y, 
            double epsilon)
{
    
    struct FunctionTrain * temp = function_train_copy(x);
    function_train_scale(temp,a);
    struct FunctionTrain * z = function_train_sum(temp,*y);
    function_train_free(*y); *y = NULL;
    if (epsilon > 0){
        *y = function_train_round(z,epsilon);
        function_train_free(z); z = NULL;
    }
    else{
        *y = z;
    }
    function_train_free(temp); temp = NULL;
}

/***********************************************************//**
    Computes 
    \f$
        \langle x,y \rangle
    \f$

    \param x [in] - first function train
    \param y [in] - second function train

    \return out - inner product between two function trains
***************************************************************/
double c3dot(struct FunctionTrain * x, struct FunctionTrain * y)
{
    double out = function_train_inner(x,y);
    return out;
}

//////////////////////////////////////////////////////////////////////
// Blas type interface 2
/***********************************************************//**
    Computes 
    \f[
        y \leftarrow alpha \sum_{i=1}^n \texttt{round}(\texttt{product}(A[i*inca],x),epsilon) +
         beta y 
    \f]
    \f[
        y \leftarrow \texttt{round}(y,epsilon)
    \f]
    
    \note
    Also rounds after every summation
***************************************************************/
void c3gemv(double alpha, size_t n, struct FT1DArray * A, size_t inca,
        struct FunctionTrain * x,double beta, struct FunctionTrain * y,
        double epsilon)
{

    size_t ii;
    if (y == NULL){
        struct BoundingBox * bds = function_train_bds(x);
        y = function_train_constant(x->dim,0.0, bds,NULL);
        bounding_box_free(bds);
    }
    else{
        function_train_scale(y,beta);
    }

    if (epsilon > 0){
        struct FunctionTrain * runinit = function_train_product(A->ft[0],x);
        struct FunctionTrain * run = function_train_round(runinit,epsilon);
        for (ii = 1; ii < n; ii++){
            struct FunctionTrain * temp = 
                function_train_product(A->ft[ii*inca],x);
            struct FunctionTrain * tempround = function_train_round(temp,epsilon);
            c3axpy(1.0,tempround,&run,epsilon);
            function_train_free(temp); temp = NULL;
            function_train_free(tempround); tempround = NULL;
        }
        c3axpy(alpha,run,&y,epsilon);
        function_train_free(run); run = NULL;
        function_train_free(runinit); runinit = NULL;
    }
    else{
        struct FunctionTrain * run = function_train_product(A->ft[0],x);
        for (ii = 1; ii < n; ii++){
            struct FunctionTrain * temp = 
                function_train_product(A->ft[ii*inca],x);
            c3axpy(1.0,temp,&run,epsilon);
            function_train_free(temp); temp = NULL;
        }
        c3axpy(alpha,run,&y,epsilon);
        function_train_free(run); run = NULL;
    }
       
}

//////////////////////////////////////////////////////////////////////
// Blas type interface 1 (for ft arrays

/***********************************************************//**
    Computes for \f$ i=1 \ldots n\f$
    \f[
        y[i*incy] \leftarrow \texttt{round}(a * x[i*incx] + y[i*incy],epsilon)
    \f]

***************************************************************/
void c3vaxpy(size_t n, double a, struct FT1DArray * x, size_t incx, 
            struct FT1DArray ** y, size_t incy, double epsilon)
{
    size_t ii;
    for (ii = 0; ii < n; ii++){
        c3axpy(a,x->ft[ii*incx],&((*y)->ft[ii*incy]),epsilon);
    }
}

/***********************************************************//**
    Computes for \f$ i=1 \ldots n\f$
    \f[
        z \leftarrow alpha\sum_{i=1}^n y[incy*ii]*x[ii*incx] + beta * z
    \f]

***************************************************************/
void c3vaxpy_arr(size_t n, double alpha, struct FT1DArray * x, 
                size_t incx, double * y, size_t incy, double beta,
                struct FunctionTrain ** z, double epsilon)
{
    if (*z == NULL)
    {
        struct BoundingBox * bds = function_train_bds(x->ft[0]);
        *z = function_train_constant(x->ft[0]->dim,0.0, bds,NULL);
        bounding_box_free(bds); bds = NULL;
    }
    else{
        function_train_scale(*z,beta);
    }

    size_t ii;
    for (ii = 0; ii < n; ii++){
        c3axpy(alpha*y[incy*ii],x->ft[ii*incx], z,epsilon);
    }
}

/***********************************************************//**
    Computes 
    \f[
        z \leftarrow \texttt{round}(a\sum_{i=1}^n \texttt{round}(x[i*incx]*y[i*incy],epsilon) + beta * z,epsilon)
        
    \f]
    
    \note
    Also rounds after every summation.

***************************************************************/
void c3vprodsum(size_t n, double a, struct FT1DArray * x, size_t incx,
                struct FT1DArray * y, size_t incy, double beta,
                struct FunctionTrain ** z, double epsilon)
{
    if (*z == NULL)
    {
        struct BoundingBox * bds = function_train_bds(x->ft[0]);
        *z = function_train_constant(x->ft[0]->dim,0.0, bds,NULL);
        bounding_box_free(bds); bds = NULL;
    }
    else{
        function_train_scale(*z,beta);
    }
    
    size_t ii;

    if (epsilon > 0){
        struct FunctionTrain * runinit = 
            function_train_product(x->ft[0],y->ft[0]);
        struct FunctionTrain * run = function_train_round(runinit,epsilon);
        for (ii = 1; ii < n; ii++){
            struct FunctionTrain * tempinit = 
                function_train_product(x->ft[ii*incx],y->ft[ii*incy]);
            struct FunctionTrain * temp = function_train_round(tempinit,epsilon);
            c3axpy(1.0,temp,&run,epsilon);
            function_train_free(temp); temp = NULL;
            function_train_free(tempinit); tempinit = NULL;
        }
        c3axpy(a,run,z,epsilon);
        function_train_free(run); run = NULL;
        function_train_free(runinit); runinit = NULL;
    }
    else{
        struct FunctionTrain * run = 
            function_train_product(x->ft[0],y->ft[0]);
        for (ii = 1; ii < n; ii++){
            struct FunctionTrain * temp = 
                function_train_product(x->ft[ii*incx],y->ft[ii*incy]);
            c3axpy(1.0,temp,&run,epsilon);
            function_train_free(temp); temp = NULL;
        }
        c3axpy(a,run,z,epsilon);
        function_train_free(run); run = NULL;
    }
}

/***********************************************************//**
    Computes for \f$ i = 1 \ldots m \f$
    \f[
        y[i*incy] \leftarrow alpha*\sum_{j=1}^n \texttt{product}(A[i,j*lda],x[j*incx]) + beta * y[i*incy]
    \f]
    
    \note
    Rounds with tolerance epsilon after summation and multiplication. Not shown to avoid clutter.

***************************************************************/
void c3vgemv(size_t m, size_t n, double alpha, struct FT1DArray * A, size_t lda,
        struct FT1DArray * x, size_t incx, double beta, struct FT1DArray ** y,
        size_t incy, double epsilon)
{

    size_t ii;
    if (*y == NULL){
        *y = ft1d_array_alloc(m);
        assert (incy == 1);
        struct BoundingBox * bds = function_train_bds(x->ft[0]);
        for (ii = 0; ii < m; ii++){
            (*y)->ft[ii] = function_train_constant(bds->dim,0.0, bds,NULL);
        }
        bounding_box_free(bds);
    }
    else{
        ft1d_array_scale(*y,m,incy,beta);
    }

    struct FT1DArray * run = ft1d_array_alloc(m); 
    for (ii = 0; ii < m; ii++){
        struct FT1DArray ftatemp;
        ftatemp.ft = A->ft+ii;
        c3vprodsum(n,1.0,&ftatemp,lda,x,incx,0.0,&(run->ft[ii*incy]),epsilon);
    }
    c3vaxpy(m,alpha,run,1,y,incy,epsilon);

    ft1d_array_free(run); run = NULL;
}

/***********************************************************//**
    Computes for \f$ i = 1 \ldots m \f$
    \f[
        y[i*incy] \leftarrow alpha*\sum_{j=1}^n \texttt{product}(A[i,j],B[j*incb]) + beta * y[i*incy]
    \f]
    
    \note
    Rounds with tolerance epsilon after summation and multiplication. Not shown to avoid clutter.
    trans = 0 means not to transpose A
    trans = 1 means to transpose A
***************************************************************/
void c3vgemv_arr(int trans, size_t m, size_t n, double alpha, double * A, 
        size_t lda, struct FT1DArray * B, size_t incb, double beta,
        struct FT1DArray ** y, size_t incy, double epsilon)
{
    size_t ii;
    if (*y == NULL){
        assert(incy == 1);
        *y = ft1d_array_alloc(m);
        struct BoundingBox * bds = function_train_bds(B->ft[0]);
        for (ii = 0; ii < m; ii++){
            (*y)->ft[ii] = function_train_constant(bds->dim,0.0, bds,NULL);
        }
        bounding_box_free(bds);
    }
    else{
        ft1d_array_scale(*y,m,incy,beta);
    }
    
    if (trans == 0){
        for (ii = 0; ii < m; ii++){
            c3vaxpy_arr(n,alpha,B,incb,A+ii,lda,beta,&((*y)->ft[ii*incy]),epsilon);
        }
    }
    else if (trans == 1){
        for (ii = 0; ii < m; ii++){
            c3vaxpy_arr(n,alpha,B,incb,A+ii*lda,1,beta,&((*y)->ft[ii*incy]),epsilon);
        }
    }

    
}
