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





/** \file cross2d.c
 * Provides algorithms for cross approximation of 2d funcs
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#include "lib_funcs.h"
#include "quasimatrix.h"
#include "linalg.h"

/** \struct SkeletonDecomp
 *  \brief Defines a skeleton decomposition of a two-dimensional function \f$ f(x,y) \f$
 *  \var SkeletonDecomp::r
 *  rank of the function
 *  \var SkeletonDecomp::xqm
 *  quasimatrix representing functions of *x*
 *  \var SkeletonDecomp::yqm
 *  quasimatrix representing functions of *y*
 *  \var SkeletonDecomp::skeleton
 *  skeleton matrix
 */
struct SkeletonDecomp
{
    size_t r;
    struct Quasimatrix * xqm;
    struct Quasimatrix * yqm;
    double * skeleton; // rows are x, cols are y
};

/** \struct Cross2dargs
 *  \brief Contains arguments for cross approximation
 *  \var Cross2dargs::r 
 *  rank of approximation
 *  \var Cross2dargs::delta
 *  convergence criteria
 *  \var Cross2dargs::fclass
 *  approximation classes for each dimension
 *  \var Cross2dargs::sub_type
 *  approximation sub types for each dimension
 *  \var Cross2dargs::verbose
 *  verbosity level (0,1,2)
 * */
struct Cross2dargs
{
    size_t r;
    double delta;

    enum function_class fclass[2];
    void * approx_args [2];
    int verbose;
};

/*********************************************************//**
   Create cross2d args
**************************************************************/
struct Cross2dargs * cross2d_args_create(size_t r, double delta,
                                         enum function_class fc,
                                         int verbose)
{
    struct Cross2dargs * cargs = malloc(sizeof(struct Cross2dargs));
    assert (cargs != NULL);
    cargs->r = r;
    cargs->delta = delta;
    cargs->fclass[0] = fc;
    cargs->fclass[1] = fc;
    cargs->approx_args[0] = NULL;
    cargs->approx_args[1] = NULL;
    cargs->verbose = verbose;
    return cargs;
}

void cross2d_args_destroy(struct Cross2dargs * cargs)
{
    free(cargs);
    cargs = NULL;
}

void cross2d_args_set_approx_args(struct Cross2dargs * cargs, void * aopts)
{
    cargs->approx_args[0] = aopts;
    cargs->approx_args[1] = aopts;
}

/*********************************************************//**
    Get rank
**************************************************************/
size_t cross2d_args_get_rank(const struct Cross2dargs * cargs)
{
    assert (cargs != NULL);
    return cargs->r;
}

/*********************************************************//**
    Inner product between two functions represented in 
    skeleton format by their quasimatrices

    \param[in] A1 - x quasimatrix  of function 1
    \param[in] B1 - y quasimatrix  of function 1
    \param[in] A2 - x quasimatrix  of function 2
    \param[in] B2 - y quasimatrix  of function 2

    \return mag : \f$ int (A1B1^T)(A2B2^T) dx dy \f$
**************************************************************/
static double
skeleton_decomp_inner_help(const struct Quasimatrix * A1,
                           const struct Quasimatrix * B1,
                           const struct Quasimatrix * A2,
                           const struct Quasimatrix * B2)
{
    size_t ii, jj;
    double int1, int2;
    double mag = 0.0;
    //printf("here! %zu,%zu\n",A1->n,A2->n);
    size_t a1n = quasimatrix_get_size(A1);
    size_t a2n = quasimatrix_get_size(A2);
    struct GenericFunction *a1f,*a2f,*b1f,*b2f;
        
    for (ii = 0; ii < a1n; ii++){
        for (jj = 0; jj < a2n; jj++){
            a1f = quasimatrix_get_func(A1,ii);
            a2f = quasimatrix_get_func(A2,jj);
            b1f = quasimatrix_get_func(B1,ii);
            b2f = quasimatrix_get_func(B2,jj);
            int1 = generic_function_inner(a1f,a2f);
            //printf("int1=%3.2f\n",int1);
            int2 = generic_function_inner(b1f,b2f);
            //printf("int2=%3.2f\n",int2);
            mag += int1 * int2;
            //printf("(int1,int2) = (%3.5f,%3.5f)\n",int1,int2);
        }
    }
    //printf("mag = %3.2f \n", mag);
    
    return mag;
}

/*********************************************************//**
    Check the convergence of cross2d

    \param[in] s1 - skeleton decomp 1
    \param[in] s2 - skeleton decomp 2

    \return dist - relative distance between 
                   the two approximations
                   \f$ ||1 - 2||^2/ ||1||^2 \f$
*************************************************************/
static double
check_cross_2d_convergence(struct SkeletonDecomp * s1,
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

    double den = skeleton_decomp_inner_help(t1,s1->yqm,
                                            t1,s1->yqm);
    //printf("den = %3.5f\n",den);
    double num = skeleton_decomp_inner_help(t2,s2->yqm,t2,
                                            s2->yqm) + den - 
        2.0 * skeleton_decomp_inner_help(t1,s1->yqm,
                                         t2,s2->yqm);
    
    double dist = num/den;

    free(eye1);
    free(eye2);
    quasimatrix_free(t1);
    quasimatrix_free(t2);
    //double dist = den;
    return dist;
}


////////////////////////////////////////////////////////

/*********************************************************//**
    Allocate memory for a skeleton decomposition

    \param[in] r - rank

    \return skd - skeleton decomposition
*************************************************************/
struct SkeletonDecomp * skeleton_decomp_alloc(size_t r)
{

    struct SkeletonDecomp * skd;
    if ( NULL == (skd = malloc(sizeof(struct SkeletonDecomp)))){
        fprintf(stderr, 
                "failed to allocate memory for skeleton decomposition.\n");
        exit(1);
    }
    skd->r = r;
    skd->xqm = quasimatrix_alloc(r);
    skd->yqm = quasimatrix_alloc(r);
    skd->skeleton = calloc_double(r*r);
    return skd;
}

/*********************************************************//**
    Copy a skeleton decomposition

    \param[in] skd - skeleton decomposition
    \return snew  - copied skeleton decomposition
*************************************************************/
struct SkeletonDecomp *
skeleton_decomp_copy(const struct SkeletonDecomp * skd)
{
    struct SkeletonDecomp * snew;
    if ( NULL == (snew = malloc(sizeof(struct SkeletonDecomp)))){
        fprintf(stderr, 
                "failed to allocate memory for skeleton decomposition.\n");
        exit(1);
    }
    double * eye = calloc_double(skd->r * skd->r);
    size_t ii;
    for (ii = 0; ii < skd->r; ii++) eye[ii*skd->r +ii] = 1.0;
    snew->r = skd->r;
    snew->xqm = qmm(skd->xqm,eye, skd->r);
    snew->yqm = qmm(skd->yqm,eye, skd->r);
    snew->skeleton = calloc_double(skd->r * skd->r);

    memmove(snew->skeleton,skd->skeleton,skd->r *skd->r* sizeof(double));
    
    free(eye);eye=NULL;
    return snew;
}

/********************************************************//**
    Free memory allocated to skeleton decomposition

    \param[in,out] skd - skeleton decomposition
*************************************************************/
void skeleton_decomp_free(struct SkeletonDecomp * skd)
{
    if (skd != NULL){
        quasimatrix_free(skd->xqm);
        quasimatrix_free(skd->yqm);
        free(skd->skeleton);
        free(skd);skd=NULL;
    }
}

/********************************************************//**
    Get skeleton
*************************************************************/
double * skeleton_get_skeleton(const struct SkeletonDecomp * skd)
{
    assert (skd != NULL);
    return skd->skeleton;
}

/*********************************************************//**
    Allocate and initialize skeleton decomposition 
    with a set of pivots and a given approximation

    \param[in] f      - function to approximate
    \param[in] args   - function arguments
    \param[in] bounds - bounds on function
    \param[in] cargs  - cross2d args
    \param[in] pivx   - x pivots
    \param[in] pivy   - y pivots

    \return skeleton decomposition
*************************************************************/
struct SkeletonDecomp * 
skeleton_decomp_init2d_from_pivots(
    double (*f)(double,double,void *),
    void * args, const struct BoundingBox * bounds,
    const struct Cross2dargs * cargs,
    const double * pivx, const double * pivy)

{

    size_t r = cargs->r;
    struct SkeletonDecomp * skd = skeleton_decomp_alloc(r);

    struct FiberCut ** fx;  
    struct FiberCut ** fy;
    
    double * lb = bounding_box_get_lb(bounds);
    double * ub = bounding_box_get_ub(bounds);

    fx = fiber_cut_2darray(f,args,0,r, pivy);
    quasimatrix_free(skd->xqm);
    skd->xqm = quasimatrix_approx_from_fiber_cuts(
            r, fiber_cut_eval2d, fx, cargs->fclass[0], 
            cargs->sub_type[0],
            lb[0],ub[0], cargs->approx_args[0]);
    fiber_cut_array_free(r, fx);

    fy = fiber_cut_2darray(f,args,1,r, pivx);
    quasimatrix_free(skd->yqm);
    skd->yqm = quasimatrix_approx_from_fiber_cuts(
            r, fiber_cut_eval2d, fy, cargs->fclass[1], 
            cargs->sub_type[1],
            lb[1],ub[1], cargs->approx_args[1]);
    fiber_cut_array_free(r, fy);

    
    size_t ii,jj;
    double * cmat = calloc_double(r*r);
    for (ii = 0; ii <r; ii++){
        for (jj = 0; jj <r; jj++){
            cmat[ii * r + jj] = f(pivx[jj],pivy[ii], args);
        }
    }

    /*
    printf("cmat = ");
    dprint2d_col(r,r,cmat);
    */
    pinv(r,r,r,cmat,skd->skeleton,1e-15);

    free(cmat);cmat=NULL;
    return skd;
}

/*********************************************************//**
    Evaluate a skeleton decomposition

    \param[in] skd - skeleton decomposition
    \param[in] x   - x-location to evaluate
    \param[in] y   - y-location to evaluate

    \return out - evaluation
*************************************************************/
double skeleton_decomp_eval(const struct SkeletonDecomp * skd,
                            double x, double y)
{
    double out = 0.0;
    struct Quasimatrix * t1 = qmm(skd->xqm, skd->skeleton, skd->r);
    size_t n = quasimatrix_get_size(t1);
    size_t ii;
    struct GenericFunction *gf1, *gf2;
    for (ii = 0; ii < n; ii++){
        gf1 = quasimatrix_get_func(t1,ii);
        gf2 = quasimatrix_get_func(skd->yqm,ii);
        out += generic_function_1d_eval(gf1,x) * 
            generic_function_1d_eval(gf2,y);
    }
    quasimatrix_free(t1);
    return out;
}

/*********************************************************//**
    Cross approximation of a two dimensional function f(x,y)

    \param[in]     f        - function
    \param[in]     args     - function arguments
    \param[in]     bounds   - bounds on input space
    \param[in,out] skd_init - initial skeleton decomposition,
                              changed in func
    \param[in,out] pivx     - x values for skeleton
    \param[in,out] pivy     - y values for skeleton
    \param[in]     cargs    - algorithm parameters

    \return skd - skeleton decomposition
*************************************************************/
struct SkeletonDecomp *
cross_approx_2d(double (*f)(double, double, void *),
                void * args, 
                struct BoundingBox * bounds,
                struct SkeletonDecomp ** skd_init,
                double * pivx, double * pivy,
                struct Cross2dargs * cargs)
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

    double * lb = bounding_box_get_lb(bounds);
    double * ub = bounding_box_get_ub(bounds);

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
                r, fiber_cut_eval2d, fy, cargs->fclass[1],
                cargs->sub_type[1],
                lb[1],ub[1], cargs->approx_args[1]);
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
                r, fiber_cut_eval2d, fx, 
                cargs->fclass[0], cargs->sub_type[0],
                lb[0],ub[0], cargs->approx_args[0]);
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
