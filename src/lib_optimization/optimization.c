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


/** \file optimization.c
 * Provides routines for optimization
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "array.h"
#include "linalg.h"
#include "optimization.h"

/***********************************************************//**
    Minimization using Newtons method

    \param[in,out] start - starting point and resulting solution
    \param[in] dim - dimension
    \param[in] step_size - usually 1.0 
    \param[in] tol  - absolute tolerance for gradient convergence
    \param[in] g  - function for gradient evaluation
    \param[in] h  - function for hessian evaluation
    \param[in] args - arguments to gradient and hessian
***************************************************************/
void
newton(double ** start, size_t dim, double step_size, double tol,
        double * (*g)(double *, void *),
        double * (*h)(double *, void *), void * args)
{
        
    int done = 0;
    double * p = calloc_double(dim);
    size_t * piv = calloc_size_t(dim);
    size_t one = 1;
    double diff;
    //double diffrel;
    double den;
    size_t ii;
    while (done == 0)
    {
        double * grad = g(*start,args);
        double * hess = h(*start, args);
        int info;
        dgesv_(&dim,&one,hess,&dim,piv,grad,&dim, &info);
        assert(info == 0);
        
        diff = 0.0;
        //diffrel = 0.0;
        den = 0.0;
        for (ii = 0; ii < dim; ii++){
            //printf("sol[%zu]=%G\n",ii,grad[ii]);
            den += pow((*start)[ii],2);
            (*start)[ii] = (*start)[ii] - step_size*grad[ii];
            diff += pow(step_size*grad[ii],2);
        }
        diff = sqrt(diff);
        /*
        if (den > 1e-15)
            diffrel = diff / sqrt(den);
        else{
            diffrel = den;
        }
        */
        
        if (diff < tol){
            done = 1;
        }
        free(grad); grad = NULL;
        free(hess); hess = NULL;
    }
    free(piv); piv = NULL;
    free(p); p = NULL;
}

/***********************************************************//**
    Projected Gradient damped newton

    \param[in]     d          - dimension
    \param[in]     lb         - lower bounds of box constraints
    \param[in]     ub         - upper bounds of box constraints
    \param[in,out] x          - starting/final point
    \param[in,out] fval       - final function value
    \param[in,out] grad       - gradient at final point
    \param[in,out] hess       - hessian at final point
    \param[in,out] space      - allocated space for computation 
                                (size (2*d) array)
    \param[in]     f          - objective function
    \param[in]     fargs      - arguments to object function
    \param[in]     g          - gradient of objective function
    \param[in]     gargs      - arguments to gradient function
    \param[in]     h          - hessian 
    \param[in]     hargs      - arguments to hessian
    \param[in]     tol        - convergence tolerance for 
                                distance between iterates \f$x\f$
    \param[in]     maxiter    - maximum number of iterations
    \param[in]     maxsubiter - maximum number of backtrack
                                iterations
    \param[in]     alpha      - backtrack parameter (0,0.5)
    \param[in]     beta       - backtrack paremeter (0,1.0)
    \param[in]     verbose    - 0: np output, 1: output

    \return  0    - success
            -20+? - failure in gradient (gradient outputs ?)
            -30+? - failure in hessian (hessian outputs ?)
             1    - maximum number of iterations reached
         other    - error in backward_line_search 
                   (see that function)
    \note See Convex Optimization 
          (boyd and Vandenberghe) page 466
***************************************************************/
int box_damp_newton(size_t d, double * lb, double * ub,
                    double * x, double * fval, double * grad,
                    double * hess,
                    double * space,
                    double (*f)(double*,void*),void * fargs,
                    int (*g)(double*,double*,void*),void* gargs,
                    int (*h)(double*,double*,void*),void* hargs,
                    double tol,size_t maxiter,size_t maxsubiter,
                    double alpha, double beta, int verbose)
{
    *fval = f(x,fargs);
    
    int res = g(x,grad,gargs);
    if (res != 0){
        return -20+res;
    }
    res = h(x,hess,hargs);
    if (res !=0 ){
        return -30+res;
    }
    
//    printf("here here\n");
    int one=1,info;
    size_t * piv = calloc_size_t(d);
    memmove(space+d,grad,d*sizeof(double));
    dgesv_(&d,&one,hess,&d,piv,space+d,&d,&info);
    free(piv); piv = NULL;
    assert(info == 0);

    double eta = cblas_ddot(d,grad,1,space+d,1);
    if (verbose > 0){
        printf("Iteration:0 (fval,||g||) = (%3.5G,%3.5G)\n",*fval,eta);
    }
//    printf("eta = %G\n",eta);
    if ( (eta*eta/2.0) < tol){
        return 0;
    }

    size_t iter = 1;
    double fvaltemp;
    int onbound = 0;
    while (eta > tol){
        memmove(space,x,d*sizeof(double));
        for (size_t ii = 0; ii <d; ii++){ space[d+ii] *= -1.0;}
        
        fvaltemp = *fval;
        res = backtrack_line_search_bc(d,lb,ub,space,
                                       fvaltemp,space+d,
                                       x,fval,grad,alpha,
                                       beta,f,fargs,maxsubiter);
//        printf("%G,%G\n",x[0],x[1]);
        
        if (res != 0){
            return res;
        }
        iter += 1;
        if (iter > maxiter){
            //          printf("Warning: max iter in newton method reached\n");
//            printf("on bound = %d\n",onbound);
            return 1;
        }
        res = g(x,grad,gargs);
        if (res != 0){
            return -20 + res;
        }
        res = h(x,hess,hargs);
        if (res != 0){
            return -30 + res;
        }

        memmove(space+d,grad,d*sizeof(double));
        piv = calloc_size_t(d);
        dgesv_(&d,&one,hess,&d,piv,space+d,&d,&info);
        free(piv); piv = NULL;
        assert(info == 0);

        eta = cblas_ddot(d,grad,1,space+d,1);
        onbound = 0;
        for (size_t ii = 0; ii < d; ii++){
            if ((x[ii] <= lb[ii]) || (x[ii] >= ub[ii])){
                onbound = 1;
                break;
            }
        }
        if (onbound == 1){
            eta = 0.0;
            for (size_t ii = 0; ii < d; ii++){
                eta += pow(x[ii]-space[ii],2);
            }
//            eta = sqrt(eta);
        }

        if (verbose > 0){
            printf("Iteration:%zu (fval,||g||) = (%3.5G,%3.5G)\n",iter,*fval,eta);
            printf("\t Onbound = %d\n",onbound);
        }
        
    }
    free(piv); piv = NULL;
    return 0;
}



/***********************************************************//**
    Gradient descent (with inexact backtracking)

    \param[in]     d          - dimension
    \param[in,out] x          - starting/final point
    \param[in,out] fval       - final function value
    \param[in,out] grad       - gradient at final point
    \param[in,out] space      - allocated space for computation (size (2*d) array)
    \param[in]     f          - objective function
    \param[in]     fargs      - arguments to object function
    \param[in]     g          - gradient of objective function
    \param[in]     gargs      - arguments to gradient function
    \param[in]     tol        - convergence tolerance for norm of gradient
    \param[in]     maxiter    - maximum number of iterations
    \param[in]     maxsubiter - maximum number of backtrack iterations
    \param[in]     alpha      - backtrack parameter (0,0.5)
    \param[in]     beta       - backtrack paremeter (0,1.0)
    \param[in]     verbose    - 0: np output, 1: output

    \return  0 - success
            -20+? - failure in gradient (gradient function outputs ?)
             1 - maximum number of iterations reached
         other - error in backward_line_search (see that function)
    \note See Convex Optimization (boyd and Vandenberghe) page 466
***************************************************************/
int gradient_descent(size_t d, double * x, double * fval, double * grad,
                     double * space,
                     double (*f)(double *,void*),void * fargs,
                     int (*g)(double *,double*,void*), void * gargs,
                     double tol,size_t maxiter, size_t maxsubiter,
                     double alpha, double beta, int verbose)
{
    int res = g(x,grad,gargs);
    if (res != 0){
        return -20+res;
    }
    *fval = f(x,fargs);
    double eta = sqrt(cblas_ddot(d,grad,1,grad,1));
    if (verbose > 0){
        printf("Iteration:%d (fval,||g||) = (%G,%G)\n",1,*fval,eta);
    }
    if (eta < tol){
        return 0;
    }
    size_t iter = 1;
    double fvaltemp;
    while (eta > tol){
        
        // copy current point
        memmove(space,x,d*sizeof(double));
        // create current search direction (-grad)
        memmove(space+d,grad,d*sizeof(double));
        for (size_t ii = 0; ii <d; ii++){ space[d+ii] *= -1.0;}
        
        fvaltemp = *fval;
        res = backtrack_line_search(d,space,fvaltemp,space+d,
                                    x,fval,grad,alpha,
                                    beta,f,fargs,maxsubiter);

        if (res != 0){
            return res;
        }
        iter += 1;
        if (iter > maxiter){
            printf("Warning: max number of iterations (%zu) of gradient descent reached\n",iter);
            return 1;
        }
        res = g(x,grad,gargs);
        if (res != 0){
            return -20 + res;
        }
        eta = sqrt(cblas_ddot(d,grad,1,grad,1));
        if (verbose > 0){
            printf("Iteration:%zu (fval,||g||) = (%G,%G)\n",iter,*fval,eta);
        }
    }
    
    return 0;
}

/***********************************************************//**
    Box constrained Gradient descent (with inexact backtracking 
    projected gradient linesearch) 

    \param[in]     d          - dimension
    \param[in]     lb         - lower bounds
    \param[in]     ub         - upper bounds
    \param[in,out] x          - starting/final point
    \param[in,out] fval       - final function value
    \param[in,out] grad       - gradient at final point
    \param[in,out] space      - allocated space for computation (size (2*d) array)
    \param[in]     f          - objective function
    \param[in]     fargs      - arguments to object function
    \param[in]     g          - gradient of objective function
    \param[in]     gargs      - arguments to gradient function
    \param[in]     tol        - convergence tolerance for iterates
    \param[in]     maxiter    - maximum number of iterations
    \param[in]     maxsubiter - maximum number of backtrack iterations
    \param[in]     alpha      - backtrack parameter (0,0.5)
    \param[in]     beta       - backtrack paremeter (0,1.0)
    \param[in]     verbose    - 0: np output, 1: output

    \return  0 - success
            -20+? - failure in gradient (gradient function outputs ?)
             1 - maximum number of iterations reached
         other - error in backward_line_search (see that function)

***************************************************************/
int box_pg_descent(size_t d, double * lb, double * ub,
                   double * x, double * fval, double * grad,
                   double * space,
                   double (*f)(double *,void*),void * fargs,
                   int (*g)(double *,double*,void*), 
                   void * gargs,
                   double tol,size_t maxiter, size_t maxsubiter,
                   double alpha, double beta, int verbose)
{
    int res = g(x,grad,gargs);
    if (res != 0){
        return -20+res;
    }
    *fval = f(x,fargs);
    double eta = sqrt(cblas_ddot(d,grad,1,grad,1));
    if (verbose > 0){
        printf("Iteration:%d (fval,||g||) = (%G,%G)\n",1,*fval,eta);
    }
    if (eta < tol){
        return 0;
    }
    size_t iter = 1;
    double fvaltemp;
    while (eta > tol){
        
        // copy current point
        memmove(space,x,d*sizeof(double));
        // create current search direction (-grad)
        memmove(space+d,grad,d*sizeof(double));
        for (size_t ii = 0; ii <d; ii++){ space[d+ii] *= -1.0;}
        
        fvaltemp = *fval;
        res = backtrack_line_search_bc(d,lb,ub,space,fvaltemp,
                                       space+d,x,fval,grad,alpha,
                                       beta,f,fargs,maxsubiter);

//        printf("newx = (%G,%G)\n",x[0],x[1]);

        if (res != 0){
            printf("Warning: backtrack return %d\n",res);
            return res;
        }
        iter += 1;
        if (iter > maxiter){
            printf("Warning: max number of iterations (%zu) of gradient descent reached\n",iter);
            return 1;
        }
        res = g(x,grad,gargs);
        if (res != 0){
            return -1;
        }
        
        eta = 0.0;
        for (size_t ii = 0; ii < d; ii++){
            eta += pow(x[ii]-space[ii],2);
        }
        eta = sqrt(eta);
        if (verbose > 0){
            printf("Iteration:%zu (fval,||x_k-x_k-1||) = (%G,%G)\n",iter,*fval,eta);
        }
    }
    
    return 0;
}

/***********************************************************//**
    Backtracking Line search                                                           

    \param[in]     d       - dimension
    \param[in]     x       - base point
    \param[in]     fx      - objective function value at x
    \param[in]     p       - search direction
    \param[in,out] newx    - new location (x + t*p)
    \param[in,out] fnx     - objective function value f(newx)
    \param[in]     grad    - gradient at x
    \param[in]     alpha   - specifies accepted decrease (0,0.5) typically (0.01 - 0.3)
    \param[in]     beta    - (0,1) typically (0.1, 0.8) corresponding to (crude search, less crude)
    \param[in]     f       - objective function
    \param[in]     fargs   - arguments to objective function
    \param[in]     maxiter - maximum number of iterations

    \return  0 - success
            -1 - alpha not within correct bounds
            -2 - beta not within correct bounds
             1 - maximum number of iterations reached
    \note See Convex Optimization (boyd and Vandenberghe) page 464
***************************************************************/
int backtrack_line_search(size_t d, double * x, double fx, double * p, double * newx, 
                          double * fnx,
                          double * grad, double alpha, double beta, 
                          double (*f)(double * x, void * args), void * fargs, 
                          size_t maxiter)
{
    if ((alpha <= 0.0) || (alpha >= 0.5)){
        printf("line search alpha (%G) is not (0,0.5)\n",alpha);
        return -1;
    }
    if ((beta <= 0.0) && (beta >= 1.0)) {
        printf("line search beta (%G) is not (0,1)\n",beta);
        return -2;
    }

    double t = 1.0;
    memmove(newx,x,d*sizeof(double));

    size_t iter = 1;
    
    double dg = cblas_ddot(d,grad,1,p,1);
    double checkval = fx + alpha*t*dg;
    cblas_daxpy(d,t,p,1,newx,1);
    *fnx = f(newx,fargs);
    iter += 1;
    
//    printf("newx (%G,%G)\n",newx[0],newx[1]);
    int res = 0;
    while (*fnx > checkval){
        if (iter >= maxiter){
            res = 1;
            printf("Warning: maximum number of iterations (%zu) of line search reached\n",iter);
            break;
        }
        t = beta * t;

        checkval = fx + alpha*t*dg;
       
        memmove(newx,x,d*sizeof(double));
        cblas_daxpy(d,t,p,1,newx,1);
        *fnx = f(newx,fargs);
        iter += 1 ;
    }
    
    return res;
}

/***********************************************************//**
    Backtracking Line search with projection onto box constraints

    \param[in]     d       - dimension
    \param[in]     lb      - lower bounds for each dimension
    \param[in]     ub      - upper bounds for each dimension
    \param[in]     x       - base point
    \param[in]     fx      - objective function value at x
    \param[in]     p       - search direction
    \param[in,out] newx    - new location (x + t*p)
    \param[in,out] fnx     - objective function value f(newx)
    \param[in]     grad    - gradient at x
    \param[in]     alpha   - specifies accepted decrease (0,0.5) typically (0.01 - 0.3)
    \param[in]     beta    - (0,1) typically (0.1, 0.8) corresponding to (crude search, less crude)
    \param[in]     f       - objective function
    \param[in]     fargs   - arguments to objective function
    \param[in]     maxiter - maximum number of iterations

    \return  0 - success
            -1 - alpha not within correct bounds
            -2 - beta not within correct bounds
             1 - maximum number of iterations reached
    \note See Convex Optimization (boyd and Vandenberghe) page 464
***************************************************************/
int backtrack_line_search_bc(size_t d, double * lb, double * ub,
                             double * x, double fx, double * p, double * newx, 
                             double * fnx,
                             double * grad, double alpha, double beta, 
                             double (*f)(double * x, void * args), void * fargs, 
                             size_t maxiter)
{
    if ((alpha <= 0.0) || (alpha >= 0.5)){
        printf("line search alpha (%G) is not (0,0.5)\n",alpha);
        return -1;
    }
    if ((beta <= 0.0) && (beta >= 1.0)) {
        printf("line search beta (%G) is not (0,1)\n",beta);
        return -2;
    }
    for (size_t ii = 0; ii < d; ii++){
        if (x[ii] < lb[ii]){
            printf("line search starting point violates constraints\n");
            return -3;
        }
        if (x[ii] > ub[ii]){
            printf("line search starting point violates constraints\n");
            return -3;
        }
    }

    double t = 1.0;
    memmove(newx,x,d*sizeof(double));

    size_t iter = 1;
    
    double dg = cblas_ddot(d,grad,1,p,1);
    double checkval = fx + alpha*t*dg;
    cblas_daxpy(d,t,p,1,newx,1);
    for (size_t ii = 0; ii < d; ii++){
        if (newx[ii] < lb[ii]){
            newx[ii] = lb[ii];
        }
        if (newx[ii] > ub[ii]){
            newx[ii] = ub[ii];
        }
    }
    *fnx = f(newx,fargs);
    iter += 1;
    
//    printf("newx (%G,%G)\n",newx[0],newx[1]);
    int res = 0;
    while (*fnx > checkval){
        if (iter >= maxiter){
            res = 1;
            printf("Warning: maximum number of iterations (%zu) of line search reached\n",iter);
            break;
        }
        t = beta * t;

        checkval = fx + alpha*t*dg;
       
        memmove(newx,x,d*sizeof(double));
        cblas_daxpy(d,t,p,1,newx,1);
        for (size_t ii = 0; ii < d; ii++){
            if (newx[ii] < lb[ii]){
                newx[ii] = lb[ii];
            }
            if (newx[ii] > ub[ii]){
                newx[ii] = ub[ii];
            }
        }
        *fnx = f(newx,fargs);
        iter += 1 ;
    }
    
    return res;
}
