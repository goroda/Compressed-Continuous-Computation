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



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "array.h"
#include "quadrature.h"
#include "legquadrules.h"
#include "linalg.h"

/***********************************************************//**
*     Compute trapezoidal rule weights for uniformly distributed quadrature 
*     points
*
*     \param[in] N - number of elements
*     \param[in] h - distance between  neighboring points
*   
*     \return weights - weights of the trapezoidal rule
*****************************************************************/
double * trap_w(size_t N, double h)
{
    size_t ii;
    double * weights = calloc_double(N);
    weights[0] = 0.5*h;
    for (ii = 1; ii < N-1; ii++){
        weights[ii] = 1.0*h;
    }
    weights[N-1] = 0.5*h;
    return weights;
}

/***********************************************************//**
*     Compute Simpson rule weights for uniformly distributed 
*     quadrature points
*
*     \param[in] N - number of elements
*     \param[in] h - distance between neighboring points
*
*     \return weights  - weights of the Simpson rule
*************************************************************/
double * simpson_w(size_t N, double h)
{
    double * weights = calloc_double(N);
    size_t ii;
    int odd = N % 2;
    weights[0] = 1.0 / 3.0 * h; 
    for (ii = 1; ii < N-1; ii++)
    {
        if (ii % 2 == 0){
            weights[ii] = 2.0 / 3.0 * h;
        }
        else{
            weights[ii] = 4.0 / 3.0 * h;
        }
    }
    if (odd == 0){
        weights[N-1] = 0.5*h;
        weights[N-2] = 1.0/3.0 * h + 0.5*h;
    }
    else{
        weights[N-1] = 1.0 / 3.0 * h;
    }
    return weights;
}

/***********************************************************//**
*     Compute the points and weights of Chebyshev-Gauss quadrature
*
*       \param[in]     N       - number of elements
*       \param[in,out] pts     - quadrature nodes (space already alloc)
*       \param[in,out] weights - weights (space alread alloc)
*
*       \return 0 if successfull
*************************************************************/
int cheb_gauss(size_t N, double * pts, double * weights){
    size_t ii;
    for (ii = 0; ii < N; ii++){
        weights[ii] = M_PI / (double) N;
        pts[ii] = cos( (2.0 * (double) (ii+1) - 1.0) * M_PI / 
                    2.0 / (double) N);
    }
    return 0;
}

/***********************************************************//**
*     Compute the points and weights of Clenshaw-Curtis quadrature
*              
*     \param[in]     N       - number of elements
*     \param[in,out] pts     - quadrature nodes (space already alloc)
*     \param[in,out] weights - weights (space alread alloc)
*************************************************************/
void clenshaw_curtis(size_t N, double * pts, double * weights){
    size_t ii, jj;
    if (N == 1){
        pts[0] = 0;
        weights[0] = 2.0;
        return;
    }
    for (ii = 1; ii < N-1; ii++){
        pts[ii] = cos( M_PI * (double)(N-1-ii) / (double) (N-1));
    }
    pts[0] = -1.0;
    if ( N % 2 == 1){
        pts[ (N-1)/2] = 0.0;
    }
    pts[N-1] = 1.0;

    double temp1, temp2;
    for (ii = 0; ii < N; ii++){
        temp1 = (double) ii * M_PI / (double)(N-1);
        weights[ii] = 1.0; 
        for (jj = 1; jj <=(N-1)/2; jj++){
            if (2 * jj  == (N-1)){
                temp2 = 1.0;
            }
            else{
                temp2 = 2.0;
            }

            weights[ii] -=  temp2 * cos (2.0 * (double) jj * temp1) / 
                                (double) ( 4 * pow(jj,2) - 1);
        }
    }

    weights[0] = weights[0] / (double) (N-1);
    for (ii = 1; ii < N-1; ii++){
        weights[ii] = 2.0 * weights[ii] / (double) (N-1);
    }
    weights[N-1] = weights[N-1] / (double) (N-1);
}

/***********************************************************//**
*     Rescale Clenshaw-Curtis quadrature to lie between [a,b]
*
*       \param[in]     N   - number of elements
*       \param[in,out] pts - quadrature nodes (space already alloc)
*       \param[in,out] wts - weights (space alread alloc)
*       \param[in]     a   - lower bound
*       \param[in]     b   - upper bound
*************************************************************/
void rescale_cc(size_t N, double * pts, double * wts, double a, double b){
    
    size_t ii;
    for (ii = 0; ii < N; ii++){
        pts[ii] = 0.5 * ((pts[ii]+1.0)*b - (pts[ii]-1.0)*a);
        wts[ii] = 0.5 * ( b- a) * wts[ii];
    }
}

/***********************************************************//**
*     Compute the points and weights of Fejer second rule 
*              
*     \param[in]     N       - number of elements
*     \param[in,out] pts     - quadrature nodes (space already alloc)
*     \param[in,out] weights - weights (space alread alloc)
*************************************************************/
void fejer2(size_t N, double * pts, double * weights){
    if (N > 2){
        double * ptemp = calloc_double(N+2);
        double * wtemp = calloc_double(N+2);

        clenshaw_curtis(N+2,ptemp,wtemp);
        memmove(pts,ptemp+1,N*sizeof(double));
        memmove(weights,wtemp+1,N*sizeof(double));
        //dprint(N,pts);
        free(ptemp); ptemp = NULL;
        free(wtemp); wtemp = NULL;
    }
    else{
        clenshaw_curtis(N,pts,weights);
    }
}
 
/***********************************************************//**
*     Compute the points and weights of Gauss-Hermite quadrature
*
*     \param[in]     N       - number of elements
*     \param[in,out] pts     - quadrature nodes (space already alloc)
*     \param[in,out] weights - weights (space alread alloc)
*
*     \return 0 if successful

*     \note
*     weight function is \f$ w(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}\f$
*************************************************************/
int gauss_hermite(size_t N, double * pts, double * weights){
    
    size_t ii;
    double * offdiag = calloc_double(N);
    double * evec = calloc_double(N*N);
    double * work = calloc_double(2*N-2);
    int info;
    for (ii = 0; ii < N; ii++){
        pts[ii] = 0.0;
        offdiag[ii] = sqrt((double)(ii+1));
    }
    int M = N;
    int M2 = N;
    dstev_("V", &M, pts, offdiag, evec, &M2, work, &info);

    if ( info > 0) {
        fprintf(stderr, "Gauss-Hermite quadrature failed because eigenvalues did not converge \n");
        exit(1);
    }
    else if ( info < 0) {
        fprintf(stderr, "The %d-th argument of dstev_ has an illegal value",info);
        exit(1);
    }
    
    for (ii = 0; ii < N; ii++){
        weights[ii] = evec[ii*N] * evec[ii*N];// * sqrt(2*M_PI);
    }
    free(offdiag); offdiag = NULL;
    free(evec); evec = NULL;
    free(work); work = NULL;
    
    return info;
}

/***********************************************************//**
*     Compute the points and weights of Gauss-Legendre quadrature
*
*     \param[in]     N       - number of elements
*     \param[in,out] pts     - quadrature nodes (space already alloc)
*     \param[in,out] weights - weights (space alread alloc)
*
*     \note
*            Here the quadrature and the points are computed 
*            assuming a weight function of .5. Thus it will differ
*            from some results online because the weights computed
*            here will be half of the weights computed with a weight
*            function of 1
*************************************************************/
void gauss_legendre(size_t N, double * pts, double * weights){
    //*
    if (N == 1){
        pts[0] = 0.0;
        weights[0] = 1.0;
    }
    else if (N < 200){
        getLegPtsWts(N,pts,weights);
        //size_t ii;
        //for (ii = 0; ii < N; ii++){
        //    weights[ii] /= 2.0;
        //}
    }
    else{
        size_t ii;
        double temp; 
        double * offdiag = calloc_double(N);
        double * evec = calloc_double(N*N);
        double * work = calloc_double(2*N-2);
        int info;
        for (ii = 0; ii < N; ii++){
            pts[ii] = 0.0;
            temp = (double)ii + 1.0;
            //temp = (double)ii;
            offdiag[ii] = temp / sqrt( (2.0 * temp + 1.0) * (2.0 * temp -1.0) );
            //offdiag[ii] = 0.5  / sqrt( 1.0 - pow(2.0 * temp, -2.0));
            //offdiag[ii] = temp / sqrt((2.0 * temp +1.0));
        }

        int M = N;
        int M2 = N;
        dstev_("V", &M, pts, offdiag, evec, &M2, work, &info);

        if ( info > 0) {
            fprintf(stderr, "Gauss-Hermite quadrature failed because eigenvalues did not converge \n");
            exit(1);
        }
        else if ( info < 0) {
            fprintf(stderr, "The %d-th argument of dstev_ has an illegal value",info);
            exit(1);
        }
        
        for (ii = 0; ii < N; ii++){
            //weights[ii] = evec[N*N-2 + ii] * evec[N*N-2 + ii];
            //weights[ii] = evec[N*N-2 + ii] * evec[N*N-2 + ii];
            //weights[ii] = 2.0* evec[ii] * evec[ii];
            weights[ii] =  evec[ii*N] * evec[ii*N];
        }
        free(offdiag); offdiag = NULL;
        free(evec); evec = NULL;
        free(work); work = NULL;
    }
}

