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


/** \file polynomials.c
 * Provides routines for manipulating orthogonal polynomials
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <assert.h>
#include <complex.h>

#include "futil.h"

#ifndef ZEROTHRESH
#define ZEROTHRESH  1e0 * DBL_EPSILON
#endif

#include "stringmanip.h"
#include "array.h"
#include "fft.h"
#include "lib_quadrature.h"
#include "linalg.h"
#include "legtens.h"
#include "fourier.h"

inline static double fourierortho(size_t n){
    (void) n;
    return 2.0*M_PI;
}

/********************************************************//**
*   Initialize a Fourier Basis
*
*   \return p - polynomial
*************************************************************/
struct OrthPoly * init_fourier_poly(){
    
    struct OrthPoly * p;
    if ( NULL == (p = malloc(sizeof(struct OrthPoly)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    p->ptype = FOURIER;
    p->an = NULL;
    p->bn = NULL;
    p->cn = NULL;
    
    p->lower = 0.0;
    p->upper = 2.0 * M_PI;

    p->const_term = 0.0;
    p->lin_coeff = 0.0;
    p->lin_const = 0.0;

    p->norm = fourierortho;

    return p;
}


/********************************************************//**
*   Evaluate a fourier expansion
*
*   \param[in] poly - polynomial expansion
*   \param[in] x    - location at which to evaluate
*
*   \return out - polynomial value
*************************************************************/
double fourier_expansion_eval(const struct OrthPolyExpansion * poly, double x)
{
    assert (poly != NULL);
    assert (poly->kristoffel_eval == 0);


    double x_norm = space_mapping_map(poly->space_transform, x);

    double complex val = cexp((double _Complex)I*x_norm);
    double complex out = 0.0;
    double complex basis = 1.0;
    for (size_t ii = 0; ii < poly->num_poly; ii++){        
        out = out + poly->ccoeff[ii]*basis;
        basis = basis * val;
    }

    basis = 1.0 / val;
    for (size_t ii = 1; ii < poly->num_poly; ii++){
        out = out + conj(poly->ccoeff[ii])*basis;        
        basis = basis / val;
    }
    
    return creal(out);
}

/********************************************************//**
*   Evaluate the derivative of orth normal polynomial expansion
*
*   \param[in] poly - pointer to orth poly expansion
*   \param[in] x    - location at which to evaluate
*
*
*   \return out - value of derivative
*************************************************************/
double fourier_expansion_deriv_eval(const struct OrthPolyExpansion * poly, double x)
{
    assert (poly != NULL);
    assert (poly->kristoffel_eval == 0);

    double x_norm = space_mapping_map(poly->space_transform,x);

    double complex val = cexp((double _Complex)I*x_norm);
    double complex out = 0.0;
    double complex basis = val;
    for (size_t ii = 1; ii < poly->num_poly; ii++){        
        out = out + poly->ccoeff[ii]*basis * ii * (double _Complex)I;
        basis = basis * val;
    }

    basis = 1.0 / val;
    for (size_t ii = 1; ii < poly->num_poly; ii++){
        out = out - conj(poly->ccoeff[ii])*basis * ii * (double _Complex)I;        
        basis = basis / val;
    }

    double rout = creal(out);
    rout *= space_mapping_map_deriv(poly->space_transform,x);
    return rout;
}

/********************************************************//**
   Compute an expansion for the derivtive

   \param[in] p - orthogonal polynomial expansion
   
   \return derivative
*************************************************************/
struct OrthPolyExpansion * fourier_expansion_deriv(const struct OrthPolyExpansion * p)
{
    if (p == NULL) return NULL;
    assert (p->kristoffel_eval == 0);

    double dx = space_mapping_map_deriv(p->space_transform, 0.0);
    struct OrthPolyExpansion * out = orth_poly_expansion_copy(p);
    out->ccoeff[0] = 0.0;
    for (size_t ii = 1; ii < p->num_poly; ii++){
        out->ccoeff[ii] *= (double _Complex)I*ii*dx;
    }
    return out;
}

/********************************************************//**
   Compute an expansion for the second derivative

   \param[in] p - orthogonal polynomial expansion
   
   \return derivative
*************************************************************/
struct OrthPolyExpansion * fourier_expansion_dderiv(const struct OrthPolyExpansion * p)
{
    if (p == NULL) return NULL;
    assert (p->kristoffel_eval == 0);

    double dx = space_mapping_map_deriv(p->space_transform, 0.0);
    struct OrthPolyExpansion * out = orth_poly_expansion_copy(p);
    out->ccoeff[0] = 0.0;
    for (size_t ii = 1; ii < p->num_poly; ii++){
        out->ccoeff[ii] *= -1.0*(double)(ii*ii)*dx*dx;
    }
    return out;
}

/********************************************************//**
*  Approximating a function that can take a vector of points as
*  input
*
*  \param[in,out] poly - orthogonal polynomial expansion
*  \param[in]     f    - wrapped function
*  \param[in]     opts - approximation options
*
*  \return 0 - no problems, > 0 problem
*
*  \note  Maximum quadrature limited to 200 nodes
*************************************************************/
int
fourier_expansion_approx_vec(struct OrthPolyExpansion * poly,
                             struct Fwrap * f,
                             const struct OpeOpts * opts)
{
    (void) opts;
    int return_val = 1;
    /* printf("I am here!\n"); */


    /* size_t N = poly->num_poly; */
    size_t N = (poly->num_poly-1)*2;
    /* size_t N = 8; */
    size_t nquad = N;
    double frac = 2.0*M_PI/(nquad);


    double * quad_pts = calloc_double(N);
    for (size_t ii = 0; ii < nquad; ii++){
        quad_pts[ii] = frac * ii;
    }
    
    /* printf("what!\n"); */
    double * pts = calloc_double(nquad);
    double * fvals = calloc_double(nquad);
    for (size_t ii = 0; ii < nquad; ii++){
        pts[ii] = space_mapping_map_inverse(poly->space_transform,
                                            quad_pts[ii]);
        /* pts[ii] = -M_PI + ii * frac; */
    }
    

    // Evaluate functions

    return_val = fwrap_eval(nquad,pts,fvals,f);
    if (return_val != 0){
        return return_val;
    }

    /* printf("pts = "); dprint(nquad, pts); */
    /* printf("fvals = "); dprint(nquad, fvals); */

    double complex * coeff = malloc(nquad * sizeof(double complex));
    double complex * fvc = malloc(nquad * sizeof(double complex));
    for (size_t ii = 0; ii < nquad; ii++){
        fvc[ii] = fvals[ii];
    }

    int fft_res = fft(N, fvc, 1, coeff, 1);
    if (fft_res != 0){
        fprintf(stderr, "fft is not successfull\n");
        return 1;
    }

    /* printf("\n"); */
    /* for (size_t ii = 0; ii < N; ii++){ */
    /*     fprintf(stdout, "%3.5G %3.5G\n", creal(coeff[ii]), cimag(coeff[ii])); */
    /* } */


    /* printf("copy coeffs %zu= \n", poly->num_poly); */
    /* poly->num_poly = N/2; */
    poly->ccoeff[0] = coeff[0]/N;
    for (size_t ii = 1; ii < poly->num_poly; ii++){
        poly->ccoeff[ii] = coeff[ii]/N;
        /* printf("ii = %zu %3.5G %3.5G\n", ii, creal(poly->ccoeff[ii]), */
        /*        cimag(poly->ccoeff[ii])); */
    }
    
    /* poly->num_poly = N/2; */
    /* for (size_t ii = 0; ii < poly->num_poly/2+1; ii++){ */
    /*     /\* printf("ii = %zu\n", ii); *\/ */
    /*     poly->ccoeff[ii] = coeff[ii]/N; */
    /* } */
    /* for (size_t ii = 1; ii < poly->num_poly/2; ii++){ */
    /*     poly->ccoeff[ii+poly->num_poly/2] = coeff[poly->num_poly-ii]/N; */
    /* } */

    free(pts); pts = NULL;
    free(quad_pts); quad_pts = NULL;
    free(fvals); fvals = NULL;
    free(fvc); fvc = NULL;
    free(coeff); coeff = NULL;
    return return_val;
}

/********************************************************//**
*   Integrate a fourier expansion
*
*   \param[in] poly - polynomial to integrate
*
*   \return out - integral of approximation
*************************************************************/
double fourier_integrate(const struct OrthPolyExpansion * poly)
{
    double out = 0.0;

    double m = space_mapping_map_inverse_deriv(poly->space_transform, 0.0);
    out = creal(poly->ccoeff[0] * 2.0 * M_PI);
    out = out * m;
    return out;
}

typedef struct Pair
{
    const struct OrthPolyExpansion * a;
    const struct OrthPolyExpansion * b;
} pair_t;

static int prod_eval(size_t n, const double * x, double * out, void * args)
{
    pair_t * pairs = args;
    double e1, e2;
    for (size_t ii = 0; ii < n; ii++){
        e1 = orth_poly_expansion_eval(pairs->a, x[ii]);
        e2 = orth_poly_expansion_eval(pairs->b, x[ii]);
        out[ii] = e1 * e2;
    }
    return 0;
}

/********************************************************//**
   Multiply two fourier series'

   \param[in] a - first fs
   \param[in] b - second fs

   \return c - product
*************************************************************/
struct OrthPolyExpansion *
fourier_expansion_prod(const struct OrthPolyExpansion * a,
                       const struct OrthPolyExpansion * b)
{

    /* (void) a; */
    /* (void) b; */
    /* fprintf(stderr, "HAVE NOT FINISHED IMPLEMENTING PRODUCT of fourier\n"); */
    /* exit(1); */
    double lb = a->lower_bound;
    double ub = a->upper_bound;

    struct OpeOpts * opts = ope_opts_alloc(FOURIER);
    ope_opts_set_lb(opts, lb);
    ope_opts_set_ub(opts, ub);
    size_t n = a->num_poly-1;
    /* ope_opts_set_start(opts, 2*n+1); */
    ope_opts_set_start(opts, n+1);    

    /* printf("n = %zu\n", n); */
    /* struct OrthPolyExpansion * fourier = */
    /*     orth_poly_expansion_init_from_opts(opts, 2*n+1); */
    struct OrthPolyExpansion * fourier =
        orth_poly_expansion_init_from_opts(opts, n+1);

    if (n > 0){
        struct Fwrap * fw = fwrap_create(1, "general-vec");
        pair_t pairs = {a, b};
        fwrap_set_fvec(fw, prod_eval, &pairs);
        int res = orth_poly_expansion_approx_vec(fourier, fw, opts);
        assert (res == 0);
        fwrap_destroy(fw); fw = NULL;
    }
    ope_opts_free(opts); opts = NULL;
    return fourier;
}

/********************************************************//**
*   Multiply by scalar and add two expansions
*
*   \param[in] a  - scaling factor for first polynomial
*   \param[in] x  - first polynomial
*   \param[in] y  - second polynomial
*
*   \return 0 if successfull 1 if error with allocating more space for y
*
*   \note
*       Computes z=ax+by, where x and y are polynomial expansionx
*       Requires both polynomials to have the same upper
*       and lower bounds
*
**************************************************************/
int fourier_expansion_axpy(double a, const struct OrthPolyExpansion * x,
                           struct OrthPolyExpansion * y)
{

    if (x->num_poly <= y->num_poly){
        for (size_t ii = 0; ii < x->num_poly; ii++){
            y->ccoeff[ii] += a * x->ccoeff[ii];
            if (cabs(y->ccoeff[ii]) < ZEROTHRESH){
                y->ccoeff[ii] = 0.0;
            }
        }
    }
    else{
        if (x->num_poly > y->nalloc){
            //printf("hereee\n");
            y->nalloc = x->num_poly+10;
            double complex * temp = realloc(y->ccoeff, (y->nalloc)*sizeof(double complex));
            if (temp == NULL){
                return 0;
            }
            else{
                y->ccoeff = temp;
                for (size_t ii = y->num_poly; ii < y->nalloc; ii++){
                    y->ccoeff[ii] = 0.0;
                }
            }
            //printf("finished\n");
        }
        for (size_t ii = y->num_poly; ii < x->num_poly; ii++){
            y->ccoeff[ii] = a * x->ccoeff[ii];
            if (cabs(y->ccoeff[ii]) < ZEROTHRESH){
                y->ccoeff[ii] = 0.0;
            }
        }
        for (size_t ii = 0; ii < y->num_poly; ii++){
            y->ccoeff[ii] += a * x->ccoeff[ii];
            if (cabs(y->ccoeff[ii]) < ZEROTHRESH){
                y->ccoeff[ii] = 0.0;
            }
        }
        y->num_poly = x->num_poly;
        size_t nround = y->num_poly;
        for (size_t ii = 0; ii < y->num_poly-1;ii++){
            if (cabs(y->ccoeff[y->num_poly-1-ii]) > ZEROTHRESH){
                break;
            }
            else{
                nround = nround-1;
            }
        }
        y->num_poly = nround;
    }
    
    return 0;
}


/* /\********************************************************\//\** */
/* *   Multiply by scalar and add two orthgonal  */
/* *   expansions of the same family together */
/* * */
/* *   \param[in] a - scaling factor for first polynomial */
/* *   \param[in] x - first polynomial */
/* *   \param[in] b - scaling factor for second polynomial */
/* *   \param[in] y  second polynomial */
/* * */
/* *   \return p - orthogonal polynomial expansion */
/* * */
/* *   \note  */
/* *       Computes z=ax+by, where x and y are polynomial expansionx */
/* *       Requires both polynomials to have the same upper  */
/* *       and lower bounds */
/* *    */
/* *************************************************************\/ */
/* struct OrthPolyExpansion * */
/* orth_poly_expansion_daxpby(double a, struct OrthPolyExpansion * x, */
/*                            double b, struct OrthPolyExpansion * y) */
/* { */
/*     /\* */
/*     printf("a=%G b=%G \n",a,b); */
/*     printf("x=\n"); */
/*     print_orth_poly_expansion(x,0,NULL); */
/*     printf("y=\n"); */
/*     print_orth_poly_expansion(y,0,NULL); */
/*     *\/ */
    
/*     //double diffa = cabs(a-ZEROTHRESH); */
/*     //double diffb = cabs(b-ZEROTHRESH); */
/*     size_t ii; */
/*     struct OrthPolyExpansion * p ; */
/*     //if ( (x == NULL && y != NULL) || ((diffa <= ZEROTHRESH) && (y != NULL))){ */
/*     if ( (x == NULL && y != NULL)){ */
/*         //printf("b = %G\n",b); */
/*         //if (diffb <= ZEROTHRESH){ */
/*         //    p = orth_poly_expansion_init(y->p->ptype,1,y->lower_bound, y->upper_bound); */
/*        // } */
/*        // else{     */
/*             p = orth_poly_expansion_init(y->p->ptype, */
/*                         y->num_poly, y->lower_bound, y->upper_bound); */
/*             space_mapping_free(p->space_transform); */
/*             p->space_transform = space_mapping_copy(y->space_transform); */
/*             for (ii = 0; ii < y->num_poly; ii++){ */
/*                 p->coeff[ii] = y->coeff[ii] * b; */
/*             } */
/*         //} */
/*         orth_poly_expansion_round(&p); */
/*         return p; */
/*     } */
/*     //if ( (y == NULL && x != NULL) || ((diffb <= ZEROTHRESH) && (x != NULL))){ */
/*     if ( (y == NULL && x != NULL)){ */
/*         //if (a <= ZEROTHRESH){ */
/*         //    p = orth_poly_expansion_init(x->p->ptype,1, x->lower_bound, x->upper_bound); */
/*        // } */
/*         //else{ */
/*             p = orth_poly_expansion_init(x->p->ptype, */
/*                         x->num_poly, x->lower_bound, x->upper_bound); */
/*             space_mapping_free(p->space_transform); */
/*             p->space_transform = space_mapping_copy(x->space_transform); */
/*             for (ii = 0; ii < x->num_poly; ii++){ */
/*                 p->coeff[ii] = x->coeff[ii] * a; */
/*             } */
/*         //} */
/*         orth_poly_expansion_round(&p); */
/*         return p; */
/*     } */

/*     size_t N = x->num_poly > y->num_poly ? x->num_poly : y->num_poly; */

/*     p = orth_poly_expansion_init(x->p->ptype, */
/*                     N, x->lower_bound, x->upper_bound); */
/*     space_mapping_free(p->space_transform); */
/*     p->space_transform = space_mapping_copy(x->space_transform); */
    
/*     size_t xN = x->num_poly; */
/*     size_t yN = y->num_poly; */

/*     //printf("diffa = %G, x==NULL %d\n",diffa,x==NULL); */
/*     //printf("diffb = %G, y==NULL %d\n",diffb,y==NULL); */
/*    // assert(diffa > ZEROTHRESH); */
/*    // assert(diffb > ZEROTHRESH); */
/*     if (xN > yN){ */
/*         for (ii = 0; ii < yN; ii++){ */
/*             p->coeff[ii] = x->coeff[ii]*a + y->coeff[ii]*b;            */
/*             //if ( cabs(p->coeff[ii]) < ZEROTHRESH){ */
/*             //    p->coeff[ii] = 0.0; */
/*            // } */
/*         } */
/*         for (ii = yN; ii < xN; ii++){ */
/*             p->coeff[ii] = x->coeff[ii]*a; */
/*             //if ( cabs(p->coeff[ii]) < ZEROTHRESH){ */
/*             //    p->coeff[ii] = 0.0; */
/*            // } */
/*         } */
/*     } */
/*     else{ */
/*         for (ii = 0; ii < xN; ii++){ */
/*             p->coeff[ii] = x->coeff[ii]*a + y->coeff[ii]*b;            */
/*             //if ( cabs(p->coeff[ii]) < ZEROTHRESH){ */
/*             //    p->coeff[ii] = 0.0; */
/*            // } */
/*         } */
/*         for (ii = xN; ii < yN; ii++){ */
/*             p->coeff[ii] = y->coeff[ii]*b; */
/*             //if ( cabs(p->coeff[ii]) < ZEROTHRESH){ */
/*             //    p->coeff[ii] = 0.0; */
/*             //} */
/*         } */
/*     } */

/*     orth_poly_expansion_round(&p); */
/*     return p; */
/* } */

/* //////////////////////////////////////////////////////////////////////////// */
/* // Algorithms */

/* /\********************************************************\//\** */
/* *   Obtain the real roots of a standard polynomial */
/* * */
/* *   \param[in]     p     - standard polynomial */
/* *   \param[in,out] nkeep - returns how many real roots tehre are */
/* * */
/* *   \return real_roots - real roots of a standard polynomial */
/* * */
/* *   \note */
/* *   Only roots within the bounds are returned */
/* *************************************************************\/ */
/* double * */
/* standard_poly_real_roots(struct StandardPoly * p, size_t * nkeep) */
/* { */
/*     if (p->num_poly == 1) // constant function */
/*     {    */
/*         double * real_roots = NULL; */
/*         *nkeep = 0; */
/*         return real_roots; */
/*     } */
/*     else if (p->num_poly == 2){ // linear */
/*         double root = -p->coeff[0] / p->coeff[1]; */
        
/*         if ((root > p->lower_bound) && (root < p->upper_bound)){ */
/*             *nkeep = 1; */
/*         } */
/*         else{ */
/*             *nkeep = 0; */
/*         } */
/*         double * real_roots = NULL; */
/*         if (*nkeep == 1){ */
/*             real_roots = calloc_double(1); */
/*             real_roots[0] = root; */
/*         } */
/*         return real_roots; */
/*     } */
    
/*     size_t nrows = p->num_poly-1; */
/*     //printf("coeffs = \n"); */
/*     //dprint(p->num_poly, p->coeff); */
/*     while (cabs(p->coeff[nrows]) < ZEROTHRESH ){ */
/*     //while (cabs(p->coeff[nrows]) < DBL_MIN){ */
/*         nrows--; */
/*         if (nrows == 1){ */
/*             break; */
/*         } */
/*     } */

/*     //printf("nrows left = %zu \n",  nrows); */
/*     if (nrows == 1) // linear */
/*     { */
/*         double root = -p->coeff[0] / p->coeff[1]; */
/*         if ((root > p->lower_bound) && (root < p->upper_bound)){ */
/*             *nkeep = 1; */
/*         } */
/*         else{ */
/*             *nkeep = 0; */
/*         } */
/*         double * real_roots = NULL; */
/*         if (*nkeep == 1){ */
/*             real_roots = calloc_double(1); */
/*             real_roots[0] = root; */
/*         } */
/*         return real_roots; */
/*     } */
/*     else if (nrows == 0) */
/*     { */
/*         double * real_roots = NULL; */
/*         *nkeep = 0; */
/*         return real_roots; */
/*     } */

/*     // transpose of the companion matrix */
/*     double * t_companion = calloc_double((p->num_poly-1)*(p->num_poly-1)); */
/*     size_t ii; */
    

/*    // size_t m = nrows; */
/*     t_companion[nrows-1] = -p->coeff[0]/p->coeff[nrows]; */
/*     for (ii = 1; ii < nrows; ii++){ */
/*         t_companion[ii * nrows + ii-1] = 1.0; */
/*         t_companion[ii * nrows + nrows-1] = -p->coeff[ii]/p->coeff[nrows]; */
/*     } */
/*     double * real = calloc_double(nrows); */
/*     double * img = calloc_double(nrows); */
/*     int info; */
/*     int lwork = 8 * nrows; */
/*     double * iwork = calloc_double(8 * nrows); */
/*     //double * vl; */
/*     //double * vr; */
/*     int n = nrows; */

/*     //printf("hello! n=%d \n",n); */
/*     dgeev_("N","N", &n, t_companion, &n, real, img, NULL, &n, */
/*            NULL, &n, iwork, &lwork, &info); */
    
/*     //printf("info = %d", info); */

/*     free (iwork); */
    
/*     int * keep = calloc_int(nrows); */
/*     *nkeep = 0; */
/*     // the 1e-10 is kinda hacky */
/*     for (ii = 0; ii < nrows; ii++){ */
/*         //printf("real[ii] - p->lower_bound = %G\n",real[ii]-p->lower_bound); */
/*         //printf("real root = %3.15G, imag = %G \n",real[ii],img[ii]); */
/*         //printf("lower thresh = %3.20G\n",p->lower_bound-1e-8); */
/*         //printf("zero thresh = %3.20G\n",1e-8); */
/*         //printf("upper thresh = %G\n",p->upper_bound+ZEROTHRESH); */
/*         //printf("too low? %d \n", real[ii] < (p->lower_bound-1e-8)); */
/*         if ((cabs(img[ii]) < 1e-7) &&  */
/*             (real[ii] > (p->lower_bound-1e-8)) &&  */
/*             //(real[ii] >= (p->lower_bound-1e-7)) &&  */
/*             (real[ii] < (p->upper_bound+1e-8))) { */
/*             //(real[ii] <= (p->upper_bound+1e-7))) { */
        
/*             //\* */
/*             if (real[ii] < p->lower_bound){ */
/*                 real[ii] = p->lower_bound; */
/*             } */
/*             if (real[ii] > p->upper_bound){ */
/*                 real[ii] = p->upper_bound; */
/*             } */
/*             //\*\/ */

/*             keep[ii] = 1; */
/*             *nkeep = *nkeep + 1; */
/*             //printf("keep\n"); */
/*         } */
/*         else{ */
/*             keep[ii] = 0; */
/*         } */
/*     } */
    
/*     /\* */
/*     printf("real portions roots = "); */
/*     dprint(nrows, real); */
/*     printf("imag portions roots = "); */
/*     for (ii = 0; ii < nrows; ii++) printf("%E ",img[ii]); */
/*     printf("\n"); */
/*     //dprint(nrows, img); */
/*     *\/ */

/*     double * real_roots = calloc_double(*nkeep); */
/*     size_t counter = 0; */
/*     for (ii = 0; ii < nrows; ii++){ */
/*         if (keep[ii] == 1){ */
/*             real_roots[counter] = real[ii]; */
/*             counter++; */
/*         } */

/*     } */
    
/*     free(t_companion); */
/*     free(real); */
/*     free(img); */
/*     free(keep); */

/*     return real_roots; */
/* } */

/* static int dblcompare(const void * a, const void * b) */
/* { */
/*     const double * aa = a; */
/*     const double * bb = b; */
/*     if ( *aa < *bb){ */
/*         return -1; */
/*     } */
/*     return 1; */
/* } */

/* /\********************************************************\//\** */
/* *   Obtain the real roots of a legendre polynomial expansion */
/* * */
/* *   \param[in]     p     - orthogonal polynomial expansion */
/* *   \param[in,out] nkeep - returns how many real roots tehre are */
/* * */
/* *   \return real_roots - real roots of an orthonormal polynomial expansion */
/* * */
/* *   \note */
/* *       Only roots within the bounds are returned */
/* *       Algorithm is based on eigenvalues of non-standard companion matrix from */
/* *       Roots of Polynomials Expressed in terms of orthogonal polynomials */
/* *       David Day and Louis Romero 2005 */
/* * */
/* *       Multiplying by a factor of sqrt(2*N+1) because using orthonormal, */
/* *       rather than orthogonal polynomials */
/* *************************************************************\/ */
/* double *  */
/* legendre_expansion_real_roots(struct OrthPolyExpansion * p, size_t * nkeep) */
/* { */

/*     double * real_roots = NULL; // output */
/*     *nkeep = 0; */

/*     double m = (p->upper_bound - p->lower_bound) /  */
/*             (p->p->upper - p->p->lower); */
/*     double off = p->upper_bound - m * p->p->upper; */

/*     orth_poly_expansion_round(&p); */
/*    // print_orth_poly_expansion(p,3,NULL); */
/*     //printf("last 2 = %G\n",p->coeff[p->num_poly-1]); */
/*     size_t N = p->num_poly-1; */
/*     //printf("N = %zu\n",N); */
/*     if (N == 0){ */
/*         return real_roots; */
/*     } */
/*     else if (N == 1){ */
/*         if (cabs(p->coeff[N]) <= ZEROTHRESH){ */
/*             return real_roots; */
/*         } */
/*         else{ */
/*             double root = -p->coeff[0] / p->coeff[1]; */
/*             if ( (root >= -1.0-ZEROTHRESH) && (root <= 1.0 - ZEROTHRESH)){ */
/*                 if (root <-1.0){ */
/*                     root = -1.0; */
/*                 } */
/*                 else if (root > 1.0){ */
/*                     root = 1.0; */
/*                 } */
/*                 *nkeep = 1; */
/*                 real_roots = calloc_double(1); */
/*                 real_roots[0] = m*root+off; */
/*             } */
/*         } */
/*     } */
/*     else{ */
/*         /\* printf("I am here\n"); *\/ */
/*         double * nscompanion = calloc_double(N*N); // nonstandard companion */
/*         size_t ii; */
/*         double hnn1 = - (double) (N) / (2.0 * (double) (N) - 1.0); */
/*         /\* double hnn1 = - 1.0 / p->p->an(N); *\/ */
/*         nscompanion[1] = 1.0; */
/*         /\* nscompanion[(N-1)*N] += hnn1 * p->coeff[0] / p->coeff[N]; *\/ */
/*         nscompanion[(N-1)*N] += hnn1 * p->coeff[0] / (p->coeff[N] * sqrt(2*N+1)); */
/*         for (ii = 1; ii < N-1; ii++){ */
/*             assert (cabs(p->p->bn(ii)) < 1e-14); */
/*             double in = (double) ii; */
/*             nscompanion[ii*N+ii-1] = in / ( 2.0 * in + 1.0); */
/*             nscompanion[ii*N+ii+1] = (in + 1.0) / ( 2.0 * in + 1.0); */

/*             /\* nscompanion[(N-1)*N + ii] += hnn1 * p->coeff[ii] / p->coeff[N]; *\/ */
/*             nscompanion[(N-1)*N + ii] += hnn1 * p->coeff[ii] * sqrt(2*ii+1)/ p->coeff[N] / sqrt(2*N+1); */
/*         } */
/*         nscompanion[N*N-2] += (double) (N-1) / (2.0 * (double) (N-1) + 1.0); */

        
/*         /\* nscompanion[N*N-1] += hnn1 * p->coeff[N-1] / p->coeff[N]; *\/ */
/*         nscompanion[N*N-1] += hnn1 * p->coeff[N-1] * sqrt(2*(N-1)+1)/ p->coeff[N] / sqrt(2*N+1); */
        
/*         //printf("good up to here!\n"); */
/*         //dprint2d_col(N,N,nscompanion); */

/*         int info; */
/*         double * scale = calloc_double(N); */
/*         //\* */
/*         //Balance */
/*         int ILO, IHI; */
/*         //printf("am I here? N=%zu \n",N); */
/*         //dprint(N*N,nscompanion); */
/*         dgebal_("S", (int*)&N, nscompanion, (int *)&N,&ILO,&IHI,scale,&info); */
/*         //printf("yep\n"); */
/*         if (info < 0){ */
/*             fprintf(stderr, "Calling dgebl had error in %d-th input in the legendre_expansion_real_roots function\n",info); */
/*             exit(1); */
/*         } */

/*         //printf("balanced!\n"); */
/*         //dprint2d_col(N,N,nscompanion); */

/*         //IHI = M1; */
/*         //printf("M1=%zu\n",M1); */
/*         //printf("ilo=%zu\n",ILO); */
/*         //printf("IHI=%zu\n",IHI); */
/*         //\*\/ */

/*         double * real = calloc_double(N); */
/*         double * img = calloc_double(N); */
/*         //printf("allocated eigs N = %zu\n",N); */
/*         int lwork = 8 * (int)N; */
/*         //printf("got lwork\n"); */
/*         double * iwork = calloc_double(8*N); */
/*         //printf("go here"); */

/*         //dgeev_("N","N", &N, nscompanion, &N, real, img, NULL, &N, */
/*         //        NULL, &N, iwork, &lwork, &info); */
/*         dhseqr_("E","N",(int*)&N,&ILO,&IHI,nscompanion,(int*)&N,real,img,NULL,(int*)&N,iwork,&lwork,&info); */
/*         //printf("done here"); */

/*         if (info < 0){ */
/*             fprintf(stderr, "Calling dhesqr had error in %d-th input in the legendre_expansion_real_roots function\n",info); */
/*             exit(1); */
/*         } */
/*         else if(info > 0){ */
/*             //fprintf(stderr, "Eigenvalues are still uncovered in legendre_expansion_real_roots function\n"); */
/*            // printf("coeffs are \n"); */
/*            // dprint(p->num_poly, p->coeff); */
/*            // printf("last 2 = %G\n",p->coeff[p->num_poly-1]); */
/*            // exit(1); */
/*         } */

/*       //  printf("eigenvalues \n"); */
/*         size_t * keep = calloc_size_t(N); */
/*         for (ii = 0; ii < N; ii++){ */
/*             /\* printf("(%3.15G, %3.15G)\n",real[ii],img[ii]); *\/ */
/*             if ((cabs(img[ii]) < 1e-6) && (real[ii] > -1.0-1e-12) && (real[ii] < 1.0+1e-12)){ */
/*                 if (real[ii] < -1.0){ */
/*                     real[ii] = -1.0; */
/*                 } */
/*                 else if (real[ii] > 1.0){ */
/*                     real[ii] = 1.0; */
/*                 } */
/*                 keep[ii] = 1; */
/*                 *nkeep = *nkeep + 1; */
/*             } */
/*         } */
        
        
/*         if (*nkeep > 0){ */
/*             real_roots = calloc_double(*nkeep); */
/*             size_t counter = 0; */
/*             for (ii = 0; ii < N; ii++){ */
/*                 if (keep[ii] == 1){ */
/*                     real_roots[counter] = real[ii]*m+off; */
/*                     counter++; */
/*                 } */
/*             } */
/*         } */
     

/*         free(keep); keep = NULL; */
/*         free(iwork); iwork  = NULL; */
/*         free(real); real = NULL; */
/*         free(img); img = NULL; */
/*         free(nscompanion); nscompanion = NULL; */
/*         free(scale); scale = NULL; */
/*     } */

/*     if (*nkeep > 1){ */
/*         qsort(real_roots, *nkeep, sizeof(double), dblcompare); */
/*     } */
/*     return real_roots; */
/* } */

/* /\********************************************************\//\** */
/* *   Obtain the real roots of a chebyshev polynomial expansion */
/* * */
/* *   \param[in]     p     - orthogonal polynomial expansion */
/* *   \param[in,out] nkeep - returns how many real roots tehre are */
/* * */
/* *   \return real_roots - real roots of an orthonormal polynomial expansion */
/* * */
/* *   \note */
/* *       Only roots within the bounds are returned */
/* *       Algorithm is based on eigenvalues of non-standard companion matrix from */
/* *       Roots of Polynomials Expressed in terms of orthogonal polynomials */
/* *       David Day and Louis Romero 2005 */
/* * */
/* *       Multiplying by a factor of sqrt(2*N+1) because using orthonormal, */
/* *       rather than orthogonal polynomials */
/* *************************************************************\/ */
/* double *  */
/* chebyshev_expansion_real_roots(struct OrthPolyExpansion * p, size_t * nkeep) */
/* { */
/*     /\* fprintf(stderr, "Chebyshev real_roots not finished yet\n"); *\/ */
/*     /\* exit(1); *\/ */
/*     double * real_roots = NULL; // output */
/*     *nkeep = 0; */

/*     double m = (p->upper_bound - p->lower_bound) /  (p->p->upper - p->p->lower); */
/*     double off = p->upper_bound - m * p->p->upper; */


/*     /\* printf("coeff pre truncate = "); dprint(p->num_, p->coeff); *\/ */
/*     /\* for (size_t ii = 0; ii < p->num_poly; ii++){ *\/ */
/*     /\*     if (cabs(p->coeff[ii]) < 1e-13){ *\/ */
/*     /\*         p->coeff[ii] = 0.0; *\/ */
/*     /\*     } *\/ */
/*     /\* } *\/ */
/*     orth_poly_expansion_round(&p); */
    
/*     size_t N = p->num_poly-1; */
/*     if (N == 0){ */
/*         return real_roots; */
/*     } */
/*     else if (N == 1){ */
/*         if (cabs(p->coeff[N]) <= ZEROTHRESH){ */
/*             return real_roots; */
/*         } */
/*         else{ */
/*             double root = -p->coeff[0] / p->coeff[1]; */
/*             if ( (root >= -1.0-ZEROTHRESH) && (root <= 1.0 - ZEROTHRESH)){ */
/*                 if (root <-1.0){ */
/*                     root = -1.0; */
/*                 } */
/*                 else if (root > 1.0){ */
/*                     root = 1.0; */
/*                 } */
/*                 *nkeep = 1; */
/*                 real_roots = calloc_double(1); */
/*                 real_roots[0] = m*root+off; */
/*             } */
/*         } */
/*     } */
/*     else{ */
/*         /\* printf("I am heare\n"); *\/ */
/*         /\* dprint(N+1, p->coeff); *\/ */
/*         double * nscompanion = calloc_double(N*N); // nonstandard companion */
/*         size_t ii; */

/*         double hnn1 = 0.5; */
/*         double gamma = p->coeff[N]; */
        
/*         nscompanion[1] = 1.0; */
/*         nscompanion[(N-1)*N] -= hnn1*p->coeff[0] / gamma; */
/*         for (ii = 1; ii < N-1; ii++){ */
/*             assert (cabs(p->p->bn(ii)) < 1e-14); */
            
/*             nscompanion[ii*N+ii-1] = 0.5; // ii-th column */
/*             nscompanion[ii*N+ii+1] = 0.5; */

/*             // update last column */
/*             nscompanion[(N-1)*N + ii] -= hnn1 * p->coeff[ii] / gamma; */
/*         } */
/*         nscompanion[N*N-2] += 0.5; */
/*         nscompanion[N*N-1] -= hnn1 * p->coeff[N-1] / gamma; */
        
/*         //printf("good up to here!\n"); */
/*         /\* dprint2d_col(N,N,nscompanion); *\/ */

/*         int info; */
/*         double * scale = calloc_double(N); */
/*         //\* */
/*         //Balance */
/*         int ILO, IHI; */
/*         //printf("am I here? N=%zu \n",N); */
/*         //dprint(N*N,nscompanion); */
/*         dgebal_("S", (int*)&N, nscompanion, (int *)&N,&ILO,&IHI,scale,&info); */
/*         //printf("yep\n"); */
/*         if (info < 0){ */
/*             fprintf(stderr, "Calling dgebl had error in %d-th input in the chebyshev_expansion_real_roots function\n",info); */
/*             exit(1); */
/*         } */

/*         //printf("balanced!\n"); */
/*         //dprint2d_col(N,N,nscompanion); */

/*         //IHI = M1; */
/*         //printf("M1=%zu\n",M1); */
/*         //printf("ilo=%zu\n",ILO); */
/*         //printf("IHI=%zu\n",IHI); */
/*         //\*\/ */

/*         double * real = calloc_double(N); */
/*         double * img = calloc_double(N); */
/*         //printf("allocated eigs N = %zu\n",N); */
/*         int lwork = 8 * (int)N; */
/*         //printf("got lwork\n"); */
/*         double * iwork = calloc_double(8*N); */
/*         //printf("go here"); */

/*         //dgeev_("N","N", &N, nscompanion, &N, real, img, NULL, &N, */
/*         //        NULL, &N, iwork, &lwork, &info); */
/*         dhseqr_("E","N",(int*)&N,&ILO,&IHI,nscompanion,(int*)&N,real,img,NULL,(int*)&N,iwork,&lwork,&info); */
/*         //printf("done here"); */

/*         if (info < 0){ */
/*             fprintf(stderr, "Calling dhesqr had error in %d-th input in the legendre_expansion_real_roots function\n",info); */
/*             exit(1); */
/*         } */
/*         else if(info > 0){ */
/*             //fprintf(stderr, "Eigenvalues are still uncovered in legendre_expansion_real_roots function\n"); */
/*            // printf("coeffs are \n"); */
/*            // dprint(p->num_poly, p->coeff); */
/*            // printf("last 2 = %G\n",p->coeff[p->num_poly-1]); */
/*            // exit(1); */
/*         } */

/*        /\* printf("eigenvalues \n"); *\/ */
/*         size_t * keep = calloc_size_t(N); */
/*         for (ii = 0; ii < N; ii++){ */
/*             /\* printf("(%3.15G, %3.15G)\n",real[ii],img[ii]); *\/ */
/*             if ((cabs(img[ii]) < 1e-6) && (real[ii] > -1.0-1e-12) && (real[ii] < 1.0+1e-12)){ */
/*             /\* if ((real[ii] > -1.0-1e-12) && (real[ii] < 1.0+1e-12)){                 *\/ */
/*                 if (real[ii] < -1.0){ */
/*                     real[ii] = -1.0; */
/*                 } */
/*                 else if (real[ii] > 1.0){ */
/*                     real[ii] = 1.0; */
/*                 } */
/*                 keep[ii] = 1; */
/*                 *nkeep = *nkeep + 1; */
/*             } */
/*         } */

/*         /\* printf("nkeep = %zu\n", *nkeep); *\/ */
        
/*         if (*nkeep > 0){ */
/*             real_roots = calloc_double(*nkeep); */
/*             size_t counter = 0; */
/*             for (ii = 0; ii < N; ii++){ */
/*                 if (keep[ii] == 1){ */
/*                     real_roots[counter] = real[ii]*m+off; */
/*                     counter++; */
/*                 } */
/*             } */
/*         } */
     

/*         free(keep); keep = NULL; */
/*         free(iwork); iwork  = NULL; */
/*         free(real); real = NULL; */
/*         free(img); img = NULL; */
/*         free(nscompanion); nscompanion = NULL; */
/*         free(scale); scale = NULL; */
/*     } */

/*     if (*nkeep > 1){ */
/*         qsort(real_roots, *nkeep, sizeof(double), dblcompare); */
/*     } */
/*     return real_roots; */
/* } */

/* /\********************************************************\//\** */
/* *   Obtain the real roots of a orthogonal polynomial expansion */
/* * */
/* *   \param[in] p     - orthogonal polynomial expansion */
/* *   \param[in] nkeep - returns how many real roots tehre are */
/* * */
/* *   \return real_roots - real roots of an orthonormal polynomial expansion */
/* * */
/* *   \note */
/* *       Only roots within the bounds are returned */
/* *************************************************************\/ */
/* double * */
/* orth_poly_expansion_real_roots(struct OrthPolyExpansion * p, size_t * nkeep) */
/* { */
/*     double * real_roots = NULL; */
/*     enum poly_type ptype = p->p->ptype; */
/*     switch(ptype){ */
/*     case LEGENDRE: */
/*         real_roots = legendre_expansion_real_roots(p,nkeep);    */
/*         break; */
/*     case STANDARD:         */
/*         assert (1 == 0); */
/*         //x need to convert polynomial to standard polynomial first */
/*         //real_roots = standard_poly_real_roots(sp,nkeep); */
/*         //break; */
/*     case CHEBYSHEV: */
/*         real_roots = chebyshev_expansion_real_roots(p,nkeep); */
/*         break; */
/*     case HERMITE: */
/*         assert (1 == 0); */
/*     } */
/*     return real_roots; */
/* } */

/* /\********************************************************\//\** */
/* *   Obtain the maximum of an orthogonal polynomial expansion */
/* * */
/* *   \param[in] p - orthogonal polynomial expansion */
/* *   \param[in] x - location of maximum value */
/* * */
/* *   \return maxval - maximum value */
/* *    */
/* *   \note */
/* *       if constant function then just returns the left most point */
/* *************************************************************\/ */
/* double orth_poly_expansion_max(struct OrthPolyExpansion * p, double * x) */
/* { */
    
/*     double maxval; */
/*     double tempval; */

/*     maxval = orth_poly_expansion_eval(p,p->lower_bound); */
/*     *x = p->lower_bound; */

/*     tempval = orth_poly_expansion_eval(p,p->upper_bound); */
/*     if (tempval > maxval){ */
/*         maxval = tempval; */
/*         *x = p->upper_bound; */
/*     } */
    
/*     if (p->num_poly > 2){ */
/*         size_t nroots; */
/*         struct OrthPolyExpansion * deriv = orth_poly_expansion_deriv(p); */
/*         double * roots = orth_poly_expansion_real_roots(deriv,&nroots); */
/*         if (nroots > 0){ */
/*             size_t ii; */
/*             for (ii = 0; ii < nroots; ii++){ */
/*                 tempval = orth_poly_expansion_eval(p, roots[ii]); */
/*                 if (tempval > maxval){ */
/*                     *x = roots[ii]; */
/*                     maxval = tempval; */
/*                 } */
/*             } */
/*         } */

/*         free(roots); roots = NULL; */
/*         orth_poly_expansion_free(deriv); deriv = NULL; */
/*     } */
/*     return maxval; */
/* } */

/* /\********************************************************\//\** */
/* *   Obtain the minimum of an orthogonal polynomial expansion */
/* * */
/* *   \param[in]     p - orthogonal polynomial expansion */
/* *   \param[in,out] x - location of minimum value */
/* * */
/* *   \return minval - minimum value */
/* *************************************************************\/ */
/* double orth_poly_expansion_min(struct OrthPolyExpansion * p, double * x) */
/* { */

/*     double minval; */
/*     double tempval; */

/*     minval = orth_poly_expansion_eval(p,p->lower_bound); */
/*     *x = p->lower_bound; */

/*     tempval = orth_poly_expansion_eval(p,p->upper_bound); */
/*     if (tempval < minval){ */
/*         minval = tempval; */
/*         *x = p->upper_bound; */
/*     } */
    
/*     if (p->num_poly > 2){ */
/*         size_t nroots; */
/*         struct OrthPolyExpansion * deriv = orth_poly_expansion_deriv(p); */
/*         double * roots = orth_poly_expansion_real_roots(deriv,&nroots); */
/*         if (nroots > 0){ */
/*             size_t ii; */
/*             for (ii = 0; ii < nroots; ii++){ */
/*                 tempval = orth_poly_expansion_eval(p, roots[ii]); */
/*                 if (tempval < minval){ */
/*                     *x = roots[ii]; */
/*                     minval = tempval; */
/*                 } */
/*             } */
/*         } */
/*         free(roots); roots = NULL; */
/*         orth_poly_expansion_free(deriv); deriv = NULL; */
/*     } */
/*     return minval; */
/* } */

/* /\********************************************************\//\** */
/* *   Obtain the maximum in absolute value of an orthogonal polynomial expansion */
/* * */
/* *   \param[in]     p     - orthogonal polynomial expansion */
/* *   \param[in,out] x     - location of maximum */
/* *   \param[in]     oargs - optimization arguments  */
/* *                          required for HERMITE otherwise can set NULL */
/* * */
/* *   \return maxval : maximum value (absolute value) */
/* * */
/* *   \note */
/* *       if no roots then either lower or upper bound */
/* *************************************************************\/ */
/* double orth_poly_expansion_absmax( */
/*     struct OrthPolyExpansion * p, double * x, void * oargs) */
/* { */

/*     //printf("in absmax\n"); */
/*    // print_orth_poly_expansion(p,3,NULL); */
/*     //printf("%G\n", orth_poly_expansion_norm(p)); */

/*     enum poly_type ptype = p->p->ptype; */
/*     if (oargs != NULL){ */

/*         struct c3Vector * optnodes = oargs; */
/*         double mval = cabs(orth_poly_expansion_eval(p,optnodes->elem[0])); */
/*         *x = optnodes->elem[0]; */
/*         double cval = mval; */
/*         if (ptype == HERMITE){ */
/*             mval *= exp(-pow(optnodes->elem[0],2)/2.0); */
/*         } */
/*         *x = optnodes->elem[0]; */
/*         for (size_t ii = 0; ii < optnodes->size; ii++){ */
/*             double val = cabs(orth_poly_expansion_eval(p,optnodes->elem[ii])); */
/*             double tval = val; */
/*             if (ptype == HERMITE){ */
/*                 val *= exp(-pow(optnodes->elem[ii],2)/2.0); */
/*                 //printf("ii=%zu, x = %G. val=%G, tval=%G\n",ii,optnodes->elem[ii],val,tval); */
/*             } */
/*             if (val > mval){ */
/* //                printf("min achieved\n"); */
/*                 mval = val; */
/*                 cval = tval; */
/*                 *x = optnodes->elem[ii]; */
/*             } */
/*         } */
/* //        printf("optloc=%G .... cval = %G\n",*x,cval); */
/*         return cval; */
/*     } */
/*     else if (ptype == HERMITE){ */
/*         fprintf(stderr,"Must specify optimizatino arguments\n"); */
/*         fprintf(stderr,"In the form of candidate points for \n"); */
/*         fprintf(stderr,"finding the absmax of hermite expansion\n"); */
/*         exit(1); */
        
/*     } */
/*     double maxval; */
/*     double norm = orth_poly_expansion_norm(p); */
    
/*     if (norm < ZEROTHRESH) { */
/*         *x = p->lower_bound; */
/*         maxval = 0.0; */
/*     } */
/*     else{ */
/*         //printf("nroots=%zu\n", nroots); */
/*         double tempval; */

/*         maxval = cabs(orth_poly_expansion_eval(p,p->lower_bound)); */
/*         *x = p->lower_bound; */

/*         tempval = cabs(orth_poly_expansion_eval(p,p->upper_bound)); */
/*         if (tempval > maxval){ */
/*             maxval = tempval; */
/*             *x = p->upper_bound; */
/*         } */
/*         if (p->num_poly > 2){ */
/*             size_t nroots; */
/*             struct OrthPolyExpansion * deriv = orth_poly_expansion_deriv(p); */
/*             double * roots = orth_poly_expansion_real_roots(deriv,&nroots); */
/*             if (nroots > 0){ */
/*                 size_t ii; */
/*                 for (ii = 0; ii < nroots; ii++){ */
/*                     tempval = cabs(orth_poly_expansion_eval(p, roots[ii])); */
/*                     if (tempval > maxval){ */
/*                         *x = roots[ii]; */
/*                         maxval = tempval; */
/*                     } */
/*                 } */
/*             } */

/*             free(roots); roots = NULL; */
/*             orth_poly_expansion_free(deriv); deriv = NULL; */
/*         } */
/*     } */
/*     //printf("done\n"); */
/*     return maxval; */
/* } */



