// Copyright (c) 2014-2016, Massachusetts Institute of Technology
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

/** \file qmarray.c
 * Provides algorithms for qmarray manipulation
 */

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "stringmanip.h"
#include "array.h"
#include "linalg.h"


/** \struct Qmarray
 * \brief Defines a matrix-valued function (a Quasimatrix array)
 * \var Qmarray::nrows
 * number of rows
 * \var Qmarray::ncols
 * number of columns
 * \var Qmarray::funcs
 * functions in column-major order 
 */
struct Qmarray {
    size_t nrows;
    size_t ncols;
    struct GenericFunction ** funcs; // fortran order
};


/***********************************************************//**
    Allocate space for a qmarray

    \param[in] nrows - number of rows of quasimatrix
    \param[in] ncols - number of cols of quasimatrix

    \return qmarray
***************************************************************/
struct Qmarray * qmarray_alloc(size_t nrows, size_t ncols){
    
    struct Qmarray * qm = NULL;
    if ( NULL == (qm = malloc(sizeof(struct Qmarray)))){
        fprintf(stderr, "failed to allocate memory for qmarray.\n");
        exit(1);
    }
    qm->nrows = nrows;
    qm->ncols = ncols;
    if ( NULL == (qm->funcs=malloc(nrows*ncols*sizeof(struct GenericFunction *)))){
        fprintf(stderr, "failed to allocate memory for qmarray.\n");
        exit(1);
    }
    size_t ii;
    for (ii = 0; ii < nrows*ncols; ii++){
        qm->funcs[ii] = NULL;
    }
    return qm;
}

/***********************************************************//**
    Free memory allocated to qmarray 

    \param[in] qm - qmarray
***************************************************************/
void qmarray_free(struct Qmarray * qm){
    
    if (qm != NULL){
        size_t ii = 0;
        for (ii = 0; ii < qm->nrows*qm->ncols; ii++){
            generic_function_free(qm->funcs[ii]);
        }
        free(qm->funcs);
        free(qm);qm=NULL;
    }
}

/***********************************************************//**
    (deep)copy qmarray

    \param[in] qm - qmarray

    \return qmo  
***************************************************************/
struct Qmarray * qmarray_copy(struct Qmarray * qm)
{
    struct Qmarray * qmo = qmarray_alloc(qm->nrows,qm->ncols);
    size_t ii;
    for (ii = 0; ii < qm->nrows*qm->ncols; ii++){
        qmo->funcs[ii] = generic_function_copy(qm->funcs[ii]);
    }

    return qmo;
}


/***********************************************************//**
    Create a Qmarray of zero functions

    \param[in] ptype - polynomial type
    \param[in] nrows - number of rows of quasimatrix
    \param[in] ncols - number of cols of quasimatrix
    \param[in] lb    - lower bound of functions
    \param[in] ub    - upper bound of functions

    \return qmarray
***************************************************************/
struct Qmarray * qmarray_zeros(enum poly_type ptype,size_t nrows,
                               size_t ncols,double lb, double ub){
    
    struct Qmarray * qm = qmarray_alloc(nrows,ncols);
    size_t ii;
    for (ii = 0; ii < nrows*ncols; ii++){
        qm->funcs[ii] = generic_function_constant(0.0,POLYNOMIAL,
                                                  &ptype,lb,ub,NULL);
    }
    return qm;
}

/********************************************************//**
*    Create a qmarray consisting of pseudo-random orth poly expansion
*   
*   \param[in] ptype    - polynomial type
*   \param[in] nrows    - number of rows
*   \param[in] ncols    - number of columns
*   \param[in] maxorder - maximum order of the polynomial
*   \param[in] lower    - lower bound of input
*   \param[in] upper    - upper bound of input
*
*   \return qm - qmarray
************************************************************/
struct Qmarray *
qmarray_poly_randu(enum poly_type ptype, 
                   size_t nrows, size_t ncols, 
                   size_t maxorder, double lower, double upper)
{
    struct Qmarray * qm = qmarray_alloc(nrows,ncols);
    size_t ii;
    for (ii = 0; ii < nrows*ncols; ii++){
        qm->funcs[ii] = generic_function_poly_randu(ptype, maxorder,
                                                    lower,upper);
    }
    return qm;
}




// getters and setters
/**********************************************************//**
    Return a function                                                           
**************************************************************/
struct GenericFunction *
qmarray_get_func(const struct Qmarray * qma, size_t row, size_t col)
{
    assert (qma != NULL);
    assert (row < qma->nrows);
    assert (col < qma->ncols);

    return qma->funcs[col*qma->nrows+row];
}

/**********************************************************//**
    Set the row of a quasimatrix array to a given quasimatrix by copying

    \param[in,out] qma - qmarray
    \param[in]     row - row to set
    \param[in]     qm  - quasimatrix to copy
**************************************************************/
void
qmarray_set_row(struct Qmarray * qma, size_t row, const struct Quasimatrix * qm)
{
    size_t ii;
    for (ii = 0; ii < qma->ncols; ii++){
        generic_function_free(qma->funcs[ii*qma->nrows+row]);
        struct GenericFunction * f = quasimatrix_get_func(qm,ii);
        qma->funcs[ii*qma->nrows+row] = generic_function_copy(f);
    }
}

/**********************************************************//**
    Extract a copy of a quasimatrix from a row of a qmarray

    \param[in] qma - qmarray
    \param[in] row - row to copy

    \return qm - quasimatrix
**************************************************************/
struct Quasimatrix *
qmarray_extract_row(const struct Qmarray * qma, size_t row)
{

    struct Quasimatrix * qm = quasimatrix_alloc(qma->ncols);
    size_t ii;
    for (ii = 0; ii < qma->ncols; ii++){
        quasimatrix_set_func(qm,qma->funcs[ii*qma->nrows+row],ii);
        /* qm->funcs[ii] = generic_function_copy(qma->funcs[ii*qma->nrows+row]); */
    }
    return qm;
}


/**********************************************************//**
    Set the column of a quasimatrix array to a given quasimatrix by copying

    \param[in,out] qma - qmarray to modify
    \param[in]     col - column to set
    \param[in]     qm  - quasimatrix to copy
**************************************************************/
void
qmarray_set_column(struct Qmarray * qma, size_t col, 
                   const struct Quasimatrix * qm)
{
    size_t ii;
    for (ii = 0; ii < qma->nrows; ii++){
        generic_function_free(qma->funcs[col*qma->nrows+ii]);
        struct GenericFunction * f = quasimatrix_get_func(qm,ii);
        qma->funcs[col*qma->nrows+ii] = generic_function_copy(f);
    }
}

/***********************************************************//**
    Function qmarray_set_column_gf

    Purpose: Set the column of a quasimatrix array 
             to a given array of generic functions

    \param[in,out] qma - qmarray
    \param[in]     col - column to set
    \param[in]     gf  - array of generic functions
***************************************************************/
void
qmarray_set_column_gf(struct Qmarray * qma, size_t col, 
                      struct GenericFunction ** gf)
{
    size_t ii;
    for (ii = 0; ii < qma->nrows; ii++){
        generic_function_free(qma->funcs[col*qma->nrows+ii]);
        qma->funcs[col*qma->nrows+ii] = generic_function_copy(gf[ii]);
    }
}

/*********************************************************//**
    Extract a copy of a quasimatrix from a column of a qmarray

    \param[in] qma - qmarray
    \param[in] col - column to copy

    \return quasimatrix
*************************************************************/
struct Quasimatrix *
qmarray_extract_column(const struct Qmarray * qma, size_t col)
{

    struct Quasimatrix * qm = quasimatrix_alloc(qma->nrows);
    size_t ii;
    for (ii = 0; ii < qma->nrows; ii++){
        quasimatrix_set_func(qm,qma->funcs[col*qma->nrows+ii],ii);
        /* qm->funcs[ii] = generic_function_copy(qma->funcs[col*qma->nrows+ii]); */
    }
    return qm;
}
