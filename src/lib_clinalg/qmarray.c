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
#include "lib_funcs.h"

#include "quasimatrix.h"
#include "qmarray.h"

/* #define ZEROTHRESH 1e2*DBL_EPSILON */
/* #define ZERO 1e2*DBL_EPSILON */

/* #define ZEROTHRESH 1e0*DBL_EPSILON */
/* #define ZERO 1e0*DBL_EPSILON */

#define ZEROTHRESH 1e2*DBL_EPSILON
#define ZERO 1e2*DBL_EPSILON

/* #define ZEROTHRESH 1e-200 */
/* #define ZERO 1e-200 */

#ifndef VQMALU
    #define VQMALU 0
#endif

#ifndef VQMAMAXVOL
    #define VQMAMAXVOL 0
#endif

#ifndef VQMAHOUSEHOLDER
    #define VQMAHOUSEHOLDER 0
#endif

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
struct Qmarray * qmarray_copy(const struct Qmarray * qm)
{
    struct Qmarray * qmo = qmarray_alloc(qm->nrows,qm->ncols);
    size_t ii;
    for (ii = 0; ii < qm->nrows*qm->ncols; ii++){
        qmo->funcs[ii] = generic_function_copy(qm->funcs[ii]);
    }

    return qmo;
}

/***********************************************************//**
    Perform an elementwise operator on a qmarray

    \param[in] qm - qmarray
    \param[in] op - operator

    \return qmo  
***************************************************************/
struct Qmarray * qmarray_operate_elements(const struct Qmarray * qm,
                                          const struct Operator * op)
{
    struct Qmarray * qmo = qmarray_alloc(qm->nrows,qm->ncols);
    if (op != NULL){
        for (size_t ii = 0; ii < qm->nrows*qm->ncols; ii++){
            qmo->funcs[ii] = op->f(qm->funcs[ii], op->opts);
        }
    }
    else{
        for (size_t ii = 0; ii < qm->nrows*qm->ncols; ii++){
            qmo->funcs[ii] = generic_function_copy(qm->funcs[ii]);
        }
    }

    return qmo;
}

/***********************************************************//**
    Serialize a qmarray

    \param[in,out] ser       - stream to serialize to
    \param[in]     qma       - quasimatrix array
    \param[in,out] totSizeIn - if NULL then serialize, if not NULL
                               then return size;

    \return ser shifted by number of bytes
***************************************************************/
unsigned char * 
qmarray_serialize(unsigned char * ser, const struct Qmarray * qma, 
                  size_t *totSizeIn)
{

    // nrows -> ncols -> func -> func-> ... -> func
    size_t ii;
    size_t totSize = 2*sizeof(size_t);
    size_t size_temp;
    for (ii = 0; ii < qma->nrows * qma->ncols; ii++){
        serialize_generic_function(NULL,qma->funcs[ii],&size_temp);
        totSize += size_temp;
    }
    if (totSizeIn != NULL){
        *totSizeIn = totSize;
        return ser;
    }
    
    //printf("in serializing qmarray\n");
    unsigned char * ptr = ser;
    ptr = serialize_size_t(ptr, qma->nrows);
    ptr = serialize_size_t(ptr, qma->ncols);
    for (ii = 0; ii < qma->nrows * qma->ncols; ii++){
       /* printf("on function number (%zu/%zu)\n",ii+1,qma->nrows * qma->ncols); */
       // print_generic_function(qma->funcs[ii],3,NULL);
        ptr = serialize_generic_function(ptr, qma->funcs[ii],NULL);
    }
    return ptr;
}

/***********************************************************//**
    Deserialize a qmarray

    \param[in]     ser - serialized qmarray
    \param[in,out] qma - qmarray

    \return ptr - shifted ser after deserialization
***************************************************************/
unsigned char *
qmarray_deserialize(unsigned char * ser, struct Qmarray ** qma)
{
    unsigned char * ptr = ser;

    size_t nrows, ncols;
    ptr = deserialize_size_t(ptr, &nrows);
    ptr = deserialize_size_t(ptr, &ncols);
    /* printf("deserialized rows,cols = (%zu,%zu)\n",nrows,ncols); */
    
    *qma = qmarray_alloc(nrows, ncols);

    size_t ii;
    for (ii = 0; ii < nrows * ncols; ii++){
        ptr = deserialize_generic_function(ptr, &((*qma)->funcs[ii]));
    }
    
    return ptr;
}

/***********************************************************//**
    Save a qmarray in text format

    \param[in]     qma  - quasimatrix array
    \param[in,out] fp   - stream to which to write
    \param[in]     prec - precision with which to print

***************************************************************/
void qmarray_savetxt(const struct Qmarray * qma, FILE * fp, size_t prec)
{

    // nrows -> ncols -> func -> func-> ... -> func
    
    assert (qma != NULL);
    fprintf(fp,"%zu ",qma->nrows);
    fprintf(fp,"%zu ",qma->ncols);
    for (size_t ii = 0; ii < qma->nrows * qma->ncols; ii++){
        generic_function_savetxt(qma->funcs[ii],fp,prec);
    }
}

/***********************************************************//**
    Load a qmarray saved in text format

    \param[in,out] fp - stream to save to

    \return qmarray
***************************************************************/
struct Qmarray * qmarray_loadtxt(FILE * fp)
{
    size_t nrows, ncols;
    int num = fscanf(fp,"%zu ",&nrows);
    assert (num == 1);
    num =fscanf(fp,"%zu ",&ncols);
    assert(num == 1);
    struct Qmarray * qma = qmarray_alloc(nrows, ncols);

    size_t ii;
    for (ii = 0; ii < nrows * ncols; ii++){
        qma->funcs[ii] = generic_function_loadtxt(fp);
    }
    
    return qma;
}

/***********************************************************//**
    Create a qmarray by approximating 1d functions

    \param[in] nrows - number of rows of quasimatrix
    \param[in] ncols - number of cols of quasimatrix
    \param[in] fapp  - approximation arguments
    \param[in] fw    - wrapped function

    \return qmarray
***************************************************************/
struct Qmarray * 
qmarray_approx1d(size_t nrows, size_t ncols, struct OneApproxOpts * fapp,
                 struct Fwrap * fw)
                 /* double (**funcs)(double, void *), */
                 /* void ** args,  */
                 /* enum function_class fc, void * st, double lb, */
                 /* double ub, void * aopts) */
{
    struct Qmarray * qm = qmarray_alloc(nrows,ncols);
    size_t ii;
    for (ii = 0; ii < nrows*ncols; ii++){
        fwrap_set_which_eval(fw,ii);
        qm->funcs[ii] = 
            generic_function_approximate1d(fapp->fc,fapp->aopts,fw);
    }
    return qm;
}

/***********************************************************
    Create a qmarray from a fiber_cuts array

    \param[in] nrows    - number of rows of qmarray
    \param[in] ncols    - number of columns of qmarray
    \param[in] f        - functions
    \param[in] fcut     - array of fiber cuts
    \param[in] fc       - function class of each column
    \param[in] sub_type - sub_type of each column
    \param[in] lb       - lower bound of inputs to functions
    \param[in] ub       - upper bound of inputs to functions
    \param[in] aopts    - approximation options

    \return qmarray
***************************************************************/
/* struct Qmarray *  */
/* qmarray_from_fiber_cuts(size_t nrows, size_t ncols,  */
/*                     double (*f)(double, void *),struct FiberCut ** fcut,  */
/*                     enum function_class fc, void * sub_type, double lb, */
/*                     double ub, void * aopts) */
/* { */
/*     struct Qmarray * qm = qmarray_alloc(nrows,ncols); */
/*     size_t ii; */
/*     for (ii = 0; ii < nrows*ncols; ii++){ */
/*         qm->funcs[ii] = generic_function_approximate1d(f, fcut[ii],  */
/*                                     fc, sub_type, lb, ub, aopts); */
/*     } */
/*     return qm; */
/* } */



/***********************************************************//**
    Generate a qmarray with orthonormal columns consisting of
    one dimensional functions of linear elements on a certain grid

    \param[in] nrows - number of rows
    \param[in] ncols - number of columns
    \param[in] grid  - grid size

    \return qmarray with orthonormal columns

    \note
        - Not super efficient because of copies
***************************************************************/
/* struct Qmarray * */
/* qmarray_orth1d_linelm_grid(size_t nrows,size_t ncols, struct c3Vector * grid) */
/* { */
/*     struct Qmarray * qm = qmarray_alloc(nrows,ncols); */
/*     generic_function_array_orth1d_linelm_columns(qm->funcs,nrows,ncols,grid); */
/*     return qm; */
/* } */

/***********************************************************//**
    Generate a qmarray with orthonormal rows

    \param[in] nrows - number of rows
    \param[in] ncols - number of columns
    \param[in] opts  - approximation optsion

    \return qmarray with orthonormal rows

    \note
        Not super efficient because of copies
***************************************************************/
struct Qmarray *
qmarray_orth1d_rows(size_t nrows, size_t ncols, struct OneApproxOpts * opts)
{

    struct Qmarray * qm = qmarray_alloc(nrows,ncols);
    
    struct GenericFunction ** funcs = NULL;
    if ( NULL == (funcs = malloc(nrows*sizeof(struct GenericFunction *)))){
        fprintf(stderr, "failed to allocate memory for quasimatrix.\n");
        exit(1);
    }
    size_t ii, jj,kk;
    for (ii = 0; ii < nrows; ii++){
        funcs[ii] = NULL;
    }
    /* printf("wwwhat\n"); */
    generic_function_array_orth(nrows, funcs, opts->fc, opts->aopts);
    /* printf("wwwhere\n"); */
    /* for (size_t ii = 0; ii < nrows; ii++){ */
    /*     printf("after array_orth ii = %zu\n", ii); */
    /*     print_generic_function(funcs[ii], 4, NULL, stdout);         */
    /* } */
    
    struct GenericFunction * zero = generic_function_constant(0.0,opts->fc,opts->aopts);
    
    size_t onnon = 0;
    size_t onorder = 0;
    for (jj = 0; jj < nrows; jj++){
        assert (onorder < nrows);
        qm->funcs[onnon*nrows+jj] = generic_function_copy(funcs[onorder]);
        double normfunc = generic_function_norm(qm->funcs[onnon*nrows+jj]);
        /* printf("normfunc rows = %3.15G\n", normfunc); */
        if (isnan(normfunc)){
            printf("normfunc rows = %3.15G\n", normfunc);
            printf("num_rows = %zu\n", nrows);
            printf("num_cols = %zu\n", ncols);
            printf("onorder = %zu\n", onorder);
            print_generic_function(qm->funcs[onnon*nrows+jj], 4, NULL, stdout);
            exit(1);
        }
        for (kk = 0; kk < onnon; kk++){
            qm->funcs[kk*nrows+jj] = generic_function_copy(zero);
        }
        for (kk = onnon+1; kk < ncols; kk++){
            qm->funcs[kk*nrows+jj] = generic_function_copy(zero);
        }
        onnon = onnon+1;
        if (onnon == ncols){
            onorder = onorder+1;
            onnon = 0;
        }
    }
    //printf("max order rows = %zu\n",onorder);
    
    for (ii = 0; ii < nrows;ii++){
        generic_function_free(funcs[ii]);
        funcs[ii] = NULL;
    }
    free(funcs); funcs=NULL;
    generic_function_free(zero); zero = NULL;

    return qm;
}

/***********************************************************//**
    Generate a qmarray with orthonormal columns consisting of
    one dimensional functions

    \param[in] nrows - number of rows
    \param[in] ncols - number of columns
    \param[in] opts  - options

    \return qmarray with orthonormal columns

    \note
        - Not super efficient because of copies
***************************************************************/
struct Qmarray *
qmarray_orth1d_columns(size_t nrows,size_t ncols,struct OneApproxOpts * opts)
{

    struct Qmarray * qm = qmarray_alloc(nrows,ncols);
    
    struct GenericFunction ** funcs = NULL;
    if ( NULL == (funcs = malloc(ncols*sizeof(struct GenericFunction *)))){
        fprintf(stderr, "failed to allocate memory for quasimatrix.\n");
        exit(1);
    }
    /* size_t ii, jj; //,kk; */
    for (size_t ii = 0; ii < ncols; ii++){
        funcs[ii] = NULL;
    }

    struct GenericFunction * zero = generic_function_constant(0.0,opts->fc,opts->aopts);
    generic_function_array_orth(ncols,funcs,opts->fc,opts->aopts);
    /* if (nrows >= ncols){ */
    /*     generic_function_array_orth(1,funcs,opts->fc,opts->aopts); */
    /*     for (size_t col = 0; col < ncols; col++){ */
    /*         for (size_t row = 0; row < nrows; row++){ */
    /*             if (row != col){ */
    /*                 qm->funcs[col*nrows+row] = generic_function_copy(zero); */
    /*             } */
    /*             else{ */
    /*                 qm->funcs[col*nrows+row] = generic_function_copy(funcs[0]); */
    /*             } */
                
    /*         } */
    /*     } */
    /* } */
    /* else{ */

    /*     size_t ngen = ncols / nrows + 1; */
    /*     generic_function_array_orth(ngen+1,funcs,opts->fc,opts->aopts); */
    /*     size_t oncol = 0; */
    /*     /\* int minnum = 0; *\/ */
    /*     size_t maxnum = 0; */
    /*     int converged = 0; */
    /*     printf("nrows=%zu,ncols=%zu,ngen=%zu\n",nrows,ncols,ngen); */
    /*     while (converged == 0){ */
    /*         size_t count = 0; */
    /*         printf("maxnum = %zu\n",maxnum); */
    /*         /\* print_generic_function(funcs[maxnum],3,NULL); *\/ */
    /*         for (size_t col = oncol; col < oncol+nrows; col++){ */
    /*             printf("col = %zu/%zu\n",col,ncols); */
    /*             if (col == ncols){ */
    /*                 printf("conveged!\n"); */
    /*                 converged = 1; */
    /*                 break; */
    /*             } */
    /*             for (size_t row = 0; row < nrows; row++){ */
    /*                 printf("row=%zu/%zu\n",row,nrows); */
    /*                 if (row != count){ */
    /*                     qm->funcs[col*nrows+row] = generic_function_copy(zero); */
    /*                 } */
    /*                 else{ */
    /*                     qm->funcs[col*nrows+row] = generic_function_copy(funcs[maxnum]); */
    /*                 } */
    /*             } */
    /*             count++; */
    /*         } */
    /*         maxnum += 1; */
    /*         oncol += nrows; */
    /*         if (maxnum > ngen){ */
    /*             fprintf(stderr,"Cannot generate enough orthonormal polynomials\n"); */
    /*             assert (1 == 0); */
    /*         } */
    /*     } */
    /*     printf("we are out\n"); */
    /* }         */
        size_t onnon = 0;
        size_t onorder = 0;
        for (size_t jj = 0; jj < ncols; jj++){
            qm->funcs[jj*nrows+onnon] = generic_function_copy(funcs[onorder]);
            double normfunc = generic_function_norm(qm->funcs[jj*nrows+onnon]);
            /* printf("normfunc cols = %3.15G %d \n", normfunc, isnan(normfunc)); */
            if (isnan(normfunc)){
                printf("normfunc cols = %3.15G\n", normfunc);
                printf("num_rows = %zu\n", nrows);
                printf("num_cols = %zu\n", ncols);
                print_generic_function(qm->funcs[jj*nrows+onnon], 4, NULL, stdout);

                generic_function_free(qm->funcs[jj*nrows+onnon]);
                qm->funcs[jj*nrows+onnon] = generic_function_copy(zero);
                exit(1);
            }

            for (size_t kk = 0; kk < onnon; kk++){
                qm->funcs[jj*nrows+kk] = generic_function_copy(zero);
            }
            for (size_t kk = onnon+1; kk < nrows; kk++){
                qm->funcs[jj*nrows+kk] = generic_function_copy(zero);
            }
            onnon = onnon+1;
            if (onnon == nrows){
                onorder = onorder+1;
                onnon = 0;
            }
        }

    //printf("max order rows = %zu\n",onorder);
    
    for (size_t ii = 0; ii < ncols; ii++){
        if (funcs[ii] != NULL){
            generic_function_free(funcs[ii]);
            funcs[ii] = NULL;
        }
    }
    free(funcs); funcs=NULL;
    generic_function_free(zero); zero = NULL;
    /* printf("returned\n"); */
    /* return qm; */
    /* struct Qmarray * qm = qmarray_alloc(nrows,ncols); */
    /* struct Qmarray * qmtemp = qmarray_alloc(ncols,1); */
    /* generic_function_array_orth1d_columns(qm->funcs,qmtemp->funcs, */
    /*                                       fc,st,nrows,ncols,lb,ub); */
    
    /* qmarray_free(qmtemp); qmtemp = NULL; */
    return qm;
}

/***********************************************************//**
    Create a Qmarray of zero functions

    \param[in] nrows - number of rows of quasimatrix
    \param[in] ncols - number of cols of quasimatrix
    \param[in] opts  - options


    \return qmarray
***************************************************************/
struct Qmarray *
qmarray_zeros(size_t nrows,size_t ncols, struct OneApproxOpts * opts)
{
    struct Qmarray * qm = qmarray_alloc(nrows,ncols);
    size_t ii;
    for (ii = 0; ii < nrows*ncols; ii++){
        qm->funcs[ii] = generic_function_constant(0.0,opts->fc,opts->aopts);
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
    Return a typed function                                                           
**************************************************************/
void *
qmarray_get_func_base(const struct Qmarray * qma, size_t row, size_t col)
{
    assert (qma != NULL);
    assert (row < qma->nrows);
    assert (col < qma->ncols);

    return qma->funcs[col*qma->nrows+row]->f;
}

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
void qmarray_set_column(struct Qmarray * qma, size_t col, 
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

/***********************************************************//**
    get number of columns
***************************************************************/
size_t qmarray_get_ncols(const struct Qmarray * qma)
{
    assert (qma != NULL);
    return qma->ncols;
}

/***********************************************************//**
    get number of rows
***************************************************************/
size_t qmarray_get_nrows(const struct Qmarray * qma)
{
    assert (qma != NULL);
    return qma->nrows;
}

//////////////////////////////////////////////////////////////////

/***********************************************************//**
    Quasimatrix array - matrix multiplication

    \param[in] Q - quasimatrix array
    \param[in] R - matrix  (fortran order)
    \param[in] b - number of columns of R

    \return B - qmarray
***************************************************************/
struct Qmarray * qmam(const struct Qmarray * Q, const double * R, size_t b)
{
    size_t nrows = Q->nrows;
    struct Qmarray * B = qmarray_alloc(nrows,b);
    size_t ii,jj;
    for (jj = 0; jj < nrows; jj++){
        for (ii = 0; ii < b; ii++){
            B->funcs[ii*nrows+jj] = // Q[jj,:]B[:,ii]
                generic_function_lin_comb2(Q->ncols,
                    Q->nrows, Q->funcs + jj, 1, R + ii*Q->ncols);
        }
    }
    return B;
}

/***********************************************************//**
    Transpose Quasimatrix array - matrix multiplication

    \param[in] Q - quasimatrix array
    \param[in] R - matrix  (fortran order)
    \param[in] b - number of columns of R

    \return B - qmarray
***************************************************************/
struct Qmarray *
qmatm(const struct Qmarray * Q,const double * R, size_t b)
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

    \param[in] R  - matrix  (fortran order)
    \param[in] Q  - quasimatrix array
    \param[in] b  - number of rows of R

    \return B - qmarray (b x Q->ncols)
***************************************************************/
struct Qmarray *
mqma(const double * R, const struct Qmarray * Q, size_t b)
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

    \param[in] a - qmarray 1
    \param[in] b - qmarray 2
 
    \return c - qmarray 
***************************************************************/
struct Qmarray *
qmaqma(const struct Qmarray * a, const struct Qmarray * b)
{
    struct Qmarray * c = qmarray_alloc(a->nrows, b->ncols);
    size_t ii,jj;
    for (jj = 0; jj < b->ncols; jj++){
        for (ii = 0; ii < a->nrows; ii++){
            // a[ii,:]b[:,jj]
            c->funcs[jj*c->nrows+ii] =
                generic_function_sum_prod(a->ncols, a->nrows, 
                                          a->funcs + ii, 1,
                                          b->funcs + jj*b->nrows);
        }
    }
    return c;
}

/***********************************************************//**
    Transpose qmarray - qmarray mutliplication

    \param[in] a - qmarray 1
    \param[in] b - qmarray 2

    \return c - qmarray : c = a^T b
***************************************************************/
struct Qmarray *
qmatqma(const struct Qmarray * a, const struct Qmarray * b)
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

    \param[in] a  - qmarray 1
    \param[in] b  - qmarray 2

    \return c - qmarray : c = a b^T
***************************************************************/
struct Qmarray *
qmaqmat(const struct Qmarray * a, const struct Qmarray * b)
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

    \param[in] a  - qmarray 1
    \param[in] b - qmarray 2

    \return c - qmarray 
***************************************************************/
struct Qmarray *
qmatqmat(const struct Qmarray * a, const struct Qmarray * b)
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

    \param[in] a  - qmarray 1
    \param[in] b  - qmarray 2

    \return array of integrals of  \f$ c = \int a(x)^T b(x) dx \f$
***************************************************************/
double *
qmatqma_integrate(const struct Qmarray * a,const struct Qmarray * b)
{
    double * c = calloc_double(a->ncols*b->ncols);
    size_t nrows = a->ncols;
    size_t ncols = b->ncols;
    size_t ii,jj;
    for (jj = 0; jj < ncols; jj++){
        for (ii = 0; ii < nrows; ii++){
            // c[jj*nrows+ii] = a[:,ii]^T b[jj,:]
            c[jj*nrows+ii] = generic_function_inner_sum(b->nrows, 1, 
                    a->funcs + ii*a->nrows, 1, b->funcs + jj*b->nrows);
        }
    }
    return c;
}

/***********************************************************//**
    Integral of qmarray - transpose qmarray mutliplication

    \param[in] a - qmarray 1
    \param[in] b - qmarray 2

    \return array of integrals of  \f$ c = \int a(x) b(x)^T dx \f$
***************************************************************/
double *
qmaqmat_integrate(const struct Qmarray * a,const struct Qmarray * b)
{
    double * c = calloc_double(a->nrows*b->nrows);
    size_t nrows = a->nrows;
    size_t ncols = b->nrows;
    size_t ii,jj;
    for (jj = 0; jj < ncols; jj++){
        for (ii = 0; ii < nrows; ii++){
            // c[jj*nrows+ii] = a[:,ii]^T b[jj,:]
            c[jj*nrows+ii] =  
                generic_function_inner_sum(a->ncols, 
                                           a->nrows, 
                                           a->funcs + ii, 
                                           b->nrows, b->funcs + jj);
        }
    }
    return c;
}

/***********************************************************//**
    Integral of Transpose qmarray - transpose qmarray mutliplication

    \param[in] a - qmarray 1
    \param[in] b - qmarray 2

    \return array of integrals
***************************************************************/
double *
qmatqmat_integrate(const struct Qmarray * a,const struct Qmarray * b)
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
    (Weighted) Integral of Transpose qmarray - transpose qmarray mutliplication

    \param[in] a - qmarray 1
    \param[in] b - qmarray 2

    \return array of integral values

    \note 
    In order for this function to make sense a(x) and b(x)
    shoud have the same underlying basis
***************************************************************/
double *
qmatqmat_integrate_weighted(const struct Qmarray * a,const struct Qmarray * b)
{
    double * c = calloc_double(a->ncols*b->nrows);
    size_t ii,jj;
    for (jj = 0; jj < b->nrows; jj++){
        for (ii = 0; ii < a->ncols; ii++){
            // c[jj*a->ncols+ii] = a[:,ii]^T b[jj,:]
            //c[ii*b->nrows+jj] =  generic_function_sum_prod_integrate(b->ncols, 1, 
            //        a->funcs + ii*a->nrows, b->nrows, b->funcs + jj);

            c[ii*b->nrows+jj] =
                generic_function_inner_weighted_sum(b->ncols, 1, 
                                                    a->funcs + ii*a->nrows, b->nrows, b->funcs + jj);
        }
    }
    return c;
}

/***********************************************************//**
    Kronecker product between two qmarrays

    \param[in] a - qmarray 1
    \param[in] b - qmarray 2

    \return c -  qmarray kron(a,b)
***************************************************************/
struct Qmarray * qmarray_kron(const struct Qmarray * a,const struct Qmarray * b)
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
    Integral of kronecker product between two qmarrays

    \param[in] a - qmarray 1
    \param[in] b - qmarray 2

    \return c -  \f$ \int kron(a(x),b(x)) dx \f$
***************************************************************/
double * qmarray_kron_integrate(const struct Qmarray * a, const struct Qmarray * b)
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
    Weighted integral of kronecker product between two qmarrays

    \param[in] a - qmarray 1
    \param[in] b - qmarray 2

    \return c - \f$ \int kron(a(x),b(x)) w(x) dx \f$

    \note 
    In order for this function to make sense a(x) and b(x)
    shoud have the same underlying basis
***************************************************************/
double * qmarray_kron_integrate_weighted(const struct Qmarray * a, const struct Qmarray * b)
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
                    c[column*totrows + row] = generic_function_inner_weighted(
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

    \param[in] a - vector,array (b->ncols * c->ncols);
    \param[in] b - qmarray 1
    \param[in] c - qmarray 2

    \return d - qmarray  b->ncols x (c->ncols)
***************************************************************/
struct Qmarray *
qmarray_vec_kron(const double * a,const struct Qmarray * b,const struct Qmarray * c)
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

    \param[in] a - vector,array (b->nrows * c->nrows);
    \param[in] b - qmarray 1
    \param[in] c - qmarray 2

    \return d - b->ncols x (c->ncols)
***************************************************************/
double *
qmarray_vec_kron_integrate(const double * a, const struct Qmarray * b,const struct Qmarray * c)
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
    If a is vector, b and c are quasimatrices then computes
            \f$ \int a^T kron(b(x),c(x)) w(x) dx  \f$

    \param[in] a - vector,array (b->nrows * c->nrows);
    \param[in] b - qmarray 1
    \param[in] c - qmarray 2

    \return d -  b->ncols x (c->ncols)

    \note 
    In order for this function to make sense b(x) and c(x)
    shoud have the same underlying basis
***************************************************************/
double *
qmarray_vec_kron_integrate_weighted(const double * a, const struct Qmarray * b,const struct Qmarray * c)
{
    double * d = NULL;

    struct Qmarray * temp = qmatm(b,a,c->nrows); // b->ncols * c->ncols
    //printf("got qmatm\n");
    //d = qmaqma(temp,c);
    d = qmatqmat_integrate_weighted(c,temp);
    qmarray_free(temp); temp = NULL;
    return d;
}

/***********************************************************//**
    If a is a matrix, b and c are qmarrays then computes
             a kron(b,c) fast
    
    \param[in] r - number of rows of a
    \param[in] a - array (k, b->nrows * c->nrows);
    \param[in] b - qmarray 1
    \param[in] c - qmarray 2

    \return d - qmarray  (k, b->ncols * c->ncols)

    \note
        Do not touch this!! Who knows how it happens to be right...
        but it is tested (dont touch the test either. and it seems to be right
***************************************************************/
struct Qmarray *
qmarray_mat_kron(size_t r, const double * a, const struct Qmarray * b, const struct Qmarray * c)
{
    struct Qmarray * temp = qmarray_alloc(r*b->nrows,c->ncols);
    generic_function_kronh(1, r, b->nrows, c->nrows, c->ncols, a,
                            c->funcs,temp->funcs);

    struct Qmarray * d = qmarray_alloc(r, b->ncols * c->ncols);
    generic_function_kronh2(1,r, c->ncols, b->nrows,b->ncols,
                        b->funcs,temp->funcs, d->funcs);

    qmarray_free(temp); temp = NULL;
    return d;
}

/***********************************************************//**
    If a is a matrix, b and c are qmarrays then computes
             kron(b,c)a fast
    
    \param[in] r - number of rows of a
    \param[in] a - array (b->ncols * c->ncols, a);
    \param[in] b - qmarray 1
    \param[in] c - qmarray 2

    \return d - qmarray  (b->nrows * c->nrows , k)

    \note
        Do not touch this!! Who knows how it happens to be right...
        but it is tested (dont touch the test either. and it seems to be right
***************************************************************/
struct Qmarray *
qmarray_kron_mat(size_t r, const double * a, const struct Qmarray * b,
                 const struct Qmarray * c)
{
    struct Qmarray * temp = qmarray_alloc(c->nrows,b->ncols*r);
    generic_function_kronh(0, r, b->ncols, c->nrows, c->ncols, a,
                            c->funcs,temp->funcs);

    struct Qmarray * d = qmarray_alloc(b->nrows * c->nrows,r);
    generic_function_kronh2(0,r, c->nrows, b->nrows,b->ncols,
                        b->funcs,temp->funcs, d->funcs);

    qmarray_free(temp); temp = NULL;
    return d;
}

// The algorithms below really are not cache-aware and might be slow ...
void qmarray_block_kron_mat(char type, int left, size_t nlblocks,
        struct Qmarray ** lblocks, struct Qmarray * right, size_t r,
        double * mat, struct Qmarray * out)
{   
    if (left == 1)
    {

        size_t ii;
        if (type == 'D'){
            size_t totrows = 0;
            for (ii = 0; ii < nlblocks; ii++){
                totrows += lblocks[ii]->nrows;    
            }
            struct GenericFunction ** gfarray = 
                generic_function_array_alloc(r * right->ncols * totrows);
            
            generic_function_kronh(1, r, totrows, right->nrows, right->ncols,
                    mat, right->funcs, gfarray);
        

            size_t width = 0;
            size_t  runrows = 0;
            size_t jj,kk,ll;
            for (ii = 0; ii < nlblocks; ii++){
                struct GenericFunction ** temp = 
                    generic_function_array_alloc(right->ncols * r * lblocks[ii]->nrows);

                for (jj = 0; jj < right->ncols; jj++){
                    for (kk = 0; kk < lblocks[ii]->nrows; kk++){
                        for (ll = 0; ll < r; ll++){
                            temp[ll + kk*r + jj * lblocks[ii]->nrows*r] =
                                gfarray[runrows + ll + kk*r + jj*totrows*r];
                        }
                    }
                }
                generic_function_kronh2(1,r,right->ncols, 
                        lblocks[ii]->nrows, lblocks[ii]->ncols,
                        lblocks[ii]->funcs, temp, 
                        out->funcs + width * out->nrows);
                width += lblocks[ii]->ncols * right->ncols;
                runrows += r * lblocks[ii]->nrows;
                free(temp); temp = NULL;
            }
            generic_function_array_free(gfarray, r * right->ncols * totrows);
        }
        else if (type == 'R'){
            size_t row1 = lblocks[0]->nrows;
            for (ii = 1; ii < nlblocks; ii++){
                assert( row1 == lblocks[ii]->nrows );
            }
            
            struct GenericFunction ** gfarray = 
                generic_function_array_alloc(r * right->ncols * row1);

            generic_function_kronh(1,r, row1, right->nrows, right->ncols,
                    mat, right->funcs, gfarray);

            
            size_t width = 0;
            for (ii = 0; ii < nlblocks; ii++){
                generic_function_kronh2(1,r, right->ncols, row1,
                        lblocks[ii]->ncols, lblocks[ii]->funcs, gfarray,
                        out->funcs + width * out->nrows);
                width += lblocks[ii]->ncols * right->ncols;        
            }
            generic_function_array_free(gfarray,r * right->ncols * row1);
        }
        else if (type == 'C'){
            
            size_t col1 = lblocks[0]->ncols;
            size_t totrows = 0;
            for (ii = 1; ii < nlblocks; ii++){
                assert( col1 == lblocks[ii]->ncols );
                totrows += lblocks[ii]->nrows;    
            }

            struct Qmarray * temp = qmarray_alloc(r, col1*nlblocks*right->ncols);
            qmarray_block_kron_mat('D',1,nlblocks,lblocks,right,r,mat,temp);
            
            size_t jj,kk,ll;
            size_t startelem = 0;
            for (ii = 0; ii < col1; ii++){
                for (jj = 0; jj < right->ncols; jj++){
                    for (kk = 0; kk < r; kk++){
                        out->funcs[kk + jj*r + ii * r*right->ncols] =
                            generic_function_copy(
                                temp->funcs[startelem + kk + jj*r + ii * r*right->ncols]);

                    }
                }
            }
            startelem += r * right->ncols * col1;
            for (ll = 1; ll < nlblocks; ll++){
                for (ii = 0; ii < col1; ii++){
                    for (jj = 0; jj < right->ncols; jj++){
                        for (kk = 0; kk < r; kk++){
                            generic_function_axpy(1.0,
                                temp->funcs[startelem + kk + jj*r + ii * r*right->ncols],
                                out->funcs[kk + jj*r + ii * r*right->ncols]);

                        }
                    }
                }
                startelem += r * right->ncols * col1;
            }

            qmarray_free(temp); temp = NULL;

        }
    }
    else{
        size_t ii;
        if (type == 'D'){
            size_t totcols = 0;
            size_t totrows = 0;
            for (ii = 0; ii < nlblocks; ii++){
                totcols += lblocks[ii]->ncols;    
                totrows += lblocks[ii]->nrows;
            }


            struct GenericFunction ** gfarray = 
                generic_function_array_alloc(r * right->nrows * totcols);
            
            generic_function_kronh(0, r, totcols, right->nrows, right->ncols,
                    mat, right->funcs, gfarray);
        
           // printf("number of elements in gfarray is %zu\n",r * right->nrows * totcols);
            size_t runcols = 0;
            size_t runrows = 0;
            size_t jj,kk,ll;
           // printf("done with computing kronh1\n");
            for (ii = 0; ii < nlblocks; ii++){

                size_t st1a = right->nrows;
                size_t st1b = lblocks[ii]->ncols;
                size_t st = st1a * st1b * r; 
                struct GenericFunction ** temp = generic_function_array_alloc(st);
                
                size_t rga = right->nrows;;
                size_t lastind = right->nrows * totcols;

                for (jj = 0; jj < r; jj++){
                    for (kk = 0; kk < st1b; kk++){
                        for (ll = 0; ll < st1a; ll++){
                            temp [ll + kk*st1a + jj*st1a*st1b] =
                                gfarray[ll + (kk+runcols)*rga  + jj*lastind];
                        }
                    }
                }
                
                size_t st2a = right->nrows;
                size_t st2b = lblocks[ii]->nrows;
                size_t st2 = st2a * st2b * r;

                struct GenericFunction ** temp2 = 
                    generic_function_array_alloc(st2);

                generic_function_kronh2(0,r,st2a, st2b, st1b,
                        lblocks[ii]->funcs, temp, temp2);
                
                for (jj = 0; jj < st2a; jj++){
                    for (kk = 0; kk < st2b; kk++){
                        for (ll = 0; ll < r; ll++){
                            out->funcs[jj + (runrows + kk)*st2a + ll*totrows* st2a] =
                                temp2[jj + kk*st2a + ll*st2a*st2b];
                        }
                    }
                }

                runcols += lblocks[ii]->ncols;
                runrows += lblocks[ii]->nrows;
                free(temp); temp = NULL;
                free(temp2); temp2 = NULL;
            }
            generic_function_array_free(gfarray, r*right->nrows*totcols);
        }
        else if (type == 'R')
        {

            size_t totcols = lblocks[0]->ncols;
            size_t row1 = lblocks[0]->nrows;
            for (ii = 1; ii < nlblocks; ii++){
                totcols += lblocks[ii]->ncols;    
                assert ( lblocks[ii]->nrows == row1);
            }

            struct Qmarray * temp = 
                qmarray_alloc(row1*nlblocks*right->nrows, r);
            qmarray_block_kron_mat('D',0,nlblocks,lblocks,right,r,mat,temp);

            size_t jj,kk,ll;
            size_t ind1,ind2;
            size_t runrows = 0;
            for (ii = 0; ii < nlblocks; ii++){
                //printf(" on block %zu\n",ii);
                if (ii == 0){
                    for (jj = 0; jj < r; jj++){
                        for (kk = 0; kk < row1; kk++){
                            for (ll = 0; ll < right->nrows; ll++){
                                ind1 = ll + kk * right->nrows + 
                                        jj * row1 * right->nrows;
                                ind2 = ll + kk * right->nrows + 
                                        jj * row1*nlblocks * right->nrows;
                                out->funcs[ind1] = 
                                    generic_function_copy(temp->funcs[ind2]);
                            }
                        }
                    }
                }
                else{
                    for (jj = 0; jj < r; jj++){
                        for (kk = 0; kk < row1; kk++){
                            for (ll = 0; ll < right->nrows; ll++){
                                ind1 = ll + kk * right->nrows + 
                                            jj * row1 * right->nrows;
                                ind2 = ll + (kk+runrows) * right->nrows + 
                                            jj * row1 *nlblocks* right->nrows;
                                generic_function_axpy(
                                        1.0,
                                        temp->funcs[ind2],
                                        out->funcs[ind1]);
                            }
                        }
                    }
                }
                runrows += row1;
            }
            qmarray_free(temp); temp = NULL;
        }
        else if (type == 'C'){
            size_t totrows = lblocks[0]->nrows;
            size_t col1 = lblocks[0]->ncols;
            for (ii = 1; ii < nlblocks; ii++){
                totrows += lblocks[ii]->nrows;    
                assert ( lblocks[ii]->ncols == col1);
            }

            struct GenericFunction ** gfarray = 
                generic_function_array_alloc(r * right->nrows * col1);
            
            generic_function_kronh(0, r, col1, right->nrows, right->ncols,
                    mat, right->funcs, gfarray);
            
            size_t jj,kk,ll,ind1,ind2;
            size_t runrows = 0;
            for (ii = 0; ii < nlblocks; ii++){
                size_t st2a = right->nrows;
                size_t st2b = lblocks[ii]->nrows;
                size_t st2 = st2a * st2b * r;

                struct GenericFunction ** temp2 = 
                    generic_function_array_alloc(st2);

                generic_function_kronh2(0,r,st2a, st2b, lblocks[ii]->ncols,
                        lblocks[ii]->funcs, gfarray, temp2);

                for (jj = 0; jj < r; jj++){
                    for (kk = 0; kk < st2b; kk++){
                        for (ll = 0; ll < right->nrows; ll++){
                            ind2 = ll + kk * right->nrows + 
                                jj * st2b * right->nrows;
                            ind1 = ll + (kk + runrows)*right->nrows + 
                                jj* totrows * right->nrows;
                            out->funcs[ind1] = temp2[ind2];
                        }
                    }
                }
                runrows += lblocks[ii]->nrows;
                free(temp2); temp2 = NULL;
            }
            
            generic_function_array_free(gfarray,
                                    r * right->nrows * col1);

        }
    }
}


/***********************************************************//**
    Integrate all the elements of a qmarray

    \param[in] a - qmarray to integrate

    \return out - array containing integral of every function in a
***************************************************************/
double * qmarray_integrate(const struct Qmarray * a)
{
    
    double * out = calloc_double(a->nrows*a->ncols);
    size_t ii, jj;
    for (ii = 0; ii < a->ncols; ii++){
        for (jj = 0; jj < a->nrows; jj++){
            out[ii*a->nrows + jj] = 
                generic_function_integrate(a->funcs[ii*a->nrows+jj]);
        }
    }
    
    return out;
}

/***********************************************************//**
    Weighted integration of all the elements of a qmarray

    \param[in] a - qmarray to integrate

    \return out - array containing integral of every function in a

    \note Computes \f$ \int f(x) w(x) dx \f$ for every univariate function
    in the qmarray
    
    w(x) depends on underlying parameterization
    for example, it is 1/2 for legendre (and default for others),
    gauss for hermite,etc
***************************************************************/
double * qmarray_integrate_weighted(const struct Qmarray * a)
{
    
    double * out = calloc_double(a->nrows*a->ncols);
    size_t ii, jj;
    for (ii = 0; ii < a->ncols; ii++){
        for (jj = 0; jj < a->nrows; jj++){
            out[ii*a->nrows + jj] = 
                generic_function_integrate_weighted(a->funcs[ii*a->nrows+jj]);
        }
    }
    
    return out;
}

/***********************************************************//**
    Norm of a qmarray

    \param[in] a - first qmarray

    \return L2 norm
***************************************************************/
double qmarray_norm2(const struct Qmarray * a)
{
    double out = 0.0;
    size_t ii;
    for (ii = 0; ii < a->ncols * a->nrows; ii++){
//        print_generic_function(a->funcs[ii],0,NULL);
        out += generic_function_inner(a->funcs[ii],a->funcs[ii]);
        // printf("out in qmarray norm = %G\n",out);
    }
    if (out < 0){
        fprintf(stderr,"Inner product between two qmarrays cannot be negative %G\n",out);
        exit(1);
        //return 0.0;
    }

    return sqrt(fabs(out));
}

/***********************************************************//**
    Two norm difference between two functions

    \param[in] a - first qmarray
    \param[in] b - second qmarray

    \return out - difference in 2 norm
***************************************************************/
double qmarray_norm2diff(const struct Qmarray * a,const struct Qmarray * b)
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
        //return 0.0;
    }
    generic_function_array_free(temp, a->ncols*a->nrows);
    //free(temp);
    temp = NULL;
    return sqrt(out);
}

/***********************************************************//**
    Compute \f[ y \leftarrow ax + y \f]

    \param[in]     a - scale for first qmarray
    \param[in]     x - first qmarray
    \param[in,out] y - second qmarray
***************************************************************/
void qmarray_axpy(double a, const struct Qmarray * x, struct Qmarray * y)
{
    assert (x->nrows == y->nrows );
    assert (x->ncols == y->ncols );
    
    size_t ii;
    for (ii = 0; ii < x->nrows * x->ncols; ii++){
        generic_function_axpy(a,x->funcs[ii],y->funcs[ii]);
    }
}

struct Zeros {
    double * zeros;
    size_t N;
};

int eval_zpoly(size_t n, const double * x,double * out, void * args){

    struct Zeros * z = args;
    for (size_t jj = 0; jj < n; jj++){
        out[jj] = 1.0;
        size_t ii;
        for (ii = 0; ii < z->N; ii++){
            out[jj] *= (x[jj] - z->zeros[ii]);
        }
    }
    return 0;
}

void create_any_L(struct GenericFunction ** L, size_t nrows, 
                  size_t upto,size_t * piv, double * px,
                  struct OneApproxOpts * approxopts, void * optargs)
                  /* double lb, double ub, void * optargs) */
{
    
    //create an arbitrary quasimatrix array that has zeros at piv[:upto-1],px[:upto-1]
    // and one at piv[upto],piv[upto] less than one every else
    size_t ii,jj;
    size_t * count = calloc_size_t(nrows);
    double ** zeros = malloc( nrows * sizeof(double *));
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
            L[ii] = generic_function_constant(1.0,approxopts->fc,approxopts->aopts);//&ptype,lb,ub,NULL);
        }
        else{
            struct Zeros zz;
            zz.zeros = zeros[ii];
            zz.N = count[ii];
            struct Fwrap * fw = fwrap_create(1,"general-vec");
            fwrap_set_fvec(fw,eval_zpoly,&zz);
            L[ii] = generic_function_approximate1d(approxopts->fc,approxopts->aopts,fw);
            fwrap_destroy(fw);
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

        //for (ii = 0; ii < nrows; ii++){
        //    print_generic_function(L[ii],3,NULL);
        //}
    }

    //double tval =
    generic_function_array_absmax(nrows, 1, L,&amind, &xval,optargs);
    px[upto] = xval;
    piv[upto] = amind;
    double val = generic_function_1d_eval(L[piv[upto]],px[upto]);
    generic_function_array_scale(1.0/val,L,nrows);
    if (VQMALU){
        printf("got new val = %G\n", val);
        printf("Values of new array at indices\n ");
        for (size_t zz = 0; zz < upto+1; zz++){
            double eval = generic_function_1d_eval(L[piv[zz]],px[zz]);
            printf("\t ind=%zu x=%G val=%G\n",piv[zz],px[zz],eval);
        }
    }

    assert (fabs(val) > ZEROTHRESH);

}

/* static void create_any_L_nodal(struct GenericFunction ** L, */
/*                                enum function_class fc, */
/*                                size_t nrows,  */
/*                                size_t upto,size_t * piv, double * px, */
/*                                struct OneApproxOpts * approxargs, void * optargs) */
/* { */
    
/*     //create an arbitrary quasimatrix array that has zeros at piv[:upto-1],px[:upto-1] */
/*     // and one at piv[upto],piv[upto] less than one every else */
/* //    printf("creating any L!\n"); */

/*     assert (1 == 0); */
/*     double lb,ub; */
        
/*     struct c3Vector * optnodes = NULL; */
/*     piv[upto] = 0; */
/*     if (optargs != NULL){ */
/*         /\* printf("Warning: optargs is not null in create_any_L_linelm so\n"); *\/ */
/*         /\* printf("         might not be able to guarantee that the new px\n"); *\/ */
/*         /\* printf("         is going to lie on a desired node\n"); *\/ */
/*         optnodes = optargs; */
/*         assert (optnodes->size > 3); */
/*         px[upto] = optnodes->elem[1]; */
/*     } */
/*     else{ */
/*         assert (1 == 0); */
/*         /\* px[upto] = lb + ub/(ub-lb) * randu(); *\/ */
/*     } */

    
/*     double * zeros = calloc_double(upto); */
/*     size_t iz = 0; */
/*     for (size_t ii = 0; ii < upto; ii++){ */
/*         if (piv[ii] == 0){ */
/*             if ( (fabs(px[ii] - lb) > 1e-15) && (fabs(px[ii]-ub) > 1e-15)){ */
/*                 zeros[iz] = px[ii]; */
/*                 iz++; */
/*             } */
/*         } */
/*     } */

/*     int exists = 0; */
/*     size_t count = 2; */
/*     do { */
/*         exists = 0; */
/*         if (optargs != NULL){ */
/*             assert (count < optnodes->size); */
/*         } */
/*         for (size_t ii = 0; ii < iz; ii++){ */
/*             if (fabs(px[ii] - px[upto]) < 1e-15){ */
/*                 exists = 1; */
/*                 if (optargs != NULL){ */
/*                     px[upto] = optnodes->elem[count]; */
/*                 } */
/*                 else{ */
/*                     px[upto] = lb + ub/(ub-lb) * randu(); */
/*                 } */
/*                 count++; */
/*                 break; */
/*             } */
/*         } */
/*     }while (exists == 1); */
        
/*     /\* printf("before\n"); *\/ */
/*     /\* dprint(upto,px); *\/ */
/*     /\* iprint_sz(upto,piv); *\/ */
/*     /\* printf("%G\n",px[upto]); *\/ */
/*     /\* dprint(iz,zeros); *\/ */
/*     /\* L[0] = generic_function_onezero(LINELM,px[upto],iz,zeros,lb,ub); *\/ */
/*     L[0] = generic_function_onezero(fc,px[upto],iz,zeros,lb,ub);     */

/*     for (size_t ii = 1; ii < nrows;ii++){ */
/*         L[ii] = generic_function_constant(0.0,fc,approxargs); */
/*     } */

/* //    for (size_t ii = 0; ii < upto; ii++){ */
/* //        double eval = generic_function_1d_eval( */
/* //            L[piv[ii]],px[ii]); */
/*         //      printf("eval should be 0 = %G\n",eval); */
/* //    } */
/* //    double eval = generic_function_1d_eval(L[piv[upto]],px[upto]); */
/* //    printf("eval should be 1 = %G\n", eval); */
/* //    printf("after\n"); */
    
/*     free(zeros); zeros = NULL; */
/* } */

/***********************************************************//**
    Compute the LU decomposition of a quasimatrix array of 1d functioins

    \param[in]     A       - qmarray to decompose
    \param[in,out] L       - qmarray representing L factor
    \param[in,out] u       - allocated space for U factor
    \param[in,out] piv     - row of pivots 
    \param[in,out] px      - x values of pivots 
    \param[in]     app     - approximation arguments
    \param[in]     optargs - optimization arguments

    \return info = 0 full rank <0 low rank ( rank = A->n + info )
***************************************************************/
int qmarray_lu1d(struct Qmarray * A, struct Qmarray * L, double * u,
                 size_t * piv, double * px, struct OneApproxOpts * app,
                 void * optargs)
{
    int info = 0;
    
    size_t ii,kk;
    double val, amloc;
    size_t amind;
    struct GenericFunction ** temp = NULL;
    
    for (kk = 0; kk < A->ncols; kk++){

        if (VQMALU){
            printf("\n\n\n\n\n");
            printf("lu kk=%zu out of %zu \n",kk,A->ncols-1);
            printf("norm=%G\n",generic_function_norm(A->funcs[kk*A->nrows]));
        }
    
        val = generic_function_array_absmax(A->nrows, 1, A->funcs + kk*A->nrows, 
                                            &amind, &amloc,optargs);
        //printf("after absmax\n");
        piv[kk] = amind;
        px[kk] = amloc;
        //printf("absmax\n");

        val = generic_function_1d_eval(A->funcs[kk*A->nrows+amind], amloc);
        if (VQMALU){
            printf("locmax=%3.15G\n",amloc);
            printf("amindmax=%zu\n",amind);
            printf("val of max =%3.15G\n",val);
            struct GenericFunction ** At = A->funcs+kk*L->nrows;
            double eval2 = generic_function_1d_eval(At[amind],amloc);
            printf("eval of thing %G\n",eval2);
            //print_generic_function(A->funcs[kk*A->nrows+amind],0,NULL);
        }
        if (fabs(val) <= ZEROTHRESH) {
            if (VQMALU){
                printf("creating any L\n");
            }

            if ((A->funcs[kk*A->nrows]->fc == LINELM) ||
                (A->funcs[kk*A->nrows]->fc == CONSTELM)){
                //printf("lets go!\n");
                /* create_any_L_linelm(L->funcs+kk*L->nrows, */
                /*                     L->nrows,kk,piv,px,app,optargs); */
                generic_function_array_onezero(
                    L->funcs+kk*L->nrows,
                    L->nrows, 
                    app->fc,
                    kk,piv,px,app->aopts);
            }
            else{
                create_any_L(L->funcs+kk*L->nrows,L->nrows,
                             kk,piv,px,app,optargs);
            }
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
                                           0.0, 1, NULL, 1, 
                                           L->funcs + kk * L->nrows);
        }

        if (VQMALU){
            // printf("got new val = %G\n", val);
            printf("Values of new array at indices\n ");
            struct GenericFunction ** Lt = L->funcs+kk*L->nrows;

            for (size_t zz = 0; zz < kk; zz++){
                double eval = generic_function_1d_eval(Lt[piv[zz]],px[zz]);
                printf("\t ind=%zu x=%G val=%G\n",piv[zz],px[zz],eval);
            }
            printf("-----------------------------------\n");
        }

        //printf("k start here\n");
        for (ii = 0; ii < A->ncols; ii++){
            if (VQMALU){
                printf("In lu qmarray ii=%zu/%zu \n",ii,A->ncols);
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
                //print_generic_function(A->funcs[ii*A->nrows],0,NULL);
                //printf("norm pre=%G\n",generic_function_norm(A->funcs[ii*A->nrows]));
            }
            //printf("got this daxpby\n");
            qmarray_set_column_gf(A,ii,temp);
            if (VQMALU){
                //printf("norm post=%G\n",generic_function_norm(A->funcs[ii*A->nrows]));
            }
            generic_function_array_free(temp,A->nrows);
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
    Compute the LU decomposition of a quasimatrix array of 1d functioins

    \param[in]     A       - qmarray to decompose
    \param[in,out] L       - qmarray representing L factor
    \param[in,out] u       - allocated space for U factor
    \param[in,out] ps      - pivot set
    \param[in]     app     - approximation arguments
    \param[in]     optargs - optimization arguments

    \return info = 0 full rank <0 low rank ( rank = A->n + info )
    \note THIS FUNCTION IS NOT READY YET
***************************************************************/
int qmarray_lu1d_piv(struct Qmarray * A, struct Qmarray * L, double * u,
                     struct PivotSet * ps, struct OneApproxOpts * app,
                     void * optargs)
{
    assert (app != NULL);
    assert (1 == 0);
    int info = 0;
    
    size_t ii,kk;
    double val;
    struct GenericFunction ** temp = NULL;
    
    for (kk = 0; kk < A->ncols; kk++){

        struct Pivot * piv = pivot_set_get_pivot(ps,kk);
        if (VQMALU){
            printf("\n\n\n\n\n");
            printf("lu kk=%zu out of %zu \n",kk,A->ncols-1);
            printf("norm=%G\n",generic_function_norm(A->funcs[kk*A->nrows]));
        }
    
        val = generic_function_array_absmax_piv(A->nrows,1,A->funcs + kk*A->nrows,
                                                piv,optargs);
        
        val = generic_function_1darray_eval_piv(A->funcs + kk*A->nrows,piv);
        if (VQMALU){
            printf("val of max =%3.15G\n",val);
            printf("eval of thing %G\n",val);
        }
        if (fabs(val) <= ZEROTHRESH) {
            if (VQMALU){
                printf("creating any L\n");
            }

            if (A->funcs[kk*A->nrows]->fc == LINELM){
                assert (1 == 0);
                /* generic_function_array_onezero( */
                /*     L->funcs+kk*L->nrows, */
                /*     L->nrows,  */
                /*     app->fc, */
                /*     kk,piv,px,app->aopts); */
            }
            else{
                assert (1 == 0);
                /* create_any_L(L->funcs+kk*L->nrows,L->nrows, */
                /*              kk,piv,px,app,optargs); */
            }

            /* if (VQMALU){ */
            /*     printf("done creating any L\n"); */
            /* } */
        }
        else{
            generic_function_array_daxpby2(A->nrows,1.0/val, 1, 
                                           A->funcs + kk*A->nrows, 
                                           0.0, 1, NULL, 1, 
                                           L->funcs + kk * L->nrows);
        }

        if (VQMALU){
            // printf("got new val = %G\n", val);
            printf("Values of new array at indices\n ");
            struct GenericFunction ** Lt = L->funcs+kk*L->nrows;

            for (size_t zz = 0; zz < kk; zz++){
                struct Pivot * pp = pivot_set_get_pivot(ps,zz);
                double eval = generic_function_1darray_eval_piv(Lt,pp);
                size_t ind = pivot_get_ind(pp);
                //printf("\t ind=%zu x=%G val=%G\n",piv[zz],px[zz],eval);
                printf("\t ind=%zu val=%G\n",ind,eval);
            }
            printf("-----------------------------------\n");
        }

        //printf("k start here\n");
        for (ii = 0; ii < A->ncols; ii++){
            if (VQMALU){
                printf("In lu qmarray ii=%zu/%zu \n",ii,A->ncols);
            }
            if (ii == kk){
                u[ii*A->ncols+kk] = val;
            }
            else{
                u[ii*A->ncols+kk] = 
                    generic_function_1darray_eval_piv(A->funcs + ii*A->nrows,
                                                      piv);
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
                //print_generic_function(A->funcs[ii*A->nrows],0,NULL);
                //printf("norm pre=%G\n",generic_function_norm(A->funcs[ii*A->nrows]));
            }
            //printf("got this daxpby\n");
            qmarray_set_column_gf(A,ii,temp);
            if (VQMALU){
                //printf("norm post=%G\n",generic_function_norm(A->funcs[ii*A->nrows]));
            }
            generic_function_array_free(temp,A->nrows);
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


/***********************************************************//**
    Perform a greedy maximum volume procedure to find the 
    maximum volume submatrix of a qmarray

    \param[in]     A       - qmarray
    \param[in,out] Asinv   - submatrix inv
    \param[in,out] pivi    - row pivots 
    \param[in,out] pivx    - x value in row pivots 
    \param[in]     appargs - approximation arguments
    \param[in]     optargs - optimization arguments

    \return info = 
            0 converges,
            < 0 no invertible submatrix,
            >0 rank may be too high,
    
    \note
        naive implementation without rank 1 updates
***************************************************************/
int qmarray_maxvol1d(struct Qmarray * A, double * Asinv, size_t * pivi, 
                     double * pivx, struct OneApproxOpts * appargs, void * optargs)
{
    //printf("in maxvolqmarray!\n");

    int info = 0;

    size_t r = A->ncols;

    struct Qmarray * L = qmarray_alloc(A->nrows, r);
    double * U = calloc_double(r * r);
    
    struct Qmarray * Acopy = qmarray_copy(A);
    
    if (VQMAMAXVOL){
        printf("==*=== \n\n\n\n In MAXVOL \n");
        /* size_t ll; */
        /* for (ll = 0; ll < A->nrows * A->ncols; ll++){ */
        /*     printf("%G\n", generic_function_norm(Acopy->funcs[ll])); */
        /* } */
    }

    //print_qmarray(Acopy,0,NULL);
    info =  qmarray_lu1d(Acopy,L,U,pivi,pivx,appargs,optargs);
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
    /* if (VQMAMAXVOL){ */
    /*     printf("built U\nn"); */
    /* } */

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
        
    /* if (VQMAMAXVOL){ */
    /*     printf("took pseudoinverse \nn"); */
    /* } */
    struct Qmarray * B = qmam(A,Asinv,r);

    /* if (VQMAMAXVOL){ */
    /*     printf("did qmam \n"); */
    /* } */
    size_t maxrow, maxcol;
    double maxx, maxval, maxval2;
    
    //printf("B size = (%zu,%zu)\n",B->nrows, B->ncols);
    // BLAH 2

    /* if (VQMAMAXVOL){ */
    /*     printf("do absmax1d\n"); */
    /* } */
    qmarray_absmax1d(B, &maxx, &maxrow, &maxcol, &maxval,optargs);
    if (VQMAMAXVOL){
        printf("maxx=%G,maxrow=%zu maxcol=%zu maxval=%G\n",maxx,maxrow, maxcol, maxval);
    }
    size_t maxiter = 30;
    size_t iter =0;
    double delta = 1e-2;
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
        qmarray_absmax1d(B, &maxx, &maxrow, &maxcol, &maxval2,optargs);
        if (VQMAMAXVOL){
            printf("maxx=%G,maxrow=%zu maxcol=%zu maxval=%G\n",maxx,maxrow, maxcol, maxval2);
        }
        
        /* if (fabs(maxval2-maxval)/fabs(maxval) < 1e-2){ */
        /*     break; */
        /* } */
        if (iter > maxiter){
            if (VQMAMAXVOL){
                printf("maximum iterations (%zu) reached\n",maxiter);
            }
            break;
        }
        maxval = maxval2;
        //printf("maxval=%G\n",maxval);
        iter++;
    }

    if (iter > 0){
        pivi[maxcol] = maxrow;
        pivx[maxcol] = maxx;
    }
    
    if (VQMAMAXVOL){
        printf("qmarray_maxvol indices and pivots before removing duplicates\n");
        iprint_sz(r,pivi);
        dprint(r,pivx);
    }
    //printf("done\n");
    //if ( r < 0){
    if ( r > 1){
        double lb = generic_function_get_lb(A->funcs[0]);
        double ub = generic_function_get_ub(A->funcs[0]);
        remove_duplicates(r, &pivi, &pivx,lb,ub);
        for (ii = 0; ii < r; ii++){
            for (jj = 0; jj < r; jj++){
                U[jj * r + ii] = generic_function_1d_eval(
                        A->funcs[jj*A->nrows + pivi[ii]], pivx[ii]);
            }
        }
        pinv(r,r,r,U,Asinv,ZEROTHRESH);
    }

    if (VQMAMAXVOL){
        printf("qmarray_maxvol indices and pivots after removing duplicates\n");
        iprint_sz(r,pivi);
        dprint(r,pivx);
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

    \param[in,out] A - qmarray to triangularize (overwritten)
    \param[in,out] E - qmarray with orthonormal columns
    \param[in,out] V - allocated space for a qmarray (will store reflectors)
    \param[in,out] R - allocated space upper triangular factor

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
    //print_qmarray(E,0,NULL);
    //printf("qm array\n");
    struct GenericFunction ** vfuncs = NULL;
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
        quasimatrix_free_funcs(v);
        
        /* free(v->funcs);v->funcs=NULL; */
        //printf("array daxpby nrows = %zu\n",A->nrows);
        //printf("epostalpha_qma = \n");
        //print_generic_function(E->funcs[ii*E->nrows],3,NULL);
        //printf("Apostalpha_qma = \n");
        //print_generic_function(A->funcs[ii*A->nrows],3,NULL);

        vfuncs = generic_function_array_daxpby(A->nrows,rho,1,
                                              E->funcs+ii*E->nrows,-1.0,
                                              1, A->funcs+ii*A->nrows);
        quasimatrix_set_funcs(v,vfuncs);
        
        /* printf("sup!\n"); */
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
                struct GenericFunction ** tvfuncs = NULL;
                tvfuncs = quasimatrix_get_funcs(tempv);
                //printf("compute temp\n");
                //print_quasimatrix(tempv, 0, NULL);
                temp1 =  generic_function_inner_sum(A->nrows,1,tvfuncs,1,
                                                    E->funcs + jj*E->nrows);
                /* quasimatrix_set_funcs(tempv,tvfuncs); */
                
                //printf("temp1= %G\n",temp1);
                /* generic_function_array_free(vfuncs,A->nrows); */
                quasimatrix_free_funcs(v);
                /* vfuncs = quasimatrix_get_funcs(v); */
                vfuncs = generic_function_array_daxpby(A->nrows,1.0,1,tvfuncs,
                                                       -temp1,1,E->funcs+jj*E->nrows);
                quasimatrix_set_funcs(v,vfuncs);
                //printf("k ok= %G\n",temp1);
                quasimatrix_free(tempv);
            }
        }
        //*/

        /* printf("compute sigma\n"); */
        //print_generic_function(v->funcs[0],3,NULL);
        /* sigma = generic_function_array_norm(A->nrows, 1, vfuncs); */
        sigma = quasimatrix_norm(v);
        /* printf("get it!\n"); */
        if (sigma < ZEROTHRESH){
            sigma = 0.0;
        }

        if (VQMAHOUSEHOLDER){
            printf(" \t ... sigma = %3.15G ZEROTHRESH=%3.15G\n",
                            sigma,ZEROTHRESH);
        }
            //printf(" \t ... sigma = %3.15G ZEROTHRESH=%3.15G\n",
            //                sigma,ZEROTHRESH);
        
        //printf("v2preqma = \n");
        //print_generic_function(v->funcs[0],3,NULL);
        double ttol = ZEROTHRESH;
        struct GenericFunction ** v2fs;
        if (fabs(sigma) <= ttol){
            //printf("here sigma too small!\n");
            v2 = quasimatrix_alloc(A->nrows);
            quasimatrix_free_funcs(v2);
            /* free(v2->funcs);v2->funcs=NULL; */
            //just a copy
            v2fs = generic_function_array_daxpby(A->nrows,1.0, 1, 
                                                 E->funcs+ii*E->nrows, 0.0, 1, NULL);
            quasimatrix_set_funcs(v2,v2fs);
        }
        else {
            //printf("quasi daxpby\n");
            //printf("sigma = %G\n",sigma);
            /* printf("we are here\n"); */
            v2 = quasimatrix_daxpby(1.0/sigma, v, 0.0, NULL);
            /* printf("and now there\n"); */
            v2fs = quasimatrix_get_funcs(v2);
            //printf("quasi got it daxpby\n");
        }
        quasimatrix_free(v);
        qmarray_set_column(V,ii,v2);
        
        //printf("v2qma = \n");
        //print_generic_function(v2->funcs[0],3,NULL);
        //printf("go into cols \n");
        for (jj = ii+1; jj < A->ncols; jj++){

            if (VQMAHOUSEHOLDER){
                printf("\t On sub-iter %zu\n", jj);
            }
            //printf("Aqma = \n");
            //print_generic_function(A->funcs[jj*A->nrows],3,NULL);
            temp1 = generic_function_inner_sum(A->nrows,1,v2fs,1,
                                               A->funcs+jj*A->nrows);

            //printf("\t\t temp1 pre = %3.15G DBL_EPS=%G\n", temp1,DBL_EPSILON);
            if (fabs(temp1) < ZEROTHRESH){
                temp1 = 0.0;
            }

            atemp = quasimatrix_alloc(A->nrows);
            quasimatrix_free_funcs(atemp);
            struct GenericFunction ** afuncs = quasimatrix_get_funcs(atemp);
            afuncs = generic_function_array_daxpby(A->nrows, 1.0, 1, 
                                                  A->funcs+jj*A->nrows,
                                                  -2.0 * temp1, 1, 
                                                  v2fs);
            quasimatrix_set_funcs(atemp,afuncs);
            
            //printf("atemp = \n");
            //print_generic_function(atemp->funcs[0], 0, NULL);
            R[jj * A->ncols + ii] = generic_function_inner_sum(E->nrows, 1,
                        E->funcs + ii*E->nrows, 1, afuncs);


            //double aftemp = 
            //    generic_function_array_norm(A->nrows,1,atemp->funcs);

            //double eftemp = 
            //    generic_function_array_norm(E->nrows,1,E->funcs+ii*E->nrows);

            if (VQMAHOUSEHOLDER){
                //printf("\t \t ... temp1 = %3.15G\n",temp1);
                //printf("\t\t e^T atemp= %3.15G\n", R[jj*A->ncols+ii]);
            }

            v = quasimatrix_alloc(A->nrows);
            quasimatrix_free_funcs(v);
            /* free(v->funcs);v->funcs=NULL; */
            vfuncs = generic_function_array_daxpby(A->nrows, 1.0, 1, 
                                                   afuncs,
                                                   -R[jj * A->ncols + ii], 1,
                                                   E->funcs + ii*E->nrows);
            quasimatrix_set_funcs(v,vfuncs);


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
    /* printf("done?\n"); */
    //printf("qmarray v after = \n");
    //print_qmarray(V,0,NULL);

    //printf("qmarray E after = \n");
    //print_qmarray(E,0,NULL);
    /* printf("\n\n\n\n\n\n\n\n\n Ending QMARRAY_HOUSEHOLDER\n"); */
    return 0;
}

/***********************************************************//**
    Compute the Q for the QR factor from householder reflectors
    for a Quasimatrix Array

    \param[in,out] Q - qmarray E obtained afterm qmarray_householder
                       becomes overwritten
    \param[in,out] V - Householder functions, obtained from qmarray_householder

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
        /* printf("INCREMENT COUNTER \n"); */
        for (jj = ii; jj < Q->ncols; jj++){

            temp1 = generic_function_inner_sum(V->nrows,1,V->funcs + ii*V->nrows,
                                        1, Q->funcs + jj*Q->nrows);
            //printf("temp1=%G\n",temp1);    
            q2 = quasimatrix_alloc(Q->nrows);
            quasimatrix_free_funcs(q2);
            struct GenericFunction ** q2funcs = quasimatrix_get_funcs(q2);
            /* printf("lets go\n"); */
            q2funcs = generic_function_array_daxpby(Q->nrows, 1.0, 1, 
                        Q->funcs + jj * Q->nrows, -2.0 * temp1, 1, 
                        V->funcs + ii * V->nrows);
            /* printf("why?!\n"); */
            quasimatrix_set_funcs(q2,q2funcs);
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
    /* printf("done\n"); */
    //printf("************Q after*************\n");
    //print_qmarray(Q, 0, NULL);
    //printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");

    return info;
}

/***********************************************************//**
    Compute the householder triangularization of a 
    quasimatrix array. (LQ= A), Q orthonormal, 
    L lower triangular

    \param[in,out] A - qmarray to triangularize (destroyed 
    \param[in,out] E - qmarray with orthonormal rows
    \param[in,out] V - allocated space for a qmarray (will store reflectors)
    \param[in,out] L - allocated space lower triangular factor

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

    struct GenericFunction ** vfuncs = NULL;
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
        quasimatrix_free_funcs(v);
        
        vfuncs = generic_function_array_daxpby(A->ncols, rho, E->nrows, 
                    E->funcs+ii, -1.0, A->nrows, A->funcs+ii);
        quasimatrix_set_funcs(v,vfuncs);
        
        // improve orthogonality
        //*
        if (ii > 1){
            struct Quasimatrix * tempv = NULL;
            for (jj = 0; jj < ii-1; jj++){
                tempv = quasimatrix_copy(v);
                struct GenericFunction ** tvfuncs = quasimatrix_get_funcs(tempv);
                temp1 =  generic_function_inner_sum(A->ncols,1,tvfuncs,E->nrows,
                                                    E->funcs + jj);
                quasimatrix_free_funcs(v);
                vfuncs = generic_function_array_daxpby(A->ncols,1.0,1,tvfuncs,
                                                       -temp1,E->nrows,E->funcs+jj);
                quasimatrix_set_funcs(v,vfuncs);
                quasimatrix_free(tempv);
            }
        }
        //*/
        sigma = quasimatrix_norm(v);//generic_function_array_norm(v->n, 1, v->funcs);
        if (sigma < ZEROTHRESH){
            sigma = 0.0;
        }
        //
        double ttol = ZEROTHRESH;

        if (VQMAHOUSEHOLDER){
            printf(" \t ... sigma = %G ttol=%G \n",sigma,ttol);
        }
        struct GenericFunction ** v2funcs;
        
        if (fabs(sigma) <= ttol){
            //printf("HERE SIGMA TOO SMALL\n");
            v2 = quasimatrix_alloc(A->ncols);
            quasimatrix_free_funcs(v2);
             //just a copy
            v2funcs = generic_function_array_daxpby(E->ncols,1.0, E->nrows, 
                                                    E->funcs+ii, 0.0, 1, NULL);
            quasimatrix_set_funcs(v2,v2funcs);
        }
        else {
            v2 = quasimatrix_daxpby(1.0/sigma, v, 0.0, NULL);
            v2funcs = quasimatrix_get_funcs(v2);
        }
        quasimatrix_free(v);
        qmarray_set_row(V,ii,v2);

        for (jj = ii+1; jj < A->nrows; jj++){

            if (VQMAHOUSEHOLDER){
                //printf("\t On sub-iter %zu\n", jj);
            }

            temp1 = generic_function_inner_sum(A->ncols,1,v2funcs,
                                               A->nrows, A->funcs+jj);

            if (fabs(temp1) < ZEROTHRESH){
                temp1 = 0.0;
            }
            //if (VQMAHOUSEHOLDER)
            //    printf("\t\t temp1= %3.15G\n", temp1);

            atemp = quasimatrix_alloc(A->ncols);
            quasimatrix_free_funcs(atemp);
            struct GenericFunction ** afuncs = NULL;
            /* free(atemp->funcs); */
            afuncs = generic_function_array_daxpby(A->ncols, 1.0, 
                                                   A->nrows, A->funcs+jj,
                                                   -2.0 * temp1, 1, 
                                                   v2funcs);
            quasimatrix_set_funcs(atemp,afuncs);

            L[ii * A->nrows + jj] = generic_function_inner_sum(E->ncols, 
                                                               E->nrows,
                                                               E->funcs + ii, 1,
                                                               afuncs);

            //if (VQMAHOUSEHOLDER)
            //    printf("\t\t e^T atemp= %3.15G\n", L[ii*A->nrows+jj]);

            v = quasimatrix_alloc(A->ncols);
            quasimatrix_free_funcs(v);
            /* free(v->funcs);v->funcs=NULL; */
            /* vfuncs = quasimatrix_get_funcs(v); */
            vfuncs = generic_function_array_daxpby(A->ncols, 1.0, 1,
                                                   afuncs, -L[ii * A->nrows + jj],
                                                   E->nrows, E->funcs + ii);
            quasimatrix_set_funcs(v,vfuncs);
            
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

    \param[in,out] Q - qmarray E obtained afterm qmarray_householder_rows (overwrite)
    \param[in,out] V - Householder functions, obtained from qmarray_householder

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
            quasimatrix_free_funcs(q2);
            struct GenericFunction ** q2funcs = quasimatrix_get_funcs(q2);
            q2funcs = generic_function_array_daxpby(Q->ncols, 1.0, Q->nrows, 
                        Q->funcs + jj, -2.0 * temp1, V->nrows, 
                        V->funcs + ii);
            quasimatrix_set_funcs(q2,q2funcs);
            
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

    \param[in]     dir - type either "QR" or "LQ"
    \param[in,out] A   - qmarray to triangularize (destroyed in call)
    \param[in,out] R   - allocated space upper triangular factor
    \param[in]     app - approximation arguments

    \return Q 
***************************************************************/
struct Qmarray *
qmarray_householder_simple(char * dir,struct Qmarray * A,double * R,
                           struct OneApproxOpts * app)
{
    
    size_t ncols = A->ncols;

    // generate two quasimatrices needed for householder
    struct Qmarray * Q = NULL;
    if (strcmp(dir,"QR") == 0){
        
        Q = qmarray_orth1d_columns(A->nrows,ncols,app);

        
        if (app->fc != PIECEWISE){
            int out = qmarray_qr(A,&Q,&R,app);
            assert (out == 0);
        }
        else{
            struct Qmarray * V = qmarray_alloc(A->nrows,ncols);
            int out = qmarray_householder(A,Q,V,R);
            assert(out == 0);
            out = qmarray_qhouse(Q,V);
            assert(out == 0);
            qmarray_free(V);
        }
    }
    else if (strcmp(dir, "LQ") == 0){
        /* printf("lets go LQ!\n"); */
        Q = qmarray_orth1d_rows(A->nrows,ncols,app);
        /* printf("got orth1d_rows!\n"); */
        if (app->fc != PIECEWISE){
            //free(R); R = NULL;
            int out = qmarray_lq(A,&Q,&R,app);
            assert (out == 0);
        }
        else{
            struct Qmarray * V = qmarray_alloc(A->nrows,ncols);
            int out = qmarray_householder_rows(A,Q,V,R);
            /* printf("\n\n\n\n"); */
            /* printf("right after qmarray_householder_rows \n"); */
            /* print_qmarray(Q,0,NULL); */

            /* printf("\n\n\n\n"); */
            /* printf("VV  right after qmarray_householder_rows \n"); */
            /* print_qmarray(V,0,NULL); */

            assert(out == 0);
            out = qmarray_qhouse_rows(Q,V);

            /* printf("\n\n\n\n"); */
            /* printf("right after qmarray_qhouse_rows \n"); */
            /* print_qmarray(Q,0,NULL); */
            /* printf("\n\n\n\n"); */

            assert(out == 0);
            qmarray_free(V); V = NULL;
        }
    }
    else{
        fprintf(stderr, "No clear QR/LQ decomposition for type=%s\n",dir);
        exit(1);
    }
    return Q;
}

/***********************************************************
    Compute the householder triangularization of a 
    qmarray. for nodal basis (grid) functions of a fixed
    grid

    \param[in]     dir - type either "QR" or "LQ"
    \param[in,out] A   - qmarray to triangularize (destroyed in call)
    \param[in,out] R   - allocated space upper triangular factor
    \param[in]     grid - grid of locations at which to

    \return Q 
***************************************************************/
/* struct Qmarray * */
/* qmarray_householder_simple_grid(char * dir, struct Qmarray * A, double * R, */
/*                                 struct c3Vector * grid) */
/* { */
    
/*     for (size_t ii = 0; ii < A->nrows * A->ncols; ii++){ */
/*         assert(A->funcs[ii]->fc == LINELM); */
/*     } */

/*     struct Qmarray * Q = NULL; */
/*     if (strcmp(dir,"QR") == 0){ */
/*         Q = qmarray_orth1d_linelm_grid(A->nrows, A->ncols, grid); */
/*         int out = qmarray_qr(A,&Q,&R); */
/*         assert (out == 0); */
/*     } */
/*     else if (strcmp(dir, "LQ") == 0){ */
/*         struct Qmarray * temp = qmarray_orth1d_linelm_grid(A->nrows,A->ncols,grid); */
/*         Q = qmarray_transpose(temp); */
/*         int out = qmarray_lq(A,&Q,&R); */
/*         assert(out == 0); */
/*         qmarray_free(temp); temp = NULL; */
/*     } */
/*     else{ */
/*         fprintf(stderr, "No clear QR/LQ decomposition for type=%s\n",dir); */
/*         exit(1); */
/*     } */
/*     return Q; */
/* } */

/***********************************************************//**
    Compute the svd of a quasimatrix array Udiag(lam)vt = A

    \param[in,out] A    - qmarray to get SVD (destroyed)
    \param[in,out] U    - qmarray with orthonormal columns
    \param[in,out] lam  - singular values
    \param[in,out] vt   - matrix containing right singular vectors
    \param[in]     opts - options for approximations

    \return info - if not == 0 then error
***************************************************************/
int qmarray_svd(struct Qmarray * A, struct Qmarray ** U, double * lam, 
                double * vt, struct OneApproxOpts * opts)
{

    int info = 0;

    double * R = calloc_double(A->ncols * A->ncols);
    struct Qmarray * temp = qmarray_householder_simple("QR",A,R,opts);

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

    \param[in,out] A     - qmarray to get SVD (destroyed)
    \param[in,out] U     - qmarray with orthonormal columns
    \param[in,out] lam   - singular values 
    \param[in,out] vt    - matrix containing right singular vectors
    \param[in,out] delta - threshold
    \param[in]     opts  - approximation options

    \return rank - rank of the qmarray
    
    \note
        *U*, *lam*, and *vt* are allocated in this function
***************************************************************/
size_t qmarray_truncated_svd(
    struct Qmarray * A, struct Qmarray ** U, 
    double ** lam, double ** vt, double delta,
    struct OneApproxOpts * opts)
{

    //int info = 0;
    double * R = calloc_double(A->ncols * A->ncols);
    struct Qmarray * temp = qmarray_householder_simple("QR",A,R,opts);

    double * u = NULL;
    //printf("R = \n");
    //dprint2d_col(A->ncols,A->ncols,R);
    size_t rank = truncated_svd(A->ncols, A->ncols, A->ncols, 
                                R, &u, lam, vt, delta);
    
    qmarray_free(*U);
    *U = qmam(temp,u,rank);
    
    qmarray_free(temp);
    free(u);
    free(R);
    return rank;
}

/***********************************************************//**
    Compute the location of the (abs) maximum element in aqmarray consisting of 1 dimensional functions

    \param[in]     qma     - qmarray 1
    \param[in,out] xmax    - x value at maximum
    \param[in,out] rowmax  - row of maximum
    \param[in,out] colmax  - column of maximum
    \param[in,out] maxval  - absolute maximum value
    \param[in]     optargs - optimization arguments
***************************************************************/
void qmarray_absmax1d(struct Qmarray * qma, double * xmax, size_t * rowmax,
                      size_t * colmax, double * maxval, void * optargs)
{
    size_t combrowcol;
    *maxval = generic_function_array_absmax(qma->nrows * qma->ncols,
                                            1, qma->funcs, 
                                            &combrowcol, xmax,optargs);
    
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

    \param[in] a - qmarray 1
    \param[in] b - qmarray 2

    \return c - \f$ [a b] \f$
***************************************************************/
struct Qmarray * qmarray_stackh(const struct Qmarray * a,const struct Qmarray * b)
{
    assert (a != NULL);
    assert (b != NULL);
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

    \param[in] a - qmarray 1
    \param[in] b - qmarray 2

    \return  c : \f$ [a; b] \f$
***************************************************************/
struct Qmarray * qmarray_stackv(const struct Qmarray * a, const struct Qmarray * b)
{
    assert (a != NULL);
    assert (b != NULL);
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

    \param[in] a - qmarray

    \return qmarray of functions representing the derivatives at all values x
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

/***********************************************************//**
    Compute the second derivative of every function in the qmarray

    \param[in] a - qmarray

    \return qmarray of functions representing the derivatives at all values x
***************************************************************/
struct Qmarray * qmarray_dderiv(struct Qmarray * a)
{

    struct Qmarray * b = qmarray_alloc(a->nrows, a->ncols);
    size_t ii;
    for (ii = 0; ii < a->nrows*a->ncols; ii++){
        b->funcs[ii] = generic_function_dderiv(a->funcs[ii]);
    }
    return b;
}

/***********************************************************//**
    Compute the second derivative of every function in the qmarray enforcing periodic boundaries

    \param[in] a - qmarray

    \return qmarray of functions representing the derivatives at all values x
***************************************************************/
struct Qmarray * qmarray_dderiv_periodic(struct Qmarray * a)
{

    struct Qmarray * b = qmarray_alloc(a->nrows, a->ncols);
    size_t ii;
    for (ii = 0; ii < a->nrows*a->ncols; ii++){
        b->funcs[ii] = generic_function_dderiv_periodic(a->funcs[ii]);
    }
    return b;
}

/***********************************************************//**
    Evaluate the derivative of every function in the qmarray

    \param[in]     a   - qmarray
    \param[in]     x   - location at which to evaluate
    \param[in,out] out - evaluations (previously allocated)
***************************************************************/
void qmarray_deriv_eval(const struct Qmarray * a, double x, double * out)
{


    size_t ii;
    for (ii = 0; ii < a->nrows*a->ncols; ii++){
        out[ii] = generic_function_deriv_eval(a->funcs[ii],x);
    }
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


/***********************************************************//**
    Evaluate a qmarray

    \param[in]     qma - quasimatrix array
    \param[in]     x   - location at which to evaluate
    \param[in,out] out - allocated array to store output (qma->nrows * qma->ncols)
***************************************************************/
void qmarray_eval(struct Qmarray * qma, double x, double * out)
{
    size_t ii;
    for (ii = 0; ii < qma->nrows*qma->ncols; ii++){
        out[ii] = generic_function_1d_eval(qma->funcs[ii],x);
    }
}

/***********************************************************//**
    Get number of parameters describing a particular function
    within a quasimatrix array

    \param[in] qma - quasimatrix array
    \param[in] row - row
    \param[in] col - col

    \returns number of parameters
***************************************************************/
size_t qmarray_func_get_nparams(const struct Qmarray * qma,
                                size_t row, size_t col)
{
    struct GenericFunction * gf = qma->funcs[row + col * qma->nrows];
    return generic_function_get_num_params(gf);
}
                                

/***********************************************************//**
    Get number of parameters describing the qmarray

    \param[in]     qma   - quasimatrix array
    \param[in,out] nfunc - number of parameters in the 
                           function with most parameters

    \returns - total number of parameters
***************************************************************/
size_t qmarray_get_nparams(const struct Qmarray * qma, size_t * nfunc)
{

     size_t size = qma->nrows*qma->ncols;
     size_t ntot = 0;
     size_t maxfun = 0;
     for (size_t ii = 0; ii < size; ii++){
         size_t nparam = generic_function_get_num_params(qma->funcs[ii]);
         ntot += nparam;
         
         if (nparam > maxfun){
             maxfun = nparam;
         }
     }
     if (nfunc != NULL){
         *nfunc = maxfun;
     }
     
     return ntot;
}

/***********************************************************//**
    Get Qmarray parameters

    \param[in]      qma     - quasimatrix array whose parameters to update
    \param[in,out]  param   - location to store parameters

    \returns total number of parameters
***************************************************************/
size_t qmarray_get_params(struct Qmarray * qma, double * param)
{
     size_t size = qma->nrows*qma->ncols;
     size_t onind = 0;
     size_t nparam;
     for (size_t ii = 0; ii < size; ii++){
         if (param == NULL){
             nparam = generic_function_get_num_params(qma->funcs[ii]);
         }
         else{
             nparam = generic_function_get_params(qma->funcs[ii],
                                                  param+onind);
         }
         onind += nparam;
     }
     return onind;
}

/***********************************************************//**
    Update parameters of a single generic function

    \param[in,out]  qma    - quasimatrix array whose parameters to update
    \param[in]      row    - row of the function to update
    \param[in]      col    - column of the function
    \param[in]      nparam - total number of parameters
    \param[in]      param  - parameters with which to update
***************************************************************/
void qmarray_uni_update_params(struct Qmarray * qma, size_t row, size_t col,
                           size_t nparam, const double * param)
{


     size_t ind = col * qma->nrows + row;
     size_t nparam_expect = generic_function_get_num_params(qma->funcs[ind]);
     if (nparam_expect != nparam){
         fprintf(stderr, "Trying to update a univariate function in\n");
         fprintf(stderr, "qmarray_uni_update_params with the wrong number of parameters\n");
         exit(1);
     }
     generic_function_update_params(qma->funcs[ind],nparam,param);
}

/***********************************************************//**
    Update Qmarray parameters

    \param[in,out]  qma     - quasimatrix array whose parameters to update
    \param[in]      nparams - total number of parameters
    \param[in]      param   - parameters with which to update
***************************************************************/
void qmarray_update_params(struct Qmarray * qma, size_t nparams, const double * param)
{

     size_t size = qma->nrows*qma->ncols;
     size_t onind = 0;
     for (size_t ii = 0; ii < size; ii++){
         size_t nparam = generic_function_get_num_params(qma->funcs[ii]);
         generic_function_update_params(qma->funcs[ii],nparam,param+onind);
         onind += nparam;
     }
     if (onind != nparams){
         fprintf(stderr,"Error updating the parameters of a Qmarray\n");
         fprintf(stderr,"number of parameters expected is not the\n");
         fprintf(stderr,"number of actual parameters\n");
         exit(1);
     }
}

/***********************************************************//**
    Evaluate the gradient of a qmarray with respect to the parameters
    of the function at a set of evaluation locations

    \param[in]     qma       - quasimatrix array
    \param[in]     N         - number of locations at which to evaluate
    \param[in]     x         - location at which to evaluate
    \param[in]     incx      - increment between inputs
    \param[in,out] out       - allocated array to store output (qma->nrows * qma->ncols)
    \param[in]     incout    - increment of output
    \param[in,out] grad      - compute gradient if not null (store gradient here)
    \param[in]     incg      - incremement of gradient
    \param[in,out] workspace - big enough to handle the largest number of params of any one function 
***************************************************************/
void qmarray_param_grad_eval(struct Qmarray * qma, size_t N,
                             const double * x, size_t incx,
                             double * out, size_t incout,
                             double * grad, size_t incg,
                             double * workspace)
{
    size_t ii;
    /* printf("\t cmoooonn\n"); */
    size_t size = qma->nrows*qma->ncols;
    /* printf("\t evaluate! size=%zu\n",size); */
    if (out != NULL){
        /* printf("1darray_eval2N\n"); */
        generic_function_1darray_eval2N(size,qma->funcs,N,x,incx,out,incout);
        /* printf("got 1darray_eval2N incout = %zu\n",incout); */
        for (ii = 0; ii < size; ii++){
            if (isnan(out[ii])){
                fprintf(stderr,"Warning, evaluation in qmarray_param_grad_eval is nan\n");
                fprintf(stderr,"ii=%zu,size=%zu,incout=%zu\n",ii,size,incout);
                fprintf(stderr,"x = %G\n",x[ii*incx]);
                exit(1);
            }
            else if (isinf(out[ii])){
                fprintf(stderr,"Warning, evaluation in qmarray_param_grad_eval inf\n");
                exit(1);
            }
        }
        /* printf("finished checking!\n"); */
        
        /* printf("out = "); dprint(size,out); */
    }

    if (grad != NULL){
        for (size_t jj = 0; jj < N; jj++){
            size_t onparam = 0;
            for (ii = 0; ii < size; ii++){
                size_t nparam = generic_function_get_num_params(qma->funcs[ii]);
                generic_function_param_grad_eval(qma->funcs[ii],1,x + jj*incx,workspace);
                for (size_t kk = 0; kk < nparam; kk++){
                    // only one element changes so copy everything
                    for(size_t zz = 0; zz < size; zz++){
                        grad[onparam*size+zz + jj * incg] = 0.0;
                    }
                    // and change the one element
                    grad[onparam*size+ii + jj * incg] = workspace[kk];
                    if (isnan(workspace[kk])){
                        fprintf(stderr,"Warning, gradient in qmarray_param_grad_eval is nan\n");
                        exit(1);
                    }
                    else if (isinf(workspace[kk])){
                        fprintf(stderr,"Warning, gradient in qmarray_param_grad_eval is inf\n");
                        exit(1);
                    }

                    onparam++;
                }
            }
        }
    }
    /* printf("done there\n"); */
}

/***********************************************************//**
    Evaluate the gradient of a qmarray with respect to the parameters
    of the function at a set of evaluation locations

    \param[in]     qma       - quasimatrix array
    \param[in]     N         - number of locations at which to evaluate
    \param[in,out] out       - allocated array to store output (qma->nrows * qma->ncols)
    \param[in]     incout    - increment of output
    \param[in]     grad      - computed gradient
    \param[in]     incg      - incremement of gradient

***************************************************************/
void qmarray_linparam_eval(struct Qmarray * qma, size_t N,
                           double * out, size_t incout,
                           const double * grad, size_t incg)

{

    size_t size = qma->nrows*qma->ncols;
    size_t nparam,onparam;
    size_t ii,jj,indout,indgrad;
    double * param;
    for (ii = 0; ii < N; ii++){
        /* printf("ii = %zu\n",ii); */
        onparam=0;
        indout = incout*ii;
        indgrad = ii*incg;
        for (jj = 0; jj < size; jj++){
            param = generic_function_get_params_ref(qma->funcs[jj],&nparam);
            /* dprint(nparam,grad+ii*incg + onparam); */
            out[jj + indout] = cblas_ddot(nparam,param,1,
                                          grad + indgrad + onparam, 1);
            onparam += nparam;
        }
    }
}

/***********************************************************//**
    Sum the L2 norms of each function

    \param[in]     qma   - quasimatrix array
    \param[in]     scale - increment between inputs
    \param[in,out] grad  - compute gradient if not null (store gradient here)

    \returns sum of scale * inner_product of each function 
    \note 
    If gradient is not null then it adds scaled values of the new gradients to the
    existing gradient. Gradient is stored function first, row second, column third
***************************************************************/
double qmarray_param_grad_sqnorm(struct Qmarray * qma, double scale, double * grad)
{
    /* printf("\t cmoooonn\n"); */
    size_t size = qma->nrows*qma->ncols;
    double out = 0.0;
    if (grad == NULL){
        for (size_t ii = 0; ii < size; ii++){
            out += scale * generic_function_inner(qma->funcs[ii],qma->funcs[ii]);
        }
    }
    else{
        size_t runparam=0;
        for (size_t ii = 0; ii < size; ii++){
            out += scale * generic_function_inner(qma->funcs[ii],qma->funcs[ii]);
            size_t nparam = generic_function_get_num_params(qma->funcs[ii]);
            int res = generic_function_squared_norm_param_grad(qma->funcs[ii],
                                                               scale, grad+runparam);
            assert (res == 0);
            runparam += nparam;
        }
    }

    return out;
}

/***********************************************************//**
    Evaluate the gradient of a qmarray with respect to the parameters
    of the function, store gradient in sparse format, and multiply on the left
    by vector

    \param[in]     qma      - quasimatrix array
    \param[in]     N        - number of locations at which to evaluate
    \param[in]     x        - location at which to evaluate
    \param[in]     incx     - increment between inputs
    \param[in,out] out      - allocated array to store output (qma->nrows * qma->ncols)
    \param[in]     incout   - increment of output
    \param[in,out] grad     - compute gradient if not null (store gradient here)
    \param[in]     incg     - incremement of gradient between each evaluation
    \param[in]     left     - set of vectors to multiply each gradient
    \param[in,out] mult_out - multiplied out values 
    \param[in]     nparam   - expected number of parameters


    \note
       mult_out is a flattened array, increment between values for different
       *x* values is nparam * qma->ncols
       mult_out[]
       
***************************************************************/
void qmarray_param_grad_eval_sparse_mult(struct Qmarray * qma, size_t N,
                                         const double * x, size_t incx,
                                         double * out, size_t incout,
                                         double * grad, size_t incg,
                                         double * left, double * mult_out, size_t nparam)
{

    /* printf("\t cmoooonn\n"); */
    size_t size = qma->nrows*qma->ncols;
    /* printf("\t evaluate! size=%zu\n",size); */
    if (out != NULL){
        generic_function_1darray_eval2N(size,qma->funcs,N,x,incx,out,incout);
    }

    if (grad != NULL){
        size_t inc_out = nparam * qma->ncols;
        for (size_t jj = 0; jj < N; jj++){
            // zero every output
            if (mult_out != NULL){
                for (size_t ii = 0; ii < nparam * qma->ncols; ii++){
                    mult_out[jj * inc_out + ii] = 0.0;
                }
            }
            
            size_t onnum = 0;
            for (size_t ii = 0; ii < qma->ncols; ii++){
                for (size_t kk = 0; kk < qma->nrows; kk++){
                    size_t onfunc = kk + ii * qma->nrows;
                    size_t nparamf = generic_function_get_num_params(qma->funcs[onfunc]);
                    generic_function_param_grad_eval(qma->funcs[onfunc],1,
                                                     x + jj*incx,
                                                     grad + onnum + jj * incg);

                    if (left != NULL){
                        /* printf("%zu,%zu,%zu\n",ii,kk,onnum); */
                        size_t active_left = kk + jj * qma->nrows;


                        // modifying onnum vector
                        size_t on_output = onnum * qma->ncols + jj * inc_out;

                        // only need to update *ith* element 
                        for (size_t ll = 0; ll < nparamf; ll++){
                            size_t modelem = ll * qma->ncols + ii;
                            mult_out[on_output + modelem] =
                                grad[onnum + jj * incg + ll] * left[active_left];
                        }
                    }
                    onnum += nparamf;
                }
            }
            if (mult_out != NULL){
                assert (nparam == onnum);
            }
        }
    }
    /* printf("done there\n"); */
}


/***********************************************************//**
    Interpolate a qmarray using a nodal basis at particular values

    \param[in] qma - quasimatrix array
    \param[in] N   - number of nodes to form nodal basis
    \param[in] x   - locations of nodal basis

    \return Qmarray with interpolated functions at a nodal basis
***************************************************************/
struct Qmarray * qmarray_create_nodal(struct Qmarray * qma, size_t N, double * x)
{
    struct Qmarray * qmaout = qmarray_alloc(qma->nrows,qma->ncols);
    for (size_t ii = 0; ii < qma->nrows * qma->ncols; ii++){
        qmaout->funcs[ii] = generic_function_create_nodal(qma->funcs[ii],N,x);
    }
    return qmaout;
}


/***********************************************************//**
    Determine whether kristoffel weighting is active

    \param[in] qma - qmarray

    \return 1 if active, 0 otherwise

    \note It is assumed that if kristoffel is active on one function
    it is active on all
***************************************************************/
int qmarray_is_kristoffel_active(const struct Qmarray * qma)
{

    int active = generic_function_is_kristoffel_active(qma->funcs[0]);
    for (size_t ii = 1; ii < qma->nrows * qma->ncols; ii++){
        if (generic_function_is_kristoffel_active(qma->funcs[ii]) != active){
            fprintf(stderr, "Kristoffel weighting cannot be active on some univariate functions\n");
            fprintf(stderr, "and inactive on other ones\n");
            exit(1);
        }
    }
    return active;
}

void qmarray_activate_kristoffel(struct Qmarray * qma)
{
    for (size_t ii = 0; ii < qma->nrows * qma->ncols; ii++){
        generic_function_activate_kristoffel(qma->funcs[ii]);
    }
}

void qmarray_deactivate_kristoffel(struct Qmarray * qma)
{
    for (size_t ii = 0; ii < qma->nrows * qma->ncols; ii++){
        generic_function_deactivate_kristoffel(qma->funcs[ii]);
    }
}


/***********************************************************//**
    Get the kristoffel normalization factor                                                            

    \param[in] qma - quasimatrix array
    \param[in] x   - location at which to obtain normalization

    \return normalization factor

    \note Assumes that each univariate function has the same weight
***************************************************************/
double qmarray_get_kristoffel_weight(const struct Qmarray * qma,
                                     double x)
{
    double weight = generic_function_get_kristoffel_weight(qma->funcs[0],x);
    /* printf("qmarray kri weight = %G\n",weight); */
    return weight;
}




/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////


int qmarray_qr(struct Qmarray * A, struct Qmarray ** Q, double ** R, 
               struct OneApproxOpts * app)
{

    size_t ii,kk,ll;

    size_t nrows = A->nrows;
    size_t ncols = A->ncols;
    if ( (*Q) == NULL){
        *Q = qmarray_orth1d_columns(nrows,ncols,app);
    }

   // print_qmarray(*Q,0,NULL);
    
    if ((*R) == NULL){
        *R = calloc_double(ncols*ncols);
    }
    
    struct Qmarray * V = qmarray_alloc(nrows,ncols);
    for (ii = 0; ii < nrows * ncols; ii++){
        V->funcs[ii] = generic_function_alloc_base(1);    
    }
    
    /* printf("\n\n\n ------------------------------ \n"); */
    /* printf("in qmarray QR\n"); */
    /* for (size_t ii = 0; ii < ncols; ii++){ */
    /*     struct GenericFunction * gf = qmarray_get_func(*Q,0,ii); */
    /*     double integral = generic_function_integral(gf); */
    /*     double norm = generic_function_norm(gf); */
    /*     printf("\nintegral = %G, norm=%G\n",integral,norm); */
    /*     /\* print_generic_function(gf,0,NULL); *\/ */
    /* } */


    double s, rho, alpha, sigma,v;
    for (ii = 0; ii < ncols; ii++){
       /* printf("ii = %zu \n", ii); */

        rho = generic_function_array_norm(nrows,1,A->funcs+ii*nrows);
        (*R)[ii*ncols+ii] = rho;
        alpha = generic_function_inner_sum(nrows,1,
                                           (*Q)->funcs+ii*nrows,1,
                                           A->funcs+ii*nrows);
        
        /* printf("\t rho=%G, \t  alpha=%G\n",rho,alpha); */
        if (fabs(alpha) < ZERO){
            alpha = 0.0;
            s = 0.0;
        }
        else{
            s = -alpha / fabs(alpha);    
            if (s < 0.0){
                generic_function_array_flip_sign(nrows,1,
                                                 (*Q)->funcs+ii*nrows);
            }
        }

//        printf("s = %G\n",s);

        for (kk = 0; kk < nrows; kk++){
            generic_function_weighted_sum_pa(
                rho,(*Q)->funcs[ii*nrows+kk],-1.0,
                A->funcs[ii*nrows+kk],
                &(V->funcs[ii*nrows+kk]));
        }

        //improve orthogonality
        for (size_t zz = 0; zz < ii; zz++){
            v = generic_function_inner_sum(nrows,1,
                                           V->funcs+ii*nrows,
                                           1, (*Q)->funcs+zz*nrows);
            generic_function_array_axpy(nrows,-v,(*Q)->funcs+zz*nrows,
                                    V->funcs+ii*nrows);
        }

        sigma = 0.0;
        for (kk = 0; kk < nrows; kk++){
            sigma += generic_function_inner(V->funcs[ii*nrows+kk],
                                            V->funcs[ii*nrows+kk]);
        }

        /* printf("\t sigma = %G\n",sigma); */
        sigma = sqrt(sigma);
        // weird that this line is needed for stabilization
        // Need to add reorthoganlization
        //sigma = sigma + 1e-16;
        if (fabs(sigma) < ZERO){
            //printf("less than zero sigma = %G,Zero=%G\n",sigma,ZERO);
            //V=E
            for (kk = 0; kk < nrows; kk++){
                generic_function_free(V->funcs[ii*nrows+kk]);
                V->funcs[ii*nrows+kk] = NULL;
                V->funcs[ii*nrows+kk] = generic_function_copy(
                    (*Q)->funcs[ii*nrows+kk]);
                //generic_function_copy_pa((*Q)->funcs[ii*nrows+kk],V->funcs[ii*nrows+kk]);
            }
        }
        else{
            /* printf("\t scale that bad boy, sigma less? \n"); */
            generic_function_array_scale(1.0/sigma,V->funcs+ii*nrows,
                                         nrows);
        }
        
        //printf("start inner loop\n");
        double ev = generic_function_inner_sum(nrows,1,
                                               (*Q)->funcs+ii*nrows,1,
                                               V->funcs+ii*nrows);
        for (kk = ii+1; kk < ncols; kk++){
            double temp = 
                generic_function_inner_sum(nrows,1,
                                           V->funcs+ii*nrows,1,
                                           A->funcs+kk*nrows);
            (*R)[kk*ncols+ii]=
                generic_function_inner_sum(nrows,1,
                                           (*Q)->funcs+ii*nrows,1,
                                           A->funcs+kk*nrows);
            (*R)[kk*ncols+ii] += (-2.0 * ev * temp);
            for (ll = 0; ll < nrows; ll++){
                int success =
                    generic_function_axpy(-2.0*temp,
                                          V->funcs[ii*nrows+ll],
                                          A->funcs[kk*nrows+ll]);
                if (success == 1){
                    return 1;
                }
                //assert (success == 0);
                success = generic_function_axpy(
                    -(*R)[kk*ncols+ii],
                    (*Q)->funcs[ii*nrows+ll],
                    A->funcs[kk*nrows+ll]);

                if (success == 1){
                    return 1;
                }
//                assert (success == 0);
            }
        }
    }

    /* printf("\nafter getting the reflectors V = \n"); */
    /* for (size_t ii = 0; ii < ncols; ii++){ */
    /*     struct GenericFunction * gf = qmarray_get_func(V,0,ii); */
    /*     double integral = generic_function_integral(gf); */
    /*     double norm = generic_function_norm(gf); */
    /*     printf("integral = %G, norm=%G\n",integral,norm); */
    /*     /\* print_generic_function(gf,0,NULL); *\/ */
    /* } */

    /* printf("\n ------------------------------ \n\n\n");     */
    // form Q
    size_t counter = 0;
    ii = ncols-1;
    size_t jj;
    while (counter < ncols){
        
        for (jj = ii; jj < ncols; jj++){
            double val = generic_function_inner_sum(nrows,1,
                    (*Q)->funcs+jj*nrows, 1, V->funcs+ii*nrows);

            for (ll = 0; ll < nrows; ll++){
                generic_function_axpy(-2.0*val,V->funcs[ii*nrows+ll],
                                        (*Q)->funcs[jj*nrows+ll]);

            }
        }

        ii = ii - 1;
        counter += 1;
    }
    
    qmarray_free(V); V = NULL;
    return 0;
}

int qmarray_lq(struct Qmarray * A, struct Qmarray ** Q, double ** L,
               struct OneApproxOpts * app)
{

    size_t ii,kk,ll;

    size_t nrows = A->nrows;
    size_t ncols = A->ncols;

    if ( (*Q) == NULL){
        *Q = qmarray_orth1d_rows(nrows,ncols,app);
    }

    if ((*L) == NULL){
        *L = calloc_double(nrows*nrows);
    }
    
    struct Qmarray * V = qmarray_alloc(nrows,ncols);
    for (ii = 0; ii < nrows * ncols; ii++){
        V->funcs[ii] = generic_function_alloc_base(1);    
    }

    double s, rho, alpha, sigma, v;
    for (ii = 0; ii < nrows; ii++){
        //printf("ii = %zu \n", ii);

        rho = generic_function_array_norm(ncols,nrows,A->funcs+ii);
        (*L)[ii*nrows+ii] = rho;
        alpha = generic_function_inner_sum(ncols,nrows,(*Q)->funcs+ii,nrows, A->funcs+ii);
        
        //printf("alpha=%G\n",alpha);
        if (fabs(alpha) < ZERO){
            alpha = 0.0;
            s = 0.0;
        }
        else{
            s = -alpha / fabs(alpha);    
            if (s < 0.0){
                generic_function_array_flip_sign(ncols,nrows,(*Q)->funcs+ii);
            }
        }
        
        for (kk = 0; kk < ncols; kk++){
            generic_function_weighted_sum_pa(
                rho,(*Q)->funcs[kk*nrows+ii],-1.0,
                A->funcs[kk*nrows+ii],
                &(V->funcs[kk*nrows+ii]));
        }

        //improve orthogonality
        for (size_t zz = 0; zz < ii; zz++){
            v = generic_function_inner_sum(ncols,nrows,
                                           V->funcs+ii,
                                           nrows, (*Q)->funcs+zz);
            for (size_t llz = 0; llz < ncols; llz++){
                generic_function_axpy(-v,(*Q)->funcs[llz*nrows+zz],
                                      V->funcs[llz*nrows+ii]);
            }
        }

        sigma = 0.0;
        for (kk = 0; kk < ncols;kk++){
            sigma += generic_function_inner(V->funcs[kk*nrows+ii],
                                            V->funcs[kk*nrows+ii]);
        }

        sigma = sqrt(sigma);
        // weird that this line is needed for stabilization
        // Need to add reorthoganlization
        //sigma = sigma + ZERO*1e-3;
        if (fabs(sigma) < ZERO){
            //printf("less than zero sigma = %G,Zero=%G\n",sigma,ZERO);
            //V=E
            for (kk = 0; kk < ncols; kk++){
                generic_function_free(V->funcs[kk*nrows+ii]);
                V->funcs[kk*nrows+ii] = NULL;
                V->funcs[kk*nrows+ii] = generic_function_copy((*Q)->funcs[kk*nrows+ii]);
                //generic_function_copy_pa((*Q)->funcs[ii*nrows+kk],V->funcs[ii*nrows+kk]);
            }
        }
        else{
            for (kk = 0; kk < ncols; kk++){
                generic_function_scale(1.0/sigma,V->funcs[kk*nrows+ii]);
            }
        }
        
        //printf("start inner loop\n");
        double ev = generic_function_inner_sum(ncols,nrows,(*Q)->funcs+ii,nrows,V->funcs+ii);
        for (kk = ii+1; kk < nrows; kk++){
            double temp = generic_function_inner_sum(ncols,nrows,V->funcs+ii,nrows,A->funcs+kk);
            (*L)[ii*nrows+kk]=generic_function_inner_sum(ncols,nrows,(*Q)->funcs+ii,nrows,A->funcs+kk);
            (*L)[ii*nrows+kk] += (-2.0 * ev * temp);
            for (ll = 0; ll < ncols; ll++){
                int success = generic_function_axpy(-2.0*temp,V->funcs[ll*nrows+ii],A->funcs[ll*nrows+kk]);
                assert (success == 0);
                success = generic_function_axpy(-(*L)[ii*nrows+kk],
                        (*Q)->funcs[ll*nrows+ii],A->funcs[ll*nrows+kk]);
                assert (success == 0);
            }
        }
    }
    
    // form Q
    size_t counter = 0;
    ii = nrows-1;
    size_t jj;
    while (counter < nrows){
        
        for (jj = ii; jj < nrows; jj++){
            double val = generic_function_inner_sum(ncols,nrows,
                    (*Q)->funcs+jj, nrows, V->funcs+ii);

            for (ll = 0; ll < ncols; ll++){
                generic_function_axpy(-2.0*val,V->funcs[ll*nrows+ii],
                                        (*Q)->funcs[ll*nrows+jj]);

            }
        }

        ii = ii - 1;
        counter += 1;
    }
    
    qmarray_free(V); V = NULL;
    return 0;
}

int qmarray_qr_gs(struct Qmarray * A, double ** R)
{
    size_t nrows = A->nrows;
    size_t ncols = A->ncols;
    if ((*R) == NULL){
        *R = calloc_double(ncols*ncols);
    }
    
    for (size_t ii = 0; ii < ncols; ii++ ){
        (*R)[ii*ncols+ii] = generic_function_array_norm(nrows,1,A->funcs + ii*nrows);
        if ((*R)[ii*ncols+ii] > ZERO){
            generic_function_array_scale(1.0/(*R)[ii*ncols+ii],A->funcs+ii*nrows,nrows);
            for (size_t jj = ii+1; jj < ncols; jj++){
                (*R)[jj*ncols+ii] = generic_function_inner_sum(nrows,1,A->funcs+ii*nrows,
                                                               1, A->funcs + jj*nrows);
                generic_function_array_axpy(nrows,-(*R)[jj*ncols+ii],A->funcs+ii*nrows,A->funcs+jj*nrows);
            }
        }
        else{
            printf("warning!!\n");
            printf("norm = %G\n",(*R)[ii*ncols+ii]);
            assert(1 == 0);
        }
    }
    return 0;
}


int qmarray_lq_gs(struct Qmarray * A, double ** R)
{
    size_t nrows = A->nrows;
    size_t ncols = A->ncols;
    if ((*R) == NULL){
        *R = calloc_double(nrows*nrows);
    }
    
    for (size_t ii = 0; ii < nrows; ii++ ){
        (*R)[ii*nrows+ii] = generic_function_array_norm(ncols,nrows,A->funcs + ii);
        if ((*R)[ii*nrows+ii] > ZERO){
            for (size_t jj = 0; jj < ncols; jj++){
                generic_function_scale(1.0/(*R)[ii*nrows+ii],A->funcs[jj*nrows+ii]);                
            }
            for (size_t jj = ii+1; jj < nrows; jj++){
                (*R)[ii*nrows+jj] = generic_function_inner_sum(ncols,nrows,A->funcs+ii,
                                                               nrows, A->funcs + jj);
                for (size_t kk = 0; kk < ncols; kk++){
                    generic_function_axpy(-(*R)[ii*nrows+jj],
                                          A->funcs[kk*nrows+ii],
                                          A->funcs[kk*nrows+jj]);
                }
            }
        }
        else{
            printf("norm = %G\n",(*R)[ii*nrows+ii]);
            assert (1 == 0);
        }
    }
    return 0;
}


void print_qmarray(struct Qmarray * qm, size_t prec, void * args, FILE *fp)
{

    printf("Quasimatrix Array (%zu,%zu)\n",qm->nrows, qm->ncols);
    printf("=========================================\n");
    size_t ii,jj;
    for (ii = 0; ii < qm->nrows; ii++){
        for (jj = 0; jj < qm->ncols; jj++){
            printf("(%zu, %zu)\n",ii,jj);
            print_generic_function(qm->funcs[jj*qm->nrows+ ii],prec,args,fp);
            printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
        }
    }
}
