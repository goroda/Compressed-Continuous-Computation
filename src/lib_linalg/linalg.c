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






#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "array.h"

#include "linalg.h"

#ifndef LAPACKWARN
    #define LAPACKWARN 0
#endif

double maxabselem(const double * a, size_t * elem, size_t N){
    
    double val = fabs(a[0]);
    double temp;
    size_t ii = 0;
    *elem = ii;
    for (ii = 1; ii < N; ii++){
        temp = fabs(a[ii]);
        if (temp > val){
            val = temp;
            *elem = ii;
        }
    }
    return val;
}

void rows_to_top_col(double *A, double *top, int *rows, size_t m, size_t ncols){
    // COLUMN MAJOR ORDER
    /* ONLY FOR A TALL AND THIN MATRIX  A = m x ncols; len(rows) = ncols*/
    size_t ii, jj;
    for (ii = 0; ii < ncols; ii++){
        for (jj = 0; jj < ncols; jj++){
            top[ii*ncols + jj] = A[ii*m+ (size_t) rows[jj]];
            A[ii*m + (size_t) rows[jj]] = A[ii*m + jj];
            A[ii*m + jj] = top[ii*ncols+jj];
        }
    }
}
void getrows(const double * A, double * top, size_t * rows, size_t arows, size_t acols){
    // column major order
    size_t ii, jj;
    for (jj = 0; jj < acols; jj++){
        for (ii = 0; ii < acols; ii++){
            top[jj*acols + ii] = A[jj*arows+rows[ii]];
        }
    }
}


/***********************************************************//**
    Computes *N* vector-matrix multiplications

    \param[in]     N    - number of evaluations
    \param[in]     r1   - number of rows
    \param[in]     r2   - number of columns
    \param[in]     A    - an N x r1 matrix
    \param[in]     inca - increment between rows of *A*
    \param[in]     B    - an r1 x r2 x N multidimensional array
    \param[in]     incb - increment between the beginning of matrices (r1 x r2, if compact)
    \param[in,out] C    - an N x r2 matrix
    \param[in]     incc - increment between rows of output

    \note
    Computes
    \f[
        C[j,:] = A[j,:]B[:,:,j]
    \f]
***************************************************************/
void c3linalg_multiple_vec_mat(size_t N, size_t r1, size_t r2,
                               const double * A, size_t inca,
                               const double * B, size_t incb, double * C,size_t incc)
{
    for (size_t jj = 0; jj < N; jj++){
        cblas_dgemv(CblasColMajor,CblasTrans,
                    r1,r2,1.0,
                    B + jj * incb, r1,
                    A + jj * inca, 1, 0.0, C + jj*incc,1);
    }
}

/***********************************************************//**
    Computes *N* matrix-vector multiplications

    \param[in]     N    - number of evaluations
    \param[in]     r1   - number of rows
    \param[in]     r2   - number of columns
    \param[in]     A    - an N x r2 matrix
    \param[in]     inca - increment between rows of *A*
    \param[in]     B    - an r1 x r2 x N multidimensional array
    \param[in]     incb - increment between the beginning of matrices (r1 x r2, if compact)
    \param[in,out] C    - an N x r1 matrix
    \param[in]     incc - increment between rows of output

    \note
    Computes
    \f[
        (C[j,:])^T = B[:,:,j](A[j,:])^T
    \f]
***************************************************************/
void c3linalg_multiple_mat_vec(size_t N, size_t r1, size_t r2,
                               const double * A, size_t inca,
                               const double * B, size_t incb, double * C,size_t incc)
{
    for (size_t jj = 0; jj < N; jj++){
        cblas_dgemv(CblasColMajor,CblasNoTrans,
                    r1, r2, 1.0, B + jj * incb, r1,
                    A + jj * inca,  1, 0.0, C + jj * incc, 1);
    }
}


/***********************************************************//*
    Function qr

    Purpose: Compute the Q factor of a QR decomposition

    Parameters:
        - M (IN) - Number of rows of a matrix
        - N (IN) - Number of columns of a matrixt
        - a (in/out) - matrix in column order (FORTRAN) 
                     gets becomes the Q factor
        - lda (IN) - LDA (see lapack)

    Returns: info (if != 0 then something is wrong -- see lapack)
***************************************************************/
int
qr(size_t m, size_t n, double * a, size_t lda)
{
    
    // Workspace and status variables
    double * work;
    if (NULL == ( work = malloc(sizeof(double)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }

    size_t tau_size = m < n ? m: n;
    double * tau;
    if (NULL == ( tau = calloc(tau_size, sizeof(double)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }

    int lworkinit = -1;
    int info = 0;

    //get optimal workspace size
    dgeqrf_((int*)&m,(int*)&n, a, (int*)&lda, tau, work, &lworkinit, &info);
    if (info) printf("error in getting workspace for qr %d ",info);

    // optimal workspace is returned in *work
    int lwork = (int) lrint(work[0]); 
    free(work);
    if (NULL == ( work = malloc((size_t)lwork * sizeof(work[0])))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }

    // do actual computation
    dgeqrf_((int*)&m,(int*)&n, a, (int*)&lda, tau, work, &lwork, &info);
    if (info) printf("error in QR computation  %d ",info);
     
    //printf("m=%d n=%d\n ", m, n);
    dorgqr_((int*)&m,(int*)&n, (int *)&tau_size, a,(int*) &lda, tau, work, &lwork, &info); // third element is min(n,m)
    if (info) printf("error in computing Q of QR %d ",info);
    free(work);
    free(tau);

    return info;
}


void
rq_with_rmult(size_t m, size_t n, double * a, size_t lda, size_t bm, 
                size_t bn, double * b, size_t ldb) 
{
    // Workspace and status variables
    double * work;
    if (NULL == ( work = malloc(sizeof(double)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }

    size_t tau_size = m < n ? m: n;
    double * tau;
    if (NULL == ( tau = calloc(tau_size, sizeof(double)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }

    int lworkinit = -1;
    int info = 0;

    //get optimal workspace size
    dgerqf_((int*)&m,(int*)&n, a,(int*) &lda, tau, work, &lworkinit, &info);
    if (info) printf("error in getting workspace for qr %d ",info);

    // optimal workspace is returned in *work
    size_t lwork = (size_t) work[0]; 
    free(work);
    if (NULL == ( work = malloc(lwork * sizeof(work[0])))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }

    // do actual computation
    dgerqf_((int*)&m,(int*)&n, a, (int*)&lda, tau, work, (int*)&lwork, &info);
    if (info) printf("error in QR computation  %d ",info);
     
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, bm, bn, 1.0, 
           a+(n-m)*m, m, b, ldb);

    //printf("m=%d n=%d\n ", m, n);
    dorgrq_((int*)&m,(int*)&n, (int*)&tau_size, a, (int*)&lda, tau, work, (int*)&lwork, &info); // third element is min(n,m)
    if (info) printf("error in computing Q of QR %d ",info);
    free(work);
    free(tau);
}


/*************************************************************//**
    Function svd

    Purpose: Compute the singular value decomposition

    Parameters:
        - M (IN) - Number of rows of a matrix
        - N (IN) - Number of columns of a matrixt
        - lda (IN) - LDA (see lapack)
        - a (in/out) - matrix in column order (FORTRAN) 
                     gets destroyed by SVD call
        - u (in/out) - left singular values in column order
        - s (in/out) - array of singular values 
        - vt (in/out) - right singular values in column order

    Returns: Nothing. (void function)
**************************************************************/
void svd(size_t m, size_t n, size_t lda, double *a, double *u, double *s, 
            double *vt)
{
    size_t numberOfSingularValues = m < n ? m:n;

    // Workspace and status variables
    double * work = calloc_double(1);

    int lworkinit = -1;
    int *iwork;
    if (NULL == ( iwork = malloc(8*numberOfSingularValues*sizeof(int)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    int info = 0;
    

    //get optimal workspace size
    dgesdd_("A", (int*)&m, (int *)&n, a, (int*)&lda, s, u, (int*)&m, vt, (int *)&n, work, &lworkinit, iwork, &info);
    if (info) printf("error in getting workspace %d \n",info);

    // optimal workspace is returned in *work
    size_t lwork = (size_t)work[0]; 
    free(work);
    if (NULL == ( work = malloc(lwork * sizeof(work[0])))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }

    // do actual computation
    dgesdd_("A",(int*)&m,(int *)&n,a,(int*)&lda,s,u,(int*)&m,vt,(int *)&n,work,(int*)&lwork,iwork,&info);
    if (info > 0){
        if (LAPACKWARN == 1){
            if (info) printf("error in SVD computation  %d\n ",info);
        }
    }
    else if (info < 0){
        if (info) printf("error in SVD computation  %d\n ",info);
    }

    free(work);
    free(iwork);
}

/*************************************************************//**
    Compute the truncated singular value decomposition

    \param[in]     m     - Number of rows of a matrix
    \param[in]     n     - Number of columns of a matrixt
    \param[in]     lda   - LDA (see lapack)
    \param[in,out] a     - matrix in column order (FORTRAN) gets 
                           destroyed by SVD call
    \param[in,out] u     - left singular values in column order 
    \param[in,out] s     - array of singular values 
    \param[in,out] vt    - right singular values in column order 
    \param[in]     delta - truncation level such that sum(s^2) <= delta

    \return rank
    
    \note 
        Will never return a rank 0 approximation
**************************************************************/
size_t truncated_svd(size_t m, size_t n, size_t lda, double *a, 
                     double **u, 
                     double **s, double **vt, double delta)
{
    size_t numberOfSingularValues = m < n ? m:n;
    double * u_temp;
    if (NULL == ( u_temp = calloc(m*m,sizeof(double)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    double * vt_temp;
    if (NULL == ( vt_temp = calloc(n*n,sizeof(double)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    double * s_temp;
    if (NULL == ( s_temp = calloc(numberOfSingularValues, sizeof(double)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }

    svd(m, n, lda, a, u_temp, s_temp, vt_temp);

    size_t ii;
    double sum = 0.0;
    size_t rank = numberOfSingularValues;
    /* printf("sing before "); */
    /* dprint(numberOfSingularValues,s_temp); */
    //for (ii = numberOfSingularValues-1; ii >= 0; ii--){
    for (ii = 0; ii < numberOfSingularValues; ii++){
        /* printf("ii = %zu\n",ii); */
        sum+=pow(s_temp[numberOfSingularValues - ii- 1],2);
        if (sum >= delta){
            /* printf("sum=%G delta=%G\n",sum,delta); */
            break;
        } 
        if (rank == 1) break;
        rank -= 1;
    }

    /* printf("num keep = %zu\n ",rank); */

    *u = calloc_double(m*rank);
    memmove((*u), u_temp, rank * m * sizeof(double));
    
    *vt = calloc_double(n*rank);
    for (ii = 0; ii < n; ii++){
        memmove( (*vt)+ii*rank, vt_temp + ii * n, rank * sizeof(double));
    }
    
    *s = calloc_double(rank);
    memmove((*s), s_temp, rank * sizeof(double));
    
    free(u_temp);
    free(vt_temp);
    free(s_temp);
    return rank;
}

/*************************************************************//**
    Compute the Pseudoinverse 

    \param[in]     m      - Number of rows of a matrix
    \param[in]     n      - Number of columns of a matrix
    \param[in]     lda    - stride of matrix
    \param[in,out] a      - matrix in column order (gets destroyed/overwritten)
    \param[in,out] ainv   - pseudoinverse (M*N)
    \param[in]     cutoff - singular value cutoff

    \returns rank of matrix
**************************************************************/
size_t
pinv(size_t m, size_t n, size_t lda, double *a, double * ainv, double cutoff){
    
    size_t ii;
    size_t numberOfSingularValues = m < n ? m:n;
    double * u = calloc_double(m*m);
    double * vt = calloc_double(n*n);
    double * s = calloc_double(numberOfSingularValues);
    
    // not sure about thir dargument
    svd(m, n, lda, a, u, s, vt); //note changed from m to lda
    
    double * smat = calloc_double(m*n);

    size_t rank = 0;
    for (ii = 0; ii < numberOfSingularValues; ii++){
        if (fabs(s[ii]) < cutoff){ 
            smat[ii*m+ii] = 0.0;
        }
        else{
            smat[ii*m+ii] = 1.0/s[ii];
            rank += 1;
        }
    }
    
    double * temp = calloc_double(m*n);
    
    cblas_dgemm(CblasColMajor,CblasTrans,CblasTrans, n, m, n, 1.0, 
                    vt, n, smat, m, 0.0, temp, n);

    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans, n, m, m, 1.0, 
                    temp, n, u, m, 0.0, ainv, n);
    
    free(temp);
    free(smat);
    free(u);
    free(vt);
    free(s);

    return rank;
}

/*************************************************************//**
    Function norm2

    Purpose: compute the l2 norm of a vector

    Parameters:
        - vec (IN) - vector whose norm to take
        - N (IN)   - number of elements in the vector

    Returns: norm of the array
**************************************************************/
double 
norm2(double * vec, int N)
{
    double out = 0.0;
    int ii;
    for (ii = 0; ii < N; ii++){
        out += pow(vec[ii],2);
    }
    out = sqrt(out);
    return out;
}

/*************************************************************//**
    Function norm2diff

    Purpose: compute the l2 norm difference of two vectors

    Parameters:
        - a (IN) - vector
        - b (IN) - vector
        - N (IN)   - number of elements in each vector

    Returns: norm of the difference between *a* and *b*
**************************************************************/
double 
norm2diff(double * a, double * b, int N)
{

    double out = 0.0;
    int ii;
    for (ii = 0; ii < N; ii++){
        out += pow(a[ii]-b[ii],2.0);
    }
    out = sqrt(out);
    return out;
}

/*************************************************************//**
    Function mean

    Purpose: compute the mean of a vector

    Parameters:
        - vec (IN) - vector whose norm to take
        - N (IN)   - number of elements in the vector

    Returns: mean of the array
**************************************************************/
double 
mean(double * vec, size_t N)
{
    double out = 0.0;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        out += vec[ii];
    }
    out = out / N;
    return out;
}

/*************************************************************//**
    Function mean_size_t

    Purpose: compute the mean of a vector

    Parameters:
        - vec (IN) - vector whose norm to take
        - N (IN)   - number of elements in the vector

    Returns: mean of the array
**************************************************************/
double 
mean_size_t(size_t * vec, size_t N)
{
    double out = 0.0;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        out += (double) vec[ii];
    }
    out = out /  (double) N;
    return out;
}


/*************************************************************//**
    Function kron

    Purpose: matrix kronecker product

    Parameters:
        - a (IN) - first matrix
        - b (IN) - second matrix

    Returns: matrix kronecker product of *a* and *b*
**************************************************************/
struct mat * 
kron(const struct mat * a, const struct mat * b)
{
    struct mat * m; 
    if (NULL == ( m = malloc(sizeof(struct mat)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    m->nrows = a->nrows * b->nrows;
    m->ncols = a->ncols * b->ncols;
    m->vals = calloc_double(m->nrows *m->ncols);
    
    size_t ii,jj,kk,ll,row,col;
    for (ii = 0; ii < a->nrows; ii++){
        for (jj = 0; jj < a->ncols; jj++){
            for (kk = 0; kk < b->nrows; kk++){
                for (ll = 0; ll < b->ncols; ll++){
                    row = ii * b->nrows + kk;
                    col = jj * b->ncols + ll;
                    m->vals[row* m->ncols + col] = a->vals[ii*a->ncols + jj] * 
                                                   b->vals[kk*b->ncols + ll];
                }
            }
        }
    }
    return m;
}

/*************************************************************//**
    Function kron_col

    Purpose: matrix kronecker product with column major
             entries

    Parameters: (see blas type parameters)
        - n (IN) - number of rows of first matrix
        - m (IN) - number of columns of first matrix
        - lda (IN) - space between elements consecutive rows (see blas)
                   (also considered as stride)
        - a (IN) - first array
        - k (IN) - number of rows of second matrix
        - l (IN) - number of columns of second matrix
        - ldb (IN) - space between elements consecutive rows (see blas)
        - b (IN) - second array
        - ldc (in) -  space between elements consecutive rows (see blas)
        - out (IN/OUT) - kron(a,b)

    Returns: Nothing

    Notes:     

        Not optimized for access in order of storage
**************************************************************/
void
kron_col(int n, int m, double * a, int lda, int k, int l, double * b, int ldb, double * out, int ldc)
{
   
    int ii, jj, kk, ll;
    int row = 0;
    int col = 0;
    for (ii = 0; ii < n; ii++){
        for (jj = 0; jj < m; jj++){
            for (kk = 0; kk < k; kk++){
                for (ll = 0; ll < l; ll++){
                    row = ii * k + kk;
                    col = jj * l + ll;
                    out[col* ldc + row] = a[jj * lda + ii] * 
                                          b[ll * ldb + kk];
                }
            }
        }
    }
}

/*************************************************************//**
    Function vec_kron

    Purpose: compute v^T kron(A,B) fast
             

    Parameters: (see blas type parameters)
        - m (IN) - number of rows of first matrix
        - n (IN) - number of columns of first matrix
        - lda (IN) - space between elements consecutive rows (see blas)
                   (also considered as stride)
        - a (IN) - first array
        - k (IN) - number of rows of second matrix
        - l (IN) - number of columns of second matrix
        - ldb (IN) - space between elements consecutive rows (see blas)
        - b (IN) - second array
        - ldc (in) -  space between elements consecutive rows (see blas)
        - out (IN/OUT) - v kron(a,b)

    Returns: Nothing

**************************************************************/
void vec_kron(size_t m, size_t n, double * a, size_t lda, size_t k, size_t l, 
        double * b, size_t ldb, double * v, double scale, double * out)
{
    
    double * temp = calloc_double(m * l);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, l, m, k, 1.0,
                b, ldb, v, k, 0.0, temp, m);
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, l, m, 1.0,
                    a, lda, temp, m, scale, out, n);
    free(temp);
}

/* long double version of previous function */
void vec_kronl(size_t m, size_t n, double * a, size_t lda, size_t k, size_t l, 
        double * b, size_t ldb, long double * v, double scale, long double * out)
{
    
    double * temp = calloc_double(m * l);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, l, m, k, 1.0,
                b, ldb, (double *) v, k, 0.0, temp, m);
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, l, m, 1.0,
                    a, lda, temp, m, scale, (double *) out, n);
    free(temp);
}
/*************************************************************//**
    Function AddFiber

    Purpose: Add one dimensional fiber to fiber list
             
    Parameters:
        - head (IN/OUT) - Head node of fiber_list LinkedList
        - index (IN)    - Index label of added nodes
        - vals (IN)     - array of fiber values
        - nVals (IN)    - number of elements in the fiber

    Returns: Nothing.
**************************************************************/
void AddFiber(struct fiber_list ** head, size_t index, double * vals, size_t nVals)
{
    struct fiber_list * newNode;
    if (NULL == (newNode = malloc(sizeof(struct fiber_list)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    newNode->index = index;
    newNode->vals = calloc_double(nVals);
    memmove(newNode->vals, vals, nVals*sizeof(double));
    newNode->next = *head;
    *head = newNode;
}

/*************************************************************//**
    Function IndexExists

    Purpose: Check if data for index exists
             
    Parameters:
        - head (IN)  - Head node of fiber_list LinkedList
        - index (IN) - Index label of fiber

    Returns: 
        - -1 if index in the linked list of fibers doesnt exist.
        - >-1 if index in the linked list of fibers doesnt exist.
**************************************************************/
int IndexExists(struct fiber_list * head, size_t index){
    // if exists == -1 then index does not exist in list
    struct fiber_list * current = head;
    int exists = -1;
    size_t iteration = 0;
    while (current != NULL){
        if (current->index == index){
            exists = iteration;
            break;
        }
        current = current->next;
        iteration+=1;
    }
    return exists;
}

/*************************************************************//**
    Function getIndex

    Purpose: get the *index*th fiber (NOTE NOT THE INDEX LABEL)
             
    Parameters:
        - head (IN)  - Head node of fiber_list LinkedList
        - index (IN) - Index of the head LinkedList to get

    Returns: pointer to the array of the *index*th fiber
**************************************************************/
double * getIndex(struct fiber_list * head, size_t index){
    struct fiber_list * current = head;
    size_t count = 0;
    for (count = 0; count < index; count++){
        current = current->next;
    }
    return current->vals;
}

/*************************************************************//**
    Function DeleteFiberList

    Purpose: free the memory of the fiber list linked list
             
    Parameters:
        - head (IN)  - Head node of fiber_list LinkedList

    Returns: Nothing
**************************************************************/
void DeleteFiberList(struct fiber_list ** head){
    // This is a bit inefficient and doesnt free the last node
    struct fiber_list * current = *head;
    struct fiber_list * next;
    while (current != NULL){
        next = current->next;
        free(current->vals);
        free(current);
        current = next;
    }
    *head = NULL;
}

/*************************************************************//**
    Function init_skf

    Purpose: Initialize skeleton decomposition structure
             
    Parameters:
        - skd (IN/OUT)  - Skeleton decomposition to initialize
        - rank (IN)     - rank of the skeleton decomposition
        - n (IN)        - number of rows
        - m (IN)        - number of columns

    Returns: Nothing
**************************************************************/
void 
init_skf(struct sk_decomp ** skd, size_t rank, size_t n, size_t m)
{
    
    if (NULL == ( (*skd) = malloc(sizeof(struct sk_decomp)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    (*skd)->rank = rank;
    (*skd)->n = n;
    (*skd)->m = m;
    (*skd)->success = 0;
    (*skd)->rows_kept = calloc_size_t(rank);
    (*skd)->cols_kept = calloc_size_t(rank);

    if (NULL == ( (*skd)->row_vals = malloc(sizeof(struct fiber_info)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    (*skd)->row_vals->nfibers = 0;
    (*skd)->row_vals->head = NULL;


    if (NULL == ( (*skd)->col_vals = malloc(sizeof(struct fiber_info)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    (*skd)->col_vals->nfibers = 0;
    (*skd)->col_vals->head = NULL;


    if (NULL == ( (*skd)->cross_inv = calloc(rank * rank, sizeof(double)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
}

/*************************************************************//**
    Function sk_decomp_to_full

    Purpose: get the entire matrix from the skeleton decomposition
             
    Parameters:
        - skd (IN)  - Skeleton decomposition
        - out (OUT) - row-major order of the full array

    Returns: Nothing
**************************************************************/
void sk_decomp_to_full(struct sk_decomp * skd, double * out){
    
    double * temp = calloc_double(skd->n * skd->rank); 
    double * C = calloc_double(skd->n * skd->rank);
    double * R = calloc_double(skd->rank * skd->m);

    size_t ii,jj;
    int exists;
    double * t1;
    double * t2;
    for (ii = 0; ii < skd->rank; ii++){
        exists = IndexExists(skd->row_vals->head, skd->rows_kept[ii]);
        t1 = getIndex(skd->row_vals->head, (size_t) exists);
        memmove(R+ii*skd->m, t1, skd->m*sizeof(double));

        exists = IndexExists(skd->col_vals->head, skd->cols_kept[ii]);
        t2 = getIndex(skd->col_vals->head, (size_t) exists);
        for (jj = 0; jj < skd->n; jj++){
            C[jj*skd->rank+ii] = t2[jj];    
        }
    }
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, skd->n, skd->rank, 
        skd->rank, 1.0, C, skd->rank, skd->cross_inv, skd->rank, 0.0, temp, skd->rank);

    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, skd->n, skd->m, 
        skd->rank, 1.0, temp, skd->rank, R, skd->m, 0.0, out, skd->m);

    free(temp);
    free(C);
    free(R);
}

/*************************************************************//**
    Function free_skf

    Purpose: Free the skeleton decomposition structure
             
    Parameters:
        - skd (IN/OUT)  - Skeleton decomposition

    Returns: Nothing
**************************************************************/
void
free_skf(struct sk_decomp ** skd){
    free((*skd)->rows_kept); 
    free((*skd)->cols_kept);
    
    DeleteFiberList(&((*skd)->row_vals->head));
    free((*skd)->row_vals); 

    DeleteFiberList(&((*skd)->col_vals->head));
    free((*skd)->col_vals); 

    free((*skd)->cross_inv); 

    free(*skd);
    *skd = NULL;
}

/*************************************************************//**
    Function skeleton_func2

    Purpose: Compute a skeleton decomposition of a two 
             dimensional function
             
    Parameters:
        - Ap (IN/OUT)   - two dimensional function
        - args (IN/OUT) - arguments to the function
        - skd (IN/OUT)  - skeleton decomposition structure
        - x (IN)        - array of *x* values at which Ap is eval
        - y (IN)        - array of *y* values at which Ap is eval
        - delta (IN)    - convergence tolerance

    Returns: 
        - 2 didn't converge
        - 1 no error
        - -1 if maximum volume algorithm had an error
**************************************************************/
int 
skeleton_func2(int (*Ap)(double *, double, size_t, size_t, 
                   double *, void *),
                   void * args, struct sk_decomp ** skd, 
                   double * x, double * y, double delta)
{
    
    size_t n = (*skd)->n; // nrows
    size_t m = (*skd)->m; // ncols
    size_t r = (*skd)->rank;

    size_t ii, jj, iteration;
    int info, converged;

    double * R;
    double * C;
    double * tempa;
    double * tempb;
    double * tempmr;
    double * tau;
    double * work;
    double * Aold;
    double * Anew;
    size_t * ipiv;
    size_t lwork = n*n;
    int mv_success;
    
    int exists;
    double dfro, fro; //differences
    //int info;
    size_t siz = sizeof(double);
    
    if (NULL == (R = calloc(n*r, siz))) printf("Error in calloc\n");
    if (NULL == (C = calloc(m*r, siz))) printf("Error in calloc\n");
    if (NULL == (tempa = calloc(r*r, siz))) printf("Error in calloc\n");
    if (NULL == (tempmr = calloc(m*r, siz))) printf("Error in calloc\n");
    if (NULL == (tempb = calloc(r*r, siz))) printf("Error in calloc\n");
    if (NULL == (tau = malloc(r * siz))) printf("Error in malloc\n");
    if (NULL == (work = malloc(lwork* siz))) printf("Error in malloc\n");
    if (NULL == (ipiv = malloc(r* siz))) printf("Error in malloc\n");
    if (NULL == (Aold = calloc(n*m, siz))) printf("Error in calloc\n");
    if (NULL == (Anew = calloc(n*m, siz))) printf("Error in calloc\n");

    // get columns
    // column major order
    for (ii = 0; ii < r; ii++){
        exists = IndexExists((*skd)->col_vals->head, (*skd)->cols_kept[ii]);
        if (exists == -1){
        //if (exists > -10){
            (*Ap)(x, y[(*skd)->cols_kept[ii]], 0, n, R+ii*n, args);
            AddFiber(&((*skd)->col_vals->head), (*skd)->cols_kept[ii], R+ii*n, n);
            (*skd)->col_vals->nfibers+=1;
        }
        else{
            memmove(R+ii*n, getIndex((*skd)->col_vals->head,(size_t) exists), n*sizeof(double));
        }
    }
    
    converged = 0;
    iteration = 0;
    while (converged == 0){

        //obtain qr decomposition of R
        qr(n,r,R,n);

        //compute new row indices
        mv_success = maxvol_rhs(R, n,r, (*skd)->rows_kept, tempa);
        if (mv_success != 0) {
            converged = -1;
            break;
        }
    
        // get rows
        for (ii = 0; ii < r; ii++){
            //(*Ap)(y, x[rows[ii]], 1, m, C+ii*m, args);
            exists = IndexExists((*skd)->row_vals->head, (*skd)->rows_kept[ii]);
            if (exists == -1){
            //if (exists > -10){
                (*Ap)(y, x[(*skd)->rows_kept[ii]], 1, m, C+ii*m, args);
                AddFiber(&((*skd)->row_vals->head), (*skd)->rows_kept[ii], C+ii*m, m);
                (*skd)->row_vals->nfibers+=1;
            }
            else{
                memmove(C+ii*m, getIndex((*skd)->row_vals->head,(size_t) exists), m*sizeof(double));
            }

        }

        dgeqrf_((int*)&m,(int*)&r, C, (int*)&m, tau, work, (int*)&lwork, &info);
        if (info < 0) printf("Something wrong with computing QR of C\n ");
        if (info > 0) printf("Input %d of QR is not correct\n ", info);
        dorgqr_((int*)&m,(int*)&r, (int*)&r, C, (int*)&m, tau, work, (int*)&lwork, &info); // third element is min(n,m)
        if (info < 0) printf("Something wrong with QR of C\n ");
        if (info > 0) printf("Input %d of QR is not correct\n ", info);
        
        
        // compute new column indices
        mv_success = maxvol_rhs(C,m,r,(*skd)->cols_kept,tempb);
        mv_success = maxvol_rhs(R, n,r, (*skd)->rows_kept, tempa);
        if (mv_success != 0) {
            converged = -1;
            break;
        }

        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, m, r, r, 1.0, C, m, tempb, r, 0.0, tempmr, m);
        for (ii = 0; ii < r; ii++){
            exists = IndexExists((*skd)->col_vals->head, (*skd)->cols_kept[ii]);
            if (exists == -1){
            //if (exists > -10){
                (*Ap)(x, y[(*skd)->cols_kept[ii]], 0, n, R+ii*n, args);
                AddFiber(&((*skd)->col_vals->head), (*skd)->cols_kept[ii], R+ii*n, n);
                (*skd)->col_vals->nfibers+=1;
            }
            else{
                memmove(R+ii*n, getIndex((*skd)->col_vals->head,(size_t) exists), n*sizeof(double));
            }
        }
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans, n, m, r, 1.0, R, n, tempmr, m, 0.0, Anew, n);
 
        dfro = norm2diff(Anew,Aold, n*m);
        memmove(Aold,Anew,n*m*sizeof(double));
        if (iteration > 0){
            fro = norm2(Anew, n*m);
            printf("Iteration:%zu, %3.5f, diff=%3.5f\n", iteration, dfro, dfro/fro);
            if (dfro/fro < delta) converged = 1;
            if (iteration > n) converged = 2;
        }
        iteration+=1;
    }
    
    // build the inverse
    double * ainv = calloc((*skd)->rank * (*skd)->rank, sizeof(double));

    // column major order
    double * across =  calloc((*skd)->rank * (*skd)->rank, sizeof(double));
    double * t1;
    for (ii = 0; ii < (*skd)->rank; ii++){
        exists = IndexExists((*skd)->col_vals->head, (*skd)->cols_kept[ii]);
        t1 = getIndex((*skd)->col_vals->head, (size_t) exists);
        for (jj = 0; jj < (*skd)->rank; jj++){
            // row jj column ii
            across[ii*(*skd)->rank + jj] = t1[(*skd)->rows_kept[jj]];
        }
    }
    
    size_t rank_cross = pinv((*skd)->rank, (*skd)->rank, (*skd)->rank, across, ainv, 1e-15);

    // convert to row major
    (*skd)->cross_rank = rank_cross;
    for (ii = 0; ii < (*skd)->rank; ii++){
        for (jj = 0; jj < (*skd)->rank; jj++){
            (*skd)->cross_inv[ii*(*skd)->rank + jj] = ainv[jj*(*skd)->rank + ii];
        }
    }

    free(ainv);
    free(across);

    free(R);
    free(C);
    free(tau);
    free(work);
    free(tempa);
    free(tempb);
    free(tempmr);
    free(ipiv);
    free(Anew);
    free(Aold);
    
    return converged;
}

/*************************************************************//**
    Function skeleton_func

    Purpose: Compute a skeleton decomposition of a two 
             dimensional function
             
    Parameters:
        - A (IN/OUT)   - two dimensional function
        - aargs (IN/OUT) - arguments to the function
        - n (IN) - number of rows
        - m (IN) - number of columns
        - r (IN) - rank
        - rows (IN/OUT) - rows chosen for the decomposition
        - cols (IN/OUT) - cols chosen for the decomposition
        - delta (IN)    - convergence tolerance

    Returns: 
        - 2 didn't converge
        - 1 no error
        - -1 if maximum volume algorithm had an error

    Notes:

        If r is smaller than the true rank of A, large errors may occur
**************************************************************/
int skeleton_func(double (*A)(int,int, int, void *), void * aargs,
                size_t n, size_t m,
                size_t r, size_t * rows, size_t * cols, double delta)
{
    size_t ii, jj, iteration;
    int info, converged;

    double * R;
    double * C;
    double * tempa;
    double * tempb;
    double * tempmr;
    double * tau;
    double * work;
    double * Aold;
    double * Anew;
    size_t * ipiv;
    size_t lwork = n*n;
    int mv_success;
    
    double dfro, fro; //differences
    //int info;
    size_t siz = sizeof(double);

    if (NULL == (R = calloc(n*r, siz))) printf("Error in calloc\n");
    if (NULL == (C = calloc(m*r, siz))) printf("Error in calloc\n");
    if (NULL == (tempa = calloc(r*r, siz))) printf("Error in calloc\n");
    if (NULL == (tempmr = calloc(m*r, siz))) printf("Error in calloc\n");
    if (NULL == (tempb = calloc(r*r, siz))) printf("Error in calloc\n");
    if (NULL == (tau = malloc(r * siz))) printf("Error in malloc\n");
    if (NULL == (work = malloc(lwork* siz))) printf("Error in malloc\n");
    if (NULL == (ipiv = malloc(r* siz))) printf("Error in malloc\n");
    if (NULL == (Aold = calloc(n*m, siz))) printf("Error in calloc\n");
    if (NULL == (Anew = calloc(n*m, siz))) printf("Error in calloc\n");

    // get columns
    // column major order
    for (ii = 0; ii < r; ii++){
        for (jj = 0; jj < n; jj++){
            R[ii*n+jj] = (*A)(jj,cols[ii], 0, aargs);
        }
    }
    
    converged = 0;
    iteration = 0;
    while (converged == 0){
        //obtain qr decomposition of R
        dgeqrf_((int *)&n,(int*)&r, R, (int *)&n, tau, work, (int*)&lwork, &info);
        if (info < 0) printf("Something wrong with computing QR of R\n ");
        if (info > 0) printf("Input %d of QR is not correct\n ", info);
        dorgqr_((int *)&n,(int*)&r, (int*)&r, R, (int *)&n, tau, work, (int*)&lwork, &info); // third element is min(n,m)
        if (info < 0) printf("Something wrong with QR of R\n ");
        if (info > 0) printf("Input %d of QR is not correct\n ", info);
        
        //compute new row indices
        mv_success = maxvol_rhs(R, n,r, rows, tempa);
        if (mv_success != 0) {
            converged = -1;
            break;
        }
    
        // get rows
        for (ii = 0; ii < r; ii++){
            for (jj = 0; jj < m; jj++){
                //C[ii*m + jj] = A[ rows[ii] * m + jj ];
                C[ii * m + jj] = (*A)(rows[ii], jj, 1, aargs);

            }
        }

        dgeqrf_((int*)&m,(int*)&r, C, (int*)&m, tau, work, (int*)&lwork, &info);
        if (info < 0) printf("Something wrong with computing QR of C\n ");
        if (info > 0) printf("Input %d of QR is not correct\n ", info);
        dorgqr_((int*)&m,(int*)&r, (int*)&r, C, (int*)&m, tau, work, (int*)&lwork, &info); // third element is min(n,m)
        if (info < 0) printf("Something wrong with QR of C\n ");
        if (info > 0) printf("Input %d of QR is not correct\n ", info);
        
        
        // compute new column indices
        mv_success = maxvol_rhs(C,m,r,cols,tempb);
        mv_success = maxvol_rhs(R, n,r, rows, tempa);
        if (mv_success != 0) {
            converged = -1;
            break;
        }

        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, m, r, r, 1.0, C, m, tempb, r, 0.0, tempmr, m);
        for (ii = 0; ii < r; ii++){
            for (jj = 0; jj < n; jj++){
                //R[ii*n+jj] = A[jj*m+cols[ii]];
                R[ii*n+jj] = (*A)(jj,cols[ii],0, aargs);
            }
        }
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans, n, m, r, 1.0, R, n, tempmr, m, 0.0, Anew, n);
 
        dfro = norm2diff(Anew,Aold, n*m);
        memmove(Aold,Anew,n*m*sizeof(double));
        if (iteration > 0){
            fro = norm2(Anew, n*m);
            printf("Iteration:%zu, %3.2f, diff=%3.2f\n", iteration, dfro, dfro/fro);
            if (dfro/fro < delta) converged = 1;
            if (iteration > n) converged = 2;
        }
        iteration+=1;
    }

    free(R);
    free(C);
    free(tau);
    free(work);
    free(tempa);
    free(tempb);
    free(tempmr);
    free(ipiv);
    free(Anew);
    free(Aold);
    
    return converged;
}

/*************************************************************//**
    Function skeleton

    Purpose: Compute a skeleton decomposition of a matrix
             
    Parameters:
        - A  - matrix in row major order
        - n (IN) - number of rows
        - m (IN) - number of columns
        - r (IN) - rank
        - rows (IN/OUT) - rows chosen for the decomposition
        - cols (IN/OUT) - cols chosen for the decomposition
        - delta (IN)    - convergence tolerance

    Returns: 
        - 2 didn't converge
        - 1 no error
        - -1 if maximum volume algorithm had an error

    Notes: 

        If r is smaller than the true rank of A, large errors may occur
**************************************************************/
int skeleton(double * A, size_t n, size_t m,
                size_t r, size_t * rows, 
                size_t * cols, double delta)
{
    size_t ii, jj,  iteration;
    int info, converged;

    double * R;
    double * C;
    double * tempa;
    double * tempb;
    double * tempmr;
    double * tau;
    double * work;
    double * Aold;
    double * Anew;
    size_t * ipiv;
    size_t lwork = n*n;
    int mv_success;
    
    double dfro, fro; //differences
    //int info;
    size_t siz = sizeof(double);

    if (NULL == (R = calloc(n*r, siz))) printf("Error in calloc\n");
    if (NULL == (C = calloc(m*r, siz))) printf("Error in calloc\n");
    if (NULL == (tempa = calloc(r*r, siz))) printf("Error in calloc\n");
    if (NULL == (tempmr = calloc(m*r, siz))) printf("Error in calloc\n");
    if (NULL == (tempb = calloc(r*r, siz))) printf("Error in calloc\n");
    if (NULL == (tau = malloc(r * siz))) printf("Error in malloc\n");
    if (NULL == (work = malloc(lwork* siz))) printf("Error in malloc\n");
    if (NULL == (ipiv = malloc(r* siz))) printf("Error in malloc\n");
    if (NULL == (Aold = calloc(n*m, siz))) printf("Error in calloc\n");
    if (NULL == (Anew = calloc(n*m, siz))) printf("Error in calloc\n");

    // column major order
    for (ii = 0; ii < r; ii++){
        for (jj = 0; jj < n; jj++){
            R[ii*n+jj] = A[jj*m+cols[ii]];
        }
    }
    
    converged = 0;
    iteration = 0;
    while (converged == 0){
        //obtain qr decomposition of R
        dgeqrf_((int *)&n,(int*)&r, R, (int *)&n, tau, work, (int*)&lwork, &info);
        if (info < 0) printf("Something wrong with computing QR of R\n ");
        if (info > 0) printf("Input %d of QR is not correct\n ", info);
        dorgqr_((int *)&n,(int*)&r, (int*)&r, R, (int *)&n, tau, work, (int*)&lwork, &info); // third element is min(n,m)
        if (info < 0) printf("Something wrong with QR of R\n ");
        if (info > 0) printf("Input %d of QR is not correct\n ", info);
        
        //compute new row indices
        mv_success = maxvol_rhs(R, n,r, rows, tempa);
        if (mv_success != 0) {
            converged = -1;
            break;
        }

        for (ii = 0; ii < r; ii++){
            for (jj = 0; jj < m; jj++){
                C[ii*m + jj] = A[ rows[ii] * m + jj ];
            }
        }

        dgeqrf_((int*)&m,(int*)&r, C, (int*)&m, tau, work, (int*)&lwork, &info);
        if (info < 0) printf("Something wrong with computing QR of C\n ");
        if (info > 0) printf("Input %d of QR is not correct\n ", info);
        dorgqr_((int*)&m,(int*)&r, (int*)&r, C, (int*)&m, tau, work, (int*)&lwork, &info); // third element is min(n,m)
        if (info < 0) printf("Something wrong with QR of C\n ");
        if (info > 0) printf("Input %d of QR is not correct\n ", info);
        
        
        // compute new column indices
        mv_success = maxvol_rhs(C,m,r,cols,tempb);
        mv_success = maxvol_rhs(R, n,r, rows, tempa);
        if (mv_success != 0) {
            converged = -1;
            break;
        }

        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, m, r, r, 1.0, C, m, tempb, r, 0.0, tempmr, m);
        for (ii = 0; ii < r; ii++){
            for (jj = 0; jj < n; jj++){
                R[ii*n+jj] = A[jj*m+cols[ii]];
            }
        }
        cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans, n, m, r, 1.0, R, n, tempmr, m, 0.0, Anew, n);
 
        dfro = norm2diff(Anew,Aold, n*m);
        memmove(Aold,Anew,n*m*sizeof(double));
        if (iteration > 0){
            fro = norm2(Anew, n*m);
            printf("Iteration:%zu, %3.2f, diff=%3.2f\n", iteration, dfro, dfro/fro);
            if (dfro/fro < delta) converged = 1;
            if (iteration > n) converged = 2;
        }
        iteration+=1;
    }

    free(R);
    free(C);
    free(tau);
    free(work);
    free(tempa);
    free(tempb);
    free(tempmr);
    free(ipiv);
    free(Anew);
    free(Aold);
    
    return converged;
}
 
/*************************************************************//**
    Function maxvol_rhs

    Purpose: Compute the rows of a maximum volume submatrix
             of a tall and skinny matrix
             
    Parameters:
        - A (IN) - matrix in column major storage
        - n (IN) - number of total rows
        - r (IN) - number of columns 
        - rows (IN/OUT) - rows creating maximum volume submatrix
        - Asinv (OUT)   - Inverse of the maximum volume submatrix

    Returns: 
          -  0 converges
          -  <0 rank is too low
          -  >0 rank may be too high

    Notes:
    
        Inverse is untested
**************************************************************/
int maxvol_rhs(const double * A, size_t n, size_t r, size_t * rows, double * Asinv){
    
    double delta = 0.01;
    
    // temporary variables
    size_t siz = sizeof(double);
    double maxelem;
    size_t maxloc;
    size_t ii, jj, row, col;
    size_t *allrows = calloc(n, sizeof(size_t));
    size_t temp;

    // solver stuff
    int * ipiv = calloc(r, sizeof(size_t));
    int * ipiv2 = calloc(r, sizeof(size_t));
    int info;
    int info2;
    size_t lwork = r*r;
    double * work = calloc(lwork, siz);

    double * Atemp = calloc(lwork, siz);
    double * temp2;
    double * B = calloc(n*r, siz);
    
    // compute LU of A to obtain pivots

    temp2 = calloc_double(n*r);
    memmove(temp2, A, n*r*siz);
    dgetrf_((int*)&r,(int*)&r, temp2, (int*)&r, ipiv, &info); //lu decomp
    if (info < 0) printf("Input %d of LU is not correct\n ", info);
    //if (info > 0) printf("Rank is too high, suggest lowering it\n");

    free(temp2);
    if (info < 0){
        printf("couldn't find invertible submatrix in maxvol\n");
        free(ipiv);
        free(ipiv2);
        free(work);
        free(allrows);
        free(Atemp);
        free(B);
        return info;     
    }

    //printf("first 3 \n");
    for (ii = 0; ii < n; ii++){
        allrows[ii] = ii;
    }

    for (ii = 0; ii < r; ii++){
        jj = (size_t) ipiv[ii];
        
        if (jj != ii){
            temp = allrows[ii];
            allrows[ii] = allrows[jj];
            allrows[jj] = temp;
        }
    }
    for (ii = 0; ii < r; ii++){
        rows[ii] = allrows[ii];
    }
    

    getrows(A, Asinv, rows,n,r);
    dgetrf_((int*)&r,(int*)&r, Asinv, (int*)&r, ipiv2, &info2); 
    //if (info2 > 0) { printf("Ainv is singular in maxvol_rhs\n "); exit(1); }
    //if (info2 < 0) { printf("Input %d of LU is not correct in maxvol_rhs\n ", info2); exit(1); }
        
    dgetri_((int*)&r, Asinv, (int*)&r, ipiv2, work, (int*)&lwork, &info2); //invert
    
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, n, r, r, 1.0, A,n, Asinv, r, 0.0, B, n);
    
    rows_to_top_col(B, Atemp, ipiv, n,r);

    size_t iter = 0;
    int done = 0;
    double * x = calloc_double(n);
    double * y = calloc_double(r);
    double * tl = calloc_double(r);
    double * tr = calloc_double(r);
    double * ei;
    while (done == 0){
        maxelem = maxabselem(B, &maxloc, n*r);

        //printf("maxelem = %3.2f\n", maxelem);
        if (maxelem < (1.0+delta)){
            break;
        }
        if (iter > n*1){
            printf("Number of iterations exceeded max in maxvol\n");
            break;
        }

        //rank1 update
        row = maxloc % n;
        col = (maxloc - row) / n;

        for (ii = 0; ii< r; ii++){
            //col major
            y[ii] = B[ii * n + row];
            x[ii] = B[col * n + ii];
        }
        y[col] -= 1.0;
        for (ii = r; ii < n; ii++){
            x[ii] = B[col * n + ii];
        }
        x[col]-=1;
        x[row]+=1;

        temp = allrows[row];
        allrows[row] = allrows[col];
        allrows[col] = temp;
        rows[col] = temp;
        
        for (ii = 0; ii< r; ii ++){
            tr[ii] = B[ii*n+row] - B[ii*n + col];   
        }
        ei = calloc_double(r);
        ei[col] = 1.0; 
        cblas_dgemv(CblasColMajor, CblasNoTrans, r, r, 1.0, Asinv, r, ei, 1, 0.0, tl, 1); 
        free(ei);
    
        // update square inverse
        cblas_dger(CblasColMajor, r, r, -1.0/B[col*n+row], tl,1, tr,1, Asinv, r);

        // Update B
        cblas_dger(CblasColMajor, n, r, -1.0/B[col*n+row], x, 1, y, 1, B, n);
        
        iter+=1;
    }
    
    free(ipiv);
    free(ipiv2);
    free(work);
    free(allrows);
    free(Atemp);
    free(B);


    free(x);
    free(y);
    free(tl);
    free(tr);
    return 0;
}


/*************************************************************//**
    Perform linear ls

    min ||Ax-y||

    \param[in] nrows - number of rows of A
    \param[in] ncols - number of columns of A
    \param[in,out] A - A matrix stored in column major ordser
    \param[in,out] y - right hand size
    \param[in,out] x - solution

    \note
    Thin wrapper around cblas_dgels
**************************************************************/
void linear_ls(size_t nrows, size_t ncols, double * A, double * y, double * x)
{
    // Workspace and status variables
    double * work;
    if (NULL == ( work = malloc(sizeof(double)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }

    int lworkinit = -1;
    int info = 0;
    //get optimal workspace size
    int nrhs=1;
    /* char trans[2] = "N"; */
    int rank;
    double rcond = -1;
    size_t mindim = nrows < ncols ? nrows : ncols;
    size_t maxdim = nrows > ncols ? nrows : ncols;

    /* printf("mindim = %zu  maxdim = %zu\n",mindim, maxdim); */
    double * tempy = calloc_double(maxdim);
    memmove(tempy,y,nrows*sizeof(double));
    
    double * s = calloc_double(maxdim);
    size_t * iwork = calloc_size_t(2);
    /* dgels_(trans,&nrows,&ncols,&nrhs,A,&nrows,y,&nrows,work,&lworkinit,&info); */

    /* printf("compute workspace\n"); */
    dgelsd_((int*)&nrows,(int*)&ncols,(int*)&nrhs,A,(int*)&nrows,tempy,(int*)&maxdim,s,&rcond,(int*)&rank,work,&lworkinit,(int*)iwork,&info);
    free(iwork); iwork = NULL;
    
    size_t optimal_work = (size_t) work[0];
    free(work); work = NULL;
    if (NULL == ( work = malloc(optimal_work * sizeof(double)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }

    /* printf("compute workspace, run\n"); */
    /* int lwork = mindim + mindim * 2; */
    int lwork = optimal_work;
    iwork = calloc_size_t(optimal_work * mindim);
    /* dgels_(trans,&nrows,&ncols,&nrhs,A,&nrows,y,&nrows,work,&lwork,&info); */
    dgelsd_((int*)&nrows,(int*)&ncols,(int*)&nrhs,A,(int*)&nrows,tempy,(int*)&maxdim,s,&rcond,(int*)&rank,work,&lwork,(int*)iwork,&info);
    if (info != 0){
        fprintf(stderr, "Error computing linear regression\n");
        exit(1);
    }

    /* printf("in linalg\n"); */
    /* dprint(nrows,tempy); */
    
    memmove(x,tempy,ncols * sizeof(double));
    free(work); work = NULL;
    free(iwork); iwork = NULL;
    free(s); s = NULL;
    free(tempy); tempy = NULL;

    
}
