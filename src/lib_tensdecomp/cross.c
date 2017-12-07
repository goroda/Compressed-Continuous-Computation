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
#include "cross.h"
#include "tt_multilinalg.h"

/********************************************************//**
    Function AddCrossFiber

    Purpose: Add one dimensional fiber from TTcross to fiber list
             
    Parameters:
        - head (in/out)  - head node of the list
        - pre_length (in) - length of pre_indices
        - post_length (in) - length of post_indices
        - pre_index (in) - pre_index
        - post_index (in ) -post index
        - vals (in) - fiber values
        - nVals (in) - number of Fiber values
        
    Returns: Nothing.
***********************************************************/
void AddCrossFiber(struct cross_fiber_list ** head, size_t pre_length, 
                    size_t post_length, size_t * pre_index, 
                    size_t * post_index, double * vals, size_t nVals)
{
    struct cross_fiber_list * newNode;
    if (NULL == (newNode = malloc(sizeof(struct cross_fiber_list)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }

    newNode->pre_length  = pre_length;
    newNode->post_length = post_length;
    if (pre_length == 0){
        newNode->pre_index = NULL;
    }
    else{
        newNode->pre_index = calloc_size_t(pre_length);
        memmove(newNode->pre_index, pre_index, pre_length * sizeof(size_t));
    }
    
    if (post_length == 0){
        newNode->post_index = NULL;
    }
    else{
        newNode->post_index = calloc_size_t(post_length);
        memmove(newNode->post_index, post_index, post_length * sizeof(size_t));
    }
    newNode->vals = calloc_double(nVals);
    memmove(newNode->vals, vals, nVals*sizeof(double));
    newNode->next = *head;
    *head = newNode;
}

/********************************************************//**
    Function CrossIndexExists

    Purpose: Check if data for index exists
             
    Parameters:
        - head (in/out)  - head node of the list
        - pre_index (in) - pre_index
        - post_index (in ) -post index
        

    Returns: 
        - -1 if index in the linked list of fibers doesnt exist.
        - >-1 if index in the linked list of fibers doesnt exist.
***********************************************************/
int 
CrossIndexExists(struct cross_fiber_list * head, size_t * pre_index, 
                 size_t * post_index)
{
    // if exists == -1 then index does not exist in list
    struct cross_fiber_list * current = head;
    int exists = -1;
    size_t iteration = 0;
    int pre_check;
    int post_check;
    size_t jj;
    while (current != NULL){
        
        pre_check = 0;
        post_check = 0;
   
        pre_check = 1;
        for (jj = 0; jj < current->pre_length; jj++){
            if (current->pre_index[jj] != pre_index[jj]){
                current = current->next;
                pre_check = 0;
                break;
            }
        }
        if (pre_check == 1) {
            post_check = 1;
            for (jj = 0; jj < current->post_length; jj++){
                if (current->post_index[jj] != post_index[jj]){
                    current = current->next;
                    post_check = 0;
                    break;
                }
            }
        }
        if ((pre_check == 1) && (post_check == 1)){
            exists = iteration;
            break;
        }
        iteration+=1;
    }
    return exists;
}

/********************************************************//**
    Function getCrossFiberListIndex

    Purpose: get the *index*th fiber (NOTE NOT THE INDEX LABEL)
             
    Parameters:
        - head (IN)  - Head node of fiber_list LinkedList
        - index (IN) - Index of the head LinkedList to get

    Returns: pointer to the array of the *index*th fiber
***********************************************************/
double * getCrossFiberListIndex(struct cross_fiber_list * head, size_t index)
{
    struct cross_fiber_list * current = head;
    size_t count = 0;
    for (count = 0; count < index; count++){
        current = current->next;
    }
    return current->vals;
}

/********************************************************//**
    Function DeleteCrossFiberList

    Purpose: free the memory of the cross fiber list linked list
             
    Parameters:
        - head (IN)  - Head node of cross fiber_list LinkedList

    Returns: Nothing
***********************************************************/
void DeleteCrossFiberList(struct cross_fiber_list ** head)
{
    // This is a bit inefficient and doesnt free the last node
    struct cross_fiber_list * current = *head;
    struct cross_fiber_list * next;
    while (current != NULL){
        next = current->next;
        if (current->pre_index != NULL){
            free(current->pre_index);
        }
        if (current->post_index != NULL){
            free(current->post_index);
        }
        free(current->vals);
        free(current);
        current = next;
    }
    *head = NULL;
}

struct tt_cross_opts * 
init_cross_opts(size_t dim, size_t *ranks, size_t *nvals)
{
    
    size_t ii, jj;
    struct tt_cross_opts * opts;
    if (NULL == (opts = malloc(sizeof(struct tt_cross_opts)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    opts->verbose = 0;
    opts->maxiter = 10;
    opts->epsilon = 1e-12;
    opts->success = 0;

    opts->dim = dim;
    opts->ranks = calloc_size_t(opts->dim+1);
    memmove(opts->ranks, ranks, (dim+1) * sizeof(size_t));

    opts->nvals = calloc_size_t(opts->dim);
    memmove(opts->nvals, nvals, dim * sizeof(size_t));
 
    if (NULL == (opts->right =
                malloc((opts->dim) * sizeof(struct index_set *)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }

    if (NULL == (opts->left =
                malloc((opts->dim) * sizeof(struct index_set *)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    
    size_t temp;
    for (ii = 0; ii < opts->dim; ii++){
        if (NULL == (opts->right[ii] = malloc( sizeof(struct index_set)))){
            fprintf(stderr, "failed to allocate memory.\n");
            exit(1);
        }
        temp = opts->dim-ii-1;
        opts->right[ii]->dim = temp; 
        opts->right[ii]->N = opts->ranks[ii+1];
        
        if (NULL == (opts->right[ii]->indices = malloc( 
                        opts->right[ii]->N * sizeof(size_t * )))){
            fprintf(stderr, "failed to allocate memory.\n");
            exit(1);
        }
        if (temp == 0){
            for (jj = 0; jj < opts->ranks[ii+1]; jj++){
                opts->right[ii]->indices[jj] = NULL;
            }
        }
        else{
            for (jj = 0; jj < opts->ranks[ii+1]; jj++){
                opts->right[ii]->indices[jj] = calloc_size_t(opts->right[ii]->dim);
            }
        }

        if (NULL == (opts->left[ii] = malloc( sizeof(struct index_set)))){
            fprintf(stderr, "failed to allocate memory.\n");
            exit(1);
        }
        
        opts->left[ii]->dim = ii;
        opts->left[ii]->N = opts->ranks[ii];
        
        if (NULL == (opts->left[ii]->indices = malloc( 
                        opts->left[ii]->N * sizeof(size_t *)))){
            fprintf(stderr, "failed to allocate memory.\n");
            exit(1);
        }
        if (ii == 0){
            for (jj = 0; jj < opts->ranks[ii]; jj++){
                opts->left[ii]->indices[jj] = NULL;
            }
        }
        else{
            for (jj = 0; jj < opts->ranks[ii]; jj++){
                opts->left[ii]->indices[jj] = 
                        calloc_size_t(opts->left[ii]->dim);
            }
        }
    }
    
    if (NULL == (opts->fibers = malloc(opts->dim * 
                                sizeof(struct cross_fiber_info)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    for (ii = 0; ii < opts->dim; ii++){
        if (NULL == (opts->fibers[ii] = 
                     malloc(sizeof(struct cross_fiber_list)))){
            fprintf(stderr, "failed to allocate memory.\n");
            exit(1);
        }
        opts->fibers[ii]->nfibers = 0;
        opts->fibers[ii]->head = NULL;
    }
    return opts;   
}

void create_naive_index_set(struct index_set * r, size_t * nvals)
{   
    size_t ii;
    size_t onDim = 0;
    for ( ii = 0; ii < r->N; ii++)
    {
        if (nvals[onDim] < (ii+1)){
            onDim += 1;
        }
        if( r->indices[ii] != NULL){
            r->indices[ii][onDim] =  ii;
        }
    }   
}

struct tt_cross_opts * 
init_cross_opts_with_naive_set(size_t dim, size_t * ranks, size_t * nvals)
{
    struct tt_cross_opts * opt = init_cross_opts(dim, ranks, nvals);
    size_t ii;
    for (ii = 0; ii < opt->dim-1; ii++){
        create_naive_index_set(opt->right[ii], nvals + ii);
        create_naive_index_set(opt->left[ii], nvals);
    }
    return opt;
}

void print_index_set(struct index_set * r){
    
    size_t ii;
    for (ii = 0; ii < r->N; ii++){
        if (r->indices[ii] == NULL){
            printf("NULL\n");
        }
        else{
            iprint_sz(r->dim, r->indices[ii]);
        }
    }
}

void free_index_set(struct index_set ** ind){

    size_t ii;
    for (ii = 0; ii < (*ind)->N; ii++){
        if ((*ind)->indices[ii] != NULL){
            free((*ind)->indices[ii]);
        }
    }
    free( (*ind)->indices);
    free( (*ind) );
}

void free_cross_opts(struct tt_cross_opts ** opts)
{
    free( (*opts)->ranks);
    free( (*opts)->nvals);
    size_t ii;
    for (ii = 0; ii < (*opts)->dim; ii++){
        free_index_set(&(*opts)->right[ii]);
        free_index_set(&(*opts)->left[ii]);
        DeleteCrossFiberList(&((*opts)->fibers[ii]->head));
        free((*opts)->fibers[ii]);
    }
    free((*opts)->right);
    free((*opts)->left);
    free((*opts)->fibers);

    free(*opts);
}


void wrap_func_for_cross(size_t nVals, size_t * indices, size_t fiber_index, 
                         double * out, void * fopts)
{
    
    struct func_to_array * af = (struct func_to_array *) fopts;
    
    size_t ii, jj;
    double * pt = calloc_double(af->dim);
    for (ii = 0; ii < nVals; ii++){
        indices[fiber_index] = ii;
        for (jj = 0; jj < af->dim; jj++){
            pt[jj] = af->pts[jj][indices[jj]];
        }
        out[ii] = af->f(pt, af->args);
    }
    free(pt);
}

void 
wrap_full_tensor_for_cross(size_t nVals, size_t * indices, size_t fiber_index, 
                            double * out, void * full_tensor)
{

    struct tensor * t = (struct tensor *) full_tensor ;
    size_t ii;
    for (ii = 0; ii < nVals; ii++){
        indices[fiber_index] = ii;
        out[ii] = tensor_elem(t, indices);
    }

}

void 
big_to_multi(size_t bigInd, size_t ndim, const size_t * nPerDim, 
             size_t * multi_ind)
{
    // assume fortran ordering
    size_t b = bigInd;
    size_t prod = iprod_sz(ndim-1, nPerDim);
    size_t jj;
    for (jj = ndim-1; jj>0; jj--){
        multi_ind[jj] = (size_t) lrint(floor(b / prod));
        b = b - multi_ind[jj] * prod;
        prod = prod / nPerDim[jj-1];
    }
    multi_ind[0] = b;
}

size_t * combine_ind_with_space(size_t nn, size_t * n, size_t mm, size_t * m)
{   
    size_t * out = calloc_size_t(nn+mm+1);
    if (nn > 0) memmove(out, n, nn * sizeof(size_t));
    if (mm > 0) memmove(out + nn+1, m, mm * sizeof(size_t));

    return out;
}

void build_fiber_mat(struct tt_cross_opts * opts, size_t ii, size_t kk, 
            size_t core, const double * fiber, double * Ak, int leftright)
{
    size_t jj;
    if (leftright == 1){
        for (jj = 0; jj < opts->nvals[core]; jj++){
            Ak[kk + jj * opts->ranks[core] + 
                ii * opts->ranks[core] * opts->nvals[core] ] = fiber[jj];
        }
    }
    else{
        for (jj = 0; jj < opts->nvals[core]; jj++){
            Ak[ii + jj * opts->ranks[core+1] + 
                kk * opts->ranks[core+1] * opts->nvals[core] ] = fiber[jj];
        }
    }
}

double * 
cross_setup(void (*A)(size_t, size_t *, size_t, double *, void *), 
        struct tt_cross_opts * opts, size_t core, int leftright, void * args)
{
    size_t ii, kk;
    double * Ak = calloc_double(opts->ranks[core] * opts->nvals[core] * 
                                opts->ranks[core+1]);

    size_t * combined_ind;
    double * fiber;
    int index;
    for (ii = 0; ii < opts->ranks[core+1]; ii++){ // loop over columns of C
        for (kk = 0; kk < opts->ranks[core]; kk++){
            
            index = CrossIndexExists(opts->fibers[core]->head,
                             opts->left[core]->indices[kk], 
                             opts->right[core]->indices[ii]);

            if (index == -1) {
                fiber = calloc_double(opts->nvals[core]);
                combined_ind = combine_ind_with_space( opts->left[core]->dim,
                            opts->left[core]->indices[kk],
                            opts->right[core]->dim,
                            opts->right[core]->indices[ii]);

                (*A)(opts->nvals[core], combined_ind, core, fiber, args);
                
                AddCrossFiber(&(opts->fibers[core]->head), 
                               opts->left[core]->dim,
                               opts->right[core]->dim, 
                               opts->left[core]->indices[kk],
                               opts->right[core]->indices[ii], fiber, 
                               opts->nvals[core]);
                opts->fibers[core]->nfibers++;
        
                build_fiber_mat(opts,ii,kk,core, fiber, Ak, leftright);

                free(combined_ind);
                free(fiber);
            }
            else{
                fiber = getCrossFiberListIndex(opts->fibers[core]->head, 
                                                (size_t) index);
                build_fiber_mat(opts,ii,kk,core, fiber, Ak, leftright);
            }
        }
    }
    return Ak;
}

int 
process_first_core_lr(struct tt * a, 
        void (*A)(size_t, size_t *, size_t, double *, void *), 
        struct tt_cross_opts * opts, void * args)
{

    double * Ainv = calloc_double(opts->ranks[1] * opts->ranks[1]);
    size_t * newrows = calloc_size_t(opts->ranks[1]);
    
    //printf("Now!\n");
    //double * Ak = cross_setup(A, opts, 0, args);
    double * Ak = cross_setup(A, opts, 0, 1, args);
    //printf("Then!\n");
    qr(opts->ranks[0] * opts->nvals[0], opts->ranks[1], Ak, 
            opts->ranks[0] * opts->nvals[0]);
    int mv_success = maxvol_rhs(Ak,opts->ranks[0] * opts->nvals[0],
                            opts->ranks[1],newrows,Ainv);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                opts->ranks[0] * opts->nvals[0], opts->ranks[1], 
                opts->ranks[1], 1.0, Ak, opts->ranks[0] * opts->nvals[0],
                Ainv, opts->ranks[1], 0.0, a->cores[0]->vals, 
                opts->ranks[0] * opts->nvals[0]);
    size_t jj;
    for (jj = 0; jj < opts->ranks[1]; jj++){
        opts->left[1]->indices[jj][0] = newrows[jj];
    }
    free(newrows);
    free(Ainv);
    free(Ak);
    return mv_success;
}

int 
process_last_core_rl(struct tt * a, 
            void (*A)(size_t, size_t *, size_t, double *, void *), 
            struct tt_cross_opts * opts, void * args)
{
    
    size_t kk = opts->dim-1;
    double * Ainv = calloc_double(opts->ranks[kk] * opts->ranks[kk]);
    size_t * newrows = calloc_size_t(opts->ranks[kk]);
    
    double * Ak = cross_setup(A, opts, kk, 0, args);
    qr(opts->ranks[kk+1] * opts->nvals[kk], opts->ranks[kk], Ak, 
            opts->ranks[kk+1] * opts->nvals[kk]);

    int mv_success = maxvol_rhs(Ak,opts->ranks[kk+1] * opts->nvals[kk],
                            opts->ranks[kk],newrows,Ainv);
    
    //printf("Then!\n");
    double * Atemp = calloc_double(opts->nvals[kk] * opts->ranks[kk] * 
                                   opts->ranks[kk+1]);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                opts->nvals[kk] * opts->ranks[kk+1], opts->ranks[kk],
                opts->ranks[kk], 1.0,
                Ak, opts->ranks[kk+1] * opts->nvals[kk],  Ainv, 
                opts->ranks[kk], 0.0, Atemp,
                opts->ranks[kk+1] * opts->nvals[kk]);
    size_t ii,zz,ll;
    //transpose
    for (ii = 0; ii <opts->ranks[kk]; ii++){
        for (zz = 0; zz < opts->nvals[kk]; zz++){
            for (ll = 0; ll < opts->ranks[kk+1]; ll++){
                a->cores[kk]->vals[ii + zz * opts->ranks[kk] + 
                        ll * opts->ranks[kk] *opts->nvals[kk]] =
                    Atemp[ll + zz * opts->ranks[kk+1] + 
                        ii * opts->ranks[kk+1] * opts->nvals[kk]];
            }
        }
    }
    free(Atemp);

    //printf("When!\n");
    size_t jj;
    for (jj = 0; jj < opts->right[kk-1]->N; jj++){
        opts->right[kk-1]->indices[jj][0] = newrows[jj];
    }
    //printf("hen!\n");
    free(newrows);
    free(Ainv);
    free(Ak);
    return mv_success;
}

int 
process_cores_rl(struct tt * a, size_t kk, 
        void (*A)(size_t, size_t *, size_t, double *, void *), 
        struct tt_cross_opts * opts, void * args)
{
    
    size_t nValsMulti[2];
    size_t newInd[2];
    //int kk = opts->dim-1;
    double * Ainv = calloc_double(opts->ranks[kk] * opts->ranks[kk]);
    size_t * newrows = calloc_size_t(opts->ranks[kk]);
    
    double * Ak = cross_setup(A, opts, kk, 0, args);
    qr(opts->ranks[kk+1] * opts->nvals[kk], opts->ranks[kk], Ak, 
            opts->ranks[kk+1] * opts->nvals[kk]);

    int mv_success = maxvol_rhs(Ak,opts->ranks[kk+1] * opts->nvals[kk],
                            opts->ranks[kk],newrows,Ainv);
    
    //printf("Then!\n");
    double * Atemp = calloc_double(opts->nvals[kk] * opts->ranks[kk] * 
                                   opts->ranks[kk+1]);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                opts->nvals[kk] * opts->ranks[kk+1], opts->ranks[kk],
                opts->ranks[kk], 1.0,
                Ak, opts->ranks[kk+1] * opts->nvals[kk],  Ainv, 
                opts->ranks[kk], 0.0, Atemp,
                opts->ranks[kk+1] * opts->nvals[kk]);
    size_t ii,zz, ll;
    //transpose
    for (ii = 0; ii <opts->ranks[kk]; ii++){
        for (zz = 0; zz < opts->nvals[kk]; zz++){
            for (ll = 0; ll < opts->ranks[kk+1]; ll++){
                a->cores[kk]->vals[ii + zz * opts->ranks[kk] + 
                        ll * opts->ranks[kk] *opts->nvals[kk]] =
                    Atemp[ll + zz * opts->ranks[kk+1] + 
                        ii * opts->ranks[kk+1] * opts->nvals[kk]];
            }
        }
    }
    free(Atemp);
    
    nValsMulti[0] = opts->ranks[kk];
    nValsMulti[1] = opts->nvals[kk];
    size_t jj;
    //printf("core = %zu \n",kk);
    for (jj = 0; jj < opts->right[kk-1]->N; jj++) {
        big_to_multi(newrows[jj], 2, nValsMulti, newInd);
        memmove(opts->right[kk-1]->indices[jj]+1, 
                opts->right[kk]->indices[newInd[0]], 
                opts->right[kk]->dim * sizeof(size_t));
        opts->right[kk-1]->indices[jj][0] = newInd[1];
    }
    //printf("done!\n");

    //printf("hen!\n");
    free(newrows);
    free(Ainv);
    free(Ak);
    return mv_success;
}

int process_cores_lr(struct tt * a, size_t kk, 
        void (*A)(size_t, size_t *, size_t, double *, void *), 
        struct tt_cross_opts * opts, void * args)
{
    //kk is the core

    size_t nValsMulti[2];
    size_t newInd[2];
    double * Ainv = calloc_double(opts->ranks[kk+1] * opts->ranks[kk+1]);
    size_t  * newrows = calloc_size_t(opts->ranks[kk+1]);
    double * Ak = cross_setup(A, opts, kk, 1, args);
    
    qr(opts->ranks[kk] * opts->nvals[kk], opts->ranks[kk+1], Ak, 
            opts->ranks[kk] * opts->nvals[kk]);

    int mv_success = maxvol_rhs(Ak,opts->ranks[kk] * opts->nvals[kk],
                            opts->ranks[kk+1],newrows,Ainv);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                opts->ranks[kk] * opts->nvals[kk], opts->ranks[kk+1], 
                opts->ranks[kk+1], 1.0, Ak, opts->ranks[kk] * opts->nvals[kk],
                Ainv, opts->ranks[kk+1], 0.0, a->cores[kk]->vals, 
                opts->ranks[kk] * opts->nvals[kk]);

    

    // update left index of kk+1
    nValsMulti[0] = opts->ranks[kk];
    nValsMulti[1] = opts->nvals[kk];
    size_t jj;
    for (jj = 0; jj < opts->ranks[kk+1]; jj++) {
        big_to_multi(newrows[jj], 2, nValsMulti, newInd);
        memmove(opts->left[kk+1]->indices[jj], 
                opts->left[kk]->indices[newInd[0]], kk * sizeof(size_t));
        opts->left[kk+1]->indices[jj][kk] = newInd[1];
    }
    free(newrows);
    free(Ainv);
    free(Ak);

    return mv_success;
}



struct tt * 
tt_cross(void (*A)(size_t, size_t *, size_t, double *, void *), 
        struct tt_cross_opts * opts, void * args)
{
    size_t ii,kk;
    struct tt * a; // for left->right sweep

    init_tt_alloc(&a, opts->dim, opts->nvals);
    a->ranks[0] = 1;
    for (ii = 0; ii < opts->dim; ii++){
        a->ranks[ii+1] = opts->ranks[ii+1];
        if (NULL == ( a->cores[ii] = malloc(sizeof(struct tensor)))){
            fprintf(stderr, "failed to allocate memory.\n");
            exit(1);
        }
        a->cores[ii]->nvals = calloc_size_t(3);
        a->cores[ii]->nvals[0] = a->ranks[ii];
        a->cores[ii]->nvals[1] = a->nvals[ii];
        a->cores[ii]->nvals[2] = a->ranks[ii+1];
        //a->cores[ii]->vals = calloc_double(opts->ranks[ii] * 
        //                        opts->ranks[ii+1] * opts->nvals[ii]);
        
        a->cores[ii]->vals = cross_setup(A,opts,ii,1, args);
    }
    
   // printf("a ranks ");
   // iprint_sz(a->dim+1,a->ranks);
   // printf("allgood\n");
    
    int done = 0;
    int mv_success;
    double * Ak;
    size_t iter = 0;
    struct tt * aold;
    struct tt * sum_tt;
    double norm_old;
    double norm_sum;
    double diff;
    while (done == 0){
       // printf("weird?\n");
        aold = copy_tt(a);
        ttscal(aold,-1.0);
      //  printf("tweed?\n");
        norm_old = tt_norm(aold);

        //printf("here cross!!\n");
        mv_success = process_first_core_lr(a,A,opts,args);
        for (kk = 1; kk < opts->dim-1; kk++){
            mv_success = process_cores_lr(a,kk, A,opts,args);
        }
        kk = opts->dim-1;
        Ak = cross_setup(A, opts, kk,1, args);
        memmove(a->cores[kk]->vals, Ak, opts->ranks[kk] * opts->nvals[kk] * 
                    opts->ranks[kk+1] * sizeof(double));
        free(Ak);

        mv_success = process_last_core_rl(a,A,opts,args);
        for (kk = opts->dim-2; kk > 0; kk--){
            mv_success = process_cores_rl(a,kk, A,opts,args);
        }
        if (mv_success != 0){
            printf("maxvol success =%d\n",mv_success);
        }
        kk = 0;
        //kk = opts->dim-2;
        Ak = cross_setup(A, opts, 0, 1, args); //nk * rk[-1] x r[0]
        memmove(a->cores[kk]->vals, Ak, opts->ranks[kk] * opts->nvals[kk] * 
                    opts->ranks[kk+1] * sizeof(double));
        free(Ak);
        
        sum_tt = ttadd(a,aold);
        norm_sum = tt_norm(sum_tt);
        diff = norm_sum/norm_old;
        if (opts->verbose == 1){
            printf("||TT_old - TT_new|| is %E (iteration %zu) \n",diff,iter+1);
        }
        freett(aold);
        freett(sum_tt);

        iter++;
        if (diff < opts->epsilon){
            done = 1;
        }
        if (iter > opts->maxiter){
            done = 1;
        }
    }
    return a;
}

struct tt *
tt_cross_adapt(void (*A)(size_t, size_t *, size_t, double *, void *), 
    struct tt_cross_opts * opts, size_t kickrank, size_t maxiter, 
    double epsilon, void * args)
{
    int done = 1;
    size_t ii, jj;
    struct tt * a;
    struct tt * rounded;
    size_t iter = 0;
    size_t temp;

    a = tt_cross(A,opts,args);
    rounded = tt_round(a, epsilon);
    freett(a);
    for (ii = 1; ii < rounded->dim; ii++){
        if (opts->ranks[ii] == rounded->ranks[ii]) {
            done = 0;
            opts->ranks[ii] += kickrank;
        }
    }
    
    if (maxiter == 0){
        return rounded;
    }
    while (done == 0){
        freett(rounded);
        if (opts->verbose == 1){
            printf("Adapting Ranks for TT_Cross iteration=%zu --- avg rank"
                "= %3.5f \n", iter+1, mean_size_t(opts->ranks, opts->dim+1));
        }
        for (ii = 0; ii < opts->dim; ii++){
            
            for (jj = 0; jj < opts->right[ii]->N; jj++){
                free(opts->right[ii]->indices[jj]);
            }
            for (jj = 0; jj < opts->left[ii]->N; jj++){
                free(opts->left[ii]->indices[jj]);
            }
            free(opts->right[ii]->indices);
            free(opts->left[ii]->indices);

            temp = opts->dim-ii-1;
            opts->right[ii]->dim = temp; 
            opts->right[ii]->N = opts->ranks[ii+1];

            if (NULL == (opts->right[ii]->indices = malloc( 
                            opts->right[ii]->N * sizeof(size_t * )))){
                fprintf(stderr, "failed to allocate memory.\n");
                exit(1);
            }
            if (temp == 0){
                for (jj = 0; jj < opts->ranks[ii+1]; jj++){
                    opts->right[ii]->indices[jj] = NULL;
                }
            }
            else{
                for (jj = 0; jj < opts->ranks[ii+1]; jj++){
                    opts->right[ii]->indices[jj] = 
                                calloc_size_t(opts->right[ii]->dim);
                }
            }

            opts->left[ii]->dim = ii;
            opts->left[ii]->N = opts->ranks[ii];
            if (NULL == (opts->left[ii]->indices = malloc( 
                            opts->left[ii]->N * sizeof(size_t *)))){
                fprintf(stderr, "failed to allocate memory.\n");
                exit(1);
            }
            if (ii == 0){
                for (jj = 0; jj < opts->ranks[ii]; jj++){
                    opts->left[ii]->indices[jj] = NULL;
                }
            }
            else{
                for (jj = 0; jj < opts->ranks[ii]; jj++){
                    opts->left[ii]->indices[jj] = 
                            calloc_size_t(opts->left[ii]->dim);
                }
            }
        }

        for (ii = 0; ii < opts->dim-1; ii++){
            create_naive_index_set(opts->right[ii], opts->nvals + ii);
            create_naive_index_set(opts->left[ii], opts->nvals);
        }
        iter += 1;
        done = 1;
        a = tt_cross(A,opts,args);
        rounded = tt_round(a, epsilon);
        freett(a);
        for (ii = 1; ii < rounded->dim; ii++){
            if (opts->ranks[ii] == rounded->ranks[ii]) {
                done = 0;
                opts->ranks[ii] += kickrank;
            }
        }
        if (iter > maxiter){
            done = 1;
        }
    }
    return rounded;
}

