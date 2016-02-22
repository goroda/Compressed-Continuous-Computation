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

/** \file indmanage.c
 * Provides routines for managing index sets associated with cross approximation algorithm
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "array.h"
#include "indmanage.h"

struct CrossNode
{
    size_t n;
    double * x;
    struct CrossNode * next;
};

struct CrossNode * cross_node_alloc()
{
    struct CrossNode * cn = NULL;
    cn = malloc(sizeof(struct CrossNode));
    if (cn == NULL){
        fprintf(stderr,"Cannot Allocate CrossNode\n");
        exit(1);
    }
    cn->n = 0;
    cn->x = NULL;
    cn->next = NULL;
    return cn;
}

void cross_node_free(struct CrossNode * cn)
{
    if (cn != NULL){
        free(cn->x); cn->x = NULL;
        cross_node_free(cn->next);
        cn->next = NULL;
        free(cn); cn = NULL;
    }
}

void cross_node_add(struct CrossNode ** cn, size_t n, double * x)
{
    if (*cn == NULL){
        *cn = cross_node_alloc();
        (*cn)->n = n;
        (*cn)->x = calloc_double(n);
        memmove((*cn)->x,x,n*sizeof(double));
        (*cn)->next = NULL;
    }
    else{
        cross_node_add(&((*cn)->next),n,x);
    }
}

void cross_node_add_nested(struct CrossNode ** cn, int left, size_t nold, double * xold, double xnew)
{
    size_t n = nold + 1;
    double * x = calloc_double(n);
    if (left == 0){
        memmove(x,xold,nold*sizeof(double));
        x[nold] = xnew;
    }
    else{
        memmove(x+1,xold,nold*sizeof(double));
        x[0] = xnew;
    }
    cross_node_add(cn,n,x);
    free(x); x = NULL;
}

struct CrossNode * cross_node_get(struct CrossNode * cn, size_t ind)
{
    struct CrossNode * out = cn;
    size_t iter = 0;
    while (iter < ind){
        out = out->next;
        iter++;
    }
    return out;
}

void print_cross_nodes(struct CrossNode * cn){
    if (cn == NULL){
        printf("NULL\n");
    }
    else{
        while(cn->next != NULL){
            printf(" ==> ");
            dprint(cn->n,cn->x);
            cn = cn->next;
        }
        printf(" ==> ");
        dprint(cn->n,cn->x);
        printf(" NULL\n");
    }
}

////////////////////////////////////////////////////////////////////////
struct CrossIndex * cross_index_alloc(size_t dind)
{
    struct CrossIndex * ci = NULL;
    ci = malloc(sizeof(struct CrossIndex));
    if (ci == NULL){
        fprintf(stderr,"Cannot Allocate Cross Index\n");
        exit(1);
    }
    ci->d = dind;
    ci->n = 0;
    ci->nodes = NULL;
    return ci;
}

void cross_index_free(struct CrossIndex * ci)
{
    if (ci != NULL){
        cross_node_free(ci->nodes); ci->nodes = NULL;
        free(ci); ci = NULL;
    }
}

void cross_index_add_index(struct CrossIndex * cn, size_t d, double * x)
{
    assert (d == cn->d);
    cn->n += 1;
    cross_node_add(&(cn->nodes),d,x);
}

void cross_index_add_nested(struct CrossIndex * cn, int left, 
                            size_t dold, double * xold, double xnew)
{
    assert ( (dold+1) == cn->d);
    cn->n += 1;
    cross_node_add_nested(&(cn->nodes),left,dold,xold,xnew);
}

struct CrossNode * cross_index_get_node(struct CrossIndex * c, size_t ind)
{
    assert (ind < c->n);
    return cross_node_get(c->nodes,ind);
}

struct CrossIndex *
cross_index_create_nested(int newfirst, int left, 
                          size_t sizenew, size_t nopts,
                          double * newopts, struct CrossIndex * old)
{
    // need to make sure can add enough unique points
    assert ( (old->n * nopts) >= sizenew);
    size_t dold = old->d;
    struct CrossIndex * ci = cross_index_alloc(dold+1);
    
    // add new options before reusing nodes from old

    if (newfirst == 1){
        struct CrossNode * oc = old->nodes;
        for (size_t ii = 0; ii < old->n; ii++){
            for (size_t jj = 0; jj < nopts; jj++){
                cross_index_add_nested(ci,left,oc->n,oc->x,newopts[jj]);
                if (ci->n == sizenew){
                    return ci;
                }
            }
            oc = oc->next;
        }
    }
    else{
        for (size_t jj = 0; jj < nopts; jj++){
            struct CrossNode * oc = old->nodes;
            for (size_t ii = 0; ii < old->n; ii++){
                cross_index_add_nested(ci,left,oc->n,oc->x,newopts[jj]);
                if (ci->n == sizenew){
                    return ci;
                }
                oc = oc->next;
            }
        }
    }
    fprintf(stderr,"Something went wrong creating a nested index set\n");
    cross_index_free(ci);
    return NULL;
}

struct CrossIndex *
cross_index_create_nested_ind(int left, size_t sizenew, size_t * indold,
                              double * newx, struct CrossIndex * old)
{
    size_t dold = old->d;
    struct CrossIndex * ci = cross_index_alloc(dold+1);

    for (size_t ii = 0; ii < sizenew; ii++){
        struct CrossNode * cn = cross_index_get_node(old,indold[ii]);
        cross_index_add_nested(ci,left,cn->n,cn->x,newx[ii]);
    }
    return ci;
}

double **
cross_index_merge_wspace(struct CrossIndex * left, struct CrossIndex * right)
{
    double ** vals = NULL;
    if ( (left != NULL) && (right != NULL) ){
        
        vals = malloc(right->n * left->n * sizeof(double *));
        if (vals == NULL){
            fprintf(stderr, "Cannot allocate values for merging CrossIndex\n");
            exit(1);
        }
        size_t dl = left->d;
        size_t dr = right->d;
        size_t d = dl+dr+1;
        struct CrossNode * cl = left->nodes;
        struct CrossNode * cr = right->nodes;
        size_t onind;
        for (size_t ii = 0; ii < right->n;ii++){
            cl = left->nodes;
            for (size_t jj = 0; jj < left->n;jj++){
                onind = ii*left->n+jj;
                vals[onind] = calloc_double(d);
                memmove(vals[onind],cl->x,dl*sizeof(double));
                memmove(vals[onind]+dl+1,cr->x,dr*sizeof(double));
                cl = cl->next;
            }
            cr = cr->next;
        }
    }
    else if (left == NULL){
        assert (right != NULL);
        vals = malloc(right->n * sizeof(double *));
        if (vals == NULL){
            fprintf(stderr, "Cannot allocate values for merging CrossIndex\n");
            exit(1);
        }
        size_t d = right->d+1;
        struct CrossNode * cn = right->nodes;
        for (size_t ii = 0; ii < right->n; ii++){
            vals[ii] = calloc_double(d);
            memmove(vals[ii]+1,cn->x,(d-1)*sizeof(double));
            cn = cn->next;
        }
    }
    else if (right == NULL){
        assert (left != NULL);
        
        vals = malloc(left->n * sizeof(double *));
        if (vals == NULL){
            fprintf(stderr, "Cannot allocate values for merging CrossIndex\n");
            exit(1);
        }
        size_t d = left->d+1;
        struct CrossNode * cn = left->nodes;
        for (size_t ii = 0; ii < left->n; ii++){
            vals[ii] = calloc_double(d);
            memmove(vals[ii],cn->x,(d-1)*sizeof(double));
            cn = cn->next;
        }
    }

    return vals;
}

double **
cross_index_merge(struct CrossIndex * left, struct CrossIndex * right)
{
    double ** vals = NULL;
    assert (left != NULL);
    assert (right != NULL);
    assert (left->n == right->n);
    vals = malloc(right->n * sizeof(double *));
    if (vals == NULL){
        fprintf(stderr, "Cannot allocate values for merging CrossIndex\n");
        exit(1);
    }
    size_t dl = left->d;
    size_t dr = right->d;
    size_t d = dl+dr;
    struct CrossNode * cl = left->nodes;
    struct CrossNode * cr = right->nodes;
    for (size_t ii = 0; ii < right->n;ii++){
        cl = left->nodes;
        vals[ii] = calloc_double(d);
        memmove(vals[ii],cl->x,dl*sizeof(double));
        memmove(vals[ii]+dl,cr->x,dr*sizeof(double));
        cl = cl->next;
        cr = cr->next;
    }
    return vals;
}

void cross_index_array_initialize(size_t dim, struct CrossIndex ** ci,
                                  int allnull, int reverse,
                                  size_t * sizes, double ** vals)
{

    if (allnull == 1){
        for (size_t ii = 0; ii < dim; ii++){
            ci[ii] = NULL;
        }
    }
    else{
        assert (reverse == 1);
        assert (dim > 1);
        ci[dim-1] = NULL;
        ci[dim-2] = cross_index_alloc(1);
        for (size_t jj = 0; jj < sizes[dim-1];jj++){
            cross_index_add_index(ci[dim-2],1,&(vals[dim-1][jj]));
        }
        size_t ind = dim-2;
        for (size_t ii = 0; ii < dim-2; ii++){
            ind = ind-1;
            //printf("ind = %zu size=%zu, vals = \n",ind,sizes[ind+1]);
            //dprint(sizes[ind+1],vals[ind+1]);
            ci[ind] = cross_index_create_nested(1,1,sizes[ind+1],
                                                sizes[ind+1],
                                                vals[ind+1], ci[ind+1]);
            
        }
    }
}


void cross_index_copylast(struct CrossIndex * ci, size_t ntimes)
{

    if (ci != NULL){
        struct CrossNode * last = cross_index_get_node(ci,ci->n-1);
        for (size_t ii = 0; ii < ntimes; ii++){
            cross_index_add_index(ci,last->n,last->x);
        }
    }
    
}


void print_cross_index(struct CrossIndex * cn)
{
    if (cn != NULL){
        printf("Cross Index\n");
        printf("-----------\n");
        printf("Number of nodes = %zu\n ",cn->n);
        printf("Dim of nodes    = %zu\n",cn->d);
        printf("Nodes are \n");
        print_cross_nodes(cn->nodes);
    }
    else{
        printf("Cross index is NULL\n");
    }
}
