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
    size_t size_elem;
    void * x;
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

void cross_node_add(struct CrossNode ** cn, size_t n, void * x, size_t size_elem)
{
    if (*cn == NULL){
        *cn = cross_node_alloc();
        (*cn)->n = n;
        (*cn)->size_elem = size_elem;
        (*cn)->x = malloc(n*size_elem);
        assert ((*cn)->x != NULL);
        memmove((*cn)->x,x,n*size_elem);
        (*cn)->next = NULL;
    }
    else{
        cross_node_add(&((*cn)->next),n,x,size_elem);
    }
}

void cross_node_add_nested(struct CrossNode ** cn, int left, size_t nold, void * xold,
                           size_t nnew, void * xnew, size_t size_elem)
{
    size_t n = nold + nnew;
    void * x = malloc(n * size_elem);
    assert (x != NULL);
    
    if (left == 0){
        memmove(x,xold,size_elem * nold ) ;
        memmove((char*)x+( size_elem * nold ),xnew,size_elem * nnew);
    }
    else{
        memmove(x,xnew,size_elem * nnew ) ;
        memmove((char*)x+( size_elem * nnew ),xold,size_elem * nold);
    }
    cross_node_add(cn,n,x,sizeof(double));
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

void cross_index_add_index(struct CrossIndex * cn, size_t d, void * x, size_t size_elem)
{
    assert (d == cn->d);
    cn->n += 1;
    cross_node_add(&(cn->nodes),d,x,size_elem);
}

void cross_index_add_nested(struct CrossIndex * cn, int left, 
                            size_t dold, void * xold, size_t dnew, void * xnew, size_t size_elem)
{
    assert ( (dold+dnew) == cn->d);
    cn->n += 1;
    cross_node_add_nested(&(cn->nodes),left,dold,xold,dnew,xnew,size_elem);
}

struct CrossNode * cross_index_get_node(struct CrossIndex * c, size_t ind)
{
    assert (ind < c->n);
    return cross_node_get(c->nodes,ind);
}


struct CrossIndex * cross_index_copy(struct CrossIndex * ci)
{
    /* printf("cross_index_copy\n"); */
    if (ci == NULL){
        return NULL;
    }
    /* printf("in cross_index_copy\n"); */
    struct CrossIndex * ci_new = cross_index_alloc(ci->d);
    for (size_t ii = 0; ii < ci->n; ii++){
        /* printf("ii=%zu/%zu\n",ii,ci->n); */
        struct CrossNode * node = cross_index_get_node(ci,ii);
        cross_index_add_index(ci_new,node->n,node->x,node->size_elem);
    }
    return ci_new;
}

void * cross_index_get_node_value(struct CrossIndex * c, size_t ind, size_t *n)
{
    if (c == NULL){
        return NULL;
    }
    assert (ind < c->n);
    struct CrossNode * temp = cross_node_get(c->nodes,ind);
    if (temp == NULL){
        *n = 0;
        return NULL;
    }
    void * x = temp->x;
    *n = temp->n;
    return x;
}

struct CrossIndex *
cross_index_create_nested(int method, int left, 
                          size_t sizenew, size_t nopts,
                          void * newopts, struct CrossIndex * old)
{
    // need to make sure can add enough unique points
    assert ( (old->n * nopts) >= sizenew);
    size_t dold = old->d;
    struct CrossIndex * ci = cross_index_alloc(dold+1);

    if (method == 1){ // add new options before reusing nodes from old
        struct CrossNode * oc = old->nodes;

        for (size_t ii = 0; ii < old->n; ii++){
            for (size_t jj = 0; jj < nopts; jj++){
                cross_index_add_nested(ci,left,oc->n,oc->x,1,(char *)newopts + jj * oc->size_elem,
                                       oc->size_elem);
                if (ci->n == sizenew){
                    return ci;
                }
            }
            oc = oc->next;
        }
    }
    else if(method == 0){ // add old opts before reusing nodes from old
        for (size_t jj = 0; jj < nopts; jj++){
            struct CrossNode * oc = old->nodes;
            for (size_t ii = 0; ii < old->n; ii++){
                cross_index_add_nested(ci,left,oc->n,oc->x,1,
                                       (char *)newopts + jj * oc->size_elem,
                                       oc->size_elem);
                if (ci->n == sizenew){
                    return ci;
                }
                oc = oc->next;
            }
        }
    }
    else if (method == 2){
        assert (sizenew <= nopts);
        if (old->n < sizenew){
            struct CrossNode * oc = old->nodes;
            for (size_t jj = 0; jj < old->n; jj++){
                cross_index_add_nested(ci,left,oc->n,oc->x,1,
                                       (char *)newopts + jj * oc->size_elem,
                                       oc->size_elem);
                oc = oc->next;
            }
            oc = old->nodes;
            for (size_t jj = old->n; jj < sizenew; jj++){
                cross_index_add_nested(ci,left,oc->n,oc->x,1,
                                       (char *)newopts + jj * oc->size_elem,
                                       oc->size_elem);
            }
        }
        else{
            struct CrossNode * oc = old->nodes;
            for (size_t jj = 0; jj < sizenew; jj++){
                cross_index_add_nested(ci,left,oc->n,oc->x,1,
                                       (char *)newopts + jj * oc->size_elem,
                                       oc->size_elem);
                oc = oc->next;
            }
        }

    }
    else{
        fprintf(stderr,"Something wrong creating a nested index set\n");
        cross_index_free(ci);
        return NULL;
    }
    return ci;
}

struct CrossIndex *
cross_index_create_nested_ind(int left, size_t sizenew, 
                              size_t * indold,
                              void * newx, struct CrossIndex * old)
{
    size_t dold = old->d;
    struct CrossIndex * ci = cross_index_alloc(dold+1);

    for (size_t ii = 0; ii < sizenew; ii++){
        struct CrossNode * cn = cross_index_get_node(old,indold[ii]);
        cross_index_add_nested(ci,left,cn->n,cn->x,1,
                               (char *)newx + ii * cn->size_elem,
                               cn->size_elem);
    }
    return ci;
}

double **
cross_index_merge_wspace(struct CrossIndex * left,
                         struct CrossIndex * right)
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
    /* if (right == NULL){ */
    /*     vals = malloc(left->n * sizeof(double *)); */
    /*     if (vals == NULL){ */
    /*         fprintf(stderr, "Cannot allocate values for merging CrossIndex\n"); */
    /*         exit(1); */
    /*     }    */
    /*     size_t dl = left->d; */
    /*     size_t d = dl + 1;  */
    /*     for (size_t ii = 0; ii < left->n; ii++){ */
    /*         vals[ii] */
    /*     } */
    /* } */

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
                                  size_t * sizes, void ** vals, size_t elem_size)
{

    int newfirst = 2;
    if (allnull == 1){
        for (size_t ii = 0; ii < dim; ii++){
            ci[ii] = NULL;
        }
    }
    else if (reverse == 1){
        /* assert (reverse == 1); */
        assert (dim > 1);
        ci[dim-1] = NULL;
        ci[dim-2] = cross_index_alloc(1);
        for (size_t jj = 0; jj < sizes[dim-1];jj++){
            cross_index_add_index(ci[dim-2],1,(char *)(vals[dim-1]) + jj*elem_size,elem_size);
        }
        size_t ind = dim-2;
        for (size_t ii = 0; ii < dim-2; ii++){
            ind = ind-1;
            /* printf("ind = %zu size=%zu, vals = \n",ind,sizes[ind+1]); */
            /* dprint(sizes[ind+1],vals[ind+1]); */
            ci[ind] = 
                cross_index_create_nested(newfirst,1,sizes[ind+1],
                                          sizes[ind+1],
                                          vals[ind+1], ci[ind+1]);
            
        }
    }
    else{
//        newfirst = 1;
        ci[0] = NULL;
        ci[1] = cross_index_alloc(1);
//        printf("on first one\n");
        for (size_t jj = 0; jj < sizes[0]; jj++){
            cross_index_add_index(ci[1],1,(char *)(vals[0]) + jj * elem_size,elem_size);
        }
//        printf("hello\n");
        /* printf("index set[1] = \n"); */
        /* print_cross_index(ci[1]); */
        for (size_t ii = 2; ii < dim; ii++){
            /* printf("ii=%zu x options are\n",ii); */
            /* dprint(sizes[ii-1],vals[ii]); */
            ci[ii] = cross_index_create_nested(newfirst,0,sizes[ii-1],sizes[ii-1],vals[ii-1],ci[ii-1]);
            /* printf("resulting cross index is\n"); */
            /* print_cross_index(ci[ii]); */
        }
    }
    
}




void cross_index_copylast(struct CrossIndex * ci, size_t ntimes)
{

    if (ci != NULL){
        struct CrossNode * last = cross_index_get_node(ci,ci->n-1);
        for (size_t ii = 0; ii < ntimes; ii++){
            cross_index_add_index(ci,last->n,last->x,last->size_elem);
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

