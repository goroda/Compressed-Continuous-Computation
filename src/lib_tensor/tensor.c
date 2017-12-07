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
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "array.h"
#include "linalg.h"
#include "tensor.h"

/********************************************************//**
    Function init_tensor

    Purpose: Initialize a tensor

    Parameters:
        - tensor (IN) - uninitialized tensor
        - dim (IN)    - number of dimensions of the tensor
        - nvals (IN)  - number of values per dimension

    Returns: Nothing.
***********************************************************/
void 
init_tensor(struct tensor ** a, size_t dim, size_t * nvals)
{
    if (NULL == ( (*a) = malloc(sizeof(struct tensor)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    (*a)->dim = dim;
    (*a)->nvals = calloc_size_t(dim);
    memmove((*a)->nvals, nvals, dim * sizeof(size_t));

    size_t nEntries = iprod_sz(dim, nvals);
    (*a)->vals = calloc_double(nEntries);
}

/********************************************************//**
    Function tensor_elem

    Purpose: Get an element of a tensor 

    Parameters:
        - elem (IN) - indices of the element

    Returns: value of the array
***********************************************************/
double tensor_elem(struct tensor * t, size_t * elem){
    
    size_t ii;
    size_t big_index = elem[0];
    size_t mult = t->nvals[0];
    for (ii = 1; ii < t->dim; ii++){
        big_index += elem[ii] * mult;
        mult *= t->nvals[ii];
    }
    return t->vals[big_index];
}

/********************************************************//**
    Function read_tensor_3d

    Purpose: read a 3d tensor in from memory

    Parameters:
        - fp (IN) - file pointer

    Returns: tensor
***********************************************************/
struct tensor * read_tensor_3d(FILE * fp)
{
    size_t ii, jj, kk;
    struct tensor * a;

    size_t nvals[3];
    nvals[0] = (size_t)( fgetc(fp)-'0');
    fgetc(fp); 
    nvals[1] = (size_t) ( fgetc(fp)-'0');
    fgetc(fp); 
    nvals[2] = (size_t) ( fgetc(fp)-'0');
    //iprint(3, nvals);
    init_tensor(&a, 3, nvals);
    fgetc(fp);
    char * buffer= malloc(2*nvals[2] * sizeof(double)/sizeof(char) * sizeof(char));
    char * pch;    

    buffer = fgets(buffer, sizeof(buffer), fp);
    if (buffer == NULL){
        fprintf(stderr, "failed to read in the 3d tensor file.\n");
        exit(1);
    }
    pch = strtok(buffer, " \n");

    for (ii = 0; ii < nvals[1]; ii++){
        for (jj = 0; jj < nvals[0]; jj++){
            kk = 0;
            while (pch != NULL)
            {
                sscanf(pch, "%lf",
                    &(a->vals[jj + ii * nvals[0] + kk * nvals[0] * nvals[1]]));
                //printf("%f ",(a->vals[jj + ii * nvals[0] + kk * nvals[0] * nvals[1]]));
                //printf("%s ", pch);
                pch = strtok(NULL, " \n");
                kk++;
            }
            buffer = fgets(buffer, sizeof(buffer), fp);
            if (buffer == NULL){
                fprintf(stderr, "failed to read in the 3d tensor file.\n");
                exit(1);
            }
            //printf("\n");
        }
    }
    free(buffer);
    //printf("%s ", buffer);

    return a;
}

/********************************************************//**
    Function free_tensor

    Purpose: Free the memory allocated to a tensor

    Parameters:
        - tensor (IN) - uninitialized tensor

    Returns: Nothing.
***********************************************************/
void 
free_tensor(struct tensor ** a)
{
    free((*a)->vals);
    free((*a)->nvals);
    free((*a));
}

/********************************************************//**
    Function check_right_ortho

    Purpose: Check if a 3d tensor is right orthogonalized
             i.e. sum_{i=1}^n Q[:,i,:] Q[:,i,:]^T = I

    Parameters:
        - core (IN) - three dimensional tensor representing a TT core

    Returns: Nothing. void function (though should check if its idenity)
***********************************************************/
void check_right_ortho(struct tensor * core){
    size_t jj, kk;
    size_t * nvals = core->nvals;
    double * temp;
    if (NULL == ( temp = calloc(nvals[0]*nvals[0], sizeof(double)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    // check right orthogonality
    for (jj = 0; jj < nvals[1]; jj ++){
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nvals[0], nvals[0], 
                nvals[2], 1.0,
                core->vals + jj * nvals[0],
                nvals[0] * nvals[1],
                core->vals + jj * nvals[0],
                nvals[0] * nvals[1],
                1.0,
                temp,
                nvals[0]);
    }
    printf("check right orth \n");
    for (jj = 0; jj < nvals[0]; jj++){
        for (kk = 0; kk < nvals[0]; kk++){
            printf("%3.2f ",temp[jj + nvals[0] * kk]);
        }
        printf("\n");
    }
    free(temp);

}
/********************************************************//**
    Function tensor_stack2h_3d

    Purpose: stack two 3d tensors horizontally 
             C[:,i,:] = [ A[:,i,:] B[:,i,:] ]

    Parameters:
        - a (in) - left core
        - b (in) - right core

    Returns: core - horizontally concatenated core
***********************************************************/
struct tensor * 
tensor_stack2h_3d(const struct tensor * a, const struct tensor * b)
{
    // change this function later
    size_t nvals[3];
    nvals[0] = a->nvals[0];
    nvals[1] = a->nvals[1];
    nvals[2] = a->nvals[2] + b->nvals[2];
    struct tensor * core;
    init_tensor(&core, 3, nvals);
    memmove(core->vals, a->vals, nvals[0] * nvals[1] *
                    a->nvals[2] * sizeof(double));

    memmove(core->vals + nvals[0] * nvals[1] * a->nvals[2], b->vals, 
        nvals[0] * nvals[1] * b->nvals[2] * sizeof(double));

    return core;
}

/********************************************************//**
    Function tensor_stack2v_3d

    Purpose: stack two 3d tensors vertically
             C[:,i,:] = [ A[:,i,:];
                          B[:,i,:] ]

    Parameters:
        - a (in) - top core
        - b (in) - bottom core

    Returns: core - vertically concatenated core
***********************************************************/
struct tensor * 
tensor_stack2v_3d(const struct tensor * a, const struct tensor * b)
{
    // change this function later
    size_t ii,jj;
    size_t nvals[3];
    nvals[0] = a->nvals[0] + b->nvals[0];
    nvals[1] = a->nvals[1];
    nvals[2] = a->nvals[2];
    struct tensor * core;
    init_tensor(&core, 3, nvals);
    
    for (ii = 0; ii < nvals[1]; ii++){
        for (jj = 0; jj < nvals[2]; jj++){
            memmove(core->vals + ii * nvals[0] + jj * nvals[0] * nvals[1], 
                    a->vals + ii * a->nvals[0] + jj * a->nvals[0] * nvals[1],
                    a->nvals[0] * sizeof(double));

            memmove(core->vals + a->nvals[0] + ii * nvals[0] + jj * nvals[0] * nvals[1], 
                    b->vals + ii * b->nvals[0] + jj * b->nvals[0] * nvals[1],
                    b->nvals[0] * sizeof(double));
        }
    }
    return core;
}

/********************************************************//**
    Function tensor_blockdiag_3d

    Purpose: create a block diagonal of 3d tensors
             C[:,i,:] = [ A[:,i,:] 0;
                          0 B[:,i,:] ]

    Parameters:
        - a (in) - top left core
        - b (in) - bottom right core

    Returns: core - block diagonal core
***********************************************************/
struct tensor * 
tensor_blockdiag_3d(const struct tensor * a, const struct tensor * b)
{
    size_t nvals_ur[3];
    nvals_ur[0] = a->nvals[0];
    nvals_ur[1] = a->nvals[1];
    nvals_ur[2] = b->nvals[2];
    struct tensor * upper_right;
    init_tensor(&upper_right, 3, nvals_ur);
    
    size_t nvals_ll[3];
    nvals_ll[0] = b->nvals[0];
    nvals_ll[1] = b->nvals[1];
    nvals_ll[2] = a->nvals[2];
    struct tensor * lower_left;
    init_tensor(&lower_left, 3, nvals_ll);
    
    struct tensor * up = tensor_stack2h_3d(a, upper_right);
    struct tensor * down = tensor_stack2h_3d(lower_left, b);
    struct tensor * core = tensor_stack2v_3d(up,down);
    
    free_tensor(&upper_right);
    free_tensor(&lower_left);
    free_tensor(&up);
    free_tensor(&down);
    return core;
}

/********************************************************//**
    Function print_tensor_3d

    Purpose: display a 3d tensor

    Parameters:
        - a (in) - TT 1
        - opt (in) - way to display
                   if 1 then  A[:,0,:]
                                  |
                                  |
                              A[:,n-1,:]

    Returns: Nothing. (void function)
***********************************************************/
void print_tensor_3d(struct tensor * a, int opt){
    size_t ii,jj,kk;
    if (opt == 1){
        for (ii = 0; ii < a->nvals[1]; ii ++){
            printf(" Tensor [:,%zu,:] \n",ii);
            for (jj = 0; jj < a->nvals[0]; jj++){
                for (kk = 0; kk < a->nvals[2]; kk++){
                    printf("%3.2f ", a->vals[jj + ii * a->nvals[0] + kk * a->nvals[0]* a->nvals[1]]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}

/********************************************************//**
    Function tensor_ones_3d

    Purpose: create a 3d tensor filled with ones

    Parameters:
        - rankl (in) - left rank
        - rankr (in) - right rank
        - N (in) - number of elements

    Returns: core - TT core

***********************************************************/
struct tensor * tensor_ones_3d(size_t rankl, size_t N, size_t rankr)
{
    size_t nvals[3];
    nvals[0] = rankl;
    nvals[1] = N;
    nvals[2] = rankr;
    struct tensor * core;
    init_tensor(&core, 3, nvals);
    
    size_t ii;
    for (ii = 0; ii < iprod_sz(3, nvals); ii++){
        core->vals[ii] = 1.0;
    }
    return core;
}

struct tensor *
tensor_x_3d(size_t N, const double * x)
{
    struct tensor * core;
    size_t nvals[3];
    nvals[0] = 1;
    nvals[1] = N;
    nvals[2] = 1;
    init_tensor(&core, 3, nvals);
    memmove(core->vals, x, N*sizeof(double));
    return core;
}

/*
Computes sum
given ttc A, vector b, vector c, compute
sum_{i} A[:,i,:]*b[i]
*/
double * tensor_sum2(const struct tensor * ttc, const double * b)//, double * out)
{
    // maybe should figure out how to vectorize this and use blas library
    
    double * out = calloc_double(ttc->nvals[0] * ttc->nvals[2]);
    size_t ii, jj, kk;
    for (ii = 0; ii < ttc->nvals[1]; ii++){
        for (jj = 0; jj < ttc->nvals[2]; jj++){
            for (kk = 0; kk < ttc->nvals[0]; kk++){
                out[jj*ttc->nvals[0]+kk] += b[ii] * 
                    ttc->vals[kk + ii * ttc->nvals[0] + jj * ttc->nvals[0] * ttc->nvals[1]];
            }
        }
    }
    return out;
}

// kron of each element
struct tensor * tensor_kron_3d(const struct tensor * a, const struct tensor * b)
{
    
    size_t nvals[3];
    nvals[0] = a->nvals[0] * b->nvals[0];
    nvals[1] = a->nvals[1];
    nvals[2] = a->nvals[2] * b->nvals[2];
    struct tensor * out;
    init_tensor(&out, 3, nvals);
    size_t ii;
    for (ii = 0; ii < nvals[1]; ii++){
        kron_col(a->nvals[0], a->nvals[2], a->vals + ii * a->nvals[0], a->nvals[0]* a->nvals[1],
             b->nvals[0], b->nvals[2], b->vals + ii * b->nvals[0] , b->nvals[0]*b->nvals[1],
             out->vals + ii * nvals[0], out->nvals[0]*out->nvals[1]);
    }
    return out;
}



/********************************************************//**
    Function tensor_copy_3d

    Purpose: copy a 3d tensor

    Parameters:
        - tensor (IN) - 3d tensor
        - transpose (IN)  - whether or not to transpose

    Returns: new core
***********************************************************/
struct tensor * tensor_copy_3d(const struct tensor * t, int transpose){
    
    struct tensor * out;
    if (transpose == 0){
        init_tensor(&out, 3, t->nvals);
        memmove(out->vals, t->vals, iprod_sz(3, t->nvals) * sizeof(double));
    }
    else{
        size_t nvals[3];
        nvals[0] = t->nvals[2];
        nvals[1] = t->nvals[1];
        nvals[2] = t->nvals[0];
        init_tensor(&out, 3, t->nvals);
        size_t ii,jj,kk;
        for (ii = 0; ii < nvals[0]; ii++){
            for (jj = 0; jj < nvals[1]; jj++){
                for(kk=0; kk < nvals[2]; kk++){
                    out->vals[ii + jj * nvals[0] + kk * nvals[0] * nvals[1]] =
                    t->vals[kk + jj * nvals[2] + ii * nvals[1] * nvals[2]];
                }
            }
        }
    }
    return out;
}

