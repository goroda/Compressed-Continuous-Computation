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


/** \file parameterization.c
 * Provides routines for parameterizing the FT
 */

#include <string.h>
#include <assert.h>
#include <math.h>
#include <cblas.h>

#include <time.h>

#include "ft.h"
#include "lib_linalg.h"
#include "superlearn_util.h"


#include "parameterization.h"


/***********************************************************//**
    Specify what type of structure exists in the parameterization
    
    \param[in] ftp - parameterized function train
    
    \returns LINEAR_ST if linear or NONE_ST if nonlinear
***************************************************************/
enum FTPARAM_ST ft_param_extract_structure(const struct FTparam * ftp)
{
    enum FTPARAM_ST structure = LINEAR_ST;

    for (size_t ii = 0; ii < ftp->dim; ii++){
        int islin = multi_approx_opts_linear_p(ftp->approx_opts,ii);
        if (islin == 0){
            structure = NONE_ST;
            break;
        }
    }
    return structure;
}


/***********************************************************//**
    Allocate parameterized function train

    \param[in] dim    - size of input space
    \param[in] aopts  - approximation options
    \param[in] params - parameters
    \param[in] ranks  - ranks (dim+1,)

    \return parameterized FT
***************************************************************/
struct FTparam *
ft_param_alloc(size_t dim,
               struct MultiApproxOpts * aopts,
               double * params, size_t * ranks)
{
    if (ranks == NULL){
        fprintf(stderr, "ranks for parameterized ft are not specified in allocation\n");
        exit(1);
    }
    if (aopts == NULL){
        fprintf(stderr, "approximation options for parameterized ft are not specified in allocation\n");
        exit(1);
    }
    struct FTparam * ftr = malloc(sizeof(struct FTparam));
    if (ftr == NULL){
        fprintf(stderr, "Cannot allocate FTparam structure\n");
        exit(1);
    }
    ftr->approx_opts = aopts;
    ftr->dim = dim;
    
    ftr->nparams_per_core = calloc_size_t(ftr->dim);
    ftr->nparams = 0;
    size_t nuni = 0; // number of univariate functions
    for (size_t jj = 0; jj < ftr->dim; jj++)
    {
        nuni += ranks[jj]*ranks[jj+1];
        ftr->nparams_per_core[jj] = ranks[jj]*ranks[jj+1] * multi_approx_opts_get_dim_nparams(aopts,jj);
        ftr->nparams += ftr->nparams_per_core[jj];
    }
    
    ftr->nparams_per_uni = calloc_size_t(nuni);
    ftr->max_param_uni = 0;
    size_t onind = 0;
    for (size_t jj = 0; jj < ftr->dim; jj++){

        for (size_t ii = 0; ii < ranks[jj]*ranks[jj+1]; ii++){
            ftr->nparams_per_uni[onind] = multi_approx_opts_get_dim_nparams(aopts,jj);
            if (ftr->nparams_per_uni[onind] > ftr->max_param_uni){
                ftr->max_param_uni = ftr->nparams_per_uni[onind];
            }
            onind++;
        }
    }


    
    ftr->ft = function_train_zeros(aopts,ranks);
    ftr->params = calloc_double(ftr->nparams);
    if (params != NULL){
        memmove(ftr->params,params,ftr->nparams*sizeof(double));
        function_train_update_params(ftr->ft,ftr->params);
    }
    return ftr;
}

/***********************************************************//**
    Free memory allocated for FT parameterization structure
    
    \param[in,out] ftr - parameterized FT
***************************************************************/
void ft_param_free(struct FTparam * ftr)
{
    if (ftr != NULL){
        function_train_free(ftr->ft); ftr->ft = NULL;
        free(ftr->nparams_per_uni); ftr->nparams_per_uni = NULL;
        free(ftr->nparams_per_core); ftr->nparams_per_core = NULL;
        free(ftr->params); ftr->params = NULL;
        free(ftr); ftr = NULL;
    }
}


/**********************************************************//**
    Copy a function-train parameterization
    
    \param[in] ftp - parameterized FT

    \returns copied
***************************************************************/
struct FTparam * ft_param_copy(struct FTparam * ftp)
{
    struct FTparam * copied = ft_param_alloc(ftp->dim, ftp->approx_opts, ftp->params,
                                             function_train_get_ranks(ftp->ft));
    return copied;
}

/***********************************************************//**
    Get a parameter

    \param[in] ftp   - parameterized function train
    \param[in] index - index of parameter

    \return value of the parameters
***************************************************************/
double ft_param_get_param(const struct FTparam * ftp, size_t index)
{
    if (ftp->nparams > 0){
        if (index < ftp->nparams){
            return ftp->params[index];
        }
        else{
            fprintf(stderr,"Index of parameter to retrieve exceeds number of parameters\n");
            exit(1);
        }
    }
    else{
        fprintf(stderr,"No parameters have yet been specified\n");
        exit(1);
    }
}

/***********************************************************//**
    Get parameters

    \param[in] ftp   - parameterized function train
    \param[in] nparams - number of parameters to copy
    \param[in, out] params - empty parameter array

    \return number of parameters copied
***************************************************************/
size_t ft_param_get_params(const struct FTparam * ftp, size_t nparams, double * params)
{
    if (ftp->nparams <= 0) {
        fprintf(stderr,"No parameters have yet been specified\n");
        exit(1);
    } 
    else if (nparams <= ftp->nparams){
        for (size_t i=0; i<nparams; i++) {
            params[i] = ftp->params[i];
        }
        return nparams;
    }
    else{
        for (size_t i=0; i<ftp->nparams; i++) {
            params[i] = ftp->params[i];
        }
        return ftp->nparams;
    }
}

/***********************************************************//**
    Get number of parameters 

    \param[in] ftp - parameterized FTP

    \return number of parameters
***************************************************************/
size_t ft_param_get_nparams(const struct FTparam * ftp)
{
    return ftp->nparams;
}


/***********************************************************//**
    Get number of dimensions 

    \param[in] ftp - parameterized FTP

    \return number of dimensions
***************************************************************/
size_t ft_param_get_dim(const struct FTparam * ftp)
{
    assert (ftp != NULL);
    return ftp->dim;
}

/***********************************************************//**
    Update the parameters of an FT

    \param[in,out] ftp    - parameterized FTP
    \param[in]     params - new parameter values
***************************************************************/
void ft_param_update_params(struct FTparam * ftp, const double * params)
{
    memmove(ftp->params,params,ftp->nparams * sizeof(double) );
    function_train_update_params(ftp->ft,ftp->params);
}


/***********************************************************//**
    Get the number of parameters of an FT for univariate functions
    >= ranks_start

    \param[in] ftp        - parameterized FTP
    \param[in] rank_start - starting ranks for which to obtain number of parameters (dim-1,)
***************************************************************/
size_t ft_param_get_nparams_restrict(const struct FTparam * ftp, const size_t * rank_start)
{
    size_t nparams = 0;
    size_t ind = 0;
    for (size_t kk = 0; kk < ftp->dim; kk++){
        for (size_t jj = 0; jj < ftp->ft->ranks[kk+1]; jj++){
            for (size_t ii = 0; ii < ftp->ft->ranks[kk]; ii++){
                if (kk > 0){
                    if ( (ii >= rank_start[kk-1]) || (jj >= rank_start[kk]) ){
                        nparams += ftp->nparams_per_uni[ind];
                    }
                }
                else{ // kk == 0
                    if (jj >= rank_start[kk]){
                        nparams += ftp->nparams_per_uni[ind];
                    }
                }
                ind++;
            }
        }
    }
    return nparams;
}

/***********************************************************//**
    Update the parameters of an FT for univariate functions
    >= ranks_start

    \param[in,out] ftp        - parameterized FTP
    \param[in]     params     - parameters for univariate functions at locations >= ranks_start
    \param[in]     rank_start - starting ranks for which to obtain number of parameters (dim-1,)

    \note
    As always FORTRAN ordering (columns first, then rows)
***************************************************************/
void ft_param_update_restricted_ranks(struct FTparam * ftp,
                                      const double * params, const size_t * rank_start)
{

    size_t ind = 0;
    size_t onparam_new = 0;
    size_t onparam_general = 0;
    for (size_t kk = 0; kk < ftp->dim; kk++){
        for (size_t jj = 0; jj < ftp->ft->ranks[kk+1]; jj++){
            for (size_t ii = 0; ii < ftp->ft->ranks[kk]; ii++){
                for (size_t ll = 0; ll < ftp->nparams_per_uni[ind]; ll++){
                    if (kk > 0){
                        if ( (ii >= rank_start[kk-1]) || (jj >= rank_start[kk]) ){
                            ftp->params[onparam_general] = params[onparam_new];
                            onparam_new++;
                        }
                    }
                    else{ // kk == 0
                        if (jj >= rank_start[kk]){
                            ftp->params[onparam_general] = params[onparam_new];
                            onparam_new++;
                        }
                    }

                    onparam_general++;
                }
                ind++;
            }
        }
        
    }
    function_train_update_params(ftp->ft,ftp->params);
}

/***********************************************************//**
    Update the parameters of an FT for univariate functions
    < ranks_start

    \param[in,out] ftp        - parameterized FTP
    \param[in]     params     - parameters for univariate functions at locations < ranks_start
    \param[in]     rank_start - threshold of ranks at which not to update

    \note
    As always FORTRAN ordering (columns first, then rows)
***************************************************************/
void ft_param_update_inside_restricted_ranks(struct FTparam * ftp,
                                             const double * params, const size_t * rank_start)
{

    size_t ind = 0;
    size_t onparam_new = 0;
    size_t onparam_general = 0;
    for (size_t kk = 0; kk < ftp->dim; kk++){
        for (size_t jj = 0; jj < ftp->ft->ranks[kk+1]; jj++){
            for (size_t ii = 0; ii < ftp->ft->ranks[kk]; ii++){
                for (size_t ll = 0; ll < ftp->nparams_per_uni[ind]; ll++){
                    if (kk > 0){
                        if ( (ii < rank_start[kk-1]) && (jj < rank_start[kk]) ){
                            ftp->params[onparam_general] = params[onparam_new];
                            onparam_new++;
                        }
                    }
                    else{ // kk == 0
                        if (jj < rank_start[kk]){
                            ftp->params[onparam_general] = params[onparam_new];
                            onparam_new++;
                        }
                    }

                    onparam_general++;
                }
                ind++;
            }
        }
        
    }
    function_train_update_params(ftp->ft,ftp->params);
}


/***********************************************************//**
    Update the parameters of an FT for a specific core

    \param[in,out] ftp    - parameterized FTP
    \param[in]     core   - core to update
    \param[in]     params - parameters
***************************************************************/
void ft_param_update_core_params(struct FTparam * ftp, size_t core, const double * params)
{
    size_t runparam = 0;
    for (size_t ii = 0; ii < core; ii ++){
        runparam += ftp->nparams_per_core[ii];
    }
    function_train_core_update_params(ftp->ft,core,
                                      ftp->nparams_per_core[core],
                                      params);

    memmove(ftp->params + runparam,params,ftp->nparams_per_core[core] * sizeof(double) );
}


/***********************************************************//**
    Update the parameterization of an FT

    \param[in,out] ftp       - parameterized FTP
    \param[in]     opts      - new approximation options
    \param[in]     new_ranks - new ranks
    \param[in]     new_vals  - new parameters values
***************************************************************/
void ft_param_update_structure(struct FTparam ** ftp,
                               struct MultiApproxOpts * opts,
                               size_t * new_ranks, double * new_vals)
{
    // just overwrites
    size_t dim = (*ftp)->dim;
    ft_param_free(*ftp); *ftp = NULL;
    *ftp = ft_param_alloc(dim,opts,new_vals,new_ranks);
}


/***********************************************************//**
    Get a reference to an array storing the number of parameters per core                      

    \param[in] ftp - parameterized FTP

    \returns number of parameters per core
***************************************************************/
size_t * ft_param_get_nparams_per_core(const struct FTparam * ftp)
{
    assert (ftp != NULL);
    return ftp->nparams_per_core;
}

/***********************************************************//**
    Get the number of parameters of a core                      

    \param[in] ftp  - parameterized FTP
    \param[in] core - index of core

    \returns number of parameters per core
***************************************************************/
size_t ft_param_get_core_nparams(const struct FTparam * ftp, size_t core)
{
    assert (ftp != NULL);
    assert (core < ftp->dim);
    return ftp->nparams_per_core[core];
}

/***********************************************************//**
    Get a reference to the underlying FT

    \param[in] ftp - parameterized FTP

    \returns a function train
***************************************************************/
struct FunctionTrain * ft_param_get_ft(const struct FTparam * ftp)
{
    assert(ftp != NULL);
    return ftp->ft;
}

/***********************************************************//**
    Create a parameterization that is initialized to a constant

    \param[in,out] ftp     - parameterized FTP
    \param[in]     val     - number of data points
    \param[in]     perturb - perturbation to zero elements
***************************************************************/
void ft_param_create_constant(struct FTparam * ftp, double val,
                              double perturb)
{
    size_t * ranks = function_train_get_ranks(ftp->ft);

    struct FunctionTrain * const_ft =
        function_train_constant(val,ftp->approx_opts);

    // free previous parameters
    free(ftp->params); ftp->params = NULL;
    ftp->params = calloc_double(ftp->nparams);
    for (size_t ii = 0; ii < ftp->nparams; ii++){
        ftp->params[ii] = perturb*(randu()*2.0-1.0);
    }

    size_t onparam = 0;
    size_t onfunc = 0;
    for (size_t ii = 0; ii < ftp->dim; ii++){

        size_t mincol = 1;
        size_t maxcol = ranks[ii+1];
        if (mincol > maxcol){
            mincol = maxcol;
        }

        size_t minrow = 1;
        size_t maxrow = ranks[ii];
        if (minrow > maxrow ){
            minrow = maxrow;
        }

        size_t nparam_temp = function_train_core_get_nparams(const_ft,ii,NULL);
        double * temp_params = calloc_double(nparam_temp);
        function_train_core_get_params(const_ft,ii,temp_params);
        size_t onparam_temp = 0;

        /* printf("on core = %zu\n,ii"); */
        for (size_t col = 0; col < mincol; col++){
            for (size_t row = 0; row < minrow; row++){

                size_t nparam_temp_func =
                    function_train_func_get_nparams(const_ft,ii,row,col);

                size_t minloop = nparam_temp_func;
                size_t maxloop = ftp->nparams_per_uni[onfunc];
                if (maxloop < minloop){
                    minloop = maxloop;
                }
                for (size_t ll = 0; ll < minloop; ll++){
                    /* ftp->params[onparam] = temp_params[onparam_temp]; */
                    ftp->params[onparam] += temp_params[onparam_temp];
                    /* ftp->params[onparam] += 0.001*randn(); */
                    onparam++;
                    onparam_temp++;
                }
                for (size_t ll = minloop; ll < maxloop; ll++){
                    /* ftp->params[onparam] = 0.0; */
                    onparam++;
                }
                onfunc++;
            }
            
            for (size_t row = minrow; row < maxrow; row++){
                onparam += ftp->nparams_per_uni[onfunc];
                onfunc++;
            }
        }
        for (size_t col = mincol; col < maxcol; col++){
            for (size_t row = 0; row < maxrow; row++){
                onparam += ftp->nparams_per_uni[onfunc];
                onfunc++;
            }
        }

        free(temp_params); temp_params = NULL;
    }

    // update the function train
    function_train_update_params(ftp->ft,ftp->params);
    function_train_free(const_ft); const_ft = NULL;
}

/******************************************Rf*****************//**
    Create a parameterization from a linear least squares fit to 
    x and y

    \param[in,out] ftp     - parameterized FTP
    \param[in]     N       - number of data points
    \param[in]     x       - features
    \param[in]     y       - labels
    \param[in]     perturb - perturbation to zero elements
    \param[in]     seed    - seed to use for random number generation
    \note
    If ranks are < 2 then performs a constant fit at the mean of the data
    Else creates top 2x2 blocks to be a linear least squares fit and sets everything
    else to 1e-12
***************************************************************/
void ft_param_create_from_lin_ls(struct FTparam * ftp, size_t N,
                                 const double * x, const double * y,
                                 double perturb, const unsigned int * seed)
{

    // perform LS
    size_t * ranks = function_train_get_ranks(ftp->ft);

    // create A matrix (change from row major to column major)
    double * A = calloc_double(N * (ftp->dim+1));
    for (size_t ii = 0; ii < ftp->dim; ii++){
        for (size_t jj = 0; jj < N; jj++){
            A[ii*N+jj] = x[jj*ftp->dim+ii];
        }
    }
    for (size_t jj = 0; jj < N; jj++){
        A[ftp->dim*N+jj] = 1.0;
    }

    
    double * b = calloc_double(N);
    memmove(b,y,N * sizeof(double));

    double * weights = calloc_double(ftp->dim+1);

    /* printf("A = \n"); */
    /* dprint2d_col(N,ftp->dim,A); */

    // includes offset
    linear_ls(N,ftp->dim+1,A,b,weights);

    /* printf("weights = "); dprint(ftp->dim,weights); */
    /* for (size_t ii = 0; ii < ftp->dim;ii++){ */
    /*     weights[ii] = randn(); */
    /* } */
    
    // now create the approximation
    double * a = calloc_double(ftp->dim);
    for (size_t ii = 0; ii < ftp->dim; ii++){
        a[ii] = weights[ftp->dim]/(double)ftp->dim;
    }
    struct FunctionTrain * linear_temp = function_train_linear(weights,1,a,1,ftp->approx_opts);

    struct FunctionTrain * const_temp = function_train_constant(weights[ftp->dim],ftp->approx_opts);
    /* function_train_free(ftp->ft); */
    /* ftp->ft = function_train_copy(temp); */
    
    // free previous parameters
    free(ftp->params); ftp->params = NULL;

    if (seed != NULL){
        /* printf("fixed seed to %u\n", *seed);         */
        srand(*seed);
    }

    ftp->params = calloc_double(ftp->nparams);
    for (size_t ii = 0; ii < ftp->nparams; ii++){
        ftp->params[ii] = perturb*(randu()*2.0-1.0);
    }

    size_t onparam = 0;
    size_t onfunc = 0;
    for (size_t ii = 0; ii < ftp->dim; ii++){


        size_t mincol = 2;
        size_t maxcol = ranks[ii+1];
        if (mincol > maxcol){
            mincol = maxcol;
        }

        size_t minrow = 2;
        size_t maxrow = ranks[ii];
        if (minrow > maxrow ){
            minrow = maxrow;
        }
        struct FunctionTrain * temp = linear_temp;
        if ((mincol == 1) && (minrow == 1)){
            temp = const_temp;
        }        

        size_t nparam_temp = function_train_core_get_nparams(temp,ii,NULL);
        double * temp_params = calloc_double(nparam_temp);
        function_train_core_get_params(temp,ii,temp_params);
        size_t onparam_temp = 0;

        /* printf("on core = %zu\n,ii"); */
        for (size_t col = 0; col < mincol; col++){
            for (size_t row = 0; row < minrow; row++){


                size_t nparam_temp_func = function_train_func_get_nparams(temp,ii,row,col);

                size_t minloop = nparam_temp_func;
                size_t maxloop = ftp->nparams_per_uni[onfunc];
                if (maxloop < minloop){
                    minloop = maxloop;
                }
                for (size_t ll = 0; ll < minloop; ll++){
                    /* ftp->params[onparam] = temp_params[onparam_temp]; */
                    ftp->params[onparam] += temp_params[onparam_temp];
                    /* ftp->params[onparam] += 0.001*randn(); */
                    onparam++;
                    onparam_temp++;
                }
                for (size_t ll = minloop; ll < maxloop; ll++){
                    /* ftp->params[onparam] = 0.0; */
                    onparam++;
                }
                onfunc++;
            }
            
            for (size_t row = minrow; row < maxrow; row++){
                onparam += ftp->nparams_per_uni[onfunc];
                onfunc++;
            }
        }
        for (size_t col = mincol; col < maxcol; col++){
            for (size_t row = 0; row < maxrow; row++){
                onparam += ftp->nparams_per_uni[onfunc];
                onfunc++;
            }
        }

        free(temp_params); temp_params = NULL;
    }


    // update the function train
    function_train_update_params(ftp->ft,ftp->params);

    /* fprintf(stdout, "Function train parameters = \n"); */
    /* dprint(ftp->nparams, ftp->params); */
    
    function_train_free(const_temp); const_temp = NULL;
    function_train_free(linear_temp); linear_temp = NULL;
    free(a); a = NULL;
    
    free(A); A = NULL;
    free(b); b = NULL;
    free(weights); weights = NULL;
}


/***********************************************************//**
    Evaluate an ft that has each univariate core parameterized with linear parameters

    \param[in,out] ftp        - parameterized FTP
    \param[in]     grad_evals - gradient of each univariate function wrt the parameter
    \param[in,out] mem        - workspace (number of univariate functions, 1)

    \return evaluation
***************************************************************/
double ft_param_eval_lin(struct FTparam * ftp, const double * grad_evals, double * mem)
{

    /* printf("ft_param_eval_lin\n"); */
    size_t onuni = 0;
    size_t onparam = 0;
    size_t * ranks = function_train_get_ranks(ftp->ft);

    // Except for the first core
    // previous core runs form indmem to indmem + ranks[kk]
    // new core evaluation runs from indmeme+ranks[kk] to indmeme+ranks[kk}+ranks[kk+1]
    size_t indmem = 0;
    assert (ranks[0] == 1);

    for (size_t col = 0; col < ranks[1]; col++){
        mem[col] = cblas_ddot(ftp->nparams_per_uni[onuni],grad_evals+onparam,1,
                              ftp->params + onparam,1);
        onparam += ftp->nparams_per_uni[onuni];
        onuni++;
    }

    for (size_t kk = 1; kk < ftp->dim; kk++){
        for (size_t col = 0; col < ranks[kk+1]; col++){
            mem[indmem+ranks[kk]+col] = 0.0;
            for (size_t row = 0; row < ranks[kk]; row++){
                double dd = cblas_ddot(ftp->nparams_per_uni[onuni],grad_evals+onparam,1,
                                       ftp->params + onparam,1);
                mem[indmem+ranks[kk]+col] += mem[indmem+row] * dd;
                onparam += ftp->nparams_per_uni[onuni];
                onuni++;
            }
        }
        if (kk > 0){
            indmem += ranks[kk];
        }
    }

    double out = mem[indmem];
    /* free(mem); mem = NULL; */
    /* printf("got it\n"); */
    return out;
}

/***********************************************************//**
    Evaluate an ft that has each univariate core parameterized with linear parameters

    \param[in,out] ftp        - parameterized FTP
    \param[in]     grad_evals - gradient of each univariate function wrt the parameter
    \param[in,out] grad       - gradient wrt each parameter in each univariate function
    \param[in,out] mem        - workspace (number of univariate functions, 1)
    \param[in,out] evals      - workspace (number of univariate functions)

    \return evaluation
***************************************************************/
double ft_param_gradeval_lin(struct FTparam * ftp, const double * grad_evals,
                             double * grad, double * mem, double * evals)
{


    size_t onuni = 0;
    size_t onparam = 0;
    size_t * ranks = function_train_get_ranks(ftp->ft);

    // Except for the first core
    // previous core runs form indmem to indmem + ranks[kk]
    // new core evaluation runs from indmeme+ranks[kk] to indmeme+ranks[kk}+ranks[kk+1]
    size_t indmem = 0;
    assert (ranks[0] == 1);

    // Forward sweep
    for (size_t kk = 0; kk < ftp->dim; kk++){
        for (size_t col = 0; col < ranks[kk+1]; col++){
            if (kk > 0){
                mem[indmem+ranks[kk]+col] = 0.0;
                for (size_t row = 0; row < ranks[kk]; row++){
                    evals[onuni] = cblas_ddot(ftp->nparams_per_uni[onuni],
                                             grad_evals + onparam, 1,
                                             ftp->params + onparam,1);
                    mem[indmem+ranks[kk]+col] += mem[indmem+row] * evals[onuni];
                    onparam += ftp->nparams_per_uni[onuni];
                    onuni++;
                }

            }
            else{
                mem[col] = cblas_ddot(ftp->nparams_per_uni[onuni],
                                      grad_evals + onparam, 1,
                                      ftp->params + onparam,1);
                evals[col] = mem[col];
                onparam += ftp->nparams_per_uni[onuni];
                onuni++;
            }
        }
        if (kk > 0){
            indmem += ranks[kk];
        }
    }


    double out = mem[indmem];
    size_t backind=0,r1,r2;

    // backward sweep
    // mem[indmem] is the product of all the cores
    // mem[indmem-r1] is the product of all but the last core
    for (size_t zz = 0; zz < ftp->dim-1; zz++)
    {
        backind = ftp->dim-1-zz;
        r1 = ranks[backind];
        r2 = ranks[backind+1];

        indmem -= r1;

        onuni = onuni - r1*r2;
        onparam = onparam - ftp->nparams_per_core[backind];

        /* printf("onparam start = %zu\n",onparam); */
        if (backind < ftp->dim-1){
            for (size_t ii = 0; ii < r2; ii++){
                double right_mult = mem[indmem + r1 + ii];
                for (size_t kk = 0; kk < r1; kk++){
                    double left_mult = mem[indmem + kk];
                    for (size_t ll = 0; ll < ftp->nparams_per_uni[onuni];ll++){
                        grad[onparam] =  left_mult*grad_evals[onparam]*right_mult;
                        onparam++;
                    }
                    onuni++;
                }
            }
            onparam -= ftp->nparams_per_core[backind];
            onuni -= r1*r2;
            cblas_dgemv(CblasColMajor, CblasNoTrans,
                        r1,r2,1.0,
                        evals + onuni, r1,
                        mem + indmem + r1, 1,
                        0.0,mem + indmem, 1);
                            
        }
        else{
            /* printf("LETS GO indmem=%zu!\n",indmem); */
            for (size_t kk = 0; kk < r1; kk++){
                double left_mult = mem[indmem + kk];
                for (size_t ll = 0; ll < ftp->nparams_per_uni[onuni]; ll++){
                    grad[onparam] = left_mult*grad_evals[onparam];
                    onparam++;
                }
                mem[indmem + kk] = evals[onuni];
                onuni++;
            }
            onparam -= ftp->nparams_per_core[backind];
            onuni -= r1*r2;
        }
    }

    backind = 0;
    /* printf("done with all but first, backind = %zu\n",backind); */
    // handle first core;
    r1 = ranks[backind];
    r2 = ranks[backind+1];

    /* // current core will run from indmem to indmem + r2; */
    indmem -= r1;

    onuni = onuni - r1*r2;
    onparam = onparam - ftp->nparams_per_core[backind];

    for (size_t ii = 0; ii < r2; ii++){
        double right_mult = mem[indmem + r1 + ii];
        for (size_t ll = 0; ll < ftp->nparams_per_uni[onuni]; ll++){
            grad[onparam] =  grad_evals[onparam]*right_mult;
            onparam++;
        }
        onuni++;
    }

    return out;
}


/***********************************************************//**
    Evaluate the gradient of the ft with respect to each parameter

    \param[in,out] ftp        - parameterized FTP
    \param[in]     N          - number of data points
    \param[in]     x          - locations at which to evaluate
    \param[in,out] grad       - gradient wrt each parameter in each univariate function
    \param[in,out] grad_evals - workspace (number of univariate functions)
    \param[in,out] mem        - workspace (number of univariate functions)
    \param[in,out] evals      - workspace (number of univariate functions)

***************************************************************/
void ft_param_gradevals(struct FTparam * ftp, size_t N, const double * x,
                         double * grad,
                         double * grad_evals,
                         double * mem, double * evals)
{
    for (size_t ii=0; ii<N; ii++){
        ft_param_gradeval(ftp, x + ii*ftp->dim, grad + ii*ftp->nparams, grad_evals, mem, evals);
    }
}

/***********************************************************//**
    Evaluate the gradient of the ft with respect to each parameter

    \param[in,out] ftp        - parameterized FTP
    \param[in]     x          - location at which to evaluate
    \param[in,out] grad       - gradient wrt each parameter in each univariate function
    \param[in,out] grad_evals - workspace (number of univariate functions)
    \param[in,out] mem        - workspace (number of univariate functions)
    \param[in,out] evals      - workspace (number of univariate functions)

    \return evaluation
***************************************************************/
double ft_param_gradeval(struct FTparam * ftp, const double * x,
                         double * grad,
                         double * grad_evals,
                         double * mem, double * evals)
{

    size_t onuni = 0;
    size_t onparam = 0;
    size_t * ranks = function_train_get_ranks(ftp->ft);
    struct FunctionTrain * ft = ftp->ft;
    // Except for the first core
    // previous core runs form indmem to indmem + ranks[kk]
    // new core evaluation runs from indmeme+ranks[kk] to indmeme+ranks[kk}+ranks[kk+1]
    size_t indmem = 0;
    assert (ranks[0] == 1);

    // Forward sweep
    for (size_t kk = 0; kk < ftp->dim; kk++){
        for (size_t col = 0; col < ranks[kk+1]; col++){
            if (kk > 0){
                mem[indmem+ranks[kk]+col] = 0.0;
                for (size_t row = 0; row < ranks[kk]; row++){
                    evals[onuni] = generic_function_param_grad_eval2(
                        ft->cores[kk]->funcs[row + col * ranks[kk]],x[kk],grad_evals + onparam);
                        
                    mem[indmem+ranks[kk]+col] += mem[indmem+row] * evals[onuni];
                    onparam += ftp->nparams_per_uni[onuni];
                    onuni++;
                }
            }
            else{
                mem[col] =
                    generic_function_param_grad_eval2(ft->cores[kk]->funcs[col],x[kk],grad_evals + onparam);

                evals[col] = mem[col];
                onparam += ftp->nparams_per_uni[onuni];
                onuni++;
            }
        }
        if (kk > 0){
            indmem += ranks[kk];
        }
    }

    double out = mem[indmem];
    size_t backind=0,r1,r2;

    // backward sweep
    // mem[indmem] is the product of all the cores
    // mem[indmem-r1] is the product of all but the last core
    for (size_t zz = 0; zz < ftp->dim-1; zz++)
    {
        backind = ftp->dim-1-zz;
        r1 = ranks[backind];
        r2 = ranks[backind+1];

        indmem -= r1;

        onuni = onuni - r1*r2;
        onparam = onparam - ftp->nparams_per_core[backind];

        /* printf("onparam start = %zu\n",onparam); */
        if (backind < ftp->dim-1){
            for (size_t ii = 0; ii < r2; ii++){
                double right_mult = mem[indmem + r1 + ii];
                for (size_t kk = 0; kk < r1; kk++){
                    double left_mult = mem[indmem + kk];
                    for (size_t ll = 0; ll < ftp->nparams_per_uni[onuni];ll++){
                        grad[onparam] =  left_mult*grad_evals[onparam]*right_mult;
                        onparam++;
                    }
                    onuni++;
                }
            }
            onparam -= ftp->nparams_per_core[backind];
            onuni -= r1*r2;
            cblas_dgemv(CblasColMajor, CblasNoTrans,
                        r1,r2,1.0,
                        evals + onuni, r1,
                        mem + indmem + r1, 1,
                        0.0,mem + indmem, 1);
                            
        }
        else{
            /* printf("LETS GO indmem=%zu!\n",indmem); */
            for (size_t kk = 0; kk < r1; kk++){
                double left_mult = mem[indmem + kk];
                for (size_t ll = 0; ll < ftp->nparams_per_uni[onuni]; ll++){
                    grad[onparam] = left_mult*grad_evals[onparam];
                    onparam++;
                }
                mem[indmem + kk] = evals[onuni];
                onuni++;
            }
            onparam -= ftp->nparams_per_core[backind];
            onuni -= r1*r2;
        }
    }

    backind = 0;
    /* printf("done with all but first, backind = %zu\n",backind); */
    // handle first core;
    r1 = ranks[backind];
    r2 = ranks[backind+1];

    /* // current core will run from indmem to indmem + r2; */
    indmem -= r1;

    onuni = onuni - r1*r2;
    onparam = onparam - ftp->nparams_per_core[backind];

    for (size_t ii = 0; ii < r2; ii++){
        double right_mult = mem[indmem + r1 + ii];
        for (size_t ll = 0; ll < ftp->nparams_per_uni[onuni]; ll++){
            grad[onparam] =  grad_evals[onparam]*right_mult;
            onparam++;
        }
        onuni++;
    }

    return out;
}

static void update_running_lr(struct FTparam * ftp, size_t core, double * running_lr, double * running_lr_up, double * evals)
{
    size_t * ranks = function_train_get_ranks(ftp->ft);
    if (core > 0){
        cblas_dgemv(CblasColMajor,CblasTrans,ranks[core],ranks[core+1],1.0,evals,ranks[core],running_lr,1,
                    0.0,running_lr_up,1);
                    
    }
    else{
        memmove(running_lr_up,evals,ranks[1]*sizeof(double));
    }
}

static void update_running_rl(struct FTparam * ftp, size_t core,
                              double * running_rl, double * running_rl_up,
                              double * evals)
{
    size_t * ranks = function_train_get_ranks(ftp->ft);
    if (core < ftp->dim-1){
        cblas_dgemv(CblasColMajor,CblasNoTrans,
                    ranks[core],ranks[core+1], 1.0, evals, ranks[core], running_rl, 1,
                    0.0, running_rl_up,1);
    }
    else{
        memmove(running_rl_up,evals,ranks[ftp->dim-1] * sizeof(double));
    }
}


/***********************************************************//**
    Helper function for ALS optimization
***************************************************************/
void process_sweep_left_right_lin(struct FTparam * ftp, size_t current_core, double * grad_evals,
                                  double * running_lr, double * running_lr_up)
{
    size_t * ranks = function_train_get_ranks(ftp->ft);
    size_t r1 = ranks[current_core];
    size_t r2 = ranks[current_core+1];
    size_t onparam = 0;
    size_t onuni = 0;
    for (size_t ii = 0; ii < current_core; ii++){
        onparam += ftp->nparams_per_core[ii];
        onuni += ranks[ii]*ranks[ii+1];
    }

    if (current_core > 0){
        double eval;
        for (size_t ii = 0; ii < r2; ii++){
            running_lr_up[ii] = 0.0;
            for (size_t jj = 0; jj < r1; jj++){
                eval = cblas_ddot(ftp->nparams_per_uni[onuni],grad_evals+onparam,1,ftp->params + onparam,1);
                running_lr_up[ii] += eval * running_lr[jj];
                onparam += ftp->nparams_per_uni[onuni];
                onuni++;
            }
        }
    }
    else{
        for (size_t ii = 0; ii < r2; ii++){
            running_lr_up[ii] = cblas_ddot(ftp->nparams_per_uni[onuni],grad_evals+onparam,1,ftp->params + onparam,1);
            onparam += ftp->nparams_per_uni[onuni];
            onuni++;
        }
    }
}

/***********************************************************//**
    Helper function for ALS optimization
***************************************************************/
void process_sweep_right_left_lin(struct FTparam * ftp, size_t current_core, double * grad_evals,
                                  double * running_rl, double * running_rl_up)
{
    size_t * ranks = function_train_get_ranks(ftp->ft);
    size_t r1 = ranks[current_core];
    size_t r2 = ranks[current_core+1];
    size_t onparam = 0;
    size_t onuni = 0;
    for (size_t ii = 0; ii < current_core; ii++){
        onparam += ftp->nparams_per_core[ii];
        onuni += ranks[ii]*ranks[ii+1];
    }
    
    if (current_core < (ftp->dim-1)){
        double eval;
        for (size_t jj = 0; jj < r1; jj++){
            running_rl_up[jj] = 0.0;
        }
        for (size_t ii = 0; ii < r2; ii++){
            for (size_t jj = 0; jj < r1; jj++){
                eval = cblas_ddot(ftp->nparams_per_uni[onuni],grad_evals + onparam,1,ftp->params + onparam,1);
                running_rl_up[jj] += eval * running_rl[ii];
                onparam += ftp->nparams_per_uni[onuni];
                onuni++;
            }
        }
    }
    else{
        for (size_t ii = 0; ii < r1; ii++){
            running_rl_up[ii] = cblas_ddot(ftp->nparams_per_uni[onuni],grad_evals + onparam,1,ftp->params + onparam,1);
            onparam += ftp->nparams_per_uni[onuni];
            onuni++;
        }
    }
}


/***********************************************************//**
    Helper function for ALS optimization
***************************************************************/
void process_sweep_left_right(struct FTparam * ftp, size_t current_core, double x, double * evals,
                              double * running_lr, double * running_lr_up)
{
    size_t * ranks = function_train_get_ranks(ftp->ft);
    size_t r1 = ranks[current_core];
    size_t r2 = ranks[current_core+1];
    qmarray_param_grad_eval(ftp->ft->cores[current_core],1,&x,1,evals,r1*r2,NULL,0,NULL);
    update_running_lr(ftp,current_core,running_lr,running_lr_up, evals);
}

/***********************************************************//**
    Helper function for ALS optimization
***************************************************************/
void process_sweep_right_left(struct FTparam * ftp, size_t current_core, double x, double * evals,
                              double * running_rl, double * running_rl_up)
{

    size_t * ranks = function_train_get_ranks(ftp->ft);
    size_t r1 = ranks[current_core];
    size_t r2 = ranks[current_core+1];
    qmarray_param_grad_eval(ftp->ft->cores[current_core],1,&x,1,evals,r1*r2,NULL,0,NULL);
    update_running_rl(ftp,current_core,running_rl,running_rl_up, evals);
}


/***********************************************************//**
    Run Gibbs Sampler using Linear Parameterization of FT

    \param[in,out] ftp         - parameterized function train
    \param[in]     N           - number of data points
    \param[in]     x           - training samples
    \param[in]     y           - training labels
    \param[in]     init_sample - original sample to start at
    \param[in]     prior_cov   - prior covariance matrix
    \param[in]     prior_mean  - prior mean
    \param[in]     noise_var   - noise variance
    \param[in]     Nsamples    - number of samples requested
    \param[in,out] out         - space to put samples in

***************************************************************/
void sample_gibbs_linear(struct FTparam * ftp, size_t N, double * x, double * y, 
        double * init_sample, double * prior_cov, double * prior_mean, double noise_var, size_t Nsamples, double * out)
{
    struct SLMemManager * mem = sl_mem_manager_alloc(ftp->dim,N,ftp->nparams,LINEAR_ST);
    sl_mem_manager_check_structure(mem,ftp,x);

    // Initialize with Initial Sample
    ft_param_update_params(ftp, init_sample);
    for (size_t ii=0; ii<ftp->nparams; ii++){
        out[ii] = init_sample[ii];
    }

    // Calculate PseudoInverse of all core covariance matrices
    // COULD BE ROW MAJOR WHICH WOULD MESS UP CALCULATIONS
    size_t running_nparams = 0;
    double inv_prior_cov[ftp->nparams*ftp->nparams];
    for (size_t core=0; core<ftp->dim; core++){
        double core_prior_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];

        for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++) {
            for (size_t jj=0; jj<ftp->nparams_per_core[core]; jj++) {
                core_prior_cov[ftp->nparams_per_core[core]*jj + ii] = prior_cov[ftp->nparams*running_nparams + ftp->nparams*jj + running_nparams + ii];
            }
        }
        double inv_core_prior_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
        // This destroys core_prior_cov
        pinv(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
                core_prior_cov, inv_core_prior_cov, 0.0); // inv_cov = np.linalg.pinv(core_prior_cov)


        for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++) {
            for (size_t jj=0; jj<ftp->nparams_per_core[core]; jj++) {
                inv_prior_cov[ftp->nparams*running_nparams + ftp->nparams*jj + running_nparams + ii] = inv_core_prior_cov[ftp->nparams_per_core[core]*jj + ii];
            }
        }
        
        running_nparams += ftp->nparams_per_core[core];
    }

    // Invert the noise and Make it a matrix
    double inv_noise_var = 1/noise_var;
    double inv_noise_cov[N*N];
    for (size_t ii=0; ii<N; ii++){
        for (size_t jj=0; jj<N; jj++){
            if (ii==jj){
                inv_noise_cov[N*jj + ii] = inv_noise_var;
            } else {
                inv_noise_cov[N*jj +ii] = 0.0;
            }
        }
    }

    // init forward sweep
    for (size_t ii = 0; ii < N; ii++){
        process_sweep_right_left_lin(ftp, ftp->dim-1, mem->lin_structure_vals + ii * ftp->nparams,
                                        NULL, mem->running_rl[ftp->dim-1][ii]);
    }
    
    // THIS STOPS BEFORE EVALUATING THE LAST CORE, SHOULD NOT BREAK ANYTHING
    for (size_t zz = ftp->dim-2; zz > 0; zz--){
        for (size_t ii = 0; ii < N; ii++){
            process_sweep_right_left_lin(ftp,zz,mem->lin_structure_vals + ii * ftp->nparams,
                                            mem->running_rl[zz+1][ii],mem->running_rl[zz][ii]);
        }   
    }
    
    for (size_t iter = 1; iter < Nsamples; iter++){

        running_nparams = 0;
        for (size_t core = 0; core < ftp->dim; core++){

            if (core > 0 && core < ftp->dim-1){
                for (size_t ii = 0; ii < N; ii++){
                    ft_param_core_gradeval_lin(ftp,core, mem->grad->vals + ii * ftp->nparams_per_core[core],
                                                        mem->running_lr[core-1][ii],mem->running_rl[core+1][ii],
                                                        mem->lin_structure_vals + ii * ftp->nparams);
                }
            } else if (core == 0){
                for (size_t ii = 0; ii < N; ii++){
                    ft_param_core_gradeval_lin(ftp,core, mem->grad->vals + ii * ftp->nparams_per_core[core],
                                                        NULL,mem->running_rl[core+1][ii],
                                                        mem->lin_structure_vals + ii * ftp->nparams);
                }
            } else {
                for (size_t ii = 0; ii < N; ii++){
                    ft_param_core_gradeval_lin(ftp,core, mem->grad->vals + ii * ftp->nparams_per_core[core],
                                                        mem->running_lr[core-1][ii],NULL,
                                                        mem->lin_structure_vals + ii * ftp->nparams);
                }
            }

            
            
            // Sample Core
            // ROW MAJOR??
            double inv_core_prior_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++) {
                for (size_t jj=0; jj<ftp->nparams_per_core[core]; jj++) {
                    inv_core_prior_cov[ftp->nparams_per_core[core]*jj + ii] = inv_prior_cov[ftp->nparams*running_nparams + ftp->nparams*jj + running_nparams + ii];
                }
            }

            double core_prior_mean[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                core_prior_mean[ii] = prior_mean[running_nparams + ii];
            }

            

            //cblas_dgemm(Col, TransOpA, TransOpB, M, N, K, al, A, lda, B, ldb, beta, C, ldc)
            // op(A) = M x K
            // op(B) = K x N
            //    C  = M x N
            // lda = cols of A = M or K
            // ldb = cols of B = K or N
            // ldc = cols of C = M


            // mem->grad->vals PASS IN -----------------------------------------------------------------------------------

            // temp = np.dot( self.inv_noise_cov, core_grad )
            // N x N times N x Nparams
            double inv_noise_x_grad[N*ftp->nparams_per_core[core]]; // N x nparams_core
            cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, N, ftp->nparams_per_core[core], N, 1.0, inv_noise_cov, N, mem->grad->vals, ftp->nparams_per_core[core], 0.0, inv_noise_x_grad, ftp->nparams_per_core[core]);
            // cblas_dgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, ftp->nparams_per_core[core], N, 1.0, inv_noise_cov, N, mem->grad->vals, N, 0.0, inv_noise_x_grad, N);
            

            // np.dot(core_grad.T, temp)
            // Nparams x N times N x Nparams
            double gradT_x_noise_x_grad[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]]; // nparams_core x nparams_core
            cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], N, 1.0, mem->grad->vals, ftp->nparams_per_core[core], inv_noise_x_grad, ftp->nparams_per_core[core], 0.0, gradT_x_noise_x_grad, ftp->nparams_per_core[core]);
            // cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], N, 1.0, mem->grad->vals, N, inv_noise_x_grad, N, 0.0, gradT_x_noise_x_grad, ftp->nparams_per_core[core]);

            // term = np.dot(core_grad.T, np.dot( self.inv_noise_cov, core_grad ))
            double mean[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]*ftp->nparams_per_core[core]; ii++){
                mean[ii] = gradT_x_noise_x_grad[ii] + inv_core_prior_cov[ii];
            }

            double inv_mean[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // DOES THIS REQUIRE ROW OR COLUMN MAJOR??
            pinv(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
                mean, inv_mean, 0.0);
            
            // mean_inv = np.linalg.pinv(term + inv_cov)

            double noise_x_data[N];
            // cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, 1.0, inv_noise_cov, N, y, 1, 0.0, noise_x_data, 1);
            cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, inv_noise_cov, N, y, 1, 0.0, noise_x_data, 1);

            double data_term[ftp->nparams_per_core[core]];
            // cblas_dgemv(CblasColMajor, CblasTrans, N, ftp->nparams_per_core[core], 1.0, mem->grad->vals, N, noise_x_data, 1, 0.0, data_term, 1);
            cblas_dgemv(CblasRowMajor, CblasTrans, N, ftp->nparams_per_core[core], 1.0, mem->grad->vals, ftp->nparams_per_core[core], noise_x_data, 1, 0.0, data_term, 1);
            
            // data_term = np.dot(core_grad.T, np.dot( self.inv_noise_cov, self.Ydata ))

            double prior_term[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // cblas_dgemv(CblasColMajor, CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, inv_core_prior_cov, ftp->nparams_per_core[core], core_prior_mean, 1, 0.0, prior_term, 1);
            cblas_dgemv(CblasRowMajor, CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, inv_core_prior_cov, ftp->nparams_per_core[core], core_prior_mean, 1, 0.0, prior_term, 1);

            // prior_term = np.dot(inv_cov, core_prior_mean)

            double data_plus_prior[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                data_plus_prior[ii] = data_term[ii] + prior_term[ii];
            }

            double post_mean[ftp->nparams_per_core[core]];
            // cblas_dgemv(CblasColMajor,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, inv_mean, ftp->nparams_per_core[core], data_plus_prior, 1, 0.0, post_mean,1);
            cblas_dgemv(CblasRowMajor,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, inv_mean, ftp->nparams_per_core[core], data_plus_prior, 1, 0.0, post_mean,1);

            // post_mean = np.dot(mean_inv, data_term + prior_term)

            
            
            // double term_plus_inv_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // for (size_t ii=0; ii<ftp->nparams_per_core[core]*ftp->nparams_per_core[core]; ii++){
            //     term_plus_inv_cov[ii] = gradT_x_noise_x_grad[ii] + inv_core_prior_cov[ii];
            // }

            // double post_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // pinv(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
            //     term_plus_inv_cov, post_cov, 0.0);

            // post_cov = np.linalg.pinv(term + inv_cov)

            double u[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            double s[ftp->nparams_per_core[core]];
            double vt[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // svd(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
            //         post_cov, u, s, vt);
            svd(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
                    inv_mean, u, s, vt);


            double s_mat[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                for (size_t jj=0; jj<ftp->nparams_per_core[core]; jj++){
                    if (ii==jj){
                        s_mat[jj*ftp->nparams_per_core[core] + ii] = sqrt(s[ii]);
                    } else {
                        s_mat[jj*ftp->nparams_per_core[core] + ii] = 0;
                    }
                }
            }
            
            double sqrtcov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, u, ftp->nparams_per_core[core], s_mat, ftp->nparams_per_core[core], 0.0, sqrtcov, ftp->nparams_per_core[core]);
            // cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, u, ftp->nparams_per_core[core], s_mat, ftp->nparams_per_core[core], 0.0, sqrtcov, ftp->nparams_per_core[core]);

            // u, s, v = np.linalg.svd(post_cov)
            // sqrtcov = np.dot(u, np.sqrt(np.diag(s)))

            double rand_arr[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                rand_arr[ii] = randn();
            }

            double bef_core_sample[ftp->nparams_per_core[core]];
            // cblas_dgemv(CblasRowMajor,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, sqrtcov, ftp->nparams_per_core[core], rand_arr, 1, 0.0, bef_core_sample,1);
            cblas_dgemv(CblasColMajor,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, sqrtcov, ftp->nparams_per_core[core], rand_arr, 1, 0.0, bef_core_sample,1);

            double core_sample[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                core_sample[ii] = bef_core_sample[ii] + post_mean[ii];
            }

            // if (core==0 && iter==1){
            //     for (size_t ii=0; ii<ftp->nparams_per_core[core]*ftp->nparams_per_core[core]; ii++){
            //         printf("%ld: %f\n",ii, u[ii]);
            //     }
            //     return;
            // }


            // core_sample = post_mean + np.dot(sqrtcov, np.random.randn(core_nparams))

            // Update Core

            // PASS RESULTS IN --------------------------------------------------------------------------
            ft_param_update_core_params(ftp,core,core_sample);


            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                out[iter*ftp->nparams + running_nparams + ii] = core_sample[ii];
            }

            // Calculate running left 

            running_nparams += ftp->nparams_per_core[core];


            // forward sweep
            if ((core > 0) && (core < ftp->dim-1)){
                for (size_t zz = 0; zz < N; zz++){
                    process_sweep_left_right_lin(ftp, core, mem->lin_structure_vals + zz * ftp->nparams,
                                                    mem->running_lr[core-1][zz], mem->running_lr[core][zz]);
                }
            }
            else if (core == 0){
                for (size_t zz = 0; zz < N; zz++){
                    process_sweep_left_right_lin(ftp, core, mem->lin_structure_vals + zz * ftp->nparams,
                                                    NULL, mem->running_lr[core][zz]);
                }
            }
            
        }


        // backward sweep
        for (size_t zz = 0; zz < N; zz++){
            process_sweep_right_left_lin(ftp, ftp->dim-1, mem->lin_structure_vals + zz * ftp->nparams,
                                            NULL, mem->running_rl[ftp->dim-1][zz]);
        }
            
        for (size_t jj = 1; jj < ftp->dim-1; jj++){
            size_t ii = ftp->dim-1-jj;

            for (size_t zz = 0; zz < N; zz++){
                process_sweep_right_left_lin(ftp,ii,mem->lin_structure_vals + zz * ftp->nparams,
                                                mem->running_rl[ii+1][zz],mem->running_rl[ii][zz]);
            }   
        }

    }
    
    sl_mem_manager_free(mem); mem = NULL;
}

/***********************************************************//**
    Run Gibbs Sampler using Linear Parameterization of FT 
        and sample noise

    \param[in,out] ftp         - parameterized function train
    \param[in]     N           - number of data points
    \param[in]     x           - training samples
    \param[in]     y           - training labels
    \param[in]     init_sample - original sample to start at
    \param[in]     prior_cov   - prior covariance matrix
    \param[in]     prior_mean  - prior mean
    \param[in]     noise_alpha - noise precision prior gamma distribution - alpha
    \param[in]     noise_theta - noise precision prior gamma distribution - theta
    \param[in]     Nsamples    - number of samples requested
    \param[in,out] out         - space to put samples in

***************************************************************/
void sample_gibbs_linear_noise(struct FTparam * ftp, size_t N, double * x, double * y, 
        double * init_sample, double * prior_cov, double * prior_mean, int noise_alpha, 
        double noise_theta, size_t Nsamples, double * out)
{
    struct SLMemManager * mem = sl_mem_manager_alloc(ftp->dim,N,ftp->nparams,LINEAR_ST);
    sl_mem_manager_check_structure(mem,ftp,x);

    // Initialize with Initial Sample
    ft_param_update_params(ftp, init_sample+1);
    for (size_t ii=0; ii<ftp->nparams+1; ii++){
        out[ii] = init_sample[ii];
    }

    // Calculate PseudoInverse of all core covariance matrices
    size_t running_nparams = 0;
    double inv_prior_cov[ftp->nparams*ftp->nparams];
    for (size_t core=0; core<ftp->dim; core++){
        double core_prior_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];

        for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++) {
            for (size_t jj=0; jj<ftp->nparams_per_core[core]; jj++) {
                core_prior_cov[ftp->nparams_per_core[core]*jj + ii] = prior_cov[ftp->nparams*running_nparams + ftp->nparams*jj + running_nparams + ii];
            }
        }
        double inv_core_prior_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
        // This destroys core_prior_cov
        pinv(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
                core_prior_cov, inv_core_prior_cov, 0.0); // inv_cov = np.linalg.pinv(core_prior_cov)


        for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++) {
            for (size_t jj=0; jj<ftp->nparams_per_core[core]; jj++) {
                inv_prior_cov[ftp->nparams*running_nparams + ftp->nparams*jj + running_nparams + ii] = inv_core_prior_cov[ftp->nparams_per_core[core]*jj + ii];
            }
        }
        
        running_nparams += ftp->nparams_per_core[core];
    }


    // init forward sweep
    for (size_t ii = 0; ii < N; ii++){
        process_sweep_right_left_lin(ftp, ftp->dim-1, mem->lin_structure_vals + ii * ftp->nparams,
                                        NULL, mem->running_rl[ftp->dim-1][ii]);
    }
    
    
    for (size_t zz = ftp->dim-2; zz > 0; zz--){
        for (size_t ii = 0; ii < N; ii++){
            process_sweep_right_left_lin(ftp,zz,mem->lin_structure_vals + ii * ftp->nparams,
                                            mem->running_rl[zz+1][ii],mem->running_rl[zz][ii]);
        }   
    }

    // Calculate Noise alpha
    // new_alpha = self.alpha + 0.5*self.Xdata.shape[0]
    int new_alpha = noise_alpha + round(0.5*N);

    // double inv_noise_cov[N*N];
    double * inv_noise_cov = malloc(N*N * sizeof(double));
    for (size_t ii=0; ii<N; ii++){
        for (size_t jj=0; jj<N; jj++){
            inv_noise_cov[N*jj +ii] = 0.0;
        }
    }
    

    // Begin iterations
    for (size_t iter = 1; iter < Nsamples; iter++){

        // Sample Noise
        double ft_evals[N];
        const double * x_in = (const double *) x;
        function_train_evals(ftp->ft, N, x_in, ft_evals);

        double residual = 0;
        for (size_t ii=0; ii<N; ii++){
            residual += (y[ii] - ft_evals[ii])*(y[ii] - ft_evals[ii]);
        }
        double new_beta = (1/noise_theta) + 0.5*residual;

        // Requires alpha >= 1
        double noise_sample = log(randu());
        for (int ii=1; ii<new_alpha; ii++){
            noise_sample += log(randu());
        }
        noise_sample = (-1/new_beta)*noise_sample;

        out[iter*(ftp->nparams+1)] = noise_sample;

        // Fill in identity
        for (size_t ii=0; ii<N; ii++){
            inv_noise_cov[N*ii + ii] = noise_sample;
        }

        // Loop through cores
        running_nparams = 0;
        for (size_t core = 0; core < ftp->dim; core++){

            if (core > 0 && core < ftp->dim-1){
                for (size_t ii = 0; ii < N; ii++){
                    ft_param_core_gradeval_lin(ftp,core, mem->grad->vals + ii * ftp->nparams_per_core[core],
                                                        mem->running_lr[core-1][ii],mem->running_rl[core+1][ii],
                                                        mem->lin_structure_vals + ii * ftp->nparams);
                }
            } else if (core == 0){
                for (size_t ii = 0; ii < N; ii++){
                    ft_param_core_gradeval_lin(ftp,core, mem->grad->vals + ii * ftp->nparams_per_core[core],
                                                        NULL,mem->running_rl[core+1][ii],
                                                        mem->lin_structure_vals + ii * ftp->nparams);
                }
            } else {
                for (size_t ii = 0; ii < N; ii++){
                    ft_param_core_gradeval_lin(ftp,core, mem->grad->vals + ii * ftp->nparams_per_core[core],
                                                        mem->running_lr[core-1][ii],NULL,
                                                        mem->lin_structure_vals + ii * ftp->nparams);
                }
            }

            
            // Sample Core
            double inv_core_prior_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++) {
                for (size_t jj=0; jj<ftp->nparams_per_core[core]; jj++) {
                    inv_core_prior_cov[ftp->nparams_per_core[core]*jj + ii] = inv_prior_cov[ftp->nparams*running_nparams + ftp->nparams*jj + running_nparams + ii];
                }
            }

            double core_prior_mean[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                core_prior_mean[ii] = prior_mean[running_nparams + ii];
            }



            //cblas_dgemm(Col, TransOpA, TransOpB, M, N, K, al, A, lda, B, ldb, beta, C, ldc)
            // op(A) = M x K
            // op(B) = K x N
            //    C  = M x N
            // lda = cols of A = M or K
            // ldb = cols of B = K or N
            // ldc = cols of C = M


            // temp = np.dot( self.inv_noise_cov, core_grad )
            // N x N times N x Nparams
            double inv_noise_x_grad[N*ftp->nparams_per_core[core]]; // N x nparams_core
            cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, N, ftp->nparams_per_core[core], N, 1.0, inv_noise_cov, N, mem->grad->vals, ftp->nparams_per_core[core], 0.0, inv_noise_x_grad, ftp->nparams_per_core[core]);
            

            // np.dot(core_grad.T, temp)
            // Nparams x N times N x Nparams
            double gradT_x_noise_x_grad[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]]; // nparams_core x nparams_core
            cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], N, 1.0, mem->grad->vals, ftp->nparams_per_core[core], inv_noise_x_grad, ftp->nparams_per_core[core], 0.0, gradT_x_noise_x_grad, ftp->nparams_per_core[core]);

            // term = np.dot(core_grad.T, np.dot( self.inv_noise_cov, core_grad ))
            double mean[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]*ftp->nparams_per_core[core]; ii++){
                mean[ii] = gradT_x_noise_x_grad[ii] + inv_core_prior_cov[ii];
            }

            double inv_mean[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // pinv(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
            //     mean, inv_mean, 0.0);
            
            // mean_inv = np.linalg.pinv(term + inv_cov)

            size_t m = ftp->nparams_per_core[core];
            double cutoff = 0.0;
            double u_[m*m];
            double vt_[m*m];
            double s_[m];
            
            // not sure about thir dargument
            svd(m, m, m, mean, u_, s_, vt_); //note changed from m to lda
            
            double * smat = calloc_double(m*m);
            double * smat_sqrt = calloc_double(m*m);

            for (size_t ii = 0; ii < m; ii++){
                if (fabs(s_[ii]) < cutoff){ 
                    smat[ii*m+ii] = 0.0;
                    smat_sqrt[ii*m+ii] = 0.0;
                }
                else{
                    smat[ii*m+ii] = 1.0/s_[ii];
                    smat_sqrt[ii*m+ii] = sqrt(1.0/s_[ii]);
                }
            }
            
            double temp[m*m];
            
            cblas_dgemm(CblasColMajor,CblasTrans,CblasTrans, m, m, m, 1.0, 
                            vt_, m, smat, m, 0.0, temp, m);

            free(smat);
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans, m, m, m, 1.0, 
                            temp, m, u_, m, 0.0, inv_mean, m);










            double noise_x_data[N];
            cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, inv_noise_cov, N, y, 1, 0.0, noise_x_data, 1);

            double data_term[ftp->nparams_per_core[core]];
            cblas_dgemv(CblasRowMajor, CblasTrans, N, ftp->nparams_per_core[core], 1.0, mem->grad->vals, ftp->nparams_per_core[core], noise_x_data, 1, 0.0, data_term, 1);
            
            // data_term = np.dot(core_grad.T, np.dot( self.inv_noise_cov, self.Ydata ))

            double prior_term[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            cblas_dgemv(CblasRowMajor, CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, inv_core_prior_cov, ftp->nparams_per_core[core], core_prior_mean, 1, 0.0, prior_term, 1);

            // prior_term = np.dot(inv_cov, core_prior_mean)

            double data_plus_prior[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                data_plus_prior[ii] = data_term[ii] + prior_term[ii];
            }

            double post_mean[ftp->nparams_per_core[core]];
            cblas_dgemv(CblasRowMajor,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, inv_mean, ftp->nparams_per_core[core], data_plus_prior, 1, 0.0, post_mean,1);

            // post_mean = np.dot(mean_inv, data_term + prior_term)

            
            
            // double term_plus_inv_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // for (size_t ii=0; ii<ftp->nparams_per_core[core]*ftp->nparams_per_core[core]; ii++){
            //     term_plus_inv_cov[ii] = gradT_x_noise_x_grad[ii] + inv_core_prior_cov[ii];
            // }

            // double post_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // pinv(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
            //     term_plus_inv_cov, post_cov, 0.0);

            // post_cov = np.linalg.pinv(term + inv_cov)

            // double u[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // double s[ftp->nparams_per_core[core]];
            // double vt[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // // svd(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
            // //         post_cov, u, s, vt);
            // svd(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
            //         inv_mean, u, s, vt);

            // double s_mat[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
            //     for (size_t jj=0; jj<ftp->nparams_per_core[core]; jj++){
            //         if (ii==jj){
            //             s_mat[jj*ftp->nparams_per_core[core] + ii] = sqrt(s[ii]);
            //         } else {
            //             s_mat[jj*ftp->nparams_per_core[core] + ii] = 0.0;
            //         }
            //     }
            // }
            
            double sqrtcov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, vt_, ftp->nparams_per_core[core], smat_sqrt, ftp->nparams_per_core[core], 0.0, sqrtcov, ftp->nparams_per_core[core]);
            free(smat_sqrt);
            // u, s, v = np.linalg.svd(post_cov)
            // sqrtcov = np.dot(u, np.sqrt(np.diag(s)))

            double rand_arr[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                rand_arr[ii] = randn();
            }

            double bef_core_sample[ftp->nparams_per_core[core]];
            cblas_dgemv(CblasColMajor,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, sqrtcov, ftp->nparams_per_core[core], rand_arr, 1, 0.0, bef_core_sample,1);

            double core_sample[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                core_sample[ii] = bef_core_sample[ii] + post_mean[ii];
            }
            // core_sample = post_mean + np.dot(sqrtcov, np.random.randn(core_nparams))

            // Update Core

            ft_param_update_core_params(ftp,core,core_sample);

            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                out[iter*(ftp->nparams+1) + running_nparams + ii + 1] = core_sample[ii];
            }

            // Calculate running left 

            running_nparams += ftp->nparams_per_core[core];


            // forward sweep
            if ((core > 0) && (core < ftp->dim-1)){
                for (size_t zz = 0; zz < N; zz++){
                    process_sweep_left_right_lin(ftp, core, mem->lin_structure_vals + zz * ftp->nparams,
                                                    mem->running_lr[core-1][zz], mem->running_lr[core][zz]);
                }
            }
            else if (core == 0){
                for (size_t zz = 0; zz < N; zz++){
                    process_sweep_left_right_lin(ftp, core, mem->lin_structure_vals + zz * ftp->nparams,
                                                    NULL, mem->running_lr[core][zz]);
                }
            }
            
        }


        // backward sweep
        for (size_t zz = 0; zz < N; zz++){
            process_sweep_right_left_lin(ftp, ftp->dim-1, mem->lin_structure_vals + zz * ftp->nparams,
                                            NULL, mem->running_rl[ftp->dim-1][zz]);
        }
            
        for (size_t jj = 1; jj < ftp->dim-1; jj++){
            size_t ii = ftp->dim-1-jj;

            for (size_t zz = 0; zz < N; zz++){
                process_sweep_right_left_lin(ftp,ii,mem->lin_structure_vals + zz * ftp->nparams,
                                                mem->running_rl[ii+1][zz],mem->running_rl[ii][zz]);
            }   
        }

    }
    
    sl_mem_manager_free(mem); mem = NULL;
    free(inv_noise_cov);
}


/***********************************************************//**
    Run Gibbs Sampler using Linear Parameterization of FT,
        sample noise precision, and use a hierarchical prior 
        variance across all the cores

    \param[in,out] ftp          - parameterized function train
    \param[in]     N            - number of data points
    \param[in]     x            - training samples
    \param[in]     y            - training labels
    \param[in]     init_sample  - original sample to start at
    \param[in]     prior_alphas - prior alpha for prior variance gamma distribution
    \param[in]     prior_thetas - prior theta 
    \param[in]     noise_alpha  - noise precision prior gamma distribution - alpha
    \param[in]     noise_theta  - noise precision prior gamma distribution - theta
    \param[in]     Nsamples     - number of samples requested
    \param[in,out] out          - space to put samples in

***************************************************************/
void sample_hier_group_gibbs_linear_noise(struct FTparam * ftp, size_t N, double * x, double * y, 
        double * init_sample, int prior_alpha, double prior_theta, int noise_alpha, 
        double noise_theta, size_t Nsamples, double * out)
{
    struct SLMemManager * mem = sl_mem_manager_alloc(ftp->dim,N,ftp->nparams,LINEAR_ST);
    sl_mem_manager_check_structure(mem,ftp,x);

    // Initialize with Initial Sample
    ft_param_update_params(ftp, init_sample+2);
    for (size_t ii=0; ii<ftp->nparams+2; ii++){
        out[ii] = init_sample[ii];
    }


    // init forward sweep
    for (size_t ii = 0; ii < N; ii++){
        process_sweep_right_left_lin(ftp, ftp->dim-1, mem->lin_structure_vals + ii * ftp->nparams,
                                        NULL, mem->running_rl[ftp->dim-1][ii]);
    }
    
    
    for (size_t zz = ftp->dim-2; zz > 0; zz--){
        for (size_t ii = 0; ii < N; ii++){
            process_sweep_right_left_lin(ftp,zz,mem->lin_structure_vals + ii * ftp->nparams,
                                            mem->running_rl[zz+1][ii],mem->running_rl[zz][ii]);
        }   
    }
    // Calculate Noise alpha
    // new_alpha = self.alpha + 0.5*self.Xdata.shape[0]
    int new_n_alpha = noise_alpha + round(0.5*N);

    // Calculate prior alpha
    int new_p_alpha = prior_alpha + round(0.5*ftp->nparams);

    // double inv_noise_cov[N*N];
    double * inv_noise_cov = malloc(N*N * sizeof(double));
    for (size_t ii=0; ii<N; ii++){
        for (size_t jj=0; jj<N; jj++){
            inv_noise_cov[N*jj +ii] = 0.0;
        }
    }

    // Begin iterations
    for (size_t iter = 1; iter < Nsamples; iter++){

        // Sample Noise
        double ft_evals[N];
        const double * x_in = (const double *) x;
        function_train_evals(ftp->ft, N, x_in, ft_evals);

        double residual = 0;
        for (size_t ii=0; ii<N; ii++){
            residual += (y[ii] - ft_evals[ii])*(y[ii] - ft_evals[ii]);
        }
        double new_n_beta = (1/noise_theta) + 0.5*residual;

        // Requires alpha >= 1
        double noise_sample = log(randu());
        for (int ii=1; ii<new_n_alpha; ii++){
            noise_sample += log(randu());
        }
        noise_sample = (-1/new_n_beta)*noise_sample;

        out[iter*(ftp->nparams+2)] = noise_sample;

        // Fill in identity
        for (size_t ii=0; ii<N; ii++){
            inv_noise_cov[N*ii + ii] = noise_sample;
        }


        // Sample Prior Variance

        double squared_sum = 0;
        for (size_t ii=0; ii<ftp->nparams; ii++){
            squared_sum += ftp->params[ii]*ftp->params[ii];
        }

        double new_p_beta = (1/prior_theta) + 0.5*squared_sum;

        double var_sample = log(randu());
        for (int ii=1; ii<new_p_alpha; ii++){
            var_sample += log(randu());
        }
        var_sample = -(1/new_p_beta)*var_sample;

        out[iter*(ftp->nparams+2)+1] = var_sample;

        // Loop through cores
        size_t running_nparams = 0;
        for (size_t core = 0; core < ftp->dim; core++){

            if (core > 0 && core < ftp->dim-1){
                for (size_t ii = 0; ii < N; ii++){
                    ft_param_core_gradeval_lin(ftp,core, mem->grad->vals + ii * ftp->nparams_per_core[core],
                                                        mem->running_lr[core-1][ii],mem->running_rl[core+1][ii],
                                                        mem->lin_structure_vals + ii * ftp->nparams);
                }
            } else if (core == 0){
                for (size_t ii = 0; ii < N; ii++){
                    ft_param_core_gradeval_lin(ftp,core, mem->grad->vals + ii * ftp->nparams_per_core[core],
                                                        NULL,mem->running_rl[core+1][ii],
                                                        mem->lin_structure_vals + ii * ftp->nparams);
                }
            } else {
                for (size_t ii = 0; ii < N; ii++){
                    ft_param_core_gradeval_lin(ftp,core, mem->grad->vals + ii * ftp->nparams_per_core[core],
                                                        mem->running_lr[core-1][ii],NULL,
                                                        mem->lin_structure_vals + ii * ftp->nparams);
                }
            }



            // Create Inverse Prior Covariance Matrix
            double inv_core_prior_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++) {
                for (size_t jj=0; jj<ftp->nparams_per_core[core]; jj++) {
                    if (ii==jj){
                        inv_core_prior_cov[ftp->nparams_per_core[core]*jj + ii] = var_sample;
                    } else {
                        inv_core_prior_cov[ftp->nparams_per_core[core]*jj + ii] = 0.0;
                    }
                }
            }

            double core_prior_mean[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                core_prior_mean[ii] = 0;
            }


            // Sample Core
            //cblas_dgemm(Col, TransOpA, TransOpB, M, N, K, al, A, lda, B, ldb, beta, C, ldc)
            // op(A) = M x K
            // op(B) = K x N
            //    C  = M x N
            // lda = cols of A = M or K
            // ldb = cols of B = K or N
            // ldc = cols of C = M


            // temp = np.dot( self.inv_noise_cov, core_grad )
            // N x N times N x Nparams
            double inv_noise_x_grad[N*ftp->nparams_per_core[core]]; // N x nparams_core
            cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, N, ftp->nparams_per_core[core], N, 1.0, inv_noise_cov, N, mem->grad->vals, ftp->nparams_per_core[core], 0.0, inv_noise_x_grad, ftp->nparams_per_core[core]);

            // np.dot(core_grad.T, temp)
            // Nparams x N times N x Nparams
            double gradT_x_noise_x_grad[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]]; // nparams_core x nparams_core
            cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], N, 1.0, mem->grad->vals, ftp->nparams_per_core[core], inv_noise_x_grad, ftp->nparams_per_core[core], 0.0, gradT_x_noise_x_grad, ftp->nparams_per_core[core]);

            // term = np.dot(core_grad.T, np.dot( self.inv_noise_cov, core_grad ))
            double mean[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]*ftp->nparams_per_core[core]; ii++){
                mean[ii] = gradT_x_noise_x_grad[ii] + inv_core_prior_cov[ii];
            }


            // PSEUDO INVERSE
            double inv_mean[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // pinv(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
            //     mean, inv_mean, 0.0);

            
            size_t m = ftp->nparams_per_core[core];
            double cutoff = 0.0;
            double u_[m*m];
            double vt_[m*m];
            double s_[m];
            
            // not sure about thir dargument
            svd(m, m, m, mean, u_, s_, vt_); //note changed from m to lda
            
            double * smat = calloc_double(m*m);
            double * smat_sqrt = calloc_double(m*m);

            for (size_t ii = 0; ii < m; ii++){
                if (fabs(s_[ii]) < cutoff){ 
                    smat[ii*m+ii] = 0.0;
                    smat_sqrt[ii*m+ii] = 0.0;
                }
                else{
                    smat[ii*m+ii] = 1.0/s_[ii];
                    smat_sqrt[ii*m+ii] = sqrt(1.0/s_[ii]);
                }
            }
            
            double temp[m*m];
            
            cblas_dgemm(CblasColMajor,CblasTrans,CblasTrans, m, m, m, 1.0, 
                            vt_, m, smat, m, 0.0, temp, m);

            free(smat);
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans, m, m, m, 1.0, 
                            temp, m, u_, m, 0.0, inv_mean, m);
            
            





            
            // mean_inv = np.linalg.pinv(term + inv_cov)

            double noise_x_data[N];
            cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, inv_noise_cov, N, y, 1, 0.0, noise_x_data, 1);

            double data_term[ftp->nparams_per_core[core]];
            cblas_dgemv(CblasRowMajor, CblasTrans, N, ftp->nparams_per_core[core], 1.0, mem->grad->vals, ftp->nparams_per_core[core], noise_x_data, 1, 0.0, data_term, 1);
            
            // data_term = np.dot(core_grad.T, np.dot( self.inv_noise_cov, self.Ydata ))

            double prior_term[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            cblas_dgemv(CblasRowMajor, CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, inv_core_prior_cov, ftp->nparams_per_core[core], core_prior_mean, 1, 0.0, prior_term, 1);

            // prior_term = np.dot(inv_cov, core_prior_mean)

            double data_plus_prior[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                data_plus_prior[ii] = data_term[ii] + prior_term[ii];
            }

            double post_mean[ftp->nparams_per_core[core]];
            cblas_dgemv(CblasRowMajor,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, inv_mean, ftp->nparams_per_core[core], data_plus_prior, 1, 0.0, post_mean,1);

            // post_mean = np.dot(mean_inv, data_term + prior_term)

            
            
            // double term_plus_inv_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // for (size_t ii=0; ii<ftp->nparams_per_core[core]*ftp->nparams_per_core[core]; ii++){
            //     term_plus_inv_cov[ii] = gradT_x_noise_x_grad[ii] + inv_core_prior_cov[ii];
            // }

            // double post_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // pinv(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
            //     term_plus_inv_cov, post_cov, 0.0);

            // post_cov = np.linalg.pinv(term + inv_cov)



            // double u[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // double s[ftp->nparams_per_core[core]];
            // double vt[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // svd(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
            //         inv_mean, u, s, vt);


            // double s_mat[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
            //     for (size_t jj=0; jj<ftp->nparams_per_core[core]; jj++){
            //         if (ii==jj){
            //             s_mat[jj*ftp->nparams_per_core[core] + ii] = sqrt(s[ii]);
            //         } else {
            //             s_mat[jj*ftp->nparams_per_core[core] + ii] = 0.0;
            //         }
            //     }
            // }
            
            double sqrtcov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // THIS WAS PREVIOUSLY U, BUT NOW WE NEED TO TRANSPOSE V
            cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, vt_, ftp->nparams_per_core[core], smat_sqrt, ftp->nparams_per_core[core], 0.0, sqrtcov, ftp->nparams_per_core[core]);
            free(smat_sqrt);

            // u, s, v = np.linalg.svd(post_cov)
            // sqrtcov = np.dot(u, np.sqrt(np.diag(s)))

            double rand_arr[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                rand_arr[ii] = randn();
            }

            double bef_core_sample[ftp->nparams_per_core[core]];
            cblas_dgemv(CblasColMajor,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, sqrtcov, ftp->nparams_per_core[core], rand_arr, 1, 0.0, bef_core_sample,1);

            double core_sample[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                core_sample[ii] = bef_core_sample[ii] + post_mean[ii];
            }
            // core_sample = post_mean + np.dot(sqrtcov, np.random.randn(core_nparams))

            // Update Core

            ft_param_update_core_params(ftp,core,core_sample);

            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                out[iter*(ftp->nparams+2) + running_nparams + ii + 2] = core_sample[ii];
            }

            // Calculate running left 

            running_nparams += ftp->nparams_per_core[core];


            // forward sweep
            if ((core > 0) && (core < ftp->dim-1)){
                for (size_t zz = 0; zz < N; zz++){
                    process_sweep_left_right_lin(ftp, core, mem->lin_structure_vals + zz * ftp->nparams,
                                                    mem->running_lr[core-1][zz], mem->running_lr[core][zz]);
                }
            }
            else if (core == 0){
                for (size_t zz = 0; zz < N; zz++){
                    process_sweep_left_right_lin(ftp, core, mem->lin_structure_vals + zz * ftp->nparams,
                                                    NULL, mem->running_lr[core][zz]);
                }
            }
            
        }


        // backward sweep
        for (size_t zz = 0; zz < N; zz++){
            process_sweep_right_left_lin(ftp, ftp->dim-1, mem->lin_structure_vals + zz * ftp->nparams,
                                            NULL, mem->running_rl[ftp->dim-1][zz]);
        }
            
        for (size_t jj = 1; jj < ftp->dim-1; jj++){
            size_t ii = ftp->dim-1-jj;

            for (size_t zz = 0; zz < N; zz++){
                process_sweep_right_left_lin(ftp,ii,mem->lin_structure_vals + zz * ftp->nparams,
                                                mem->running_rl[ii+1][zz],mem->running_rl[ii][zz]);
            }   
        }

    }
    
    sl_mem_manager_free(mem); mem = NULL;
    free(inv_noise_cov);
}


/***********************************************************//**
    Run Gibbs Sampler using Linear Parameterization of FT,
      sample noise precision, and use a hierarchical prior 
      variance along each core

    \param[in,out] ftp          - parameterized function train
    \param[in]     N            - number of data points
    \param[in]     x            - training samples
    \param[in]     y            - training labels
    \param[in]     init_sample  - original sample to start at
    \param[in]     prior_alphas - prior alpha list for prior variance gamma distribution
    \param[in]     prior_thetas - prior theta list 
    \param[in]     noise_alpha  - noise precision prior gamma distribution - alpha
    \param[in]     noise_theta  - noise precision prior gamma distribution - theta
    \param[in]     Nsamples     - number of samples requested
    \param[in,out] out          - space to put samples in
    \param[in,out] prior_out    - space to put variance samples in

***************************************************************/
void sample_hier_ind_gibbs_linear_noise(struct FTparam * ftp, size_t N, double * x, double * y, 
        double * init_sample, double * prior_alphas, double * prior_thetas, int noise_alpha, 
        double noise_theta, size_t Nsamples, double * out, double* prior_out)
{
    // CREATE A GAMMA DISTRIBUTION OVER THE PRIOR VARIANCE 
    // POSSIBLE CREATE INDIVIDUAL GAMMA DISTRIBUTIONS OVER EACH DIMENSION

    struct SLMemManager * mem = sl_mem_manager_alloc(ftp->dim,N,ftp->nparams,LINEAR_ST);
    sl_mem_manager_check_structure(mem,ftp,x);

    // Initialize with Initial Sample
    ft_param_update_params(ftp, init_sample+1);
    for (size_t ii=0; ii<ftp->nparams+1; ii++){
        out[ii] = init_sample[ii];
    }


    // init forward sweep
    for (size_t ii = 0; ii < N; ii++){
        process_sweep_right_left_lin(ftp, ftp->dim-1, mem->lin_structure_vals + ii * ftp->nparams,
                                        NULL, mem->running_rl[ftp->dim-1][ii]);
    }
    
    
    for (size_t zz = ftp->dim-2; zz > 0; zz--){
        for (size_t ii = 0; ii < N; ii++){
            process_sweep_right_left_lin(ftp,zz,mem->lin_structure_vals + ii * ftp->nparams,
                                            mem->running_rl[zz+1][ii],mem->running_rl[zz][ii]);
        }   
    }

    // Calculate Noise alpha
    // new_alpha = self.alpha + 0.5*self.Xdata.shape[0]
    int new_n_alpha = noise_alpha + round(0.5*N);

    double * inv_noise_cov = malloc(N*N * sizeof(double));
    for (size_t ii=0; ii<N; ii++){
        for (size_t jj=0; jj<N; jj++){
            inv_noise_cov[N*jj +ii] = 0.0;
        }
    }

    // Begin iterations
    for (size_t iter = 1; iter < Nsamples; iter++){

        // Sample Noise
        double ft_evals[N];
        const double * x_in = (const double *) x;
        function_train_evals(ftp->ft, N, x_in, ft_evals);

        double residual = 0;
        for (size_t ii=0; ii<N; ii++){
            residual += (y[ii] - ft_evals[ii])*(y[ii] - ft_evals[ii]);
        }
        double new_n_beta = (1/noise_theta) + 0.5*residual;

        // Requires alpha >= 1
        double noise_sample = log(randu());
        for (int ii=1; ii<new_n_alpha; ii++){
            noise_sample += log(randu());
        }
        noise_sample = (-1/new_n_beta)*noise_sample;

        out[iter*(ftp->nparams+1)] = noise_sample;

        // Fill in identity
        for (size_t ii=0; ii<N; ii++){
            inv_noise_cov[N*ii + ii] = noise_sample;
        }


        // Loop through cores
        size_t running_nparams = 0;
        for (size_t core = 0; core < ftp->dim; core++){

            if (core > 0 && core < ftp->dim-1){
                for (size_t ii = 0; ii < N; ii++){
                    ft_param_core_gradeval_lin(ftp,core, mem->grad->vals + ii * ftp->nparams_per_core[core],
                                                        mem->running_lr[core-1][ii],mem->running_rl[core+1][ii],
                                                        mem->lin_structure_vals + ii * ftp->nparams);
                }
            } else if (core == 0){
                for (size_t ii = 0; ii < N; ii++){
                    ft_param_core_gradeval_lin(ftp,core, mem->grad->vals + ii * ftp->nparams_per_core[core],
                                                        NULL,mem->running_rl[core+1][ii],
                                                        mem->lin_structure_vals + ii * ftp->nparams);
                }
            } else {
                for (size_t ii = 0; ii < N; ii++){
                    ft_param_core_gradeval_lin(ftp,core, mem->grad->vals + ii * ftp->nparams_per_core[core],
                                                        mem->running_lr[core-1][ii],NULL,
                                                        mem->lin_structure_vals + ii * ftp->nparams);
                }
            }

            // Calculate prior alpha
            int new_p_alpha = round(prior_alphas[core] + 0.5*ftp->nparams_per_core[core]); //round(0.5*ftp->nparams_per_core[core]);

            // Sample Prior Variance

            // ONLY FOR THIS CORE OF PARAMETERS
            size_t runparam = 0;
            for (size_t ii = 0; ii < core; ii ++){
                runparam += ftp->nparams_per_core[ii];
            }

            double squared_sum = 0;
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                squared_sum += ftp->params[runparam + ii]*ftp->params[runparam + ii];
            }

            double new_p_beta = (1/prior_thetas[core]) + 0.5*squared_sum;

            double var_sample = log(randu());
            for (int ii=1; ii<new_p_alpha; ii++){
                var_sample += log(randu());
            }
            var_sample = -(1/new_p_beta)*var_sample;

            prior_out[(iter-1)*ftp->dim + core] = var_sample;


            // Create Inverse Prior Covariance Matrix
            double inv_core_prior_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++) {
                for (size_t jj=0; jj<ftp->nparams_per_core[core]; jj++) {
                    if (ii==jj){
                        inv_core_prior_cov[ftp->nparams_per_core[core]*jj + ii] = var_sample; // CHECK THIS
                    } else {
                        inv_core_prior_cov[ftp->nparams_per_core[core]*jj + ii] = 0.0;
                    }
                }
            }

            double core_prior_mean[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                core_prior_mean[ii] = 0;
            }


            // Sample Core
            //cblas_dgemm(Col, TransOpA, TransOpB, M, N, K, al, A, lda, B, ldb, beta, C, ldc)
            // op(A) = M x K
            // op(B) = K x N
            //    C  = M x N
            // lda = cols of A = M or K
            // ldb = cols of B = K or N
            // ldc = cols of C = M


            // temp = np.dot( self.inv_noise_cov, core_grad )
            // N x N times N x Nparams
            double inv_noise_x_grad[N*ftp->nparams_per_core[core]]; // N x nparams_core
            cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, N, ftp->nparams_per_core[core], N, 1.0, inv_noise_cov, N, mem->grad->vals, ftp->nparams_per_core[core], 0.0, inv_noise_x_grad, ftp->nparams_per_core[core]);
            

            // np.dot(core_grad.T, temp)
            // Nparams x N times N x Nparams
            double gradT_x_noise_x_grad[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]]; // nparams_core x nparams_core
            cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], N, 1.0, mem->grad->vals, ftp->nparams_per_core[core], inv_noise_x_grad, ftp->nparams_per_core[core], 0.0, gradT_x_noise_x_grad, ftp->nparams_per_core[core]);

            // term = np.dot(core_grad.T, np.dot( self.inv_noise_cov, core_grad ))
            double mean[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]*ftp->nparams_per_core[core]; ii++){
                mean[ii] = gradT_x_noise_x_grad[ii] + inv_core_prior_cov[ii];
            }

            double inv_mean[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // pinv(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
            //       mean, inv_mean, 0.0);

            size_t m = ftp->nparams_per_core[core];
            double cutoff = 0.0;
            double u_[m*m];
            double vt_[m*m];
            double s_[m];
            
            // not sure about thir dargument
            svd(m, m, m, mean, u_, s_, vt_); //note changed from m to lda
            
            double * smat = calloc_double(m*m);
            double * smat_sqrt = calloc_double(m*m);

            for (size_t ii = 0; ii < m; ii++){
                if (fabs(s_[ii]) < cutoff){ 
                    smat[ii*m+ii] = 0.0;
                    smat_sqrt[ii*m+ii] = 0.0;
                }
                else{
                    smat[ii*m+ii] = 1.0/s_[ii];
                    smat_sqrt[ii*m+ii] = sqrt(1.0/s_[ii]);
                }
            }
            
            double temp[m*m];
            
            cblas_dgemm(CblasColMajor,CblasTrans,CblasTrans, m, m, m, 1.0, 
                            vt_, m, smat, m, 0.0, temp, m);

            free(smat);
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans, m, m, m, 1.0, 
                            temp, m, u_, m, 0.0, inv_mean, m);




            
            // mean_inv = np.linalg.pinv(term + inv_cov)

            double noise_x_data[N];
            cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, inv_noise_cov, N, y, 1, 0.0, noise_x_data, 1);

            double data_term[ftp->nparams_per_core[core]];
            cblas_dgemv(CblasRowMajor, CblasTrans, N, ftp->nparams_per_core[core], 1.0, mem->grad->vals, ftp->nparams_per_core[core], noise_x_data, 1, 0.0, data_term, 1);
            
            // data_term = np.dot(core_grad.T, np.dot( self.inv_noise_cov, self.Ydata ))

            double prior_term[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            cblas_dgemv(CblasRowMajor, CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, inv_core_prior_cov, ftp->nparams_per_core[core], core_prior_mean, 1, 0.0, prior_term, 1);

            // prior_term = np.dot(inv_cov, core_prior_mean)

            double data_plus_prior[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                data_plus_prior[ii] = data_term[ii] + prior_term[ii];
            }

            double post_mean[ftp->nparams_per_core[core]];
            cblas_dgemv(CblasRowMajor,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, inv_mean, ftp->nparams_per_core[core], data_plus_prior, 1, 0.0, post_mean,1);

            // post_mean = np.dot(mean_inv, data_term + prior_term)

            
            
            // double term_plus_inv_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // for (size_t ii=0; ii<ftp->nparams_per_core[core]*ftp->nparams_per_core[core]; ii++){
            //     term_plus_inv_cov[ii] = gradT_x_noise_x_grad[ii] + inv_core_prior_cov[ii];
            // }

            // double post_cov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // pinv(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
            //     term_plus_inv_cov, post_cov, 0.0);

            // post_cov = np.linalg.pinv(term + inv_cov)

            // double u[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // double s[ftp->nparams_per_core[core]];
            // double vt[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // svd(ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 
            //         inv_mean, u, s, vt);


            // double s_mat[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            // for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
            //     for (size_t jj=0; jj<ftp->nparams_per_core[core]; jj++){
            //         if (ii==jj){
            //             s_mat[jj*ftp->nparams_per_core[core] + ii] = sqrt(s[ii]);
            //         } else {
            //             s_mat[jj*ftp->nparams_per_core[core] + ii] = 0.0;
            //         }
            //     }
            // }
            
            double sqrtcov[ftp->nparams_per_core[core]*ftp->nparams_per_core[core]];
            cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, vt_, ftp->nparams_per_core[core], smat_sqrt, ftp->nparams_per_core[core], 0.0, sqrtcov, ftp->nparams_per_core[core]);
            free(smat_sqrt);

            // u, s, v = np.linalg.svd(post_cov)
            // sqrtcov = np.dot(u, np.sqrt(np.diag(s)))

            double rand_arr[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                rand_arr[ii] = randn();
            }

            double bef_core_sample[ftp->nparams_per_core[core]];
            cblas_dgemv(CblasColMajor,CblasNoTrans, ftp->nparams_per_core[core], ftp->nparams_per_core[core], 1.0, sqrtcov, ftp->nparams_per_core[core], rand_arr, 1, 0.0, bef_core_sample,1);

            double core_sample[ftp->nparams_per_core[core]];
            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                core_sample[ii] = bef_core_sample[ii] + post_mean[ii];
            }
            // core_sample = post_mean + np.dot(sqrtcov, np.random.randn(core_nparams))

            // Update Core

            ft_param_update_core_params(ftp,core,core_sample);

            for (size_t ii=0; ii<ftp->nparams_per_core[core]; ii++){
                out[iter*(ftp->nparams+1) + running_nparams + ii + 1] = core_sample[ii];
            }

            // Calculate running left 

            running_nparams += ftp->nparams_per_core[core];


            // forward sweep
            if ((core > 0) && (core < ftp->dim-1)){
                for (size_t zz = 0; zz < N; zz++){
                    process_sweep_left_right_lin(ftp, core, mem->lin_structure_vals + zz * ftp->nparams,
                                                    mem->running_lr[core-1][zz], mem->running_lr[core][zz]);
                }
            }
            else if (core == 0){
                for (size_t zz = 0; zz < N; zz++){
                    process_sweep_left_right_lin(ftp, core, mem->lin_structure_vals + zz * ftp->nparams,
                                                    NULL, mem->running_lr[core][zz]);
                }
            }
            
        }


        // backward sweep
        for (size_t zz = 0; zz < N; zz++){
            process_sweep_right_left_lin(ftp, ftp->dim-1, mem->lin_structure_vals + zz * ftp->nparams,
                                            NULL, mem->running_rl[ftp->dim-1][zz]);
        }
            
        for (size_t jj = 1; jj < ftp->dim-1; jj++){
            size_t ii = ftp->dim-1-jj;

            for (size_t zz = 0; zz < N; zz++){
                process_sweep_right_left_lin(ftp,ii,mem->lin_structure_vals + zz * ftp->nparams,
                                                mem->running_rl[ii+1][zz],mem->running_rl[ii][zz]);
            }   
        }

    }
    
    sl_mem_manager_free(mem); mem = NULL;
    free(inv_noise_cov);
}

/***********************************************************//**
    Evaluate the gradient of the ft with respect to each parameter in a core

    \param[in,out] ftp          - parameterized FTP
    \param[in]     core         - core under consideration   
    \param[in]     N            - number of data points
    \param[in]     x            - locations at which to evaluate
    \param[in,out] grad         - gradient wrt each parameter in each univariate function
    \param[in]     calc_running - Boolean - Calculate Left and Right Evaluations
    \param[in,out] running_lr   - evaluation of cores from the left
    \param[in,out] running_rl   - evaluation of the cores from the right
    \param[in,out] running_eval - workspace (number of univariate functions)

***************************************************************/
void ft_param_core_gradevals(struct FTparam * ftp, size_t core, size_t N, double * x,
                              double * grad, int calc_running, double * running_lr, double * running_rl,
                              double * running_eval)

{
    size_t maxp;
    size_t nparam = qmarray_get_nparams(ftp->ft->cores[core],&maxp);
    size_t dim = ftp->dim;
    assert(ftp->ft->ranks[0] == 1);
    assert(ftp->ft->ranks[dim] == 1);

    size_t left_rank = ftp->ft->ranks[core];
    size_t right_rank = ftp->ft->ranks[core+1];

    if (calc_running != 0){
        struct SLMemManager * slm = sl_mem_manager_alloc(dim,N,ftp->nparams,LINEAR_ST);
        sl_mem_manager_check_structure(slm,ftp,x);
        
        for (size_t ii = 0; ii < N; ii++){
            process_sweep_right_left(ftp,dim-1,x[ii*dim + dim-1],slm->running_eval,
                                    NULL,slm->running_rl[dim-1][ii]);
        }
        for (size_t zz = ftp->dim-2; zz > core; zz--){
            for (size_t ii = 0; ii < N; ii++){
                process_sweep_right_left(ftp,zz,x[ii*dim + zz],slm->running_eval,
                                        slm->running_rl[zz+1][ii],slm->running_rl[zz][ii]);
            }   
        }

        for (size_t ii = 0; ii < N; ii++){
            process_sweep_left_right(ftp,0,x[ii*dim],slm->running_eval,NULL,slm->running_lr[0][ii]);
        }
        for (size_t zz = 1; zz < core; zz++){
            for (size_t ii = 0; ii < N; ii++){
                process_sweep_left_right(ftp,zz,x[ii*dim + zz],slm->running_eval,slm->running_lr[zz-1][ii],
                                        slm->running_lr[zz][ii]);
            }
        }

        
        for (size_t ii=0; ii<N; ii++){
            if (core > 0 && core < dim-1){
                ft_param_core_gradeval(ftp,core,x[core + ii*dim], grad + ii*nparam,
                                                        slm->running_lr[core-1][ii], slm->running_rl[core+1][ii],
                                                        slm->running_eval);
            } else if (core == dim-1) {
                ft_param_core_gradeval(ftp,core,x[core + ii*dim], grad + ii*nparam,
                                                        slm->running_lr[core-1][ii], NULL,
                                                        slm->running_eval);
            } else if (core == 0) {
                ft_param_core_gradeval(ftp,core,x[core + ii*dim], grad + ii*nparam,
                                                        NULL, slm->running_rl[core+1][ii],
                                                        slm->running_eval);
            }
        }

        sl_mem_manager_free(slm);
        slm  = NULL;
    } else {
        if (core > 0 && core < dim-1){
            for (size_t ii=0; ii<N; ii++){
                ft_param_core_gradeval(ftp,core,x[core + ii*dim], grad + ii*nparam,
                                                        running_lr + left_rank*ii, running_rl + right_rank*ii,
                                                        running_eval);
            }
        } else if (core == dim-1) {
            for (size_t ii=0; ii<N; ii++){
                ft_param_core_gradeval(ftp,core,x[core + ii*dim], grad + ii*nparam,
                                                        running_lr + left_rank*ii, NULL,
                                                        running_eval);
            }
        } else if (core == 0) {
            for (size_t ii=0; ii<N; ii++){
                ft_param_core_gradeval(ftp,core,x[core + ii*dim], grad + ii*nparam,
                                                        NULL, running_rl + right_rank*ii,
                                                        running_eval);
            }
        }
    }
}

/***********************************************************//**
    Evaluate the gradient of the ft with respect to each parameter in a core

    \param[in,out] ftp        - parameterized FTP
    \param[in]     core       - core under consideration   
    \param[in]     x          - location at which to evaluate
    \param[in,out] grad       - gradient wrt each parameter in each univariate function
    \param[in,out] running_lr - evaluation of cores from the left
    \param[in,out] running_rl - evaluation of the cores from the right
    \param[in,out] grad_evals - workspace (number of univariate functions)

    \return evaluation (also grad is the key item here)
***************************************************************/
double ft_param_core_gradeval(struct FTparam * ftp, size_t core, double x,
                              double * grad,  double * running_lr,double * running_rl,
                              double * grad_evals)

{

    size_t onparam = 0,nparam;

    struct FunctionTrain * ft = ftp->ft;
    size_t * ranks = function_train_get_ranks(ft);

    size_t r1 = ranks[core];
    size_t r2 = ranks[core+1];
    double out = 0.0, eval;
    
    if ((core > 0) && (core < ftp->dim-1)){
        for (size_t ii = 0; ii < r2; ii++){
            double right_mult = running_rl[ii];
            for (size_t kk = 0; kk < r1; kk++){
                double left_mult = running_lr[kk];
                struct Qmarray * qma = ft->cores[core];
                // compute grad for univariate function
                eval = generic_function_param_grad_eval2(qma->funcs[kk + ii * r1],x,grad_evals);
                nparam = generic_function_get_num_params(qma->funcs[kk + ii * r1]);
                for (size_t ll = 0; ll < nparam ;ll++){
                    //there are no shared parameters between univariate functions
                    grad[onparam] =  left_mult*grad_evals[ll]*right_mult; 
                    onparam++;
                }
                out += left_mult* right_mult * eval;
            }
        }
    }
    else if (core == 0){
        for (size_t ii = 0; ii < r2; ii++){
            double right_mult = running_rl[ii];
            struct Qmarray * qma = ft->cores[core];
            // compute grad for univariate function
            eval = generic_function_param_grad_eval2(qma->funcs[ii],x,grad_evals);
            nparam = generic_function_get_num_params(qma->funcs[ii]);
            for (size_t ll = 0; ll < nparam ;ll++){
                //there are no shared parameters between univariate functions
                grad[onparam] = grad_evals[ll]*right_mult; 
                onparam++;
            }
            out += right_mult * eval;
        }
    }
    else{/* if (core == (ftp->dim-1)){ */
        for (size_t kk = 0; kk < r1; kk++){
            double left_mult = running_lr[kk];
            struct Qmarray * qma = ft->cores[core];
            // compute grad for univariate function
            eval = generic_function_param_grad_eval2(qma->funcs[kk],x,grad_evals);
            nparam = generic_function_get_num_params(qma->funcs[kk]);
            for (size_t ll = 0; ll < nparam ;ll++){
                //there are no shared parameters between univariate functions
                grad[onparam] =  left_mult*grad_evals[ll]; 
                onparam++;
            }
            out += left_mult* eval;
        }
    }

    return out;
}

/***********************************************************//**
    Evaluate the ft 
    for a function with unviariate functions that are parameterized linearly

    \param[in,out] ftp        - parameterized FTP
    \param[in]     core       - core under consideration   
    \param[in,out] running_lr - evaluation of cores from the left
    \param[in,out] running_rl - evaluation of the cores from the right
    \param[in]     grad_evals - gradient wrt each parameter of each 
                                univariate function in the ft

    \return evaluation
***************************************************************/
double ft_param_core_eval_lin(struct FTparam * ftp, size_t core,
                              double * running_lr, double * running_rl,
                              const double * grad_evals)
                              
{

    /* printf("ft_param_eval_lin\n"); */
    size_t onuni = 0;
    size_t onparam = 0;
    size_t * ranks = function_train_get_ranks(ftp->ft);

    double out = 0.0;
    for (size_t kk = 0; kk < core; kk++){
        for (size_t ii = 0; ii < ranks[kk] * ranks[kk+1]; ii++){
            onparam += ftp->nparams_per_uni[onuni];
            onuni++;
        }
    }
    
    if ((core > 0) && (core < ftp->dim-1)){
        double t;
        for (size_t col = 0; col < ranks[core+1]; col++){
            for (size_t row = 0; row < ranks[core]; row++){
                t = cblas_ddot(ftp->nparams_per_uni[onuni],
                               grad_evals+onparam,1,
                               ftp->params + onparam,1);
                t *= running_lr[row] * running_rl[col];
                out += t;
                onparam += ftp->nparams_per_uni[onuni];
                onuni++;
            }
        }
    }
    else if (core == ftp->dim-1){
        for (size_t row = 0; row < ranks[ftp->dim-1]; row++){
            out += cblas_ddot(ftp->nparams_per_uni[onuni],
                              grad_evals+onparam,1,
                              ftp->params + onparam,1) * running_lr[row];

            onparam += ftp->nparams_per_uni[onuni];
            onuni++;
        }
    }
    else{ // core == 0
        for (size_t col = 0; col < ranks[1]; col++){
            out += cblas_ddot(ftp->nparams_per_uni[onuni],
                              grad_evals+onparam,1,
                              ftp->params + onparam,1) *
                running_rl[col];
            onparam += ftp->nparams_per_uni[onuni];
            onuni++;
        }
    }

    return out;
}


/***********************************************************//**
    Evaluate the gradient of a function with respect to the parameters
    of a particular core for a function with unviariate functions that are parameterized linearly

    \param[in,out] ftp        - parameterized FTP
    \param[in]     core       - core under consideration   
    \param[in,out] grad       - gradient wrt each parameter in the core
    \param[in,out] running_lr - evaluation of cores from the left
    \param[in,out] running_rl - evaluation of the cores from the right
    \param[in]     grad_evals - gradient wrt each parameter of each univariate function in the ft

    \return evaluation
***************************************************************/
double ft_param_core_gradeval_lin(struct FTparam * ftp, size_t core,
                                  double * grad,  double * running_lr,double * running_rl,
                                  double * grad_evals)

{

    /* printf("ft_param_eval_lin\n"); */
    size_t onuni = 0;
    size_t onparam = 0;
    size_t * ranks = function_train_get_ranks(ftp->ft);
    size_t ongrad = 0;
    double out = 0.0;
    for (size_t kk = 0; kk < core; kk++){
        onparam += ftp->nparams_per_core[kk];
        onuni += ranks[kk]*ranks[kk+1];
    }
    
    if ((core > 0) && (core < ftp->dim-1)){
        double t;
        for (size_t col = 0; col < ranks[core+1]; col++){
            for (size_t row = 0; row < ranks[core]; row++){
                t = cblas_ddot(ftp->nparams_per_uni[onuni],
                               grad_evals+onparam,1,
                               ftp->params + onparam,1);
                t *= running_lr[row] * running_rl[col];
                out += t;

                double t2 = running_lr[row] * running_rl[col];
                for (size_t ll = 0; ll < ftp->nparams_per_uni[onuni]; ll++){
                    //there are no shared parameters between univariate functions
                    grad[ongrad] =  t2 * grad_evals[onparam];
                    onparam++;
                    ongrad++;
                }
                
                onuni++;
            }
        }
    }
    else if (core == ftp->dim-1){
        for (size_t row = 0; row < ranks[ftp->dim-1]; row++){
            out += cblas_ddot(ftp->nparams_per_uni[onuni],
                              grad_evals+onparam,1,
                              ftp->params + onparam,1) * running_lr[row];

            for (size_t ll = 0; ll < ftp->nparams_per_uni[onuni];ll++){
                //there are no shared parameters between univariate functions
                grad[ongrad] =  running_lr[row] * grad_evals[onparam];
                onparam++;
                ongrad++;
            }
            onuni++;
        }
    }
    else{ // core == 0
        for (size_t col = 0; col < ranks[1]; col++){
            out += cblas_ddot(ftp->nparams_per_uni[onuni],
                              grad_evals+onparam,1,
                              ftp->params + onparam,1) * running_rl[col];


            for (size_t ll = 0; ll < ftp->nparams_per_uni[onuni]; ll++){
                //there are no shared parameters between univariate functions
                grad[ongrad] =  running_rl[col] * grad_evals[onparam];
                onparam++;
                ongrad++;
            }
            onuni++;
        }
    }

    return out;
}

static void vec_mat(size_t N, size_t M, const double * vec, const double * mat, double * out)
{
    // vec is 1 x N
    // mat is N x M

    for (size_t jj = 0; jj < M; jj++){
        out[jj] = 0.0;
        for (size_t ii = 0; ii < N; ii++){
            out[jj] += vec[ii]*mat[ii + jj*N];
        }
    }
    
}

static double hess_ij(size_t dim,
                      size_t core_ii,size_t core_jj,
                      size_t row_ii, size_t col_ii, size_t row_jj, size_t col_jj,
                      double * evals, double grad_ii, double grad_jj,
                      size_t maxrank, size_t * ranks)
{
    assert (core_jj > core_ii);
    double * left_eval = calloc_double(maxrank);
    double * space = calloc_double(maxrank);
    size_t num_vals = 0;
    size_t on_eval = 0;
    if (core_ii > 0){
        for (size_t ii = 0; ii < ranks[1]; ii++){
            left_eval[ii] = evals[ii];
        }
        num_vals = ranks[1];
        on_eval+= ranks[1];
        for (size_t core = 1; core < core_ii; core++){
            vec_mat(ranks[core],ranks[core+1],left_eval,evals+on_eval,space);
            num_vals = ranks[core+1];
            memmove(left_eval,space,num_vals * sizeof(double));
            on_eval += ranks[core]*ranks[core+1];
        }

        double val = left_eval[row_ii];
        for (size_t ii = 0; ii < ranks[core_ii+1]; ii++){
            if (ii != col_ii){
                left_eval[ii] = 0.0;
            }
            else{
                left_eval[ii] = val * grad_ii;/* evals[on_eval + row_ii + col_ii*ranks[core_ii]]; */
            }
        }
        on_eval += ranks[core_ii]*ranks[core_ii+1];
        num_vals = ranks[core_ii+1];
    }
    else{
        left_eval[col_ii] = grad_ii;
        on_eval  = ranks[1];
        num_vals = ranks[1];
    }

    // Now middle cores
    // should really do core immediately after to take advantage of sparsity
    for (size_t core = core_ii+1; core < core_jj; core++){
        vec_mat(ranks[core],ranks[core+1],left_eval,evals+on_eval,space);
        num_vals = ranks[core+1];
        memmove(left_eval,space,num_vals * sizeof(double));
        on_eval += ranks[core]*ranks[core+1];        
    }

    /* printf("core_ii = %zu, row_ii = %zu, col_ii = %zu ",core_ii,row_ii,col_ii); */
    /* dprint(num_vals,left_eval); */
    
    // now multiply by core with gradient
    double val = left_eval[row_jj] * grad_jj; /* evals[on_eval + row_jj + col_jj*ranks[core_jj]]; */
    for (size_t ii = 0; ii < ranks[core_jj+1]; ii++)
    {
        if (ii != col_jj){
            left_eval[ii] = 0.0;
        }
        else{
            left_eval[ii] = val;
        }
    }
    num_vals = ranks[core_jj+1];
    on_eval += ranks[core_jj]*ranks[core_jj+1];

    // Now do the last cores
    for (size_t core = core_jj+1; core < dim; core++){
        vec_mat(ranks[core],ranks[core+1],left_eval,evals+on_eval,space);
        num_vals = ranks[core+1];
        memmove(left_eval,space,num_vals * sizeof(double));
        on_eval += ranks[core]*ranks[core+1];        
    }


    double ret = left_eval[0];
    free(left_eval); left_eval = NULL;
    free(space); space = NULL;
    return ret;
}

/***********************************************************//**
    Evaluate the hessian times vector

    \param[in,out] ftp      - parameterized FTP
    \param[in]     x        - location at which to evaluate
    \param[in]     vec      - vector by which to right-multiply the hessian
    \param[in,out] hess_vec - final result, must be allocated on entrance

    \return evaluation

    \note 
    This algorithm may not be fully optimized to take advantage of
    all available structure. It needs to be reviewed
***************************************************************/
double ft_param_hessvec(struct FTparam * ftp, const double * x,
                        const double * vec,
                        double * hess_vec)
{


    struct FunctionTrain * ft = ftp->ft;
    size_t * ranks = function_train_get_ranks(ft);
    size_t maxrank = ranks[0];
    for (size_t ii = 0; ii < ft->dim; ii++){
        if (ranks[ii] > maxrank){
            maxrank = ranks[ii];
        }
    }

    size_t r1,r2;
    size_t onuni = 0;
    size_t onparam = 0;
    size_t onparam_hess = 0;
    double * evals = calloc_double(ftp->nparams);
    double * d1_evals = calloc_double(ftp->nparams);
    /* double * d2_evals = calloc_double(maxrank*maxrank*ftp->nparams); */


    double eval = 0.0;

    // Precompute all relavent info
    for (size_t ii = 0; ii < ftp->dim; ii++){
        r1 = ranks[ii];
        r2 = ranks[ii+1];
        for (size_t jj = 0; jj < r1*r2; jj++){
            evals[onuni] = generic_function_param_grad_eval2(ft->cores[ii]->funcs[jj],x[ii],d1_evals + onparam);
            /* generic_function_param_hess(ft->cores[ii]->funcs[jj],x[ii],d2_evals + onparam_hess); */

            onparam_hess += (ftp->nparams_per_uni[onuni] * ftp->nparams_per_uni[onuni]);
            onparam += ftp->nparams_per_uni[onuni];
            onuni++;
        }
    }

    // Assemble
    for (size_t ii = 0; ii < ftp->nparams; ii++){
        hess_vec[ii] = 0.0;
    }
    

    double h;
    size_t param_ii = 0;
    size_t onuni_ii = 0;
    size_t start_params_col = 0;
    size_t start_onuni_jj = 0;
    for (size_t core_ii = 0; core_ii < ftp->dim; core_ii++){
        start_params_col += ftp->nparams_per_core[core_ii];
        start_onuni_jj += ranks[core_ii]*ranks[core_ii+1];
        /* printf("core_ii = %zu\n",core_ii); */
        for (size_t col_ii = 0; col_ii < ranks[core_ii+1]; col_ii++){
            /* printf("\t col_ii = %zu\n",col_ii); */
            for (size_t row_ii = 0; row_ii < ranks[core_ii]; row_ii++){ // end of indices describing row of hessian
                /* printf("\t\t row_ii = %zu\n",row_ii); */
                /* printf("\t\t onuni_ii = %zu\n",onuni_ii); */
                for (size_t param = 0; param < ftp->nparams_per_uni[onuni_ii]; param++){
                    /* printf("\t\t\t param = %zu, d1_evals = %G\n",param,d1_evals[param_ii]); */
                    // handle diagonal *element* rest of core/core derivatives are zero
                    // TODO -- compute diagonal components

                    // handle the rest
                    size_t param_jj = start_params_col;
                    size_t onuni_jj = start_onuni_jj;
                    /* printf("starting onuni_jj = %zu\n",onuni_jj); */
                    for (size_t core_jj = core_ii+1; core_jj < ftp->dim; core_jj++){
                        /* printf("core_jj = %zu\n",core_jj); */
                        for (size_t col_jj = 0; col_jj < ranks[core_jj+1]; col_jj++){
                            /* printf("\t col_jj = %zu\n",col_jj); */
                            for (size_t row_jj = 0; row_jj < ranks[core_jj]; row_jj++){
                                /* printf("\t\t row_jj = %zu, on_uni_jj = %zu, params = %zu\n",row_jj,onuni_jj,ftp->nparams_per_uni[onuni_jj]); */

                                // TODO - Leverage Block structure
                                /* if (col_ii == row_jj){ */
                                    for (size_t param_col = 0; param_col < ftp->nparams_per_uni[onuni_jj]; param_col++){
                                        h = hess_ij(ftp->dim,
                                                    core_ii, core_jj, row_ii, col_ii, row_jj,col_jj,
                                                    evals,d1_evals[param_ii],d1_evals[param_jj],maxrank,ranks);

                                    
                                        hess_vec[param_ii] += h * vec[param_jj];
                                        /* printf("param_jj = %zu\n",param_jj); */
                                        hess_vec[param_jj] += h * vec[param_ii];
                                        param_jj++;
                                    }
                                /* } */
                                /* else{ */
                                /*     param_jj += ftp->nparams_per_uni[onuni_jj]; */
                                /* } */
                                
                                onuni_jj++;
                            }
                        }
                    }

                    param_ii++;
                }
                onuni_ii++;
            }
        }
    }

    free(evals); evals = NULL;
    free(d1_evals); d1_evals = NULL;
    return eval;
}
