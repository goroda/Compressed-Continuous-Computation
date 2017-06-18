// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016-2017 Sandia Corporation

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

#include "ft.h"
#include "lib_linalg.h"

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

/***********************************************************//**
    Create a parameterization from a linear least squares fit to 
    x and y

    \param[in,out] ftp     - parameterized FTP
    \param[in]     N       - number of data points
    \param[in]     x       - features
    \param[in]     y       - labels
    \param[in]     perturb - perturbation to zero elements

    \note
    If ranks are < 2 then performs a constant fit at the mean of the data
    Else creates top 2x2 blocks to be a linear least squares fit and sets everything
    else to 1e-12
***************************************************************/
void ft_param_create_from_lin_ls(struct FTparam * ftp, size_t N,
                                 const double * x, const double * y,
                                 double perturb)
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

static void update_running_rl(struct FTparam * ftp, size_t core, double * running_rl, double * running_rl_up, double * evals)
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

