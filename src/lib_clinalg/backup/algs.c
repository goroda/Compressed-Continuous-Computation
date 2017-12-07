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





/** \file algs.c
 * Provides routines for performing continuous linear algebra and 
 * working with function trains
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>

#include "qmarray_qr.h"
#include "lib_funcs.h"
#include "indmanage.h"
#include "algs.h"
#include "array.h"
#include "linalg.h"

#define ZEROTHRESH 1e2*DBL_EPSILON
//#define ZEROTHRESH 0.0
//#define ZEROTHRESH  1e-14
//#define ZEROTHRESH 1e-20S

#ifndef VPREPCORE
    #define VPREPCORE 0
#endif

#ifndef VFTCROSS
    #define VFTCROSS 0
#endif


////////////////////////////////////////////////////////////////////////////
// function_train



// utility function for function_train_cross (not in header file)
struct Qmarray *
prepCore(size_t ii, size_t nrows, 
         double(*f)(double *, void *), void * args,
         struct BoundingBox * bd,
         struct CrossIndex ** left_ind,struct CrossIndex ** right_ind, 
         struct FtCrossArgs * cargs, struct FtApproxArgs * fta, int t)
{

    enum function_class fc;
    void * sub_type = NULL;
    void * approx_args = NULL;
    struct FiberCut ** fcut = NULL;  
    struct Qmarray * temp = NULL;
    double ** vals = NULL;
    size_t ncols, ncuts;
    size_t dim = bd->dim;
    
    fc = ft_approx_args_getfc(fta,ii);
    sub_type = ft_approx_args_getst(fta,ii);
    approx_args = ft_approx_args_getaopts(fta, ii);
    //printf("sub_type prep_core= %d\n",*(int *)ft_approx_args_getst(fta,ii));
    //printf("t=%d\n",t);
    if (t == 1){
        ncuts = nrows;
        ncols = 1;
        /* printf("left index of merge = \n"); */
        /* print_cross_index(left_ind[ii]); */
        /* printf("right index of merge = \n"); */
        /* print_cross_index(right_ind[ii]); */
//        vals = cross_index_merge(left_ind[ii],right_ind[ii-1]);
        vals = cross_index_merge_wspace(left_ind[ii],NULL);
        fcut = fiber_cut_ndarray(f,args, dim, ii, ncuts, vals);
    }
    else if (t == -1){
        ncuts = cargs->ranks[ii+1];
        ncols = ncuts;
//        vals = cross_index_merge(left_ind[ii+1],right_ind[ii]);
        vals = cross_index_merge_wspace(NULL,right_ind[ii]);
        fcut = fiber_cut_ndarray(f,args, dim, ii, ncuts, vals);
    }
    else{
        if (left_ind[ii] == NULL){
            ncuts = right_ind[ii]->n;
        }
        else if (right_ind[ii] == NULL){
            ncuts = left_ind[ii]->n;
        }
        else{
            ncuts = left_ind[ii]->n * right_ind[ii]->n;
        }
        vals = cross_index_merge_wspace(left_ind[ii],right_ind[ii]);
        fcut = fiber_cut_ndarray(f,args, dim, ii, ncuts, vals);
        ncols = ncuts / nrows;
    }
    if (VPREPCORE){
        printf("compute from fibercuts t = %d\n",t);
        for (size_t kk = 0; kk < nrows*ncols; kk++){
            dprint(dim,vals[kk]);
        }
    }
    temp = qmarray_from_fiber_cuts(nrows, ncols,
                    fiber_cut_eval, fcut, fc, sub_type,
                    bd->lb[ii],bd->ub[ii], approx_args);

    //print_qmarray(temp,0,NULL);
    if (VPREPCORE){
        printf("computed!\n");
    }

    free_dd(ncuts,vals); vals = NULL;
    fiber_cut_array_free(ncuts, fcut); fcut = NULL;
    
    
    return temp;
}



/***********************************************************//**
    Cross approximation of a of a dim-dimensional function
    (with a fixed grid)

    \param[in]     f         - function
    \param[in]     args      - function arguments
    \param[in]     bd        -  bounds on input space
    \param[in,out] ftref     - initial ftrain, changed in func
    \param[in,out] left_ind  - left indices(first element should be NULL)
    \param[in,out] right_ind - right indices(last element should be NULL)
    \param[in]     cargs     - algorithm parameters
    \param[in]     apargs    - approximation arguments

    \return function train decomposition of \f$ f \f$

    \note
    both left and right indices are nested
***************************************************************/
struct FunctionTrain *
ftapprox_cross_grid(double (*f)(double *, void *), void * args,
                    struct BoundingBox * bd,
                    struct FunctionTrain * ftref,
                    struct CrossIndex ** left_ind,
                    struct CrossIndex ** right_ind,
                    struct FtCrossArgs * cargs,
                    struct FtApproxArgs * apargs,
                    struct c3Vector ** grid
    )
{
    size_t dim = bd->dim;
    int info;
    size_t nrows, ii, oncore;
    struct Qmarray * temp = NULL;
    struct Qmarray * Q = NULL;
    struct Qmarray * Qt = NULL;
    double * R = NULL;
    size_t * pivind = NULL;
    double * pivx = NULL;
    double diff, diff2, den;
    
    struct FunctionTrain * ft = function_train_alloc(dim);
    memmove(ft->ranks, cargs->ranks, (dim+1)*sizeof(size_t));

    struct FunctionTrain * fti = function_train_copy(ftref);
    struct FunctionTrain * fti2 = NULL;

    int done = 0;
    size_t iter = 0;

    while (done == 0){
        if (cargs->verbose > 0)
            printf("cross iter=%zu \n",iter);
      
        // left right sweep;
        nrows = 1;
        for (ii = 0; ii < dim-1; ii++){
            if (cargs->verbose > 1){
                printf(" ............. on left-right sweep (%zu/%zu)\n",ii,dim-1);
            }
            //printf("ii=%zu\n",ii);
            pivind = calloc_size_t(ft->ranks[ii+1]);
            pivx = calloc_double(ft->ranks[ii+1]);
            
            if (VFTCROSS){
                printf( "prepCore \n");
                printf( "left index set = \n");
                print_cross_index(left_ind[ii]);
                printf( "right index set = \n");
                print_cross_index(right_ind[ii]);
            }
            temp = prepCore(ii,nrows,f,args,bd,left_ind,right_ind,
                            cargs,apargs,0);

            if (VFTCROSS == 2){
                printf ("got it \n");
                //print_qmarray(temp,0,NULL);
                struct Qmarray * tempp = qmarray_copy(temp);
                printf("core is \n");
                //print_qmarray(tempp,0,NULL);
                R = calloc_double(temp->ncols * temp->ncols);
                Q = qmarray_householder_simple("QR", temp,R);
                printf("R=\n");
                dprint2d_col(temp->ncols, temp->ncols, R);
//                print_qmarray(Q,0,NULL);

                struct Qmarray * mult = qmam(Q,R,temp->ncols);
                //print_qmarray(Q,3,NULL);
                double difftemp = qmarray_norm2diff(mult,tempp);
                printf("difftemp = %3.15G\n",difftemp);
                qmarray_free(tempp);
                qmarray_free(mult);
            }
            else{
                R = calloc_double(temp->ncols * temp->ncols);
                /* Q = qmarray_householder_simple("QR", temp,R); */
                Q = qmarray_householder_simple_grid("QR",temp,R,grid[ii]);
            }

            info = qmarray_maxvol1d(Q,R,pivind,pivx,grid[ii]);

            if (VFTCROSS){
                printf( " got info=%d\n",info);
                printf("indices and pivots\n");
                iprint_sz(ft->ranks[ii+1],pivind);
                dprint(ft->ranks[ii+1],pivx);
            }

            if (info < 0){
                fprintf(stderr, "no invertible submatrix in maxvol in cross\n");
            }
            if (info > 0){
                fprintf(stderr, " error in qmarray_maxvol1d \n");
                exit(1);
            }

            cross_index_free(left_ind[ii+1]);
            if (ii > 0){
                left_ind[ii+1] =
                    cross_index_create_nested_ind(0,ft->ranks[ii+1],pivind,
                                                  pivx,left_ind[ii]);
            }
            else{
                left_ind[ii+1] = cross_index_alloc(1);
                for (size_t zz = 0; zz < ft->ranks[1]; zz++){
                    cross_index_add_index(left_ind[1],1,&(pivx[zz]));
                }
            }
            
            qmarray_free(ft->cores[ii]); ft->cores[ii]=NULL;
            ft->cores[ii] = qmam(Q,R, temp->ncols);
            nrows = left_ind[ii+1]->n;

            qmarray_free(temp); temp = NULL;
            qmarray_free(Q); Q = NULL;
            free(pivind); pivind =NULL;
            free(pivx); pivx = NULL;
            free(R); R=NULL;

        }
        ii = dim-1;
        if (cargs->verbose > 1){
            printf(" ............. on left-right sweep (%zu/%zu)\n",ii,dim-1);
        }
        qmarray_free(ft->cores[ii]); ft->cores[ii] = NULL;
        ft->cores[ii] = prepCore(ii,cargs->ranks[ii],f,args,bd,
                                 left_ind,right_ind,cargs,apargs,1);

        if (VFTCROSS == 2){
            printf ("got it \n");
            //print_qmarray(ft->cores[ii],0,NULL);
            printf("integral = %G\n",function_train_integrate(ft));
            struct FunctionTrain * tprod = function_train_product(ft,ft);
            printf("prod integral = %G\n",function_train_integrate(tprod));
            printf("norm2 = %G\n",function_train_norm2(ft));
            print_qmarray(tprod->cores[0],0,NULL);
            //print_qmarray(tprod->cores[1],0,NULL);
            function_train_free(tprod);
        }

        if (VFTCROSS){
            printf("\n\n\n Index sets after Left-Right cross\n");
            for (ii = 0; ii < dim; ii++){
                printf("ii = %zu\n",ii);
                printf( "left index set = \n");
                print_cross_index(left_ind[ii]);
                printf( "right index set = \n");
                print_cross_index(right_ind[ii]);
            }
            printf("\n\n\n");
        }
        
        //printf("compute difference \n");
        //printf("norm fti = %G\n",function_train_norm2(fti));
        //printf("norm ft = %G\n",function_train_norm2(ft));
        diff = function_train_relnorm2diff(ft,fti);
        //printf("diff = %G\n",diff);
        //den = function_train_norm2(ft);
        //diff = function_train_norm2diff(ft,fti);
        //if (den > ZEROTHRESH){
        //    diff /= den;
       // }

        if (cargs->verbose > 0){
            den = function_train_norm2(ft);
            printf("...... New FT norm L/R Sweep = %E\n",den);
            printf("...... Error L/R Sweep = %E\n",diff);
        }
        
        if (diff < cargs->epsilon){
            done = 1;
            break;
        }
        
        /* function_train_free(fti); fti=NULL; */
        /* fti = function_train_copy(ft); */
        
        //printf("copied \n");
        //printf("copy diff= %G\n", function_train_norm2diff(ft,fti));

        ///////////////////////////////////////////////////////
        // right-left sweep
        for (oncore = 1; oncore < dim; oncore++){
            
            ii = dim-oncore;

            if (cargs->verbose > 1){
                printf(" ............. on right_left sweep (%zu/%zu)\n",ii,dim-1);
            }

            nrows = ft->ranks[ii];

            if (VFTCROSS){
                printf("do prep\n");
                printf( "left index set = \n");
                print_cross_index(left_ind[ii]);
                printf( "right index set = \n");
                print_cross_index(right_ind[ii]);
            }
            //printf("prep core\n");
            temp = prepCore(ii,nrows,f,args,bd,left_ind,right_ind,cargs,apargs,0);
            //printf("prepped core\n");

            R = calloc_double(temp->nrows * temp->nrows);
            Q = qmarray_householder_simple_grid("LQ", temp,R,grid[ii]);
            Qt = qmarray_transpose(Q);
            pivind = calloc_size_t(ft->ranks[ii]);
            pivx = calloc_double(ft->ranks[ii]);

            info = qmarray_maxvol1d(Qt,R,pivind,pivx,grid[ii]);
            
            if (VFTCROSS){
                printf("got info=%d\n",info);
                printf("indices and pivots\n");
                iprint_sz(ft->ranks[ii],pivind);
                dprint(ft->ranks[ii],pivx);
            }

            //printf("got maxvol\n");
            if (info < 0){
                fprintf(stderr, "noinvertible submatrix in maxvol in rl cross\n");
            }

            if (info > 0){
                fprintf(stderr, " error in qmarray_maxvol1d \n");
                exit(1);
            }
            qmarray_free(Q); Q = NULL;

            //printf("pivx \n");
            //dprint(ft->ranks[ii], pivx);

            Q = qmam(Qt,R, temp->nrows);

            qmarray_free(ft->cores[ii]); ft->cores[ii] = NULL;
            ft->cores[ii] = qmarray_transpose(Q);

            cross_index_free(right_ind[ii-1]);
            if (ii < dim-1){
                //printf("are we really here? oncore=%zu,ii=%zu\n",oncore,ii);
                right_ind[ii-1] =
                    cross_index_create_nested_ind(1,ft->ranks[ii],pivind,
                                                  pivx,right_ind[ii]);
            }
            else{
                //printf("lets update the cross index ii=%zu\n",ii);
                right_ind[ii-1] = cross_index_alloc(1);
                for (size_t zz = 0; zz < ft->ranks[ii]; zz++){
                    cross_index_add_index(right_ind[ii-1],1,&(pivx[zz]));
                }
                //printf("updated\n");
            }

            qmarray_free(temp); temp = NULL;
            qmarray_free(Q); Q = NULL;
            qmarray_free(Qt); Qt = NULL;
            free(pivind);
            free(pivx);
            free(R); R=NULL;

        }

        ii = 0;
        qmarray_free(ft->cores[ii]); ft->cores[ii] = NULL;

        if (cargs->verbose > 1)
            printf(" ............. on right_left sweep (%zu/%zu)\n",ii,dim-1);
        ft->cores[ii] = prepCore(ii,1,f,args,bd,left_ind,right_ind,
                                 cargs,apargs,-1);
        if (cargs->verbose > 1)
            printf(" ............. done with right left sweep\n");
 

        if (VFTCROSS){
            printf("\n\n\n Index sets after Right-left cross\n");
            for (ii = 0; ii < dim; ii++){
                printf("ii = %zu\n",ii);
                printf( "left index set = \n");
                print_cross_index(left_ind[ii]);
                printf( "right index set = \n");
                print_cross_index(right_ind[ii]);
            }
            printf("\n\n\n");
        }

        diff = function_train_relnorm2diff(ft,fti);
        if (fti2 != NULL){
            diff2 = function_train_relnorm2diff(ft,fti2);
        }
        else{
            diff2 = diff;
        }

        //den = function_train_norm2(ft);
        //diff = function_train_norm2diff(ft,fti);
        //if (den > ZEROTHRESH){
        //    diff /= den;
       // }

        if (cargs->verbose > 0){
            den = function_train_norm2(ft);
            printf("...... New FT norm R/L Sweep = %3.9E\n",den);
            printf("...... Error R/L Sweep = %E,%E\n",diff,diff2);
        }

        if ( (diff2 < cargs->epsilon) || (diff < cargs->epsilon)){
            done = 1;
            break;
        }

        function_train_free(fti2); fti2 = NULL;
        fti2 = function_train_copy(fti);
        function_train_free(fti); fti=NULL;
        fti = function_train_copy(ft);

        iter++;
        if (iter  == cargs->maxiter){
            done = 1;
            break;
        }
    }

    function_train_free(fti); fti=NULL;
    function_train_free(fti2); fti2=NULL;
    return ft;
}



/***********************************************************//**
    An interface for cross approximation of a function

    \param[in] f      - function
    \param[in] args   - function arguments
    \param[in] bds    - bounding box 
    \param[in] xstart - location for first fibers 
                        (if null then middle of domain)
    \param[in] fca    - cross approximation args, 
                        if NULL then default exists
    \param[in] apargs - function approximation arguments 
                        (if null then defaults)

    \return function train decomposition of f

    \note
    Nested indices both left and right
***************************************************************/
struct FunctionTrain *
function_train_cross(double (*f)(double *, void *), void * args, 
                     struct BoundingBox * bds,
                     double ** xstart,
                     struct FtCrossArgs * fca,
                     struct FtApproxArgs * apargs)
{   
    size_t dim = bds->dim;
    
    double ** init_x = NULL;
    struct FtCrossArgs * fcause = NULL;
    struct FtApproxArgs * fapp = NULL;
    
    double * init_coeff = darray_val(dim,1.0/ (double) dim);
    
    size_t ii;

    if (fca != NULL){
        fcause = ft_cross_args_copy(fca);
    }
    else{
        size_t init_rank = 5;
        fcause = ft_cross_args_alloc(dim,init_rank);
    }

    if ( xstart == NULL) {
        init_x = malloc_dd(dim);
        // not used in cross because left->right first
        //init_x[0] = calloc_double(fcause->ranks[1]); 
        
        init_x[0] = linspace(bds->lb[0],bds->ub[0],
                                fcause->ranks[1]);
        for (ii = 0; ii < dim-1; ii++){
            init_x[ii+1] = linspace(bds->lb[ii+1],bds->ub[ii+1],
                                    fcause->ranks[ii+1]);
//            dprint(fcause->ranks[ii+1],init_x[ii+1]);
        }

    }
    else{
        init_x = xstart;
    }
    
    /* dprint(fcause->ranks[1],init_x[0]); */
    /* for (size_t zz = 0; zz < dim-1; zz++){ */
    /*     dprint(fcause->ranks[zz+1],init_x[zz+1]); */
    /* } */
    
    enum poly_type ptype;
    if (apargs != NULL){
        fapp = apargs;
    }
    else{
        ptype = LEGENDRE;
        fapp = ft_approx_args_createpoly(dim,&ptype,NULL);
    }
    
    assert (dim <= 1000);
    struct CrossIndex * isl[1000];
    struct CrossIndex * isr[1000];
    /* cross_index_array_initialize(dim,isl,1,0,NULL,NULL); */
    /* cross_index_array_initialize(dim,isr,0,1,fcause->ranks,init_x); */
    /* struct FunctionTrain * ftref = function_train_constant_d(fapp,1.0,bds); */

    cross_index_array_initialize(dim,isr,0,1,fcause->ranks,init_x);
    cross_index_array_initialize(dim,isl,0,0,fcause->ranks+1,init_x);
    /* exit(1); */
    struct FunctionTrain * ftref = function_train_alloc(dim);
/*     printf("ranks are !!!\n"); */
/*     iprint_sz(dim+1,fcause->ranks); */
    /* printf("create ftref!\n"); */
    for (ii = 0; ii < dim; ii++){

        /* printf("-------------------------\n"); */
        /* printf("ii = %zu\n",ii); */
        /* printf( "left index set = \n"); */
        /* print_cross_index(isl[ii]); */
        /* printf( "right index set = \n"); */
        /* print_cross_index(isr[ii]); */
        /* printf("-------------------------------\n"); */
        size_t nrows = fcause->ranks[ii];
        size_t ncols = fcause->ranks[ii+1];
        ftref->cores[ii] = prepCore(ii,nrows,f,args,bds,isl,isr,fcause,fapp,0);
        ftref->ranks[ii] = nrows;
        ftref->ranks[ii+1] = ncols;
     
    }

    /* exit(1); */
    struct FunctionTrain * ft  = NULL;
    /* printf("start cross!\n"); */
    ft = ftapprox_cross_rankadapt(f,args,bds,ftref,isl,
                                  isr,fcause,fapp);
    
    
    if (xstart == NULL){
        free_dd(dim, init_x); //init_x = NULL;
    }
    if (apargs == NULL){
        ft_approx_args_free(fapp); fapp = NULL;
    }
    ft_cross_args_free(fcause);

    function_train_free(ftref); ftref = NULL;
    for (ii = 0; ii < dim;ii++){
        cross_index_free(isl[ii]);
        cross_index_free(isr[ii]);
    }
    free(init_coeff); init_coeff = NULL;
    return ft;
}


/***********************************************************//**
    An interface for cross approximation of a function
    on an unbounded domain

    \param[in] f        - function
    \param[in] args     - function arguments
    \param[in] xstart   - location for first fibers 
                          (if null then middle of domain)
    \param[in] fcain    - cross approximation args, 
                          if NULL then default exists
    \param[in] apargsin - function approximation arguments 
                          (if null then defaults)
    \param[in] foptin   - optimization arguments                          

    \return function train decomposition of f

    \note
    Nested indices both left and right
***************************************************************/
struct FunctionTrain *
function_train_cross_ub(double (*f)(double *, void *), void * args,
                        size_t dim,
                        double ** xstart,
                        struct FtCrossArgs * fcain,
                        struct FtApproxArgs * apargsin,
                        struct FiberOptArgs * foptin)
{
    enum poly_type ptype = HERMITE;
    struct FtCrossArgs  * fca   = NULL;
    struct FtApproxArgs * aparg = NULL;
    struct FiberOptArgs * fopt  = NULL;
    struct OpeAdaptOpts * aopts = NULL;
    double ** startnodes = NULL;;
    struct c3Vector * optnodes = NULL;

    if (fcain != NULL){
        fca = ft_cross_args_copy(fcain);
    }
    else{
        size_t init_rank = 5;
        fca = ft_cross_args_alloc(dim,init_rank);
    }

    if (foptin != NULL){
        fopt = foptin;
    }
    else{
        // default pivot / fiber locations
        size_t N = 100;
        double * x = linspace(-10.0,10.0,N);
        optnodes = c3vector_alloc(N,x);
        fopt = fiber_opt_args_bf_same(dim,optnodes);
        free(x); x = NULL;
        ft_cross_args_set_maxrank_all(fca,N);
    }
    ft_cross_args_set_optargs(fca,fopt);

    if (apargsin != NULL){
        aparg =apargsin;
    }
    else{
        aopts = ope_adapt_opts_alloc();
        ope_adapt_opts_set_start(aopts,5);
        ope_adapt_opts_set_maxnum(aopts,30);
        ope_adapt_opts_set_coeffs_check(aopts,2);
        ope_adapt_opts_set_tol(aopts,1e-5);
        aparg = ft_approx_args_createpoly(dim,&ptype,aopts);
        
    }

    if (xstart != NULL){
        startnodes = xstart;
    }
    else{
        startnodes = malloc_dd(dim);
        double lbs = -2.0;
        double ubs = 2.0;
        startnodes[0] = calloc_double(fca->ranks[1]); 
        for (size_t ii = 0; ii < dim-1; ii++){
            startnodes[ii+1] = linspace(lbs,ubs,fca->ranks[ii+1]);
        }
    }

    struct BoundingBox * bds = bounding_box_init(dim,-DBL_MAX,DBL_MAX);
    
    struct FunctionTrain * ft = NULL;
    ft = function_train_cross(f,args,bds,startnodes,fca,aparg);
    bounding_box_free(bds); bds = NULL;

    if (foptin == NULL){
        c3vector_free(optnodes); optnodes = NULL;
        fiber_opt_args_free(fopt); fopt = NULL;
    }
    if (apargsin == NULL){
        ope_adapt_opts_free(aopts); aopts = NULL;
        ft_approx_args_free(aparg); aparg = NULL;
    }
    ft_cross_args_free(fca); fca = NULL;
    if (xstart == NULL){
        free_dd(dim,startnodes);
        startnodes = NULL;
    }
    return ft;
}


/***********************************************************//**
    Cross approximation of a of a dim-dimensional function with rank adaptation

    \param[in]     f      - function
    \param[in]     args   - function arguments
    \param[in]     bds    - bounds on input space
    \param[in,out] ftref  - initial ftrain decomposition, 
                            changed in func
    \param[in,out] isl    - left indices (first element should be NULL)
    \param[in,out] isr    - right indices (last element should be NULL)
    \param[in]     fca    - algorithm parameters, 
                            if NULL then default paramaters used
    \param[in]     apargs - function approximation args 

    \return function train decomposition of f

    \note
    both left and right indices are nested
***************************************************************/
struct FunctionTrain *
ftapprox_cross_rankadapt( double (*f)(double *, void *),
                          void * args,
                          struct BoundingBox * bds, 
                          struct FunctionTrain * ftref, 
                          struct CrossIndex ** isl, 
                          struct CrossIndex ** isr,
                          struct FtCrossArgs * fca,
                          struct FtApproxArgs * apargs)
                
{
    size_t dim = bds->dim;
    double eps = fca->epsround;
    size_t kickrank = fca->kickrank;

    struct FunctionTrain * ft = NULL;


    ft = ftapprox_cross(f,args,bds,ftref,isl,isr,fca,apargs);

    if (fca->adapt == 0){
        return ft;
    }
    // printf("found left index\n");
    // print_cross_index(isl[1]);
    // printf("found right index\n");
    //print_cross_index(isr[0]);
    
    //return ft;
    if (fca->verbose > 0){
        printf("done with first cross... rounding\n");
    }
    size_t * ranks_found = calloc_size_t(dim+1);
    memmove(ranks_found,fca->ranks,(dim+1)*sizeof(size_t));
    
    struct FunctionTrain * ftc = function_train_copy(ft);
    struct FunctionTrain * ftr = function_train_round(ft, eps);
    /* printf("rounded ranks = "); iprint_sz(dim+1,ftr->ranks); */
    //struct FunctionTrain * ftr = function_train_copy(ft);
    //return ftr; 
    //printf("DOOONNTT FORGET MEEE HEERREEEE \n");
    int adapt = 0;
    size_t ii;
    for (ii = 1; ii < dim; ii++){
        /* printf("%zu == %zu\n",ranks_found[ii],ftr->ranks[ii]); */
        if (ranks_found[ii] == ftr->ranks[ii]){
            if (fca->ranks[ii] < fca->maxranks[ii-1]){
                adapt = 1;
                size_t kicksize; 
                if ( (ranks_found[ii]+kickrank) <= fca->maxranks[ii-1]){
                    kicksize = kickrank;
                }
                else{
                    kicksize = fca->maxranks[ii-1] -ranks_found[ii];
                }
                fca->ranks[ii] = ranks_found[ii] + kicksize;
                
                // simply repeat the last nodes
                // this is not efficient but I don't have a better
                // idea. Could do it using random locations but
                // I don't want to.
                cross_index_copylast(isr[ii-1],kicksize);
            
                ranks_found[ii] = fca->ranks[ii];
            }
        }
    }

    //printf("adapt here! adapt=%zu\n",adapt);

    size_t iter = 0;
//    double * xhelp = NULL;
    while ( adapt == 1 )
    {
        adapt = 0;
        if (fca->verbose > 0){
            printf("adapting \n");
            printf("Increasing rank\n");
            iprint_sz(ft->dim+1,fca->ranks);
        }
        
        function_train_free(ft); ft = NULL;
        ft = ftapprox_cross(f, args, bds, ftc, isl, isr, fca,apargs);
         
        function_train_free(ftc); ftc = NULL;
        function_train_free(ftr); ftr = NULL;

        //printf("copying\n");
        function_train_free(ftc); ftc = NULL;
        ftc = function_train_copy(ft);
        //printf("rounding\n");
        function_train_free(ftr); ftr = NULL;
        //ftr = function_train_copy(ft);//, eps);
        ftr = function_train_round(ft, eps);
        //printf("done rounding\n");
        for (ii = 1; ii < dim; ii++){
            if (ranks_found[ii] == ftr->ranks[ii]){
                if (fca->ranks[ii] < fca->maxranks[ii-1]){
                    adapt = 1;
                    size_t kicksize; 
                    if ( (ranks_found[ii]+kickrank) <= fca->maxranks[ii-1]){
                        kicksize = kickrank;
                    }
                    else{
                        kicksize = fca->maxranks[ii-1] -ranks_found[ii];
                    }
                    fca->ranks[ii] = ranks_found[ii] + kicksize;
                
            
                    // simply repeat the last nodes
                    // this is not efficient but I don't have a better
                    // idea. Could do it using random locations but
                    // I don't want to.
                    cross_index_copylast(isr[ii-1],kicksize);
            
                    ranks_found[ii] = fca->ranks[ii];
                }
            }
        }

        iter++;
        /* if (iter == fca->maxiteradapt) { */
        /*     adapt = 0; */
        /* } */
        //adapt = 0;

    }
    function_train_free(ft); ft = NULL;
    function_train_free(ftc); ftc = NULL;
    free(ranks_found);
    return ftr;
}



struct vfdata
{   
    double (*f)(double *,size_t,void *);
    size_t ind;
    void * args;
};

double vfunc_eval(double * x, void * args)
{
    struct vfdata * data = args;
    double out = data->f(x,data->ind,data->args);
    return out;
}

/***********************************************************//**
    An interface for cross approximation of a vector-valued function

    \param[in] f      - function
    \param[in] args   - function arguments
    \param[in] nfuncs - function arguments
    \param[in] bds    - bounding box 
    \param[in] xstart - location for first fibers 
                        (if null then middle of domain)
    \param[in] fcain  - cross approximation args, if NULL then 
                        default exists
    \param[in] apargs - function approximation arguments 
                        (if null then defaults)

    \return function train decomposition of f

    \note
    Nested indices both left and right
***************************************************************/
struct FT1DArray *
ft1d_array_cross(double (*f)(double *, size_t, void *), void * args, 
                size_t nfuncs,
                struct BoundingBox * bds,
                double ** xstart,
                struct FtCrossArgs * fcain,
                struct FtApproxArgs * apargs)
{   
    struct vfdata data;
    data.f = f;
    data.args = args;

    struct FT1DArray * fta = ft1d_array_alloc(nfuncs);
    for (size_t ii = 0; ii < nfuncs; ii++){
        data.ind = ii;
        struct FtCrossArgs * fca = ft_cross_args_copy(fcain);
        fta->ft[ii] = function_train_cross(vfunc_eval,&data,bds,xstart,fca,apargs);
        ft_cross_args_free(fca); fca = NULL;
    }
    return fta;
}



//////////////////////////////////////////////////////////////////////
// Blas type interface 1
//




