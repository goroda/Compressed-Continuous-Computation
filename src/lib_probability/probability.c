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

/** \file probability.c
 * Provides routines for working with probability with function trains
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "stringmanip.h"
#include "array.h"

#include "lib_optimization.h"
#include "probability.h"
#include "linalg.h"
#include "lib_clinalg.h"

/***********************************************************//**
    Allocate a linear transform Ax +b

    \param[in] dimin  - input dimension 
    \param[in] dimout - output dimension
    \param[in] A      - slopes of transformation
    \param[in] b      - offset of transformation
    \param[in] det  - pointer to a double representing the determinant of A

    \return linear transform struct

    \note
        Makes a copy of A/b. if A/b are NULL then allocates
        space for them.
***************************************************************/
struct LinearTransform *
linear_transform_alloc(size_t dimin, size_t dimout, double * A, 
                        double * b, double det)
{
    struct LinearTransform * lt = malloc(sizeof(struct LinearTransform));
    if (lt == NULL){
        fprintf(stderr, "Failed to allocate LinearTransform\n");
        exit(1);
    }
    
    lt->dimin = dimin;
    lt->dimout = dimout;

    lt->A = calloc_double(dimin*dimout);
    if (A != NULL){
        memmove(lt->A,A,dimin*dimout*sizeof(double));
        lt->det = det;
    }

    lt->b = calloc_double(dimout);
    if (b!= NULL){
        memmove(lt->b,b,dimout*sizeof(double));
    }
    
    // these are only allocated if necessary!!
    lt->Ainv = NULL;
    lt->binv = NULL;

    return lt;
}

/***********************************************************//**
    Copy a linear transform

    \param[in] lt - linear transform to copy

    \return newlt - copied linear transform
***************************************************************/
struct LinearTransform * linear_transform_copy(struct LinearTransform * lt)
{
    struct LinearTransform * newlt = 
        linear_transform_alloc(lt->dimin, lt->dimout, lt->A, lt->b, lt->det);

    if (lt->Ainv != NULL){
        newlt->Ainv = calloc_double(lt->dimin*lt->dimout);
        memmove(newlt->Ainv, lt->Ainv, lt->dimin * lt->dimout * sizeof(double));
        newlt->binv = calloc_double(lt->dimin);
        memmove(newlt->binv, lt->binv, lt->dimin * sizeof(double));
        newlt->detinv = lt->detinv;
    }
    return newlt;
    
}

/***********************************************************//**
    Frees memory allocated to Linear Transform

    \param[in,out] lt - linear transform to free
***************************************************************/
void linear_transform_free(struct LinearTransform * lt){
    
    if (lt != NULL){
        free(lt->A); lt->A = NULL;
        free(lt->b); lt->b = NULL;

        free(lt->Ainv); lt->Ainv = NULL;
        free(lt->binv); lt->binv = NULL;


        free(lt); lt = NULL;
    }
}

/***********************************************************//**
    Apply a linear transform y = Ax + b
    
    \param[in] dimx - dimension of input
    \param[in] dimb - dimension of output
    \param[in] A    - A matrix
    \param[in] x    - input vector
    \param[in] b    - bvector

    \return transformed vector

    \note should specialize this to particular transforms ...
***************************************************************/
double * 
linear_transform_apply(size_t dimx, size_t dimb, double * A, 
                            double * x, double * b)
{
    double * xout = calloc_double(dimb);
    memmove(xout,b,dimb*sizeof(double));
    cblas_dgemv(CblasColMajor, CblasNoTrans, dimb,dimx,1.0,A,
                    dimb,x,1,1.0,xout,1);
    //printf("xout[0]=%G\n",xout[0]);
    return xout;
}

/***********************************************************//**
    Invert a linear transform

    \param[in,out] lt - linear transform 
***************************************************************/
void linear_transform_invert(struct LinearTransform * lt)
{
    if (lt->Ainv == NULL){
        
        if (lt->mt == LT){
            lt->Ainv = calloc_double(lt->dimin*lt->dimout);
            memmove(lt->Ainv, lt->A, lt->dimin * lt->dimout * sizeof(double));
            int info;
            dtrtri_("L","N",&(lt->dimout),lt->Ainv,&(lt->dimout),&info);
            if (info != 0){
                fprintf(stderr, "Error inverting lower triangular lin transform %d\n",info);
            }
            size_t ii, jj;
            lt->binv = calloc_double(lt->dimin);
            for (ii = 0; ii < lt->dimout; ii++){
                lt->binv[ii] = -lt->b[ii];
                for (jj = 0; jj < ii; jj++){
                    lt->Ainv[ii*lt->dimout+jj] = 0.0;//lt->Ainv[jj*lt->dimout+ii];
                }
            }
            
            cblas_dtrmv(CblasColMajor,CblasLower,CblasNoTrans,CblasNonUnit,lt->dimout,lt->Ainv,
                lt->dimout,lt->binv,1);
            //cblas_dsymv(CblasColMajor,CblasLower,lt->dimout,-1.0,lt->Ainv,
            //                lt->dimout,lt->b,1,0.0,lt->binv,1);

            lt->detinv = 1.0/lt->det;
            lt->mti = LT;


        }
        else{
            fprintf(stderr, "Cannot invert this type of transform\n");
        }
    }
}

/***********************************************************//**
    Serialize a linear transform

    \param[in,out] ser       - stream to serialize to
    \param[in]     lt        - function train
    \param[in,out] totSizeIn - if NULL then serialize, if not NULL then return size

    \return ptr - ser shifted by number of ytes
***************************************************************/
unsigned char * 
linear_transform_serialize(unsigned char * ser, struct LinearTransform * lt,
                size_t *totSizeIn)
{   
    int invexists = (lt->Ainv != NULL);
    
    size_t totSize = (lt->dimin * lt->dimout) * sizeof(double) + sizeof(size_t)  + // A
                     (lt->dimout) * sizeof(double) + sizeof(size_t) +//
                     sizeof(size_t) + sizeof(size_t) + // dimin dimout
                     sizeof(double) + sizeof(enum lt_matrix_type) + // det and mat type
                     sizeof(int);  // flag for whether inverse exists
    
    if (invexists){
        totSize += (lt->dimin * lt->dimout) * sizeof(double) + sizeof(size_t) + // Ainv
                     (lt->dimin) * sizeof(double) + sizeof(size_t) + //binv
                     sizeof(double) + sizeof(int); // det and mat type

    }

    if (totSizeIn != NULL){
        *totSizeIn = totSize;
        return ser;
    }

    unsigned char * ptr = ser;
    ptr = serialize_size_t(ptr, lt->dimin);
    ptr = serialize_size_t(ptr, lt->dimout);
    ptr = serialize_int(ptr, lt->mt);
    ptr = serialize_int(ptr, invexists);
    ptr = serialize_doublep(ptr,lt->A, lt->dimin * lt->dimout);
    ptr = serialize_doublep(ptr,lt->b, lt->dimout);
    ptr = serialize_double(ptr,lt->det);

    if (invexists){
        ptr = serialize_int(ptr, lt->mti);
        ptr = serialize_doublep(ptr,lt->Ainv, lt->dimin * lt->dimout);
        ptr = serialize_doublep(ptr,lt->binv, lt->dimin);
        ptr = serialize_double(ptr,lt->detinv);
    }

    return ptr;
}   

/********************************************************//**
*   deserialize a linear transform
*
*   \param[in]     ser - string to deserialize
*   \param[in,out] lt  - linear transform
*
*   \return ptr - ser + number of bytes of linear transform
*************************************************************/
unsigned char * 
linear_transform_deserialize(unsigned char * ser, struct LinearTransform ** lt)
{
    if ( NULL == ( (*lt) = malloc(sizeof(struct LinearTransform)))){
        fprintf(stderr, "failed to allocate memory for linear transform.\n");
        exit(1);
    }
    int invexists;
    unsigned char * ptr;
    int mat_type;
    ptr = deserialize_size_t(ser,&((*lt)->dimin));
    ptr = deserialize_size_t(ptr,&((*lt)->dimout));
    size_t Asize;
    size_t bsize;
    ptr = deserialize_int(ptr,&mat_type);
    (*lt)->mt = (enum lt_matrix_type) mat_type;
    ptr = deserialize_int(ptr,&invexists);
    ptr = deserialize_doublep(ptr,&((*lt)->A),&Asize);
    ptr = deserialize_doublep(ptr,&((*lt)->b),&bsize);
    ptr = deserialize_double(ptr,&((*lt)->det));
    if (invexists){
        ptr = deserialize_int(ptr,&mat_type);
        (*lt)->mti = (enum lt_matrix_type) mat_type;
        ptr = deserialize_doublep(ptr,&((*lt)->Ainv),&Asize);
        ptr = deserialize_doublep(ptr,&((*lt)->binv),&bsize);
        ptr = deserialize_double(ptr,&((*lt)->detinv));
    }
    else{
        (*lt)->Ainv = NULL;
        (*lt)->binv = NULL;
    }
    return ptr;
}

/***********************************************************//**
    Allocate a Probability Density function

    \return pdf - allocated pdf (every membor is NULL)
***************************************************************/
struct ProbabilityDensity *
probability_density_alloc()
{
    struct ProbabilityDensity * pdf = malloc(sizeof(struct ProbabilityDensity));
    if (pdf == NULL){
        fprintf(stderr, "Failed to allocate LinearTransform\n");
        exit(1);
    }
    
    pdf->pdf = NULL;
    pdf->lt = NULL;
    pdf->transform = 0;
    pdf->extra = NULL;
    return pdf;
}

/***********************************************************//**
    Frees memory allocated to ProbabilityDensity

    \param[in,out] pdf - pdf to free
***************************************************************/
void probability_density_free(struct ProbabilityDensity * pdf){
    
    if (pdf != NULL){
        function_train_free(pdf->pdf); pdf->pdf = NULL;
        linear_transform_free(pdf->lt); pdf->lt = NULL;
        assert (pdf->extra == NULL); // dont have this yet
        free(pdf); pdf=NULL;
    }
}

/***********************************************************//**
    Serialize a probability density function 

    \param[in,out] ser       - stream to serialize to
    \param[in]     pdf       - density function
    \param[in,out] totSizeIn - if NULL then serialize, if not NULL then return size

    \return ptr - ser shifted by number of bytes
***************************************************************/
unsigned char * 
probability_density_serialize(unsigned char * ser, 
            struct ProbabilityDensity * pdf, size_t * totSizeIn)
{
    assert(pdf->extra == NULL);
    size_t totSize;
    if (totSizeIn != NULL){
        size_t size_pdf, size_lt;
        function_train_serialize(NULL,pdf->pdf,&size_pdf);
        linear_transform_serialize(NULL,pdf->lt,&size_lt);
        totSize = size_pdf + size_lt + sizeof(int) + sizeof(enum pdf_type);
        *totSizeIn = totSize;
        return ser;
    }

    unsigned char * ptr = ser;
    ptr = serialize_int(ptr,pdf->transform);
    ptr = serialize_int(ptr,pdf->type);
    ptr = function_train_serialize(ptr,pdf->pdf,NULL);
    ptr = linear_transform_serialize(ptr,pdf->lt,NULL);
    return ptr;
}

/********************************************************//**
*   deserialize a probability density function
*
*   \param[in] ser - string to deserialize
*   \param[in] pdf - density function
*
*   \return ptr - ser + number of bytes of linear transform
*************************************************************/
unsigned char * 
probability_density_deserialize(unsigned char * ser, 
        struct ProbabilityDensity ** pdf)
{
    unsigned char * ptr = ser;
    int ptype;
    *pdf = probability_density_alloc();
    ptr = deserialize_int(ptr,&((*pdf)->transform));
    ptr = deserialize_int(ptr,&ptype);
    (*pdf)->type = (enum pdf_type) ptype;
    ptr = function_train_deserialize(ptr,&((*pdf)->pdf));
    ptr = linear_transform_deserialize(ptr,&((*pdf)->lt));
    
    return ptr;
}

/***********************************************************//**
    Save a probability density function to a file
    
    \param[in] pdf      - pdf to to save
    \param[in] filename - name of file to save to

    \return success (1) or failure (0) of opening the file
***************************************************************/
int probability_density_save(struct ProbabilityDensity * pdf, char * filename)
{
    FILE *fp;
    fp =  fopen(filename, "w");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open %s\n", filename);
        return 0;
    }

    size_t totsize;
    probability_density_serialize(NULL,pdf,&totsize);
    unsigned char * data = malloc(totsize+sizeof(size_t));
    if (data == NULL){
        fprintf(stderr, "can't allocate space for saving density\n");
        return 0;
    }

    unsigned char * ptr = serialize_size_t(data,totsize);
    ptr = probability_density_serialize(ptr,pdf,NULL);
    
    fwrite(data,sizeof(unsigned char),totsize+sizeof(size_t),fp);

    free(data); data = NULL;
    fclose(fp);
    return 1;
}

/***********************************************************//**
    Load a probability density function from a file;
    
    \param[in] filename - filename to load

    \return pdf if successfull NULL otherwise
***************************************************************/
struct ProbabilityDensity * probability_density_load(char * filename)
{
    FILE *fp;
    fp =  fopen(filename, "r");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open %s\n", filename);
        return NULL;
    }

    size_t totsize;
    size_t k = fread(&totsize,sizeof(size_t),1,fp);
    if ( k != 1){
        printf("error reading file %s\n",filename);
        return NULL;
    }

    unsigned char * data = malloc(totsize);
    if (data == NULL){
        fprintf(stderr, "can't allocate space for loading density\n");
        return NULL;
    }

    k = fread(data,sizeof(unsigned char),totsize,fp);

    struct ProbabilityDensity * pdf = NULL;
    probability_density_deserialize(data,&pdf);
    
    free(data); data = NULL;
    fclose(fp);
    return pdf;
}


double pdf_stdmvn_helper(double x, size_t dim, void * args)
{
    size_t * totdim = args;
    assert (dim < *totdim);
    double scale = 1.0 / sqrt(pow(2.0*M_PI,*totdim));
    scale = pow(scale,1.0/( (double) *totdim));
    double out = scale * exp(-0.5*x*x);
    return out;
}

/***********************************************************//**
    Construct a standard normal pdf
    
    \param[in] dim - number of dimensions

    \return gaussian pdf 
***************************************************************/
struct ProbabilityDensity * probability_density_standard_normal(size_t dim)
{
    struct BoundingBox * bds = bounding_box_init_std(dim);
    size_t ii; 
    for (ii = 0; ii < dim; ii++){
        bds->lb[ii] = -10.0;
        bds->ub[ii] = 10.0;
    }

    double hmin = 1e-3;
    double delta = 1e-5;
    size_t init_rank = 3;
    double round_tol = 1e-3;
    double cross_tol = 1e-8;
    struct C3Approx * c3a = c3approx_create(CROSS,dim,bds->lb,bds->ub);
    c3approx_init_lin_elem(c3a);
    c3approx_set_lin_elem_delta(c3a,delta);
    c3approx_set_lin_elem_hmin(c3a,hmin);
    c3approx_init_cross(c3a,init_rank,0);
    c3approx_set_round_tol(c3a,round_tol);
    c3approx_set_cross_tol(c3a,cross_tol);
    
    struct FtApproxArgs * fapp = c3approx_get_approx_args(c3a);
    struct ProbabilityDensity * pdf = probability_density_alloc();
    pdf->pdf = function_train_rankone(dim,pdf_stdmvn_helper,&dim,bds,fapp);
    pdf->type = GAUSSIAN;
    
    bounding_box_free(bds);
    c3approx_destroy(c3a);

    return pdf;
}


/***********************************************************//**
    construct a multivariate normal distribution
    
    \param[in] dim  - number of dimensions
    \param[in] mean - mean of distribution
    \param[in] cov  - covariance of distribution (lower triangular)

    \return gaussian pdf 
***************************************************************/
struct ProbabilityDensity * 
probability_density_mvn(size_t dim, double * mean, double * cov)
{
    struct ProbabilityDensity * pdf = 
            probability_density_standard_normal(dim);
    
    //printf("in mvn!\n");
    pdf->transform = 1;
    pdf->lt = linear_transform_alloc(dim,dim,NULL,NULL,0.0);

    memmove(pdf->lt->b, mean, sizeof(double) * dim);
    memmove(pdf->lt->A, cov, dim*dim*sizeof(double));
    
    // compute cholesky
    int info;
    dpotrf_("L",&dim,pdf->lt->A,&dim,&info);
    pdf->lt->mt = LT;
    if (info != 0){
        fprintf(stdout,"Warning: cholesky finished with info=%d\n",info);
    }
    size_t ii, jj;
    pdf->lt->det = 1.0;
    for (ii = 0; ii < dim; ii++){
        pdf->lt->det *= pdf->lt->A[ii*dim+ii];
        for (jj = 0; jj < ii; jj++){
            pdf->lt->A[ii*dim+jj] = 0.0;
        }
    }
    
    pdf->type = GAUSSIAN;
    return pdf;
}

/***********************************************************//**
    Generate a Sample from a probability density function
    
    \param[in] pdf - probability density function

    \return sample
***************************************************************/
double * probability_density_sample(struct ProbabilityDensity * pdf)
{   
    double * out = NULL;
    if (pdf->type != GAUSSIAN){
        fprintf(stderr,"Cannot sample from a non-Gaussian pdf");
        exit(1);
    }
    if (pdf->type == GAUSSIAN){
        size_t dim = pdf->pdf->dim;
        double * norm = calloc_double(dim);
        size_t ii;
        for (ii = 0; ii < dim; ii++){
            norm[ii] = randn();
        }
        out = linear_transform_apply(dim,dim,pdf->lt->A,
                            norm,pdf->lt->b);
        free(norm); norm = NULL;
    }
    return out;
}

/***********************************************************//**
    Construct a laplace approximation
    
    \param[in] gradLogPost - function computes gradient of negative log posterior
    \param[in] hessLogPost - function computes hessian of negative log posterior
    \param[in] args        - arguments to grad and hess functions
    \param[in] dim         - number of dimensions
    \param[in] start       - start of optimization

    \return gaussian pdf 
***************************************************************/
struct ProbabilityDensity * 
probability_density_laplace(double *(*gradLogPost)(double * x, void * args), 
                            double *(*hessLogPost)(double * x, void * args),
                            void * args, size_t dim, double * start)
{
    double tol = 1e-4;
    double * mean = calloc_double(dim);
    memmove(mean,start,dim*sizeof(double));

    /* struct c3Opt * opt = c3opt_alloc(BFGS,dim); */
    /* c3opt_add_lb(opt,lb); */
    /* c3opt_add_ub(opt,ub); */
    /* c3opt_add_objective(opt,c3opt_f,NULL); */
    /* c3opt_set_verbose(opt,0); */

    /* res = c3opt_minimize(opt,start,&val); */

    printf("do newtons method\n");
    newton(&mean, dim, 1.0, tol, gradLogPost,hessLogPost,args);
    double * hess = hessLogPost(mean,args);
    double * cov = calloc_double(dim*dim);
    pinv(dim,dim,dim,hess,cov,0.0);

    struct ProbabilityDensity * pdf =
        probability_density_mvn(dim, mean, cov);

    free(mean); mean = NULL;
    free(hess); hess = NULL;
    free(cov); cov= NULL;
    
    return pdf;
}

/***********************************************************//**
    Evaluate the probability density function   

    \param[in] pdf - pdf to evaluate
    \param[in] x   - location at which to evaluate

    \return out - evaluation
***************************************************************/
double probability_density_eval(struct ProbabilityDensity * pdf, double * x)
{
    double out = 0.0;
    double * lb = probability_density_lb_base(pdf);
    double * ub = probability_density_ub_base(pdf);

    //assert(pdf->transform == 0); // havent implemented transform yet
    if (pdf->transform == 1){
        //printf("not implemented yet!\n");
        linear_transform_invert(pdf->lt);
        double * temp = linear_transform_apply(pdf->lt->dimout,
                pdf->lt->dimin, pdf->lt->Ainv, x, pdf->lt->binv);

        size_t ii;
        int good = 1;
        for (ii = 0; ii < pdf->pdf->dim; ii++){
            if (temp[ii] < lb[ii]){
                good = 0;
                out = 0.0;
                break;
            }
            else if (temp[ii] > ub[ii]){
                good = 0;
                out = 0.0;
                break;
            }
        }
        if (good == 1){
            out = function_train_eval(pdf->pdf,temp) * fabs(pdf->lt->detinv);
        }
                   
        free(temp); temp = NULL;
    }   
    else{
        size_t ii;
        int good = 1;
        for (ii = 0; ii < pdf->pdf->dim; ii++){
            if (x[ii] < lb[ii]){
                good = 0;
                out = 0.0;
                break;
            }
            else if (x[ii] > ub[ii]){
                good = 0;
                out = 0.0;
                break;
            }
        }
        if (good == 1){
            out = function_train_eval(pdf->pdf,x);
        }
    }
    free(lb); lb = NULL;
    free(ub); ub = NULL;
    return out;
}

/***********************************************************//**
    Evaluate the gradient of the log probability density function   

    \param[in] pdf - gradient of log(pdf) is obtained
    \param[in] x   - location at which to obtain the gradient

    \return out - gradient of log pdf at *x*
***************************************************************/
double * 
probability_density_log_gradient_eval(struct ProbabilityDensity * pdf, 
                                        double * x)
{
    double * out = NULL;
    //printf("here!\n");
    struct FT1DArray * grad = function_train_gradient(pdf->pdf);
    //printf("got grad\n");
    if (pdf->transform == 0){
        double val = 1.0 / probability_density_eval(pdf,x);
        out = ft1d_array_eval(grad,x);
        size_t ii;
        for (ii = 0; ii < grad->size; ii++){
            out[ii] = out[ii] * val;
        }
    }
    else if (pdf->transform == 1){
        if (pdf->lt->Ainv == NULL){
            linear_transform_invert(pdf->lt);
        }
        double * xtemp = linear_transform_apply(pdf->lt->dimout,
                pdf->lt->dimin, pdf->lt->Ainv, x, pdf->lt->binv);
        double val = 1.0 / function_train_eval(pdf->pdf, xtemp);
        double * temp = ft1d_array_eval(grad,xtemp);
        out = calloc_double(pdf->pdf->dim);
        cblas_dgemv(CblasColMajor, CblasTrans, pdf->pdf->dim, pdf->pdf->dim,
                                    val, pdf->lt->Ainv, pdf->pdf->dim,temp, 1,
                                    0.0, out, 1);
        free(xtemp); xtemp = NULL;
        free(temp); temp = NULL;
    }
    else{
        fprintf(stderr,
            "gradient of LOG pdf with transform type %d is not available\n",
            pdf->transform);
    }
    ft1d_array_free(grad); grad = NULL;
    
    return out;
}

/***********************************************************//**
    Evaluate the hessian of the log probability density function   

    \param[in] pdf - hessian of log(pdf) is obtained
    \param[in] x   - location at which to obtain the hessian

    \return out - hessian of log pdf at *x*
***************************************************************/
double * 
probability_density_log_hessian_eval(struct ProbabilityDensity * pdf, 
                                        double * x)
{
    double * out = NULL;
    if (pdf->transform == 0){
        double val = 1.0 / probability_density_eval(pdf,x);
        double val_squared = pow(val,2);

        struct FT1DArray * grad = function_train_gradient(pdf->pdf);
        double * grad_eval = ft1d_array_eval(grad, x);

        struct FT1DArray * hess = function_train_hessian(pdf->pdf);
        double * hess_eval = ft1d_array_eval(hess,x);

        out = calloc_double(pdf->pdf->dim * pdf->pdf->dim);
        size_t ii,jj;
        for (ii = 0; ii < pdf->pdf->dim; ii++){
            for (jj = 0; jj < pdf->pdf->dim; jj++){
                out[jj*pdf->pdf->dim+ii] = 
                    -val_squared * grad_eval[ii] * grad_eval[jj] +
                    val * hess_eval[jj*pdf->pdf->dim+ii];
            }
        }

        free(grad_eval); grad_eval = NULL;
        free(hess_eval); hess_eval = NULL;

        ft1d_array_free(grad); grad = NULL;
        ft1d_array_free(hess); hess = NULL;
    }
    else if (pdf->transform == 1){
        if (pdf->lt->Ainv == NULL){
            linear_transform_invert(pdf->lt);
        }
        

        double * xtemp = linear_transform_apply(pdf->lt->dimout,
                pdf->lt->dimin, pdf->lt->Ainv, x, pdf->lt->binv);

        double val =  1.0 / function_train_eval(pdf->pdf,xtemp);
        double val_squared = pow(val,2.0);

        struct FT1DArray * grad = function_train_gradient(pdf->pdf);
        double * grad_eval = ft1d_array_eval(grad, xtemp);

        struct FT1DArray * hess = function_train_hessian(pdf->pdf);
        double * hess_eval = ft1d_array_eval(hess,xtemp);
    
        out = calloc_double(pdf->lt->dimout * pdf->lt->dimout);

        size_t ii, jj;
        
        double * grad_prod_temp = calloc_double(pdf->lt->dimout);
        for (ii = 0; ii < pdf->lt->dimout; ii++){
            grad_prod_temp[ii] = 
                    cblas_ddot(pdf->lt->dimout,grad_eval,1,
                        pdf->lt->Ainv + pdf->lt->dimout*ii,1);
        }
       
        double * temp = calloc_double(pdf->lt->dimout);
        for (ii = 0; ii < pdf->lt->dimout; ii++){
            cblas_dgemv(CblasColMajor, CblasTrans, pdf->pdf->dim, 
                    pdf->pdf->dim, val, hess_eval, pdf->pdf->dim,
                        pdf->lt->Ainv+ii*pdf->lt->dimout, 1, 0.0, temp, 1);

            for (jj = 0; jj < pdf->lt->dimout; jj++){
                out[jj*pdf->lt->dimout + ii] = - grad_prod_temp[jj];
                out[jj*pdf->lt->dimout + ii] *= grad_prod_temp[ii];
                out[jj*pdf->lt->dimout + ii] *= val_squared;


                out[jj*pdf->lt->dimout+ii] += 
                    cblas_ddot(pdf->lt->dimout,temp,1,
                        pdf->lt->Ainv+jj*pdf->lt->dimout,1);
            }
        }
        
        free(grad_prod_temp); grad_prod_temp = NULL;

        free(temp); temp = NULL;
        free(grad_eval); grad_eval = NULL;
        free(hess_eval); hess_eval = NULL;

        free(xtemp); xtemp = NULL;
        ft1d_array_free(grad); grad = NULL;
        ft1d_array_free(hess); hess = NULL;
    }
    else{
        fprintf(stderr,
            "gradient of LOG pdf with transform type %d is not available\n",
            pdf->transform);
    }
    
    return out;
}


/***********************************************************//**
    Compute the mean from the pdf

    \param[in] pdf whose mean to compute

    \return out - mean of the pdf

    \note
        I can make this faster by precomputing the integrals of each
        core first.
***************************************************************/
double * probability_density_mean(struct ProbabilityDensity * pdf)
{
    
    size_t ii;
    size_t dimft = pdf->pdf->dim; // dimension of ft
    size_t dimpdf = dimft; // dimension of pdf variable
    double * mean = NULL;
    struct BoundingBox * bds = bounding_box_init_std(dimft);
    for (ii = 0; ii < dimft; ii++){
        bds->lb[ii] = 
            generic_function_get_lower_bound(pdf->pdf->cores[ii]->funcs[0]);
        bds->ub[ii] = 
            generic_function_get_upper_bound(pdf->pdf->cores[ii]->funcs[0]);
    }
    if (pdf->transform == 1){
        dimpdf = pdf->lt->dimout;   
        mean = calloc_double(dimpdf);
        double * offset = calloc_double(dimft);
        for (ii = 0; ii < dimpdf; ii++){
            offset[0] = pdf->lt->b[ii];
            struct FunctionTrain * ftlin = 
                function_train_linear2(LINELM,NULL,dimft,
                                       bds,pdf->lt->A+ii,dimpdf,
                                       offset,1,NULL);
            mean[ii] = function_train_inner(ftlin,pdf->pdf);
            //printf("mean[%zu]=%G\n",ii,mean[ii]);
            function_train_free(ftlin);
        }
        free(offset);
    }
    else{
        mean = calloc_double(dimft);
        for (ii = 0; ii < dimpdf; ii++){
            struct Qmarray * temp = qmarray_copy(pdf->pdf->cores[ii]);
            struct Qmarray * tempx = qmarray_alloc(1,1);
            tempx->funcs[0] = 
                generic_function_linear(1.0,0.0,LINELM,NULL,
                                    bds->lb[ii],bds->ub[ii],NULL);

            qmarray_free(pdf->pdf->cores[ii]);
            pdf->pdf->cores[ii] = qmarray_kron(tempx,temp);
            
            mean[ii] = function_train_integrate(pdf->pdf);

            qmarray_free(pdf->pdf->cores[ii]);
            pdf->pdf->cores[ii] = qmarray_copy(temp);

            qmarray_free(temp);
            qmarray_free(tempx);
        }
    }
    

    bounding_box_free(bds); bds = NULL;
    return mean;
}

/***********************************************************//**
    Compute the covariance from the pdf

    \param[in] pdf - pdf whose mean to compute

    \return covariance matrix

    \note
        I can make this faster by precomputing the integrals of each
        core first.
***************************************************************/
double * probability_density_cov(struct ProbabilityDensity * pdf)
{
    size_t ii, jj;
    size_t dimft = pdf->pdf->dim; // dimension of ft
    size_t dimpdf = dimft; // dimension of pdf variable
    double * cov = NULL;
    
    struct BoundingBox * bds = bounding_box_init_std(dimft);
    for (ii = 0; ii < dimft; ii++){
        bds->lb[ii] = generic_function_get_lower_bound(pdf->pdf->cores[ii]->funcs[0]);
        bds->ub[ii] = generic_function_get_upper_bound(pdf->pdf->cores[ii]->funcs[0]);
    }

    double * mean = probability_density_mean(pdf);

    if (pdf->transform  == 1){

        dimpdf = pdf->lt->dimout;   
        cov = calloc_double(dimpdf*dimpdf);
        double * offset = calloc_double(dimft);
        for (ii = 0; ii < dimpdf; ii++){
            offset[0] = pdf->lt->b[ii] - mean[ii];
            struct FunctionTrain * ftleft = 
                function_train_linear2(LINELM,NULL,dimft,bds,
                                       pdf->lt->A+ii,dimpdf,offset,1,NULL);

            struct FunctionTrain * temp = function_train_product(ftleft,ftleft);
            cov[ii*dimpdf+ii] = function_train_inner(temp,pdf->pdf);
            function_train_free(temp);

            for (jj = ii+1; jj < dimpdf; jj++){
                offset[0] = pdf->lt->b[jj] - mean[jj];
                struct FunctionTrain * ftright = 
                    function_train_linear2(LINELM,NULL,dimft,bds,
                                           pdf->lt->A+jj,dimpdf,offset,1,NULL);

                struct FunctionTrain * ftprod = NULL;
                ftprod = function_train_product(ftleft,ftright);
                cov[ii*dimpdf+jj] = function_train_inner(ftprod,pdf->pdf);

                cov[jj*dimpdf+ii] = cov[ii*dimpdf+jj];
                function_train_free(ftright); ftright = NULL;
                function_train_free(ftprod); ftprod = NULL;

                function_train_free(ftright);
                function_train_free(ftprod);
            }
            function_train_free(ftleft);
        }
        free(offset);
    }
    else{
        cov = calloc_double(dimpdf * dimpdf);
//        enum poly_type ptype = LEGENDRE;
        struct FunctionTrain * ftc = function_train_copy(pdf->pdf);
        struct FunctionTrain * xvals = function_train_alloc(dimft);
        for (ii = 0; ii < dimpdf; ii++){
            double lb = generic_function_get_lower_bound(ftc->cores[ii]->funcs[0]);
            double ub = generic_function_get_upper_bound(ftc->cores[ii]->funcs[0]);
            xvals->cores[ii] = qmarray_alloc(1,1);
            xvals->cores[ii]->funcs[0] = 
                generic_function_linear(1.0,-mean[ii],LINELM,NULL,lb,ub,NULL);
        }

        for (ii = 0; ii < dimpdf; ii++){

            struct Qmarray * temp = qmarray_kron(xvals->cores[ii],pdf->pdf->cores[ii]);

            qmarray_free(ftc->cores[ii]);
            ftc->cores[ii] = qmarray_kron(xvals->cores[ii],temp);
            cov[ii*dimpdf+ii] = function_train_integrate(ftc);

            qmarray_free(ftc->cores[ii]);
            //maybe dont have to copy bc freeing later
            ftc->cores[ii] = qmarray_copy(temp); 
            qmarray_free(temp);
            for (jj = ii+1; jj < dimpdf; jj++){
                qmarray_free(ftc->cores[jj]);
                ftc->cores[jj] = qmarray_kron(xvals->cores[jj],pdf->pdf->cores[jj]);
                
                cov[ii*dimpdf+jj] = function_train_integrate(ftc);
                cov[jj*dimpdf+ii] = cov[ii*dimpdf+jj];

                qmarray_free(ftc->cores[jj]);
                ftc->cores[jj] = qmarray_copy(pdf->pdf->cores[jj]);
            }
            qmarray_free(ftc->cores[ii]);
            ftc->cores[ii] = qmarray_copy(pdf->pdf->cores[ii]);
        }
        function_train_free(xvals);
        function_train_free(ftc);
    }   
    bounding_box_free(bds); bds=NULL;
    free(mean); mean=NULL;
    return cov;
}

/***********************************************************//**
    Compute the variance (only diagonal of covariance matrix) from the pdf

    \param[in] pdf - pdf whose mean to compute

    \return out - variance of all variables

    \note
        I can make this faster by precomputing the integrals of each
        core first.
***************************************************************/
double * probability_density_var(struct ProbabilityDensity * pdf)
{

    size_t ii;
    size_t dimft = pdf->pdf->dim; // dimension of ft
    size_t dimpdf = dimft; // dimension of pdf variable
    double * var = NULL;

    struct BoundingBox * bds = bounding_box_init_std(dimft);
    for (ii = 0; ii < dimft; ii++){
        bds->lb[ii] = generic_function_get_lower_bound(pdf->pdf->cores[ii]->funcs[0]);
        bds->ub[ii] = generic_function_get_upper_bound(pdf->pdf->cores[ii]->funcs[0]);
    }

    double * mean = probability_density_mean(pdf);

    if (pdf->transform  == 1){

        dimpdf = pdf->lt->dimout;   
        var = calloc_double(dimpdf);
        double * offset = calloc_double(dimft);
        for (ii = 0; ii < dimpdf; ii++){
            offset[0] = pdf->lt->b[ii] - mean[ii];
            struct FunctionTrain * ftleft = 
                function_train_linear2(LINELM,NULL,dimft,bds,
                                       pdf->lt->A+ii,dimpdf,offset,1,NULL);
            
            struct FunctionTrain * temp = function_train_product(ftleft,ftleft);
            var[ii] = function_train_inner(temp,pdf->pdf);
            function_train_free(temp);
            function_train_free(ftleft);
        }
        free(offset); offset=NULL;
    }
    else{
        var = calloc_double(dimpdf);

        struct FunctionTrain * ftc = function_train_copy(pdf->pdf);
        struct FunctionTrain * xvals = function_train_alloc(dimft);
        for (ii = 0; ii < dimpdf; ii++){
            double lb = generic_function_get_lower_bound(ftc->cores[ii]->funcs[0]);
            double ub = generic_function_get_upper_bound(ftc->cores[ii]->funcs[0]);
            xvals->cores[ii] = qmarray_alloc(1,1);
            xvals->cores[ii]->funcs[0] = 
                generic_function_linear(1.0,-mean[ii],LINELM,NULL,lb,ub,NULL);
        }

        for (ii = 0; ii < dimpdf; ii++){
            struct Qmarray * temp = qmarray_kron(xvals->cores[ii],pdf->pdf->cores[ii]);

            qmarray_free(ftc->cores[ii]);
            ftc->cores[ii] = qmarray_kron(xvals->cores[ii],temp);
            qmarray_free(temp); temp = NULL;
            var[ii] = function_train_integrate(ftc);

            qmarray_free(ftc->cores[ii]);
            //maybe dont have to copy bc freeing later
            ftc->cores[ii] = qmarray_copy(pdf->pdf->cores[ii]); 
        }
        function_train_free(xvals); xvals = NULL;
        function_train_free(ftc); ftc = NULL;
    }
    
    bounding_box_free(bds);
    free(mean); mean = NULL;
    return var;
}

/***********************************************************//**
    Get the lower bounds of the underlying (transformed) pdf

    \param[in] pdf - pdf whose lower bounds to obtain

    \return  lb - array of lower bounds
***************************************************************/
double * probability_density_lb_base(struct ProbabilityDensity * pdf)
{
    size_t dim = pdf->pdf->dim;
    double * lb = calloc_double(dim);
    size_t ii;
    for (ii = 0; ii < dim; ii++){
        lb[ii] = generic_function_get_lower_bound(
                    pdf->pdf->cores[ii]->funcs[0]);
    }
    return lb;
}

/***********************************************************//**
    Get the upper bounds of the underlying (transformed) pdf

    \param[in] pdf - pdf whose lower bounds to obtain

    \return  ub - array of upper bounds
***************************************************************/
double * probability_density_ub_base(struct ProbabilityDensity * pdf)
{
    size_t dim = pdf->pdf->dim;
    double * ub = calloc_double(dim);
    size_t ii;
    for (ii = 0; ii < dim; ii++){
        ub[ii] = generic_function_get_upper_bound(
                    pdf->pdf->cores[ii]->funcs[0]);
    }
    return ub;
}

struct LikeParamDataCouple {
    double * param;
    double * data;
};

double like_poisson_helper(double x, size_t dim, void * args)
{
    struct LikeParamDataCouple * pdc  = args;
    double out = pdc->data[dim] * x - 
                    pdc->param[0] * exp(x);
    //printf("out=%G\n",out);
    out = exp(out);
    return out;
}

/***********************************************************//**
    Allocate a likelihood function

    \param[in] datadim  - dimension of the data
    \param[in] data     - data
    \param[in] paramdim - dimension of extra parameters
    \param[in] param    - parameters
    \param[in] inputdim - dimension of the input
    \param[in] lb       - lower bound of each dimension
    \param[in] ub       - upper bound of each dimension
    \param[in] type     - likelihood type

    \return like - likelihood function
***************************************************************/
struct Likelihood *
likelihood_alloc(size_t datadim, double * data, size_t paramdim, 
                 double * param, size_t inputdim, 
                 double lb, double ub, enum likelihood_type type)
{

    struct Likelihood * like = malloc(sizeof(struct Likelihood));
    if (like == NULL){
        fprintf(stderr, "Failed to allocate Likelihood\n");
        exit(1);
    }
    
    like->datadim = datadim;
    like->paramdim = paramdim;
    like->inputdim = inputdim;
    like->type = type;
    if (type == POISSON_LIKE){
        enum poly_type ptype = LEGENDRE;
        size_t start_num = 7;
        size_t c_check = 2;
        struct OpeAdaptOpts ao;
        ao.start_num = start_num;
        ao.coeffs_check = c_check;
        ao.tol = 1e-10;
        
        struct FtApproxArgs * fapp = 
                ft_approx_args_createpoly(inputdim,&ptype,&ao);
    
        struct BoundingBox * bds = bounding_box_init_std(inputdim);
        size_t ii; 
        for (ii = 0; ii < inputdim; ii++){
            bds->lb[ii] = lb; 
            bds->ub[ii] = ub;
        }

        struct LikeParamDataCouple args = {param,data};
    
        //printf("compute here\n");
        like->like = function_train_rankone(inputdim,like_poisson_helper,
                                                &args,bds,fapp);
        //printf("compute there\n");
        like->loglike = 0;

        bounding_box_free(bds);
        ft_approx_args_free(fapp);
    }
    else{
        like->like = NULL;
    }
    return like;
}

/***********************************************************//**
    Allocate and initialize a Gaussian likelihood

    \param[in] noise_type - type of noise
    \param[in] ndata      - number of data points
    \param[in] datadim    - dimension of the data
    \param[in] data       - data (ndata * datadim,)
    \param[in] paramdim   - dimension of parameters 
    \param[in] param      - parameters defining noise
    \param[in] inputdim   - dimension of the input
    \param[in] meanfunc   - function array evaluate the mean of the Gaussian
                            (ndata * datadim,)

    \return like - likelihood function
    
    \note 
        noise_type = 0 
            - paramdim should be = 1, param indicates noise std)
        noise_type = 1 
            - paramdim should be = datadim, 
              param specifies diff noise std for each dimension)
***************************************************************/
struct Likelihood * likelihood_gaussian(int noise_type, size_t ndata,
    size_t datadim, double * data, size_t paramdim, double * param,
    size_t inputdim, struct FT1DArray * meanfunc)
{

    struct Likelihood * like = malloc(sizeof(struct Likelihood));
    if (like == NULL){
        fprintf(stderr, "Failed to allocate Likelihood\n");
        exit(1);
    }
    
    like->type = GAUSSIAN_LIKE;
    like->datadim = datadim;
    like->paramdim = paramdim;
    like->inputdim = inputdim;
    
    struct FunctionTrain * temp = NULL;
    struct FunctionTrain * temp2 = NULL;

    size_t ii,jj;
    like->logextra = 0.0;
    for (jj = 0; jj < ndata; jj++){
//        printf("jj = %zu\n",jj);
        struct FT1DArray * moff = ft1d_array_alloc(datadim);
        for (ii = 0; ii < datadim; ii++){
//            printf("ii=%zu, data=%G \n",ii,data[jj*datadim+ii]);
            moff->ft[ii] = function_train_afpb(1.0,-data[jj*datadim+ii],
                            meanfunc->ft[jj*datadim+ii],1e-12);
        }
//        printf("got moff\n");

        if (noise_type == 0){
            assert (paramdim == 1);
            double * coeff = darray_val(datadim,-0.5/pow(param[0],2));
            temp = ft1d_array_sum_prod(datadim,coeff,moff,moff,1e-10);
            free(coeff); coeff = NULL;
            //printf("here! %G \n", log(pow(pow(pow(param[0],2),datadim),-0.5)));
            like->logextra += log(pow(2.0*M_PI,- ((double) datadim) /2.0) *
                                  pow(pow(param[0] * param[0],datadim),-0.5));
            //printf("datadim = %zu\n",datadim);
            //printf("log extra is now = %G\n", like->logextra);
        }
        else if (noise_type == 1){
            assert (paramdim == datadim);
            double * coeff = calloc_double(paramdim);

            double tempdet = 1.0;
            for (ii = 0; ii < paramdim; ii++){
                coeff[ii] = -0.5 / pow(param[ii],2);
                tempdet *= pow(param[ii],2);
            }
            like->logextra += log( pow(2*M_PI,-(double) datadim/2.0) * 
                                   pow(tempdet,-0.5));
            temp = ft1d_array_sum_prod(datadim,coeff,moff,moff,1e-10);
            free(coeff); coeff = NULL;
        }
        else{
            fprintf(stderr,
                    "Noise type (%d) not available for Gaussian likelihood",
                    noise_type); 
            exit(1);
        }
        if (jj == 0){
            like->like = function_train_copy(temp);
        }
        else{
            temp2 = function_train_sum(temp,like->like);
            function_train_free(like->like); like->like = NULL;

            like->like = function_train_round(temp2,1e-10);
            function_train_free(temp2); temp2 = NULL;

        }
        function_train_free(temp); temp = NULL;
        ft1d_array_free(moff); moff = NULL;
       
    }
    like->loglike = 1;


    return like;
}

/***********************************************************//**
    Gaussian Likelihood for Linear Regression to infer parameters a_i in

    \f$ f(x) = a_0 + \sum_{i=1}^d a_i x_i \f$

    \param[in] dim        - dimension of *x*
    \param[in] N          - number of data points
    \param[in] data       - array of one dimensional data points
    \param[in] covariates - (dim * N) array of *x* locations
    \param[in] noise      - standard deviation of noise
    \param[in] bds        - boundaries for inference of coefficients *a_i*

    \return like - likelihood function
***************************************************************/
struct Likelihood * likelihood_linear_regression(size_t dim, size_t N, 
    double * data, double * covariates, double noise, struct BoundingBox * bds)
{
//    enum poly_type ptype = LEGENDRE;
    struct FT1DArray * meanfunc = ft1d_array_alloc(N);
    size_t ii;
    double * pt = calloc_double(dim+1);
    pt[0] = 1.0;
    for (ii = 0; ii < N; ii++){
        memmove(pt+1,covariates + ii*dim,dim*sizeof(double));    
//        meanfunc->ft[ii] = function_train_linear(POLYNOMIAL,&ptype,dim+1,bds,pt,NULL);
        meanfunc->ft[ii] = function_train_linear(LINELM,NULL,dim+1,bds,pt,NULL);
    }

    struct Likelihood * like = 
            likelihood_gaussian(0,N,1,data,1,&noise,dim+1,meanfunc);
    
    free(pt);
    pt = NULL;
    ft1d_array_free(meanfunc); 
    meanfunc = NULL;
    return like;
}

/***********************************************************//**
    Free memory allocated to a likelihood function

    \param like [inout] - likelihood function
***************************************************************/
void likelihood_free(struct Likelihood * like)
{
    if (like != NULL){
        function_train_free(like->like); like->like = NULL;
        free(like); like = NULL;
    }
}

struct LikeLinCouple
{
    struct Likelihood * like;
    struct LinearTransform * lt;
};

double like_transform_helper(double * x, void * args)
{   
    struct LikeLinCouple * lc = args;
    double * xnew = linear_transform_apply(lc->lt->dimout, lc->lt->dimin, 
                lc->lt->A,x,lc->lt->b);
    
    //*
    //printf("xnew = ");
    //dprint(lc->lt->dimin, x);
    //*/
    double out = 0.0;
    int outofbounds = 0;
    size_t ii;
    for (ii = 0; ii < lc->lt->dimout; ii++){
        double lb = generic_function_get_lower_bound(lc->like->like->cores[ii]->funcs[0]);
        if (xnew[ii] < lb){
            xnew[ii] = lb;
            //out = 0.0;
            //outofbounds = 1;
            //break;
        }
        double ub = generic_function_get_upper_bound(lc->like->like->cores[ii]->funcs[0]);
        if (xnew[ii] > ub){
            xnew[ii] = ub;
            //out = 0.0;
            //outofbounds = 1;
            //break;
        }
    }
    
    if (outofbounds == 0){
        out = function_train_eval(lc->like->like,xnew);
    }

    //printf("xnew = ");
    //dprint(lc->lt->dimout, xnew);

    //printf("out=%G\n",out);

    free(xnew); xnew = NULL;
    return out;
}

/***********************************************************//**
    Transform the likelihood using a linear variable transformation

    \param[in] like - likelihood 
    \param[in] lt   - linear transformation
    \param[in] bds  - domain

    \return newlike - new likelihood
***************************************************************/
struct Likelihood *
likelihood_transform(struct Likelihood * like, struct LinearTransform * lt,
                     struct BoundingBox * bds)
{

    struct LikeLinCouple lc = {like,lt};
        

//
    size_t dim = lt->dimin;
    size_t init_rank = 2;
    double hmin = 1e-2;
    double delta = 1e-5;
    double round_tol = 1e-3;
    double cross_tol = 1e-5;
    struct C3Approx * c3a = c3approx_create(CROSS,dim,bds->lb,bds->ub);
    c3approx_init_lin_elem(c3a);
    c3approx_set_lin_elem_delta(c3a,delta);
    c3approx_set_lin_elem_hmin(c3a,hmin);
    c3approx_init_cross(c3a,init_rank,0);
    c3approx_set_round_tol(c3a,round_tol);
    c3approx_set_cross_tol(c3a,cross_tol);

    /* struct FtCrossArgs temp; */
    /* struct FtApproxArgs * fapp = NULL; */
    //size_t ii;
    /* size_t * init_ranks = calloc_size_t(dim+1); */
    /* for (ii = 0; ii < dim ;ii++){ */
    /*     init_ranks[ii] = init_rank; */
    /* } */
    /* init_ranks[0] = 1; */
    /* init_ranks[dim] = 1; */
        
    /* struct OpeAdaptOpts aopts; */
    /* aopts.start_num = 10; */
    /* aopts.coeffs_check = 0; */
    /* aopts.tol = 1e-5; */

    /* enum poly_type ptype = LEGENDRE; */
    /* fapp = ft_approx_args_createpoly(dim,&ptype,&aopts); */

    /* temp.dim = dim; */
    /* temp.ranks = init_ranks; */
    /* temp.epsilon = 1e-5; */
    /* temp.maxiter = 10; */
    /* temp.verbose = 2; */
    
    /* temp.epsround = 100.0*DBL_EPSILON; */
    /* temp.kickrank = 4; */
    /* temp.maxiteradapt = 5; */
    
    struct Likelihood * newlike = malloc(sizeof(struct Likelihood));
    if (newlike == NULL){
        fprintf(stderr, "Failed to allocate Likelihood\n");
        exit(1);
    }

    newlike->datadim = like->datadim;
    newlike->paramdim = 0;
    newlike->inputdim = dim;
    newlike->type = GENERIC_LIKE;
    //printf("here!!!\n");
    /* newlike->like = function_train_cross(like_transform_helper,&lc,bds,NULL,&temp,fapp); */
    newlike->like = c3approx_do_cross(c3a,like_transform_helper,&lc);
    //printf("there!!!\n");

    /* free(init_ranks); init_ranks = NULL; */
    /* ft_approx_args_free(fapp); fapp = NULL; */
            
    return newlike; 
}

/***********************************************************//**
    Compute the gradient of the log posterior from bayes rule
    
    \param[in] x    - location at which to obtain the gradient log posterior
    \param[in] args - BayesRule structure holding likelihood and prior

    \return out - gradient of log posterior
***************************************************************/
double * bayes_rule_log_gradient(double * x, void * args){
    
    struct BayesRule * br = args;
    size_t dim = br->like->inputdim;


    double * gradp = probability_density_log_gradient_eval(br->prior,x);
    

    struct FT1DArray * gradl = function_train_gradient(br->like->like);
    double * gradl_eval = ft1d_array_eval(gradl,x);


    double * out = calloc_double(dim);
    size_t ii;
    if (br->like->loglike == 0){
        double likeval = function_train_eval(br->like->like,x);
        for (ii = 0; ii < dim; ii++){
            out[ii] = 1.0/likeval * gradl_eval[ii] + gradp[ii];
        }
    }
    else{
        for (ii = 0; ii < dim; ii++){
            out[ii] = gradl_eval[ii] + gradp[ii];
        }
    }


    ft1d_array_free(gradl); gradl = NULL;
    free(gradl_eval); gradl_eval = NULL;
    free(gradp); gradp = NULL;

    return out;
}

/***********************************************************//**
    Compute the gradient of the negative log posterior from bayes rule
    
    \param[in] x    - location at which to obtain the hessian of log posterior
    \param[in] args - BayesRule structure holding likelihood and prior

    \return out - gradient of negative log posterior
***************************************************************/
double * bayes_rule_log_gradient_negative(double * x, void * args){
    
    struct BayesRule * br = args;
    size_t dim = br->like->inputdim;

    double * grad = bayes_rule_log_gradient(x,args);
    size_t ii;
    for (ii=0; ii < dim; ii++){
        grad[ii] *= -1.0;
    }
    return grad;
}

/***********************************************************//**
    Compute the hessian of the log posterior from bayes rule
    
    \param[in] x    - location at which to obtain the hessian of log posterior
    \param[in] args - BayesRule structure holding likelihood and prior

    \return out - hessian of log posterior
***************************************************************/
double * bayes_rule_log_hessian(double * x, void * args){
    
    struct BayesRule * br = args;
    size_t dim = br->like->inputdim;

    double * hessp = probability_density_log_hessian_eval(br->prior,x);


    struct FT1DArray * hessl = function_train_hessian(br->like->like);
    double * hessl_eval = ft1d_array_eval(hessl,x);


    double * out = calloc_double(dim*dim);
    size_t ii,jj;
    if (br->like->loglike == 0){

        double likeval = function_train_eval(br->like->like,x);
        double likeval_squared = pow(likeval,2);

        struct FT1DArray * gradl = function_train_gradient(br->like->like);
        double * gradl_eval = ft1d_array_eval(gradl,x);
        for (ii = 0; ii < dim; ii++){
            for (jj = 0; jj < dim; jj++){
                out[ii*dim+jj] = 
                    -1.0/likeval_squared * gradl_eval[ii] * gradl_eval[jj] +
                     1.0/likeval * hessl_eval[ii*dim+jj] + hessp[ii*dim+jj];
            }
        }

        ft1d_array_free(gradl); gradl = NULL;
        free(gradl_eval); gradl_eval = NULL;
    }
    else{
        for (ii = 0; ii < dim; ii++){
            for (jj = 0; jj < dim; jj++){
                out[ii*dim+jj] = hessl_eval[ii*dim+jj] + hessp[ii*dim+jj];
            }
        }
    }

    ft1d_array_free(hessl); hessl = NULL;
    free(hessl_eval); hessl_eval = NULL;

    free(hessp); hessp = NULL;

    return out;
}

/***********************************************************//**
    Compute the hessian of the negative log posterior from bayes rule
    
    \param x [in] - location at which to obtain the hessian of log posterior
    \param args [in] - BayesRule structure holding likelihood and prior

    \return out - hessian of negative log posterior
***************************************************************/
double * bayes_rule_log_hessian_negative(double * x, void * args){
    
    struct BayesRule * br = args;
    size_t dim = br->like->inputdim;

    double * hess = bayes_rule_log_hessian(x,args);
    size_t ii,jj;
    for (ii=0; ii < dim; ii++){
        for (jj=0; jj < dim; jj++){
            hess[ii*dim+jj] *= -1.0;
        }
    }
    return hess;
}

/***********************************************************//**
    Compute the Laplace approximation to Bayes rule

    \param[in] br - Bayes Rule structure holding likelihood and prior

    \return posterior - posterior distribution
***************************************************************/
struct ProbabilityDensity * bayes_rule_laplace(struct BayesRule * br)
{
    double * start = probability_density_mean(br->prior);
    size_t dim = br->like->inputdim;
    struct ProbabilityDensity * pdf = 
        probability_density_laplace(bayes_rule_log_gradient_negative,
                                    bayes_rule_log_hessian_negative,
                                    br, dim, start);
    
    free(start); start = NULL;
    return pdf;
}

struct BrTrans
{
    struct BayesRule * br;
    struct LinearTransform * lt;
    double mult;
};

double bayes_rule_evaluate(double * x, void * arg)
{
    struct BrTrans * brt = arg;

    //printf("x in = \n");
    //dprint(brt->lt->dimin, x);

    double * xtemp = linear_transform_apply(brt->lt->dimout,
                brt->lt->dimin, brt->lt->A, x, brt->lt->b);
    
    /* printf("xtemp = \n"); */
    /* dprint(brt->lt->dimin, xtemp); */

    double prior_val = probability_density_eval(brt->br->prior,xtemp);
    double like_val = function_train_eval(brt->br->like->like,xtemp);
    //printf("likeval first = %G\n",like_val);
    if (brt->br->like->loglike == 1){
        like_val = exp(like_val + brt->br->like->logextra);
    }

    /* printf("priorval = %G\n",prior_val); */
    /* printf("likeval = %G\n",like_val); */
    /* printf("brt->mult = %G\n",brt->mult); */
    
    free(xtemp); xtemp = NULL;
    double out = prior_val * like_val * brt->mult;
    
    /* printf("out = %G\n",out); */

    return out;
}

double bayes_rule_log(double * x, void * arg)
{
    struct BrTrans * brt = arg;

    //printf("x in = \n");
    //dprint(brt->lt->dimin, x);

    double * xtemp = linear_transform_apply(brt->lt->dimout,
                brt->lt->dimin, brt->lt->A, x, brt->lt->b);
    
    //printf("xtemp = \n");
    //dprint(brt->lt->dimin, xtemp);

    double prior_val = log(probability_density_eval(brt->br->prior,xtemp));
    double like_val = function_train_eval(brt->br->like->like,xtemp) 
                        + brt->br->like->logextra;
    

    //printf("priorval = %G\n",prior_val);
    //printf("likeval = %G\n",like_val);
    
    free(xtemp); xtemp = NULL;
    double out = 1.0/((prior_val + like_val)/800);
    
    //printf("out = %G\n",out);

    return out;
}

/***********************************************************//**
    Compute the posterior from Bayes Rule

    \param br [in] - BayesRule structure holding likelihood and prior

    \return posterior - posterior distribution
***************************************************************/
struct ProbabilityDensity * bayes_rule_compute(struct BayesRule * br)
{
    size_t dim = br->like->inputdim;

    // laplace approximation
    struct ProbabilityDensity * lap = bayes_rule_laplace(br);
    
    // Initialize transform with laplace approximation
    struct ProbabilityDensity * posterior = probability_density_alloc();
    posterior->type = GENERAL;
    posterior->transform = 1;
    posterior->lt = linear_transform_alloc(dim,dim,NULL,NULL,0.0);

    memmove(posterior->lt->b, lap->lt->b, sizeof(double) * dim);
    memmove(posterior->lt->A, lap->lt->A, dim*dim*sizeof(double));
    posterior->lt->mt = lap->lt->mt;

    /* printf("mean = \n"); */
    /* dprint(dim,posterior->lt->b); */
        
    struct BrTrans brt;
    brt.br = br;
    brt.lt = posterior->lt;
    brt.mult = 1.0;
    
    //double normalize;
    double prior_val = log(probability_density_eval(br->prior,lap->lt->b));
    double like_val = function_train_eval(br->like->like,lap->lt->b);
    if (br->like->loglike == 0){
        like_val = log(like_val);
    }
    /* printf("log prior at map = %G\n",prior_val); */
    /* printf("log likelihood at map = %G\n",like_val); */
    br->like->logextra = -(like_val+prior_val);
    /* printf("log extra = %G\n",br->like->logextra); */
    double val_at_map = bayes_rule_evaluate(lap->lt->b,&brt);
    /* printf("the value at the map is %G\n",val_at_map); */
    brt.mult = 0.5/val_at_map;
    val_at_map = bayes_rule_evaluate(lap->lt->b,&brt);
    /* printf("the value at the map 2 is %G\n",val_at_map); */
    
    /* printf("BAYESRUL IS BROKE!!!\n"); */
    /* exit(1); */
    //
    struct BoundingBox * bds = bounding_box_init(dim,-6.0,6.0);
    double hmin = 1e-2;
    double delta = 1e-5;
    size_t init_rank = 3;
    double round_tol = 1e-3;
    double cross_tol = 1e-8;
    struct C3Approx * c3a = c3approx_create(CROSS,dim,bds->lb,bds->ub);
    c3approx_init_lin_elem(c3a);
    c3approx_set_lin_elem_delta(c3a,delta);
    c3approx_set_lin_elem_hmin(c3a,hmin);
    c3approx_init_cross(c3a,init_rank,0);
    c3approx_set_round_tol(c3a,round_tol);
    c3approx_set_cross_tol(c3a,cross_tol);
    
    /* struct FunctionTrain * roughpdf = c3approx_do_cross(c3a,bayes_rule_log,&brt); */
    
    /* double * temp = calloc_double(dim); */
    /* double postval = function_train_eval(roughpdf,temp); */
    /* double postval_check = bayes_rule_log(temp,&brt); */
    /* free(temp); temp = NULL; */
    /* printf("log post at map = %G, %G\n", 1.0/postval,1.0/postval_check); */

    /* double normalize = function_train_integrate(roughpdf); */
    /* printf("normalizing constant is %G \n", normalize); */
    /* printf("integral of log %G \n", normalize); */

    /* brt.mult = fabs(1.0/normalize); */
    /* function_train_free(roughpdf); roughpdf = NULL; */

    /* exit(1); */
    /* posterior->pdf =  */
    /*     function_train_cross(bayes_rule_evaluate, &brt, bds, NULL,NULL,NULL); */
    posterior->pdf = c3approx_do_cross(c3a,bayes_rule_evaluate,&brt);

    double normalize = function_train_integrate(posterior->pdf);
    /* printf("second normalizing constant is %G \n", normalize); */
    function_train_scale(posterior->pdf,1.0/normalize);

//    exit(1);

    bounding_box_free(bds); bds = NULL;
    probability_density_free(lap); lap = NULL;
    return posterior;
}


