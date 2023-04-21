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
#include <assert.h>

#include "array.h"
#include "stringmanip.h"

double * calloc_double(const size_t N){
    
    double * arrp; 
    if (NULL == ( arrp = calloc(N,sizeof(double)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    return arrp;
}

double ** malloc_dd(const size_t N){
    double ** x;
    if (NULL == ( x = malloc(N * sizeof(double *)))){
        fprintf(stderr, "failed to allocate memory for double pointer.\n");
        exit(1);
    }
    return x;
}

int * calloc_int(const size_t N){

    int * arrp; 
    if (NULL == ( arrp = calloc(N,sizeof(int)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    return arrp;
}

size_t * calloc_size_t(const size_t N){

    size_t * arrp = NULL;; 
    if (NULL == ( arrp = calloc(N,sizeof(size_t)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }
    return arrp;
}

void copy_dd(size_t N, size_t d, double ** dest, double ** src)
{
    size_t ii;
    for (ii = 0; ii < N; ii++){
        memmove(dest[ii],src[ii],d*sizeof(double));
    }
}

void free_dd(size_t N, double ** arr){
    if (arr != NULL){
        size_t ii;
        for (ii = 0; ii < N ;ii++){
            free(arr[ii]); arr[ii] = NULL;
        }
        free(arr); arr = NULL;
    }
}

/*************************************************************//**
    Print a double array

    \param[in] N   - Number of elements in array
    \param[in] arr - array to print
****************************************************************/
void 
dprint(const size_t N, const double * arr){
    size_t ii = 0;
    for (ii = 0; ii < N; ii++) printf("%3.5E ",arr[ii]);
    printf("\n");
}

/*************************************************************//**
    Print a two dimensional double array stored as a double pointer in Row-Major (C) order

    \param[in] N   - number of rows
    \param[in] M   - number of cols
    \param[in] arr - array to print
****************************************************************/
void dprint2d(const size_t N, const size_t M, const double * arr){
    size_t ii, jj;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < M; jj++){
            printf("%3.2f ", arr[ii*M+jj]);
        }
        printf("\n");
    }
}

/*************************************************************//**
    Print a two dimensional double array stored as a
    double pointer in col-Major (C) order

    \param N [in] - Number of rows
    \param M [in] - Number of cols
    \param arr [in] - array to print
****************************************************************/
void dprint2d_col(const size_t N, const size_t M, const double * arr){
    size_t ii, jj;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < M; jj++){
            printf("%3.2f ", arr[ii + jj*N]);
        }
        printf("\n");
    }
}

void dprint2dd(const size_t N, const size_t M, double ** arr){
    
    size_t ii;
    for (ii = 0; ii < N; ii++){
        dprint(M,arr[ii]);
    }
}

/*************************************************************//**
    Print an integer array
        
    \param[in] N   - Number of elements
    \param[in] arr - array to print
****************************************************************/
void iprint(size_t N, const int * arr){
    size_t ii = 0;
    for (ii = 0; ii < N; ii++) printf("%d ",arr[ii]);
    printf("\n");
}

/*************************************************************//**
    Print a size_t array
        
    \param[in] N   - Number of elements
    \param[in] arr - array to print
****************************************************************/
void iprint_sz(size_t N, const size_t * arr){
    size_t ii = 0;
    for (ii = 0; ii < N; ii++) printf("%zu ",arr[ii]);
    printf("\n");
}

/*************************************************************//**
    Create a zero array of doubles
        
    \param N [in] - Number of elements

    \return arrp - pointer to array
****************************************************************/
double * dzeros(const size_t N)
{
    double * arrp = calloc_double(N);
    return arrp;
}

/*************************************************************//**
    Create a zero array of integers

    \param N [in] - Number of elements

    \return arrp - pointer to array
****************************************************************/
int * izeros(const size_t N)
{
    int * arrp = calloc_int(N);
    return arrp;
}

/*************************************************************//**
    Create a double array of ones
        
    \param N [in] - Number of elements

    \return arrp - pointer to array
****************************************************************/
double * dones(const size_t N)
{
    double * arrp = calloc_double(N);
    size_t ii = 0;
    for (ii = 0; ii < N; ii++) arrp[ii]=1.0;
    return arrp;
}

/*************************************************************//**
    Create a pseudorandom double array with each element between [-1,1]
        
    \param N [in] - Number of elements

    \return arrp - pointer to array
****************************************************************/
double * drandu(const size_t N)
{
    double * arrp = calloc_double(N);
    size_t ii = 0;
    for (ii = 0; ii < N; ii++) arrp[ii]=randu()*2.0-1.0;
    return arrp;
}

/*************************************************************//**
    Create a double array consisting of the same value in each element
        
    \param N [in] - Number of elements
    \param val [in] -  value of each element

    \return arrp - pointer to array
****************************************************************/
double * darray_val(const size_t N, double val)
{
    double * arrp = calloc_double(N);
    size_t ii = 0;
    for (ii = 0; ii < N; ii++) arrp[ii]=val;
    return arrp;
}

/*************************************************************//**
    Create a double array of *N* linearly spaced 
    pointers between *min* and *max* (inclusive)
        
    \param min [in] - lower bound
    \param max [in] - upper bound
    \param N [in]   - Number of elements

    \return arrp - pointer to array
****************************************************************/
double * linspace(const double min, const double max, const size_t N){

    double * arrp = calloc_double(N);
    double dx = (max-min)/(double) (N-1);
    size_t ii;
    for (ii = 0; ii < N-1; ii++){
        arrp[ii] = (double)ii*dx + min;
    }
    arrp[N-1] = max;
    return arrp;
}

void dd_row_linspace(double ** dd, size_t ii, double lb, double ub, size_t N)
{
    dd[ii] = linspace(lb,ub,N);
}

/*************************************************************//**
    Create a double array of *N* log spaced
    pointers between 10^min and 10^ max
        
    \param min [in] - lower bound
    \param max [in] - upper bound
    \param N [in]   - Number of elements

    \return arrp - pointer to array
****************************************************************/
double * logspace(int min, int max, const size_t N){
    
    double mmin = (double) min;
    double mmax = (double) max;
    double * arrp = calloc_double(N);
    double dx = (mmax-mmin)/(double) (N-1);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        arrp[ii] = pow(10, (double)ii*dx + mmin );
    }
    return arrp;
}


/*************************************************************//**
    Create an array of points obtained by starting at
    *start* and taking steps of size *step* until
    a value greater than equal to *stop* is obtained
        
    \param start [in] - starting value
    \param  stop [in] - upper bound on ending value
    \param step [in] - step size
    \param N [out]  - size of array

    \return arrp - pointer to array
****************************************************************/
double * 
arange(const double start, const double stop, const double step, size_t * N){
    
    *N = (size_t) lrint(floor((stop-start)/step)) + 1;
    size_t ii;
    double * arrp = calloc_double(*N);
    for (ii = 0; ii < *N; ii++){
        arrp[ii] = start + step*(double) ii;
    }
    return arrp;
}

/*************************************************************//**
    Create a diagonal "matrix" from array
        
    \param N [in]   - Number of elements
    \param arr [in] - upper bound

    \return arrp - pointer to array
****************************************************************/
double * diag(const size_t N, double * arr){

    double * arrp = calloc_double(N*N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        arrp[ii*N+ii] = arr[ii];
    }
    return arrp;
}

/*************************************************************//**
    Concatenate the columns of two arrays
        
    \param rows [in]  - number of rows
    \param cols1 [in] - columns of first array
    \param cols2 [in] - columns of second array
    \param arr1 [in] - first array
    \param arr2 [in] - second array

    \return arrp - pointer to array
****************************************************************/
double * 
dconcat_cols(size_t rows, size_t cols1, size_t cols2, 
        double * arr1, double * arr2)
{

    double * arrp = calloc_double(rows*(cols1+cols2));
    memmove(arrp,arr1, rows*cols1*sizeof(double));
    memmove(arrp+rows*cols1,arr2,rows*cols2*sizeof(double));
    return arrp;
}


/*************************************************************//**
    Compute the product of the elements of a double array
        
    \param N [in] - size of array
    \param in [in] - array

    \return prod - Product of elements of *in*
****************************************************************/
double dprod(const size_t N, const double * in)
{
    double prod = 1.0;
    size_t i;
    for (i=0; i<N; i++) prod = prod * in[i];
    return prod;
}

/*************************************************************//**
    Compute the product of the elements of an integer array
        
    \param N [in] - size of array
    \param in [in] - array

    \return prod - product of elements of *in*
****************************************************************/
int iprod(const size_t N, const int * in)
{
    int prod = 1;
    size_t i;
    for (i=0; i<N; i++) prod = prod * in[i];
    return prod;
}

/*************************************************************//**
    Compute the product of the elements of an size_t array
        
    \param N [in] - size of array
    \param in [in] - array

    \return prod - product of elements of *in*
****************************************************************/
size_t iprod_sz(const size_t N, const size_t * in)
{
    size_t prod = 1;
    size_t i;
    for (i=0; i<N; i++) prod = prod * in[i];
    return prod;
}




#define FRAC_MAX 9223372036854775807LL // 2**63-1 

/// @private
struct dbl_packed
{
    int exp;
    long long frac;
};

void pack(double x, struct dbl_packed * r)
{
    double xf = fabs(frexp(x,&r->exp))-0.5;
    if (xf < 0.0){
        r->frac = 0;
        return;
    }
    r->frac = 1 + (long long)(xf * 2.0  * (FRAC_MAX-1));
    if (x < 0.0){
        r->frac = -r->frac;
    }
}

double unpack(const struct dbl_packed * p)
{
    double xf, x;
    if (p->frac == 0){
        return 0.0;
    }
    xf = ((double)(llabs(p->frac)-1) / (FRAC_MAX-1)) /2.0;
    x = ldexp(xf + 0.5,p->exp);
    if (p->frac < 0){
        x = -x;
    }
    return x;
}

char * serialize_double_packed(double x)
{
    struct dbl_packed dbl;
    pack(x,&dbl);

    size_t nBytes = sizeof(int) + sizeof(long long);
    char * mem = malloc(nBytes);
    memcpy(mem,&(dbl.exp),sizeof(int));
    memcpy(mem+sizeof(int),&(dbl.frac),sizeof(long long));
    return mem;
}

double deserialize_double_packed(char * ser)
{
    struct dbl_packed dbl;
    memcpy(&(dbl.exp),ser,sizeof(int));
    /* ser = ser + sizeof(int); */
    memcpy(&(dbl.frac),ser+sizeof(int),sizeof(long long));

    double x = unpack(&dbl);
    return x;
}

char * serialize_double_arr_packed(size_t n, const double * x)
{
    size_t size_of_one_double = sizeof(int) + sizeof(long long);
    /* char * mem = malloc(n * (size_of_one_double)+sizeof(char)); */
    char * mem = malloc(n * 20 * sizeof(char));
    for (size_t ii = 0; ii < n*10; ii++){
        mem[ii] = '\0';
    }
    printf("n = %zu\n",n);
    size_t onbyte = 0;
    for (size_t ii = 0; ii < n; ii++){
        char * temp = serialize_double_packed(x[ii]);
        memcpy(mem+onbyte,temp,size_of_one_double);
        free(temp); temp = NULL;
        onbyte += size_of_one_double;
        printf("onbyte = %zu\n",ii);
    }

    /* mem[onbyte] = '\0'; */
    /* char t = '\0'; */
    /* memcpy(mem+onbyte,&t,sizeof(char)); */
    /* printf("%s\n",mem); */
    return mem;
}



/*************************************************************//**
    Serialize a double to string
            
    \param[in] x - double to serialize

    \return buffer - string of values
****************************************************************/
char * serialize_double_to_text(double x)
{
    size_t N = 40;
    char * buffer = NULL;
    if (NULL == (buffer = malloc(N * sizeof(char)))){
        fprintf(stderr, "failed to allocate memory.\n");
        exit(1);
    }

    int exp;
    double res = frexp(x,&exp);
               
    //printf("res=%f,exp=%d\n",res,exp);
    int cx;
    /* cx = snprintf(buffer, N, "%0.10f:%d", res,exp); */ // worked before
    cx = snprintf(buffer, N, "%3.14E:%d", res,exp);
    /* cx = snprintf(buffer, N, "%10.20f:%d", res,exp); */
    //printf("after = %s\n",buffer);
    char * temp2 = NULL;
    if (cx > (int)N){
       //printf("too much\n");
        temp2 =  realloc(buffer, (size_t) cx * sizeof(char));
        if (temp2 !=NULL ){
            buffer = temp2;
        }
        else{
            fprintf(stderr, "failed to allocate memory.\n");
            exit(1);
        }
        N = (size_t)cx;
        /* snprintf(buffer, N, "%0.20f:%d", res,exp); */
        snprintf(buffer, N, "%3.14E:%d", res,exp);
    }
    else if (cx < 0){
        fprintf(stderr, "some other error writing double \n");
        exit(1);
    }

    return buffer;
}

/*************************************************************//**
    De-Serialize text to a double

    \param num [in] - double to deserialize
    \return val - deserialized value
****************************************************************/
double deserialize_double_from_text(char * num)
{
    char * pch = bite_string(num,':');
    double f = atof(pch);
    int exp = atoi(num);
    double val = ldexp(f,exp);

    free(pch);
    return val;
}

/*************************************************************//**
    Serialize a double array
 
    \param[in] N      - number of elements in the array
    \param[in] array  - array to serialize
 
    \return buffer - string of values
****************************************************************/
char * serialize_darray_to_text(size_t N, const double * array)
{
    
    char * buffer = NULL;
    char * temp = NULL;
    //char temp[] = "HELLO";
    
    size_t ii;
    for (ii = 0; ii < N-1; ii++){
        //printf("ii=%zu (out of %zu) \n",ii, N-2);
        temp = serialize_double_to_text(array[ii]);
        //printf("temp=%s\n",temp);
        concat_string_ow(&buffer,temp);
        concat_string_ow(&buffer,",");
        free(temp); temp=NULL;
        //printf("buffer so far = %s\n",buffer+sizerun);
    }
    temp = serialize_double_to_text(array[N-1]);
    concat_string_ow(&buffer,temp);
    free(temp); temp=NULL;
    
    return buffer;
}

/*************************************************************//**
    De-Serialize text to a double array

    \param num [in] - double to serialize
    \param N [inout] - number of values in array (will determine this)
 
    \return finalvals - deserialized array
*****************************************************************/
double * deserialize_darray_from_text(char * num, size_t *N)
{
    
    char ** vals = parse_string(num,',',N);
    double * finalvals = calloc_double(*N);
    size_t ii;
    
    for (ii = 0; ii < *N; ii++){
        finalvals[ii] = deserialize_double_from_text(vals[ii]);
        free(vals[ii]);
    }
    free(vals);
    return finalvals;
}

/***********************************************************//**
    Save an array to file
    
    \param[in] rows     - number of rows
    \param[in] cols     - number of cols
    \param[in] arr      - array to save
    \param[in] filename - filename
    \param[in] type     - 0 not human readable (fast)
                        - 1 human readable (slow)

    \return success (1) or failure (0) of opening the file
***************************************************************/
int 
darray_save(size_t rows, size_t cols, double * arr, char * filename, int type)
{
    FILE *fp;
    fp =  fopen(filename, "w");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open %s\n", filename);
        return 0;
    }
    
    if (type == 0){
        size_t totsize = rows*cols*sizeof(double) + sizeof(size_t) + // for array
                                    2*sizeof(size_t); // for rows and columns

        unsigned char * data = malloc(totsize+sizeof(size_t));
        if (data == NULL){
            fprintf(stderr, "can't allocate space for saving density\n");
            return 0;
        }

        unsigned char * ptr = serialize_size_t(data,totsize);
        ptr = serialize_size_t(ptr,rows);
        ptr = serialize_size_t(ptr,cols);
        ptr = serialize_doublep(ptr,arr,rows*cols);
        
        fwrite(data,sizeof(unsigned char),totsize+sizeof(size_t),fp);
        free(data); data = NULL;
    }
    else{
        size_t ii,jj;
        for (ii = 0; ii < cols; ii++){
            fprintf(fp,"x%zu ",ii);
        }
        fprintf(fp,"\n");
        for (ii = 0; ii < rows; ii++){
            for (jj = 0; jj < cols; jj++){
                fprintf(fp,"%3.16g ",arr[jj*rows+ii]);
            }
            fprintf(fp,"\n");
        }
    
    }
    fclose(fp);

    return 1;
}

/***********************************************************//**
    Load an array to file
    
    \param[in] filename - filename
    \param[in] type     - 0 not human readable (fast)
                        - 1 human readable (slow)

    \return array
***************************************************************/
double *
darray_load(char * filename, int type)
{
    FILE *fp;
    fp =  fopen(filename, "r");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open %s\n", filename);
        return NULL;
    }
    
    double * value = NULL;
    if (type == 0){
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
        
        size_t nrows, ncols, tot;
        unsigned char * ptr = deserialize_size_t(data,&nrows);
        ptr = deserialize_size_t(ptr,&ncols);
        ptr = deserialize_doublep(ptr,&value,&tot);
        assert(tot == (nrows*ncols));
        
        free(data); data = NULL;
        fclose(fp);
    }
    else{
        fprintf(stderr,"Loading array of type=%d is not implemented\n",type);
    }
    return value;
}

struct c3Vector * c3vector_alloc(size_t d, const double * x)
{
    struct c3Vector * c3v = malloc(sizeof(struct c3Vector));
    if (c3v == NULL){
        fprintf(stderr,"Cannot Allocate c3Vector\n");
        exit(1);
    }
    c3v->size = d;
    c3v->elem = calloc_double(d);
    memmove(c3v->elem,x,d*sizeof(double));
    return c3v;
}

struct c3Vector * c3vector_copy(const struct c3Vector * x)
{
    if (x == NULL){
        return NULL;
    }
    struct c3Vector * new = c3vector_alloc(x->size, x->elem);
    return new;
}

void c3vector_free(struct c3Vector * c3v)
{
    if (c3v != NULL){
        free(c3v->elem); c3v->elem = NULL;
        free(c3v); c3v = NULL;
    }
}

struct c3Vector ** c3vector_array_alloc(size_t N)
{
    struct c3Vector ** c3v;
    c3v = malloc(N*sizeof(struct c3Vector *));
    if (c3v == NULL){
        fprintf(stderr,"Memory alloc for c3Vector array failed\n");
        exit(1);
    }
    for (size_t ii = 0; ii < N; ii++){
        c3v[ii] = NULL;
    }
    return c3v;
}

struct c3Vector ** c3vector_array_copy(size_t N, struct c3Vector ** c3v)
{
    struct c3Vector ** new = c3vector_array_alloc(N);
    for (size_t ii = 0; ii < N; ii++){
        new[ii] = c3vector_copy(c3v[ii]);
    }
    return new;
}

void c3vector_array_free(size_t N, struct c3Vector ** c3v)
{
    if (c3v != NULL){
        for (size_t ii = 0; ii < N; ii++){
            c3vector_free(c3v[ii]); c3v[ii] = NULL;
        }
        free(c3v); c3v = NULL;
    }
}

// UTILITIES
//random numbers

// uniform in 0, 1
double randu(void)
{
    int temp_val = rand();
    double val = (double)temp_val / (double) RAND_MAX;
    return val;
}

// standard normal
//kbox-mueller
static int generate = -1;
double randn(void)
{
    const double epsilon = pow(2,-32);
    const double two_pi = 2.0*3.14159265358979323846;
         
    static double z0, z1;
    if (generate == 0){
        generate = 1;
    }
    else if (generate == 1){
        generate = 0;
    }
    else if (generate == -1){
        generate = 1;
    }
    //generate = !generate;
                     
    if (!generate) {
        return z1;
    }
                        
    double u1, u2;
    do
    {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while ( u1 <= epsilon );
             
    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0;
}

// poisson
size_t poisson(double mean)
{   
    double lleft = mean;
    size_t k = 0; 
    double p = 1.0;
    double u;
    double STEP = 500;
    do { 
        k++;
        u = randu();
        p = p * u;
        if ((p < exp(1)) && lleft > 0){
            if (lleft > STEP){
                p = p * exp(STEP);
                lleft = lleft - STEP;
            }
            else{
                p = p * exp(lleft);
                lleft -= 1;
            }
        }

    } while (p > 1.0);
    
    return k-1;

}



