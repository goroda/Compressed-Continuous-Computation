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

    \param N [in] - Number of elements in array
    \param arr [in] - array to print
****************************************************************/
void 
dprint(const size_t N, double * arr){
    size_t ii = 0;
    for (ii = 0; ii < N; ii++) printf("%f ",arr[ii]);
    printf("\n");
}

/*************************************************************//**
    Print a two dimensional double array stored as a double pointer in Row-Major (C) order

    \param N [in] - number of rows
    \param M [in] - number of cols
    \param arr [in] - array to print
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
        
    \param N [in] - Number of elements
    \param arr [in] - array to print
****************************************************************/
void iprint(const size_t N, int * arr){
    size_t ii = 0;
    for (ii = 0; ii < N; ii++) printf("%d ",arr[ii]);
    printf("\n");
}

/*************************************************************//**
    Print a size_t array
        
    \param N [in] - Number of elements
    \param arr [in] - array to print
****************************************************************/
void iprint_sz(const size_t N, size_t * arr){
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

/*************************************************************//**
    Serialize a double to string
            
    \param x [in] - double to serialize

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
    cx = snprintf(buffer, N, "%0.20f:%d", res,exp);
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
        snprintf(buffer, N, "%0.20f:%d", res,exp);
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
 
    \param N [in] - number of elements in the array
    \param array [in] - array to serialize
 
    \return buffer - string of values
****************************************************************/
char * serialize_darray_to_text(size_t N, double * array)
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
    
    \param rows [in] - number of rows
    \param cols [in] - number of cols
    \param arr [in] - array to save

    \param filename [in] - filename
    \param type [in] - 0 not human readable (fast)
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
                fprintf(fp,"%3.16f ",arr[jj*rows+ii]);
            }
            fprintf(fp,"\n");
        }
    
    }
    fclose(fp);

    return 1;
}

/***********************************************************//**
    Load an array to file
    
    \param filename [in] - filename
    \param type [in] - 0 not human readable (fast)
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



