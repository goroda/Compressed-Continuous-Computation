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

#include "stringmanip.h"

void strip_blank_ends(char * strin)
{
    
    int ii = 0;
    int n;
    char * str = strin;
    while (str[ii] == ' ')
    {
        memmove(str,str+1,strlen(str)+1);
    }

    n = strlen(str);
    while (str[n]==' ' || str[n]=='\0')
        n--;
    str[n+1]='\0';
}

void strip_ends(char * strin, char left, char right)
{
    int ii = 0;
    int n;
    char * str = strin;
    while (str[ii] == left)
    {
        memmove(str,str+1,strlen(str+1));
    }
    
    n = strlen(str);
    while (str[n] == right || str[n]=='\0'){
        n--;
    }
    str[n+1] = '\0';
}

char * bite_string(char * strin, char tok)
{
    
    char * out = NULL;

    if (strin == NULL){
        return out;
    }
    //printf("here!%s\n", strin);
    size_t ii=0;
    while (strin[ii] != tok){
        ii++;
        
        if (ii == (strlen(strin)+1)){
            return out;
        }
    }

    out = malloc((ii+1)*sizeof(char));

    memcpy(out,strin,ii);
    out[ii] = '\0';
    
    memmove(strin, strin+ii+1, strlen(strin)+1-ii-1);
    return out;

}

char * bite_string2(char * strin, char * tok)
{
    char * out = NULL;
    size_t ii = 0;
    size_t cut = 0;
    size_t nMatch = 0;
    for (ii = 0; ii < strlen(strin); ii++){
        if (strin[ii] == tok[nMatch]){
            nMatch++;
        }
        else{
            nMatch = 0;
        }
        if (nMatch == strlen(tok)){
            cut = ii+1 - nMatch;
            break;
        }
    }
    
    if (cut > 0){
        out = malloc((cut+1)*sizeof(char));
        memmove(out,strin,cut);
        out[cut] = '\0';
        memmove(strin, strin+cut+strlen(tok), strlen(strin)+1-ii-1);
    }
    return out;
}

char ** parse_string(char * strin, char tok, size_t * N)
{
    
    *N = 0;
    size_t ii;
    for (ii = 0; ii < strlen(strin); ii++){
        if (strin[ii] == tok){
            *N = *N + 1;
        }
    }
    *N = *N + 1; // last one
    
    char ** out = malloc(*N * sizeof(char *));
    
    ii = 0;
    size_t jj;
    size_t track=0, new_len;
    for (jj = 0; jj < strlen(strin); jj++){
        if (strin[jj] == tok){
            new_len = jj - track + 1;
            out[ii] = malloc(new_len * sizeof(char));
            memmove(out[ii],strin+track, new_len-1);
            out[ii][new_len-1] = '\0';
            track = jj+1;

            //printf("out[ii]=%s\n",out[ii]);
            ii++;
        }
    }
    new_len = jj - track + 1;
    out[ii] = malloc(new_len * sizeof(char));
    memmove(out[ii],strin+track, new_len-1);
    out[ii][new_len-1] = '\0';

    return out;
}

char * concat_string(char * a, char * b)
{
    size_t N1;
    if (a == NULL){
        N1 = 0;
    }
    else{
        N1 = strlen(a);
    }
    size_t N2;
    if (b == NULL){
        N2 = 0;
    }
    else{
        N2 = strlen(b);
    }

    char * c;
    c = malloc((N1+N2+1)*sizeof(char));
    if (N1 > 0){
        memmove(c,a,N1);
    }
    if (N2 > 0){
        memmove(c+N1,b,N2);
    }
    c[N1+N2] = '\0';
    return c;
}

//overwrites a
void concat_string_ow(char ** a, char * b){
    
    char * c = concat_string(*a,b);
    //size_t Nc = strlen(c);
    
    free(*a); *a = NULL;
    *a = c;
}


// from http://stackoverflow.com/questions/8257714/how-to-convert-an-int-to-string-in-c/8257728#8257728
char *
itoa (int value, char *result, int base)
{
    // check that the base if valid
    if (base < 2 || base > 36) { *result = '\0'; return result; }
    
    char* ptr = result, *ptr1 = result, tmp_char;
    int tmp_value;
    
    do {
        tmp_value = value;
        value /= base;
        *ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz" [35 + (tmp_value - value * base)];
    } while ( value );
    
    // Apply negative sign
    if (tmp_value < 0) *ptr++ = '-';
    *ptr-- = '\0';
    while (ptr1 < ptr) {
        tmp_char = *ptr;
        *ptr--= *ptr1;
        *ptr1++ = tmp_char;
    }
    return result;
}

unsigned char * serialize_char(unsigned char * buffer, char value)
{
    size_t nBytes = sizeof(char); // should be 1
    memcpy(buffer, &value, nBytes);
    return buffer+nBytes;
}

unsigned char * serialize_int(unsigned char * buffer, int value)
{
    size_t nBytes = sizeof(int);
    memcpy(buffer, &value, nBytes);
    return buffer+nBytes;
}

unsigned char * deserialize_int(unsigned char * buffer, int * value)
{
    size_t nBytes = sizeof(int);
    memcpy(value,buffer,nBytes);
    return buffer + nBytes;
}

unsigned char * serialize_size_t(unsigned char * buffer, size_t value)
{
    size_t nBytes = sizeof(size_t);
    memcpy(buffer, &value, nBytes);
    return buffer+nBytes;
}

unsigned char * deserialize_size_t(unsigned char * buffer, size_t * value)
{
    size_t nBytes = sizeof(size_t);
    memcpy(value,buffer,nBytes);
    return buffer + nBytes;
}

unsigned char * serialize_double(unsigned char * buffer, double value)
{
    size_t nBytes = sizeof(double);
    memcpy(buffer, &value, nBytes);
    return buffer+nBytes;
}

unsigned char * deserialize_double(unsigned char * buffer, double * value)
{
    size_t nBytes = sizeof(double);
    memcpy(value,buffer,nBytes);
    return buffer + nBytes;
}

unsigned char * 
serialize_doublep(unsigned char * buffer, double * value, size_t N)
{
    unsigned char * ptr = serialize_size_t(buffer,N);

    size_t nBytes = N * sizeof(double); // should it be double *? No. I think;

    memcpy(ptr, value, nBytes);
    return ptr + nBytes;
}

unsigned char * 
deserialize_doublep(unsigned char * buffer, double ** value, size_t * N)
{
    
    unsigned char * ptr = deserialize_size_t(buffer,N);
    *value = malloc(*N * sizeof(double));
    size_t nBytes = *N * sizeof(double);
    memcpy(*value, ptr, nBytes);

    return ptr + nBytes;
}

/////////////////////////////////////
/////////////////////////////////////
////  Parsing  strings for data /////
/////////////////////////////////////
/////////////////////////////////////
char * mystrdup(char * str)
{
    size_t len = strlen(str)+1;
    char *p = malloc(len);
    return p ? memcpy(p, str, len) : NULL;
}

double * readtxt_double_array(char * str,size_t *nrows, size_t *ncols)
{
    char * token;
    
    char * str2 = mystrdup(str);
    char * str3 = mystrdup(str);

    /* char *save1,*save2,*save3;; */
    
    token = strtok(str,"\t ,\n");
    size_t ntot = 0;
    while (token != NULL){
        token = strtok(NULL,"\t ,\n");
        ntot++;
    }
    /* printf("total = %zu\n",ntot); */
    
    double * data = calloc(ntot,sizeof(double));
    if (data == NULL){
        fprintf(stderr, "Cannot allocate data for parsing array\n");
        exit(1);
    }

    token = strtok(str2,"\n");
    *nrows = 0;
    while (token != NULL){
        token = strtok(NULL,"\n");
        *nrows = *nrows + 1;
    }
    *ncols = ntot / *nrows;
    
    /* printf("number of rows = %zu\n",*nrows); */
    /* printf("number of cols = %zu\n",*ncols); */

    token = strtok(str3," ,\n");
    size_t ind = 0;
    /* printf("%s %zu \n",token,ind); */
    while (token != NULL){
        data[ind] = atof(token);
        ind++;
        token = strtok(NULL," ,\n");
        /* printf("%s %zu\n",token,ind); */
    }

    free(str2); str2 = NULL;
    free(str3); str3 = NULL;
    return data;
}

double * readfile_double_array(FILE * fp, size_t *nrows, size_t *ncols)
{
    fseek(fp, 0L, SEEK_END);  /* Position to end of file */
    long lFileLen = ftell(fp);     /* Get file length */
    rewind(fp);

    char * cFile = calloc((unsigned long)lFileLen + 1, sizeof(char));

    if(cFile == NULL )
    {
        printf("\nInsufficient memory to read file.\n");
        return NULL;
    }

    /* Read the entire file into cFile */
    size_t ret = fread(cFile, (unsigned long) lFileLen, 1, fp); 
    if (ret != 1){
        fprintf(stderr, "Error reading file containing double array\n");
        free(cFile); cFile = NULL;
        return NULL;
    }
    double * vals = readtxt_double_array(cFile,nrows,ncols);
    free(cFile); cFile = NULL;
    return vals;
}
