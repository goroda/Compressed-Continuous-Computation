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

/** \file fwrap.c
 * Provides basic routines for wrapping functions
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

typedef enum {ND=0, VEC, NUMFT} Ftype; 

/** \struct Fwrap
 *  \brief Interface to function pointers
 *  \var Fwrap::d
 *  number of dimensions
 *  \var Fwrap::f
 *  function
 *  \var Fwrap::fvec
 *  vectorized function
 *  \var Fwrap::fargs
 *  function arguments
 * \var Fwrap::ftype
 *  function type
 */
struct Fwrap
{
    size_t d;

    double (*f)(const double *,void *);
    int (*fvec)(size_t,const double*,double*,void*);

    void * fargs;

    Ftype ftype;
};

/**********************************************************//**
    Create an fwrapper
**************************************************************/
struct Fwrap * fwrap_create(size_t dim, const char * type)
{
    struct Fwrap * fw = malloc(sizeof(struct Fwrap));
    if (fw == NULL){
        fprintf(stderr,"Memory error creating fwrapper\n");
        exit(1);
    }

    fw->d = dim;
    if ( (strcmp(type,"general") == 0) || (type == NULL)){
        fw->ftype = ND;
    }
    else if ( (strcmp(type,"general-vec") == 0)){
        fw->ftype = VEC;
    }
    else{
        fprintf(stderr,"Unrecognized function type type %s\n",type);
        exit(1);
    }
    return fw;
}

/***********************************************************//**
    Set the function
***************************************************************/
void fwrap_set_f(struct Fwrap * fwrap, double(*f)(const double*,void*),void*arg)
{
    assert (fwrap != NULL);
    if (fwrap->ftype != ND){
        fprintf(stderr,"Must set fwrap type to ND before calling set_f\n");
        exit(1);
    }
    fwrap->f = f;
    fwrap->fargs =arg;
}

/***********************************************************//**
    Set the vectorizedfunction
***************************************************************/
void fwrap_set_fvec(struct Fwrap * fwrap, 
                    int (*f)(size_t,const double*,double*,void*),
                    void* arg)
{
    assert (fwrap != NULL);
    if (fwrap->ftype != VEC){
        fprintf(stderr,"Must set fwrap type to ND before calling set_f\n");
        exit(1);
    }
    fwrap->fvec = f;
    fwrap->fargs =arg;
}


/***********************************************************//**
    Destroy
***************************************************************/
void fwrap_destroy(struct Fwrap * fw)
{
    if (fw != NULL){
        free(fw); fw = NULL;
    }
}

/***********************************************************//**
    Evaluate
    
    \return 0 if everything is fine 
            1 if unrecognized function type
***************************************************************/
int fwrap_eval(size_t nevals, const double * x, double * out, void * fwin)
{
    struct Fwrap * fw = fwin;
    if (fw->ftype == ND){
        assert (fw->f != NULL);
        for (size_t ii = 0; ii < nevals; ii++){
            out[ii] = fw->f(x + ii*fw->d,fw->fargs);
        }
        return 0;
    }
    else if (fw->ftype == VEC){
        assert (fw->fvec != NULL);
        return fw->fvec(nevals,x,out,fw->fargs);
    }
    else{
        fprintf(stderr,"Cannot evaluate function wrapper of type %d\n",fw->ftype);
        return 1;
    }
}

