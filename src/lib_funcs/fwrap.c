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

typedef enum {ND=0, VEC, MOVEC, ARRVEC, NUMFT} Ftype; 

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
 * \var Fwrap::mofvec
 *  vectorized multi output
 * \var Fwrap::evalfunc
 * indicator for which multioutput function to evaluate
 */
struct Fwrap
{
    size_t d;

    double (*f)(const double *,void *);
    int (*fvec)(size_t,const double*,double*,void*);

    void * fargs;

    Ftype ftype;

    // special arguments if type == MOVEC (multioutput vec)
    int (*mofvec)(size_t, size_t, const double *, double *, void *);
    size_t evalfunc; // which function to eval from mofvec

    // special arguments if type == ARRVEC (array of functions)
    size_t nfuncs;
    int (**arrfuncs)(size_t,const double*,double*,void*);
    void ** farrargs;
    
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
    else if ( (strcmp(type,"mo-vec") == 0)){
        fw->ftype = MOVEC;
        fw->evalfunc = 0;
    }
    else if ( (strcmp(type,"array-vec") == 0)){
        fw->ftype = ARRVEC;
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
    Set the vectorized function
***************************************************************/
void fwrap_set_fvec(struct Fwrap * fwrap, 
                    int (*f)(size_t,const double*,double*,void*),
                    void* arg)
{
    assert (fwrap != NULL);
    if (fwrap->ftype != VEC){
        fprintf(stderr,"Must set fwrap type to VEC before calling set_f\n");
        exit(1);
    }
    fwrap->fvec = f;
    fwrap->fargs =arg;
}

/***********************************************************//**
    Set the multi output vectorized function
***************************************************************/
void fwrap_set_mofvec(struct Fwrap * fwrap, 
                      int (*f)(size_t,size_t, const double*,double*,void*),
                      void* arg)
{
    assert (fwrap != NULL);
    if (fwrap->ftype != MOVEC){
        fprintf(stderr,"Must set fwrap type to MOVEC before calling set_mofvec\n");
        exit(1);
    }
    fwrap->mofvec = f;
    fwrap->fargs =arg;
}

/***********************************************************//**
    Set which of the multi output functions to evaluate
    when one needs to evaluate them one at a time
***************************************************************/
void fwrap_set_which_eval(struct Fwrap * fwrap, size_t which)
{
    assert (fwrap != NULL);
    /* if (fwrap->ftype != MOVEC){ */
    /*     fprintf(stderr,"Must set fwrap type to MOVEC before calling which_eval\n"); */
    /*     exit(1); */
    /* } */
    fwrap->evalfunc = which;
}

/***********************************************************//**
    Get which of the multi output functions to evaluate
    when one needs to evaluate them one at a time
***************************************************************/
size_t fwrap_get_which_eval(const struct Fwrap * fwrap)
{
    assert (fwrap != NULL);
    /* if (fwrap->ftype != MOVEC){ */
    /*     fprintf(stderr,"Must set fwrap type to MOVEC before calling which_eval\n"); */
    /*     exit(1); */
    /* } */
    return fwrap->evalfunc;
}

/***********************************************************//**
   For an array of function pointers
***************************************************************/
void fwrap_set_num_funcs(struct Fwrap * fwrap, size_t nfuncs)
{
    assert (fwrap != NULL);
    if (fwrap->ftype != ARRVEC){
        fprintf(stderr,"Must set fwrap type to ARRVEC before set_num_funcs\n");
        exit(1);
    }
    fwrap->nfuncs = nfuncs;
    fwrap->arrfuncs = malloc(nfuncs * sizeof( int (*)(size_t,const double*,
                                                      double*, void *)));
    fwrap->farrargs = malloc(nfuncs * sizeof (void *));
}

/***********************************************************//**
   For an array of function pointers
***************************************************************/
void fwrap_set_func_array(struct Fwrap * fwrap,size_t ind,
                          int (*f)(size_t,const double*, double*, void *),
                          void * args)
{
    assert (fwrap != NULL);
    if (fwrap->ftype != ARRVEC){
        fprintf(stderr,"Must set fwrap type to ARRVEC before set_num_funcs\n");
        exit(1);
    }
    assert (ind < fwrap->nfuncs);
    fwrap->arrfuncs[ind] = f;
    fwrap->farrargs[ind] = args;
}

/***********************************************************//**
    Destroy
***************************************************************/
void fwrap_destroy(struct Fwrap * fw)
{
    if (fw != NULL){
        if (fw->ftype == ARRVEC){
            free(fw->arrfuncs); fw->arrfuncs = NULL;
            free(fw->farrargs); fw->farrargs = NULL;
        }
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
    else if (fw->ftype == MOVEC){
        assert (fw->mofvec != NULL);
        return fw->mofvec(nevals,fw->evalfunc,x,out,fw->fargs);
    }
    else if (fw->ftype == ARRVEC){
        assert (fw->arrfuncs != NULL);
        return fw->arrfuncs[fw->evalfunc](nevals,x,out,fw->farrargs[fw->evalfunc]);
    }
    else{
        fprintf(stderr,"Cannot evaluate function wrapper of type %d\n",fw->ftype);
        return 1;
    }
}

