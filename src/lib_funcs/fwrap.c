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





/** \file fwrap.c
 * Provides basic routines for wrapping functions
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "array.h"
#include "fwrap.h"


#ifdef COMPILE_WITH_PYTHON

struct Obj
{
    size_t dim;
    void * f;
    void * params;
};

PyMODINIT_FUNC initiate_numpy(void)
{
    Py_Initialize();
    import_array();
    #if PY_MAJOR_VERSION >= 3
    return 0;
    #endif
}


static int eval_py_obj(size_t N, const double * x, double * out, void * obj_void)
{
    struct Obj * obj = obj_void;
    size_t dim = obj->dim;
    
    npy_intp dims[2];
    dims[0] = (npy_intp) N;
    dims[1] = (npy_intp) dim;

    // Setup inputs
    PyObject * pyX = PyArray_SimpleNewFromData(2,dims,NPY_DOUBLE,(double*)x);
    
    // Call function
    PyObject * pyResult = PyObject_CallFunctionObjArgs(obj->f,pyX,obj->params,NULL);
    PyArrayObject * arr = (PyArrayObject*)PyArray_ContiguousFromAny(pyResult,NPY_DOUBLE,1,1);

    // Ensure outputs
    size_t ndims = (size_t)PyArray_NDIM(arr);
    if (ndims > 1){
        PyErr_SetString(PyExc_TypeError, "Wrapped python function must return a flattened (1d) array\n");
        return 1;
    }
    npy_intp * dimss  = (npy_intp*)PyArray_DIMS(arr);

    if ((size_t)dimss[0] != N){
        PyErr_SetString(PyExc_TypeError,
                        "Wrapped python function must return an array with the same number of rows as input\n");
        return 1;
    }

    // copy data to output
    double * vals = (double*)PyArray_DATA(arr);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = vals[ii];
    }

    Py_XDECREF(arr);
    Py_XDECREF(pyX);
    Py_XDECREF(pyResult);
    
    return 0;
}

int c3py_wrapped_eval(size_t N, const double * x, double * out, void * args)
{

    /* PyObject * py_obj = args; */
    /* printf("c3py_wrapped_eval\n"); */
    struct Obj * obj = args;
    
    int res = eval_py_obj(N,x,out,obj);
    
    return res;
}
#endif /* COMPILE_WITH_PYTHON */


typedef enum {ND=0, INTND, VEC, MOVEC, ARRVEC, NUMFT} Ftype; 

/** 
\struct Fwrap
\brief Interface to function pointers
\var Fwrap::d 
number of dimensions
\var Fwrap::f 
function
\var Fwrap::fvec 
vectorized function
\var Fwrap::intf 
function taking size_t arguments
\var Fwrap::fargs 
function arguments
\var Fwrap::ftype 
function type
\var Fwrap::mofvec 
vectorized multi output
\var Fwrap::evalfunc 
indicator for which multioutput function to evaluate
\var Fwrap::interface
flag as to whether special interface used (1 for python)
*/
struct Fwrap
{
    size_t d;

    double (*f)(const double *,void *);
    int (*fvec)(size_t,const double*,double*,void*);

    double (*intf)(const size_t *, void *);

    void * fargs;

    Ftype ftype;

    // special arguments if type == MOVEC (multioutput vec)
    int (*mofvec)(size_t, size_t, const double *, double *, void *);
    size_t evalfunc; // which function to eval from mofvec

    // special arguments if type == ARRVEC (array of functions)
    size_t nfuncs;
    int (**arrfuncs)(size_t,const double*,double*,void*);
    void ** farrargs;


    // fiber stuff
    int fiber_approx;
    size_t fiber_dim;
    size_t nfibers;
    /* double ** fiber_vals; */
    void ** fiber_vals;
    size_t onfiber;


    //Interface
    int interface;
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

    fw->interface = 0;
    fw->d = dim;
    if ( (strcmp(type,"general") == 0) || (type == NULL)){
        fw->ftype = ND;
    }
    else if ( (strcmp(type,"general-index") == 0) || (type == NULL)){
        fw->ftype = INTND;
    }
    else if ( (strcmp(type,"general-vec") == 0)){
        fw->ftype = VEC;
    }
    #ifdef COMPILE_WITH_PYTHON
    else if ( (strcmp(type,"python") == 0)){
        initiate_numpy();
        fw->ftype = VEC;
        fw->interface = 1;
    }
    #endif
    else if ( (strcmp(type,"mo-vec") == 0)){
        fw->ftype = MOVEC;
    }
    else if ( (strcmp(type,"array-vec") == 0)){
        fw->ftype = ARRVEC;
    }
    else{
        fprintf(stderr,"Wrapped function: unrecognized function type %s\n",type);
        exit(1);
    }

    fw->evalfunc = 0;


    fw->fiber_approx = 0;
    fw->fiber_dim = 0;
    fw->nfibers = 0;
    fw->fiber_vals = NULL;
    fw->onfiber = 0;
    return fw;
}

int fwrap_get_type(const struct Fwrap * fwrap)
{
    assert (fwrap != NULL);
    return fwrap->ftype;
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
    Set the function
***************************************************************/
void fwrap_set_findex(struct Fwrap * fwrap, double(*f)(const size_t*,void*),void*arg)
{
    assert (fwrap != NULL);
    if (fwrap->ftype != INTND){
        fprintf(stderr,"Must set fwrap type to INTND before calling set_findex\n");
        exit(1);
    }
    fwrap->intf = f;
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

#ifdef COMPILE_WITH_PYTHON
/***********************************************************//**
    Set a python function
***************************************************************/
void fwrap_set_pyfunc(struct Fwrap * fwrap, PyObject * args)
{

    
    struct Obj * obj = PyCapsule_GetPointer(args,NULL);
    /* printf("Got obj\n"); */
    fwrap->fvec = c3py_wrapped_eval;
    fwrap->fargs = obj;
}
#endif

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
    fwrap->evalfunc = which;
}

/***********************************************************//**
    Get which of the multi output functions to evaluate
    when one needs to evaluate them one at a time
***************************************************************/
size_t fwrap_get_which_eval(const struct Fwrap * fwrap)
{
    assert (fwrap != NULL);
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
int fwrap_eval(size_t nevals, const void * x, double * out, void * fwin)
{
    struct Fwrap * fw = fwin;
    assert (fw->ftype < NUMFT);
    /* assert (fw->ftype > 0); */
    if (fw->fiber_approx == 1){
        return fwrap_eval_fiber(nevals,x,out,fwin);
    }
    if (fw->ftype == ND){
        assert (fw->f != NULL);
        const double * xin = x;
        for (size_t ii = 0; ii < nevals; ii++){
            out[ii] = fw->f(xin + ii*fw->d,fw->fargs);
        }
        return 0;
    }
    else if (fw->ftype == INTND){
        assert (fw->intf != NULL);
        const size_t * xin = x;
        for (size_t ii = 0; ii < nevals; ii++){
            out[ii] = fw->intf(xin + ii*fw->d,fw->fargs);
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
        return fw->arrfuncs[fw->evalfunc](nevals,x,out,
                                          fw->farrargs[fw->evalfunc]);
    }
    else{
        fprintf(stderr,"Cannot evaluate function wrapper of type %d\n",fw->ftype);
        return 1;
    }
}

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////                                 ///////////////
//////////////// utilities for evaluating fibers ///////////////
////////////////                                 ///////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

/***********************************************************//**
    Initialize fiber approximation. Creates  *nfibers*
    functions that are slices of the function in dimension *ind*

    \param[in] fw      - wrapped function to slice
    \param[in] ind     - dimension along which to generate slices
    \param[in] nfibers - number of slices
***************************************************************/
void fwrap_initialize_fiber_approx(struct Fwrap * fw, size_t ind, size_t nfibers)
{

    assert (fw != NULL );
    assert (fw->ftype < NUMFT);
    /* assert (fw->ftype > -1); */

    fw->fiber_approx = 1;
    fw->fiber_dim = ind;
    fw->nfibers = nfibers;

    if (fw->ftype == INTND){
        fw->fiber_vals = malloc(nfibers * sizeof(size_t *));
        assert (fw->fiber_vals != NULL);
        for (size_t ii = 0; ii < nfibers; ii++){
            fw->fiber_vals[ii] = calloc_size_t(fw->d);
        }
    }
    else{
        fw->fiber_vals = malloc(nfibers * sizeof(double *));
        assert (fw->fiber_vals != NULL);
        for (size_t ii = 0; ii < nfibers; ii++){
            fw->fiber_vals[ii] = calloc_double(fw->d);
        }
    }
}

/***********************************************************//**
    Add fixed fiber values  

    \param[in] fw    - wrapped function to slice
    \param[in] ind   - specification of which fiber
    \param[in] nl    - number of fixed values on the left
    \param[in] left  - fixed values on the left
    \param[in] nr    - fixed values on the right
    \param[in] right - fixed values on the right
***************************************************************/
void fwrap_add_fiber(struct Fwrap * fw, size_t ind, 
                     size_t nl, const void * left, 
                     size_t nr, const void * right)
{

    assert (fw != NULL);
    assert (fw->fiber_approx == 1);
    /* printf("dim = %zu, nl=%zu, nr=%zu\n",fw->d,nl,nr); */
    assert ( (nl + nr + 1)  == fw->d);
    assert (nl == fw->fiber_dim);
    assert (ind < fw->nfibers);

    size_t size_elem;
    if (fw->ftype == INTND){
        size_elem = sizeof(size_t);
        if (nl != 0){
            memmove((size_t *)(fw->fiber_vals[ind]), left, nl * size_elem);     
        }
        if (nr != 0){
            memmove((size_t *)(fw->fiber_vals[ind]) + nl + 1, right, nr * size_elem);
        }
    }
    else{
        size_elem = sizeof(double);
        if (nl != 0){
            memmove((double *)(fw->fiber_vals[ind]), left, nl * size_elem);     
        }
        if (nr != 0){
            memmove((double *)(fw->fiber_vals[ind]) + nl + 1, right, nr * size_elem);
        }
    }

}

/***********************************************************//**
    Set which fiber I am currently on
***************************************************************/
void fwrap_set_which_fiber(struct Fwrap * fw, size_t which)
{
    assert (fw != NULL);
    assert (fw->ftype < NUMFT);
    /* assert (fw->ftype > 0); */
    assert (fw->fiber_approx == 1);
    assert (which < fw->nfibers);
    fw->onfiber = which;
    /* printf("on fiber = %zu\n",fw->onfiber); */
}

/***********************************************************//**
    Cleaned up fiber approximation

    \param[in,out] fw - wrapped function to slice
***************************************************************/
void fwrap_clean_fiber_approx(struct Fwrap * fw)
{
    fw->fiber_approx = 0;
    if (fw->fiber_vals != NULL){
        for (size_t ii = 0; ii < fw->nfibers; ii++){
            free(fw->fiber_vals[ii]);
            fw->fiber_vals[ii] = NULL;
        }
        free(fw->fiber_vals);
        fw->fiber_vals = NULL;
    }
    fw->fiber_vals = NULL;
    fw->nfibers = 0;
}

/***********************************************************//**
    Evaluate
    
    \return 0 if everything is fine 
            1 if unrecognized function type
***************************************************************/
int fwrap_eval_fiber(size_t nevals, const void * x, double * out, void * fwin)
{
    struct Fwrap * fw = fwin;

    int ret = 0;
    if (fw->ftype == INTND){
        
        assert (fw->intf != NULL);
        const size_t * xin = x;
        size_t * xeval = calloc_size_t(nevals * fw->d);
        size_t * fiber_vals = fw->fiber_vals[fw->onfiber];
        for (size_t ii = 0; ii < nevals; ii++){
            for (size_t jj = 0; jj < fw->d; jj++){
                xeval[ii*fw->d+jj] = fiber_vals[jj];
            }
            xeval[ii*fw->d + fw->fiber_dim] = xin[ii];
        }

        
        for (size_t ii = 0; ii < nevals; ii++){
            out[ii] = fw->intf(xeval + ii*fw->d,fw->fargs);
        }
    }
    else{
        const double * xin = x;
        double * xeval = calloc_double(nevals * fw->d);
        double * fiber_vals = fw->fiber_vals[fw->onfiber];
        for (size_t ii = 0; ii < nevals; ii++){
            for (size_t jj = 0; jj < fw->d; jj++){
                xeval[ii*fw->d+jj] = fiber_vals[jj];
            }
            xeval[ii*fw->d + fw->fiber_dim] = xin[ii];
        }

        
        if (fw->ftype == ND){
            assert (fw->f != NULL);
            for (size_t ii = 0; ii < nevals; ii++){
                out[ii] = fw->f(xeval + ii*fw->d,fw->fargs);
            }
        }
        else if (fw->ftype == VEC){
            assert (fw->fvec != NULL);
            ret = fw->fvec(nevals,xeval,out,fw->fargs);
        }
        else if (fw->ftype == MOVEC){
            assert (fw->mofvec != NULL);
            ret = fw->mofvec(nevals,fw->evalfunc,xeval,out,fw->fargs);
        }
        else if (fw->ftype == ARRVEC){
            assert (fw->arrfuncs != NULL);
            ret = fw->arrfuncs[fw->evalfunc](nevals,xeval,out,
                                             fw->farrargs[fw->evalfunc]);
        }
        else{
            fprintf(stderr,"Cannot evaluate function wrapper of type %d\n",fw->ftype);
            ret = 1;
        }

        free(xeval);
    }
    return ret;
}




