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


/** \file fwrap.h
 * Provides header files and structure definitions for functions in fwrap.c 
 */

#ifndef FWRAP_H
#define FWRAP_H

#include <stddef.h>

struct Fwrap;
struct Fwrap * fwrap_create(size_t, const char *);
int fwrap_get_type(const struct Fwrap *);
void fwrap_set_f(struct Fwrap *, double(*)(const double*,void*),void*);
void fwrap_set_findex(struct Fwrap *,double(*)(const size_t*,void*),void*);
void fwrap_set_fvec(struct Fwrap *, 
                    int (*)(size_t, const double*,double*,void*),void*);
void fwrap_set_mofvec(struct Fwrap *, 
                      int (*)(size_t,size_t,const double*,double*,void*),
                      void*);
void fwrap_set_which_eval(struct Fwrap *, size_t);
size_t fwrap_get_which_eval(const struct Fwrap *);

void fwrap_set_num_funcs(struct Fwrap *, size_t);
void fwrap_set_func_array(struct Fwrap *,size_t,
                          int (*)(size_t,const double*,double*,void *),
                          void *);
void fwrap_destroy(struct Fwrap *);
/* int fwrap_eval(size_t, const double *, double *, void *); */
int fwrap_eval(size_t, const void *, double *, void *);

// fibers
void fwrap_initialize_fiber_approx(struct Fwrap *, size_t, size_t);
/* void fwrap_add_fiber(struct Fwrap *, size_t,  */
/*                      size_t, const double *,  */
/*                      size_t, const double *); */
void fwrap_add_fiber(struct Fwrap *, size_t, 
                     size_t, const void *, 
                     size_t, const void *);
void fwrap_set_which_fiber(struct Fwrap *, size_t);
void fwrap_clean_fiber_approx(struct Fwrap *);
/* int fwrap_eval_fiber(size_t, const double *, double *, void *); */
int fwrap_eval_fiber(size_t, const void *, double *, void *);


// Interface stuff

#ifdef COMPILE_WITH_PYTHON
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"

int c3py_wrapped_eval(size_t N, const double * x, double * out, void * args);
void fwrap_set_pyfunc(struct Fwrap *, PyObject *);

#endif


#endif
