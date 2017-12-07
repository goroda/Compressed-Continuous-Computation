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





/** \file dmrg.h
 * Provides header files and structure definitions for functions in dmrg.c
 */

#ifndef DMRG_H
#define DMRG_H

#include "ft.h"

/** \struct QR
 *  \brief Holds Q and R for qmarray
 *  \var QR::right
 *  1 if QR and 0 if LQ
 *  \var QR::mat
 *  matrix part
 *  \var QR::mr
 *  number of rows in the matrix
 *  \var QR::mc
 *  number of columns in the matrix
 *  \var QR::Q
 *  Q term;
 */
struct QR
{
    int right;
    double * mat;
    size_t mr;
    size_t mc;
    struct Qmarray * Q;

};

struct QR * qr_reduced(const struct Qmarray *,int,struct OneApproxOpts*);
void qr_free(struct QR *);
void qr_array_free(struct QR **, size_t);
struct QR ** qr_array_alloc(size_t);


struct FunctionTrain *
dmrg_approx(struct FunctionTrain *,
            void (*)(char,size_t,size_t,double *,struct Qmarray **,void *),
            void *, double, size_t, double, int,struct MultiApproxOpts *);

#endif
