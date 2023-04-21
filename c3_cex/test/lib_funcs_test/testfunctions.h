// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016-2017 Sandia Corporation

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




#ifndef TESTF_H
#define TESTF_H

#include <stdlib.h>

int Sin3xTx2(size_t, const double *, double *, void *);
double funcderiv(double, void *);
double funcdderiv(double, void *);
int gaussbump(size_t, const double *, double *, void *);
int gaussbump_deriv(size_t, const double *, double *, void *);
int gaussbump_dderiv(size_t, const double *, double *, void *);
int gaussbump2(size_t, const double *, double *, void *);
int gaussbump2dd(size_t, const double *, double *, void *);
int sin_lift(size_t, const double *, double *, void *);
int sin_liftdd(size_t, const double *, double *, void *);
int powX2(size_t, const double *, double *, void *);
int TwoPowX3(size_t, const double *, double *, void *);

int polyroots(size_t, const double *, double *, void *);
int maxminpoly(size_t, const double *, double *, void *);
int x3minusx(size_t, const double *, double *, void *);
double x3minusxd(double, void *);
double x3minusxdd(double, void *);
#endif
