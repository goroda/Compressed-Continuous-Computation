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

int func(size_t, const double *, double *, void *);
int func2(size_t, const double *, double *, void *);
int funcp(size_t, const double *, double *, void *);
int func2p(size_t, const double *, double *, void *);
int func3(size_t, const double *, double *, void *);
int func3p(size_t, const double *, double *, void *);
int func4(size_t, const double *, double *, void *);
int func5(size_t, const double *, double *, void *);
int func6(size_t, const double *, double *, void *);

int funcnda(size_t, const double *, double *, void *);
int funcndb(size_t, const double *, double *, void *);
int funcnd1(size_t, const double *, double *, void *);
int funcnd2(size_t, const double *, double *, void *);
int disc2d(size_t, const double *, double *, void *);
int funcH4(size_t, const double *, double *,void *);
int funch1(size_t, const double *, double *, void *);
int funch2(size_t, const double *, double *, void *);
int func_not_all(size_t, const double *, double *, void *);
int sin10d(size_t, const double *, double *, void *);
int sin100d(size_t, const double *, double *, void *);
int sin1000d(size_t, const double *, double *, void *);

int funcGrad(size_t, const double *, double *, void *);
int funcHess(size_t, const double *, double *, void *);

int funcCheck2(size_t, const double *, double *, void *);
int funcCheck3(size_t, const double *, double *, void *);
int quad2d(size_t, const double *, double *, void *);

int gauss2d(size_t, const double *, double *, void *);

#endif
