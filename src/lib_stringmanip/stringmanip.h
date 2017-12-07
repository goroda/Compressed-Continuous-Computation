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






#ifndef STRINGMANIP_H
#define STRINGMANIP_H

#include <stdio.h>

void strip_ends(char * strin, char, char);
void strip_blank_ends(char *);
char * bite_string(char *, char);
char * bite_string2(char *, char *);
char ** parse_string(char *, char, size_t *);
char * concat_string(char *, char *);
void concat_string_ow(char **, char *);

char *
itoa (int value, char *result, int base);


// not very portable. need to define use some sort of standard
unsigned char * serialize_char(unsigned char *, char);
unsigned char * serialize_int(unsigned char *, int);
unsigned char * deserialize_int(unsigned char *, int *);
unsigned char * serialize_size_t(unsigned char *, size_t);
unsigned char * deserialize_size_t(unsigned char *, size_t *);
unsigned char * serialize_double(unsigned char *, double);
unsigned char * deserialize_double(unsigned char *, double *);
unsigned char * serialize_doublep(unsigned char *, double *, size_t);
unsigned char * deserialize_doublep(unsigned char *, double **, size_t *);

double * readtxt_double_array(char *,size_t *, size_t *);
double * readfile_double_array(FILE *, size_t *, size_t *);

#endif
