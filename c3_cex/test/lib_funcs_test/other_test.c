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




#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>

#include "CuTest.h"
#include "testfunctions.h"

#include "array.h"
#include "lib_linalg.h"

#include "lib_funcs.h"

void Test_Linked_List(CuTest * tc){
   
    printf("Testing functions: Linked_List \n");

    double x[5] = {0.0, 1.0, 2.0, 0.5, 0.3};
    double val = 2.0;

    size_t sv1 = 5 * sizeof(double) + sizeof(char);
    char * v1 = malloc(sv1);
    memmove(v1, x, 5 * sizeof(double));
    v1[sv1-1] = '\0';

    //printf("v1[0]=%c\n",v1[0]);

    size_t sv2 = sizeof(double) + sizeof(char);
    char * v2 = malloc(sv2);
    memmove(v2, &val, sizeof(double));
    v2[sv2-1] = '\0';

    struct Cpair * pl = cpair_create(v1,v2);
    struct PairList * ll = NULL;

    pair_push(&ll,pl);

   // print_pair_list(ll);
   // printf("============\n");
    pair_list_delete(&ll);

    //print_pair_list(ll);
    //printf("============\n");
    
    CuAssertIntEquals(tc,1,1);

    free(v1);
    free(v2);
    cpair_free(pl);
}

CuSuite * LinkedListGetSuite(){
    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_Linked_List);
    return suite;
}

