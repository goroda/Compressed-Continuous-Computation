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







/** \file monitoring.h
 * Provides header files and structure definitions for functions in monitoring.c 
 */

#ifndef MONITORING_H
#define MONITORING_H

#include <stdlib.h>
#include <stdio.h>

///////////////////////////////////////

/** \struct FunctionMonitor
 * \brief Utility to store / recall function evaluations
 * \var FunctionMonitor::ftype
 * 0 for multidimensional functions, 1 for one dimensional functions, 2 for two
 *   dimensional functions
 *  \var FunctionMonitor::dim
 *  dimension of function
 *  \var FunctionMonitor::f
 *  function to monitor
 *  \var FunctionMonitor::args
 *  function arguments
 *  \var FunctionMonitor::evals
 *  hashmap which stores evaluations
 */
struct FunctionMonitor
{
    int ftype;
    size_t dim;
    union ff {
        double (*fnd)(const double *, void *); //ftype = 0
        double (*f1d)(double, void *);// ftype = 1
        double (*f2d)(double, double, void *); // ftype = 2
    } f;
    void * args;
    struct HashtableCpair * evals;
};

struct FunctionMonitor * 
function_monitor_initnd( double (*)(const double *, void *), void *, size_t, size_t);
void function_monitor_free(struct FunctionMonitor *);
double function_monitor_eval(const double *, void *);
void function_monitor_print_to_file(struct FunctionMonitor *, FILE *);


struct storevals_main {
    int nEvals;
    struct storevals * head;
};

struct storevals {
    size_t dim;
    double * x;
    double f;
    struct storevals * next;
};

void PushVal(struct storevals **, size_t , double *, double);
void PrintVals2d(FILE *, struct storevals *);
void DeleteStoredVals(struct storevals **);


/** \struct Cpair
 * \brief A pair of chars
 * \var Cpair::a
 * first char
 * \var Cpair::b
 * second char
 */
struct Cpair {
    char * a;
    char * b;
};

struct Cpair * cpair_create(char *, char *);
struct Cpair * copy_cpair(struct Cpair *);
void print_cpair(struct Cpair *);
void cpair_free(struct Cpair *);
int cpair_isequal(struct Cpair *, struct Cpair *);

/** \struct PairList
 * \brief A linked-list of Cpair
 * \var PairList::data
 * a Cpair
 * \var PairList::next
 * pointer to next list item
 */
struct PairList {
    struct Cpair * data;
    struct PairList * next;
};

void pair_push(struct PairList **, struct Cpair *);
void print_pair_list(struct PairList *);
void pair_list_delete(struct PairList **);
size_t pair_list_len(struct PairList *);
size_t pair_list_index(struct PairList *, struct Cpair *);


/** \struct HashtableCpair
 *  \brief a hashmap of Cpairs where the keys are strings and the valrs are Cpairs
 *  \var HashtableCpair::size
 *  size of the table
 *  \var HashtableCpair::table
 *  stored values
 */
struct HashtableCpair
{
    size_t size;
    struct PairList ** table;
};

struct HashtableCpair * create_hashtable_cp(size_t);
char * lookup_key(struct HashtableCpair *, char *); 
int add_cpair(struct HashtableCpair *, struct Cpair *);
void free_hashtable_cp(struct HashtableCpair *);
size_t nstored_hashtable_cp(struct HashtableCpair *);

size_t hashsimple(size_t, char *);


#endif

