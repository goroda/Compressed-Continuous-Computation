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

/** \file linelm.h
 * Provides header files and structure definitions for functions in linelm.c
 */

#ifndef LINELM_H
#define LINELM_H

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

#include <stdlib.h>
#include <stdio.h>

/** \struct LinElemExp
 * \brief structure to represent an expansion of linear elements
 * \var LinElemExp::num_nodes
 * number of basis functions or nodes
 * \var LinElemExp::nodes
 * nodes of basis functions
 * \var LinElemExp::coeff
 * coefficients of basis functions
 * \var LinElemExp::inner
 * inner products of basis functions
 */
struct LinElemExp{

    size_t num_nodes;
    double * nodes;
    double * coeff;
    double * inner;
};


struct LinElemExp * lin_elem_exp_alloc();
struct LinElemExp * lin_elem_exp_copy(struct LinElemExp *);
void lin_elem_exp_free(struct LinElemExp *);
struct LinElemExp * lin_elem_exp_init(size_t, double *, double *);
unsigned char *
serialize_lin_elem_exp(unsigned char *, struct LinElemExp *,size_t *);
unsigned char * deserialize_lin_elem_exp(unsigned char *, 
                                         struct LinElemExp **);
double lin_elem_exp_eval(struct LinElemExp *, double);
double lin_elem_exp_integrate(struct LinElemExp *);
double lin_elem_exp_inner(struct LinElemExp *,struct LinElemExp *);
int lin_elem_exp_axpy(double, struct LinElemExp *,struct LinElemExp *);
double lin_elem_exp_norm(struct LinElemExp *);
double lin_elem_exp_max(struct LinElemExp *, double *);
double lin_elem_exp_min(struct LinElemExp *, double *);
double lin_elem_exp_absmax(struct LinElemExp *, double *,void *);

double lin_elem_exp_err_est(struct LinElemExp *, double *, short,short);

/////////////////////////////////////////////////////////////////////
/** \struct LinElemExpAopts
 * \brief Approximation options of LinElemExp
 * \var LinElemExp::num_nodes
 * number of basis functions or nodes
 * \var LinElemExp::nodes
 * nodes
 * \var LinElemExp::adapt
 * whether or not to adapt (0 or 1)
 */
struct LinElemExpAopts
{
    size_t num_nodes;
    double * nodes;
    int adapt;

    double delta;
    double hmin;
};

struct LinElemExpAopts * lin_elem_exp_aopts_alloc(size_t, double *);
void lin_elem_exp_aopts_free(struct LinElemExpAopts *);

struct LinElemXY;
void lin_elem_adapt(double (*)(double,void*), void *,
                    double, double,double, double,
                    double, double,struct LinElemXY ** x);
double lin_elem_xy_get_x(struct LinElemXY *);
double lin_elem_xy_get_y(struct LinElemXY *);
struct LinElemXY * lin_elem_xy_next(struct LinElemXY *);
void lin_elem_xy_free(struct LinElemXY *);

struct LinElemExp *
lin_elem_exp_approx(double (*)(double,void*), void*,
                    double, double,
                    struct LinElemExpAopts *);
void lin_elem_exp_scale(double, struct LinElemExp *);
void lin_elem_exp_flip_sign(struct LinElemExp *);
void lin_elem_exp_orth_basis(size_t, struct LinElemExp **);
struct LinElemExp *
lin_elem_exp_constant(double, double, double,
                      const struct LinElemExpAopts *);
double lin_elem_exp_lb(struct LinElemExp *);
double lin_elem_exp_ub(struct LinElemExp *);

void print_lin_elem_exp(struct LinElemExp *, size_t, void *, FILE *);

#endif
