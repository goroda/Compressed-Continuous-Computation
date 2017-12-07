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



/** \file futil.h
 * Some macro utilities
 */

#ifndef FUNC_UTIL_H
#define FUNC_UTIL_H

#define NOT_IMPLEMENTED_MSG(str) fprintf(stderr, "%s not yet implemented\n", str);


// MACROS TO AID GENERIC PROGRAMMING

#define GF_SWITCH_NO_OUT(operation)                                 \
    switch (gf->fc){                                                \
    case PIECEWISE:  piecewise_poly_##operation(gf->f);      break; \
    case POLYNOMIAL: orth_poly_expansion_##operation(gf->f); break; \
    case LINELM:     lin_elem_exp_##operation(gf->f);        break; \
    case CONSTELM:   const_elem_exp_##operation(gf->f);      break; \
    case KERNEL:     kernel_expansion_##operation(gf->f);    break; \
    }

#define GF_SWITCH_NO_ONEOUT(operation, fc, A, B)                     \
    switch (fc){                                                     \
    case PIECEWISE:  piecewise_poly_##operation(A,B);      break;    \
    case POLYNOMIAL: orth_poly_expansion_##operation(A,B); break;    \
    case LINELM:     lin_elem_exp_##operation(A,B);        break;    \
    case CONSTELM:   const_elem_exp_##operation(A,B);      break;    \
    case KERNEL:     kernel_expansion_##operation(A,B);    break;    \
    }

#define GF_SWITCH_TEMPOUT(operation) \
    switch (gf->fc){                                                       \
    case PIECEWISE:  temp = piecewise_poly_##operation(gf->f);      break; \
    case POLYNOMIAL: temp = orth_poly_expansion_##operation(gf->f); break; \
    case LINELM:     temp = lin_elem_exp_##operation(gf->f);        break; \
    case CONSTELM:   temp = const_elem_exp_##operation(gf->f);      break; \
    case KERNEL:     temp = kernel_expansion_##operation(gf->f);    break; \
    }

#define GF_SWITCH_ONEOUT(operation,fc,O,A)                            \
    switch (fc){                                                        \
    case PIECEWISE:  O = piecewise_poly_##operation(A);      break; \
    case POLYNOMIAL: O = orth_poly_expansion_##operation(A); break; \
    case LINELM:     O = lin_elem_exp_##operation(A);        break; \
    case CONSTELM:   O = const_elem_exp_##operation(A);      break; \
    case KERNEL:     O = kernel_expansion_##operation(A);    break; \
    }

#define GF_OPTS_SWITCH_ONEOUT(operation,fc,O,A)                            \
    switch (fc){                                                        \
    case PIECEWISE:  O = pw_poly_opts_##operation(A);      break; \
    case POLYNOMIAL: O = ope_opts_##operation(A); break; \
    case LINELM:     O = lin_elem_exp_aopts_##operation(A);        break; \
    case CONSTELM:   O = const_elem_exp_aopts_##operation(A);      break; \
    case KERNEL:     O = kernel_approx_opts_##operation(A);    break; \
    }

#define GF_SWITCH_TWOOUT(operation,fc,O,A,B)                            \
    switch (fc){                                                        \
    case PIECEWISE:  O = piecewise_poly_##operation(A,B);      break; \
    case POLYNOMIAL: O = orth_poly_expansion_##operation(A,B); break; \
    case LINELM:     O = lin_elem_exp_##operation(A,B);        break; \
    case CONSTELM:   O = const_elem_exp_##operation(A,B);      break; \
    case KERNEL:     O = kernel_expansion_##operation(A,B);    break; \
    }

#define GF_SWITCH_THREEOUT(operation,fc,O,A,B,C)                       \
    switch (fc){                                                        \
    case PIECEWISE:  O = piecewise_poly_##operation(A,B,C);      break;  \
    case POLYNOMIAL: O = orth_poly_expansion_##operation(A,B,C); break;  \
    case LINELM:     O = lin_elem_exp_##operation(A,B,C);        break;  \
    case CONSTELM:   O = const_elem_exp_##operation(A,B,C);      break;  \
    case KERNEL:     O = kernel_expansion_##operation(A,B,C);    break;  \
    }

#define GF_SWITCH_NO_THREEOUT(operation,fc,A,B,C)                       \
    switch (fc){                                                        \
    case PIECEWISE:  piecewise_poly_##operation(A,B,C);      break;  \
    case POLYNOMIAL: orth_poly_expansion_##operation(A,B,C); break;  \
    case LINELM:     lin_elem_exp_##operation(A,B,C);        break;  \
    case CONSTELM:   const_elem_exp_##operation(A,B,C);      break;  \
    case KERNEL:     kernel_expansion_##operation(A,B,C);    break;  \
    }

#define GF_SWITCH_THREE_FRONT(operation,fc,A,B,C)                  \
    switch (fc){                                                    \
    case PIECEWISE:  operation##_piecewise_poly(A,B,C);      break; \
    case POLYNOMIAL: operation##_orth_poly_expansion(A,B,C); break; \
    case LINELM:     operation##_lin_elem_exp(A,B,C);        break; \
    case CONSTELM:   operation##_const_elem_exp(A,B,C);      break; \
    case KERNEL:     operation##_kernel_expansion(A,B,C);    break; \
    }


#define GF_SWITCH_NO_FOUROUT_FRONT(operation,fc,A,B,C,D)                 \
    switch (fc){                                                        \
    case PIECEWISE:  operation##_piecewise_poly(A,B,C,D);      break;    \
    case POLYNOMIAL: operation##_orth_poly_expansion(A,B,C,D); break;    \
    case LINELM:     operation##_lin_elem_exp(A,B,C,D);        break;    \
    case CONSTELM:   operation##_const_elem_exp(A,B,C,D);      break;    \
    case KERNEL:     operation##_kernel_expansion(A,B,C,D);    break;    \
    }

#define GF_SWITCH_THREEOUT_FRONT(operation,fc,O,A,B,C)                  \
    switch (fc){                                                        \
    case PIECEWISE:  O = operation##_piecewise_poly(A,B,C);      break; \
    case POLYNOMIAL: O = operation##_orth_poly_expansion(A,B,C); break; \
    case LINELM:     O = operation##_lin_elem_exp(A,B,C);        break; \
    case CONSTELM:   O = operation##_const_elem_exp(A,B,C);      break; \
    case KERNEL:     O = operation##_kernel_expansion(A,B,C);    break; \
    }

#define GF_SWITCH_TWOOUT_FRONT(operation,fc,O,A,B)                  \
    switch (fc){                                                        \
    case PIECEWISE:  O = operation##_piecewise_poly(A,B);      break; \
    case POLYNOMIAL: O = operation##_orth_poly_expansion(A,B); break; \
    case LINELM:     O = operation##_lin_elem_exp(A,B);        break; \
    case CONSTELM:   O = operation##_const_elem_exp(A,B);      break; \
    case KERNEL:     O = operation##_kernel_expansion(A,B);    break; \
    }

#define GF_SWITCH_FOUROUT(operation,fc,O,A,B,C,D)                       \
    switch (fc){                                                        \
    case PIECEWISE:  O = piecewise_poly_##operation(A,B,C,D);      break; \
    case POLYNOMIAL: O = orth_poly_expansion_##operation(A,B,C,D); break; \
    case LINELM:     O = lin_elem_exp_##operation(A,B,C,D);        break; \
    case CONSTELM:   O = const_elem_exp_##operation(A,B,C,D);      break; \
    case KERNEL:     O = kernel_expansion_##operation(A,B,C,D);    break; \
    }

#define GF_SWITCH_SIX(operation,fc,A,B,C,D,E,F)                       \
    switch (fc){                                                        \
    case PIECEWISE:  piecewise_poly_##operation(A,B,C,D,E,F);      break; \
    case POLYNOMIAL: orth_poly_expansion_##operation(A,B,C,D,E,F); break; \
    case LINELM:     lin_elem_exp_##operation(A,B,C,D,E,F);        break; \
    case CONSTELM:   const_elem_exp_##operation(A,B,C,D,E,F);      break; \
    case KERNEL:     kernel_expansion_##operation(A,B,C,D,E,F);    break; \
    }

#define GF_IN_OUT(operation) \
struct GenericFunction * generic_function_##operation(const struct GenericFunction * gf) \
{ \
    struct GenericFunction * out = NULL;                                       \
    out = generic_function_alloc(gf->dim, gf->fc);                             \
    switch (gf->fc){                                                           \
    case PIECEWISE:  out->f = piecewise_poly_##operation(gf->f);      break;   \
    case POLYNOMIAL: out->f = orth_poly_expansion_##operation(gf->f); break;   \
    case LINELM:     out->f = lin_elem_exp_##operation(gf->f);        break;   \
    case CONSTELM:   out->f = const_elem_exp_##operation(gf->f);      break;   \
    case KERNEL:     out->f = kernel_expansion_##operation(gf->f);    break;   \
    }                                                                          \
    return out;                                                                \
}

#define GF_IN_GENOUT(operation, typeout, init) \
typeout generic_function_##operation(const struct GenericFunction * gf) \
{ \
    typeout out = init;                            \
    switch (gf->fc){                                                    \
    case PIECEWISE:  out = piecewise_poly_##operation(gf->f);      break;   \
    case POLYNOMIAL: out = orth_poly_expansion_##operation(gf->f); break;   \
    case LINELM:     out = lin_elem_exp_##operation(gf->f);        break;   \
    case CONSTELM:   out = const_elem_exp_##operation(gf->f);      break;   \
    case KERNEL:     out = kernel_expansion_##operation(gf->f);    break;   \
    }                                                                          \
    return out;                                                                \
}

#endif
