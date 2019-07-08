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





#include <stdio.h>
#include <stdlib.h>
#include "CuTest.h"
#include "unconstrained_functions.h"
#include "sgd_functions.h"

CuSuite * OptGetSuite(void);
CuSuite * BFGSGetSuite(void);
CuSuite * LBFGSGetSuite(void);
CuSuite * BGradGetSuite(void);
CuSuite * SGDGetSuite(void);

void RunAllTests(void) {
    
    printf("Running test suite for: lib_optimization\n");

    CuString * output = CuStringNew();
    CuSuite * suite = CuSuiteNew();
    
    CuSuite * opt = OptGetSuite();
    CuSuite * bfgs = BFGSGetSuite();
    CuSuite * lbfgs = LBFGSGetSuite();
    CuSuite * bgrad = BGradGetSuite();
    CuSuite * sgrad = SGDGetSuite();

    /* CuSuiteAddSuite(suite, opt); */

    create_unc_probs();

    /* CuSuiteAddSuite(suite, lbfgs); */
    /* CuSuiteAddSuite(suite, bfgs); */

    create_sgd_probs();
    CuSuiteAddSuite(suite, sgrad);

    // batch gradient doesn't work
    /* CuSuiteAddSuite(suite, bgrad); //something is wrong */

    CuSuiteRun(suite);
    CuSuiteSummary(suite, output);
    CuSuiteDetails(suite, output);
    printf("%s \n", output->buffer);
    
    CuSuiteDelete(opt);
    CuSuiteDelete(bfgs);
    CuSuiteDelete(lbfgs);
    CuSuiteDelete(bgrad);
    CuSuiteDelete(sgrad);
    
    CuStringDelete(output);
    free(suite);
   
}

int main(void) {

    RunAllTests();
}
