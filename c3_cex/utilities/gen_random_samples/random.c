// Copyright (c) 2016, Sandia Corporation. Under the terms of Contract
// DE-AC04-94AL85000, there is a non-exclusive license for use of this
// work by or on behalf of the U.S. Government. Export of this program
// may require a license from the United States Government

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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <sys/time.h>

#include "array.h"


static char * program_name;

void print_code_usage (FILE *, int) __attribute__ ((noreturn));
void print_code_usage (FILE * stream, int exit_code)
{
    fprintf(stream, "Usage: %s options \n", program_name);
    fprintf(stream,
            " -h --help      Display this usage information.\n"
            " -t --type      \"uniform\" or \"normal\" (required).\n"
            " -r --rows      Number of rows (required).\n"
            " -c --cols      Number of columns (default 1).\n"
            " -l --lower     Lower bound for uniform (default 0).\n"
            " -u --upper     Upper bound for uniform (default 1).\n"
            " -s --stddev    Standard deviation for normal (default 1).\n"
            " -m --mean      Mean for normal (default 0).\n"
        );
    exit (exit_code);
}

int main(int argc, char * argv[])
{
    struct timeval t1;
    gettimeofday(&t1, NULL);
    unsigned int seed = t1.tv_usec * t1.tv_sec;
    srand(seed);
    // sample randomly to "clean things out"
    // could be just an old-wives thing
    // the numbers look more random to me after doing this
    // before, the first number of a uniform random sample was 
    // very similar amongst different runs
    for (size_t ii = 0; ii < 50; ii++){
        rand();
    }
    
    int next_option;
    const char * const short_options = "ht:r:c:l:u:s:m:";
    const struct option long_options[] = {
        { "help"   , 0, NULL, 'h' },
        { "type"   , 0, NULL, 't' },
        { "rows"   , 1, NULL, 'r' },
        { "cols"   , 1, NULL, 'c' },
        { "lower"  , 1, NULL, 'l' },
        { "upper"  , 1, NULL, 'u' },
        { "stddev" , 1, NULL, 's' },
        { "mean"   , 1, NULL, 'm' },
        { NULL     , 0, NULL, 0   }
    };

    char * type = NULL;
    size_t nrows = 0;
    size_t ncols = 1;
    double lower = 0.0;
    double upper = 1.0;
    double std = 1.0;
    double mean = 0.0;
    
    program_name = argv[0];
    do {
        next_option = getopt_long (argc, argv, short_options, long_options, NULL);
        switch (next_option)
        {
            case 'h': 
                print_code_usage(stdout, 0);
            case 't':
                type = optarg;
                break;
            case 'r':
                nrows = strtoul(optarg,NULL,10);
                break;
            case 'c':
                ncols = strtoul(optarg,NULL,10);
                break;
            case 'l':
                lower = atof(optarg);
                break;
            case 'u':
                upper = atof(optarg);
                break;
            case 's':
                std = atof(optarg);
                break;
            case 'm':
                mean = atof(optarg);
                break;
            case '?': // The user specified an invalid option 
                print_code_usage (stderr, 1);
            case -1: // Done with options. 
                break;
            default: // Something unexpected
                abort();
        }

    } while (next_option != -1);


    if (nrows == 0){
        fprintf(stderr, "Error: Must specify number of rows\n\n");
        print_code_usage(stderr, 1);
    }

    if (type == NULL){
        fprintf(stderr, "Error: Must specify type of random variable\n\n");
        print_code_usage(stderr, 1);
    }

    if (strcmp(type,"uniform") == 0){
        for (size_t ii = 0; ii < nrows; ii++){
            for (size_t jj = 0; jj < ncols; jj++){
                fprintf(stdout,"%3.15G ", randu()*(upper-lower) + lower);
            }
            fprintf(stdout,"\n");
        }
    }
    else if (strcmp(type,"normal") == 0){
        for (size_t ii = 0; ii < nrows; ii++){
            for (size_t jj = 0; jj < ncols; jj++){
                fprintf(stdout,"%3.15G ", randn()*std + mean);
            }
            fprintf(stdout,"\n");
        }
    }
    else{
        fprintf(stderr, "Unrecognized random variable %s\n ",type);
        print_code_usage(stderr,1);
    }

    return 0;
}
