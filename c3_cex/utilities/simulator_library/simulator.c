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

#include <getopt.h>

#include "c3.h"
#include "functions.h"

static char * program_name;

void print_code_usage (FILE *, int) __attribute__ ((noreturn));
void print_code_usage (FILE * stream, int exit_code)
{
    fprintf(stream, "A Library of Functions Useful for Testing Approximation Routines. \n\n");
    fprintf(stream, "Usage: %s options \n", program_name);
    fprintf(stream,
            " -h --help      Display this usage information.\n"
            " -f --function  Which function to evaluate. \n"
            " -i --input     Input file.\n"
            " -o --output    Output file.\n"
            " -n --number    Number of data points (default 100).\n"
            " -v --verbose   Output words (default 0).\n"
            "                if -1 print the number of dimensions\n"
            "                if -2 print the name of the function\n"
            "                if -3 print the lower bound of input\n"
            "                if -4 print the upper bound of input\n"
        );
    fprintf(stream,
            "\n\nFunction options include:\n");
    for (size_t ii = 0; ii < num_funcs; ii++){
        fprintf(stream, "Function: %zu\n\n",ii+1);
        fprintf(stream, "%s",funcs[ii].message);

        fprintf(stream, "************\n");
    }
    exit (exit_code);
}

int main(int argc, char * argv[])
{
    unsigned int seed = 4;
    srand(seed);
    
    int next_option;
    const char * const short_options = "hf:i:o:n:v:";
    const struct option long_options[] = {
        { "help"     , 0, NULL, 'h' },
        { "function" , 0, NULL, 'f' },
        { "input"    , 1, NULL, 'i' },
        { "output"   , 1, NULL, 'o' },
        { "number"   , 1, NULL, 'n' },
        { "verbose"  , 1, NULL, 'v' },
        { NULL       ,  0, NULL, 0   }
    };

    char * infile   = NULL;
    size_t function = 0;
    char * outfile  = "data.dat";
    size_t num = 100;
    int verbose = 0;
    program_name = argv[0];
    do {
        next_option = getopt_long (argc, argv, short_options, long_options, NULL);
        switch (next_option)
        {
            case 'h': 
                print_code_usage(stdout, 0);
            case 'f':
                function = strtoul(optarg,NULL,10);
                break;
            case 'i':
                infile = optarg;
                break;
            case 'o':
                outfile = optarg;
                break;
            case 'n':
                num = strtoul(optarg,NULL,10);
                break;
            case 'v':
                verbose = strtoul(optarg,NULL,10);
                break;
            case '?': // The user specified an invalid option 
                print_code_usage (stderr, 1);
            case -1: // Done with options. 
                break;
            default: // Something unexpected
                abort();
        }

    } while (next_option != -1);

    // Initialized functions
    create_functions();
    if (function == 0){
        fprintf(stderr, "Must specify function from which to generate data\n\n");
        print_code_usage(stderr, 1);
    }
    if (function > num_funcs){
        fprintf(stderr, "Requested function does not exist\n\n");
        print_code_usage(stderr, 1);
    }
    size_t dim = function_get_dim(&(funcs[function-1]));
    if (verbose == -1){
        printf("%zu\n",dim);
        exit(0);
    }
    else if (verbose == -2){
        printf("%s\n",function_get_name(&funcs[function-1]));
        exit(0);
    }
    else if (verbose == -3){
        printf("%G\n",function_get_lower(&funcs[function-1]));
        exit(0);
    }
    else if (verbose == -4){
        printf("%G\n",function_get_upper(&funcs[function-1]));
        exit(0);
    }
    
    size_t nrows,ncols;
    double * xin = NULL;
    double * x = calloc_double(num * dim);
    if (infile != NULL){
        FILE * fp = fopen(infile, "rt");
        if (fp == NULL){
            fprintf(stderr,"Cannot open %s for reading data\n",infile);
            return 1;
        }
        xin = readfile_double_array(fp,&nrows,&ncols);

        if (verbose > 0){
            fprintf(stdout,"Read %zu out of %zu evaluation locations\n",num,nrows);
        }

        if (nrows < num){
            fprintf(stderr,"Cannot read %zu data points, file only contains %zu\n",num,nrows);
        }

        if (ncols < dim){
            fprintf(stderr,"Cannot read points of dim %zu, file only has %zu cols.\n",dim,ncols);
        }


        for (size_t ii = 0; ii < num; ii++){
            for (size_t jj = 0; jj < dim; jj++){
                x[ii*dim+jj] = xin[ii*ncols+jj];
            }
        }
        free(xin); xin = NULL;
        fclose(fp);
    }
    else{
        fprintf(stderr,"Missing: input file with locations to evaluate function\n");
        print_code_usage(stderr, 1);
    }

    FILE * fp = fopen(outfile, "wt");
    if (fp == NULL){
        fprintf(stderr,"Cannot open %s for writing data\n",outfile);
        return 1;
    }

    // evaluate functions
    double * vals = calloc_double(num);
    function_eval(num,x,vals,&(funcs[function-1]));
    for (size_t ii = 0; ii < num; ii++){
        fprintf(fp,"%3.15G\n",vals[ii]);
    }
    free(vals);
    fclose(fp);
    free(x);
    /* else{ */
    /*     for (size_t ii = 0; ii < ) */
    /* } */
}
