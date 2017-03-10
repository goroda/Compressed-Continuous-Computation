#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <getopt.h>

#include "stringmanip.h"
#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"

static char * program_name;

void print_code_usage (FILE *, int) __attribute__ ((noreturn));
void print_code_usage (FILE * stream, int exit_code)
{

    fprintf(stream, "Usage: %s options \n", program_name);
    fprintf(stream,
            " -h --help     Display this usage information.\n"
            " -f --ft       Input file containing saved function train (required) \n"
            " -x --xtest    Input file containing locations at which to evaluate the FT  \n"
            " -o --output   Output file, if not specified then to stdout\n"
            " -v --verbose  Output words (default 0)\n"
        );
    exit (exit_code);
}

int main(int argc, char * argv[])
{

    int next_option;
    const char * const short_options = "hf:x:o:v:";
    const struct option long_options[] = {
        { "help"    , 0, NULL, 'h' },
        { "ft"      , 1, NULL, 'f' },
        { "xtest"   , 1, NULL, 'x' },
        { "output"  , 1, NULL, 'o' },
        { "verbose" , 1, NULL, 'v' },
        { NULL      , 0, NULL, 0   }
    };

    char * infile = NULL;
    char * xfile = NULL;
    char * outfile = NULL;
    program_name = argv[0];
    int verbose = 0;
    do {
        next_option = getopt_long (argc, argv, short_options, long_options, NULL);
        switch (next_option)
        {
            case 'h': 
                print_code_usage(stdout, 0);
            case 'x':
                xfile = optarg;
                break;
            case 'f':
                infile = optarg;
                break;
            case 'o':
                outfile = optarg;
                break;
            case 'v':
                verbose = strtol(optarg,NULL,10);
                break;
            case '?': // The user specified an invalid option 
                print_code_usage (stderr, 1);
            case -1: // Done with options. 
                break;
            default: // Something unexpected
                abort();
        }

    } while (next_option != -1);

    if (verbose > 0){
        fprintf(stderr,"Cool\n");
    }
    
    if (infile == NULL){
        fprintf(stderr, "Error: Missing FT input file\n\n");
        print_code_usage(stderr,1);
    }

    /* printf("load\n"); */
    struct FunctionTrain * ft = function_train_load(infile);
    if (ft == NULL){
        fprintf(stderr, "Failure loading %s\n",infile);
        return 1;
    }
    /* printf("loaded\n"); */

    double * xtest = NULL;
    size_t ntest, dtest;
    if (xfile != NULL){
        FILE * fp = fopen(xfile,"rt");
        if (fp == NULL){
            fprintf(stderr, "Could not open file %s\n",xfile);
            return 1;
        }
        xtest = readfile_double_array(fp,&ntest,&dtest);
        if (dtest != ft->dim){
            fprintf(stderr,"Number of columns of testing points is not the same as the\n");
            fprintf(stderr,"dimension of the function train\n");
            return 1;
        }
        fclose(fp);


        FILE * fpout = stdout;
        if (outfile != NULL){
            fpout = fopen(outfile,"wt");
            if (fpout == NULL){
                fprintf(stderr, "Cannot open file %s to write output\n",outfile);
                print_code_usage(stderr,1);
            }
        }
        for (size_t ii = 0; ii < ntest; ii++){
            /* dprint(ft->dim, xtest+ii*ft->dim); */
            double val = function_train_eval(ft,xtest+ii*ft->dim);
            fprintf(fpout,"%3.15G\n",val);
        }
        if (outfile != NULL){
            fclose(fpout);
        }
    }

    function_train_free(ft); ft = NULL;
    return 0;
}
