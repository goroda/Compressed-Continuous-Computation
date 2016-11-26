#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <getopt.h>

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
            " -h --help      Display this usage information.\n"
            " -i --input     Input file"
            " -v --verbose   Output words (default 0)\n"
        );
    exit (exit_code);
}

int main(int argc, char * argv[])
{

    int next_option;
    const char * const short_options = "hi:v:";
    const struct option long_options[] = {
        { "help"    , 0, NULL, 'h' },
        { "input"   , 1, NULL, 'i' },
        { "verbose" , 1, NULL, 'v' },
        { NULL      , 0, NULL, 0   }
    };

    char * infile = "data.dat";
    program_name = argv[0];
    int verbose = 0;
    do {
        next_option = getopt_long (argc, argv, short_options, long_options, NULL);
        switch (next_option)
        {
            case 'h': 
                print_code_usage(stdout, 0);
            case 'i':
                infile = optarg;
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

    FILE * fp = fopen(infile, "rt");
    if (fp == NULL){
        fprintf(stderr,"Cannot open %s for reading data\n",infile);
        return 1;
    }


    return 0;
}
