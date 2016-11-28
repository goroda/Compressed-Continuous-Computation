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
            " -h --help      Display this usage information.\n"
            " -i --input     Input file\n"
            " -v --verbose   Output words (default 0)\n"
        );
    exit (exit_code);
}

void split_x_y(double * data, size_t nrows, size_t ncols, double **x, double **y)
{
    size_t dim = ncols-1;
    *x = calloc_double(nrows * dim);
    *y = calloc_double(nrows);
    for (size_t ii = 0; ii < nrows; ii++){
        for (size_t jj = 0; jj < dim; jj++){
            (*x)[ii*dim+jj] = data[ii * ncols + jj];
        }
        (*y)[ii] = data[(ii+1)*ncols - 1];
    }
}


void get_regression_data_from_file(FILE * fp, size_t * ndata, size_t * dim,
                                   double **x, double **y)
{
    // read rows first
    size_t ncols;
    double * vals = readfile_double_array(fp,ndata,&ncols);
    *dim = ncols-1;
    split_x_y(vals,*ndata,ncols,x,y);
    free(vals); vals = NULL;
}

int main(int argc, char * argv[])
{
    int seed = 4;
    srand(seed);
    
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

    size_t ndata, dim;
    double * x = NULL;
    double * y = NULL;
    get_regression_data_from_file(fp,&ndata,&dim,&x,&y);
    
    double lb = -1.0;
    double ub = 1.0;
    size_t ranks[6] = {1,2,2,2,2,1};
    size_t maxorder = 10;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    struct RegressAIO * aio = regress_aio_alloc(dim);
    regress_aio_add_data(aio,ndata,x,y);
    regress_aio_prep_memory(aio,a,1);
    size_t num_tot_params = regress_aio_get_num_params(aio);
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,num_tot_params);
    c3opt_set_verbose(optimizer,1);
    c3opt_add_objective(optimizer,regress_aio_LS,aio);

    double * guess = calloc_double(num_tot_params);
    for (size_t ii = 0; ii < num_tot_params; ii++){
        guess[ii] = randn();
    }

    double obj;
    int res = c3opt_minimize(optimizer,guess,&obj);

    free(x); x = NULL;
    free(y); y = NULL;
    function_train_free(a); a = NULL;
    bounding_box_free(bds); bds = NULL;
    regress_aio_free(aio); aio = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    free(guess); guess = NULL;
    

    return 0;
}
