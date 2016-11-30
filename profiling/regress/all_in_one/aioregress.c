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
            " -x --xtrain   Input file containing training locations (required) \n"
            " -y --ytrain   Input file containing training evaluations (required) \n"
            " -o --outfile  File to which to save the resulting function train \n"
            "               Does not save if this file is not specified\n"
            " -l --lower    Lower bound, same for every dimension (default -1.0)\n"
            " -u --upper    Upper bound, same for every dimension (default  1.0)\n"
            " -m --maxorder Maximum number of params in univariate approx (default 5)\n"
            " -r --rank     Starting rank for approximation (default 4)\n"
            " -v --verbose  Output words (default 0)\n"
        );
    exit (exit_code);
}

int main(int argc, char * argv[])
{
    int seed = 3;
    srand(seed);
    
    int next_option;
    const char * const short_options = "hx:y:o:l:u:m:r:v:";
    const struct option long_options[] = {
        { "help"    , 0, NULL, 'h' },
        { "xtrain"  , 1, NULL, 'x' },
        { "ytrain"  , 1, NULL, 'y' },
        { "outfile" , 1, NULL, 'o' },
        { "lower"   , 1, NULL, 'l' },
        { "upper"   , 1, NULL, 'u' },
        { "maxorder", 1, NULL, 'm' },
        { "rank"    , 1, NULL, 'r' },
        { "verbose" , 1, NULL, 'v' },
        { NULL      , 0, NULL, 0   }
    };

    char * xfile = NULL;
    char * yfile = NULL;
    char * outfile = NULL;
    program_name = argv[0];
    double lower = -1.0;
    double upper = 1.0;
    size_t maxorder = 5;
    size_t rank = 4;
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
            case 'y':
                yfile = optarg;
                break;
            case 'o':
                outfile = optarg;
                break;
            case 'l':
                lower = atof(optarg);
                break;
            case 'u':
                upper = atof(optarg);
                break;
            case 'm':
                maxorder = strtol(optarg,NULL,10);
                break;
            case 'r':
                rank = strtol(optarg,NULL,10);
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

    if ( (xfile == NULL) || (yfile == NULL)){
        fprintf(stderr, "Error: missing data files\n\n");
        print_code_usage(stderr,1);
    }
    
    FILE * fpx = fopen(xfile, "rt");
    if (fpx == NULL){
        fprintf(stderr,"Cannot open %s for reading data\n",xfile);
        return 1;
    }

    FILE * fpy = fopen(yfile, "rt");
    if (fpy == NULL){
        fprintf(stderr,"Cannot open %s for reading data\n",yfile);
        return 1;
    }

    
    size_t ndata, dim, trash;
    double * x = readfile_double_array(fpx,&ndata,&dim);
    double * y = readfile_double_array(fpy,&ndata,&trash);
    
    fclose(fpx);
    fclose(fpy);


    size_t * ranks = calloc_size_t(dim+1);
    for (size_t ii = 0; ii < dim+1; ii++){ ranks[ii] = rank; }
    ranks[0] = 1;
    ranks[dim] = 1;

    struct BoundingBox * bds = bounding_box_init(dim,lower,upper);
    struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    struct RegressAIO * aio = regress_aio_alloc(dim);
    regress_aio_add_data(aio,ndata,x,y);
    regress_aio_prep_memory(aio,a,1);
    size_t num_tot_params = regress_aio_get_num_params(aio);
    
    struct c3Opt * optimizer = c3opt_alloc(BFGS,num_tot_params);
    c3opt_set_verbose(optimizer,verbose-1);
    c3opt_add_objective(optimizer,regress_aio_LS,aio);

    double * guess = calloc_double(num_tot_params);
    for (size_t ii = 0; ii < num_tot_params; ii++){
        guess[ii] = randn();
    }

    double obj;
    int res = c3opt_minimize(optimizer,guess,&obj);
    if (verbose > 0){
        if (res < 0){
            printf("Warning: optimization did not terminate successfully (may not have reached a local minimum)\n");
        }
    }
    
    /* struct FunctionTrain * ft = function_train_round(regress_aio_get_ft(aio),1e-7,NULL); */
    struct FunctionTrain * ft = regress_aio_get_ft(aio);
    if (verbose > 0){
        double diff;
        double err = 0.0;
        double norm = 0.0;

        for (size_t ii = 0; ii < ndata; ii++){
            diff = y[ii] - function_train_eval(ft,x+ii*dim);
            err += diff*diff;
            norm += y[ii]*y[ii];
        }

        /* printf("rounded FT ranks are ");iprint_sz(dim+1,function_train_get_ranks(ft)); */
        printf("Relative error on training samples = %G\n",err/norm);
    }
    if (outfile != NULL){
        int res = function_train_save(ft,outfile);
        if (res != 1){
            fprintf(stderr,"Failure saving function train to file %s\n",outfile);
        }
    }
    
    free(x); x = NULL;
    free(y); y = NULL;
    /* function_train_free(ft); ft = NULL; */
    function_train_free(a); a = NULL;
    bounding_box_free(bds); bds = NULL;
    regress_aio_free(aio); aio = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    free(guess); guess = NULL;
    
    return 0;
}
