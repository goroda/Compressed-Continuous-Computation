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
            " -h --help                Display this usage information.\n"
            " -x --xtrain   <filename> Input file containing training locations (required) \n"
            " -y --ytrain   <filename> Input file containing training evaluations (required) \n"
            " -o --outfile  <filename> File to which to save the resulting function train \n"
            "                          Does not save if this file is not specified\n"
            " -l --lower    <val>      Lower bound, same for every dimension (default -1.0)\n"
            " -u --upper    <val>      Upper bound, same for every dimension (default  1.0)\n"
            " -m --maxorder <val>      Maximum number of params in univariate approx (default 5)\n"
            " -r --rank     <val>      Starting rank for approximation (default 4)\n"
            "    --cv                  Cross validate \n"
            " -v --verbose  <val>      Output words (default 0)\n"
        );
    exit (exit_code);
}

#define CV 1000
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
        { "cv"      , 0, NULL,  CV },
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
    int cv = 0;
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
            case CV:
                cv = 1;
                break;
            case '?': // The user specified an invalid option
                printf("invalid option %s\n\n",optarg);
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


    // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lower);
    ope_opts_set_ub(opts,upper);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    size_t * ranks = calloc_size_t(dim+1);
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        ranks[ii] = rank;
    }
    ranks[0] = 1;
    ranks[dim] = 1;

    size_t nparams = (rank*rank * (dim-2) + 2*rank) * (maxorder+1);
    double * params_start = calloc_double(nparams);

    
    for (size_t ii = 0; ii < nparams; ii++){
        params_start[ii] = 0.01*randn();
    }

    /* struct FTparam * ftp = ft_param_alloc(dim,fapp,params_start,ranks); */
    /* struct RegressOpts * ropts = regress_opts_create(AIO,FTLS,ndata,dim,x,y); */

    struct FTRegress * ftr = ft_regress_alloc(dim,fapp);
    ft_regress_set_type(ftr,AIO);
    ft_regress_set_obj(ftr,FTLS);
    ft_regress_set_data(ftr,ndata,x,1,y,1);
    ft_regress_set_discrete_parameter(ftr,"rank",rank);
    ft_regress_set_discrete_parameter(ftr,"num_param",maxorder+1);
    ft_regress_set_discrete_parameter(ftr,"opt maxiter",1000);
    ft_regress_process_parameters(ftr);


    // choose parameters using cross validation
    if (cv == 1){
        assert (1 == 0);
        /* size_t kfold = 10; */
        /* struct CrossValidate * cv = */
        /*     cross_validate_init(ndata,dim,x,y,kfold); */

        /* // set options for parameters */
        /* size_t norder_ops = 6; */
        /* size_t order_ops[6] = {1,2,3,4,5,6}; */
        /* /\* size_t nranks = 4; *\/ */
        /* /\* size_t rank_ops[6] ={1,2,3,4,5,6}; *\/ */
        /* size_t nranks = 4; */
        /* size_t rank_ops[4] ={1,3,5,6}; */
        /* /\* size_t nmiters = 3; *\/ */
        /* /\* size_t miter_ops[3]={200,500,1000}; *\/ */

        /* cross_validate_add_discrete_param(cv,"num_param",norder_ops, */
        /*                                   order_ops); */
        /* cross_validate_add_discrete_param(cv,"rank",nranks,rank_ops); */
        /* /\* cross_validate_add_discrete_param(cv,"opt maxiter", *\/ */
        /* /\*                                      nmiters,miter_ops); *\/ */
        /* cross_validate_opt(cv,ftr,verbose); */

        /* cross_validate_free(cv); cv = NULL; */
    }

    
    struct FunctionTrain * ft = ft_regress_run(ftr);
    /* struct FunctionTrain * ft = c3_regression_run(ftp,ropts); */
    
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
        /* printf("Relative error on training samples = %G\n",err/norm); */
    }
    if (outfile != NULL){
        int res = function_train_save(ft,outfile);
        if (res != 1){
            fprintf(stderr,"Failure saving function train to file %s\n",outfile);
        }
    }
    
    free(x); x = NULL;
    free(y); y = NULL;
    free(ranks); ranks = NULL;
    
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    ft_regress_free(ftr); ftr = NULL;
    /* ft_param_free(ftp);      ftp  = NULL; */
    /* regress_opts_free(ropts); ropts = NULL; */


    
    return 0;
}
