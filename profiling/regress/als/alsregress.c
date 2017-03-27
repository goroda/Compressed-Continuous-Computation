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

#include "regress.h"

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
            " -m --maxorder <val>      Maximum order of polynomial in univariate approx (default 5)\n"
            " -r --rank     <val>      Starting rank for approximation (default 4)\n"
            "    --cv-kfold <int>      Specify k for kfold cross validation \n"
            "    --cv-rank  <int>      Add rank option with which to cross validate \n"
            "    --cv-num   <int>      Add number of univariate params option with which to cross validate \n"
            " -v --verbose  <val>      Output words (default 0)\n"
        );
    exit (exit_code);
}

#define CVK 1000
#define CVR 1001
#define CVN 1002
int main(int argc, char * argv[])
{
    unsigned int seed = 3;
    srand(seed);
    
    int next_option;
    const char * const short_options = "hx:y:o:l:u:m:r:v:";
    const struct option long_options[] = {
        { "help"     , 0, NULL, 'h' },
        { "xtrain"   , 1, NULL, 'x' },
        { "ytrain"   , 1, NULL, 'y' },
        { "outfile"  , 1, NULL, 'o' },
        { "lower"    , 1, NULL, 'l' },
        { "upper"    , 1, NULL, 'u' },
        { "maxorder" , 1, NULL, 'm' },
        { "rank"     , 1, NULL, 'r' },
        { "cv-kfold" , 1, NULL,  CVK },
        { "cv-rank"  , 1, NULL,  CVR },
        { "cv-num"   , 1, NULL,  CVN },
        { "verbose"  , 1, NULL, 'v' },
        { NULL       , 0, NULL, 0   }
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

    size_t nranksalloc = 10;
    size_t * CVranks = calloc_size_t(nranksalloc);
    size_t cvrank = 0;

    size_t nnumsalloc=10;
    size_t * CVnums  = calloc_size_t(nnumsalloc);
    size_t cvnum=0;

    size_t kfold = 5;
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
                maxorder = strtoul(optarg,NULL,10);
                break;
            case 'r':
                rank = strtoul(optarg,NULL,10);
                break;
            case 'v':
                verbose = strtol(optarg,NULL,10);
                break;
            case CVK:
                kfold = strtoul(optarg,NULL,10);
                break;
            case CVR:
                cvrank += 1;
                if (cvrank > nranksalloc){
                    size_t * nnew = calloc_size_t(nranksalloc);
                    memmove(nnew,CVranks,nranksalloc*sizeof(size_t));
                    free(CVranks); CVranks = NULL;
                    CVranks = calloc_size_t(2*nranksalloc);
                    memmove(CVranks,nnew,nranksalloc*sizeof(size_t));
                    free(nnew); nnew = NULL;
                    nranksalloc = 2 * nranksalloc;
                }
                CVranks[cvrank-1] = strtoul(optarg,NULL,10);
                break;
            case CVN:
                cvnum += 1;
                if (cvnum > nnumsalloc){
                    size_t * nnew = calloc_size_t(nnumsalloc);
                    memmove(nnew,CVnums,nnumsalloc*sizeof(size_t));
                    free(CVnums); CVnums = NULL;
                    CVnums = calloc_size_t(2*nnumsalloc);
                    memmove(CVnums,nnew,nnumsalloc*sizeof(size_t));
                    free(nnew); nnew = NULL;
                    nnumsalloc = 2 * nnumsalloc;
                }
                CVnums[cvnum-1] = strtoul(optarg,NULL,10);
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


    size_t opt_maxiter=1000;
    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_maxiter(optimizer,opt_maxiter);
    struct FTRegress * ftr = ft_regress_alloc(dim,fapp,ranks);
    ft_regress_set_alg_and_obj(ftr,ALS,FTLS);

    /* ft_regress_set_als_maxsweep(ftr,20); */
    ft_regress_set_verbose(ftr,verbose);

    // choose parameters using cross validation
    struct CrossValidate * cv = NULL;
    if ((cvrank > 0) || (cvnum > 0)){
        int cvverbose = 0;
        cv = cross_validate_init(ndata,dim,x,y,kfold,cvverbose);

        struct CVOptGrid * cvgrid = cv_opt_grid_init(3);
        
        if (cvnum > 0){ // just crossvalidate on cv num
            cv_opt_grid_add_param(cvgrid,"num_param",cvnum,CVnums);
        }
        if (cvrank > 0){ // just cross validate on ranks
            cv_opt_grid_add_param(cvgrid,"rank",cvrank,CVranks);            
        }

        cross_validate_grid_opt(cv,cvgrid,ftr,optimizer);

        cross_validate_free(cv); cv = NULL; 
        cv_opt_grid_free(cvgrid); cvgrid = NULL;
    }

    
    struct FunctionTrain * ft = ft_regress_run(ftr,optimizer,ndata,x,y);


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

    function_train_free(ft); ft = NULL;
    free(CVranks); CVranks = NULL;
    free(CVnums); CVnums = NULL;
    
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    ft_regress_free(ftr); ftr = NULL;
    c3opt_free(optimizer);

    
    return 0;
}
