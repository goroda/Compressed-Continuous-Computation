#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <getopt.h>

#include "stringmanip.h"
#include "array.h"
/* #include "linalg.h" */
/* #include "lib_clinalg.h" */
/* #include "lib_funcs.h" */

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
            " -e --evalfile <filename> File with locations to evaluate, writes evaluations to <filename>.evals \n"
            " -l --lower    <val>      Lower bound, same for every dimension (default -1.0)\n"
            " -u --upper    <val>      Upper bound, same for every dimension (default  1.0)\n"
            " -n --numparam <val>      Number of parameters in univariate approx (default 5)\n"
            " -r --rank     <val>      Starting rank for approximation (default 4)\n"
            " -a --alg      <string>   Algorithm (AIO or ALS)\n"
            " -d --adapt    <bool>     Adaptation, in this case rank is upper bound\n"
            "    --reg                 Regularization weight\n"
            "    --cv-kfold <int>      Specify k for kfold cross validation \n"
            "    --cv-rank  <int>      Add rank option with which to cross validate \n"
            "    --cv-reg   <dbl>      Regularization parameter to optimizive over with CV\n"
            "    --cv-num   <int>      Add number of univariate params option with which to cross validate \n"
            " -b --basis    <string>   basis for univariate functions (poly or kernel) \n"
            " -t --tol      <dbl>      Optimization tolerance\n"
            " -v --verbose  <val>      Output words (default 0)\n"
        );
    exit (exit_code);
}

#define CVK 1000
#define CVR 1001
#define CVN 1002
#define REG 2000
#define CVREG 2001
int main(int argc, char * argv[])
{
    unsigned int seed = 3;
    srand(seed);

    int next_option;
    const char * const short_options = "hx:y:o:e:l:u:n:r:d:a:v:b:t:";
    const struct option long_options[] = {
        { "help"     , 0, NULL, 'h' },
        { "xtrain"   , 1, NULL, 'x' },
        { "ytrain"   , 1, NULL, 'y' },
        { "outfile"  , 1, NULL, 'o' },
        { "evalfile" , 1, NULL, 'e'},
        { "lower"    , 1, NULL, 'l' },
        { "upper"    , 1, NULL, 'u' },
        { "numparam" , 1, NULL, 'n' },
        { "adapt"    , 1, NULL, 'd' },
        { "rank"     , 1, NULL, 'r' },
        { "alg"      , 1, NULL, 'a' },
        { "reg"      , 1, NULL,  REG },
        { "cv-kfold" , 1, NULL,  CVK },
        { "cv-rank"  , 1, NULL,  CVR },
        { "cv-num"   , 1, NULL,  CVN },
        { "cv-reg"   , 1, NULL,  CVREG },
        { "basis"    , 1, NULL, 'b'},
        { "tol"      , 1, NULL, 't'},
        { "verbose"  , 1, NULL, 'v' },
        { NULL       , 0, NULL, 0   }
    };

    char * xfile = NULL;
    char * yfile = NULL;
    char * outfile = NULL;
    char * evalfile = NULL;
    program_name = argv[0];
    double lower = -1.0;
    double upper = 1.0;
    size_t numparam = 5;
    size_t rank = 4;
    int verbose = 0;

    double tol = 1e-10;
    size_t nranksalloc = 10;
    size_t * CVranks = calloc_size_t(nranksalloc);
    size_t cvrank = 0;

    size_t nnumsalloc=10;
    size_t * CVnums  = calloc_size_t(nnumsalloc);
    size_t cvnum=0;

    size_t nregalloc=10;
    double * CVreg  = calloc_double(nregalloc);
    size_t cvreg=0;

    size_t kfold = 5;
    double reg = 0.0;
    char alg[80] = "AIO";
    
    size_t adapt = 0;
    enum function_class fc = POLYNOMIAL;
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
            case 'e':
                evalfile = optarg;
                break;
            case 'l':
                lower = atof(optarg);
                break;
            case 'u':
                upper = atof(optarg);
                break;
            case 'n':
                numparam = strtoul(optarg,NULL,10);
                break;
            case 'r':
                rank = strtoul(optarg,NULL,10);
                break;
            case 't':
                tol = atof(optarg);
                break;
            case 'a':
                strcpy(alg,optarg);
                break;
            case 'd':
                adapt = strtoul(optarg,NULL,10);
                break;
            case 'b':
                if (strcmp(optarg,"kernel") == 0){
                    fc = KERNEL;
                }
                else if (strcmp(optarg,"poly") == 0){
                    fc = POLYNOMIAL;
                }
                else{
                    fprintf(stderr, "Error: basis type %s unrecognized\n",optarg);
                    print_code_usage(stderr,1);
                }
                break;
            case 'v':
                verbose = strtol(optarg,NULL,10);
                break;
            case REG:
                reg = atof(optarg);
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
            case CVREG:
                cvreg += 1;
                if (cvreg > nregalloc){
                    double * nnew = calloc_double(nregalloc);
                    memmove(nnew,CVreg,nregalloc*sizeof(double));
                    free(CVreg); CVreg = NULL;
                    CVreg = calloc_double(2*nregalloc);
                    memmove(CVreg,nnew,nregalloc*sizeof(double));
                    free(nnew); nnew = NULL;
                    nregalloc = 2 * nregalloc;
                }
                CVreg[cvreg-1] = atof(optarg);
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

    if (verbose > 0){
        printf("Data size: %zu x %zu \n",ndata,dim);
    }

    
    // Initialize Approximation Structure
    struct OpeOpts * opts = NULL;

    if (fabs(upper - lower) < 1e-15){
        printf("HERMITE!!!!!\n");
        opts = ope_opts_alloc(HERMITE);
    }
    else{
        opts = ope_opts_alloc(LEGENDRE);
        ope_opts_set_lb(opts,lower);
        ope_opts_set_ub(opts,upper);
    }
    
    ope_opts_set_nparams(opts,numparam);

    // Initialize kernel opts incase
    double * centers = linspace(lower,upper,numparam);
    double scale = 1.0;
    double width = pow(numparam,-0.2)*(upper-lower)/sqrt(12.0);
    width *= 1;
    struct KernelApproxOpts * kopts = kernel_approx_opts_gauss(numparam,centers,scale,width);

    struct OneApproxOpts * qmopts = NULL;
    if (fc == POLYNOMIAL){
        qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    }
    else if (fc == KERNEL){
        qmopts = one_approx_opts_alloc(KERNEL,kopts);
    }
    
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    size_t * ranks = calloc_size_t(dim+1);
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        if (adapt == 0){
            ranks[ii] = rank;
        }
        else{
            ranks[ii] = 2;
        }
    }
    ranks[0] = 1;
    ranks[dim] = 1;

    size_t opt_maxiter=50000;
    struct FTRegress * ftr = ft_regress_alloc(dim,fapp,ranks);
    if (strcmp(alg,"AIO") == 0){
        if ((reg > 0) || (cvreg > 0)){
            ft_regress_set_alg_and_obj(ftr,AIO,FTLS_SPARSEL2);
        }
        else{
            ft_regress_set_alg_and_obj(ftr,AIO,FTLS);
        }
    }
    else if (strcmp(alg,"ALS") == 0){
        if ((reg > 0) || (cvreg > 0)){
            ft_regress_set_alg_and_obj(ftr,ALS,FTLS_SPARSEL2);
        }
        else{
            ft_regress_set_alg_and_obj(ftr,ALS,FTLS);
        }
    }    
    else{
        fprintf(stderr,"\n\nAlgorithm %s is not recognized\n",alg);
        print_code_usage(stderr, 1);      
    }

        
    ft_regress_set_max_als_sweep(ftr,20);
    ft_regress_set_verbose(ftr,verbose);
    if (reg > 0){
        ft_regress_set_regularization_weight(ftr,reg);
    }

    if (adapt != 0){
        ft_regress_set_adapt(ftr,1);
        ft_regress_set_roundtol(ftr,1e-8);
        ft_regress_set_maxrank(ftr,rank);
        ft_regress_set_kickrank(ftr,1);        
    }
    
    struct c3Opt * optimizer = c3opt_create(BFGS);
    /* struct c3Opt * optimizer = c3opt_create(SGD); */
    if (verbose > 5){
        c3opt_set_verbose(optimizer,1);
    }
    /* ft_regress_set_stoch_obj(ftr,1); */
    /* c3opt_set_sgd_nsamples(optimizer,ndata); */
    opt_maxiter = 1000;
    c3opt_set_maxiter(optimizer,opt_maxiter);
    c3opt_set_gtol(optimizer,tol);
    c3opt_set_relftol(optimizer,tol);
    c3opt_set_absxtol(optimizer,tol);
    /* c3opt_ls_set_maxiter(optimizer,10); */
    
    // choose parameters using cross validation
    struct CrossValidate * cv = NULL;
    if ((cvrank > 0) || (cvnum > 0) || (cvreg > 0)){

        int cvverbose = 0;
        cv = cross_validate_init(ndata,dim,x,y,kfold,cvverbose);

        struct CVOptGrid * cvgrid = cv_opt_grid_init(3);
        if (verbose > 2){
            cv_opt_grid_set_verbose(cvgrid,verbose-1);
        }
        
        if (cvnum > 0){ // just crossvalidate on cv num
            cv_opt_grid_add_param(cvgrid,"num_param",cvnum,CVnums);
        }
        if (cvrank > 0){ // just cross validate on ranks
            cv_opt_grid_add_param(cvgrid,"rank",cvrank,CVranks);            
        }
        if (cvreg > 0){ // cv on regularization parameter
            cv_opt_grid_add_param(cvgrid,"reg_weight",cvreg,CVreg);            
        }

        cross_validate_grid_opt(cv,cvgrid,ftr,optimizer);

        cross_validate_free(cv); cv = NULL; 
        cv_opt_grid_free(cvgrid); cvgrid = NULL;        
    }

    struct FunctionTrain * ft = NULL;
    ft = ft_regress_run(ftr,optimizer,ndata,x,y);
    
    if (verbose > 0){
        double diff;
        double err = 0.0;
        double norm = 0.0;
        /* struct FunctionTrain * rounded = function_train_round(ft,1e-1,fapp); */
        for (size_t ii = 0; ii < ndata; ii++){
            diff = y[ii] - function_train_eval(ft,x+ii*dim);
            err += diff*diff;
            norm += y[ii]*y[ii];
        }

        printf("Relative error on training samples = %G\n",err/norm);
        /* printf("rounded FT ranks are ");iprint_sz(dim+1,function_train_get_ranks(rounded)); */
        /* function_train_free(rounded); rounded = NULL; */
    }
    if (outfile != NULL){
        int res = function_train_save(ft,outfile);
        if (res != 1){
            fprintf(stderr,"Failure saving function train to file %s\n",outfile);
        }

        /* printf("load function train\n"); */
        /* struct FunctionTrain * f2 = function_train_load(outfile); */
        /* printf("loaded\n"); */
    }

    if (evalfile != NULL){

        FILE * evalin = fopen(evalfile,"rt");
        if (evalin == NULL){
            fprintf(stderr, "Cannot open %s for reading\n",evalfile);
        }
        size_t neval, dimin;        
        double * z = readfile_double_array(evalin,&neval,&dimin);
        if (dimin != dim){
            fprintf(stderr, "Error: File containing testing evaluation locations\n");
            fprintf(stderr, "has more columns than dimension of training data");
            exit(1);
        }

        char evaloutfile[256];
        sprintf(evaloutfile,"%s.evals",evalfile);

        /* printf("evaloutfile = %s\n",evaloutfile); */
        FILE * evalout = fopen(evaloutfile,"wt");
        if (evalout == NULL){
            fprintf(stderr, "Error: cannot open %s for writing evaluations\n",evaloutfile);
            exit(1);
        }


        for (size_t ii = 0; ii < neval; ii++){
            double eval = function_train_eval(ft,z+ii*dim);
            for (size_t jj = 0; jj < dim; jj++){
                fprintf(evalout,"%3.15G ",z[ii*dim+jj]);
            }
            fprintf(evalout,"%3.15G\n",eval);
        }
        
        fclose(evalin);
        fclose(evalout);
        
    }
    
    free(x); x = NULL;
    free(y); y = NULL;
    free(ranks); ranks = NULL;

    function_train_free(ft);
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    ft_regress_free(ftr); ftr = NULL;
    free(CVranks); CVranks = NULL;
    free(CVnums); CVnums = NULL;
    free(CVreg); CVreg = NULL;

    free(centers); centers = NULL;
    c3opt_free(optimizer); optimizer = NULL;

    return 0;
}
