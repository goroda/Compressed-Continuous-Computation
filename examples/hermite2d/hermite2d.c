#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <getopt.h>

#include "c3.h"

static char * program_name;

void print_code_usage (FILE *, int) __attribute__ ((noreturn));
void print_code_usage (FILE * stream, int exit_code)
{

    fprintf(stream, "Usage: %s options \n", program_name);
    fprintf(stream,
            " -h --help      Display this usage information.\n"
            " -d --directory Output directory (defaults to .)\n"
            " -f --function  Which function to approximate \n"
            "                0: (default) x + y \n"
            "                1: x*y \n"
            "                2: sin(xy)\n"
            " -v --verbose   Output words (default 0)\n"
        );
    exit (exit_code);
}

double f0(double *x, void * args)
{
    assert(args == NULL);
    return x[0] + x[1];
}

double f1(double *x, void * args)
{
    assert(args == NULL);
    return x[0] * x[1];
}

double f2(double * x, void * args)
{
    assert (args == NULL);
    double out;
    out = sin(x[0] * x[1] );
    //double out = x[0]*x[1] + pow(x[0],2)*pow(x[1],2) + pow(x[1],3)*sin(x[0]);
    
    return out;
}

int main(int argc, char * argv[])
{
    int next_option;
    const char * const short_options = "hd:f:v:";
    const struct option long_options[] = {
        { "help"     , 0, NULL, 'h' },
        { "directory", 1, NULL, 'd' },
        { "function" , 1, NULL, 'd' },
        { "verbose"  , 1, NULL, 'v' },
        { NULL       , 0, NULL, 0   }
    };
    program_name = argv[0];

    char * dirout = ".";
    size_t function = 0;
    int verbose = 0;
    do {
        next_option = getopt_long (argc, argv, short_options, long_options, NULL);
        switch (next_option)
        {
            case 'h': 
                print_code_usage(stdout, 0);
            case 'd':
                dirout = optarg;
                break;
            case 'f':
                function = (size_t) strtol(optarg,NULL,10);
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

    size_t dim = 2;
    enum poly_type ptype = HERMITE;
    struct FunctionMonitor * fm = NULL;
    double (*ff)(double *, void *);

    if (function == 0){
        fm = function_monitor_initnd(f0,NULL,dim,1000*dim);
        ff = f0;
    }
    else if (function == 1){
        fm = function_monitor_initnd(f1,NULL,dim,1000*dim);
        ff = f1;
    }
    else if (function == 2){
        fm = function_monitor_initnd(f2,NULL,dim,1000*dim);
        ff = f2;
    }
    else{
        printf("Function %zu not yet implemented\n",function);
        return 1;
    }

    struct OpeAdaptOpts * aopts = ope_adapt_opts_alloc();
    ope_adapt_opts_set_start(aopts,5);
    ope_adapt_opts_set_maxnum(aopts,20);
    ope_adapt_opts_set_coeffs_check(aopts,1);
    ope_adapt_opts_set_tol(aopts,1e-4);
    struct FtApproxArgs * fapp = ft_approx_args_createpoly(dim,&ptype,aopts);
    struct FunctionTrain * ft = NULL;
    ft = function_train_cross_ub(function_monitor_eval,fm,
                                 dim,NULL,NULL,fapp,NULL);
    ope_adapt_opts_free(aopts); aopts = NULL;
    ft_approx_args_free(fapp); fapp = NULL;

    size_t nevals = nstored_hashtable_cp(fm->evals);
    if (verbose == 1){
        printf("Final ranks are "); iprint_sz(3,ft->ranks);
        printf("Number of evaluations = %zu\n",nevals);
    }

    char evals[256];
    sprintf(evals,"%s/%s.dat",dirout,"evals");
    FILE *fp;
    fp =  fopen(evals, "w");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open %s\n", evals);
        return 0;
    }
    function_monitor_print_to_file(fm,fp);
    fclose(fp);


    FILE *fp2;
    char toterrs[256];
    sprintf(toterrs,"%s/%s.dat",dirout,"recon");
    fp2 =  fopen(toterrs, "w");
    if (fp2 == NULL){
        fprintf(stderr, "cat: can't open %s\n", toterrs);
        return 0;
    }

    fprintf(fp2,"x y f f0 df0\n");
    size_t Ntest = 1000;

    double out1=0.0;
    double den=0.0;
    double pt[2];
    double v1,v2;
    for (size_t ii = 0; ii < Ntest; ii++){
        pt[0] = randn()*0.5;
        pt[1] = randn()*0.5;
        v1 = ff(pt,NULL);
        v2 = function_train_eval(ft,pt);
        fprintf(fp2, "%3.5f %3.5f %3.5f %3.5f %3.5f \n", 
                pt[0],pt[1],v1,v2,fabs(v1-v2));
        den += pow(v1,2.0);
        out1 += pow(v1-v2,2.0);
    }

    if (verbose == 1){
        printf("RMS Error of Final = %G\n", out1/den);
    }


    fclose(fp2);
    function_train_free(ft);
    function_monitor_free(fm);
    return 0;
}
