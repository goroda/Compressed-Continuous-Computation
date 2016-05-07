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
            "                0: (default) x + y + 2.0\n"
            "                1: x*y \n"
            "                2: sin(xy)\n"
            "                3: 1/sqrt(2pi)exp(-0.5 (3x^2 + 0.5xy + y^2)\n"
            " -v --verbose   Output words (default 0)\n"
        );
    exit (exit_code);
}

double f0(const double *x, void * args)
{
    (void)(args);
    return x[0] + x[1]+2.0;
//    return 1.0;
}

double f1(const double *x, void * args)
{
    (void)(args);
    return x[0] * x[1];
}

double f2(const double * x, void * args)
{
    (void)(args);
    double out;
    out = sin(x[0] * x[1]);
    //double out = x[0]*x[1] + pow(x[0],2)*pow(x[1],2) + pow(x[1],3)*sin(x[0]);
    
    return out;
}

double f3(const double * x, void * args)
{
    (void)(args);
    double quad = 3.0*x[0]*x[0] + 0.5 * x[0]*x[1]  + 1.0*x[1]*x[1];
    double out = 1/sqrt(2.0*M_PI) * exp(-quad/2.0);
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
    struct FunctionMonitor * fm = NULL;
    double (*ff)(const double *, void *);

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
    else if (function == 3){
        fm = function_monitor_initnd(f3,NULL,dim,1000*dim);
        ff = f3;
    }
    else{
        printf("Function %zu not yet implemented\n",function);
        return 1;
    }
    struct Fwrap * fw = fwrap_create(dim,"general");
    fwrap_set_f(fw,function_monitor_eval,fm);

    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    ope_opts_set_start(opts,5);
    ope_opts_set_maxnum(opts,20);
    ope_opts_set_coeffs_check(opts,1);
    ope_opts_set_tol(opts,1e-4);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);

    size_t init_rank = 2;
    double ** start = malloc_dd(dim);
    // optimization stuff
    size_t N = 100;
    double * x = linspace(-10.0,10.0,N);
    struct c3Vector * optnodes = c3vector_alloc(N,x);
    free(x); x = NULL;
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        c3approx_set_opt_opts_dim(c3a,ii,optnodes);
        start[ii] = linspace(-1.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);

    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);
    fwrap_destroy(fw);
    c3vector_free(optnodes);
    c3approx_destroy(c3a);
    free_dd(dim,start);
    one_approx_opts_free_deep(&qmopts);
    

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
        pt[0] = randn();
        pt[1] = randn();
        v1 = ff(pt,NULL);
        v2 = function_train_eval(ft,pt);
        fprintf(fp2, "%3.5f %3.5f %3.5f %3.5f %3.5f \n", 
                pt[0],pt[1],v1,v2,fabs(v1-v2));
        den += pow(v1,2.0);
        out1 += pow(v1-v2,2.0);
    }

    if (verbose == 1){
        printf("RMS Error of Final = %G\n", out1/den);
        double integral = function_train_integrate(ft);
        printf("Expectation is %G\n",integral);
    }


    fclose(fp2);
    function_train_free(ft);
    function_monitor_free(fm);
    return 0;
}
