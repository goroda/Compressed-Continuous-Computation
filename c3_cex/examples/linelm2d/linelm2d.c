#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <getopt.h>

#include "lib_interface/c3_interface.h"

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
            "                2: sin(5xy)\n"
            " -n --n         Discretization level (default 6)\n"
            " -l --lower     Lower bounds on x,y (default -1)\n"
            " -u --upper     Upper bounds on x,y (default 1)\n"
            " -v --verbose   Output words (default 0)\n"
        );
    exit (exit_code);
}

double f0(const double *x, void * args)
{
    assert(args == NULL);
    return x[0] + x[1];
}

double f1(const double *x, void * args)
{
    assert(args == NULL);
    return x[0] * x[1];
}

double f2(const double * x, void * args)
{
    assert (args == NULL);
    double out;
    out = sin(5.0 * x[0] * x[1] );
    //double out = x[0]*x[1] + pow(x[0],2)*pow(x[1],2) + pow(x[1],3)*sin(x[0]);
    
    return out;
}

int main(int argc, char * argv[])
{
    int next_option;
    const char * const short_options = "hd:f:n:l:u:v:v:";
    const struct option long_options[] = {
        { "help"     , 0, NULL, 'h' },
        { "directory", 1, NULL, 'd' },
        { "function" , 1, NULL, 'd' },
        { "n"        , 1, NULL, 'd' },
        { "lower"    , 1, NULL, 'd' },
        { "upper"    , 1, NULL, 'd' },
        { "verbose"  , 1, NULL, 'v' },
        { NULL       , 0, NULL, 0   }
    };
    program_name = argv[0];

    char * dirout = ".";
    size_t function = 0;
    size_t n = 6;
    double lb = -1.0;
    double ub = 1.0;
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
            case 'n':
                n = (size_t) strtol(optarg,NULL,10);
                break;
            case 'l':
                lb = strtod(optarg,NULL);
                break;
            case 'u':
                ub = strtod(optarg,NULL);
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
    double lbv[2] = {lb, 2*lb};
    double ubv[2] = {ub, 3*ub};


    double * xnodes = linspace(lbv[0],ubv[0],n);
    double * ynodes = linspace(lbv[1],ubv[1],n);
    struct LinElemExpAopts * aoptsx=lin_elem_exp_aopts_alloc(n,xnodes);
    struct LinElemExpAopts * aoptsy=lin_elem_exp_aopts_alloc(n,ynodes);
    struct OneApproxOpts * qmoptsx = one_approx_opts_alloc(LINELM,aoptsx);
    struct OneApproxOpts * qmoptsy = one_approx_opts_alloc(LINELM,aoptsy);
    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    size_t init_rank = 3;
    double ** start = malloc_dd(dim);
    c3approx_set_approx_opts_dim(c3a,0,qmoptsx);
    c3approx_set_approx_opts_dim(c3a,1,qmoptsy);
    start[0] = calloc_double(init_rank);
    start[0][0] = xnodes[0];
    start[0][1] = xnodes[1];
    start[0][2] = xnodes[2];
    start[1] = calloc_double(init_rank);
    start[1][0] = xnodes[0];
    start[1][1] = xnodes[1];
    start[1][2] = xnodes[2];
    
    c3approx_init_cross(c3a,init_rank,verbose,start);
    c3approx_set_adapt_kickrank(c3a,2);
    c3approx_set_cross_maxiter(c3a,10);
    c3approx_set_cross_tol(c3a,1e-7);
    c3approx_set_round_tol(c3a,1e-10);
    c3approx_set_adapt_maxrank_all(c3a,n);
    c3approx_set_verbose(c3a,verbose);


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
    else{
        printf("Function %zu not yet implemented\n",function);
        return 1;
    }

    // Done with setup
    if (verbose == 1){
        printf("nodes are\n");
        dprint(n,xnodes);
    }

    struct Fwrap * fw = fwrap_create(dim,"general");
    fwrap_set_f(fw,function_monitor_eval,fm);

    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);

    size_t nevals = nstored_hashtable_cp(fm->evals);
    size_t ntot = n*n;
    if (verbose == 1){
        printf("Final ranks are "); iprint_sz(3,ft->ranks);
        printf("Number of evaluations = %zu\n",nevals);
        printf("Number of total nodes = %zu\n",ntot);
        printf("Fraction of nodes used is %3.15G\n",(double)nevals/(double)ntot);
    }

    char evals[256];
    sprintf(evals,"%s/%s_%zu.dat",dirout,"evals",n);
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
    sprintf(toterrs,"%s/%s_%zu.dat",dirout,"recon",n);
    fp2 =  fopen(toterrs, "w");
    if (fp2 == NULL){
        fprintf(stderr, "cat: can't open %s\n", toterrs);
        return 0;
    }

    fprintf(fp2,"x y f f0 df0\n");
    size_t N1 = 40;
    size_t N2 = 40;
    double * xtest = linspace(lb,ub,N1);
    double * ytest = linspace(2*lb,3*ub,N2);

    double out1=0.0;
    double den=0.0;
    double pt[2];
    double v1,v2;
    for (size_t ii = 0; ii < N1; ii++){
        for (size_t jj = 0; jj < N2; jj++){
            pt[0] = xtest[ii]; pt[1] = ytest[jj];
            v1 = ff(pt,NULL);
            v2 = function_train_eval(ft,pt);
            fprintf(fp2, "%3.5f %3.5f %3.5f %3.5f %3.5f \n", 
                    xtest[ii], ytest[jj],v1,v2,v1-v2);
            den += pow(v1,2.0);
            out1 += pow(v1-v2,2.0);
        }
        fprintf(fp2,"\n");
    }
    if (verbose == 1){
        printf("RMS Error of Final = %G\n", out1/den);
    }


    fclose(fp2);
    free(xtest); free(ytest);
    function_train_free(ft);
    function_monitor_free(fm);

    c3approx_destroy(c3a);
    fwrap_destroy(fw);
    free_dd(2,start);
    one_approx_opts_free_deep(&qmoptsx);
    one_approx_opts_free_deep(&qmoptsx);
    free(xnodes);
    free(ynodes);
    

    return 0;
}
