#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <getopt.h>

#include "lib_funcs/fourier.h"
#include "lib_array/array.h"

static char * program_name;

void print_code_usage (FILE *, int) __attribute__ ((noreturn));
void print_code_usage (FILE * stream, int exit_code)
{

    fprintf(stream, "Usage: %s options \n", program_name);
    fprintf(stream,
            " -h --help      Display this usage information.\n"
            " -d --directory Output directory (defaults to .)\n"
            " -f --function  Which function to approximate \n"
            "                0: (default) sin(x) \n"
            " -n --n         Discretization level (default 6)\n"
            " -l --lower     Lower bounds on x,y (default -1)\n"
            " -u --upper     Upper bounds on x,y (default 1)\n"
            " -v --verbose   Output words (default 0)\n"
        );
    exit (exit_code);
}

int f0(size_t N, const double * x, double * out, void * args)
{
    assert(args == NULL);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = sin(x[ii]);
    }
    return 0;
}


int f0d(size_t N, const double * x, double * out, void * args)
{
    assert(args == NULL);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = cos(x[ii]);
    }
    return 0;
}

int f0dd(size_t N, const double * x, double * out, void * args)
{
    assert(args == NULL);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = -sin(x[ii]);
    }
    return 0;
}

int f1(size_t N, const double * x, double * out, void * args)
{
    assert(args == NULL);
    (void) x;
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = 2.0;
    }
    return 0;
}

int f1d(size_t N, const double * x, double * out, void * args)
{
    assert(args == NULL);
    (void) x;
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = 0.0;
    }
    return 0;
}

int f1dd(size_t N, const double * x, double * out, void * args)
{
    assert(args == NULL);
    (void) x;
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = 0.0;
    }
    return 0;
}

int f2(size_t N, const double * x, double * out, void * args)
{
    assert(args == NULL);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = cos(x[ii]);
    }
    return 0;
}

int f2d(size_t N, const double * x, double * out, void * args)
{
    assert(args == NULL);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = -sin(x[ii]);
    }
    return 0;
}

int f2dd(size_t N, const double * x, double * out, void * args)
{
    assert(args == NULL);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = -cos(x[ii]);
    }
    return 0;
}

int f3(size_t N, const double * x, double * out, void * args)
{
    assert(args == NULL);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = exp(-0.5*x[ii]*x[ii]);
    }
    return 0;
}

int f3d(size_t N, const double * x, double * out, void * args)
{
    assert(args == NULL);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = -x[ii]*exp(-0.5*x[ii]*x[ii]);
    }
    return 0;
}

int f3dd(size_t N, const double * x, double * out, void * args)
{
    assert(args == NULL);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = -exp(-0.5*x[ii]*x[ii]) + x[ii]*x[ii]*exp(-0.5*x[ii]*x[ii]);
    }
    return 0;
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
    size_t N = 9;
    double lb = 0.0;
    double ub = 2.0*M_PI;
    int verbose = 0;

    int  (*func)(size_t,const double*, double *, void *) = f0;
    int  (*funcd)(size_t,const double*, double *, void *) = f0d;
    int  (*funcdd)(size_t,const double*, double *, void *) = f0dd;    
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
                if (function == 0){
                    func = f0;
                    funcd = f0d;
                    funcdd = f0dd;
                }
                else if (function == 1){
                    func = f1;
                    funcd = f1d;
                    funcdd = f1dd;
                }
                else if (function == 2){
                    func = f2;
                    funcd = f2d;
                    funcdd = f2dd;
                }
                else if (function == 3){
                    func = f3;
                    funcd = f3d;
                    funcdd = f3dd;
                }
                break;
            case 'n':
                N = (size_t) strtol(optarg,NULL,10);
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
    
    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,func,NULL);

    struct OpeOpts * opts = ope_opts_alloc(FOURIER);
    ope_opts_set_lb(opts, lb);
    ope_opts_set_ub(opts, ub);
    struct OrthPolyExpansion * fourier =
        orth_poly_expansion_init_from_opts(opts, N);

    int res = orth_poly_expansion_approx_vec(fourier, fw, opts);
    printf("Result is %d\n", res);
    struct OrthPolyExpansion * fourierd = orth_poly_expansion_deriv(fourier);
    struct OrthPolyExpansion * fourierdd = orth_poly_expansion_dderiv(fourier);
    struct OrthPolyExpansion * faxpy = orth_poly_expansion_copy(fourierd);
    res = orth_poly_expansion_axpy(2.0, fourier, faxpy);
    struct OrthPolyExpansion * fprod = orth_poly_expansion_prod(fourier, fourierd);
    printf("Result is %d\n", res);

    size_t Ntest = 100;
    double * xt = linspace(lb, ub, Ntest);
    double * eval = calloc_double(Ntest);
    double * evald = calloc_double(Ntest);
    double * evaldd = calloc_double(Ntest);    
    func(100, xt, eval, NULL);
    funcd(100, xt, evald, NULL);
    funcdd(100, xt, evaldd, NULL);    
    char evals[256];
    sprintf(evals,"%s/%s_%zu_%zu.dat",dirout,"evals",N, function);
    FILE *fp;
    fp =  fopen(evals, "w");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open %s\n", evals);
        return 0;
    }
    for (size_t ii = 0; ii < Ntest; ii++){
        double val = orth_poly_expansion_eval(fourier, xt[ii]);
        double vald = orth_poly_expansion_deriv_eval(fourier, xt[ii]);
        double vald2 = orth_poly_expansion_eval(fourierd, xt[ii]);
        double valdd = orth_poly_expansion_eval(fourierdd, xt[ii]);
        double vaxpy = orth_poly_expansion_eval(faxpy, xt[ii]);
        double vprod = orth_poly_expansion_eval(fprod, xt[ii]);
        fprintf(fp, "%3.5G %3.5G %3.5G %3.5G \
                     %3.5G %3.5G %3.5G %3.5G \
                     %3.5G %3.5G %3.5G %3.5G\n",
                xt[ii], eval[ii], val, evald[ii], vald, vald2, evaldd[ii], valdd,
                2.0 * eval[ii] + evald[ii], vaxpy,
                eval[ii] * evald[ii], vprod
            );
    }

    free(xt); xt = NULL;
    free(eval); eval = NULL;
    free(evald); evald = NULL;
    free(evaldd); evaldd = NULL;    
    fclose(fp);


    fwrap_destroy(fw);
    ope_opts_free(opts);
    orth_poly_expansion_free(fourier);
    orth_poly_expansion_free(fourierd);
    orth_poly_expansion_free(fourierdd);
    orth_poly_expansion_free(faxpy);
    orth_poly_expansion_free(fprod);

    return 0;
}
