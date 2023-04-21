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
            " -v --verbose   Output words (default 0)\n"
        );
    exit (exit_code);
}

double f1(double x, void * args)
{
    assert(args == NULL);
//    return exp(10.0 * x);
    return sin(4.0*x);
}

double compute_l2_abserr(size_t N, double * xtest, struct LinElemExp * le, int func)
{

    if (func == 1){
        double err = 0.0;
        double v1,v2;
        for (size_t ii = 0; ii < N; ii++){
            v1 = lin_elem_exp_eval(le,xtest[ii]);
            v2 = f1(xtest[ii],NULL);
            err += pow(v1-v2,2);
        }
        err /= (double) N;
        
        return sqrt(err);
    }
    else{
        return 0;
    }
}

double compute_linf_abserr(size_t N, double * xtest, struct LinElemExp * le, int func)
{

    if (func == 1){
        double err = 0.0;
        double terr;
        double v1,v2;
        for (size_t ii = 0; ii < N; ii++){
            v1 = lin_elem_exp_eval(le,xtest[ii]);
            v2 = f1(xtest[ii],NULL);
            terr = fabs(v1-v2);
            if (terr > err){
                err = terr;
            }
            
        }
        return err;
    }
    else{
        return 0;
    }
}

int main(int argc, char * argv[])
{
    int next_option;
    const char * const short_options = "hd:v:";
    const struct option long_options[] = {
        { "help"     , 0, NULL, 'h' },
        { "directory", 1, NULL, 'd' },
        { "verbose"  , 1, NULL, 'v' },
        { NULL       , 0, NULL, 0   }
    };

    char * dirout = ".";
    program_name = argv[0];
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

    char filename[256];
    sprintf(filename,"%s/errs.dat",dirout);
    FILE * fp = fopen(filename,"w");
    assert (fp != NULL);
    fprintf(fp,"n l2 l2est linf linfest integral\n");
    
    size_t n = 4;
    size_t ntimes = 12;
    double lb = -2.0;
    double ub = 2.0;
    size_t ntest = 100000;
    double * xtest = calloc_double(ntest);
    for (size_t ii = 0; ii < ntest; ii++){
        xtest[ii] = randu()*(ub-lb) + lb;
    }

    for (size_t ii = 0; ii < ntimes; ii++){
        
        double * x = linspace(lb,ub,n);
        double * f = calloc_double(n);
        for (size_t jj = 0; jj<n; jj++ ){
            f[jj] = f1(x[jj],NULL);
        }
        struct LinElemExp * le = lin_elem_exp_init(n,x,f);

//        printf("evaluation at left is %G\n",f[0]);
//        printf("left coeff = %G\n",le->coeff[0]);
//        printf("value of f at left bound = %G\n",f1(lb,NULL));
//        printf("value of le at left bound = %G\n", lin_elem_exp_eval(le,lb));
//        return 1;
        double l2err = compute_l2_abserr(ntest,xtest,le,1);
        double linferr = compute_linf_abserr(ntest,xtest,le,1);
        double integral = lin_elem_exp_integrate(le);
        double * loc = calloc_double(n-1);
        double errest = lin_elem_exp_err_est(le,loc,1,0);
        double errest2 = lin_elem_exp_err_est(le,loc,1,2);
        printf("n = %zu l2=%3.5G linf=%3.5G linfest=%3.5G l2est=%3.5G integral=%3.5G \n",
               n,l2err,linferr,errest,errest2,integral);
        fprintf(fp,"%zu %3.15G %3.15G %3.15G %3.15G %3.15G\n",n,l2err,errest2,linferr,errest,integral);
        n *= 2;
        free(x); x = NULL;
        free(f); f = NULL;
        free(loc); loc = NULL;
        lin_elem_exp_free(le); le = NULL;
    }
    if (verbose == 1){
        printf("Done.!\n");
    }
    fclose(fp);
    free(xtest); xtest = NULL;
    return 0;
}
