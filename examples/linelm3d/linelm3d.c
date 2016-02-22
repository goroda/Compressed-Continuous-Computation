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
            "                0: (default) x + y + z\n"
            "                1: x*y*z \n"
            "                2: sin(5xyz)\n"
            " -n --n         Discretization level (default 6)\n"
            " -l --lower     Lower bounds on x,y (default -1)\n"
            " -u --upper     Upper bounds on x,y (default 1)\n"
            " -v --verbose   Output words (default 0)\n"
        );
    exit (exit_code);
}

double f0(double *x, void * args)
{
    assert(args == NULL);
    return x[0] + x[1] + x[2];
}

double f1(double *x, void * args)
{
    assert(args == NULL);
    return x[0] * x[1] * x[2];
}

double f2(double * x, void * args)
{
    assert (args == NULL);
    double out;
    out = sin(5.0 * x[0] * x[1] * x[2] );
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

    size_t dim = 3;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);

    struct LinElemExpAopts aopts = {n,0};
    struct FtApproxArgs * fapp = ft_approx_args_create_le(dim,&aopts);
    double * xnodes = linspace(lb,ub,n);
    struct c3Vector c3v = {n,xnodes};
    struct FiberOptArgs * fopt = fiber_opt_args_bf_same(dim,&c3v);

    size_t init_ranks[4] = {1,3,3,1};
    struct FtCrossArgs fca;
    ft_cross_args_init(&fca);
    fca.dim = dim;
    fca.ranks = init_ranks;
    fca.epsilon = 1e-7;
    fca.maxiter = 10;
    fca.epsround = 1e-10;
    fca.kickrank = 2;
    fca.maxiteradapt = 5;
    fca.verbose = verbose;
    fca.optargs = fopt;
   
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

    // Done with setup
    if (verbose == 1){
        printf("nodes are\n");
        dprint(n,xnodes);
    }
    assert (n > 3);
    double startx[3] = {xnodes[0], xnodes[2], xnodes[n-1]};
    double starty[3] = {xnodes[0], xnodes[2], xnodes[n-1]};
    double startz[3] = {xnodes[0], xnodes[2], xnodes[n-1]};
    double * start[3];
    start[0] = startx;
    start[1] = starty;
    start[2] = startz;
    struct FunctionTrain * ft = 
        function_train_cross(function_monitor_eval,fm,bds,start,&fca,fapp);

    size_t nevals = nstored_hashtable_cp(fm->evals);
    size_t ntot = n*n*n;
    if (verbose == 1){
        printf("Final ranks are "); iprint_sz(4,ft->ranks);
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


    /* FILE *fp2; */
    /* char toterrs[256]; */
    /* sprintf(toterrs,"%s/%s_%zu.dat",dirout,"recon",n); */
    /* fp2 =  fopen(toterrs, "w"); */
    /* if (fp2 == NULL){ */
    /*     fprintf(stderr, "cat: can't open %s\n", toterrs); */
    /*     return 0; */
    /* } */

    
    //fprintf(fp2,"x y f f0 df0\n");
    size_t N1 = 40;
    size_t N2 = 40;
    size_t N3 = 40;
    double * xtest = linspace(lb,ub,N1);
    double * ytest = linspace(lb,ub,N2);
    double * ztest = linspace(lb,ub,N3);

    double out1=0.0;
    double den=0.0;
    double pt[3];
    double v1,v2;
    for (size_t ii = 0; ii < N1; ii++){
        for (size_t jj = 0; jj < N2; jj++){
            for (size_t kk = 0; kk < N3; kk++){
                pt[0] = xtest[ii]; pt[1] = ytest[jj]; pt[2] = ztest[kk];
                v1 = ff(pt,NULL);
                v2 = function_train_eval(ft,pt);
                //fprintf(fp2, "%3.5f %3.5f %3.5f %3.5f %3.5f \n", 
                //        xtest[ii], ytest[jj],v1,v2,v1-v2);
                den += pow(v1,2.0);
                out1 += pow(v1-v2,2.0);
            }
        }
        //fprintf(fp2,"\n");
    }
    if (verbose == 1){
        printf("RMS Error of Final = %G\n", out1/den);
    }


    //fclose(fp2);
    free(xtest);
    free(ytest);
    free(ztest);

    function_train_free(ft);
    function_monitor_free(fm);
    ft_approx_args_free(fapp);
    fiber_opt_args_free(fopt);
    bounding_box_free(bds);
    return 0;
}
