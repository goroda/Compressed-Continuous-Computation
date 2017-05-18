#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <getopt.h>

#include "c3.h"

static char * program_name;

void print_code_usage (FILE *, int) __attribute__ ((noreturn));
void print_code_usage (FILE * stream, int exit_code)
{
    fprintf(stream, "Two examples of functional regression \n\n");
    fprintf(stream, "Usage: %s options \n", program_name);
    fprintf(stream,
            " -h --help       Display this usage information.\n"
            " -f --function   Which function to evaluate. \n"
            "                  0: 3d gaussian bump (default)\n"
            " -p --polyorder  Maximum polynomial order (default 5).\n"
            " -b --basis      Basis for spatial variable\n"
            "                  0: piecewise polynomial  (default)\n"
            "                  1: linear element\n"
            "                  2: orthonormal polynomials\n"
            " -a --adaptrank  Flag whether or not to adapt rank\n"
            "                 0: no adaptation\n"
            "                 1: adaptation (default)\n"
            " -v --verbose    Output words (default 0), 1 then show CVs, 2 then show opt progress.\n"
            " \n\n"
            " Outputs four files\n"
            " training_funcs.dat  -- training samples"
            " testing_funcs_n{number}.dat -- evaluations of true model\n"
            " testing_funcs_ft_n{number}.dat -- evaluations of reg model\n"
            " testing_funcs_diff_n{number}.dat -- difference b/w models\n"
        );
    exit (exit_code);
}


static double gauss_lb = 0.0;
static double gauss_ub = 1.0;
static double gauss_center = 0.2;
static double gauss_width = 0.05;
double gauss(const double * x, void * arg)
{
    size_t *dim = arg;
    size_t d = *dim;
    
    double center = gauss_center;
    double l = gauss_width;
    double preexp = 1.0/(l * sqrt(M_PI*2));

    double inexp = 0.0;
    for (size_t jj = 0; jj < d; jj++){
        double dx = x[jj] - center;
        inexp += dx*dx;
    }
    inexp *= -1;
    inexp /= (2.0*l*l);

    /* printf("inexp = %3.15LE\n",inexp); */
    /* printf("d = %zu\n",d); */
    /* dprint(d,x); */
    double out = preexp * exp(inexp);
    /* long double lout = preexp * exp(inexp); */
    /* printf("out = %3.15E\n",out); */
    /* printf("out = %3.15LE\n",lout); */
    return out;
}

int main(int argc, char * argv[])
{
    int next_option;
    const char * const short_options = "hf:p:b:a:v:";
    const struct option long_options[] = {
        { "help"      , 0, NULL, 'h' },
        { "function"  , 1, NULL, 'f' },
        { "polyorder" , 1, NULL, 'p' },
        { "basis"     , 1, NULL, 'b' },
        { "adaptrank" , 1, NULL, 'a' },
        { "verbose"   , 1, NULL, 'v' },
        { NULL        ,  0, NULL, 0   }
    };

    size_t maxorder = 5;
    int verbose = 0;
    size_t function = 0;
    size_t basis = 0;
    unsigned int adapt = 1;
    program_name = argv[0];
    do {
        next_option = getopt_long (argc, argv, short_options, long_options, NULL);
        switch (next_option)
        {
            case 'h': 
                print_code_usage(stdout, 0);
            case 'f':
                function = strtoul(optarg,NULL,10);
                break;
            case 'p':
                maxorder = strtoul(optarg,NULL,10);
                break;
            case 'b':
                basis = strtoul(optarg,NULL,10);
                break;
            case 'a':
                adapt = strtoul(optarg,NULL,10);
                break;
            case 'v':
                verbose = strtoul(optarg,NULL,10);
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


    size_t nloop = 6;
    double tol[6] = {1e0,1e-1,1e-2,1e-3,1e-4,1e-5};

    
    fprintf(stdout, "Dim Tol Exact Approx Nvals AbsError \n");
    for (size_t zz = 0; zz < nloop; zz++){

        struct FunctionMonitor * fm = function_monitor_initnd(gauss,&dim,dim,1000*dim);
        struct Fwrap * fw = fwrap_create(dim,"general");
        fwrap_set_f(fw,function_monitor_eval,fm);

        struct OneApproxOpts * qmopts = NULL;
        if (basis == 0){
            struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,gauss_lb,gauss_ub);
            size_t nregion = 2;
            pw_poly_opts_set_nregions(opts,nregion);
            pw_poly_opts_set_maxorder(opts,5);
            pw_poly_opts_set_minsize(opts,pow(1.0/(double)nregion,15));
            pw_poly_opts_set_coeffs_check(opts,1);
            pw_poly_opts_set_tol(opts,tol[zz]);
            qmopts = one_approx_opts_alloc(PIECEWISE,opts);
        }
        else if (basis == 1){
            double hmin = 1e-15;
            double letol = 1e-5;
            struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,gauss_lb,gauss_ub,letol,hmin);
            qmopts = one_approx_opts_alloc(LINELM,opts);
        }
        else if (basis == 2){
            struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
            ope_opts_set_lb(opts,gauss_lb);
            ope_opts_set_ub(opts,gauss_ub);
            ope_opts_set_start(opts,3);
            ope_opts_set_maxnum(opts,20);
            ope_opts_set_coeffs_check(opts,0);
            ope_opts_set_tol(opts,1e-10);
            qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
        }
        else{
            fprintf(stderr, "Basis type is not recognized\n\n\n");
            print_code_usage (stderr, 1);
        }
    

        struct C3Approx * c3a = c3approx_create(CROSS,dim);
        size_t rank = 2;
        double ** start = malloc_dd(dim);
        for (size_t kk= 0; kk < dim; kk++){
            c3approx_set_approx_opts_dim(c3a,kk,qmopts);
            start[kk] = linspace(gauss_lb+0.1,gauss_ub-0.1,rank);
        }
        c3approx_init_cross(c3a,rank,verbose,start);
        c3approx_set_cross_maxiter(c3a,1);
        c3approx_set_cross_tol(c3a,1e-14);
        c3approx_set_round_tol(c3a,1e-14);


        struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,(int)adapt);
        /* printf("ft ranks = "); iprint_sz(dim+1,function_train_get_ranks(ft)); */

        /* double intexact_ref = 1.253235e-1; */

    
        double intexact;
        if (dim == 2){
            intexact = 0.5 * gauss_width * sqrt(M_PI/2.0) *
                ((erf((gauss_center-1)/(sqrt(2.) * gauss_width)) -
                  erf((gauss_center)/(sqrt(2.) * gauss_width))) *
                 (erf((gauss_center-1)/(sqrt(2.) * gauss_width)) -
                  erf((gauss_center)/(sqrt(2.) * gauss_width))));
        }
        else{
            intexact = -0.25 * gauss_width * gauss_width * M_PI *
                ((erf((gauss_center-1)/(sqrt(2.) * gauss_width)) -
                  erf((gauss_center)/(sqrt(2.) * gauss_width))) *
                 (erf((gauss_center-1)/(sqrt(2.) * gauss_width)) -
                  erf((gauss_center)/(sqrt(2.) * gauss_width))) *
                 (erf((gauss_center-1)/(sqrt(2.) * gauss_width)) -
                  erf((gauss_center)/(sqrt(2.) * gauss_width))));
        }
        /* double intexact = 0.5 * gauss_width * sqrt(M_PI/2.0) * */
        /*     ((erf((gauss_center-1)/(sqrt(2.) * gauss_width)) - */
        /*       erf((gauss_center+1)/(sqrt(2.) * gauss_width))) * */
        /*      (erf((gauss_center-1)/(sqrt(2.) * gauss_width)) - */
        /*       erf((gauss_center+1)/(sqrt(2.) * gauss_width)))); */
        double intval = function_train_integrate(ft);
        double error = fabs(intexact - intval);
        size_t nvals = nstored_hashtable_cp(fm->evals);

    
        /* fprintf(fp, "%zu %3.15E %zu %3.15E \n", dim,,intval,nvals, error); */
        fprintf(stdout, "%zu %3.2E %3.15E %3.15E %zu %3.15E \n", dim,tol[zz],intexact,intval,nvals, error); 

        function_train_free(ft); ft = NULL;
        one_approx_opts_free_deep(&qmopts);
        function_monitor_free(fm); fm = NULL;
        fwrap_destroy(fw);
        c3approx_destroy(c3a);
        free_dd(dim,start);
    }

}
