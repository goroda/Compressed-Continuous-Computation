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
	    " -d --dim        Dimension\n"
            " -a --adaptrank  Flag whether or not to adapt rank\n"
            "                 0: no adaptation\n"
            "                 1: adaptation (default)\n"
            " -r --dumprank2  Dump rank 2 approximation data to show univariate fibers\n"
            "                 0: dont dump (default)\n"
            "                 1: dump\n"
	    " -t --tolerance  Univariate approximation tolerance\n"
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

    // original
    /* double preexp = 1.0/(l * sqrt(M_PI*2)); */

    /* double inexp = 0.0; */
    /* for (size_t jj = 0; jj < d; jj++){ */
    /*     double dx = x[jj] - center; */
    /*     inexp += dx*dx; */
    /* } */
    /* inexp *= -1; */
    /* inexp /= (2.0*l*l); */

    // normal
    double preexp =  pow(2.0*M_PI * l*l,*dim);
    preexp = 1.0/sqrt(preexp);

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
    const char * const short_options = "hf:p:b:a:d:r:t:v:";
    const struct option long_options[] = {
        { "help"      , 0, NULL, 'h' },
        { "function"  , 1, NULL, 'f' },
        { "polyorder" , 1, NULL, 'p' },
        { "basis"     , 1, NULL, 'b' },
	{ "dim"       , 1, NULL, 'd' },
        { "adaptrank" , 1, NULL, 'a' },
        { "dumprank2" , 1, NULL, 'r' },
	{ "tolerance" , 1, NULL, 't' },
        { "verbose"   , 1, NULL, 'v' },
        { NULL        ,  0, NULL, 0  }
    };

    size_t maxorder = 5;
    int verbose = 0;
    size_t function = 0;    
    size_t basis = 0;
    unsigned int adapt = 1;
    unsigned int rank_2_print_funcs = 0;
    size_t dim = 3;
    int spec_tol = 0;
    double tol_user = 0.0;
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
	    case 'd':
                dim = strtoul(optarg,NULL,10);
                break;		
            case 'r':
                rank_2_print_funcs = strtoul(optarg,NULL,10);
                break;
            case 'a':
                adapt = strtoul(optarg,NULL,10);
                break;
	    case 't':
	        spec_tol = 1;
                tol_user = atof(optarg);
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

    (void)function; // silence compiler


    size_t nregion = 3;
    if (rank_2_print_funcs == 1){
        double tol[10] = {1e0,1e-1,1e-2,1e-3,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10};
        size_t zz = 3; // use tol = 1e-1
        
        struct FunctionMonitor * fm = function_monitor_initnd(gauss,&dim,dim,1000*dim);
        struct Fwrap * fw = fwrap_create(dim,"general");
        fwrap_set_f(fw,function_monitor_eval,fm);

        struct OneApproxOpts * qmopts = NULL;
        if (basis == 0){
            struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,gauss_lb,gauss_ub);
            pw_poly_opts_set_nregions(opts,nregion);
            pw_poly_opts_set_maxorder(opts,maxorder);
            pw_poly_opts_set_minsize(opts,pow(1.0/(double)nregion,3));
            pw_poly_opts_set_coeffs_check(opts,1);
            pw_poly_opts_set_tol(opts,tol[zz]);
            qmopts = one_approx_opts_alloc(PIECEWISE,opts);
        }
        else if (basis == 1){
            double hmin = 1e-2;
            double letol = tol[zz];
            struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,gauss_lb,gauss_ub,letol,hmin);
            qmopts = one_approx_opts_alloc(LINELM,opts);
        }
        else if (basis == 2){
            struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
            ope_opts_set_lb(opts,gauss_lb);
            ope_opts_set_ub(opts,gauss_ub);
            ope_opts_set_start(opts,3);
            ope_opts_set_maxnum(opts,maxorder);
            ope_opts_set_coeffs_check(opts,1);
            ope_opts_set_tol(opts,tol[zz]);
            
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
            start[kk] = linspace(gauss_lb,gauss_ub,rank);
        }
        c3approx_init_cross(c3a,rank,verbose,start);
        c3approx_set_cross_maxiter(c3a,3);
        c3approx_set_cross_tol(c3a,1e-10);
        c3approx_set_round_tol(c3a,1e-10);

        struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);

        size_t nx = 100;
        double * x = linspace(gauss_lb,gauss_ub,nx);
        size_t * ranks = function_train_get_ranks(ft);
        if (dim == 3){
            char filename[256];
            FILE * fp;
            for (size_t core = 0; core < 3; core++){
                for (size_t jj = 0; jj < ranks[core]*ranks[core+1]; jj++){
                    struct PiecewisePoly * pw = ft->cores[core]->funcs[jj]->f;
                    sprintf(filename,"core_%zu_func_%zu.dat",core,jj);
                    fp = fopen(filename, "w");
                    fprintf(fp,"x f\n");
                    for (size_t ii = 0; ii < nx; ii++){
                        fprintf(fp,"%3.15E %3.15E\n",x[ii], piecewise_poly_eval(pw,x[ii]));
                    }
                    fclose(fp);

                    double * nodes = NULL;
                    size_t Nb;
                    piecewise_poly_boundaries(pw,&Nb, &nodes, NULL);

                    // pieces
                    sprintf(filename,"core_%zu_func_%zu_pieces.dat",core,jj);
                    fp = fopen(filename, "w");
                    fprintf(fp,"x f\n");
                    for (size_t ii = 0; ii < Nb; ii++){
                        fprintf(fp,"%3.15E %3.15E\n",nodes[ii], piecewise_poly_eval(pw,nodes[ii]));
                    }
                    fclose(fp);
                    free(nodes); nodes = NULL;
                    
                }
            }
        }
        free(x); x = NULL;
        
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
        fprintf(stdout, "%zu %3.0E %3.15E %3.15E %zu %3.15E \n", dim,tol[zz],intexact,intval,nvals, error); 
            
        
        function_train_free(ft); ft = NULL;
        one_approx_opts_free_deep(&qmopts);
        function_monitor_free(fm); fm = NULL;
        fwrap_destroy(fw);
        c3approx_destroy(c3a);
        free_dd(dim,start);
    }
    else{
    

        size_t nloop = 7;
        /* double tol[10] = {1e0,5e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5}; */
        double tol[7] = {1e1,1e-1,1e-3,1e-5,1e-7,1e-9,1e-11};
	
	if (spec_tol == 1){
	    nloop = 1;
	    tol[0] = tol_user;
	}
        fprintf(stdout, "Dim Tol Exact Approx Nvals AbsError \n");
        for (size_t zz = 0; zz < nloop; zz++){
            if (zz > nloop){
                break;
            }
            /* printf("zz = %zu, nloop = %zu\n",zz,nloop); */
            struct Fwrap * fw = fwrap_create(dim,"general");
            
            struct FunctionMonitor * fm = function_monitor_initnd(gauss,&dim,dim,1000*dim);
            fwrap_set_f(fw,function_monitor_eval,fm);
            
            /* fwrap_set_f(fw,gauss,&dim); */

            struct OneApproxOpts * qmopts = NULL;
            if (basis == 0){
                struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,gauss_lb,gauss_ub);

                pw_poly_opts_set_nregions(opts,nregion);
                pw_poly_opts_set_maxorder(opts,maxorder);
                pw_poly_opts_set_minsize(opts,pow(1.0/(double)nregion,3));
                pw_poly_opts_set_coeffs_check(opts,1);
                pw_poly_opts_set_tol(opts,tol[zz]);
                qmopts = one_approx_opts_alloc(PIECEWISE,opts);
            }
            else if (basis == 1){
                double hmin = 1e-2;
                double letol = tol[zz];
                struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,gauss_lb,gauss_ub,letol,hmin);
                qmopts = one_approx_opts_alloc(LINELM,opts);
            }
            else if (basis == 2){
                struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
                ope_opts_set_lb(opts,gauss_lb);
                ope_opts_set_ub(opts,gauss_ub);
                ope_opts_set_start(opts,4);
                ope_opts_set_maxnum(opts,maxorder);
                ope_opts_set_coeffs_check(opts,1);
                ope_opts_set_tol(opts,tol[zz]);
            
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
            c3approx_set_adapt_kickrank(c3a,1);
            c3approx_set_cross_maxiter(c3a,10);
            c3approx_set_cross_tol(c3a,1e-8);
            c3approx_set_round_tol(c3a,1e-8);


            struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,(int)adapt);

            /* printf("ft ranks = "); iprint_sz(dim+1,function_train_get_ranks(ft)); */

            /* double intexact_ref = 1.253235e-1; */

    
            double intexact = 0.0;
            if (dim == 2){
                intexact = 0.5 * gauss_width * sqrt(M_PI/2.0) *
                    ((erf((gauss_center-1)/(sqrt(2.) * gauss_width)) -
                      erf((gauss_center)/(sqrt(2.) * gauss_width))) *
                     (erf((gauss_center-1)/(sqrt(2.) * gauss_width)) -
                      erf((gauss_center)/(sqrt(2.) * gauss_width))));
            }
            else if (dim == 3){
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
            fprintf(stdout, "%zu %3.0E %3.15E %3.15E %zu %3.15E \n", dim,tol[zz],intexact,intval,nvals, error); 
        
            function_train_free(ft); ft = NULL;
            one_approx_opts_free_deep(&qmopts);
            function_monitor_free(fm); fm = NULL;
            fwrap_destroy(fw);
            c3approx_destroy(c3a);
            free_dd(dim,start);

        }
    }

}
