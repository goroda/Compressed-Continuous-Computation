
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>

#include "c3.h"

double func(const double * x, void * args)
{
    size_t *dim = args;

    double out = 0.0;
    size_t ii;
    for (ii = 0; ii < *dim; ii++){
        out += x[ii]*x[ii];
    }

    double alpha = 1e-12;
    out = 1.0/(sqrt(out + alpha));
    return out;
}

double eval_error(size_t N, size_t dim, const double * x, struct FunctionTrain * ft)
{
    double diff = 0.0;
    double den = 0.0;
    for (size_t ii = 0; ii < N; ii++){
        double true_val = func(x + ii * dim, &dim);
        double eval = function_train_eval(ft, x + ii*dim);
        /* printf("x = "); */
        /* dprint(dim, x + ii * dim);         */
        /* printf("\t %3.5E, %3.5E\n", true_val, eval); */
        diff += pow(true_val - eval, 2);
        den += pow(true_val, 2);
    }
    return sqrt(diff / den);
    
}
//int main( int argc, char *argv[])
int main( void )
{
    
    size_t min_dim = 2;
    size_t delta_dim = 4;
    size_t ndims = 8;
    int verbose = 0;
    
    size_t ntols = 5;
    double * epsilons = logspace(-5, -1, ntols);

    // uncomment below for the single case rank adaptation
    /* size_t min_dim = 30; */
    /* size_t delta_dim = 4; */
    /* size_t ndims = 1; */
    /* int verbose = 1;     */

    /* size_t ntols = 1; */
    /* double * epsilons = calloc_double(1); */
    /* epsilons[0] = 1e-5; */


    
    char errors[256];
    sprintf(errors,"sing_errors.dat");

    FILE *fp;
    fp =  fopen(errors, "w");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open file\n");
        return 0;
    }
    fprintf(fp, "dim tol N relerr avgrank maxrank\n");

    size_t ii,jj;
    size_t dim;
    size_t Ntest = 100;
    for (ii = 0; ii < ndims; ii++){
        dim = min_dim + ii * delta_dim;
        printf("On dim (%zu/%zu) : %zu \n",ii,ndims,dim);

        double * test_samples = calloc_double(dim * Ntest);
        for (size_t zz = 0; zz < Ntest; zz++){
            for (size_t ll = 0; ll < dim; ll++){
                test_samples[zz * dim + ll] = 2.0 * randu() - 1.0;
            }
        }
        

        //printf("shouldbe=%G\n",shouldbe);
        int poly = 0;
        for (jj = 0; jj < ntols; jj++){
            printf("..... On tol (%zu/%zu) : %E \n", jj, ntols, epsilons[jj]);

            // Set up Function
            struct FunctionMonitor * fm = function_monitor_initnd(func,&dim,dim,1000*dim);
            struct Fwrap * fw = fwrap_create(dim,"general");
            fwrap_set_f(fw,function_monitor_eval,fm);

            // Set up 1D approximation
            struct OneApproxOpts * qmopts = NULL;
            double lb = -1.0;
            double ub = 1.0;            
            double tol = 1e-8;
            if (poly == 1){
                struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
                ope_opts_set_lb(opts, lb);
                ope_opts_set_ub(opts, ub);                
                ope_opts_set_start(opts, 6);
                ope_opts_set_coeffs_check(opts, 2);
                ope_opts_set_tol(opts, tol);
                qmopts = one_approx_opts_alloc(POLYNOMIAL, opts);
            }
            else{
                size_t nregion = 4;
                size_t maxorder = 3;
                struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE, lb, ub);
                pw_poly_opts_set_nregions(opts, nregion);
                pw_poly_opts_set_maxorder(opts, maxorder);
                pw_poly_opts_set_minsize(opts, pow(1.0/(double)nregion, 3));
                pw_poly_opts_set_coeffs_check(opts, 1);
                pw_poly_opts_set_tol(opts, tol);
                qmopts = one_approx_opts_alloc(PIECEWISE,opts);
            }

            // Perform approximation
            struct C3Approx * c3a = c3approx_create(CROSS,dim);

            size_t rank = 4;
            double ** start = malloc_dd(dim);
            for (size_t kk= 0; kk < dim; kk++){
                c3approx_set_approx_opts_dim(c3a,kk,qmopts);
                start[kk] = linspace(-1.0,1.0,rank);
            }
            c3approx_init_cross(c3a, rank, verbose, start);
            c3approx_set_adapt_maxrank_all(c3a, 30);
            c3approx_set_adapt_kickrank(c3a, 2);
            c3approx_set_cross_maxiter(c3a, 5);
            c3approx_set_cross_tol(c3a, epsilons[jj]);
            c3approx_set_round_tol(c3a, epsilons[jj]);


            struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);
            size_t * ranks = function_train_get_ranks(ft);
            printf("ranks = "); iprint_sz(dim+1, ranks);
            size_t maxrank = function_train_get_maxrank(ft);
            double avgrank = function_train_get_avgrank(ft);
            double rel_error = eval_error(Ntest, dim, test_samples, ft);
            
            //printf("intval=%G\n",intval);
            size_t nvals = nstored_hashtable_cp(fm->evals);
            printf(".......... rel error is %G, N = %zu\n", rel_error, nvals);

            fprintf(fp, "%zu %3.15E %zu %3.15E %3.15E %zu \n", 
                    dim, epsilons[jj], nvals, rel_error, avgrank, maxrank);

            function_train_free(ft); ft = NULL;
            function_monitor_free(fm); fm = NULL;
            one_approx_opts_free_deep(&qmopts);
            fwrap_destroy(fw);
            c3approx_destroy(c3a);
            free_dd(dim,start);
        }

        free(test_samples); test_samples = NULL;
    }
    free(epsilons);
    fclose(fp);
    return 0;
}
