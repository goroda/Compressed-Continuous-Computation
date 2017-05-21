
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>

#include "c3.h"

double sinsum(const double * x, void * args)
{
    size_t *dim = args;

    double out = 0.0;
    size_t ii;
    for (ii = 0; ii < *dim; ii++){
        out += x[ii];
    }

    out = sin(out);
    return out;
}

//int main( int argc, char *argv[])
int main( void )
{
    
    size_t min_dim = 10;
    size_t delta_dim = 75;
    size_t ndims = 10;

    size_t ntols = 10;
    double * epsilons = logspace(-13, -1, ntols);
    
    char integral[256];
    sprintf(integral,"integral.dat");

    FILE *fp;
    fp =  fopen(integral, "w");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open file\n");
        return 0;
    }
    fprintf(fp, "dim tol int N relerr \n");

    size_t ii,jj;
    size_t dim;
    for (ii = 0; ii < ndims; ii++){
        dim = min_dim + ii * delta_dim;
        printf("On dim (%zu/%zu) : %zu \n",ii,ndims,dim);

        double shouldbe = cimag( cpow( (cexp((double complex)I) - 1)/
                                       (double complex)I , dim));
        //printf("shouldbe=%G\n",shouldbe);

        for (jj = 0; jj < ntols; jj++){
            printf("..... On tol (%zu/%zu) : %E \n", jj,ntols, epsilons[jj]);

            struct FunctionMonitor * fm = function_monitor_initnd(sinsum,&dim,dim,1000*dim);
            struct Fwrap * fw = fwrap_create(dim,"general");
            fwrap_set_f(fw,function_monitor_eval,fm);
            struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
            ope_opts_set_lb(opts,0.0);
            ope_opts_set_start(opts,6);
            ope_opts_set_coeffs_check(opts,2);
            ope_opts_set_tol(opts,epsilons[jj]);
            struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
            
            struct C3Approx * c3a = c3approx_create(CROSS,dim);
            int verbose = 0;
            size_t rank = 2;
            double ** start = malloc_dd(dim);
            for (size_t kk= 0; kk < dim; kk++){
                c3approx_set_approx_opts_dim(c3a,kk,qmopts);
                start[kk] = linspace(0.0,1.0,rank);
            }
            c3approx_init_cross(c3a,rank,verbose,start);
            c3approx_set_cross_tol(c3a,1e-3);

            struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);
            
            double intval = function_train_integrate(ft);
            double relerr = fabs(intval-shouldbe)/fabs(shouldbe);
            //printf("intval=%G\n",intval);
            size_t nvals = nstored_hashtable_cp(fm->evals);
            printf(".......... rel error is %G, N = %zu\n", relerr,nvals);

            fprintf(fp, "%zu %3.15E %3.15E %zu %3.15E \n", 
                            dim,epsilons[jj],intval,nvals, relerr); 

            function_train_free(ft); ft = NULL;
            function_monitor_free(fm); fm = NULL;
            one_approx_opts_free_deep(&qmopts);
            fwrap_destroy(fw);
            c3approx_destroy(c3a);
            free_dd(dim,start);
        }
    }
    free(epsilons);
    fclose(fp);
    return 0;
}
