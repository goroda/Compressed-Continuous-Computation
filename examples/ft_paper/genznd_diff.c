#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "c3.h"

double discnd(const double * x, void * args)
{
     
    size_t * dim = args;
    
    int big = 0;
    size_t ii; 
    for (ii = 0; ii < *dim; ii++){
        if (x[ii] > 0.5){
            big = 1;
            break;
        }
    }

    double out = 0.0;
    if (big == 1){
        return out;
    }
    else{
        for (ii = 0; ii < *dim; ii++){
            out += 5.0 * x[ii];
        }
    }
    
    out = exp(out);
    return out;
}


double compute_diff(const double * x, size_t dim)
{
    double val = 0.0;
    for (size_t ii = 0; ii < dim; ii++){
        val += 5.0 * x[ii];
    }
    
    return 5.0 * exp(val);
}

int main()
{

    size_t dim = 10;
    double pt[10];
    for (size_t ii = 0; ii < dim; ii++){
        pt[ii] = 0.2;
    }
    double grad[10];

    char integral[256];
    sprintf(integral,"derivative.dat");

    FILE *fp;
    fp =  fopen(integral, "w");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open file\n");
        return 0;
    }
    fprintf(fp, "tol deriv N relerr \n");

    double tols[11] = {1e-2,1e-4,1e-6,1e-8,1e-10,1e-12,1e-14,1e-16,1e-18,1e-22,1e-24};
    size_t ntols = 9;
    for (size_t ii = 0; ii < ntols; ii++){
        /* aopts.epsilon = aopts.epsilon * 2.0; */
        printf("On tol (%zu/%zu) : %G \n",ii,ntols-1,tols[ii]);

        struct FunctionMonitor * fm = function_monitor_initnd(discnd,&dim,dim,1000*dim);
        struct Fwrap * fw = fwrap_create(dim,"general");
        fwrap_set_f(fw,function_monitor_eval,fm);
        struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,0.0,1.0);
        pw_poly_opts_set_nregions(opts,3);
        pw_poly_opts_set_maxorder(opts,6);
        pw_poly_opts_set_minsize(opts,pow(1.0/3.0,3));
        pw_poly_opts_set_coeffs_check(opts,1);
        pw_poly_opts_set_tol(opts,tols[ii]);

        struct OneApproxOpts * qmopts = one_approx_opts_alloc(PIECEWISE,opts);
        struct C3Approx * c3a = c3approx_create(CROSS,dim);
        int verbose = 0;
        double ** start = malloc_dd(dim);
        size_t rank = 2;
        for (size_t jj = 0; jj < dim; jj++){
            c3approx_set_approx_opts_dim(c3a,jj,qmopts);
            start[jj] = linspace(0.0,1.0,rank);
        }
        c3approx_init_cross(c3a,rank,verbose,start);
        c3approx_set_cross_tol(c3a,1e-10);
        c3approx_set_round_tol(c3a,1e-10);
        c3approx_set_cross_maxiter(c3a,10);
        
        struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);

        
        double shouldbe = dim*compute_diff(pt,dim);


        function_train_gradient_eval(ft,pt,grad);
        double gradis = 0.0;
        for (size_t zz = 0; zz < dim; zz++){
            gradis += grad[zz];
        }

        double eval = function_train_eval(ft,pt);
        printf("eval = %3.15G\n",eval);
        printf("eval shouldbe = %3.15G\n",compute_diff(pt,dim)/5.0);

        double abserr = fabs(gradis-shouldbe);
        double relerr = abserr/(fabs(shouldbe)+1e-20);
        size_t nvals = nstored_hashtable_cp(fm->evals);
        printf("shouldbe =%3.15G gradis=%3.15G,nvals=%zu\n",shouldbe,gradis,nvals);
        printf(".......... abs,rel error is %3.5E,%3.5E\n", abserr,relerr);
        fprintf(fp, "%3.15E %3.15E %zu %3.15E \n", 
                    tols[ii],gradis,nvals, relerr); 

        function_train_free(ft); ft = NULL;
        function_monitor_free(fm); fm = NULL;
        one_approx_opts_free_deep(&qmopts);
        free_dd(dim,start);
        fwrap_destroy(fw);
        c3approx_destroy(c3a);
    }

    fclose(fp);
    return 0;
}
