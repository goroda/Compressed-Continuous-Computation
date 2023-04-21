#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "c3.h"

double compute_int(size_t dim)
{
    
    double int1d = (exp(5.0*0.5)-exp(5.0*0.0))/5.0;
    double inttotal = pow(int1d,dim);
    return inttotal;
}

static size_t ncalls;
double discnd(const double * x, void * args)
{

    ncalls++;
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
    fprintf(fp, "tol deriv N relerr_deriv relerr_int \n");

    double int_shouldbe = compute_int(dim);
    double tols[11] = {1e-2,1e-4,1e-6,1e-8,1e-10,1e-12,1e-14,1e-16,1e-18,1e-20,1e-24};
    size_t ntols = 10;
    for (size_t ii = 0; ii < ntols; ii++){
        ncalls = 0;
        /* aopts.epsilon = aopts.epsilon * 2.0; */
        printf("On tol (%zu/%zu) : %G \n",ii,ntols-1,tols[ii]);

        struct Fwrap * fw = fwrap_create(dim,"general");
        
        struct FunctionMonitor * fm = function_monitor_initnd(discnd,&dim,dim,1000*dim);
        fwrap_set_f(fw,function_monitor_eval,fm);
        
        /* fwrap_set_f(fw,discnd,&dim); */
        
        struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,0.0,1.0);
        size_t nregions = 3;
        pw_poly_opts_set_nregions(opts,nregions);
        pw_poly_opts_set_maxorder(opts,6);
        pw_poly_opts_set_minsize(opts,pow(1.0/(double)nregions,3));
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
        c3approx_set_cross_tol(c3a,1e-20);
        c3approx_set_round_tol(c3a,1e-20);
        c3approx_set_cross_maxiter(c3a,10);
        
        struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);


        double int_is = function_train_integrate(ft);
        double int_err = fabs(int_is-int_shouldbe);
        double int_rel_err = int_err / fabs(int_shouldbe);

        
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
        /* size_t nvals = nstored_hashtable_cp(fm->evals); */
        size_t nvals = ncalls;
        printf("shouldbe =%3.15G gradis=%3.15G,nvals=%zu\n",shouldbe,gradis,nvals);
        printf(".......... abs,rel error is %3.5E,%3.5E\n", abserr,relerr);
        printf(".......... int error is %3.5E\n",int_rel_err);
        fprintf(fp, "%3.15E %3.15E %zu %3.15E %3.15E\n", 
                tols[ii],gradis,nvals, relerr,int_rel_err); 

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
