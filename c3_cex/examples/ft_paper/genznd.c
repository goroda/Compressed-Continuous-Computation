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


double compute_int(size_t dim)
{
    
    double int1d = (exp(5.0*0.5)-exp(5.0*0.0))/5.0;
    double inttotal = pow(int1d,dim);
    return inttotal;
}

int main()
{

    size_t min_dim = 3;
    size_t delta_dim = 20;
    size_t ndims = 7;

    char integral[256];
    sprintf(integral,"integral.dat");

    FILE *fp;
    fp =  fopen(integral, "w");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open file\n");
        return 0;
    }
    fprintf(fp, "dim tol int N relerr diff reldifferr\n");


    size_t ii,dim;

    size_t ntols = 10;
    double * epsilons = logspace(-13, -1, ntols);
    
    for (ii = 0; ii < ndims; ii++){
        dim = min_dim + ii * delta_dim;
        /* aopts.epsilon = aopts.epsilon * 2.0; */
        printf("On dim (%zu/%zu) : %zu \n",ii,ndims,dim);

        double * pt = calloc_double(dim);
        double sum = 0.0;
        for (size_t zz = 0; zz < dim; zz++){
            pt[zz] = 0.2;
            sum+= pt[zz];
        }
        double grad_should_be = dim*5.0*exp(5*sum);
        double shouldbe = compute_int(dim);
    
        for (size_t ll = 0; ll < ntols; ll++){
            printf("..... On tol (%zu/%zu) : %E \n", ll,ntols, epsilons[ll]);
            struct FunctionMonitor * fm = function_monitor_initnd(discnd,&dim,dim,1000*dim);
            struct Fwrap * fw = fwrap_create(dim,"general");
            fwrap_set_f(fw,function_monitor_eval,fm);
            struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,0.0,1.0);
            pw_poly_opts_set_nregions(opts,3);
            pw_poly_opts_set_maxorder(opts,6);
            pw_poly_opts_set_minsize(opts,1e-2);
            pw_poly_opts_set_coeffs_check(opts,1);
            /* pw_poly_opts_set_tol(opts,1e-10); */
            pw_poly_opts_set_tol(opts,epsilons[ll]);

            struct OneApproxOpts * qmopts = one_approx_opts_alloc(PIECEWISE,opts);
            struct C3Approx * c3a = c3approx_create(CROSS,dim);
            int verbose = 0;
            double ** start = malloc_dd(dim);
            size_t rank = 1;
            for (size_t jj = 0; jj < dim; jj++){
                c3approx_set_approx_opts_dim(c3a,jj,qmopts);
                start[jj] = linspace(0.0,1.0,rank);
                start[jj][0] = 0.2;
            }
            c3approx_init_cross(c3a,rank,verbose,start);
            c3approx_set_cross_tol(c3a,1e-1);
            c3approx_set_cross_maxiter(c3a,1);

        
            struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);

            double * derivs = calloc_double(dim);
            function_train_gradient_eval(ft,pt,derivs);
            double grad_is = 0.0;
            for (size_t zz = 0; zz < dim; zz++){
                grad_is += derivs[zz];
            }

            free(derivs); derivs = NULL;


            double intval = function_train_integrate(ft);// * exp(5.0*0.5*(double)dim) ;
            printf("(grad,int) = (%3.15E,%3.5E) is (%3.15E,%3.5E\n",
                   grad_should_be,shouldbe,
                   grad_is,intval);
            double relerr = fabs(intval-shouldbe)/fabs(shouldbe);
            double relerrgrad = fabs(grad_is-grad_should_be)/fabs(grad_should_be);
            printf(".......... rel error is %3.5E,%3.5E\n", relerrgrad,relerr);
            size_t nvals = nstored_hashtable_cp(fm->evals);

            fprintf(fp, "%zu %3.15E %3.15E %zu %3.15E %3.15E %3.15E\n", 
                    dim, epsilons[ll],intval,nvals, relerr,grad_is,relerrgrad); 

            function_train_free(ft); ft = NULL;
            function_monitor_free(fm); fm = NULL;
            one_approx_opts_free_deep(&qmopts);
            free_dd(dim,start);
            fwrap_destroy(fw);
            c3approx_destroy(c3a);
        }
        free(pt); pt = NULL;
    }
    free(epsilons); epsilons = NULL;
    fclose(fp);
    return 0;
}
