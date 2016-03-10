#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"

double discnd(double * x, void * args)
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

    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.minsize = 1e-5;
    aopts.coeff_check= 0;
    aopts.epsilon = 1e-5;
    aopts.other = NULL;

    char integral[256];
    sprintf(integral,"integral.dat");

    FILE *fp;
    fp =  fopen(integral, "w");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open file\n");
        return 0;
    }
    fprintf(fp, "dim int N relerr \n");

    size_t ii,kk,dim;
    for (ii = 0; ii < ndims; ii++){
        dim = min_dim + ii * delta_dim;
        aopts.epsilon = aopts.epsilon * 2.0;
        printf("On dim (%zu/%zu) : %zu \n",ii,ndims,dim);
        struct BoundingBox * bds = bounding_box_init_std(dim);
        size_t * ranks = calloc_size_t(dim+1);
        double * coeffs = calloc_double(dim);        
        double ** yr  = malloc_dd(dim);

        for (kk = 0; kk < dim; kk++){
            bds->lb[kk] = 0.0;
            ranks[kk] = 1;
            coeffs[kk] = 1.0/(double) dim;
            yr[kk] = calloc_double(1);
            yr[kk][0] = 0.3;
        }
        ranks[0] = 1; ranks[dim] = 1;
        
        struct FunctionTrain * ftref = 
            function_train_linear(POLYNOMIAL,LEGENDRE,dim, bds, coeffs,NULL);
        
        struct FunctionMonitor * fm = 
            function_monitor_initnd(discnd,&dim,dim,1000*dim);

        struct CrossIndex ** isl = malloc(dim * sizeof(struct CrossIndex *));
        struct CrossIndex ** isr = malloc(dim * sizeof(struct CrossIndex *));
        assert (isl != NULL);
        assert (isr != NULL);
        cross_index_array_initialize(dim,isl,1,0,NULL,NULL);
        cross_index_array_initialize(dim,isr,0,1,ranks,yr);

        struct FtApproxArgs * fapp = 
           ft_approx_args_createpwpoly(dim,&(aopts.ptype),&aopts);

        struct FtCrossArgs fca;
        ft_cross_args_init(&fca);
        fca.epsilon = 1e-1;
        fca.maxiter = 1;
        fca.verbose = 2;
        fca.dim = dim;
        fca.ranks = ranks;

        struct FunctionTrain * ft = 
                ftapprox_cross(function_monitor_eval, fm,
                               bds, ftref, isl, isr, &fca,fapp);
        
        
        double shouldbe = compute_int(dim);

        double intval = function_train_integrate(ft);// * exp(5.0*0.5*(double)dim) ;
        printf("shouldbe =%3.15G intval=%3.15G\n",shouldbe,intval);
        double relerr = fabs(intval-shouldbe)/fabs(shouldbe);
        printf(".......... rel error is %G\n", relerr);
        size_t nvals = nstored_hashtable_cp(fm->evals);

        fprintf(fp, "%zu %3.15E %zu %3.15E \n", 
                    dim, intval,nvals, relerr); 

        function_train_free(ft); ft = NULL;
        ft_approx_args_free(fapp); fapp = NULL;
        for (size_t ll = 0; ll < dim; ll++){
            cross_index_free(isr[ll]); isr[ll] = NULL;
            cross_index_free(isl[ll]); isl[ll] = NULL;
        }
        free(isr); isr = NULL;
        free(isl); isl = NULL;
        function_monitor_free(fm); fm = NULL;
        function_train_free(ftref); ftref = NULL;
        free(ranks); ranks = NULL;
        free_dd(dim,yr); yr = NULL;
        free(coeffs); coeffs = NULL;
        bounding_box_free(bds);
    }

    fclose(fp);
    return 0;
}
