#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>

#include "array.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"

double sinsum(double * x, void * args)
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
    
    enum poly_type ptype = LEGENDRE;
    size_t start_num = 6;
    size_t c_check = 2;
    struct OpeAdaptOpts ao;
    ao.start_num = start_num;
    ao.coeffs_check = c_check;
   
    size_t ntols = 10;
    double * epsilons = logspace(-13, -1, ntols);
    
    struct FtCrossArgs fca;
    ft_cross_args_init(&fca);
    fca.epsilon = 1e-5;
    fca.maxiter = 10;
    fca.verbose = 0;

    char integral[256];
    sprintf(integral,"integral.dat");

    FILE *fp;
    fp =  fopen(integral, "w");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open file\n");
        return 0;
    }
    fprintf(fp, "dim tol int N relerr \n");

    size_t ii,jj, kk;
    size_t dim;
    for (ii = 0; ii < ndims; ii++){
        dim = min_dim + ii * delta_dim;
        printf("On dim (%zu/%zu) : %zu \n",ii,ndims,dim);
        struct BoundingBox * bds = bounding_box_init_std(dim);
        size_t * ranks = calloc_size_t(dim+1);
        double ** yr  = malloc_dd(dim);
        double * coeffs = calloc_double(dim);
        for (kk = 0; kk < dim; kk++){
            bds->lb[kk] = 0.0;
            ranks[kk] = 2;
            yr[kk] = calloc_double(2);
            yr[kk][1] = 0.2; // set one fiber to something  than 0;
            coeffs[kk] = 1.0/(double) dim;
        }
        ranks[0] = 1; ranks[dim] = 1;

        fca.dim = dim;
        fca.ranks = ranks;

        double shouldbe = cimag( cpow( (cexp(I) - 1)/I , dim));
        //printf("shouldbe=%G\n",shouldbe);

        for (jj = 0; jj < ntols; jj++){
            printf("..... On tol (%zu/%zu) : %E \n", jj,ntols, epsilons[jj]);

            struct CrossIndex ** isl = malloc(dim * sizeof(struct CrossIndex *));
            struct CrossIndex ** isr = malloc(dim * sizeof(struct CrossIndex *));
            assert (isl != NULL);
            assert (isr != NULL);
            cross_index_array_initialize(dim,isl,1,0,NULL,NULL);
            cross_index_array_initialize(dim,isr,0,1,ranks,yr);

            ao.tol = epsilons[jj]; 
            struct FtApproxArgs * fapp = 
                ft_approx_args_createpoly(dim,&ptype,&ao);

            struct FunctionTrain * ftref = 
                    function_train_linear(dim, bds, coeffs,NULL);
            
            struct FunctionMonitor * fm = 
                    function_monitor_initnd(sinsum,&dim,dim,1000*dim);
            
            struct FunctionTrain * ft = ftapprox_cross(function_monitor_eval, fm,
                                            bds, ftref, isl, isr, &fca,fapp);
            
            
            double intval = function_train_integrate(ft);
            double relerr = fabs(intval-shouldbe)/fabs(shouldbe);
            //printf("intval=%G\n",intval);
            printf(".......... rel error is %G\n", relerr);
            size_t nvals = nstored_hashtable_cp(fm->evals);

            fprintf(fp, "%zu %3.15E %3.15E %zu %3.15E \n", 
                            dim,epsilons[jj],intval,nvals, relerr); 

            function_train_free(ft); ft = NULL;
            function_train_free(ftref); ftref = NULL;
            function_monitor_free(fm); fm = NULL;
            ft_approx_args_free(fapp); fapp = NULL;
            for (size_t ll = 0; ll < dim; ll++){
                cross_index_free(isr[ll]); isr[ll] = NULL;
                cross_index_free(isl[ll]); isl[ll] = NULL;
            }
            free(isr); isr = NULL;
            free(isl); isl = NULL;
//            index_set_array_free(dim, isr); isr = NULL;
//            index_set_array_free(dim, isl); isl = NULL;
        }

        bounding_box_free(bds); bds = NULL;
        free(ranks); ranks = NULL;
        free_dd(dim,yr); yr = NULL;
        free(coeffs); coeffs = NULL;
    }

    free(epsilons);
    fclose(fp);
    return 0;
}
