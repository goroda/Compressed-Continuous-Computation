#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"

//int main( int argc, char *argv[])
int main( void )
{

    size_t start = 2;
    size_t nDims = 100;
    size_t ii,jj, dim;
    double lb = -1.0;
    double ub = 1.0;
    for (dim = start; dim < (start+nDims); dim+=4){

        struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
        ope_opts_set_lb(opts,lb);
        ope_opts_set_ub(opts,ub);
        ope_opts_set_start(opts,5);
        ope_opts_set_coeffs_check(opts,4);
        ope_opts_set_tol(opts,1e-8);
        struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
        struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
        multi_approx_opts_set_all_same(fopts,qmopts);
        
        double * quad = calloc_double(dim * dim);
        double * coeff = calloc_double(dim);
        for (ii = 0; ii < dim; ii++){
            coeff[ii] = randu();
            for (jj = 0; jj < dim; jj++){
                quad[ii*dim+jj] = randu();
            }
        }
        struct FunctionTrain * f = function_train_quadratic(quad,coeff,fopts);
        struct FunctionTrain * fr = function_train_round(f,1e-8,fopts);
        
        size_t maxrank = fr->ranks[1];
        for (ii = 2; ii < dim; ii++){
            if (fr->ranks[ii] > maxrank){
                maxrank = fr->ranks[ii];
            }
        }

        printf("Dimension = %zu, max rank = %zu\n",dim,maxrank);
        free(quad);
        free(coeff);
        function_train_free(f);
        function_train_free(fr);
        multi_approx_opts_free(fopts); fopts = NULL;
        one_approx_opts_free(qmopts); qmopts = NULL;
        ope_opts_free(opts); opts = NULL;
    }
}
