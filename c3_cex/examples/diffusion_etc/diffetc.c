#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"

#include "c3_interface.h"


// n is number of samples
// dim is number of dimensions
// x is the location of evaluation (n x 1)
// out is allocated storage area
// args is additional arguments
int gaussian(size_t n, size_t dim, const double *x, double * out, void * args)
{
    assert (args == NULL);
    assert (dim < 4);
    
    double scale[4] = {1.0, 2.0, 3.0, 4.0};
    double width[4] = {0.1, 0.2, 0.3, 0.4};

    for (size_t ii = 0; ii < n; ii++){
        out[ii] = scale[dim] * exp(-1.0/width[dim] * pow(x[ii],2));
    }

    return 0;
}


static void all_opts_free(
    struct Fwrap * fw,
    struct OpeOpts * opts,
    struct OneApproxOpts * qmopts,
    struct MultiApproxOpts * fopts)
{
    fwrap_destroy(fw);
    ope_opts_free(opts);
    one_approx_opts_free(qmopts);
    multi_approx_opts_free(fopts);
}

int main( int argc, char *argv[])
{
    (void) (argc);
    (void) (argv);
    size_t dim = 4;
    double lb = -5;
    double ub = 5;
    
    struct Fwrap * fw = fwrap_create(1,"mo-vec");
    fwrap_set_mofvec(fw, gaussian, NULL);


    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,5);
    ope_opts_set_coeffs_check(opts,1);
    ope_opts_set_tol(opts,1e-3);
    ope_opts_set_maxnum(opts,12);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);

    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts, qmopts);

    struct FunctionTrain * ft = function_train_rankone(fopts,fw);

    double pt[4] = {0.0, 0.4, -0.2, 0.3};
    /* double pt[4] = {0.0, 0.0, 0.0, 0.0}; */
    
    double eval = function_train_eval(ft, pt);
    printf("Evaluation = %3.5G\n", eval);
    
    double integral = function_train_integrate(ft);
    printf("Integral = %3.5G\n", integral);

    function_train_scale(ft, 1.0/integral);
    double new_integral = function_train_integrate(ft);
    printf("New integral = %3.5G\n", new_integral);

    struct FT1DArray * derivative = function_train_gradient(ft);
    double * grad = ft1d_array_eval(derivative, pt);
   
    struct FunctionTrain * a = function_train_constant(1.0, fopts);
    
    struct FunctionTrain * out = dmrg_diffusion(a, ft, 1e-5, 5, 1e-10, 0, fopts);
                                                


    // free most of the memory
    all_opts_free(fw, opts, qmopts, fopts);
    function_train_free(ft);
    ft1d_array_free(derivative);
    free(grad);
    function_train_free(a);
    function_train_free(out);
        
    // Uncomment below for cross approximation
    /* struct C3Approx * c3a = c3approx_create(CROSS, dim); */
    /* int verbose = 1; */
    /* size_t init_rank = 5; */
    /* double ** start = malloc_dd(dim); */
    /* for (size_t ii = 0; ii < dim; ii++){ */
    /*     c3approx_set_approx_opts_dim(c3a,ii,qmopts); */
    /*     start[ii] = linspace(-2, 2, init_rank); */
    /* } */

    
 

    return 0;
}
