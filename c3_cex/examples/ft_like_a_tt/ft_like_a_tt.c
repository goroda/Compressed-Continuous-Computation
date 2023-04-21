#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"
#include "c3_interface.h"

/**********************************************************//**
    Compute the stride length between indices to choose
    *M* indices almost uniformly from *N* options
    N must be greater than or equal to M (N>=M)
**************************************************************/
size_t uniform_stride(size_t N, size_t M)
{
    assert (N >= M);
    size_t stride = 1;
    size_t M1 = M-1;
    size_t N1 = N-1;
    while (stride * M1 < N1){
        stride++;
    }
    stride--;
    return stride;
}

// A four dimensional function f(x,y,z,w) = x + y + z + w
// n: number of points to evaluate
// x: locations of the evaluation as a flattented array
// out: locations to store evaluation
// args: optional arguments
int fourd(size_t n, const double * x, double * out, void * args)
{
    (void)(args);

    for (size_t ii = 0; ii < n; ii++){
        out[ii] = x[ii*4] + x[ii*4+1] + x[ii*4+2] + x[ii*4+3];        
    }

    return 0;
}

int main(void)
{
    size_t dim = 4;  // specify the dimension

    // Wrap the function pointer in a special structure
    struct Fwrap * fw = fwrap_create(dim, "general-vec");
    fwrap_set_fvec(fw,fourd,NULL);

    // specify grid
    size_t N[4] = {20, 12, 6, 9};  // Discretiation size in each dimension
    double * x[4];
    

    // specify options
    int verbose = 1;
    size_t init_rank = 3;
    
    struct ConstElemExpAopts *opts[4]; 
    struct OneApproxOpts *qmopts[4];
    struct C3Approx * c3a = c3approx_create(CROSS, dim);

    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        // domain is [-1,1] in every dimension
        x[ii]= linspace(-1.0, 1.0, N[ii]); 

        // boilerplate
        opts[ii] = const_elem_exp_aopts_alloc(N[ii], x[ii]);
        qmopts[ii] = one_approx_opts_alloc(CONSTELM,opts[ii]);
        c3approx_set_approx_opts_dim(c3a,ii,qmopts[ii]);


        // initialize fibers to be uniformly distributed on grid
        start[ii] = calloc_double(init_rank);
        size_t stride = uniform_stride(N[ii], init_rank);
        for (size_t jj = 0; jj < init_rank; jj++){
            start[ii][jj] = x[ii][stride*jj];
        }        
    }
    c3approx_init_cross(c3a, init_rank, verbose, start);

    // set cross approximation options
    c3approx_set_verbose(c3a, verbose);
    c3approx_set_cross_tol(c3a, 1e-10);
    c3approx_set_cross_maxiter(c3a, 10);
    c3approx_set_round_tol(c3a, 1e-10);

    // build an adaptive approximation
    struct FunctionTrain * ft = c3approx_do_cross(c3a, fw, 0);

    // Test error
    double err = 0.0;
    double den = 0.0;
    double val;
    double pt[4];

    for (size_t ii = 0; ii < N[0]; ii++){
        for (size_t jj = 0; jj < N[1]; jj++){
            for (size_t kk = 0; kk < N[2]; kk++){
                for (size_t ll = 0; ll < N[3]; ll++){
                    pt[0] = x[0][ii]; pt[1] = x[1][jj];
                    pt[2] = x[2][kk]; pt[3] = x[3][ll];
                    fourd(1,pt,&val,NULL);
                    den += pow(val,2.0);
                    double eval = function_train_eval(ft,pt);
                    /* printf("eval = %G val = %G\n",eval,val); */
                    err += pow(val-eval,2.0);
                }
            }
        }
    }

    err = err/den;
    printf("Relative Mean Squared Error = %G\n", err/den);

    free_dd(dim, start);
    for (size_t ii = 0; ii < dim; ii++){
        free(x[ii]); x[ii] = NULL;
        one_approx_opts_free_deep(&qmopts[ii]);
    }
    fwrap_destroy(fw);
    function_train_free(ft);
    c3approx_destroy(c3a);
    return 0;
}
