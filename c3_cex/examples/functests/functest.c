#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"
#include "lib_probability.h"

#include "c3_interface.h"

double doug(const double * x, void * args)
{

    double out = 1.0;
    out += x[0] + 0.5*(3.0*x[1]*x[1] - 1.0) + x[0]*x[1];
    size_t * counter = args;
    *counter += 1;
    return out;
}

int main()
{
    size_t dx = 2;
    size_t dim = dx;
        
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_start(opts,3);
    ope_opts_set_coeffs_check(opts,3);
    ope_opts_set_maxnum(opts,3);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);

    int verbose = 1;
    size_t init_rank = 4;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(-1.0,1.0,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    c3approx_set_cross_tol(c3a,1e-4);
    c3approx_set_round_tol(c3a,1e-4);
    
    
    size_t counter = 0;
    struct Fwrap * fw = fwrap_create(dx,"general");
    fwrap_set_f(fw,doug,&counter);
    
    struct FunctionTrain * f = c3approx_do_cross(c3a,fw,1);
    free_dd(dim,start);
    one_approx_opts_free_deep(&qmopts);
    c3approx_destroy(c3a);
    fwrap_destroy(fw);

    printf("ranks are "); iprint_sz(3,f->ranks);
    printf("Number of evaluations is %zu\n",counter);
    
    double derr = 0.0;
    double dden = 0.0;
    size_t N = 100;
    double * xtest = linspace(-1.0,1.0,N);
    size_t ii,jj;
    double x[2];
    size_t counter2 = 0;
    for (ii = 0; ii < N; ii++){
        for(jj = 0; jj < N; jj++){
            x[0] = xtest[ii];
            x[1] = xtest[jj];
            double tval = doug(x,&counter2);
            double aval = function_train_eval(f,x);
            derr += pow(tval-aval,2);
            dden += pow(tval,2);
        }
    }
    printf("L2 error for displacement is %G\n",derr/dden);

    function_train_free(f);

    return 0;
}
