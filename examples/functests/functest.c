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


double doug(double * x, void * args)
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
    double lb[2] = {-1.0, -1.0};
    double ub[2] = {1.0, 1.0};
    size_t start_rank =4;
    int verbose = 1;
    struct C3Approx * c3a = c3approx_create(CROSS,dx,lb,ub);
    c3approx_init_poly(c3a,LEGENDRE);
    c3approx_init_cross(c3a,start_rank,verbose);
    c3approx_set_poly_adapt_nstart(c3a,3);
    c3approx_set_poly_adapt_ncheck(c3a,0);
    c3approx_set_poly_adapt_nmax(c3a,3);
    c3approx_set_cross_tol(c3a,1e-4);
    c3approx_set_round_tol(c3a,1e-4);

    size_t counter = 0;
    struct FunctionTrain * f = c3approx_do_cross(c3a,doug,&counter);
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
