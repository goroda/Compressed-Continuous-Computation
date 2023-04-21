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

static const double w = 4.0; // width
static const double t = 2.0; // thickness
static const double L = 100.0; // length
static const double D_0 = 2.2535; // Displacement tolerance
static const double E = 2.9e7;
static const double R = 40000.0;

double displacement(const double * x, void * args)
{
    assert (args == NULL );
    
    double X = icdf_normal(500.0,100.0,x[0]);
    double Y = icdf_normal(1000.0,100.0,x[1]);
    
    //printf("x=%G, X=%G\n",x[0],X);
    //printf("y=%G, Y=%G\n",x[1],Y);

    double dfact1 = 4*L*L*L / (E*w*t);
    double t1 = pow(Y/t/t,2);
    double t2 = pow(X/w/w,2);
    double dfact2 = dfact1 * sqrt(t1+t2);
    dfact2 = dfact2 / D_0 - 1.0;
    //printf("dfact2 = %G\n",dfact2);

    return dfact2;
}

double stress(const double * x, void * args)
{
    assert (args == NULL );
    
    double X = icdf_normal(500.0,100.0,x[0]);
    double Y = icdf_normal(1000.0,100.0,x[1]);

    double sfact1 = 600.0*Y/w/t/t;
    double sfact2 = 600.0*X/w/w/t;
    double out = sfact1 + sfact2;
    out = out / R - 1.0;
    return out;
}

int main()
{
    size_t dim = 2;
    struct Fwrap * fw = fwrap_create(dim,"general");
    fwrap_set_f(fw,displacement,NULL);
    struct Fwrap * fw2 = fwrap_create(dim,"general");
    fwrap_set_f(fw2,stress,NULL);

//    struct BoundingBox * bds = bounding_box_init(dim,0.01,.99);
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,0.01);
    ope_opts_set_ub(opts,0.99);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    int verbose = 0;
    size_t init_rank = 5;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(0.01,0.99,init_rank);
    }
    c3approx_init_cross(c3a,init_rank,verbose,start);
    c3approx_set_cross_tol(c3a,1e-10);
    c3approx_set_round_tol(c3a,1e-10);

    //ft_cross_args_set_maxrank_all(fca,init_ranks+10*5); 

    struct FunctionTrain * d = NULL; // displacement
    struct FunctionTrain * s = NULL; // stress

    d = c3approx_do_cross(c3a,fw,1);
    printf("Displacement rank is %zu.\n",d->ranks[1]);

    s = c3approx_do_cross(c3a,fw2,1);
    printf("Stress rank is %zu.\n", s->ranks[1]);
    
    double derr = 0.0;
    double dden = 0.0;
    double serr = 0.0;
    double sden = 0.0;
    size_t N = 100;
    double * xtest = linspace(0.1,0.9,N);
    size_t ii,jj;
    double x[2];
    for (ii = 0; ii < N; ii++){
        for(jj = 0; jj < N; jj++){
            x[0] = xtest[ii];
            x[1] = xtest[jj];
            double tval = displacement(x,NULL);
            double aval = function_train_eval(d,x);
            derr += pow(tval-aval,2);
            dden += pow(tval,2);

            tval = stress(x,NULL);
            aval = function_train_eval(s,x);
            serr += pow(tval-aval,2);
            sden += pow(tval,2);
        }
    }
    printf("L2 error for displacement is %G\n",derr/dden);
    printf("L2 error for stress is %G\n",serr/sden);

    function_train_free(d);
    function_train_free(s);
    free_dd(dim,start);
    one_approx_opts_free_deep(&qmopts);
    c3approx_destroy(c3a);
    fwrap_destroy(fw);
    fwrap_destroy(fw2);
    return 0;
}
