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

const double w = 4.0; // width
const double t = 2.0; // thickness
const double L = 100.0; // length
const double D_0 = 2.2535; // Displacement tolerance
const double E = 2.9e7;
const double R = 40000.0;

double displacement(double * x, void * args)
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

double stress(double * x, void * args)
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
    struct BoundingBox * bds = bounding_box_init(dim,0.01,.99);

    size_t init_ranks[3] = {1,5,1};
    struct FtCrossArgs fca;
    fca.dim = dim;
    fca.ranks = init_ranks;
    fca.ranks[dim] = 1;
    fca.epsilon = 1e-10;
    fca.maxiter = 10;
    fca.epsround = 1e-10;
    fca.kickrank = 5;
    fca.maxiteradapt = 10;
    fca.verbose = 0;

    struct FunctionTrain * d = NULL; // displacement
    struct FunctionTrain * s = NULL; // stress

    d = function_train_cross(displacement,NULL,bds,NULL,&fca,NULL);
    printf("Displacement rank is %zu.\n",d->ranks[1]);

    s = function_train_cross(stress,NULL,bds,NULL,NULL,NULL);
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
    bounding_box_free(bds);
    return 0;
}
