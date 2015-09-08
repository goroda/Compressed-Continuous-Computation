#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"

double myexp(double x, void * args)
{
    double * opts = args;
    double coeff = opts[0];
    double scale = opts[1];
    double out = exp(coeff * x + scale);
    return out;
}

double checkError(struct OrthPolyExpansion * ope, void * args){
    
    size_t ii;
    double * x = linspace(0,1,100);
    double err = 0.0;
    double terr = 0.0;
    double v1, v2;
    for (ii = 0; ii < 100; ii++){
        v1 = orth_poly_expansion_eval(ope,x[ii]);
        v2 = myexp(x[ii],args);
        terr += fabs(v2);
        err += fabs(v2-v1);
    }
    err /= terr;
    return err;
}

int main ()
{
    
    double scale[7] = {0.5, 1.0, 5.0, 10.0,
                        20.0, 50.0, 100.0};
                        
    enum poly_type ptype = LEGENDRE;
    double lb = 0.0;
    double ub = 1.0;

    size_t ii;
    size_t N = 7;
    for (ii = 0; ii < N; ii++){
        struct OpeAdaptOpts aopts;
        aopts.start_num = 6;
        aopts.coeffs_check = 1;
        aopts.tol = 1e-3;
        double coeff [2] = {1.0, scale[ii]};
        struct OrthPolyExpansion * ope = 
            orth_poly_expansion_approx_adapt(myexp,coeff,ptype,lb,ub,&aopts);
        double err = checkError(ope,coeff);
        printf("Err = %3.15G\n",err);
        orth_poly_expansion_free(ope);
    }
    
}
