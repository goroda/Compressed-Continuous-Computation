#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"

int myexp(size_t n, const double * x,double * out, void * args)
{
    double * opts = args;
    double coeff = opts[0];
    double scale = opts[1];
    for (size_t ii = 0; ii < n; ii++){
        out[ii] = exp(coeff * x[ii] + scale);
    }
    return 0;
}

double checkError(struct OrthPolyExpansion * ope, void * args){
    
    size_t ii;
    double * x = linspace(0,1,100);
    double err = 0.0;
    double terr = 0.0;
    double v1, v2;
    for (ii = 0; ii < 100; ii++){
        v1 = orth_poly_expansion_eval(ope,x[ii]);
        myexp(1,x+ii,&v2,args);
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
        struct OpeOpts * opts = ope_opts_alloc(ptype);
        ope_opts_set_lb(opts,lb);
        ope_opts_set_ub(opts,ub);
        ope_opts_set_start(opts,6);
        ope_opts_set_coeffs_check(opts,1);
        ope_opts_set_tol(opts,1e-3);

        double coeff [2] = {1.0, scale[ii]};
        struct Fwrap * fw = fwrap_create(1,"general-vec");
        fwrap_set_fvec(fw,myexp,coeff);
        struct OrthPolyExpansion * ope = 
            orth_poly_expansion_approx_adapt(opts,fw);
        double err = checkError(ope,coeff);
        printf("Err = %3.15G\n",err);
        orth_poly_expansion_free(ope);
        ope_opts_free(opts);
        fwrap_destroy(fw);
    }
    
}
