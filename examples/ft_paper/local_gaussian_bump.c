#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>

#include "array.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"

#include "c3_interface.h"


static double gauss_lb = 0.0;
static double gauss_ub = 1.0;
static double gauss_center = 0.2;
static double gauss_width = 0.05;
double gauss(const double * x, void * arg)
{
    size_t *dim = arg;
    size_t d = *dim;
    
    long double center = gauss_center;
    long double l = gauss_width;
    long double preexp = 1.0/(l * sqrt(M_PI*2));

    long double inexp = 0.0;
    for (size_t jj = 0; jj < d; jj++){
        long double dx = x[jj] - center;
        inexp += dx*dx;
    }
    inexp *= -1;
    inexp /= (2.0*l*l);

    /* printf("inexp = %3.15LE\n",inexp); */
    /* printf("d = %zu\n",d); */
    /* dprint(d,x); */
    double out = preexp * expl(inexp);
    /* long double lout = preexp * exp(inexp); */
    /* printf("out = %3.15E\n",out); */
    /* printf("out = %3.15LE\n",lout); */
    return out;
}

int main( void )
{

    size_t dim = 3;

    struct FunctionMonitor * fm = function_monitor_initnd(gauss,&dim,dim,1000*dim);
    struct Fwrap * fw = fwrap_create(dim,"general");
    fwrap_set_f(fw,function_monitor_eval,fm);
    /* fwrap_set_f(fw,gauss,&dim); */
    
    /* struct OpeOpts * opts = ope_opts_alloc(LEGENDRE); */
    /* ope_opts_set_lb(opts,gauss_lb); */
    /* ope_opts_set_ub(opts,gauss_ub); */
    /* ope_opts_set_start(opts,3); */
    /* ope_opts_set_maxnum(opts,20); */
    /* ope_opts_set_coeffs_check(opts,0); */
    /* ope_opts_set_tol(opts,1e-10); */
    /* struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts); */

    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,gauss_lb,gauss_ub);
    size_t nregion = 3;
    pw_poly_opts_set_nregions(opts,nregion);
    pw_poly_opts_set_maxorder(opts,5);
    pw_poly_opts_set_minsize(opts,pow(1.0/(double)nregion,15));
    pw_poly_opts_set_coeffs_check(opts,1);
    pw_poly_opts_set_tol(opts,1e-2);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(PIECEWISE,opts);


    /* double hmin = 1e-15; */
    /* double tol = 1e-5; */
    /* struct LinElemExpAopts * opts = lin_elem_exp_aopts_alloc_adapt(0,NULL,gauss_lb,gauss_ub,tol,hmin); */
    /* struct OneApproxOpts * qmopts = one_approx_opts_alloc(LINELM,opts); */

    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    int verbose = 0;
    size_t rank = 2;
    double ** start = malloc_dd(dim);
    for (size_t kk= 0; kk < dim; kk++){
        c3approx_set_approx_opts_dim(c3a,kk,qmopts);
        start[kk] = linspace(gauss_lb+0.1,gauss_ub-0.1,rank);
    }
    c3approx_init_cross(c3a,rank,verbose,start);
    c3approx_set_cross_tol(c3a,1e-14);
    c3approx_set_round_tol(c3a,1e-14);


    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);
    printf("ft ranks = "); iprint_sz(dim+1,function_train_get_ranks(ft));

    /* double intexact_ref = 1.253235e-1; */

    
    double intexact;
    if (dim == 2){
        intexact = 0.5 * gauss_width * sqrt(M_PI/2.0) *
            ((erf((gauss_center-1)/(sqrt(2.) * gauss_width)) -
              erf((gauss_center)/(sqrt(2.) * gauss_width))) *
             (erf((gauss_center-1)/(sqrt(2.) * gauss_width)) -
              erf((gauss_center)/(sqrt(2.) * gauss_width))));
    }
    else{
        intexact = -0.25 * gauss_width * gauss_width * M_PI *
            ((erf((gauss_center-1)/(sqrt(2.) * gauss_width)) -
              erf((gauss_center)/(sqrt(2.) * gauss_width))) *
             (erf((gauss_center-1)/(sqrt(2.) * gauss_width)) -
              erf((gauss_center)/(sqrt(2.) * gauss_width))) *
             (erf((gauss_center-1)/(sqrt(2.) * gauss_width)) -
              erf((gauss_center)/(sqrt(2.) * gauss_width))));
    }
    /* double intexact = 0.5 * gauss_width * sqrt(M_PI/2.0) * */
    /*     ((erf((gauss_center-1)/(sqrt(2.) * gauss_width)) - */
    /*       erf((gauss_center+1)/(sqrt(2.) * gauss_width))) * */
    /*      (erf((gauss_center-1)/(sqrt(2.) * gauss_width)) - */
    /*       erf((gauss_center+1)/(sqrt(2.) * gauss_width)))); */
    double intval = function_train_integrate(ft);
    double error = fabs(intexact - intval);
    size_t nvals = nstored_hashtable_cp(fm->evals);

    
    /* fprintf(fp, "%zu %3.15E %zu %3.15E \n", dim,,intval,nvals, error); */
    fprintf(stdout, "%zu %3.15E %3.15E %zu %3.15E \n", dim,intexact,intval,nvals, error); 

    function_train_free(ft); ft = NULL;
    function_monitor_free(fm); fm = NULL;
    one_approx_opts_free_deep(&qmopts);
    fwrap_destroy(fw);
    c3approx_destroy(c3a);
    free_dd(dim,start);
}
