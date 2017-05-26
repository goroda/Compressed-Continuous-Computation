#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"
#include "c3_interface.h"

double disc2d(const double * xy, void * args)
{
    assert (args == NULL);
     
    double x = xy[0];
    double y = xy[1];
    double out = 0.0;
    if ((x > 0.5) || (y > 0.5)){
        out = 0.0;    
    }
    else{
        out = exp(5.0 * x + 5.0 * y);
        //out = x+y;
    }
    return out;
}

int main(void)
{

//    if ((argc != 2) && (argc != 1)){
//       printf("Correct function call = ./genz2d \n");
//       return 0;
//    }

    size_t dim = 2;
    struct FunctionMonitor * fm = 
            function_monitor_initnd(disc2d,NULL,dim,1000*dim);

    struct Fwrap * fw = fwrap_create(dim,"general");
    fwrap_set_f(fw,function_monitor_eval,fm);
    struct PwPolyOpts * opts = pw_poly_opts_alloc(LEGENDRE,0.0,1.0);
    pw_poly_opts_set_nregions(opts,3);
    pw_poly_opts_set_maxorder(opts,6);
    pw_poly_opts_set_minsize(opts,1e-2);
    pw_poly_opts_set_coeffs_check(opts,1);
    pw_poly_opts_set_tol(opts,1e-10);
    
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(PIECEWISE,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    int verbose = 2;
    size_t rank = 1;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(0.1,1.0,rank);
    }
    c3approx_init_cross(c3a,rank,verbose,start);
    c3approx_set_cross_tol(c3a,1e-1);
    c3approx_set_cross_maxiter(c3a,3);
    c3approx_set_verbose(c3a,2);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,0);

    char final_errs[256];
    sprintf(final_errs,"final.dat");

    char evals[256];
    sprintf(evals,"evaluations.dat");
    
    FILE *fp;
    fp =  fopen(evals, "w");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open %s\n",evals);
        return 0;
    }
    function_monitor_print_to_file(fm,fp);
    fclose(fp);

    FILE *fp2;
    fp2 =  fopen(final_errs, "w");
    if (fp2 == NULL){
        fprintf(stderr, "cat: can't open %s\n", final_errs);
        return 0;
    }

    fprintf(fp2, "x y f f0 df0\n");
    double v1, v2;

    size_t ii,jj;
    size_t N1 = 40;
    size_t N2 = 40;
    double * xtest = linspace(0.0,1.0,N1);
    double * ytest = linspace(0.0,1.0,N2);
    
    //print_quasimatrix(skd_init->xqm,0,NULL);
    //print_quasimatrix(skd_init->yqm,0,NULL);

    double out1=0.0;
    double den=0.0;
    double pt[2];
    for (ii = 0; ii < N1; ii++){
        for (jj = 0; jj < N2; jj++){
            pt[0] = xtest[ii]; pt[1] = ytest[jj];
            v1 = disc2d(pt, NULL);
            v2 = function_train_eval(ft,pt);

            fprintf(fp2, "%3.5f %3.5f %3.5f %3.5f %3.5f \n", 
                    xtest[ii], ytest[jj],v1,v2,v1-v2);
            den += pow(v1,2.0);
            out1 += pow(v1-v2,2.0);
        }
    }
    printf("RMS Error of Final = %G\n", out1/den);
    
    fclose(fp2);
    function_monitor_free(fm);
    free(xtest);
    free(ytest);

    function_train_free(ft);
    one_approx_opts_free_deep(&qmopts);
    fwrap_destroy(fw);
    c3approx_destroy(c3a);

    return 0;
}
