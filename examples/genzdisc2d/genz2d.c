#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"

double disc2d(double * xy, void * args)
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

int main( int argc, char *argv[])
{

    if ((argc != 2) && (argc != 1)){
       printf("Correct function call = ./genz2d \n");
       return 0;
    }
    
    size_t dim = 2;
    
    struct BoundingBox * bds = bounding_box_init(2,0.0,1.0); 
    
    double coeffs[2] = {0.5, 0.5};
    size_t ranks[3] = {1, 1, 1};

    struct FunctionTrain * ftref = 
            function_train_linear(dim, bds, coeffs,NULL);
            
    struct FunctionMonitor * fm = 
            function_monitor_initnd(disc2d,NULL,dim,1000*dim);
            
    double * yr  = calloc_double(dim);
    struct IndexSet ** isr = index_set_array_rnested(dim, ranks, yr);
    struct IndexSet ** isl = index_set_array_lnested(dim, ranks, yr);


    struct FtCrossArgs fca;
    fca.epsilon = 1e-1;
    fca.maxiter = 1;
    fca.verbose = 2;
    fca.dim = dim;
    fca.ranks = ranks;
    
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 5;
    aopts.minsize = 1e-3;
    aopts.coeff_check= 1 ;
    aopts.epsilon=1e-3;
    aopts.other = NULL;

    enum poly_type ptype = LEGENDRE;
    struct FtApproxArgs * fapp = ft_approx_args_createpwpoly(dim,&ptype,&aopts);

    struct FunctionTrain * ft = ftapprox_cross(function_monitor_eval, fm,
                                    bds, ftref, isl, isr, &fca,fapp);
    //struct FunctionTrain * ft = function_train_round(fr, 1e-5);

    //struct FunctionTrain * ft = ftapprox_cross(disc2d, NULL,
    //                                bds, dim, ftref, isl, isr, &fca);

    free(yr);
    ft_approx_args_free(fapp);
    index_set_array_free(dim,isr);
    index_set_array_free(dim,isl);
            
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
    bounding_box_free(bds);
    function_train_free(ftref);
    function_train_free(ft);
    function_monitor_free(fm);
    free(xtest);
    free(ytest);
    return 0;
}
