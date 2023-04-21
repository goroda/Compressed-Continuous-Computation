#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"

#include "c3_interface.h"

struct gauss_opts
{
    double m1;
    double m2;
    double v11;
    double v12;
    double v22;
    struct storevals_main * sv;

};

int gauss(size_t n, const double * xin, double * out, void * args)
{

    struct gauss_opts * opts = args;

    double rho = opts->v12 / sqrt(opts->v11) / sqrt(opts->v22);

    double den = 1.0/2.0/M_PI/sqrt(opts->v11)/sqrt(opts->v22);
    den /= sqrt(1.0-rho*rho);

    double x,y;
    for (size_t ii = 0; ii < n; ii++){
        x = xin[ii*2+0];
        y = xin[ii*2+1];
        double z = pow(x-opts->m1,2.0)/opts->v11 + pow(y - opts->m2,2.0)/opts->v22;
        z -= 2.0 * rho * (x-opts->m1) * (y-opts->m2) / 
            sqrt(opts->v11) / sqrt(opts->v22);
        double num = exp(-z / 2.0 / (1.0-pow(rho,2.0)));

        out[ii] = num/den;

        if (opts->sv != NULL){
            double pt[2]; pt[0] = x; pt[1] = y;
            opts->sv->nEvals+= 1;
            PushVal(&(opts->sv->head),2, pt, out[ii]);
        }
    }

    return 0;
}

int main( int argc, char *argv[])
{

    if ((argc != 6) && (argc != 7)){
       printf("Correct function call = ./gauss m1 m2 v11 v22 rho rank\n");
       return 0;
    }
    // argv[5] is the correlation coefficient

    size_t dim = 2;
    size_t rank = 2;
    if (argc == 7){
        rank = (size_t)atol(argv[6]);
    }

    struct gauss_opts gopts;
    gopts.m1 = atof(argv[1]);
    gopts.m2 = atof(argv[2]);
    
    gopts.v11 = atof(argv[3]);
    gopts.v22 = atof(argv[4]);
    gopts.v12 = atof(argv[5]) * sqrt(gopts.v11)*sqrt(gopts.v22);

    char params[256];
    sprintf(params, "m1:%s-m2:%s-v11:%s-v22:%s-rho:%s",argv[1],argv[2],
                                                    argv[3],argv[4],argv[5]);
    
    struct storevals_main * sv = malloc(sizeof(struct storevals_main));
    sv->nEvals = 0;
    sv->head = NULL;
    gopts.sv = sv;

    struct Fwrap * fw = fwrap_create(dim,"general-vec");
    fwrap_set_fvec(fw,gauss,&gopts);
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,-1.0);
    ope_opts_set_ub(opts,1.0);
    ope_opts_set_start(opts,5);
    ope_opts_set_coeffs_check(opts,4);
    ope_opts_set_tol(opts,1e-8);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
    struct C3Approx * c3a = c3approx_create(CROSS,dim);
    int verbose = 2;
    double ** start = malloc_dd(dim);
    for (size_t ii = 0; ii < dim; ii++){
        c3approx_set_approx_opts_dim(c3a,ii,qmopts);
        start[ii] = linspace(-0.1,0.1,rank);
    }
    c3approx_init_cross(c3a,rank,verbose,start);
    struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);

    char final_errs[256];
    sprintf(final_errs,"%s-final.dat",params);

    char evals[256];
    sprintf(evals,"%s-evaluations.dat",params);
    
    FILE *fp;
    fp =  fopen(evals, "w");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open %s\n", argv[1]);
        return 0;
    }
    PrintVals2d(fp, sv->head);
    fclose(fp);
    DeleteStoredVals(&(sv->head));
    free(sv);


    FILE *fp2;
    fp2 =  fopen(final_errs, "w");
    if (fp2 == NULL){
        fprintf(stderr, "cat: can't open %s\n", argv[1]);
        return 0;
    }

    fprintf(fp2, "x y f f1 df1\n");
    double v1, v2;

    size_t ii,jj;
    size_t N1 = 40;
    size_t N2 = 40;
    double * xtest = linspace(-1.0,1.0,N1);
    double * ytest = linspace(-1.0,1.0,N2);
    
    //print_quasimatrix(skd_init->xqm,0,NULL);
    //print_quasimatrix(skd_init->yqm,0,NULL);

    gopts.sv = NULL;
    double out1=0.0;
    double den=0.0;
    double pts[2];
    for (ii = 0; ii < N1; ii++){
        for (jj = 0; jj < N2; jj++){
            pts[0] = xtest[ii];
            pts[1] = ytest[jj];
            gauss(1,pts,&v1,&gopts);
            v2 = function_train_eval(ft,pts);
            //v3 = skeleton_decomp_eval(skd_init,xtest[ii],ytest[jj]);

            fprintf(fp2, "%3.5f %3.5f %3.5f %3.5f %3.5f \n", 
                    xtest[ii], ytest[jj],v1,v2,v1-v2);
            //printf("v2=%3.2f\n",v2);
            //printf("v3=%3.2f\n",v3);
            den += pow(v1,2.0);
            out1 += pow(v1-v2,2.0);
            //printf("out2=%3.2f\n",out2);
        }
    }
    //printf("out2 = %3.2f\n",out2);
    printf("RMS Error of Final = %G\n", out1/den);
    

    fclose(fp2);
    free(xtest);
    free(ytest);

    function_train_free(ft);
    one_approx_opts_free_deep(&qmopts);
    fwrap_destroy(fw);
    c3approx_destroy(c3a);
    free_dd(dim,start);
    
    return 0;
}
