#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"

struct gauss_opts
{
    double m1;
    double m2;
    double v11;
    double v12;
    double v22;
    struct storevals_main * sv;

};

double gauss(double x, double y, void * args)
{
    struct gauss_opts * opts = args;

    double rho = opts->v12 / sqrt(opts->v11) / sqrt(opts->v22);

    double den = 1.0/2.0/M_PI/sqrt(opts->v11)/sqrt(opts->v22);
    den /= sqrt(1.0-rho*rho);
    
    double z = pow(x-opts->m1,2.0)/opts->v11 + pow(y - opts->m2,2.0)/opts->v22;
    z -= 2.0 * rho * (x-opts->m1) * (y-opts->m2) / 
                    sqrt(opts->v11) / sqrt(opts->v22);
    double num = exp(-z / 2.0 / (1.0-pow(rho,2.0)));

    double out = num/den;

    if (opts->sv != NULL){
        double pt[2]; pt[0] = x; pt[1] = y;
        opts->sv->nEvals+= 1;
        PushVal(&(opts->sv->head),2, pt, out);
    }

    return out;
}

int main( int argc, char *argv[])
{

    if ((argc != 6) && (argc != 7)){
       printf("Correct function call = ./gauss m1 m2 v11 v22 rho rank\n");
       return 0;
    }
    // argv[5] is the correlation coefficient
    
    size_t rank = 2;
    if (argc == 7){
        rank = (size_t)atol(argv[6]);
    }
    size_t dim = 2;

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

    struct BoundingBox * bounds = bounding_box_init_std(2); 
    
    double * pivx = linspace(-0.1,0.1, rank);
    double * pivy = linspace(-0.1,0.1, rank);
    
    pivx[0] = gopts.m1;
    pivy[0] = gopts.m2;
    
    struct OpeAdaptOpts opts;
    opts.start_num = 5;
    opts.coeffs_check= 4;
    opts.tol = 1e-8;


    enum function_class fc = POLYNOMIAL;
    enum poly_type p = LEGENDRE;
    struct Cross2dargs cargs;
    cargs.r = rank;
    cargs.delta = 5e-3;
    size_t kk;

    for (kk = 0; kk < dim; kk++){
        cargs.fclass[kk] = fc;
        cargs.sub_type[kk] = &p;
        cargs.approx_args[kk] = &opts;
    }
    cargs.verbose = 2;
    
    printf("xpivots = ");
    dprint(cargs.r,pivx);
    printf("pivots = ");
    dprint(cargs.r,pivy);

    struct SkeletonDecomp * skd = skeleton_decomp_init2d_from_pivots(
                    gauss, &gopts, bounds, cargs.fclass, cargs.sub_type,
                    cargs.r, pivx, pivy, cargs.approx_args);
    
    /*
    printf("skeleton = \n");
    dprint2d_col(skd->r,skd->r, skd->skeleton);
    return(0);
    */

    struct SkeletonDecomp * skd_init = skeleton_decomp_copy(skd);

    struct SkeletonDecomp * final = cross_approx_2d(gauss,&gopts,bounds,
                        &skd,pivx,pivy,&cargs);

    //printf("Params=%s\n",params);


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

    fprintf(fp2, "x y f f0 f1 df0 df1\n");
    double v1, v2, v3;

    size_t ii,jj;
    size_t N1 = 40;
    size_t N2 = 40;
    double * xtest = linspace(-1.0,1.0,N1);
    double * ytest = linspace(-1.0,1.0,N2);
    
    //print_quasimatrix(skd_init->xqm,0,NULL);
    //print_quasimatrix(skd_init->yqm,0,NULL);

    gopts.sv = NULL;
    double out1=0.0;
    double out2=0.0;
    double den=0.0;
    for (ii = 0; ii < N1; ii++){
        for (jj = 0; jj < N2; jj++){
            v1 = gauss(xtest[ii],ytest[jj],&gopts);
            v2 = skeleton_decomp_eval(final,xtest[ii],ytest[jj]);
            v3 = skeleton_decomp_eval(skd_init,xtest[ii],ytest[jj]);

            fprintf(fp2, "%3.5f %3.5f %3.5f %3.5f %3.5f %3.5f %3.5f \n", 
                    xtest[ii], ytest[jj],v1,v2,v3, v1-v2,v1-v3);
            //printf("v2=%3.2f\n",v2);
            //printf("v3=%3.2f\n",v3);
            den += pow(v1,2.0);
            out1 += pow(v1-v2,2.0);
            out2 += pow(v1-v3,2.0);
            //printf("out2=%3.2f\n",out2);
        }
    }
    //printf("out2 = %3.2f\n",out2);
    printf("RMS Error of Initial = %G\n", out2/den);
    printf("RMS Error of Final = %G\n", out1/den);
    
    fclose(fp2);
    free(pivx);
    free(pivy);
    bounding_box_free(bounds);
    skeleton_decomp_free(final);
    skeleton_decomp_free(skd);
    skeleton_decomp_free(skd_init);
    free(xtest);
    free(ytest);
    return 0;
}
