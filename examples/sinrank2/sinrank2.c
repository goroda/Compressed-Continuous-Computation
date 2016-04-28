#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"

struct sinrank2_opts
{
    struct storevals_main * sv;
};

double sinrank2(double x, double y, void * args)
{
    struct sinrank2_opts * opts = args;

    double out = sin(10.0* x  + 0.25)*(y+1.0);
    //double out = x*(y+1.0);

    if (opts->sv != NULL){
        double pt[2]; pt[0] = x; pt[1] = y;
        opts->sv->nEvals+= 1;
        PushVal(&(opts->sv->head),2, pt, out);
    }

    return out;
}

int main( int argc, char *argv[])
{

    if ((argc != 2) && (argc != 1)){
       printf("Correct function call = ./sinrank2 rank\n");
       return 0;
    }
    // argv[5] is the correlation coefficient
    
    size_t rank = 2;
    if (argc == 2){
        rank = (size_t)atol(argv[1]);
    }

    struct sinrank2_opts gopts;

    char params[256];
    sprintf(params, "rank=%zu",rank);
    

    struct storevals_main * sv = malloc(sizeof(struct storevals_main));
    sv->nEvals = 0;
    sv->head = NULL;
    gopts.sv = sv;

    struct BoundingBox * bounds = bounding_box_init_std(2); 
    
    double * pivx = linspace(-1.0,1.0, rank);
    double * pivy = linspace(-1.0,1.0, rank);

    struct OpeAdaptOpts * aopts = ope_adapt_opts_alloc();
    ope_adapt_opts_set_start(aopts,3);
    ope_adapt_opts_set_coeffs_check(aopts,2);
    ope_adapt_opts_set_tol(aopts,1e-4);
    

    enum poly_type p = LEGENDRE;
    int verbose = 2;
    struct Cross2dargs * cargs=cross2d_args_create(rank,1e-4,POLYNOMIAL,&p,verbose);
    cross2d_args_set_approx_args(cargs,aopts);

    printf("xpivots = ");
    dprint(rank,pivx);
    printf("pivots = ");
    dprint(rank,pivy);

    struct SkeletonDecomp * skd = skeleton_decomp_init2d_from_pivots(
        sinrank2, &gopts, bounds, cargs,pivx,pivy);

    
    printf("skeleton = \n");
    dprint2d_col(rank,rank, skeleton_get_skeleton(skd));
    //return(0);

    struct SkeletonDecomp * skd_init = skeleton_decomp_copy(skd);

    struct SkeletonDecomp * final = cross_approx_2d(sinrank2,&gopts,bounds,
                        &skd,pivx,pivy,cargs);

    //printf("Params=%s\n",params);
    printf("xpivots = ");
    dprint(rank,pivx);
    printf("pivots = ");
    dprint(rank,pivy);

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
            v1 = sinrank2(xtest[ii],ytest[jj],&gopts);
            //v2 = skeleton_decomp_eval(final,xtest[ii],ytest[jj]);
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
    ope_adapt_opts_free(aopts);
    cross2d_args_destroy(cargs);
    bounding_box_free(bounds);
    skeleton_decomp_free(final);
    skeleton_decomp_free(skd);
    skeleton_decomp_free(skd_init);
    free(xtest);
    free(ytest);
    return 0;
}
