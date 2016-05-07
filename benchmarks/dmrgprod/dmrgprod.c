#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"
    
int main(int argc, char * argv[])
{
    srand(time(NULL));
    if ((argc != 2)){
       printf("Correct function call = ./dmrgprodbench type\n");
       printf("Options for type name include: \n");
       printf("\t \"0\": time vs r comparison between dmrg and standard prod-round\n");
       printf("\t \"1\": run for profile purposes\n");
       printf("\t \"2\": time vs r comparison for just dmrg\n");
       return 0;
    }


    size_t dim = 4;
    enum poly_type ptype = LEGENDRE;
    struct OpeOpts * opts = ope_opts_alloc(ptype);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
    multi_approx_opts_set_all_same(fopts,qmopts);
    struct BoundingBox * bds = bounding_box_init_std(dim);
    if (strcmp(argv[1],"0") == 0){

        size_t ranks[5] = {1,4,4,4,1};
        size_t maxorder = 10;

        size_t nrepeat = 1;
        size_t nranks = 15;
        size_t jj,kk;

        double * results = calloc_double(nranks*4);
        for (jj = 0; jj < nranks; jj++){
            for (kk = 1; kk < dim; kk++){
                ranks[kk] = 2 + 1*jj;
            }

            results[jj] = (double) ranks[1];

            printf("base ranks are ");
            iprint_sz(dim+1,ranks);

            double time = 0.0;
            double time2 = 0.0;
            double mrank = 0.0;
            size_t ii;
            
            for (ii = 0; ii < nrepeat; ii++){

                struct FunctionTrain * a = NULL;
                a = function_train_poly_randu(ptype,bds,ranks,maxorder);
                struct FunctionTrain * start = NULL;
                a = function_train_constant(1.0,fopts);

                clock_t tic = clock();
                struct FunctionTrain * at = function_train_product(a,a);
                struct FunctionTrain * p = function_train_copy(at);
                struct FunctionTrain * pr = function_train_round(p,1e-10,fopts);
                clock_t toc = clock();
                time += (double)(toc - tic) / CLOCKS_PER_SEC;

                //printf("Actual ranks are ");
                //iprint_sz(dim+1,pr->ranks);
                mrank += pr->ranks[2];

                tic = clock();
                struct FunctionTrain * finish = dmrg_product(start,a,a,1e-5,10,1e-10,0,fopts);
                toc = clock();
                time2 += (double)(toc - tic) / CLOCKS_PER_SEC;

                //double diff = function_train_relnorm2diff(pr,finish);
                double diff = function_train_relnorm2diff(at,finish);
                printf("diff = %G\n",diff);
                assert (diff*diff < 1e-10);
                function_train_free(p); p = NULL;
                function_train_free(pr); p = NULL;
                function_train_free(finish); finish=NULL;

                function_train_free(a); a = NULL;
                function_train_free(at); at = NULL;
                function_train_free(start); start = NULL;
            }
            time /= nrepeat;
            time2 /= nrepeat;
            mrank /= nrepeat;
            printf("Average times are (%G,%G) max rank is %G \n", time,time2,mrank);

            results[nranks+jj] = time;
            results[2*nranks+jj] = time2;
            results[3*nranks+jj] = mrank;

        }
        darray_save(nranks,4,results,"time_vs_rank.dat",1);

    }
    else{
        size_t ranks[5] = {1,15,15,15,1};
        size_t maxorder = 10;


        struct FunctionTrain * a = function_train_poly_randu(ptype,bds,ranks,
                                                             maxorder);
        struct FunctionTrain * start = function_train_constant(1.0,fopts);

        clock_t tic = clock();
        struct FunctionTrain * finish = dmrg_product(start,a,a,1e-5,10,1e-10,0,fopts);
        clock_t toc = clock();
        printf("timing is %G\n", (double)(toc - tic) / CLOCKS_PER_SEC);

        function_train_free(a); a = NULL;
        function_train_free(start); start = NULL;
        function_train_free(finish); finish = NULL;
    }
    if (strcmp(argv[1],"2") == 0){
        size_t ranks[5] = {1,4,4,4,1};
        size_t maxorder = 10;

        size_t nrepeat = 5;
        size_t nranks = 20;
        size_t jj,kk;

        double * results = calloc_double(nranks*3);
        for (jj = 0; jj < nranks; jj++){
            for (kk = 1; kk < dim; kk++){
                ranks[kk] = 2 + 1*jj;
            }

            results[jj] = (double) ranks[1];

            printf("base ranks are ");
            iprint_sz(dim+1,ranks);

            double time2 = 0.0;
            double mrank = 0.0;
            size_t ii;
            for (ii = 0; ii < nrepeat; ii++){

                struct FunctionTrain * a = NULL;
                a = function_train_poly_randu(ptype,bds,ranks,maxorder);
                struct FunctionTrain * start = NULL;
                start = function_train_constant(1.0,fopts);

                clock_t tic = clock();
                struct FunctionTrain * finish = dmrg_product(start,a,a,1e-5,10,1e-10,0,fopts);
                clock_t toc = clock();
                time2 += (double)(toc - tic) / CLOCKS_PER_SEC;
                mrank += finish->ranks[2];

                function_train_free(finish); finish=NULL;

                function_train_free(a); a = NULL;
                function_train_free(start); start = NULL;
            }
            time2 /= nrepeat;
            mrank /= nrepeat;
            printf("Average time is (%G) max rank is %G \n",time2,mrank);

            results[nranks+jj] = time2;
            results[2*nranks+jj] = mrank;

        }
        darray_save(nranks,3,results,"time_vs_rank_dmrg.dat",1);
    }


    bounding_box_free(bds); bds = NULL;
    multi_approx_opts_free(fopts);
    one_approx_opts_free_deep(&qmopts);
    return 0;
}
