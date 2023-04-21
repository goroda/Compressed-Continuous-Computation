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
       printf("Correct function call = ./qrbench type\n");
       printf("Options for type name include: \n");
       printf("\t \"0\": time vs. order, \"1\": time vs r1, \"2\": time vs r2 \n");
       return 0;
    }

    double lb = -1.0;
    double ub = 1.0;
    size_t r1 = 25;
    size_t r2 = 25;
    size_t maxorder = 5;
    size_t nrepeats = 100;

    enum poly_type ptype = LEGENDRE;
    struct OpeOpts * opts = ope_opts_alloc(ptype);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    if (strcmp(argv[1],"0") == 0){
        size_t nOrders = 40;
        double * results = calloc_double(nOrders*2);
        size_t ii,jj;
        for (ii = 0; ii < nOrders; ii++){

            maxorder = 1 + 2*ii;
            results[ii] = maxorder;
            
            for (jj = 0; jj < nrepeats; jj++){
                struct Qmarray * mat = qmarray_poly_randu(LEGENDRE,r1,r2,maxorder,lb,ub);
                double * R = NULL;//calloc_double(r2 * r2);
                struct Qmarray * Q = NULL;

                // method 1
                clock_t tic = clock();
                qmarray_qr(mat,&Q,&R,qmopts);
                //qmarray_householder_simple("QR",mat,R);
                clock_t toc = clock();
                
                results[nOrders+ii] += (double)(toc - tic) / CLOCKS_PER_SEC;

                qmarray_free(mat); mat = NULL;
                qmarray_free(Q); Q = NULL;
                free(R); R = NULL;
            }   
            results[nOrders+ii] /= nrepeats;
            printf("On order %zu, total time (%G)\n",maxorder,results[nOrders+ii]);
        }
        darray_save(nOrders,2,results,"time_vs_order.dat",1);
        free(results); results = NULL;
    }
    else if (strcmp(argv[1],"1") == 0){
        size_t nr1s = 40;
        double * results = calloc_double(nr1s*2);
        size_t ii,jj;
        for (ii = 0; ii < nr1s; ii++){

            r1 = 1 + 2*ii;
            results[ii] = r1;
            
            for (jj = 0; jj < nrepeats; jj++){
                struct Qmarray * mat = qmarray_poly_randu(LEGENDRE,r1,r2,maxorder,lb,ub);
                double * R = calloc_double(r2 * r2);

                // method 1
                clock_t tic = clock();
                struct Qmarray * Q = qmarray_householder_simple("QR",mat,R,qmopts);
                clock_t toc = clock();
                
                results[nr1s+ii] += (double)(toc - tic) / CLOCKS_PER_SEC;

                qmarray_free(mat); mat = NULL;
                qmarray_free(Q); Q = NULL;
                free(R); R = NULL;
            }   
            results[nr1s+ii] /= nrepeats;
            printf("On r1=%zu, total time (%G)\n",r1,results[nr1s+ii]);
        }
        darray_save(nr1s,2,results,"time_vs_r1.dat",1);
        free(results); results = NULL;
    }
    else if (strcmp(argv[1],"2") == 0){
        size_t nr2s = 40;
        double * results = calloc_double(nr2s*2);
        size_t ii,jj;
        for (ii = 0; ii < nr2s; ii++){

            r2 = 1 + 2*ii;
            results[ii] = r2;
            
            for (jj = 0; jj < nrepeats; jj++){
                struct Qmarray * mat = qmarray_poly_randu(LEGENDRE,r1,r2,maxorder,lb,ub);
                double * R = calloc_double(r2 * r2);

                // method 1
                clock_t tic = clock();
                struct Qmarray * Q = qmarray_householder_simple("QR",mat,R,qmopts);
                clock_t toc = clock();
                
                results[nr2s+ii] += (double)(toc - tic) / CLOCKS_PER_SEC;

                qmarray_free(mat); mat = NULL;
                qmarray_free(Q); Q = NULL;
                free(R); R = NULL;
            }   
            results[nr2s+ii] /= nrepeats;
            printf("On r2=%zu, total time (%G)\n",r2,results[nr2s+ii]);
        }
        darray_save(nr2s,2,results,"time_vs_r2.dat",1);
        free(results); results = NULL;
    }
    one_approx_opts_free_deep(&qmopts);
    return 0;
}
