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
       printf("Correct function call = ./fmkronbench type\n");
       printf("Options for type name include: \n");
       printf("\t \"0\": time vs. order, \"1\": time vs k, \"2\": time vs r \n");
       printf("\t Recall that we are performing Ckron(A,B) where \n");
       printf("\t\t C is k x r^2 matrix and \n");
       printf("\t\t A,B are r x r matrix valued functions \n");
       return 0;
    }

    double lb = -1.0;
    double ub = 1.0;
    size_t r11 = 5;
    size_t r12 = 5;
    size_t r21 = 5;
    size_t r22 = 5;
    size_t k = 5;
    size_t maxorder = 5;
    double diff; 
    size_t nrepeats = 50;
    
    if (strcmp(argv[1],"0") == 0){
        size_t nOrders = 40;
        double * results = calloc_double(nOrders*3);
        size_t ii,jj;
        for (ii = 0; ii < nOrders; ii++){

            maxorder = 1 + ii;
            results[ii] = maxorder;
            
            for (jj = 0; jj < nrepeats; jj++){
                double * mat = drandu(k*r11*r21);
                struct Qmarray * mat1 = 
                    qmarray_poly_randu(LEGENDRE,r11,r12,
                                       maxorder,lb,ub);
                struct Qmarray * mat2 = 
                    qmarray_poly_randu(LEGENDRE,r21,r22,
                                       maxorder,lb,ub);
            

                // method 1
                clock_t tic = clock();
                struct Qmarray * mat3 = qmarray_kron(mat1,mat2);
                struct Qmarray * shouldbe = mqma(mat,mat3,k);
                clock_t toc = clock();
                
                results[nOrders+ii] += (double)(toc - tic) / CLOCKS_PER_SEC;

                // method 2
                tic = clock();
                struct Qmarray * is = qmarray_mat_kron(k,mat,mat1,mat2);
                toc = clock();
                results[2*nOrders+ii] += (double)(toc - tic) / CLOCKS_PER_SEC;

                diff = qmarray_norm2diff(shouldbe,is);
                assert ( (diff < 1e-10) == 1);

                free(mat); mat = NULL;
                qmarray_free(mat1); mat1 = NULL;
                qmarray_free(mat2); mat2 = NULL;
                qmarray_free(mat3); mat3 = NULL;
                qmarray_free(shouldbe); shouldbe = NULL;
                qmarray_free(is); is = NULL;
            }   
            results[nOrders+ii] /= nrepeats;
            results[2*nOrders+ii] /= nrepeats;
            printf("On order %zu, total times are (%G,%G)\n",maxorder,results[nOrders+ii],results[2*nOrders+ii]);
        }
        darray_save(nOrders,3,results,"time_vs_order.dat",1);
        
        free(results); results = NULL;
    }
    else if (strcmp(argv[1],"1") == 0){
        size_t nks = 20;
        double * results = calloc_double(nks*3);
        size_t ii,jj;
        for (ii = 0; ii < nks; ii++){

            k = 1 + 10*ii;
            results[ii] = (double) k;
            
            for (jj = 0; jj < nrepeats; jj++){
                double * mat = drandu(k*r11*r21);
                struct Qmarray * mat1 = 
                    qmarray_poly_randu(LEGENDRE,r11,r12,
                                       maxorder,lb,ub);
                struct Qmarray * mat2 = 
                    qmarray_poly_randu(LEGENDRE,r21,r22,
                                       maxorder,lb,ub);
            

                // method 1
                clock_t tic = clock();
                struct Qmarray * mat3 = qmarray_kron(mat1,mat2);
                struct Qmarray * shouldbe = mqma(mat,mat3,k);
                clock_t toc = clock();
                
                results[nks+ii] += (double)(toc - tic) / CLOCKS_PER_SEC;

                // method 2
                tic = clock();
                struct Qmarray * is = qmarray_mat_kron(k,mat,mat1,mat2);
                toc = clock();
                results[2*nks+ii] += (double)(toc - tic) / CLOCKS_PER_SEC;

                diff = qmarray_norm2diff(shouldbe,is);
                assert ( (diff < 1e-10) == 1);

                free(mat); mat = NULL;
                qmarray_free(mat1); mat1 = NULL;
                qmarray_free(mat2); mat2 = NULL;
                qmarray_free(mat3); mat3 = NULL;
                qmarray_free(shouldbe); shouldbe = NULL;
                qmarray_free(is); is = NULL;
            }   
            results[nks+ii] /= nrepeats;
            results[2*nks+ii] /= nrepeats;
            printf("On k=%zu, total times are (%G,%G)\n",k,results[nks+ii],results[2*nks+ii]);
        }
        darray_save(nks,3,results,"time_vs_k.dat",1);
        
        free(results); results = NULL;
    }
    else if (strcmp(argv[1],"2") == 0){
        size_t nrs = 20;
        double * results = calloc_double(nrs*3);
        size_t ii,jj;
        for (ii = 0; ii < nrs; ii++){
            
            r11 = 1 + 2*ii;
            r12 = r11;
            r21 = r11;
            r22 = r21;

            results[ii] = (double) r11;
            
            for (jj = 0; jj < nrepeats; jj++){
                double * mat = drandu(k*r11*r21);
                struct Qmarray * mat1 = qmarray_poly_randu(LEGENDRE,r11,r12,maxorder,lb,ub);
                struct Qmarray * mat2 = qmarray_poly_randu(LEGENDRE,r21,r22,maxorder,lb,ub);
            

                // method 1
                clock_t tic = clock();
                struct Qmarray * mat3 = qmarray_kron(mat1,mat2);
                struct Qmarray * shouldbe = mqma(mat,mat3,k);
                clock_t toc = clock();
                
                results[nrs+ii] += (double)(toc - tic) / CLOCKS_PER_SEC;

                // method 2
                tic = clock();
                struct Qmarray * is = qmarray_mat_kron(k,mat,mat1,mat2);
                toc = clock();
                results[2*nrs+ii] += (double)(toc - tic) / CLOCKS_PER_SEC;

                diff = qmarray_norm2diff(shouldbe,is);
                assert ( (diff < 1e-10) == 1);

                free(mat); mat = NULL;
                qmarray_free(mat1); mat1 = NULL;
                qmarray_free(mat2); mat2 = NULL;
                qmarray_free(mat3); mat3 = NULL;
                qmarray_free(shouldbe); shouldbe = NULL;
                qmarray_free(is); is = NULL;
            }   
            results[nrs+ii] /= nrepeats;
            results[2*nrs+ii] /= nrepeats;
            printf("On r=%zu, total times are (%G,%G)\n",r11,results[nrs+ii],results[2*nrs+ii]);
        }
        darray_save(nrs,3,results,"time_vs_r.dat",1);
        
        free(results); results = NULL;
    }

    return 0;
}
