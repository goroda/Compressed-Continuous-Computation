#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "quadrature.h"

inline static double 
eval_orth_poly_wp(const struct OrthPoly * rec, double p2, double p1, 
                  size_t n, double x)
{
    return (rec->an(n) * x + rec->bn(n)) * p1 + rec->cn(n) * p2; 
}

int main()
{
    // generate the three term product
    
    size_t maxorder = 200;
    size_t ii,jj,kk,ll;
    
    double * coeffs = calloc_double((maxorder/2)*(maxorder/2)*maxorder);
    struct OrthPoly * leg = init_leg_poly();

    size_t nquad = 3*maxorder+2;
    double * pt = calloc_double(nquad);
    double * wt = calloc_double(nquad);
    gauss_legendre(nquad,pt,wt);

    double * evals = calloc_double(nquad * (maxorder+1));
        
    for (kk = 0; kk < maxorder; kk++){   
        for (ll = 0; ll < nquad; ll++){
            if (kk == 0){
                evals[ll + kk*nquad] = orth_poly_eval(leg,0,pt[ll]);
            }
            else if (kk == 1){
                evals[ll + kk*nquad] = orth_poly_eval(leg,1,pt[ll]);
            }
            else {
                evals[ll + kk*nquad] = eval_orth_poly_wp(leg,evals[ll + (kk-2)*nquad],
                                            evals[ll + (kk-1)*nquad],kk, pt[ll]);
            }
        }
    }
    
    size_t total = 0;
    size_t nonzero = 0;
    double zerothresh = 1e-14;
    for (ii = 0; ii < maxorder/2; ii++){
        for (jj = 0; jj < maxorder/2; jj++){
            for (kk = 0; kk < maxorder; kk++){
                printf("ii,jj,kk = %zu,%zu,%zu\n",ii,jj,kk);
                coeffs[ii+jj*maxorder/2+kk*maxorder*maxorder/4] = 0.0;
                for (ll = 0; ll < nquad; ll++){
                    double evali = evals[ll + ii*nquad];
                    double evalj = evals[ll + jj*nquad];
                    double evalk = evals[ll + kk*nquad];
                    coeffs[ii+jj*maxorder/2+kk*maxorder*maxorder/4] += wt[ll]*evali*evalj*evalk;
                }
                coeffs[ii+jj*maxorder/2+kk*maxorder*maxorder/4] /= leg->norm(kk);
                
                if (fabs(coeffs[ii+jj*maxorder/2+kk*maxorder*maxorder/4]) > zerothresh){
                    nonzero++;
                }
                else{
                    coeffs[ii+jj*maxorder/2+kk*maxorder*maxorder/4] = 0.0;
                }
                total += 1;
            }
        }
    }
    printf("done and printing\n");
    printf("density level %zu / %zu = %G\n", nonzero,total,(double) nonzero/ (double)total);
    FILE * fp = fopen("legpolytens.dat","w");
    assert (fp != NULL);

    fprintf(fp,"%s\n\n", "#include <string.h>");
    fprintf(fp,"%s\n", "static const double lpolycoeffs[8000000] = {");
    for (ii = 0; ii < maxorder*maxorder*maxorder/4; ii++){
        fprintf(fp,"%3.15G,\n",coeffs[ii]);
    }
    fprintf(fp,"};\n");
    fclose(fp); 
    /* int success = darray_save(maxorder*maxorder*maxorder/4,1,coeffs,"legpolytens.dat",1); */
    /* assert ( success == 1); */
    free_orth_poly(leg);
    free(pt);
    free(wt);
    free(coeffs);

    return 0;
}
