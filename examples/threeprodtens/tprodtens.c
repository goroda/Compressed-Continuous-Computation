#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "quadrature.h"

int main()
{
    // generate a sequence of quadrature tables
    
    size_t maxorder = 50;
    size_t ii,jj,kk,ll;
    
    double * coeffs = calloc_double(125000);
    struct OrthPoly * leg = init_leg_poly();

    size_t nquad = 3*maxorder+1;
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

    for (ii = 0; ii < maxorder; ii++){
        for (jj = 0; jj < maxorder; jj++){
            for (kk = 0; kk < maxorder; kk++){
                printf("ii,jj,kk = %zu,%zu,%zu\n",ii,jj,kk);
                coeffs[ii+jj*maxorder+kk*maxorder*maxorder] = 0.0;
                for (ll = 0; ll < nquad; ll++){
                    double evali = evals[ll + ii*nquad];
                    double evalj = evals[ll + jj*nquad];
                    double evalk = evals[ll + kk*nquad];
                    coeffs[ii+jj*maxorder+kk*maxorder*maxorder] += wt[ll]*evali*evalj*evalk;
                }
                coeffs[ii+jj*maxorder+kk*maxorder*maxorder] /= leg->norm(kk);
            }
        }
    }
    printf("done and printing\n");
    int success = darray_save(maxorder*maxorder*maxorder,1,coeffs,"legpolytens.dat",1);
    assert ( success == 1);
    free_orth_poly(leg);
    free(pt);
    free(wt);
    free(coeffs);

    return 0;
}
