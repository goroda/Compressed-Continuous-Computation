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
    
    size_t n = 1;
    size_t maxn = 200;
    size_t ii;

    for (ii = n; ii < maxn; ii++){
        printf("working on rule %zu \n", ii);
        double * pt = calloc_double(ii);
        double * wt = calloc_double(ii);
        gauss_legendre(ii,pt,wt);
        
        double * comb = dconcat_cols(ii,1,1,pt,wt);
        char filename[256];
        sprintf(filename,"quad%zu.dat",ii);
        int success = darray_save(ii,2,comb,filename,1);
        assert(success == 1);

        free(pt); pt = NULL;
        free(wt); wt = NULL;
        free(comb); comb = NULL;
    }

    return 0;
}
