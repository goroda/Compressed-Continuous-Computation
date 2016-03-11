#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"

//int main( int argc, char *argv[])
int main( void )
{

    size_t start = 2;
    size_t nDims = 100;
    size_t ii,jj, dim;
    double lb = -1.0;
    double ub = 1.0;
    enum poly_type ptype = LEGENDRE;
    for (dim = start; dim < (start+nDims); dim+=4){
        struct BoundingBox * bounds = bounding_box_init_std(dim);
        double * quad = calloc_double(dim * dim);
        double * coeff = calloc_double(dim);
        for (ii = 0; ii < dim; ii++){
            coeff[ii] = randu();
            bounds->lb[ii] = lb;
            bounds->ub[ii] = ub;
            for (jj = 0; jj < dim; jj++){
                quad[ii*dim+jj] = randu();
            }
        }
        struct FunctionTrain * f = function_train_quadratic(POLYNOMIAL,&ptype,
                                                            dim,bounds,quad,coeff,NULL);
        struct FunctionTrain * fr = function_train_round(f,1e-8);
        
        size_t maxrank = fr->ranks[1];
        for (ii = 2; ii < dim; ii++){
            if (fr->ranks[ii] > maxrank){
                maxrank = fr->ranks[ii];
            }
        }

        printf("Dimension = %zu, max rank = %zu\n",dim,maxrank);
        bounding_box_free(bounds);
        free(quad);
        free(coeff);
        function_train_free(f);
        function_train_free(fr);

    }
}
