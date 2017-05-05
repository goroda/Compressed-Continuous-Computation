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


    FILE * fp = fopen("leg_c_seq.dat","w");
    assert (fp != NULL);
    fprintf(fp,"%s\n", "static const long double legcseqnorm[201] = {");

    FILE * fp2 = fopen("leg_a_seq.dat","w");
    assert (fp2 != NULL);
    fprintf(fp2,"%s\n", "static const long double legaseqnorm[201] = {");


    fprintf(fp2,"%3.25Lf,\n", (long double) 0.0);
    
    fprintf(fp,"%3.25Lf,\n", (long double) 0.0);
    fprintf(fp,"%3.25Lf,\n", (long double) 0.0);
    for (size_t ii = 0; ii <= maxorder; ii++){
        long double n = (long double) ii;

        // computation of c
        long double twon = 2.0 * n;
        long double t1 = sqrtl((twon+1)/(twon-3));
        long double t2 = (n-1.0)/n;
        if (ii > 1){
            fprintf(fp,"%3.25Lf,\n",  -t1*t2);
        }

        
        long double a = sqrtl(4 *n * n - 1.0)/n;

        if (ii > 0){
            fprintf(fp2,"%3.25Lf,\n",  a);
        }
    }
    
    fprintf(fp,"};\n");
    fclose(fp); 

    fprintf(fp2,"};\n");
    fclose(fp2); 


    return 0;
}
