#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <getopt.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"
 

const char * program_name;

void print_usage (FILE * stream, int exit_code)
{
    fprintf (stream, "Usage: %s options [type side]\n", program_name);
    fprintf(stream,
        " -h --help              Display this usage information.\n"
        " -o --output filename   Write output to file.\n"
        " -b --benchmark flag    Benchmark entire operation (default: flag=0) or core (flag=1)\n"   
        " -v --verbose           Print verbose messages\n");

    exit (exit_code);
}

int main(int argc, char * argv[])
{
    srand(time(NULL));
    
    int next_option;
    const char* const short_options = "ho:b:v";
    const struct option long_options[] = {
        {"help",      0, NULL, 'h'},
        {"output",    1, NULL, 'o'},
        {"benchmark", 1, NULL, 'b'},
        {"verbose",   0, NULL, 'v'},
        {NULL,        0, NULL, 0}
    };
    
    const char * output_filename = NULL;
    int benchmark = 0;
    int verbose = 0;
    program_name = argv[0];
    
    do {
        next_option = getopt_long (argc, argv, short_options, 
                                   long_options, NULL);
        switch (next_option)
        {
            case 'h': 
                print_usage(stdout, 0);
            case 'o':
                output_filename = optarg;
                break;
            case 'b':
                if (*optarg == '1'){
                    benchmark = 1;
                }
            case 'v':
                verbose = 1;
                break;
            case '?':  // invalid option
                print_usage (stderr,1);
            case -1: //done with options
                break;
            default:
                abort();
        }
    } while (next_option != -1);

    int type = 0;

    printf("will I benchmark? %d\n",benchmark);

    if (benchmark == 0)
    {
        size_t nrepeat = 10;
        double delta = 1e-8;
        size_t max_sweeps = 5;
        double epsilon = 1e-10;
        if (type == 0){

            size_t ndims = 60;
            size_t dimstep = 10;
            size_t dimstart = 2;
            size_t arank = 2;
            size_t frank = 2;
            size_t maxorder = 5;

            double * results = calloc_double(ndims*3);
            
            double time=0.0;
            size_t dim = dimstart;
            size_t ii,jj;
            for (ii = 0; ii < ndims; ii++){
                struct BoundingBox * bds = bounding_box_init(dim,-1.0,1.0);
                size_t * aranks = calloc_size_t(dim+1);
                size_t * franks = calloc_size_t(dim+1);
                aranks[0] = 1;
                franks[0] = 1;
                for (jj = 1; jj < dim; jj++){
                    aranks[jj] = arank;
                    franks[jj] = frank;
                }
                aranks[dim] = 1;
                franks[dim] = 1;
                
                size_t maxrank = 1;
                for (jj = 0; jj < nrepeat; jj++){
                    struct FunctionTrain * a = function_train_poly_randu(bds,aranks,maxorder);
                    struct FunctionTrain * f = function_train_poly_randu(bds,franks,maxorder);
                    
                    clock_t tic = clock();
                    struct FunctionTrain * sol = dmrg_diffusion(a,f,
                                            delta,max_sweeps,epsilon,verbose);
                    clock_t toc = clock();
                    time += (double)(toc - tic) / CLOCKS_PER_SEC;
                    
                    if (jj == 0){
                        size_t kk;
                        for (kk = 1; kk < dim; kk++){
                            if (sol->ranks[kk] > maxrank){
                                maxrank = sol->ranks[kk];
                            }
                        }
                    }

                    function_train_free(a); a = NULL;
                    function_train_free(f); f = NULL;
                    function_train_free(sol); sol = NULL;
                }
                time /= (double) nrepeat;
                printf("Average time for dim %zu is (%G)\n", dim,time);
                free(aranks); aranks = NULL;
                free(franks); franks = NULL;
                bounding_box_free(bds); bds = NULL;
                
                results[ii] = dim;
                results[ndims+ii] = time;
                results[2*ndims+ii] = maxrank;

                dim = dim + dimstep;
            }

            darray_save(ndims,3,results,"time_vs_dim.dat",1);
            free(results); results = NULL;
        }
    }
    

    return 0;
}
