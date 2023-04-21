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
 

static const char * program_name;

void print_usage2 (FILE * stream, int exit_code)
{
    fprintf (stream, "Usage: %s options [type side]\n", program_name);
    fprintf(stream,
        " -h --help                 Display this usage information.\n"
        " -o --output    filename   Write output to file.\n"
        " -b --benchmark flag       Benchmark entire operation: (default: flag = 0)\n"
        " -t --type      type       Type of benchmark to run\n"
        "                           0: Range over dimensions. Output cols (dim,time,maxrank)\n"
        "                           1: Range over rank. Output cols (dim,time_fast,time_dir,maxrank,round_rank)\n"
        "                           2: Range over polynomial order of a and f. Output cols (order,time,maxrank)\n"
        " -v --verbose              Print verbose messages\n");

    exit(exit_code);
}

int main(int argc, char * argv[])
{
    srand(time(NULL));
    
    int next_option;
    const char* const short_options = "ho:b:t:v";
    const struct option long_options[] = {
        {"help",      0, NULL, 'h'},
        {"output",    1, NULL, 'o'},
        {"benchmark", 1, NULL, 'b'},
        {"type",      1, NULL, 't'},
        {"verbose",   0, NULL, 'v'},
        {NULL,        0, NULL, 0}
    };
    
    const char * output_filename = NULL;
    int benchmark = 0;
    int verbose = 0;
    int type = 0;    
    program_name = argv[0];

    do {
        next_option = getopt_long (argc, argv, short_options, 
                                   long_options, NULL);
        switch (next_option)
        {
            case 'h': 
                print_usage2(stdout, 0);
                break;
            case 'o':
                output_filename = optarg;
                break;
            case 'b':
                if (*optarg == '1'){
                    benchmark = 1;
                }
                break;
            case 't':
                if (*optarg == '0'){
                    type = 0;
                }
                else if (*optarg == '1'){
                    type = 1;
                }
                else if (*optarg == '2'){
                    type = 2;
                }
                else{
                    printf("Unknown type %c\n",*optarg);
                    exit(1);
                }
                break;
            case 'v':
                verbose = 1;
                break;
            case '?':  // invalid option
                print_usage2 (stderr,1);
                break;
            case -1: //done with options
                break;
            default:
                abort();
        }
    } while (next_option != -1);
    assert (output_filename == NULL);

    
    if (benchmark == 0){
        printf("Running Benchmark for Full DMRG \n");
        if (type == 0){
            printf("\t Determining Scaling with Dimension\n");
        }
        else if (type == 1){
            printf("\t Determining Scaling with Rank of conductivity field 'a' \n");
        }
        else if (type == 2){
            printf("\t Determining Scaling with Polynomial Order \n");
        }
    }
    else{
        printf("Running Benchmark for a Core DMRG \n");
    }

//    enum function_class fc = POLYNOMIAL;
    enum poly_type ptype = LEGENDRE;
    struct OpeOpts * opts = ope_opts_alloc(ptype);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    if (benchmark == 0)
    {
        double delta = 1e-8;
        size_t max_sweeps = 5;
        double epsilon = 1e-10;
        if (type == 0){
            size_t nrepeat = 10;

            size_t ndims = 30;
            size_t dimstep = 5;
            size_t dimstart = 2;
            size_t arank = 2;
            size_t frank = 2;
            size_t maxorder = 5;

            double * results = calloc_double(ndims*3);
            
            double time=0.0;
            size_t dim = dimstart;
            size_t ii,jj;
            for (ii = 0; ii < ndims; ii++){
                struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
                multi_approx_opts_set_all_same(fopts,qmopts);
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
                time = 0.0;
                for (jj = 0; jj < nrepeat; jj++){
                    struct FunctionTrain * a = NULL; 
                    a = function_train_poly_randu(ptype,bds,aranks,maxorder);
                    struct FunctionTrain * f = NULL;
                    f = function_train_poly_randu(ptype,bds,franks,maxorder);
                    
                    clock_t tic = clock();
                    struct FunctionTrain * sol = dmrg_diffusion(a,f,
                                                                delta/(double) dim,max_sweeps,
                                                                epsilon,0,fopts);
                    clock_t toc = clock();
                    time += (double)(toc - tic) / CLOCKS_PER_SEC;
                    
                    if (jj == 0){
                        maxrank = function_train_get_maxrank(sol);
                        if (verbose > 0){
                            struct FunctionTrain * exact = exact_diffusion(a,f,fopts);
                            double diff = function_train_relnorm2diff(sol,exact);
                            printf("diff = %G\n",diff*diff);
                            function_train_free(exact); exact = NULL;
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
                multi_approx_opts_free(fopts);
                results[ii] = dim;
                results[ndims+ii] = time;
                results[2*ndims+ii] = maxrank;

                dim = dim + dimstep;
            }

            darray_save(ndims,3,results,"time_vs_dim.dat",1);
            free(results); results = NULL;
        }
        if (type == 1){

            size_t nrepeat = 20;
            size_t nranks = 30; 
            size_t rankstep = 2;
            size_t rankstart = 2;
            size_t dim = 5;
            size_t frank = 2;
            size_t maxorder = 3;
            epsilon= 1e-5;
            delta = 1e-8;

            double * results = calloc_double(nranks*4);
            struct BoundingBox * bds = bounding_box_init(dim,-1.0,1.0);
            struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
            multi_approx_opts_set_all_same(fopts,qmopts);            
            double time=0.0;
            double time2 = 0.0;
            size_t arank = rankstart;
            size_t ii,jj;
            for (ii = 0; ii < nranks; ii++){
                //frank = arank;
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
                double aroundrank = 1.0;
                time = 0.0;
                time2 = 0.0;
                for (jj = 0; jj < nrepeat; jj++){
                    struct FunctionTrain * a = NULL;
                    a = function_train_poly_randu(ptype,bds,aranks,maxorder);
                    struct FunctionTrain * f = NULL;
                    f = function_train_poly_randu(ptype,bds,franks,maxorder);

                    clock_t tic2 = clock();
                    struct FunctionTrain * exact = exact_diffusion(a,f,fopts);
                    struct FunctionTrain * rounded = function_train_round(exact,epsilon,fopts);
                    clock_t toc2 = clock();
                    time2 += (double)(toc2 - tic2) / CLOCKS_PER_SEC;

                    double normval = function_train_norm2(rounded);
                    clock_t tic = clock();
                    struct FunctionTrain * sol = dmrg_diffusion(a,f,
                                                                delta/(double) dim,max_sweeps,
                                                                epsilon*normval,0,fopts);
                    clock_t toc = clock();
                    time += (double)(toc - tic) / CLOCKS_PER_SEC;
                    
                    
                    if ((verbose > 0) && jj == 0){
                        maxrank = function_train_get_maxrank(sol);
                        iprint_sz(dim+1,function_train_get_ranks(sol));
                        struct FunctionTrain * at = function_train_round(a,1e-17,fopts);
                        aroundrank = function_train_get_avgrank(at);
                        function_train_free(at); at = NULL;
                            
                        printf("exact ranks = ");
                        iprint_sz(dim+1,function_train_get_ranks(exact));
                        printf("rounded ranks = ");
                        iprint_sz(dim+1,function_train_get_ranks(rounded));
                        printf("ALS ranks = ");
                        iprint_sz(dim+1,function_train_get_ranks(sol));
                        /* double diff = function_train_relnorm2diff(sol,exact); */
                        /* printf("diff = %G\n",diff*diff); */
                        
                    }
                    /* function_train_free(rounded); rounded = NULL; */
                    /* function_train_free(exact); exact = NULL; */
                    function_train_free(a); a = NULL;
                    function_train_free(f); f = NULL;
                    function_train_free(sol); sol = NULL;
                }
                time /= (double) nrepeat;
                time2 /= (double) nrepeat;
                printf("Average time (ALS/FULL)rank of a %zu is (%G/%G). Maxrank is %zu. Rounded ranks of a are %G\n",arank,time,time2,maxrank,aroundrank);
                free(aranks); aranks = NULL;
                free(franks); franks = NULL;
                
                results[ii] = arank;
                results[nranks+ii] = time;
                results[2*nranks+ii] = time2;
                results[3*nranks+ii] = maxrank;

                arank = arank + rankstep;
            }
            multi_approx_opts_free(fopts);
            bounding_box_free(bds); bds = NULL;
            darray_save(nranks,4,results,"time_vs_arank.dat",1);
            free(results); results = NULL;
        }
        if (type == 2){

            size_t nrepeat = 10;
            size_t nPs = 40; 
            size_t pstep = 2;
            size_t pstart = 2;
            size_t dim = 5;
            size_t frank = 2;
            size_t arank = 2;

            double * results = calloc_double(nPs*3);
            struct BoundingBox * bds = bounding_box_init(dim,-1.0,1.0);
            struct MultiApproxOpts * fopts = multi_approx_opts_alloc(dim);
            multi_approx_opts_set_all_same(fopts,qmopts);            
            
            double time = 0.0;
            size_t ii,jj;
            size_t maxorder = pstart;
            for (ii = 0; ii < nPs; ii++){
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
                time = 0.0;
                for (jj = 0; jj < nrepeat; jj++){
                    struct FunctionTrain * a = NULL;
                    a = function_train_poly_randu(ptype, bds,aranks,maxorder);
                    struct FunctionTrain * f = NULL;
                    f = function_train_poly_randu(ptype,bds,franks,maxorder);
                    
                    clock_t tic = clock();
                    struct FunctionTrain * sol = dmrg_diffusion(a,f,
                                                                delta/(double) dim,max_sweeps,
                                                                epsilon,0,fopts);
                    clock_t toc = clock();
                    time += (double)(toc - tic) / CLOCKS_PER_SEC;
                    
                    if (jj == 0){
                        maxrank = function_train_get_maxrank(sol);
                        if (verbose > 0){
                            struct FunctionTrain * exact = exact_diffusion(a,f,fopts);
                            double diff = function_train_relnorm2diff(sol,exact);
                            printf("diff = %G\n",diff*diff);
                            function_train_free(exact); exact = NULL;
                        }
                    }

                    function_train_free(a); a = NULL;
                    function_train_free(f); f = NULL;
                    function_train_free(sol); sol = NULL;
                }
                time /= (double) nrepeat;
                printf("Average time for P=%zu is (%G). Maxrank is %zu. \n",maxorder,time,maxrank);
                free(aranks); aranks = NULL;
                free(franks); franks = NULL;
                
                results[ii] = maxorder;
                results[nPs+ii] = time;
                results[2*nPs+ii] = maxrank;

                maxorder = maxorder + pstep;
            }
            multi_approx_opts_free(fopts); fopts = NULL;
            bounding_box_free(bds); bds = NULL;
            darray_save(nPs,3,results,"time_vs_order.dat",1);
            free(results); results = NULL;
        }
    }
    one_approx_opts_free_deep(&qmopts);
    return 0;
}
