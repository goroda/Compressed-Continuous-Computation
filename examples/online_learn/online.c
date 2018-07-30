#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/types.h>
#include <getopt.h>
#include <unistd.h>

#include "c3.h"

static char * program_name;

void print_code_usage (FILE *, int) __attribute__ ((noreturn));
void print_code_usage (FILE * stream, int exit_code)
{
    fprintf(stream, "Two examples of functional regression \n\n");
    fprintf(stream, "Usage: %s options \n", program_name);
    fprintf(stream,
            " -h --help       Display this usage information.\n"
            " -l --learn_rate Learning Rate.\n"
            " -v --verbose    Output words (default 0)\n"
            " \n\n"
            /* " Outputs four files\n" */
            /* " training_funcs.dat  -- training samples" */
            /* " testing_funcs_n{number}.dat -- evaluations of true model\n" */
            /* " testing_funcs_ft_n{number}.dat -- evaluations of reg model\n" */
            /* " testing_funcs_diff_n{number}.dat -- difference b/w models\n" */
        );
    exit (exit_code);
}


int main(int argc, char * argv[])
{
    int next_option;
    const char * const short_options = "hl:v:";
    const struct option long_options[] = {
        { "help"      , 0, NULL, 'h' },
        { "learn_rate", 1, NULL, 'l' },
        { "verbose"   , 1, NULL, 'v' },
        { NULL        ,  0, NULL, 0   }
    };

    program_name = argv[0];
    size_t verbose = 0;
    // learn_rate 9.99e-1 good for adadelta
    double learn_rate = 9.99e-1; 
    do {
        next_option = getopt_long (argc, argv, short_options, long_options, NULL);
        switch (next_option)
        {
            case 'h': 
                print_code_usage(stdout, 0);
            case 'v':
                verbose = strtoul(optarg,NULL,10);
                break;
            case 'l':
                learn_rate = strtod(optarg,NULL);
                break;
            case '?': // The user specified an invalid option 
                print_code_usage (stderr, 1);
            case -1: // Done with options. 
                break;
            default: // Something unexpected
                abort();
        }

    } while (next_option != -1);


    char filename[256] = "lorenz_init_010_dt_001.dat";
    size_t dim = 3;
    
    struct DataFrame * data = data_frame_load(filename);
    size_t ndata = data_frame_get_nrows(data);

    
    size_t fixed_rank = 6;
    size_t * ranks = calloc_size_t(dim+1);
    ranks[0] = 1;
    ranks[dim] = 1;
    for (size_t ii = 1; ii < dim; ii++){
        ranks[ii] = fixed_rank;
    }

    
    
    double lb = -30;
    double ub = 50;

    /* size_t adapt_kernel = 1; */
    /* size_t N = 10; */
    /* double width = pow(N,-0.2)/sqrt(12.0); */
    /* width *= 0.5; */
    /* double * x1 = linspace(lb, ub, N); */
    
    /* struct KernelApproxOpts * kopts = kernel_approx_opts_gauss(N, x1, 10.0, width); */
    /* kernel_approx_opts_set_center_adapt(kopts, adapt_kernel); */
    /* struct OneApproxOpts * ko = one_approx_opts_alloc(KERNEL,kopts); */

    size_t N = 6;
    size_t adapt_kernel = 0;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_nparams(opts,N);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    struct OneApproxOpts * ko = one_approx_opts_alloc(POLYNOMIAL, opts);

    
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    size_t nparam = 0;
    for (size_t ii = 0; ii < dim; ii++){
        multi_approx_opts_set_dim(fapp,ii,ko);
        if (adapt_kernel == 0){
            nparam += N * ranks[ii]*ranks[ii+1];
        }
        else{
            nparam += 2 * N * ranks[ii]*ranks[ii+1];
        }
    }

    printf("num params = %zu\n", nparam);


    struct RegressOpts* ropts = regress_opts_alloc(dim);
    struct FTparam* ftp[3];
    struct StochasticUpdater * su[3];

    double * grad[3];     
    double * param_start[3]; 
    for (size_t kk = 0; kk < dim; kk++){
        grad[kk] = calloc_double(nparam);
        param_start[kk] = calloc_double(nparam);
        for (size_t ii = 0; ii < nparam; ii++){
            param_start[kk][ii] = randu()*2.0-1.0;
        }
        
        ftp[kk]= ft_param_alloc(dim, fapp, param_start[kk], ranks);
        su[kk] = stochastic_updater_alloc(SU_ADADELTA);
        /* su[kk] = stochastic_updater_alloc(SU_ADAGRAD); */
        /* su[kk] = stochastic_updater_alloc(SU_MOMENTUM); */

        int res = setup_least_squares_online_learning(su[kk], learn_rate, ftp[kk], ropts);
        assert (res == 0);
    }
    
    struct Data * curr_pt = data_alloc(1, dim);
    struct Data * prev_pt = data_alloc(1, dim);
    struct Data * curr_t = data_alloc(1, 1);
    struct Data * prev_t = data_alloc(1, 1);
    size_t start_col = 1;
    size_t ind_sub[3] = {0, 1, 2};

    char fileout[256] = "out.dat";
    FILE * fp = fopen(fileout, "w");
    assert (fp != NULL);
    size_t nouter = 1;
    for (size_t jj = 0; jj < nouter; jj++){
        for (size_t ii = 1; ii < ndata; ii++){
            data_frame_get_feature(data, curr_pt, ii, start_col, dim);
            data_frame_get_feature(data, prev_pt, ii-1, start_col, dim);

            const double * curr_x = data_get_subset_ref(curr_pt, 1, ind_sub);
            const double * prev_x = data_get_subset_ref(prev_pt, 1, ind_sub);
        
            data_frame_get_feature(data, curr_t, ii, 0, 1);
            data_frame_get_feature(data, prev_t, ii-1, 0, 1);
        
            double t = data_get_subset_ref(curr_t, 1, 0)[0];
            double pt = data_get_subset_ref(prev_t, 1, 0)[0];
            double dt = t - pt;

            // learn f;
            printf("t = %3.5G ", t);
            printf("dt = %3.5G\n", dt);
            printf("pt = "); dprint(dim, prev_x);

            fprintf(fp, "%3.5G", t);
            for (size_t kk = 0; kk < dim; kk++){
                data_set_y(prev_pt, curr_x+kk); // kth output

                //prediction before update
                double eval = function_train_eval(ftp[kk]->ft, prev_x); 
                double obj_eval = stochastic_update_step(su[kk], param_start[kk], grad[kk], prev_pt);
                ft_param_update_params(ftp[kk], param_start[kk]);

                // prediction after update
                double eval_post = function_train_eval(ftp[kk]->ft, prev_x); 

                printf("\t (%3.5G, %3.5G, %3.5G, %3.5G %3.5G) \n", curr_x[kk], eval, eval_post, obj_eval,
                       cblas_ddot(nparam, grad[kk], 1, grad[kk], 1));

                fprintf(fp, " %3.5G %3.5G", curr_x[kk], eval);

            }
            fprintf(fp, "\n");

        }
    }
    fclose(fp);

    free(ranks); ranks = NULL;
    
    data_free(curr_pt); curr_pt = NULL;
    data_free(prev_pt); prev_pt = NULL;

    data_free(curr_t); curr_t = NULL;
    data_free(prev_t); prev_t = NULL;

    data_frame_free(data);              data        = NULL;
    one_approx_opts_free_deep(&ko);     ko          = NULL;
    multi_approx_opts_free(fapp);       fapp        = NULL;
    regress_opts_free(ropts);           ropts       = NULL;

    for (size_t kk = 0; kk < dim; kk++){
        free(param_start[kk]);                  param_start[kk] = NULL;
        free(grad[kk]);                         grad[kk]        = NULL;
        ft_param_free(ftp[kk]);                 ftp[kk]         = NULL;
        stochastic_updater_free(su[kk]);        su[kk]          = NULL;
    }

    return 0;
}
