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

/* #define dx 2 */
/* #define dx 100 */
static size_t dx;
#define dy 16

static char * program_name;

void print_code_usage (FILE *, int) __attribute__ ((noreturn));
void print_code_usage (FILE * stream, int exit_code)
{
    fprintf(stream, "Two examples of functional regression \n\n");
    fprintf(stream, "Usage: %s options \n", program_name);
    fprintf(stream,
            " -h --help      Display this usage information.\n"
            " -f --function  Which function to evaluate. \n"
            "                0: a_1x^2 + a_2x^2 (default)\n"
            "                1: more complicated, check code\n"
            " -n --number    Number of training samples (default 100).\n"
            " -p --polyorder Polynomial order for parameters (default 4).\n"
            " -b --basis     Basis for spatial variable\n"
            "                0: piecewise linear continuous (default)\n"
            "               >0: radial basis function kernels\n"
            " -s --rankspace Rank between spatial and paramter variables (default 4)\n"
            " -r --rankparam Rank between parameters (default 2)\n"
            " -v --verbose   Output words (default 0).\n"
            " \n\n"
            " Outputs four files\n"
            " training_funcs.dat  -- training samples"
            " testing_funcs_n{number}.dat -- evaluations of true model\n"
            " testing_funcs_ft_n{number}.dat -- evaluations of reg model\n"
            " testing_funcs_diff_n{number}.dat -- difference b/w models\n"
        );
    exit (exit_code);
}

struct Problem
{
    size_t ninput;
    size_t noutput;
    double * x;
};
typedef struct Problem prob_t;


//quadratic with parameterized coefficients
static void quadratic(prob_t * prob, double * input, double * output)
{
    for (size_t ii = 0; ii < prob->noutput; ii++){
        output[ii] = prob->x[ii]*prob->x[ii] * input[0] + prob->x[ii] * input[1]; 
    }
}

static void other(prob_t * prob, double * input, double * output)
{
    double sum = 0.0;
    for (size_t ii = 0; ii < dx/2; ii++){
        sum += (input[ii]+1)/2.0;
    }
    double coeff = 0.0;
    for (size_t ii = 0; ii < dx/2; ii++){
        coeff += (input[ii+dx/2]+1)/2.0;
    }
    for (size_t ii = 0; ii < prob->noutput; ii++){
        output[ii] = sin(sum * prob->x[ii])  + 0.1 * coeff * exp(-pow(prob->x[ii]-0.5,2)/0.01);
    }
}

static double * generate_inputs(size_t n)
{
    double * x = calloc_double(dx * n);
        
    for (size_t ii = 0; ii < n; ii++){
        x[0 + ii*dx] = randu()*2.0-1.0;
        x[1 + ii*dx] = randu()*2.0-1.0;
    }

    return x;
}

static double * generate_outputs(prob_t * prob, size_t n, double * inputs)
{
    double * y = calloc_double(n * dy);
    if (dx == 2){
        for (size_t ii = 0; ii < n; ii++){
            quadratic(prob,inputs + ii*dx,y+ii*dy);
        }
    }
    else {
        for (size_t ii = 0; ii < n; ii++){
            other(prob,inputs + ii*dx,y+ii*dy);
        }
    }

    return y;
}

static size_t create_unified_data(size_t ndata, double * inputs,
                                  double * xspace,
                                  double * outputs, double * x, double * y)
{
    size_t ondata=0;
    for (size_t ii = 0; ii < ndata; ii++){

        for (size_t jj = 0; jj < dy; jj++){
            y[ondata] = outputs[jj + ii * dy];
            for (size_t kk = 0; kk < dx; kk++){
                x[kk + ondata * (dx + 1)] = inputs[kk + ii * dx];
            }
            x[dx + ondata * (dx+1)] = xspace[jj];
            ondata++;
        }
    }

    return ondata;
}

static void save_array_with_x(size_t nrows, size_t ncols, double * x,
                       double * array, char * filename)
{
    double * temp = calloc_double(nrows * (ncols+1));
    memmove(temp,x,nrows * sizeof(double));
    memmove(temp+nrows,array,nrows*ncols*sizeof(double));

    darray_save(nrows,ncols+1,temp,filename,1);
    free(temp);
}

int main(int argc, char * argv[])
{
    int next_option;
    const char * const short_options = "hf:n:p:b:s:r:v:";
    const struct option long_options[] = {
        { "help"     , 0, NULL, 'h' },
        { "function" , 1, NULL, 'f' },
        { "number"   , 1, NULL, 'n' },
        { "polyorder", 1, NULL, 'p' },
        { "basis"    , 1, NULL, 'b' },
        { "rankspace", 1, NULL, 's' },
        { "rankparam", 1, NULL, 'r' },
        { "verbose"  , 1, NULL, 'v' },
        { NULL       ,  0, NULL, 0   }
    };

    size_t npoly = 4;
    size_t ndata = 100;
    int verbose = 0;
    size_t function = 0;
    size_t basis = 0;
    size_t rank_space = 4;
    size_t rank_param = 2;
    program_name = argv[0];
    do {
        next_option = getopt_long (argc, argv, short_options, long_options, NULL);
        switch (next_option)
        {
            case 'h': 
                print_code_usage(stdout, 0);
            case 'f':
                function = strtoul(optarg,NULL,10);
                break;
            case 'n':
                ndata = strtoul(optarg,NULL,10);
                break;
            case 'p':
                npoly = strtoul(optarg,NULL,10);
                break;
            case 'b':
                basis = strtoul(optarg,NULL,10);
                break;
            case 's':
                rank_space = strtoul(optarg,NULL,10);
                break;
            case 'r':
                rank_param = strtoul(optarg,NULL,10);
                break;
            case 'v':
                verbose = strtoul(optarg,NULL,10);
                break;
            case '?': // The user specified an invalid option 
                print_code_usage (stderr, 1);
            case -1: // Done with options. 
                break;
            default: // Something unexpected
                abort();
        }

    } while (next_option != -1);


    if (function == 1){
        dx = 100;
    }
    else{
        dx = 2;
    }
    printf("\n");
    printf("\n");
    printf("\t Functional regression setup\n");
    printf("\n");
    printf("\t Number of parameters:           %zu\n",dx);
    printf("\t Number of spatial measurements: %d\n",dy);
    printf("\t Number of data points:          %zu\n",ndata);
    printf("\t Basis:                          ");
    if (basis == 0){
        printf("piecewise continuous linear elements\n");
    }
    else{
        printf("squared exponential kernels\n");
    }
    printf("\t Rank between parameters:        %zu\n",rank_param);
    printf("\t Rank between param and space:   %zu\n",rank_space);
    printf("\t Parameter polynomial order:     %zu\n",npoly-1);
    printf("\n\n\n\n");
    
    prob_t prob;
    prob.ninput = dx;
    prob.noutput = dy;
    prob.x = linspace(0.0,1.0,dy);

    double * inputs = generate_inputs(ndata);
    double * outputs = generate_outputs(&prob,ndata,inputs);

    save_array_with_x(dy,ndata,prob.x,outputs,"training_funcs.dat");
    
    double * x = calloc_double((dx+1)*dy * ndata);
    double * y = calloc_double(ndata*dy);

    
    /* size_t ntotdata = create_unified_data(ndata,inputs,prob.x,outputs,x,y); */    
    create_unified_data(ndata,inputs,prob.x,outputs,x,y);    

    
    /* dprint2d_col(dy,ndata,y); */
    /* printf("ntotal data = %zu\n",ntotdata); */


    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_nparams(opts,npoly);
    struct OneApproxOpts * polyopts =
        one_approx_opts_alloc(POLYNOMIAL,opts);

    double width = pow(dy,-0.2)/sqrt(12.0);
    width *= 0.5;
    struct KernelApproxOpts * kopts =
        kernel_approx_opts_gauss(prob.noutput,prob.x,1.0,width);
    struct LinElemExpAopts * lopts =
        lin_elem_exp_aopts_alloc(prob.noutput,prob.x);

    struct OneApproxOpts * ko = NULL;
    if (basis == 0){
        ko = one_approx_opts_alloc(LINELM,lopts);
    }
    else{
        /* printf("width=%G\n",width); */
        ko = one_approx_opts_alloc(KERNEL,kopts);
    }

    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dx+1);    
    for (size_t ii = 0; ii < dx; ii++){
        multi_approx_opts_set_dim(fapp,ii,polyopts);
    }
    multi_approx_opts_set_dim(fapp,dx,ko);

    
    size_t * ranks = calloc_size_t(dx+1+1);
    ranks[0] = 1;
    ranks[dx+1] = 1;
    for (size_t ii = 1; ii <= dx; ii++){
        ranks[ii] = rank_param;
    }
    ranks[dx] = rank_space;

    struct FTparam * ftp = ft_param_alloc(dx+1,fapp,NULL,ranks);
    ft_param_create_from_lin_ls(ftp,ndata*dy,x,y,1e-6);
    struct RegressOpts * ropts = regress_opts_create(dx+1,AIO,FTLS);
    struct c3Opt * optimizer = c3opt_create(BFGS);
    if (verbose > 0){
        c3opt_set_verbose(optimizer,1);
    }
    c3opt_set_maxiter(optimizer,10000);
    c3opt_set_gtol(optimizer,1e-15);
    c3opt_set_relftol(optimizer,1e-12);

    struct FunctionTrain * ft_final =
        c3_regression_run(ftp,ropts,optimizer,ndata,x,y);


    
    size_t ntest = 100;
    double * test_inputs = generate_inputs(ntest);
    double * test_outputs = generate_outputs(&prob,ntest,test_inputs);

    double * ft_output = calloc_double(dy*ntest);

    double * diff = calloc_double(dy*ntest);
    double * pt = calloc_double(dx+1);
    for (size_t jj = 0; jj < ntest; jj++){
        for (size_t ii = 0; ii < dy; ii++){
            memmove(pt,test_inputs+jj*dx, dx * sizeof(double));
            pt[dx] = prob.x[ii];
            ft_output[ii + jj*dy] = function_train_eval(ft_final,pt);
            diff[ii + jj * dy] =
                ft_output[ii + jj*dy] - test_outputs[ii + jj*dy];
        }
    }

    
    double diff_se = cblas_ddot(dy*ntest,diff,1,diff,1);
    double norm_total = cblas_ddot(dy*ntest,test_outputs,1,test_outputs,1);
    printf("\n\n\n\t===================================\n\n");
    printf("\tDifference squared error = %G\n",diff_se);
    printf("\tSquared norm = %G\n",norm_total);
    printf("\n\n\n\n");
    
    char ftest[64];
    char ftest_ft[64];
    char ftest_diff[64];

    sprintf(ftest,"testing_funcs_n%zu.dat",ndata);
    sprintf(ftest_ft,"testing_funcs_ft_n%zu.dat",ndata);
    sprintf(ftest_diff,"testing_diff_n%zu.dat",ndata);
    save_array_with_x(dy,ntest,prob.x,test_outputs,ftest);
    save_array_with_x(dy,ntest,prob.x,ft_output,ftest_ft);
    save_array_with_x(dy,ntest,prob.x,diff,ftest_diff);
    /* dprint(dy,test_outputs); */
    /* dprint(dy,ft_output); */

    free(inputs); inputs = NULL;
    free(outputs); outputs = NULL;
    free(x); x = NULL;
    free(y); y = NULL;
    one_approx_opts_free_deep(&polyopts); polyopts = NULL;
    one_approx_opts_free_deep(&ko); ko = NULL; 
    multi_approx_opts_free(fapp); fapp = NULL;
    free(ranks); ranks = NULL;
    ft_param_free(ftp); ftp = NULL;
    regress_opts_free(ropts); ropts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    function_train_free(ft_final); ft_final = NULL;
    free(test_inputs); test_inputs = NULL;
    free(test_outputs); test_outputs = NULL;
    free(ft_output); ft_output = NULL;
    free(diff); diff = NULL;
    free(pt); pt = NULL;

}
