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
static double lb;
static double ub;
#define dy 32

static char * program_name;

void print_code_usage (FILE *, int) __attribute__ ((noreturn));
void print_code_usage (FILE * stream, int exit_code)
{
    fprintf(stream, "Two examples of functional regression \n\n");
    fprintf(stream, "Usage: %s options \n", program_name);
    fprintf(stream,
            " -h --help       Display this usage information.\n"
            " -f --function   Which function to evaluate. \n"
            "                  0: a_1x^2 + a_2x^2 (default)\n"
            "                  1: more complicated, check code\n"
            "                  2: sum of weighted gaussians bumps\n"
            "                  3: sum of (modulated) weighted gaussians bumps\n"
            " -n --number     Number of training samples (default 100).\n"
            " -p --polyorder  Polynomial order for parameters (default 4).\n"
            " -b --basis      Basis for spatial variable\n"
            "                  0: piecewise linear continuous (default)\n"
            "                 >0: radial basis function kernels\n"
            " -s --rankspace  Rank between spatial and paramter variables (default 4)\n"
            " -r --rankparam  Rank between parameters (default 2)\n"
            " -a --adaptrank  Flag whether or not to adapt rank\n"
            "                 0: no adaptation (default)\n"
            "                 1: adaptation\n"
            " -l --lowerbound Parameter lower bounds (default -1.0)\n"
            " -u --upperbound Parameter upper bounds (default 1.0)\n"
            " -v --verbose    Output words (default 0), 1 then show CVs, 2 then show opt progress.\n"
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
    int func;
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
        sum += (input[ii]-lb)/(ub-lb);
    }
    double coeff = 0.0;
    for (size_t ii = 0; ii < dx/2; ii++){
        coeff += (input[ii+dx/2]-lb)/(ub-lb);
    }
    for (size_t ii = 0; ii < prob->noutput; ii++){
        output[ii] = sin(sum * prob->x[ii])  + 0.05 * coeff * exp(-pow(prob->x[ii]-0.5,2)/0.01);
    }
}

static void third(prob_t * prob, double * input, double * output)
{
    double mean[6] = {0.15, 0.25, 0.45, 0.65, 0.75, 0.85};
    double stdev[6] = {0.01, 0.02, 0.03, 0.01, 0.05, 0.01};
    /* double stdev[6] = {0.05, 0.05, 0.05, 0.05, 0.05, 0.05}; */
    for (size_t ii = 0; ii < prob->noutput; ii++){
        output[ii] = 0.0;
        for (size_t jj = 0; jj < 6; jj++){
            double bump_height = 1.0 / sqrt(2.0*M_PI * stdev[jj]*stdev[jj]);
            /* double bump_height = 1.0;  */
            double exponent = - (prob->x[ii]-mean[jj])*(prob->x[ii]-mean[jj]) / (2.0 * stdev[jj] * stdev[jj]);
            output[ii] += input[jj] * bump_height * exp(exponent);
        }
    }
}

static void fourth(prob_t * prob, double * input, double * output)
{
    (void)prob;
    (void)input;
    (void)output;

    /* double mean[19] = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, */
    /*                    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95}; */

    /* double mean[6] = {0.15, 0.25, 0.45, 0.65, 0.75, 0.85}; */
    /* double stdev[6] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2}; */

    /* for (size_t ii = 0; ii < prob->noutput; ii++){ */
    /*     output[ii] = 0.0; */
    /*     double val = 0.0; */
    /*     for (size_t jj = 0; jj < prob->ninput; jj++){ */
    /*         val += (input[jj] - lb)/(ub-lb); */
    /*     } */
        /* output[ii] += sin(val + prob->x[ii]);AAAAAAAAAAAAAAAA */
        /* for (size_t jj = 0; jj < 6; jj++){ */
        /*     double exponent = -(prob->x[ii]-mean[jj])*(prob->x[ii]-mean[jj]) / (2.0 * stdev[jj] * stdev[jj]); */
        /*     output[ii] += exp(exponent); */
        /* } */

    /* } */
}


static double * generate_inputs(size_t n)
{
    double * x = calloc_double(dx * n);
        
    for (size_t ii = 0; ii < n; ii++){
        for (size_t jj = 0; jj < dx; jj++){
            x[jj + ii*dx] = randu()*(ub-lb) + lb;
        }
    }

    return x;
}

static double * generate_outputs(prob_t * prob, size_t n, double * inputs)
{
    double * y = calloc_double(n * dy);
    if (prob->func == 0){
        for (size_t ii = 0; ii < n; ii++){
            quadratic(prob,inputs + ii*dx,y+ii*dy);
        }
    }
    else if (prob->func == 1) {
        for (size_t ii = 0; ii < n; ii++){
            other(prob,inputs + ii*dx,y+ii*dy);
        }
    }
    else if (prob->func == 2){
        for (size_t ii = 0; ii < n; ii++){
            third(prob,inputs + ii*dx,y+ii*dy);
        }
    }
    else if (prob->func == 3){
        for (size_t ii = 0; ii < n; ii++){
            fourth(prob,inputs + ii*dx,y+ii*dy);
        }
    }

    /* for (size_t ii = 0; ii < n*dy; ii++){ */
    /*     y[ii] += randn()*0.01; */
    /* } */

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
    const char * const short_options = "hf:n:p:b:s:r:a:l:u:v:";
    const struct option long_options[] = {
        { "help"      , 0, NULL, 'h' },
        { "function"  , 1, NULL, 'f' },
        { "number"    , 1, NULL, 'n' },
        { "polyorder" , 1, NULL, 'p' },
        { "basis"     , 1, NULL, 'b' },
        { "rankspace" , 1, NULL, 's' },
        { "rankparam" , 1, NULL, 'r' },
        { "adaptrank" , 1, NULL, 'a' },
        { "lowerbound", 1, NULL, 'l'},
        { "upperbound", 1, NULL, 'u'},
        { "verbose"   , 1, NULL, 'v' },
        { NULL        ,  0, NULL, 0   }
    };

    size_t npoly = 4;
    size_t ndata = 100;
    int verbose = 0;
    size_t function = 0;
    size_t basis = 0;
    size_t rank_space = 4;
    size_t rank_param = 2;
    lb = -1.0;
    ub = 1.0;
    unsigned int adapt = 0;
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
            case 'a':
                adapt = strtoul(optarg,NULL,10);
                break;
            case 'l':
                lb = strtod(optarg,NULL);
                break;
            case 'u':
                ub = strtod(optarg,NULL);
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
        dx = 16;
    }
    else if (function == 0){
        dx = 2;
    }
    else if (function == 2){
        dx = 6;
    }
    else if (function == 3){
        dx = 12;
        fprintf(stderr, "Function 3 does not work yet\n");
        exit(1);
    }
    prob_t prob;
    prob.ninput = dx;
    prob.noutput = dy;
    prob.x = linspace(0.0,1.0,dy);
    prob.func = function;    
    
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
    

    double * inputs = generate_inputs(ndata);
    double * outputs = generate_outputs(&prob,ndata,inputs);

    save_array_with_x(dy,ndata,prob.x,outputs,"training_funcs.dat");
    
    double * x = calloc_double((dx+1)*dy * ndata);
    double * y = calloc_double(ndata*dy);
    
    /* size_t ntotdata = create_unified_data(ndata,inputs,prob.x,outputs,x,y); */    
    create_unified_data(ndata,inputs,prob.x,outputs,x,y);    

    
    /* dprint2d_col(dy,ndata,y); */
    /* printf("ntotal data = %zu\n",ntotdata); */

    /* size_t nparams = 5; */
    /* double * centers = linspace(lb,ub,nparams); */
    /* double scale = 1.0; */
    /* double w = pow(nparams,-0.2)*2.0/12.0; */
    /* w *= 10; */
    /* /\* printf("\t width = %G\n",width); *\/ */

    /* struct KernelApproxOpts * opts = kernel_approx_opts_gauss(nparams,centers,scale,w); */
    /* struct OneApproxOpts * polyopts = one_approx_opts_alloc(KERNEL,opts); */

    
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_nparams(opts,npoly);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    struct OneApproxOpts * polyopts = one_approx_opts_alloc(POLYNOMIAL,opts);


    double width = pow(dy,-0.2)/sqrt(12.0);
    width *= 0.5;

    struct KernelApproxOpts * kopts = kernel_approx_opts_gauss(prob.noutput,prob.x,1.0,width);
    struct LinElemExpAopts * lopts = lin_elem_exp_aopts_alloc(prob.noutput,prob.x);

    struct OneApproxOpts * ko = NULL;
    if (basis == 0){
        ko = one_approx_opts_alloc(LINELM,lopts);
    }
    else{
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

    struct FTRegress * ftr = ft_regress_alloc(dx+1,fapp,ranks);
    ft_regress_set_alg_and_obj(ftr,AIO,FTLS);

    /* ft_regress_set_alg_and_obj(ftr,AIO,FTLS_SPARSEL2); */
    ft_regress_set_regularization_weight(ftr,1e-10);
    if (adapt == 1){
        ft_regress_set_adapt(ftr,1);
        ft_regress_set_roundtol(ftr,1e-7);
        ft_regress_set_maxrank(ftr,10);
        ft_regress_set_kickrank(ftr,1);
    }
    if (verbose > 0){
        ft_regress_set_verbose(ftr,1);
    }
    

    /* struct c3Opt * optimizer = c3opt_create(SGD); */
    /* c3opt_set_absxtol(optimizer,1e-30); */
    /* c3opt_set_sgd_nsamples(optimizer,ndata*dy); */
    /* c3opt_set_sgd_learn_rate(optimizer,1e-3); */
    /* ft_regress_set_stoch_obj(ftr,1); */
    /* c3opt_set_maxiter(optimizer,50); */
    
    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_maxiter(optimizer,1000);
    if (verbose > 1){
        c3opt_set_verbose(optimizer,1);
    }
    c3opt_set_gtol(optimizer,1e-8);
    c3opt_set_relftol(optimizer,1e-10);


    struct FunctionTrain * ft_final = ft_regress_run(ftr,optimizer,ndata*dy,x,y);

    size_t ntest = 1000;
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
    printf("\tFinal ranks: "); iprint_sz(dx+2,function_train_get_ranks(ft_final));

    printf("\tDifference squared error = %G\n",diff_se);
    printf("\tSquared norm = %G\n",norm_total);
    printf("\n\n\n\n");
    
    char ftest[64];
    char ftest_ft[64];
    char ftest_diff[64];
    char ft_filename[64];

    sprintf(ft_filename,"ft_n%zu_relsquarederr_%3.5G.c3",ndata,diff_se/norm_total);
    function_train_save(ft_final,ft_filename);
    
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

    ft_regress_free(ftr); ftr = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    function_train_free(ft_final); ft_final = NULL;
    free(test_inputs); test_inputs = NULL;
    free(test_outputs); test_outputs = NULL;
    free(ft_output); ft_output = NULL;
    free(diff); diff = NULL;
    free(pt); pt = NULL;

}
