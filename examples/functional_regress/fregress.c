#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "c3.h"

/* #define dx 2 */
#define dx 100
#define dy 16

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
    for (size_t ii = 0; ii < n; ii++){
        /* quadratic(prob,inputs + ii*dx,y+ii*dy); */
        other(prob,inputs + ii*dx,y+ii*dy);
    }

    return y;
}

static size_t create_unified_data(size_t ndata, double * inputs, double * xspace,
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

int main()
{

    prob_t prob;
    prob.ninput = dx;
    prob.noutput = dy;
    prob.x = linspace(0.0,1.0,dy);

    size_t ndata = 200;
    double * inputs = generate_inputs(ndata);
    double * outputs = generate_outputs(&prob,ndata,inputs);

    save_array_with_x(dy,ndata,prob.x,outputs,"training_funcs.dat");
    
    double * x = calloc_double((dx+1)*dy * ndata);
    double * y = calloc_double(ndata*dy);

    size_t ntotdata = create_unified_data(ndata,inputs,prob.x,outputs,x,y);

    /* dprint2d_col(dy,ndata,y); */
    printf("ntotal data = %zu\n",ntotdata);

    size_t npoly = 4;
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_nparams(opts,npoly);
    struct OneApproxOpts * polyopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dx+1);

    double width = pow(dy,-0.2)/sqrt(12.0);
    width *= 0.5;
    printf("width=%G\n",width);
    struct KernelApproxOpts * kopts =
        kernel_approx_opts_gauss(prob.noutput,prob.x,1.0,width);
    struct OneApproxOpts * ko = one_approx_opts_alloc(KERNEL,kopts);
    /* struct LinElemExpAopts * lopts = lin_elem_exp_aopts_alloc(prob.noutput,prob.x); */
    /* struct OneApproxOpts * ko = one_approx_opts_alloc(LINELM,lopts); */
    for (size_t ii = 0; ii < dx; ii++){
        multi_approx_opts_set_dim(fapp,ii,polyopts);
    }
    multi_approx_opts_set_dim(fapp,dx,ko);

    size_t * ranks = calloc_size_t(dx+1+1);
    ranks[0] = 1;
    ranks[dx+1] = 1;
    for (size_t ii = 1; ii <= dx; ii++){
        ranks[ii] = 2;
    }
    ranks[dx] = 4;

    printf("here!\n");
    struct FTparam * ftp = ft_param_alloc(dx+1,fapp,NULL,ranks);
    ft_param_create_from_lin_ls(ftp,ndata*dy,x,y,1e-6);
    struct RegressOpts * ropts = regress_opts_create(dx+1,AIO,FTLS);
    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_verbose(optimizer,1);
    c3opt_set_maxiter(optimizer,10000);
    c3opt_set_gtol(optimizer,1e-15);
    c3opt_set_relftol(optimizer,1e-12);

    printf("there!\n");
    struct FunctionTrain * ft_final =
        c3_regression_run(ftp,ropts,optimizer,ndata,x,y);

    size_t ntest = 30;
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
    printf("Difference squared error = %G\n",diff_se);

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
    
}
