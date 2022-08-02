#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"
#include "parameterization.h"

#include "c3_interface.h"

static unsigned int seed = 3;

int main()
{

    srand(seed);

    size_t dim = 10;
    /* size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1}; */
    size_t ranks[11] = {1,2,2,2,3,4,2,2,2,2,1};
    double lb = -1.0;
    double ub = 1.0;
    size_t maxorder = 8;
    struct BoundingBox * bds = bounding_box_init(dim,lb,ub);
    // struct FunctionTrain * a = function_train_poly_randu(LEGENDRE,bds,ranks,maxorder);

    // create data
    size_t ndata = 100;
    double * x = calloc_double(ndata*dim);
    double * y = calloc_double(ndata);

    double noise_std = 0.01;

    printf("\n Example of the Gibbs Sampler for profiling \n");
    printf("\t  Dimensions: %ld\n", dim);
    printf("\t  LPOLY order: %ld\n", maxorder);
    printf("\t  ndata:       %ld\n", ndata);

    // // add noise
    for (size_t ii = 0 ; ii < ndata; ii++){
        y[ii] = noise_std*randn();
        for (size_t jj = 0; jj < dim; jj++){
            x[ii*dim+jj] = randu()*(ub-lb) + lb;
            y[ii] += sin(x[ii*dim+jj]);
        }
        // no noise!
        //function_train_eval(a,x+ii*dim);
        /* y[ii] += randn(); */
    }

        // Initialize Approximation Structure
    struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
    ope_opts_set_lb(opts,lb);
    ope_opts_set_ub(opts,ub);
    ope_opts_set_nparams(opts,maxorder+1);
    struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(dim);
    double * param_space = calloc_double(450);
    size_t onparam=0;

    for (size_t ii = 0; ii < dim; ii++){
        /* printf("ii = %zu\n",ii); */
        multi_approx_opts_set_dim(fapp,ii,qmopts);
        
        for (size_t jj = 0; jj < ranks[ii]; jj++){
            for (size_t kk = 0; kk < ranks[ii+1]; kk++){
                for (size_t ll = 0; ll < maxorder+1; ll++){
                    param_space[onparam] += (randu()*2.0-1.0);
                    onparam++;
                }
            }
        }
    }

    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_gtol(optimizer,1e-4);
    
    struct FTparam* ftp = ft_param_alloc(dim,fapp,param_space,ranks);
    struct RegressOpts* als_opts = regress_opts_create(dim,ALS,FTLS);
    regress_opts_set_verbose(als_opts,0);
    regress_opts_set_als_conv_tol(als_opts,1e-6);
    regress_opts_set_max_als_sweep(als_opts,100);
    
    printf("\nBefore regress\n");
    struct FunctionTrain * ft_final = c3_regression_run(ftp,als_opts,optimizer,ndata,x,y);
    printf("Before sample\n");
    // double diff = function_train_relnorm2diff(ft_final,a);
    // printf("\n\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff);
    // CuAssertDblEquals(tc,0.0,diff,1e-3);
    
    /* struct RegressOpts * aio_opts = regress_opts_create(dim,AIO,FTLS); */
    /* struct FunctionTrain * ft_final2 = c3_regression_run(ftp,aio_opts,optimizer,ndata,x,y); */
    /* double diff2 = function_train_relnorm2diff(ft_final2,a); */
    /* printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff2); */

    /* c3opt_set_gtol(optimizer,1e-10); */
    /* struct FunctionTrain * ft_final3 = c3_regression_run(ftp,als_opts,optimizer,ndata,x,y); */
    /* double diff3 = function_train_relnorm2diff(ft_final3,a); */
    /* printf("\t Relative Error: ||f - f_approx||/||f|| = %G\n",diff3); */
    /* CuAssertDblEquals(tc,0.0,diff3,1e-3); */
    
    
    double init_sample[ftp->nparams+2]; 
    init_sample[0] = 1/(noise_std*noise_std);
    init_sample[1] = 1/(0.01);

    for (size_t i=2; i<ftp->nparams+2; i++){
        init_sample[i] = ftp->params[i-2];
    }

    int prior_alpha = 100;
    double prior_theta = init_sample[1]/prior_alpha;

    double prior_alphas[dim];
    double prior_thetas[dim];
    for (size_t ii=0; ii<dim; ii++){
        prior_alphas[ii] = 100;
        prior_thetas[ii] = init_sample[1]/prior_alphas[ii];
    }

    double init_sample2[ftp->nparams+2]; 
    init_sample2[0] = 1/(noise_std*noise_std);

    for (size_t i=1; i<ftp->nparams+1; i++){
        init_sample2[i] = ftp->params[i-1];
    }

    int noise_alpha = 100;
    double noise_theta = init_sample[0]/noise_alpha;

    int Nsamples = 1000;
    double * out = calloc_double((ftp->nparams+2)*Nsamples);

    double * out2 = calloc_double((ftp->nparams+1)*Nsamples);
    double * var_out = calloc_double(dim*(Nsamples-1));

    
    // sample_hier_group_gibbs_linear_noise(ftp, ndata, x, y, init_sample, prior_alpha, prior_theta, 
    //     noise_alpha, noise_theta, Nsamples, out);

    clock_t start = clock() ;
    
    
    // sample_hier_ind_gibbs_linear_noise(ftp, ndata, x, y, init_sample2, prior_alphas, prior_thetas, 
    //     noise_alpha, noise_theta, Nsamples, out2, var_out);
    sample_hier_group_gibbs_linear_noise(ftp, ndata, x, y, init_sample, prior_alpha, prior_theta, 
        noise_alpha, noise_theta, Nsamples, out);

    clock_t end = clock() ;
    double elapsed_time = (end-start)/(double)CLOCKS_PER_SEC ;

    printf("Time: %f", elapsed_time);


    ft_param_free(ftp);             ftp         = NULL;
    regress_opts_free(als_opts);    als_opts    = NULL;
    free(param_space);              param_space = NULL;
    bounding_box_free(bds);         bds         = NULL;
    // function_train_free(a);         a           = NULL;
    function_train_free(ft_final);  ft_final    = NULL;

    /* regress_opts_free(aio_opts);    aio_opts    = NULL; */
    /* function_train_free(ft_final2); ft_final2   = NULL; */
    /* function_train_free(ft_final3); ft_final3   = NULL; */

    c3opt_free(optimizer); optimizer = NULL;
    
    one_approx_opts_free_deep(&qmopts);
    multi_approx_opts_free(fapp);
    free(x); x = NULL;
    free(y); y = NULL;
    free(out2); out2 = NULL;
    free(out); out = NULL;
    return 0;
}
