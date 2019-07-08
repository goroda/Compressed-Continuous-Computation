#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "c3.h"

#define offset 0.5
#define sigsize 1.0
#define corrlength 0.15
#define output 2
static size_t nfield = 64;


static double * buildEllipticRHS(size_t N, double lb, double ub)
{
    double * x = linspace(lb,ub,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        x[ii] = pow(x[ii],2);
    }
    return x;
}

static double * buildEllipticOP(size_t nperm, double * perm, double h)
{
    
    // dirichlet on left, neumann on right
    // Note it is negative!
    
    size_t N = nperm-1;
    double * A = calloc_double(N * N);
    
    size_t ii;
    for (ii = 0; ii < N-1; ii++){
        A[(ii+1)*N+ii] = - ( -(perm[ii+1] + perm[ii+2])/2.0/h/h );
        A[ii*N+ii] = - ( (perm[ii] + 2.0 * perm[ii+1]  + perm[ii+2])/2.0/h/h );
        A[ii*N+ii+1] = - ( -(perm[ii+1] + perm[ii+2])/2.0/h/h );
    }
    A[N*N-1] = - (perm[N] + perm[N-1])/2.0/h/h;

    return A;
}

static double * solve(size_t nfield, double * perm, double h, double lb, double ub)
{
    double * op = buildEllipticOP(nfield,perm,h);
    double * rhs = buildEllipticRHS(nfield-1,lb+h,ub);
    
    double * inv = calloc_double((nfield-1) * (nfield-1));
    pinv(nfield-1,nfield-1,nfield-1,op,inv,0.0);
    
    double * sol = calloc_double(nfield);
    cblas_dgemv(CblasColMajor,CblasNoTrans,nfield-1,nfield-1,1.0,inv,nfield-1,
                rhs,1,0.0,sol+1,1);

    free(op); op = NULL;
    free(rhs); rhs = NULL;
    free(inv); inv = NULL;

    return sol;
}

double evalExpKernel(double x1, double x2, double sig, double Lc)
{
    //double out = pow(sig,2) * exp(-fabs(x1-x2) * Lc);
    double out = sig * exp(-fabs(x1-x2) / 2.0 /Lc / Lc);
    return out;
}

double * sqrtCov(size_t N, double * eigs, double * cov)
{
    double * sqrt_cov = calloc_double(N*N);
    
    size_t lwork = N*8;
    double * work = calloc_double(lwork);
    int info;
    dsyev_("V","L",&N,cov,&N,eigs,work,&lwork,&info);
    if (info != 0){
        fprintf(stderr, "info = %d in computing sqrt of cov\n",info);
    }
    assert (info == 0);

    size_t ii,jj;
    for (ii = 0; ii < N; ii++){
        for (jj = 0; jj < N; jj++){
            sqrt_cov[ii*N+jj] = cov[(N-1-ii)*N+jj]*sqrt(eigs[N-1-ii]); // eigenvalues are stored in ascending order
        }
    }
    free(work); work = NULL;
    return sqrt_cov;
}

double * buildCovSqrt(size_t N, double * x, double * eigs, 
                        double sig, double Lc)
{
    double * cov = calloc_double(N * N);
    size_t ii,jj;
    for (ii = 0; ii < N; ii++){
        for (jj = ii; jj < N; jj++){
            cov[ii*N+jj] = evalExpKernel(x[ii],x[jj],sig,Lc);
            cov[jj*N+ii] = cov[ii*N+jj];
        }
    }
    double * sqrt_cov = sqrtCov(N,eigs,cov);
    free(cov); cov = NULL;
    return sqrt_cov;
}


double * fullRun(size_t nfield, size_t dim, double * permin, double * permout, double * sqrt_cov)
{
    double * sol = NULL;
    double h = 1.0/ (double) (nfield-1);
    if (sqrt_cov == NULL){
        assert (dim == nfield);
        sol = solve(nfield,permin, h, 0.0, 1.0);
    }
    else{
	//printf("nfield = %zu\n",nfield);
        cblas_dgemv(CblasColMajor,CblasNoTrans,nfield,dim,1.0,sqrt_cov,
                nfield, permin,1,0.0,permout,1);
    
        //memmove(perm,temp,nfield*sizeof(double));
        size_t ii; 
        for (ii = 0; ii < nfield; ii++){
            permout[ii] = exp(permout[ii] + offset);
        }
   	//printf("temp = \n");
	//dprint(nfield,temp);

        sol = solve(nfield,permout, h, 0.0, 1.0);
    }
    return sol;
}

double * compute_KL()
{
    double * xsol = linspace(0.0,1.0,nfield);
    double * eigs = calloc_double(nfield);
    double * sqrt_cov = buildCovSqrt(nfield, xsol,eigs, sigsize, corrlength);
    free(xsol); xsol = NULL;
    free(eigs); eigs = NULL;
    return sqrt_cov;
}

void save_solution(double * x, double * sol, char * filename)
{
    double * together = calloc_double(nfield*2);
    for (size_t ii = 0; ii < nfield; ii++){
        together[ii] = x[ii];
        together[ii+nfield] = sol[ii];
    }
    darray_save(nfield,2,together,filename,1);
    free(together); together = NULL;
}

double * one_eval(size_t nrand, double * permin, double * sqrt_cov)
{
    double * permout = calloc_double(nfield);
    double * vals = fullRun(nfield,nrand,permin,permout,sqrt_cov);
    free(permout); permout = NULL;
    return vals;
}

void gen_data(size_t ndata, double * x, double * y, size_t nrand, double * xdisc,  double * sqrt_cov)
{
    double * permin = calloc_double(nrand);
    double * permout = calloc_double(nfield);

    size_t on_point = 0;
    for (size_t ii = 0; ii < ndata; ii++){
        for (size_t jj = 0; jj < nrand; jj++){
            permin[jj] = randn();
        }
        
        double * vals = fullRun(nfield,nrand,permin,permout,sqrt_cov);
        dprint(nfield,vals);
        /* for (size_t jj = 0; jj < nfield; jj++){ */

            for (size_t kk = 0; kk < nrand; kk++){
                x[kk + on_point*nrand] = permin[kk];
            }
            /* x[on_point*(nrand+1)] = xdisc[jj]; */
            y[on_point] = vals[nfield-output];
            on_point++;
        /* } */
        /* save_solution(xdisc,vals,"sol.dat"); */

        free(vals); vals = NULL;
    }
    /* printf("x is\n"); */
    /* dprint2d_col(nrand+1,ndata*nfield,x); */
    /* dprint2d_col(ndata*nfield,1,y); */
    free(permin); permin = NULL;
    free(permout); permout = NULL;
}

int main()
{
    int pid = getpid();
    struct timeval t;
    gettimeofday(&t, NULL);
    srand(t.tv_usec * t.tv_sec * pid);
    
    
    size_t nrand = 3;
    size_t dimx = 1;
    size_t nvars = nrand;
    
    /* size_t ranks[3] = {1, 20, 20, 1}; */
    size_t ranks[4] = {1, 3, 3, 1};

    size_t nout = 1;
    size_t ndata_base = 10000;
    size_t ndata = ndata_base*nout;
    double * x = calloc_double(nvars*ndata);
    double * y = calloc_double(ndata);

    
    size_t ndisc = nfield;
    double lbx = 0.0;
    double ubx = 1.0;
    double * xdisc = linspace(lbx,ubx,ndisc);
    
    // reverse order!
    double * sqrt_cov = compute_KL();
    

    gen_data(ndata_base,x,y,nrand,xdisc,sqrt_cov);

    /* struct LinElemExpAopts * lopts = lin_elem_exp_aopts_alloc(ndisc,xdisc); */
    /* struct OneApproxOpts * linopts = one_approx_opts_alloc(LINELM,lopts); */
    struct KernelApproxOpts * lopts = kernel_approx_opts_gauss(ndisc,xdisc,1.0,0.2);
    struct OneApproxOpts * linopts = one_approx_opts_alloc(KERNEL,lopts);
    /* struct OpeOpts * lopts = ope_opts_alloc(LEGENDRE); */
    /* ope_opts_set_lb(lopts,0); */
    /* ope_opts_set_nparams(lopts,2); */

    
    size_t npoly = 4;
    struct OpeOpts * opts = ope_opts_alloc(HERMITE);
    ope_opts_set_nparams(opts,npoly);
    struct OneApproxOpts * polyopts = one_approx_opts_alloc(POLYNOMIAL,opts);
    struct MultiApproxOpts * fapp = multi_approx_opts_alloc(nvars);

    size_t nparams = 0;
    /* multi_approx_opts_set_dim(fapp,0,linopts); */
    for (size_t ii = 0; ii < nrand; ii++){
        multi_approx_opts_set_dim(fapp,ii,polyopts);
        nparams += ranks[ii+1]*ranks[ii]*npoly;
    }
    double * params = calloc_double(nparams);
    for (size_t jj = 0; jj < nparams; jj++){
        params[jj] = 1e-1*(randu()*2.0-1.0);
    }


    double dot_data = cblas_ddot(ndata,y,1,y,1);
    printf("dot_data = %G\n",dot_data/(double)ndata);
        

    struct FTparam * ftp = ft_param_alloc(nvars,fapp,NULL,ranks);
    ft_param_create_from_lin_ls(ftp,ndata,x,y,1e-6);
    struct RegressOpts * ropts = regress_opts_create(nvars,AIO,FTLS);
    struct c3Opt * optimizer = c3opt_create(BFGS);
    c3opt_set_verbose(optimizer,1);
    c3opt_set_maxiter(optimizer,300);
    c3opt_set_gtol(optimizer,1e-15);
    c3opt_set_relftol(optimizer,1e-12);
    
    struct FunctionTrain * ft_final = c3_regression_run(ftp,ropts,optimizer,ndata,x,y);


    double * save_field = calloc_double(3*nfield);
    double * test = calloc_double(nrand);
    for (size_t ii = 0; ii < nrand; ii++){
        test[ii] = randn();
    }
    double * evals = one_eval(nrand, test, sqrt_cov);
    /* dprint(nfield,evals); */

    double evalft = function_train_eval(ft_final,test);
    printf("eval = %G, eval ft = %G\n",evals[nfield-output],evalft);
    /* double * tf = calloc_double(nrand+1); */
    /* double * evalsft = calloc_double(nfield); */
    /* for (size_t jj = 0; jj < nfield; jj++){ */
    /*     tf[0] = xdisc[jj]; */
    /*     memmove(tf+1,test,nrand*sizeof(double)); */
    /*     evalsft[jj] = function_train_eval(ft_final,tf); */

    /*     save_field[jj] = xdisc[jj]; */
    /*     save_field[jj + nfield] = evals[jj]; */
    /*     save_field[jj + 2 * nfield] = evalsft[jj]; */
    /* } */
    /* printf("\n"); */
    /* dprint(nfield,evalsft); */

    /* darray_save(nfield,3,save_field,"true_and_ft.dat",1); */

    
    free(test); test = NULL;
    free(evals); evals =NULL;
    /* free(tf); tf = NULL; */
    /* free(evalsft); evalsft = NULL; */
    
    free(x); x = NULL;
    free(y); y = NULL;
    free(sqrt_cov); sqrt_cov = NULL;
    free(xdisc); xdisc = NULL;
    regress_opts_free(ropts); ropts = NULL;
    ft_param_free(ftp); ftp = NULL;
    
    free(params); params = NULL;
    free(xdisc); xdisc = NULL;
    one_approx_opts_free_deep(&linopts); linopts = NULL;
    one_approx_opts_free_deep(&polyopts); polyopts = NULL;
    c3opt_free(optimizer); optimizer = NULL;
    function_train_free(ft_final); ft_final = NULL;
    multi_approx_opts_free(fapp); fapp = NULL;
    return 0;
}

