#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "stringmanip.h"
#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_probability.h"

#include "c3_interface.h"
//#include "lib_funcs.h"
//
//

#define offset 0.5
#define sigsize 1.0
#define corrlength 0.15

void read_data(char * filename, size_t nrows, size_t ncols, double * data)
{
    FILE * fp;
    fp =  fopen(filename, "r");
    if (fp == NULL){
        fprintf(stderr, "cat: can't open %s\n", filename);
        return;
        //return 0;
    }
    
    char line[255];
    char * temp = NULL;
    size_t ii = 0;
    size_t jj;
    while(fgets(line,255,fp) != NULL)
    {
        //printf("neww!!\n");
        //printf("%s",line);
        for (jj = 0; jj < ncols-1; jj++){
            temp = bite_string(line,' ');
            //printf("%s\n",temp);
            sscanf(temp,"%lf",&data[jj*nrows+ii]);
            free(temp); temp = NULL;
        }
        //printf("%s\n",line);
        sscanf(line,"%lf",&data[(ncols-1)*nrows+ii]);
        ii++;
    }
    fclose(fp);
}

double * buildRHS(size_t N, double lb, double ub)
{
    double * x = linspace(lb,ub,N);
    size_t ii;
    for (ii = 0; ii < N; ii++){
        x[ii] = pow(x[ii],2);
    }
    return x;
}

double * buildOP(size_t nperm, double * perm, double h)
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

double * solve(size_t nfield, double * perm, double h, double lb, double ub)
{
    double * op = buildOP(nfield,perm,h);
    double * rhs = buildRHS(nfield-1,lb+h,ub);
    
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
    dsyev_("V","L",(int*)&N,cov,(int*)&N,eigs,work,(int*)&lwork,&info);
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

struct RunArgs {
    size_t nfield;
    size_t dim;
    double * sqrt_cov;
    size_t output;
};

double solveBlackBox(double * in, void * args)
{

    struct RunArgs * rargs = args;
    //printf("in = \n");
    //dprint(rargs->dim,in);
    
    double * perm = calloc_double(rargs->nfield);
    double * sol = fullRun(rargs->nfield,rargs->dim,in,perm,rargs->sqrt_cov);
    free(perm);
    double out = sol[rargs->output];

    free(sol); sol = NULL;

    //printf("out = %G\n",out);

    return out;
}

double solveBlackBoxUni(const double * uni, void * args)
{
    
    struct RunArgs * rargs = args;
    double * input = calloc_double(rargs->dim);
    size_t ii;
    for (ii = 0; ii < rargs->dim; ii++){
        input[ii] = icdf_normal(0.0,1.0,uni[ii]);
    }
    
    double out = solveBlackBox(input,args);
    free(input); 
    input = NULL;
    return out;
}

void proc_inputs(int argc, char * argv[], struct RunArgs * rargs)
{

    size_t nfield = 0;
    int fin_exists = 0;
    int fout_exists = 0;
    char filename[255];
    char filein[255];
    char covfile[255] = "cov.dat";
    double * sqrt_cov = NULL;
    size_t dim = 0;
    
    int ii;
    printf("Processing program inputs: \n");
    for (ii = 1; ii < argc; ii++){
        char * name = bite_string(argv[ii],'=');
        if (strcmp(name,"fileout") == 0){
            fout_exists = 1;
            printf("........filename is %s\n",argv[ii]);
            strcpy(filename,argv[ii]);
        }
        else if (strcmp(name,"filein") == 0){
            fin_exists = 1;
            printf("........Random samples loaded from %s\n",argv[ii]);
            strcpy(filein,argv[ii]);
        }
        else if (strcmp(name,"size") == 0){
            printf("........Size of field is %s\n",argv[ii]);
            nfield = (size_t) atol(argv[ii]);
        }
        else if (strcmp(name,"sqrtcov") == 0){
            printf("........Reading sqrt of covariance from %s\n", argv[ii]);
            strcpy(covfile,argv[ii]);
            sqrt_cov = darray_load(covfile,0);
        }
        else if (strcmp(name,"dim") == 0){
            printf("........Number of dimensions to use %s\n", argv[ii]);
            dim = (size_t) atol(argv[ii]);

        }
        free(name); name = NULL;
    }

    if (nfield == 0){
       printf("Correct func call = ./elliptic1d dim=<number of dims> size=<number of pts> filein=<filein> fileout=<fileout> sqrtcov=<filename>\n");
       exit(1);
    }

    size_t output = (size_t)llround(floor( 0.7 * nfield)); // fixed from cast to size_t
    double * xsol = linspace(0.0,1.0,nfield);
    if (sqrt_cov == NULL){
        printf("........Building sqrt\n");
        double * eigs = calloc_double(nfield);
        sqrt_cov = buildCovSqrt(nfield, xsol,eigs, sigsize, corrlength);
        printf("........Saving sqrt\n");
        darray_save(nfield, nfield, sqrt_cov, covfile, 0);
        
        double sumtot = 0.0;
        size_t zz;
        for (zz = 0; zz < (size_t) nfield; zz++){
            sumtot += pow(eigs[nfield-1-zz],2);
        }
        double sum = 0.0;
        for (zz = 0.0; zz < (size_t) nfield; zz++){
            sum += pow(eigs[nfield-1-zz],2);
            if (sum > 0.99 * sumtot){
                break;
            }
        }
        size_t dimmodel = zz;
        double * saveeigs = calloc_double(3*nfield);
        for (zz = 0; zz < (size_t)nfield; zz++){
            saveeigs[zz] = (double) zz;
            saveeigs[zz+nfield] = eigs[nfield-1-zz];
            saveeigs[zz+2*nfield] = eigs[nfield-1-dimmodel];
        }
        printf("dimension = %zu \n",dimmodel);
        darray_save(nfield,3,saveeigs,"eigenvalues.txt",1);
        free(eigs);
        free(saveeigs);
    }

   
    double * perm = NULL;
    double * sol = NULL;
    if (fin_exists == 0){
        perm = darray_val(nfield,1.0);
        double * permout = darray_val(nfield,1.0);
        sol = fullRun(nfield, nfield, perm, permout,NULL);
        free(permout);
    }
    else{
        perm = calloc_double(dim);
        read_data(filein,dim,1,perm);
        dprint(dim, perm);
        double * permout = darray_val(nfield,1.0);
        sol = fullRun(nfield, dim, perm, permout,sqrt_cov);
        free(permout);
    }
    
    if (fout_exists == 1){ // simulate 
        size_t kk,ll;
        size_t nsym = 20;
        double * pt = calloc_double(nfield);
        double * fields = calloc_double(nfield*(nsym+1));
        double * x = linspace(0,1,nfield);
        for (kk = 0; kk < nfield; kk++){
            fields[kk] = x[kk];
        }
        free(x);
        double * sols = calloc_double(nsym);
        for (kk = 0; kk < nsym; kk++){
            for (ll = 0; ll < nfield; ll++){
                pt[ll] = icdf_normal(0.0,1.0,randu());
            }
            //dprint(nfield, pt);
            sol = fullRun(nfield, nfield, pt,fields+(kk+1)*nfield, sqrt_cov);
            sols[kk] = sol[output];
        }
        darray_save(nfield,15,fields,filename,1);
        darray_save(nsym,1,sols,"solshisttrue.txt",1);
        free(sols);
        free(pt);
    }
    rargs->nfield = nfield;
    rargs->dim = dim;
    rargs->sqrt_cov = calloc_double(nfield * nfield);
    rargs->output = output;
    printf("Evaluating at point %G\n",xsol[output]);
    memmove(rargs->sqrt_cov, sqrt_cov, nfield*nfield * sizeof(double));
    
    
    /*
    size_t jj;
    for (jj = 0; jj < nfield; jj++){ perm[jj] = log(perm[jj]-offset); }
    double * p = dconcat_cols(nfield,1,1,xsol,sol);
    double * p2 = dconcat_cols(nfield,2,1,p,perm);
    int success = darray_save(nfield,3,p2,filename,1);
    assert (success == 1);
    free(p); p = NULL;
    free(p2); p2 = NULL;
    */

    
    free(sqrt_cov); sqrt_cov = NULL;
    free(xsol); xsol = NULL;
    free(perm); perm = NULL;
    free(sol); sol = NULL;
}

int main(int argc, char *argv[])
{
    
    struct RunArgs rargs;
    proc_inputs(argc, argv,&rargs);
    
    size_t iii,jjj;
    size_t nround = 12;
    size_t napprox = 13;
    double roundtol[12] = {5e-3,1e-3,5e-4,1e-4,5e-5,
                           1e-5,5e-6,1e-6,5e-7,1e-7,
                           5e-8,1e-8};
    //double roundtol[1] = {1e-7};
    //double roundtol[2] = {1e-9,1e-10};
    //double roundtol[2] = {1e-4,1e-6};
    //double approxtol[1] = {1e0};
    //double roundtol[3] = {1e-2,1e-5,1e-8};
    //double approxtol[3] = {1e-1,1e-3,1e-5};

    double approxtol[13] = {1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,
                           1e-4,5e-5,1e-5,5e-6,1e-6,5e-7,
                           1e-7};

    //double approxtol[1] = {1e-6};
    //double approxtol[3] = {1e-1,1e-2,1e-3};

    double lb=0.05;
    double ub=0.95;
    for (iii = 0; iii < nround; iii++){
        for (jjj = 0; jjj < napprox; jjj++){
            printf("done prcessing\n");

            size_t dim = rargs.dim;
            
            struct FunctionMonitor * fm = NULL;
            fm = function_monitor_initnd(solveBlackBoxUni,&rargs,dim,1000*dim);
            struct Fwrap * fw = fwrap_create(dim,"general");
            fwrap_set_f(fw,function_monitor_eval,fm);

            struct OpeOpts * opts = ope_opts_alloc(LEGENDRE);
            ope_opts_set_start(opts,3);
            ope_opts_set_coeffs_check(opts,1);
            ope_opts_set_tol(opts,approxtol[jjj]);
            ope_opts_set_maxnum(opts,25);
            ope_opts_set_lb(opts,lb);
            ope_opts_set_ub(opts,ub);
            struct OneApproxOpts * qmopts = one_approx_opts_alloc(POLYNOMIAL,opts);    
            struct C3Approx * c3a = c3approx_create(CROSS,dim);
            int verbose = 0;
            size_t init_rank = 5;
            double ** start = malloc_dd(dim);
            for (size_t ii = 0; ii < dim; ii++){
                c3approx_set_approx_opts_dim(c3a,ii,qmopts);
                start[ii] = linspace(lb,ub,init_rank);
            }
            c3approx_init_cross(c3a,init_rank,verbose,start);
            c3approx_set_verbose(c3a,2);
            c3approx_set_adapt_kickrank(c3a,5);
            c3approx_set_adapt_maxrank_all(c3a,init_rank + 3*5);
            c3approx_set_cross_tol(c3a,roundtol[iii]);
            c3approx_set_round_tol(c3a,roundtol[iii]);

            // cross approximation with rounding
            struct FunctionTrain * ft = c3approx_do_cross(c3a,fw,1);
            

            char ftsave[255] = "ftrain.ft";
            int success = function_train_save(ft,ftsave);
            assert (success == 1);

            //ft = function_train_load(ftsave);
            //double * pt = darray_val(rargs.dim,0.5); // this is the mean
            //double valft = function_train_eval(ft,pt);
            //double valtrue = solveBlackBoxUni(pt,&rargs);
            //free(pt); pt = NULL;
            //printf("valtrue=%G, valft=%G\n",valtrue,valft);
            
            if (rargs.dim == 2){
                size_t N = 20;
                double * x = linspace(0.05,0.95,N);
                size_t kk,ll;
                double pt[2];
                double val[400*4];
                size_t index = 0;
                for (kk = 0; kk < N; kk++){
                    pt[0] = x[kk];
                    for (ll = 0; ll < N; ll++){
                        pt[1] = x[ll];
                        val[index] = pt[0];
                        val[index+400] = pt[1];
                        val[800+kk*N+ll] = solveBlackBoxUni(pt,&rargs);;
                        val[1200+kk*N+ll] = function_train_eval(ft,pt);;
                        index++;
                    }
                }
                darray_save(N*N,4,val,"2dcontour.dat",1);
                free(x); x = NULL;
            }
            else{
                size_t N = 10000;
                size_t kk,ll;
                double errnum = 0.0;
                double errden = 0.0;
                double * x = calloc_double(rargs.dim);
                for (kk = 0; kk < N; kk++){
                    for (ll = 0; ll < rargs.dim; ll++){
                        x[ll] = randu() * (0.95-0.05) + 0.05;
                    }
                    double diff =  solveBlackBoxUni(x,&rargs) - function_train_eval(ft,x);
                    //printf("\n");
                    //dprint(rargs.dim,x);
                    //printf("diff = %G\n",pow(diff,2) / pow(solveBlackBoxUni(x,&rargs),2));
                    errden += pow(solveBlackBoxUni(x,&rargs),2);
                    errnum += pow(diff,2);
                }
                double err = errnum/errden;
                printf("err = %G\n",err);
                size_t nvals = nstored_hashtable_cp(fm->evals);
                printf("number of evaluations = %zu \n", nvals);
                printf("ranks are "); iprint_sz(dim+1,ft->ranks);

                double * data = calloc_double((rargs.dim+1)*2);
                for (kk = 0; kk < rargs.dim+1; kk++){
                    data[kk] = (double) ft->ranks[kk];
                }
                data[rargs.dim+1] = sqrt(err);
                data[rargs.dim+2] = (double) nvals;
                char fff[256];
                sprintf(fff,"rtol=%G_apptol=%G",roundtol[iii],approxtol[jjj]);
                darray_save(rargs.dim+1,2,data,fff,1);
            }
            

            function_monitor_free(fm); fm = NULL;
            function_train_free(ft); ft = NULL;
            c3approx_destroy(c3a);
            fwrap_destroy(fw);
            one_approx_opts_free_deep(&qmopts);


        }
    }
    free(rargs.sqrt_cov); rargs.sqrt_cov = NULL;
    return 0;
}
