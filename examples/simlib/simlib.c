#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "array.h"
#include "linalg.h"
#include "lib_clinalg.h"
#include "lib_funcs.h"

double otlcircuit(double * x, void * args)
{
    assert (args == NULL );
    // 6 dimensions
    // All inputs should be [-1,1]
    double Rb1 = x[0]*(150.0-50.0)/2.0 + (50.0+150.0)/2.0; // [50,150]
    double Rb2 = x[1]*(75.0-25.0)/2.0 + (25.0+75.0)/2.0; // [25,70]
    double Rf = x[2]*(3.0-0.5)/2.0 + (0.5+3.0)/2.0; // [0.5,3]
    double Rc1 = x[3]*(2.5-1.2)/2.0 + (1.2+2.5)/2.0; // [1.2, 2.5]
    double Rc2 = x[4]*(1.2-0.25)/2.0 + (1.2+0.25)/2.0; // [0.25,1.2]
    double beta = x[5]*(300.0-50.0)/2.0 + (300.0+50.0)/2.0; // [50,300]
    
    double Vb1 = 12.0 * Rb2 / (Rb1 + Rb2);
    double Vm = (Vb1 + 0.74) * beta * (Rc2 + 9.0) / (beta * (Rc2 + 9) + Rf);
    Vm += 11.35 * Rf / (beta * (Rc2 + 9.0) + Rf);
    Vm += 0.74 * Rf * beta * (Rc2 + 9.0) / ( (beta * (Rc2 + 9) + Rf) * Rc1);
    
    //printf("Vm = %G \n", Vm);
    return Vm;
}

double borehole(double * x, void * args)
{
    assert (args == NULL);
    // 8 dimensions
    double rw = x[0] * (0.15 - 0.05)/2.0 + (0.05+0.15)/2.0; // [0.05,0.15]
    double r  = x[1] * (50000.0-100.0)/2.0 + (100.0+50000.0)/2.0; // [100,50000]
    double Tu = x[2] * (115600.0 - 63070.0)/2.0 + (63070.0+115600.0)/2.0; // [63070, 115600]
    double Hu = x[3] * (1110.0 - 990.0)/2.0 + (990.0+1110.0)/2.0; // [990,1110]
    double Tl = x[4] * (116.0 - 63.1)/2.0 + (63.1+116.0)/2.0; // [63.1, 116];
    double Hl = x[5] * (820.0 - 700.0)/2.0 + (700.0+820.0)/2.0; // [700,820];
    double L  = x[6] * (1680.0 - 1120.0)/2.0 + (1120.0+1680.0)/2.0; // [1120,1680];
    double Kw = x[7] * (12045.0 - 9855.0)/2.0 + (9855.0+12045.0)/2.0; // [1500,15000];

    double f = 2.0 * M_PI * Tu * (Hu-Hl);
    f /= log(r/rw);
    f /= (1.0 + 2.0*L*Tu / (log(r/rw) * rw * rw * Kw) + Tu / Tl);
    return f;
}

double piston (double * x, void * args)
{
    assert (args == NULL);
    // 7 dimensions
    double M = x[0] * (60.0-30.0) / 2.0 + (60.0+30.0) / 2.0;
    double S = x[1] * (0.020-0.005) / 2.0 + (0.005 + 0.020) / 2.0;
    double Vo = x[2] * (0.010 - 0.002) / 2.0 + (0.002 + 0.010) / 2.0;
    double k = x[3] * (5000.0 - 1000.0) / 2.0 + (1000.0 + 5000.0) / 2.0;
    double Po = x[4] * (110000-90000)/2.0 + (110000 + 90000) / 2.0;
    double Ta = x[5] * (296.0 - 290.0) / 2.0 + (296.0 + 290.0) / 2.0;
    double To = x[6] * (360.0 - 340.0) / 2.0 + (360.0 + 340.0) / 2.0;

    double A = Po*S + 19.62*M - k*Vo / S;
    double V = S / (2.0 * k) * (sqrt(A*A + 4.0 * k * Po * Vo * Ta / To) - A);
    double C = 2.0 * M_PI * sqrt( M / (k + S*S*Po*Vo*Ta/To/V/V));
    return C;
}

double robotarm(double * x, void * args)
{
    assert (args == NULL);
    // 8 dimensions
    
    double u = 0.0;
    double v = 0.0;
    //size_t ii,jj;

    double t1 = x[0] * M_PI + M_PI;
    double t2 = t1 + (x[1]*M_PI + M_PI);
    double t3 = t2 + (x[2]*M_PI + M_PI);
    double t4 = t3 + (x[3]*M_PI + M_PI);

    u = (x[4]*0.5+0.5) * cos(t1) + 
        (x[5]*0.5+0.5) * cos(t2) + 
        (x[6]*0.5+0.5) * cos(t3) + 
        (x[7]*0.5+0.5) * cos(t4);

    v = (x[4]*0.5+0.5) * sin(t1) + 
        (x[5]*0.5+0.5) * sin(t2) + 
        (x[6]*0.5+0.5) * sin(t3) + 
        (x[7]*0.5+0.5) * sin(t4);
    
    double f = sqrt ( u*u + v*v);
    return f;
}

double wingweight (double * x, void * args)
{
    assert (args == NULL);
    // 10 dimensions
    double Sw = x[0] * (200.0-150.0)/2.0 + (350.0)/2.0;
    double Wfw = x[1] * (80.0)/2.0 + 520.0/2.0;
    double A = x[2] * (10.0-6.0)/2.0 + 16.0/2.0;
    double lam = x[3] * 20.0/2.0;
    double q = x[4] * (45.0-16.0)/2.0 + (45.0+16.0)/2.0;
    double slam = x[5] * 0.5/2.0 + 1.5/2.0;
    double tc = x[6] * (0.18-0.08)/2.0 + (0.18+0.08)/2.0;
    double Nz = x[7] * (6.0-2.5)/2.0 + (6.0+2.5)/2.0;
    double Wdg = x[8] * (2500.0-1700.0)/2.0 + (2500.0+1700.0)/2.0;
    double Wp = x[9] * (0.08-0.025)/2.0 + (0.08 + 0.025)/2.0;

    double f = 0.036 * pow(Sw,0.758) * pow(Wfw,0.0035) * 
               pow(A/pow(cos(lam*M_PI/180.0),2),0.6) * pow(q,0.006) * 
               pow(slam,0.04) * pow(  100.0*tc / cos(lam*M_PI/180.0), -0.3) * 
               pow(Nz * Wdg, 0.49);
    f += Sw * Wp;
    return f;
}


double friedman (double * x, void * args)
{
    assert (args == NULL);
    // 5 dimension
    
    double f = 10.0 * sin(M_PI * (x[0]*0.5+ 0.5) * (x[1]*0.5 + 0.5));
    f += 20.0 * pow ((x[2]*0.5+0.5) - 0.5,2) + 10.0 * (x[3]*0.5+0.5) + 
         5.0 * (x[4]*0.5+0.5);

    return f;
}

double gramlee09 (double * x, void * args)
{
    //Gramacy & LEE (2009)
    assert (args == NULL);
    // 6 dimension (last two variables not active)
    
    double f = exp(sin( pow( 0.9 * (x[0]*0.5+0.5 + 0.48),10.0 ) ));
    f += (x[1]*0.5+0.5) * (x[2]*0.5+0.5) + x[3] * 0.5 +0.5;

    return f;
}

double detpep10 (double * x, void * args)
{
    //Dette & Pepelyshev (2010)
    assert (args == NULL);
    // 8 dimensions
    
    double f = 4.0 * pow (x[0]*0.5+0.5 - 2.0 * 8.0*(x[1]*0.5+0.5) - 
                         8.0*pow(x[1]*0.5+0.5,2.0),2.0) +
                     pow(3.0-4.0*(x[1]*0.5+0.5),2) + 
               16.0 * sqrt(x[2]*0.5+0.5+1)*pow(2.0*(x[2]*0.5+0.5)-1.0,2);
    
    size_t ii,jj;
    for (ii = 4; ii < 9; ii++){
        double temp = 0.0;
        for (jj = 3; jj < ii+1; jj++){
            temp += (x[jj-1]*0.5+0.5);
        }
        f += (double) ii * log(1.0 + temp);
    }

    return f;
}

double detpep10exp (double * x, void * args)
{
    //Dette & Pepelyshev (2010) Exponential
    assert (args == NULL);
    // 3 dimensions
    
    double x1 = x[0]*0.5+0.5;
    double x2 = x[1]*0.5+0.5;
    double x3 = x[2]*0.5+0.5;
    double f = 100.0 * ( exp( -2.0 / pow(x1,1.75)) + exp(-2.0 / pow(x2,1.5)) +
                         exp( -2.0 / pow(x3,1.25)));

    return f;
}

double xy (double * x, void * args){
    
    assert (args == NULL);
    double out =  log(x[0] * x[1]+2.0) / pow(x[0]+2,2.0) * exp(10.0*x[0]*x[1]);
    return out;
}

int main( int argc, char *argv[])
{

    if ((argc != 2)){
       printf("Correct function call = ./simlib <functionname>\n");
       printf("Options for function name include: \n");
       printf("\t \"otl\", \"borehole\", \"piston\", \"robotarm\", \"wingweight\", \"xy\", \n");
       printf("\t \"friedman\", \"gramlee09\", \"detpep10\", \"detpep10exp\" \n");
       return 0;
    }

    struct FunctionMonitor * fm = NULL;
    size_t dim;
    if (strcmp(argv[1], "otl") == 0){
        printf("Running OTL circuit function \n");
        dim = 6;
        fm = function_monitor_initnd(otlcircuit,NULL,dim,1000*dim);
    }
    else if (strcmp(argv[1], "borehole") == 0){
        printf("Running Borehole function \n");
        dim = 8;
        fm = function_monitor_initnd(borehole,NULL,dim,1000*dim);
    }
    else if (strcmp(argv[1], "piston") == 0){
        printf("Running Piston function \n");
        dim = 7;
        fm = function_monitor_initnd(piston,NULL,dim,1000*dim);
    }
    else if (strcmp(argv[1], "robotarm") == 0){
        printf("Running Piston function \n");
        dim = 8;
        fm = function_monitor_initnd(robotarm,NULL,dim,1000*dim);
    }
    else if (strcmp(argv[1], "wingweight") == 0){
        printf("Running Wing Weight function \n");
        dim = 10;
        fm = function_monitor_initnd(wingweight,NULL,dim,1000*dim);
    }
    else if (strcmp(argv[1], "friedman") == 0){
        printf("Running Friedman function \n");
        dim = 5;
        fm = function_monitor_initnd(friedman,NULL,dim,1000*dim);
    }
    else if (strcmp(argv[1], "gramlee09") == 0){
        printf("Running Gramacy & Lee 2009 function \n");
        dim = 6;
        fm = function_monitor_initnd(gramlee09,NULL,dim,1000*dim);
    }
    else if (strcmp(argv[1], "detpep10") == 0){
        printf("Running Dette & Peplyshev 2010 8 dimension function \n");
        dim = 8;
        fm = function_monitor_initnd(detpep10,NULL,dim,1000*dim);
    }
    else if (strcmp(argv[1], "detpep10exp") == 0){
        printf("Running Dette & Peplyshev 2010 3 dimension exponential function \n");
        dim = 3;
        fm = function_monitor_initnd(detpep10exp,NULL,dim,1000*dim);
    }

    else if (strcmp(argv[1], "xy") == 0){
        printf("Running XY function \n");
        dim = 2;
        fm = function_monitor_initnd(xy,NULL,dim,1000*dim);
    }
    else{
        printf("%s is not a valid function \n",argv[1]);
        return 0;
    }

    struct BoundingBox * bds = bounding_box_init(dim,-1.0,1.0);

    // One dimensional approximation arguments
    enum poly_type ptype = LEGENDRE;
    struct OpeAdaptOpts ao;
    ao.start_num = 7;
    ao.coeffs_check = 2;
    ao.tol = 1e-7;
    struct FtApproxArgs * fapp = ft_approx_args_createpoly(dim,&ptype,&ao);

    /*
    struct PwPolyAdaptOpts aopts;
    aopts.ptype = LEGENDRE;
    aopts.maxorder = 7;
    aopts.coeff_check = 2;
    aopts.epsilon = 1e-4;
    aopts.minsize = 1e-1;
    aopts.nregions = 5;
    aopts.pts = NULL;

    enum poly_type ptype = LEGENDRE;
    struct FtApproxArgs * fapp = 
       ft_approx_args_createpwpoly(dim,&ptype,&aopts);
   */

    // Initialize cross approximation options
    size_t init_ranks = 5;
    struct FtCrossArgs fca;
    ft_cross_args_init(&fca);
    fca.dim = dim;
    fca.ranks = calloc_size_t(dim+1);
    size_t ii;
    fca.ranks[0] = 1;
    for (ii=1;ii<dim;ii++){ fca.ranks[ii] = init_ranks; }
    fca.ranks[dim] = 1;
    fca.epsilon = 1e-3;
    fca.maxiter = 5;
    fca.epsround = 1e-5;
    fca.kickrank = 4;
    fca.maxiteradapt = 5;
    fca.verbose = 2;

    double * center = darray_val(dim,0.0);
    
    struct FunctionTrain * ft = 
        function_train_cross(function_monitor_eval, fm, bds, center,&fca, fapp);

    size_t nvals = nstored_hashtable_cp(fm->evals);
    printf("number of evaluations = %zu \n", nvals);
    printf("ranks are ");
    iprint_sz(dim+1,ft->ranks);
    size_t nrand = 10000;
    size_t jj;
    double * testpt = calloc_double(dim);
    double errnum = 0.0;
    double errden = 0.0;
    double integral = 0.0;;
    for (ii = 0; ii < nrand; ii++){
        for (jj = 0; jj < dim; jj++){
            testpt[jj] = randu()*2.0-1.0;
        }
        double valtrue = function_monitor_eval(testpt,fm);
        double val = function_train_eval(ft, testpt);
        double diff = valtrue - val;
        //printf("\n");
        //dprint(dim, testpt);
        //printf("err = %G \n", diff/valtrue);
        errnum += pow(diff,2);
        errden += pow(valtrue,2);
        integral += valtrue;
    }
    double err = errnum/errden;
    
    integral /= (double) nrand;
    integral *= pow(2.0,dim);
    printf("integral is approx %G\n", integral);
    
    double inttest = function_train_integrate(ft);
    printf("integral ft = %G\n",inttest);

    printf("normalization = %G \n",errden);
    printf("RMSE = %G\n", sqrt(errnum / (double) nrand));
    printf("Relative RMSE = %G \n", sqrt(err));

    function_monitor_free(fm); fm = NULL;
    bounding_box_free(bds); bds = NULL;
    ft_approx_args_free(fapp); fapp = NULL;
    free(fca.ranks); fca.ranks = NULL;
    free(center); center = NULL;
    function_train_free(ft); ft = NULL;
    free(testpt); testpt = NULL;

    return 0;
}
