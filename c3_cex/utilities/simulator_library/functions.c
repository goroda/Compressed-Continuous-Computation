// Copyright (c) 2016, Sandia Corporation. Under the terms of Contract
// DE-AC04-94AL85000, there is a non-exclusive license for use of this
// work by or on behalf of the U.S. Government. Export of this program
// may require a license from the United States Government

// This file is part of the Compressed Continuous Computation (C3) toolbox
// Author: Alex A. Gorodetsky 
// Contact: goroda@mit.edu

// All rights reserved.

// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice, 
//    this list of conditions and the following disclaimer in the documentation 
//    and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its contributors 
//    may be used to endorse or promote products derived from this software 
//    without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//Code

#ifndef M_PI
#define M_PI 3.1415926535897932384626433
#endif

#include <math.h>
#include <assert.h>

#include "functions.h"

int rosen_brock_func(size_t N, const double * x, double * out, void * arg)
{
    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        double f1 = 10.0 * (x[ii*2+1] - pow(x[ii*2+0],2));
        double f2 = (1.0 - x[ii*2+0]);
        out[ii] = pow(f1,2) + pow(f2,2);
    }

    return 0;
}

int sin_sum2d(size_t N, const double * x, double * out, void * arg)
{

    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = sin(x[ii*2+0] + x[ii*2+1]);
    }

    return 0;
}

int sin_sum5d(size_t N, const double * x, double * out, void * arg)
{

    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        double temp = 0.0;
        for (size_t jj = 0; jj < 5; jj++){
            temp += x[ii*5+jj];
        }
        out[ii] = sin(temp);
    }

    return 0;
}

int sin_sum10d(size_t N, const double * x, double * out, void * arg)
{

    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        double temp = 0.0;
        for (size_t jj = 0; jj < 10; jj++){
            temp += x[ii*10+jj];
        }
        out[ii] = sin(temp);
    }

    return 0;
}

int sin_sum50d(size_t N, const double * x, double * out, void * arg)
{

    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        double temp = 0.0;
        for (size_t jj = 0; jj < 50; jj++){
            temp += x[ii*50+jj];
        }
        out[ii] = sin(temp);
    }

    return 0;
}

int sin_sum100d(size_t N, const double * x, double * out, void * arg)
{

    (void)(arg);
    for (size_t ii = 0; ii < N; ii++){
        double temp = 0.0;
        for (size_t jj = 0; jj < 100; jj++){
            temp += x[ii*100+jj];
        }
        out[ii] = sin(temp);
    }

    return 0;
}

int otlcircuit(size_t N, const double * xin, double * out, void * args)
{
    assert (args == NULL );
    for (size_t ii = 0; ii < N; ii++){
        const double * x = xin + ii * 6;
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
        out[ii] = Vm;
    }

    return 0;
}

int borehole(size_t N, const double * in, double * out, void * args)
{
    assert (args == NULL);
    for (size_t ii = 0; ii < N; ii++){
        const double * x = in + ii * 8;

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
        out[ii] = f;
    }

    return 0;
}

int piston (size_t N, const double * in, double * out, void * args)
{
    assert (args == NULL);
    for (size_t ii = 0; ii < N; ii++){
        const double * x = in + ii * 7;
        
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

        out[ii] = C;
    }
    return 0;
}


int robotarm(size_t N, const double * in, double * out, void * args)
{
    assert (args == NULL);

    for (size_t ii = 0; ii < N; ii++){
        const double * x = in + 8 * ii;
        // 8 dimensions
    
        double u = 0.0;
        double v = 0.0;
        //size_t ii,jj;
     
        double theta1,theta2,theta3,theta4;
        double L1,L2,L3,L4;
        int natural_order = 0;
        if (natural_order == 1){
            theta1 = x[0];
            theta2 = x[1];
            theta3 = x[2];
            theta4 = x[3];
    
            L1 = x[4];
            L2 = x[5];
            L3 = x[6];
            L4 = x[7];
        }
        else{
            // daniele's optimal ordering
            /* theta1 = x[0]; */
            /* theta2 = x[4]; */
            /* theta3 = x[5]; */
            /* theta4 = x[3]; */
            /* L1 = x[2]; */
            /* L2 = x[1]; */
            /* L3 = x[6]; */
            /* L4 = x[7]; */

            theta1 = x[0];
            theta2 = x[1];
            L1 = x[2];
            L2 = x[3];
            L3 = x[4];
            L4 = x[5];
            theta3 = x[6];
            theta4 = x[7];

        }

        double t1 = theta1 * M_PI + M_PI;
        double t2 = t1 + (theta2*M_PI + M_PI);
        double t3 = t2 + (theta3*M_PI + M_PI);
        double t4 = t3 + (theta4*M_PI + M_PI);

        u = (L1*0.5+0.5) * cos(t1) + 
            (L2*0.5+0.5) * cos(t2) + 
            (L3*0.5+0.5) * cos(t3) + 
            (L4*0.5+0.5) * cos(t4);

        v = (L1*0.5+0.5) * sin(t1) + 
            (L2*0.5+0.5) * sin(t2) + 
            (L3*0.5+0.5) * sin(t3) + 
            (L4*0.5+0.5) * sin(t4);
    
        double f = sqrt ( u*u + v*v);
        out[ii] = f;
    }
    return 0;
}

int wingweight (size_t N, const double * in, double * out, void * args)
{
    assert (args == NULL);
    for (size_t ii = 0; ii < N; ii++){
        const double * x = in + 10 * ii;
            
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

        out[ii] = f;
    }
    return 0;
}


int friedman (size_t N, const double * in, double * out, void * args)
{
    assert (args == NULL);

    for (size_t ii = 0; ii < N; ii++){
    
        const double * x = in + ii * 5;
        
        // 5 dimension
    
        double f = 10.0 * sin(M_PI * (x[0]*0.5+ 0.5) * (x[1]*0.5 + 0.5));
        f += 20.0 * pow ((x[2]*0.5+0.5) - 0.5,2) + 10.0 * (x[3]*0.5+0.5) + 
            5.0 * (x[4]*0.5+0.5);

        out[ii] = f;
    }
    return 0;
}

int gramlee09 (size_t N, const double * in, double * out, void * args)
{
    //Gramacy & LEE (2009)
    assert (args == NULL);

    for (size_t ii = 0; ii < N; ii++){
        const double * x = in + ii * 6;
        // 6 dimension (last two variables not active)
    
        double f = exp(sin( pow( 0.9 * (x[0]*0.5+0.5 + 0.48),10.0 ) ));
        f += (x[1]*0.5+0.5) * (x[2]*0.5+0.5) + x[3] * 0.5 +0.5;

        out[ii] = f;
    }

    return 0;
}

int detpep10 (size_t N, const double * in, double * out, void * args)
{
    //Dette & Pepelyshev (2010)
    assert (args == NULL);

    for (size_t ll = 0; ll < N; ll++){
        const double * x = in + ll * 8;
        
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
        
        out[ll] = f;
    }
    return 0;
}


int detpep10exp (size_t N, const double * in, double * out, void * args)
{
    //Dette & Pepelyshev (2010) Exponential
    assert (args == NULL);

    for (size_t ii = 0; ii < N; ii++){
        const double * x = in + ii * 3;
        // 3 dimensions

        double x1 = x[0]*0.5+0.5;
        double x2 = x[1]*0.5+0.5;
        double x3 = x[2]*0.5+0.5;
        double f = 100.0 * ( exp( -2.0 / pow(x1,1.75)) + exp(-2.0 / pow(x2,1.5)) +
                             exp( -2.0 / pow(x3,1.25)));

        out[ii] = f;

    }
    return 0;
}

int xy (size_t N, const double * in, double * out, void * args){
    
    assert (args == NULL);
    for (size_t ii = 0; ii < N; ii++){
        const double * x = in + ii * 2;
        out[ii] =  log(x[0] * x[1]+2.0) / pow(x[0]+2,2.0) * exp(10.0*x[0]*x[1]);        
    }

    return 0;
}



///////////////////////////////////////////////////////////////////////////
// other functionsx
size_t function_get_dim(void * arg)
{
    struct Function * p = arg;
    return p->dim;
}

double function_get_lower(void * arg)
{
    struct Function * p = arg;
    return p->lower;
}

double function_get_upper(void * arg)
{
    struct Function * p = arg;
    return p->upper;
}

char * function_get_name(void * arg)
{
    struct Function * p = arg;
    return p->name;
}

void function_eval(size_t N, const double * x, double * out, void * arg)
{
    struct Function * p = arg;
    p->eval(N,x,out,NULL);
}


struct Function funcs[34];
size_t num_funcs;
void create_functions()
{
    num_funcs = 16;
    
    funcs[0].dim = 2;
    funcs[0].lower = -2.0;
    funcs[0].upper = 2.0;
    funcs[0].eval = rosen_brock_func;
    funcs[0].name    = "Rosenbrock";
    funcs[0].message = "Rosenbrock\n\t 100(x_2-x_1^2)^2 + (1-x_1)^2 \n\t 2 dimensions\n";
    
    funcs[1].dim = 2;
    funcs[1].lower = 0.0;
    funcs[1].upper = 1.0;
    funcs[1].eval = sin_sum2d;
    funcs[1].name = "SinSum2d";
    funcs[1].message = "SinSum\n\t sin(x_1+x_2) \n\t 2 dimensions \n\t [0,1] \n";

    funcs[2].dim = 5;
    funcs[2].lower = 0.0;
    funcs[2].upper = 1.0;
    funcs[2].eval = sin_sum5d;
    funcs[2].name = "SinSum5d";
    funcs[2].message = "SinSum\n\t sin(x_1+x_2+x_3+x_4+x_5) \n\t 5 dimensions \n\t [0,1] \n";

    funcs[3].dim = 10;
    funcs[3].lower = 0.0;
    funcs[3].upper = 1.0;
    funcs[3].eval = sin_sum10d;
    funcs[3].name = "SinSum10d";
    funcs[3].message = "SinSum\n\t sin(x_1+x_2+x_3+x_4+x_5+x_6+x_7+x_8+x_9+x_10) \n\t 10 dimensions \n\t [0,1]\n";

    // Next couple are from
    // http://www.sfu.ca/~ssurjano/emulat.html
    funcs[4].dim = 6;
    funcs[4].lower = -1.0;
    funcs[4].upper = 1.0;
    funcs[4].eval = otlcircuit;
    funcs[4].name = "OTLCircuit";
    funcs[4].message = "OTL Circuit\n\t 6 dimensions \n\t Normalized: [-1,1] \n";

    funcs[5].dim = 8;
    funcs[5].lower = -1.0;
    funcs[5].upper = 1.0;
    funcs[5].eval = borehole;
    funcs[5].name = "Borehole";
    funcs[5].message = "Borehole\n\t 8 dimensions \n\t Normalized: [-1,1] \n";

    funcs[6].dim = 7;
    funcs[6].lower = -1.0;
    funcs[6].upper = 1.0;
    funcs[6].eval = piston;
    funcs[6].name = "Piston";
    funcs[6].message = "Piston\n\t 7 dimensions \n\t Normalized: [-1,1] \n";

    funcs[7].dim = 8;
    funcs[7].lower = -1.0;
    funcs[7].upper = 1.0;
    funcs[7].eval = robotarm;
    funcs[7].name = "RobotArm";
    funcs[7].message = "Robot Arm\n\t 8 dimensions \n\t Normalized: [-1,1] \n";

    funcs[8].dim = 10;
    funcs[8].lower = -1.0;
    funcs[8].upper = 1.0;
    funcs[8].eval = wingweight;
    funcs[8].name = "WingWeight";
    funcs[8].message = "Wing Weight\n\t 10 dimensions \n\t Normalized: [-1,1] \n";

    funcs[9].dim = 5;
    funcs[9].lower = -1.0;
    funcs[9].upper = 1.0;
    funcs[9].eval = friedman;
    funcs[9].name = "Friedman";
    funcs[9].message = "Friedman\n\t 5 dimensions \n\t Normalized: [-1,1] \n";

    funcs[10].dim = 6;
    funcs[10].lower = -1.0;
    funcs[10].upper = 1.0;
    funcs[10].eval = gramlee09;
    funcs[10].name = "GramLee2009";
    funcs[10].message = "Gramacy and Lee (2009)\n\t 6 dimensions \n\t Normalized: [-1,1] \n\t Last two variables inactive \n";

    funcs[11].dim = 8;
    funcs[11].lower = -1.0;
    funcs[11].upper = 1.0;
    funcs[11].eval = detpep10;
    funcs[11].name = "DettePep2010";
    funcs[11].message = "Dette and Pepelyshev (2010)\n\t 8 dimensions \n\t Normalized: [-1,1] \n";

    funcs[12].dim = 3;
    funcs[12].lower = -1.0;
    funcs[12].upper = 1.0;
    funcs[12].eval = detpep10exp;
    funcs[12].name = "DettePep2010Exp";
    funcs[12].message = "Dette and Pepelyshev (2010) Exponential\n\t 3 dimensions \n\t Normalized: [-1,1] \n";

    funcs[13].dim = 2;
    funcs[13].lower = -1.0;
    funcs[13].upper = 1.0;
    funcs[13].eval = xy;
    funcs[13].name = "XY";
    funcs[13].message = "XY\n\t log(x_1x_2+2) / (x_1+2)^2 * exp(10x_1x_2) \n\t 5 dimensions \n\t Normalized: [-1,1] \n";

    funcs[14].dim = 100;
    funcs[14].lower = 0.0;
    funcs[14].upper = 1.0;
    funcs[14].eval = sin_sum100d;
    funcs[14].name = "SinSum100d";
    funcs[14].message = "SinSum\n\t sin(sum_{i=1}^100 x_i) \n\t 100 dimensions \n\t [0,1] \n";

    funcs[15].dim = 50;
    funcs[15].lower = 0.0;
    funcs[15].upper = 1.0;
    funcs[15].eval = sin_sum50d;
    funcs[15].name = "SinSum50d";
    funcs[15].message = "SinSum\n\t sin(sum_{i=1}^50 x_i) \n\t 100 dimensions \n\t [0,1] \n";

}
