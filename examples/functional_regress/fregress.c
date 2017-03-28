#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "c3.h"

#define dx 2
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
        quadratic(prob,inputs + ii*dx,y+ii*dy);
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

int main()
{

    prob_t prob;
    prob.ninput = dx;
    prob.noutput = dy;
    prob.x = linspace(0.0,1.0,dy);

    size_t ndata = 2;
    double * inputs = generate_inputs(ndata);
    double * outputs = generate_outputs(&prob,ndata,inputs);

    double * x = calloc_double(dx*dy * ndata);
    double * y = calloc_double(ndata*dy);
    
    size_t ntotdata = create_unified_data(ndata,inputs,prob.x,outputs,x,y);

    printf("ntotal data = %zu\n",ntotdata);
    
}
