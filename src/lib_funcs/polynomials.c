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

/** \file polynomials.c
 * Provides routines for manipulating orthogonal polynomials
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <assert.h>

//#define ZEROTHRESH 1e-20
#define ZEROTHRESH  1e0 * DBL_EPSILON
//#define ZEROTHRESH 0.0
//#define ZEROTHRESH  1e2 * DBL_EPSILON
//#define ZEROTHRESH  1e-12

#include "stringmanip.h"
#include "array.h"
#include "polynomials.h"
#include "quadrature.h"
#include "linalg.h"
#include "legtens.h"

// Recurrence relationship sequences
double zero_seq(size_t n){ return (0.0 + 0.0*n); }
double one_seq(size_t n) { return (1.0 + 0.0*n); }
double none_seq(size_t n) { return (-1.0 + 0.0*n); }
double two_seq(size_t n) { return (2.0 + 0.0*n); }
double n_seq(size_t n) { return ((double) n); }
double nn_seq (size_t n) { return -n_seq(n); }
double lega_seq (size_t n) { return ( (double)(2.0 * n -1.0) / (double) n);}
double legc_seq (size_t n) { return ( -((double)n - 1.0)/ (double) n );}

// Orthonormality functions
double chebortho(size_t n) {
    if (n == 0){
        return M_PI;
    }
    else{
        return M_PI/2.0;
    }
}

static const double legorthoarr[201] =
            {1.000000000000000e+00, 3.333333333333333e-01,2.000000000000000e-01,
            1.428571428571428e-01,1.111111111111111e-01,9.090909090909091e-02,7.692307692307693e-02,
            6.666666666666667e-02,5.882352941176471e-02,5.263157894736842e-02,4.761904761904762e-02,
            4.347826086956522e-02,4.000000000000000e-02,3.703703703703703e-02,3.448275862068965e-02,
            3.225806451612903e-02,3.030303030303030e-02,2.857142857142857e-02,2.702702702702703e-02,
            2.564102564102564e-02,2.439024390243903e-02,2.325581395348837e-02,2.222222222222222e-02,
            2.127659574468085e-02,2.040816326530612e-02,1.960784313725490e-02,1.886792452830189e-02,
            1.818181818181818e-02,1.754385964912281e-02,1.694915254237288e-02,1.639344262295082e-02
            ,1.587301587301587e-02,1.538461538461539e-02,1.492537313432836e-02,1.449275362318841e-02
            ,1.408450704225352e-02,1.369863013698630e-02,1.333333333333333e-02,1.298701298701299e-02
            ,1.265822784810127e-02,1.234567901234568e-02,1.204819277108434e-02,1.176470588235294e-02
            ,1.149425287356322e-02,1.123595505617977e-02,1.098901098901099e-02,1.075268817204301e-02
            ,1.052631578947368e-02,1.030927835051546e-02,1.010101010101010e-02,9.900990099009901e-03
            ,9.708737864077669e-03,9.523809523809525e-03,9.345794392523364e-03,9.174311926605505e-03
            ,9.009009009009009e-03,8.849557522123894e-03,8.695652173913044e-03,8.547008547008548e-03
            ,8.403361344537815e-03,8.264462809917356e-03,8.130081300813009e-03,8.000000000000000e-03
            ,7.874015748031496e-03,7.751937984496124e-03,7.633587786259542e-03,7.518796992481203e-03
            ,7.407407407407408e-03,7.299270072992700e-03,7.194244604316547e-03,7.092198581560284e-03
            ,6.993006993006993e-03,6.896551724137931e-03,6.802721088435374e-03,6.711409395973154e-03
            ,6.622516556291391e-03,6.535947712418301e-03,6.451612903225806e-03,6.369426751592357e-03
            ,6.289308176100629e-03,6.211180124223602e-03,6.134969325153374e-03,6.060606060606061e-03
            ,5.988023952095809e-03,5.917159763313609e-03,5.847953216374269e-03,5.780346820809248e-03
            ,5.714285714285714e-03,5.649717514124294e-03,5.586592178770950e-03,5.524861878453038e-03
            ,5.464480874316940e-03,5.405405405405406e-03,5.347593582887700e-03,5.291005291005291e-03
            ,5.235602094240838e-03,5.181347150259068e-03,5.128205128205128e-03,5.076142131979695e-03
            ,5.025125628140704e-03,4.975124378109453e-03,4.926108374384237e-03,4.878048780487805e-03
            ,4.830917874396135e-03,4.784688995215311e-03,4.739336492890996e-03,4.694835680751174e-03
            ,4.651162790697674e-03,4.608294930875576e-03,4.566210045662100e-03,4.524886877828055e-03
            ,4.484304932735426e-03,4.444444444444444e-03,4.405286343612335e-03,4.366812227074236e-03
            ,4.329004329004329e-03,4.291845493562232e-03,4.255319148936170e-03,4.219409282700422e-03
            ,4.184100418410041e-03,4.149377593360996e-03,4.115226337448560e-03,4.081632653061225e-03
            ,4.048582995951417e-03,4.016064257028112e-03,3.984063745019920e-03,3.952569169960474e-03
            ,3.921568627450980e-03,3.891050583657588e-03,3.861003861003861e-03,3.831417624521073e-03
            ,3.802281368821293e-03,3.773584905660377e-03,3.745318352059925e-03,3.717472118959108e-03
            ,3.690036900369004e-03,3.663003663003663e-03,3.636363636363636e-03,3.610108303249098e-03
            ,3.584229390681004e-03,3.558718861209964e-03,3.533568904593640e-03,3.508771929824561e-03
            ,3.484320557491289e-03,3.460207612456748e-03,3.436426116838488e-03,3.412969283276451e-03
            ,3.389830508474576e-03,3.367003367003367e-03,3.344481605351171e-03,3.322259136212625e-03
            ,3.300330033003300e-03,3.278688524590164e-03,3.257328990228013e-03,3.236245954692557e-03
            ,3.215434083601286e-03,3.194888178913738e-03,3.174603174603175e-03,3.154574132492113e-03
            ,3.134796238244514e-03,3.115264797507788e-03,3.095975232198143e-03,3.076923076923077e-03
            ,3.058103975535168e-03,3.039513677811550e-03,3.021148036253776e-03,3.003003003003003e-03
            ,2.985074626865672e-03,2.967359050445104e-03,2.949852507374631e-03,2.932551319648094e-03
            ,2.915451895043732e-03,2.898550724637681e-03,2.881844380403458e-03,2.865329512893983e-03
            ,2.849002849002849e-03,2.832861189801700e-03,2.816901408450704e-03,2.801120448179272e-03
            ,2.785515320334262e-03,2.770083102493075e-03,2.754820936639119e-03,2.739726027397260e-03
            ,2.724795640326975e-03,2.710027100271003e-03,2.695417789757413e-03,2.680965147453083e-03
            ,2.666666666666667e-03,2.652519893899204e-03,2.638522427440633e-03,2.624671916010499e-03
            ,2.610966057441253e-03,2.597402597402597e-03,2.583979328165375e-03,2.570694087403599e-03
            ,2.557544757033248e-03,2.544529262086514e-03,2.531645569620253e-03,2.518891687657431e-03
            ,2.506265664160401e-03,2.493765586034913e-03};

double legortho(size_t n){
    if (n < 201){
        return legorthoarr[n];
    }
    else{
       // printf("here?! n=%zu\n",n);
        return (1.0 / (2.0 * (double) n + 1.0));
    }
}

// Helper functions
double orth_poly_expansion_eval2(double x, void * p){
    
    struct OrthPolyExpansion * temp = p;
    return orth_poly_expansion_eval(temp,x);
}

double orth_poly_expansion_eval3(double x, void * p)
{
    struct OrthPolyExpansion ** temp = p;
    double out = orth_poly_expansion_eval(temp[0],x);
    out *= orth_poly_expansion_eval(temp[1],x);

    return out;
}

struct lin_func
{
    double slope;
    double offset;
};

double eval_lin_func(double x, void * args)
{
    struct lin_func * lf = args;
    double m = lf->slope;
    double b = lf->offset;
    return m*x + b;
}

struct quad_func
{
    double scale;
    double offset;
};

double eval_quad_func(double x, void * args)
{
    struct quad_func * qf = args;
    double m = qf->scale;
    double b = qf->offset;
    return m*(x-b)*(x-b);
}


/********************************************************//**
*   Initialize a standard basis polynomial
*
*   \param num_poly [in] - number of basis
*   \param lb [in] - lower bound
*   \param ub [in] - upper bound
*
*   \return  p - polynomial
*************************************************************/
struct StandardPoly * 
standard_poly_init(size_t num_poly, double lb, double ub){
    
    struct StandardPoly * p;
    if ( NULL == (p = malloc(sizeof(struct StandardPoly)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    p->ptype = STANDARD;
    p->num_poly = num_poly;
    p->lower_bound = lb;
    p->upper_bound = ub;
    p->coeff = calloc_double(num_poly);
    return p;
}

/********************************************************//**
*   Create the polynomial representing the derivative
*   of the standard polynomial
*
*   \param p [in] - polynomial
*
*   \return dp  - derivative polynomial
*************************************************************/
struct StandardPoly * 
standard_poly_deriv(struct StandardPoly * p){
    
    struct StandardPoly * dp;
    if ( NULL == (dp = malloc(sizeof(struct StandardPoly)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    dp->ptype = STANDARD;
    dp->lower_bound = p->lower_bound;
    dp->upper_bound = p->upper_bound;
    if (p->num_poly > 1){
        dp->num_poly = p->num_poly-1;
        dp->coeff = calloc_double(dp->num_poly);
        size_t ii;
        for (ii = 1; ii < p->num_poly; ii++){
            dp->coeff[ii-1] = p->coeff[ii] * (double) ii;
        }
    }
    else{
        dp->num_poly = 1;
        dp->coeff = calloc_double(dp->num_poly);
    }

    return dp;
}

/********************************************************//**
*   free memory allocated to a standard polynomial
*
*   \param p [inout] - polynomial structure 
*
*************************************************************/
void 
standard_poly_free(struct StandardPoly * p)
{
    free(p->coeff);
    free(p);
}

/********************************************************//**
*   Initialize a Chebyshev polynomial 
*
*   \return p - polynomial
*************************************************************/
struct OrthPoly * init_cheb_poly(){
    
    struct OrthPoly * p;
    if ( NULL == (p = malloc(sizeof(struct OrthPoly)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    p->ptype = CHEBYSHEV;
    p->an = &two_seq; 
    p->bn = &zero_seq;
    p->cn = &none_seq;
    
    p->lower = -1.0;
    p->upper = 1.0;

    p->const_term = 1.0;
    p->lin_coeff = 1.0;
    p->lin_const = 0.0;

    p->norm = &chebortho;

    return p;
}

/********************************************************//**
*   Initialize a Legendre polynomial 
*
*   \return p - polynomial
*************************************************************/
struct OrthPoly * init_leg_poly(){
    
    struct OrthPoly * p;
    if ( NULL == (p = malloc(sizeof(struct OrthPoly)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    p->ptype = LEGENDRE;
    p->an = &lega_seq; 
    p->bn = &zero_seq;
    p->cn = &legc_seq;
    
    p->lower = -1.0;
    p->upper = 1.0;

    p->const_term = 1.0;
    p->lin_coeff = 1.0;
    p->lin_const = 0.0;

    p->norm = &legortho;

    return p;
}

/********************************************************//**
*   free memory allocated to a polynomial
*
*   \param p [inout]- polynomial structure to free
*************************************************************/
void 
free_orth_poly(struct OrthPoly * p)
{
    free(p);
}

/********************************************************//**
*   serialize an orthonormal polynomial
*
*   \param p - polynomial structure to serialize
*
*   \return ser - serialize polynomial
*
*   \note 
*    This is actually pretty stupid because only need to
*    serialize the type. But hey. Its good practice.
*************************************************************/
unsigned char *
serialize_orth_poly(struct OrthPoly * p)
{
    
    /*
    char start[]= "type=";
    char * ser = NULL;
    concat_string_ow(&ser,start);
    
    char temp[3];
    snprintf(temp,2,"%d",p->ptype);
    concat_string_ow(&ser,temp);
    */
    
    unsigned char * ser = malloc(sizeof(int) * sizeof(unsigned char)) ;
    serialize_int(ser, p->ptype);
    return ser;
}

/********************************************************//**
*   deserialize an orthonormal polynomial
*
*   \param ser - string to deserialize
*
*   \return poly - orthonormal polynomial
*************************************************************/
struct OrthPoly *
deserialize_orth_poly(unsigned char * ser)
{

    struct OrthPoly * poly = NULL;
    
    int type;
    deserialize_int(ser, &type);
    if (type == 0) {
        poly = init_leg_poly();
    }
    else if (type == 1){
        poly = init_cheb_poly();
    }
    return poly;
}

/********************************************************//**
*   Convert an orthogonal family polynomial of order *n*
*   to a standard_polynomial
*
*   \param p [in] - polynomial
*   \param n [in] - polynomial order
*
*   \return sp - standard polynomial
*************************************************************/
struct StandardPoly * orth_to_standard_poly(struct OrthPoly * p, size_t n)
{
    struct StandardPoly * sp = standard_poly_init(n+1,p->lower,p->upper);
    size_t ii, jj;
    if (n == 0){
        sp->coeff[n] = p->const_term;
    }
    else if (n == 1){
        sp->coeff[0] = p->lin_const;
        sp->coeff[1] = p->lin_coeff;
    }
    else{
        
        double * a = calloc_double(n+1); //n-2 poly
        a[0] = p->const_term;
        double * b = calloc_double(n+1); // n- 1poly
        b[0] = p->lin_const;
        b[1] = p->lin_coeff;
        for (ii = 2; ii < n+1; ii++){
            sp->coeff[0] = p->bn(ii) * b[0] + p->cn(ii) * a[0];
            for (jj = 1; jj < ii-1; jj++){
                sp->coeff[jj] = p->an(ii)*b[jj-1] + p->bn(ii) * b[jj] + 
                                    p->cn(ii) * a[jj];
            }
            sp->coeff[ii-1] = p->an(ii)*b[ii-2] + p->bn(ii) * b[ii-1];
            sp->coeff[ii] = p->an(ii) * b[ii-1];

            memcpy(a, b, ii * sizeof(double));
            memcpy(b, sp->coeff, (ii+1) * sizeof(double));
        }
        
        free(a);
        free(b);
    }

    return sp;
}

/********************************************************//**
*   Evaluate an orthogonal polynomial with previous two
*   polynomial orders specified
*
*   \param rec [in] - orthogonal polynomial 
*   \param  p2 [in] - if evaluating polynomial P_n(x), then p2 is P_{n-2}(x)
*   \param p1 [in] - if evaluating polynomial P_n(x), then p1 is P_{n-1}(x)
*   \param n [in] - order
*   \param x [in] - location at which to evaluate
*
*   \return out - polynomial value
*************************************************************/
double 
eval_orth_poly_wp(const struct OrthPoly * rec, double p2, double p1, 
                    size_t n, double x)
{   

    double out;
    if (rec->ptype == LEGENDRE){
        double nt = (double) n;
        double a = (2.0*nt-1.0) / nt;
        double c = -(nt-1.0)/nt;
        out = a * x * p1 + c * p2;
    }
    else{
        out = (rec->an(n) * x + rec->bn(n)) * p1 + rec->cn(n) * p2;
    }
    return out;
}

/********************************************************//**
*   Evaluate the derivative of a legendre polynomial up to a certain
*   order
*
*   \param x [in] - location at which to evaluate
*   \param order [in] - maximum order;
*
*   \return out - derivatives
*************************************************************/
double * deriv_legen_upto(double x, size_t order){
    
    double * out = calloc_double(order+1);
    if (order == 0){
        return out;   
    }
    else if( fabs(x-1.0) <= DBL_EPSILON) {
        size_t ii;
        for (ii = 1; ii < order+1; ii++){
            out[ii] = (double) (order * (order+1)/2.0);
        }
    }
    else if (fabs(x-(-1.0)) <= DBL_EPSILON){
        size_t ii;
        for (ii = 1; ii < order+1; ii+=2){
            out[ii] = (double) (order * (order+1)/2.0);
        }
        for (ii = 2; ii < order+1; ii+=2){
            out[ii] = -(double) (order * (order+1)/2.0);
        }
    }
    else if (order == 1){
        struct OrthPoly * p = init_leg_poly();
        out[1] = x * orth_poly_eval(p,order,x) - orth_poly_eval(p,order-1,x);
        out[1] = (double) order * out[1] / ( x * x - 1.0);
        free_orth_poly(p);
    }
    else{
        struct OrthPoly * p = init_leg_poly();
        double eval0 = orth_poly_eval(p,0,x);
        double eval1 = orth_poly_eval(p,1,x);
        double evaltemp;

        out[1] = x * eval1 - eval0;
        //printf("out[1]=%G\n",out[1]);
        out[1] = 1.0  * out[1] / ( x * x - 1.0);
        
        size_t ii;
        for (ii = 2; ii < order+1; ii++){
            evaltemp = eval_orth_poly_wp(p, eval0, eval1, ii, x);
            eval0 = eval1;
            eval1 = evaltemp;
            out[ii] = x * eval1 - eval0;
            out[ii] = (double) ii * out[ii] / ( x * x - 1.0);
        }
        free_orth_poly(p);
    }

    return out;
}

/********************************************************//**
*   Evaluate the derivative of a legendre polynomial of a certain order
*
*   \param x [in] - location at which to evaluate
*   \param order [in] - order of the polynomial
*
*   \return out - derivative
*************************************************************/
double deriv_legen(double x, size_t order){
    
    if (order == 0){
        return 0.0;
    }
    double out;

    if ( fabs(x-1.0) <= DBL_EPSILON) {
        out = (double) (order * (order+1)/2.0);
    }
    else if (fabs(x-(-1.0)) <= DBL_EPSILON){
        if (order % 2){ // odd
            out = (double) (order * (order+1)/2.0);
        }
        else{
            out = -(double) (order * (order+1)/2.0);
        }
    }
    else{
        struct OrthPoly * p = init_leg_poly();
        out = x * orth_poly_eval(p,order,x) - orth_poly_eval(p,order-1,x);
        //printf("out in plain = %G\n",out);
        out = (double) order * out / ( x * x - 1.0);
        free_orth_poly(p);
    }
    return out;
}

/********************************************************//**
*   Evaluate the derivative of orthogonal polynomials
*   up to a certain order
*
*   \param ptype [in] - polynomial type
*   \param order [in] - order of the polynomial
*   \param x [in] - location at which to evaluate
*
*   \return out - orthonormal polynomial expansion 
*************************************************************/
double * orth_poly_deriv_upto(enum poly_type ptype, size_t order, double x)
{
    double * out = NULL;
    if (ptype == LEGENDRE){
        out = deriv_legen_upto(x,order);
    }
    else {
        fprintf(stderr,"Have not implemented orth_poly_deriv for %d\n",ptype);
        exit(1);
    }
    return out;

}

/********************************************************//**
*   Evaluate the derivative of an orthogonal polynomial
*
*   \param ptype [in] - polynomial type
*   \param order [in] - order of the polynomial
*   \param x [in] - location at which to evaluate
*
*   \return out - orthonormal polynomial expansion 
*************************************************************/
double orth_poly_deriv(enum poly_type ptype, size_t order, double x)
{
    double out = 0.0;
    if (ptype == LEGENDRE){
        out = deriv_legen(x,order);
    }
    else {
        fprintf(stderr,"Have not implemented orth_poly_deriv for %d\n",ptype);
        exit(1);
    }
    return out;
}

/********************************************************//**
*   Evaluate an orthogonal polynomial of a given order
*
*   \param rec [in] - orthogonal polynomial 
*   \param n [in] - order
*   \param x [in] - location at which to evaluate
*
*   \return out - polynomial value
*************************************************************/
double 
orth_poly_eval(const struct OrthPoly * rec, size_t n, double x)
{   
    if (n == 0){
        return rec->const_term;
    }
    else if (n == 1){
        return rec->lin_coeff * x + rec->lin_const;
    }
    else {
        double out = (rec->an(n)*x + rec->bn(n)) * orth_poly_eval(rec,n-1,x) +
                        rec->cn(n) * orth_poly_eval(rec,n-2,x);
        return out;
    }
}


/********************************************************//**
*   Initialize an expanion of a certain orthogonal polynomial family
*            
*   \param ptype [in] - type of polynomial
*   \param num_poly [in] - number of polynomials
*   \param lb [in] - lower bound
*   \param ub [in] - upper bound
*
*   \return p - orthogonal polynomial expansion
*************************************************************/
struct OrthPolyExpansion * 
orth_poly_expansion_init(enum poly_type ptype, size_t num_poly, 
                            double lb, double ub)
{

    struct OrthPolyExpansion * p;
    if ( NULL == (p = malloc(sizeof(struct OrthPolyExpansion)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    
    switch (ptype) {
        case LEGENDRE:
            p->p = init_leg_poly();
            break;
        case CHEBYSHEV:
            p->p = init_cheb_poly();
            break;
        case STANDARD:
            break;
        //default:
        //    fprintf(stderr, "Polynomial type does not exist: %d\n ", ptype);
    }

    p->num_poly = num_poly;
    p->nalloc = num_poly+OPECALLOC;
    p->coeff = calloc_double(p->nalloc);
    //p->coeff = calloc_double(num_poly);
    p->lower_bound = lb;
    p->upper_bound = ub;

    return p;
}

/********************************************************//**
*   Copy an orthogonal polynomial expansion
*            
*   \param pin [in] - polynomial to copy
*
*   \return p - orthogonal polynomial expansion
*************************************************************/
struct OrthPolyExpansion * 
orth_poly_expansion_copy(struct OrthPolyExpansion * pin)
{

    struct OrthPolyExpansion * p;
    if ( NULL == (p = malloc(sizeof(struct OrthPolyExpansion)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    
    switch (pin->p->ptype) {
        case LEGENDRE:
            p->p = init_leg_poly();
            break;
        case CHEBYSHEV:
            p->p = init_cheb_poly();
            break;
        case STANDARD:
            break;
        //default:
        //    fprintf(stderr, "Polynomial type does not exist: %d\n ", ptype);
    }

    p->num_poly = pin->num_poly;
    p->nalloc = pin->nalloc;
    //p->coeff = calloc_double(p->num_poly);
    p->coeff = calloc_double(pin->nalloc);
    memmove(p->coeff,pin->coeff, p->num_poly * sizeof(double));
    p->lower_bound = pin->lower_bound;
    p->upper_bound = pin->upper_bound;

    return p;
}


/********************************************************//**
*   Generate a constant orthonormal polynomial expansion
*
*   \param a [in] - value of the function
*   \param ptype [in] - type of polynomial
*   \param lb [in] - lower bound
*   \param ub [in] - upper bound
*
*   \return p - orthogonal polynomial expansion
*************************************************************/
struct OrthPolyExpansion * 
orth_poly_expansion_constant(double a, enum poly_type ptype, double lb, 
                                double ub)
{

    struct OrthPolyExpansion * p = orth_poly_expansion_init(ptype,1,lb,ub);
    p->coeff[0] = a/ p->p->const_term;

    return p;
}

/********************************************************//**
*   Generate a linear orthonormal polynomial expansion
*
*   \param a [in] - value of the slope function
*   \param offset [in] - offset
*   \param ptype [in] - type of polynomial
*   \param lb [in] - lower bound
*   \param ub [in] - upper bound
*
*   \return p - orthogonal polynomial expansion
*************************************************************/
struct OrthPolyExpansion * 
orth_poly_expansion_linear(double a, double offset, enum poly_type ptype, double lb, 
                                double ub)
{
    struct OrthPolyExpansion * p = 
            orth_poly_expansion_init(ptype, 2, lb, ub);
    if (ptype == LEGENDRE){
        double m = (p->p->upper - p->p->lower) / 
                    (p->upper_bound - p->lower_bound);
        double off = p->p->upper - m * p->upper_bound;
        //printf("lb=%G,ub=%G\n",lb,ub);
        //printf("mycalc = %G\n",a/m);
        //printf("mycalc[0] = %G\n",offset-a/m*off);
        p->coeff[1] = a/m;
        p->coeff[0] = offset-off*p->coeff[1];
    }
    else{
        struct lin_func lf;
        lf.slope = a;
        lf.offset = offset;
        orth_poly_expansion_approx(eval_lin_func, &lf, p);
        printf("a=%G,offset=%G,coeff[0]=%G,coeff[1]=%G\n",a,offset,p->coeff[0],p->coeff[1]);
        printf("\n");
   }

    return p;
}

/********************************************************//**
*   Generate a linear orthonormal polynomial expansion
    a * (x-offset)^2
*
*   \param a [in] - value of the slope function
*   \param offset [in] - offset
*   \param ptype [in] - type of polynomial
*   \param lb [in] - lower bound
*   \param ub [in] - upper bound
*
*   \return p - orthogonal polynomial expansion
*************************************************************/
struct OrthPolyExpansion * 
orth_poly_expansion_quadratic(double a, double offset, enum poly_type ptype, double lb, 
                                double ub)
{
    struct OrthPolyExpansion * p = 
            orth_poly_expansion_init(ptype, 3, lb, ub);

    if (ptype == LEGENDRE){
        double m = (p->p->upper - p->p->lower) / 
                    (p->upper_bound - p->lower_bound);
        double off = p->p->upper - m * p->upper_bound;
        p->coeff[2] = 2.0*a/3.0/m/m;
        p->coeff[1] = (-2.0 * a * offset - (3.0 *m * off * p->coeff[2]))/m;
        p->coeff[0] = (a*offset*offset - p->coeff[2]/2.0*(3*off*off-1.0) - p->coeff[1] * off);
    }
    else{
        struct quad_func qf;
        qf.scale = a;
        qf.offset = offset;
        orth_poly_expansion_approx(eval_quad_func, &qf, p);
    }

    return p;
}

/********************************************************//**
*   Generate a polynomial expansion with only the
*   *order* coefficient being nonzero
*
*   \param ptype [in] - type of polynomial
*   \param order [in] - order of the polynomial
*   \param lb [in] - lower bound
*   \param ub [in] - upper bound
*
*   \return p - orthogonal polynomial expansion
*
*   \return
*       ONLY WORKS FOR LEGENDRE!!!
*       Orthogonal polynomials on [-1, 1] with weight 1.0;
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_genorder(enum poly_type ptype, size_t order, double lb, double ub)
{
    //printf("whhhh\n");
    struct OrthPolyExpansion * p = 
            orth_poly_expansion_init(ptype, order+1, lb, ub);
    //printf("wherhhh\n");
    
    double m = (p->upper_bound - p->lower_bound) / 
                    (p->p->upper - p->p->lower);
    //printf("here \n");
    //printf("order = %zu\n",order);
    //printf("nalloc = %zu\n",p->nalloc);
    p->coeff[order] = 1.0 / sqrt(p->p->norm(order)) / sqrt(2.0) / sqrt(m);
    //printf("there \n");
    return p;
}


/********************************************************//**
*   Evaluate the derivative of an orthogonal polynomial expansion
*
*   \param x [in] - location at which to evaluate
*   \param args [in] - pointer to orth poly expansion
*
*   \return out - value of derivative
*************************************************************/
double orth_poly_expansion_deriv_eval(double x, void * args)
{
    struct OrthPolyExpansion * p = args;
    double out = 0.0;
    size_t ii;
    double m = (p->p->upper - p->p->lower) / (p->upper_bound - p->lower_bound);
    double off = p->p->upper - m * p->upper_bound;
    
    //printf("ok lets go!\n");
    double xnorm = m*x + off;
    //printf("xnorm = %G\n", xnorm);
    double * derivvals = orth_poly_deriv_upto(p->p->ptype,p->num_poly-1,xnorm);
    for (ii = 0; ii < p->num_poly; ii++){
        out += p->coeff[ii] * derivvals[ii] * m;
        //printf("out = %G\n",out);
    }
    //printf("x=%G, out = %G\n",xnorm,out);
    free(derivvals); derivvals = NULL;
    return out;
}

/********************************************************//**
*   Evaluate the derivative of an orth poly expansion
*
*   \param p [in] - orthogonal polynomial expansion
*   
*   \return out - orthonormal polynomial expansion 
*
*   \note
*       Could speed this up slightly by using partial sum
*       to keep track of sum of coefficients
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_deriv(struct OrthPolyExpansion * p)
{
    struct OrthPolyExpansion * out = NULL;
    if ( p == NULL ){
        return out;
    }
    if  (p->num_poly == 1){
        out = orth_poly_expansion_constant(0.0, p->p->ptype,
                                p->lower_bound, p->upper_bound);
    }
    else{
        switch (p->p->ptype) {
            case LEGENDRE:
                out = orth_poly_expansion_init(p->p->ptype,p->num_poly-1, 
                                           p->lower_bound, p->upper_bound);
                size_t ii,jj;
                double m = (p->p->upper-p->p->lower) / 
                            (p->upper_bound - p->lower_bound);
                
                for (ii = 0; ii < p->num_poly-1; ii++){ // loop over coefficients
                    for (jj = ii+1; jj < p->num_poly; jj+=2){
                        out->coeff[ii] += p->coeff[jj];
                    }
                    out->coeff[ii] *= (double) ( 2 * (ii) + 1) * m;
                    
                }
            //orth_poly_expansion_approx(orth_poly_expansion_deriv_eval, p, out);
                break;
            case CHEBYSHEV:
                break;
            case STANDARD:
                break;
        }
    }
    return out;
}

/********************************************************//**
*   free the memory of an orthonormal polynomial expansion
*
*   \param p [inout] - orthogonal polynomial expansion
*************************************************************/
void orth_poly_expansion_free(struct OrthPolyExpansion * p){
    free_orth_poly(p->p);
    free(p->coeff);
    free(p);
}

/********************************************************//**
*   Serialize orth_poly_expansion
*   
*   \param ser [in] - location to which to serialize
*   \param p [in] - polynomial
*   \param totSizeIn - if not null then only return total size of array without serialization! if NULL then serialiaze
*
*   \return ptr : pointer to end of serialization
*************************************************************/
unsigned char *
serialize_orth_poly_expansion(unsigned char * ser, 
        struct OrthPolyExpansion * p,
        size_t * totSizeIn)
{
    // order is  ptype->lower_bound->upper_bound->orth_poly->coeff
    
    
    size_t totsize = sizeof(int) + 2*sizeof(double) + 
                       p->num_poly * sizeof(double) + sizeof(size_t);
    if (totSizeIn != NULL){
        *totSizeIn = totsize;
        return ser;
    }
    unsigned char * ptr = serialize_int(ser, p->p->ptype);
    ptr = serialize_double(ptr, p->lower_bound);
    ptr = serialize_double(ptr, p->upper_bound);
    ptr = serialize_doublep(ptr, p->coeff, p->num_poly);
    return ptr;
}

/********************************************************//**
*   Deserialize orth_poly_expansion
*
*   \param ser [in] - input string
*   \param poly [inout]: poly expansion
*
*   \return ptr - ser + number of bytes of poly expansion
*************************************************************/
unsigned char * 
deserialize_orth_poly_expansion(unsigned char * ser, 
        struct OrthPolyExpansion ** poly)
{
    
    size_t num_poly = 0;
    //size_t npoly_check = 0;
    double lower_bound = 0;
    double upper_bound = 0;
    double * coeff = NULL;
    struct OrthPoly * p = NULL;
    
    // order is  ptype->lower_bound->upper_bound->orth_poly->coeff
    p = deserialize_orth_poly(ser);
    unsigned char * ptr = ser + sizeof(int);
    ptr = deserialize_double(ptr,&lower_bound);
    ptr = deserialize_double(ptr,&upper_bound);
    ptr = deserialize_doublep(ptr, &coeff, &num_poly);
    
    if ( NULL == (*poly = malloc(sizeof(struct OrthPolyExpansion)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    (*poly)->num_poly = num_poly;
    (*poly)->lower_bound = lower_bound;
    (*poly)->upper_bound = upper_bound;
    (*poly)->coeff = coeff;
    (*poly)->nalloc = OPECALLOC;
    (*poly)->p = p;
    
    return ptr;

    
    /*
    char * param = NULL;
    char * val = NULL;
    while ((ser != NULL) && (strcmp(ser,"") != 0) ){
        param = bite_string(ser,'=');
        //printf("on param=%s\n",param);
        if (strcmp(param,"num_poly") == 0){
            val = bite_string(ser,',');
            num_poly = strtoul(val,NULL,10);
        }
        else if (strcmp(param,"lower_bound")==0){
            val = bite_string(ser,',');
            lower_bound = deserialize_double_from_text(val);
        }
        else if (strcmp(param,"upper_bound")==0){
            val = bite_string(ser,',');
            upper_bound = deserialize_double_from_text(val);
        }
        else if (strcmp(param,"coeff")==0){
            val = bite_string(ser,'{');free(val);val=NULL;
            val = bite_string(ser,'}');
            coeff = deserialize_darray_from_text(val,&npoly_check);
            free(val);val=NULL;
            val = bite_string(ser,',');
        }
        else if (strcmp(param,"orth_poly")==0){
            val = bite_string(ser,'{');free(val);val=NULL;
            val = bite_string(ser,'}');
            p = deserialize_orth_poly(val);
        }
        free(val);val=NULL;
        free(param);param=NULL;
    }
    
    assert((coeff != NULL) && (p != NULL) && (npoly_check==num_poly));

    if ( NULL == (poly = malloc(sizeof(struct OrthPolyExpansion)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    poly->num_poly = num_poly;
    poly->lower_bound = lower_bound;
    poly->upper_bound = upper_bound;
    poly->coeff = coeff;
    poly->p = p;

    return poly;
    */
}


/********************************************************//**
*   Convert an orthogonal polynomial expansion to a standard_polynomial
*
*   \param p [in] - polynomial
*
*   \return sp - standard polynomial
*************************************************************/
struct StandardPoly * 
orth_poly_expansion_to_standard_poly(struct OrthPolyExpansion * p)
{
    struct StandardPoly * sp = 
        standard_poly_init(p->num_poly,p->lower_bound,p->upper_bound);
    
    double m = (p->p->upper - p->p->lower) / (p->upper_bound - p->lower_bound);
    double off = p->p->upper - m * p->upper_bound;

    size_t ii, jj;
    size_t n = p->num_poly-1;

    sp->coeff[0] = p->coeff[0]*p->p->const_term;

    if (n > 0){
        sp->coeff[0]+=p->coeff[1] * (p->p->lin_const + p->p->lin_coeff * off);
        sp->coeff[1]+=p->coeff[1] * p->p->lin_coeff * m;
    }
    if (n > 1){
        
        double * a = calloc_double(n+1); //n-2 poly
        a[0] = p->p->const_term;
        double * b = calloc_double(n+1); // n- 1poly
        double * c = calloc_double(n+1); // n- 1poly
        b[0] = p->p->lin_const + p->p->lin_coeff * off;
        b[1] = p->p->lin_coeff * m;
        for (ii = 2; ii < n+1; ii++){ // starting at the order 2 polynomial
            c[0] = (p->p->bn(ii) + p->p->an(ii)*off) * b[0] + 
                                                        p->p->cn(ii) * a[0];
            sp->coeff[0] += p->coeff[ii] * c[0];
            for (jj = 1; jj < ii-1; jj++){
                c[jj] = (p->p->an(ii) * m) * b[jj-1] + 
                        (p->p->bn(ii) + p->p->an(ii) * off) * b[jj] + 
                        p->p->cn(ii) * a[jj];
                sp->coeff[jj] += p->coeff[ii] * c[jj];
            }
            c[ii-1] = (p->p->an(ii) * m) * b[ii-2] + 
                            (p->p->bn(ii) + p->p->an(ii) * off) * b[ii-1];
            c[ii] = (p->p->an(ii) * m) * b[ii-1];
            
            sp->coeff[ii-1] += p->coeff[ii] * c[ii-1];
            sp->coeff[ii] += p->coeff[ii] * c[ii];

            memcpy(a, b, ii * sizeof(double));
            memcpy(b, c, (ii+1) * sizeof(double));
        }
        
        free(a);
        free(b);
        free(c);
    }

    // Need to do something with lower and upper bounds!!
    return sp;
}

/********************************************************//**
*   Evaluate a legendre polynomial expansion 
*
*   \param poly [in] - polynomial expansion
*   \param x [in] - location at which to evaluate
*
*   \return out - polynomial value
*************************************************************/
double legendre_poly_expansion_eval(struct OrthPolyExpansion * poly, double x)
{
    double out = 0.0;
    double p [2];
    double pnew;
    
    double m = (poly->p->upper - poly->p->lower) / (poly->upper_bound- poly->lower_bound);
    double off = poly->p->upper - m * poly->upper_bound;
    double x_norm =  m * x + off;
    
    size_t iter = 0;
    p[0] = 1.0;
    out += p[0] * poly->coeff[iter];
    iter++;
    if (poly->num_poly > 1){
        p[1] = x_norm;
        out += p[1] * poly->coeff[iter];
        iter++;
    }   
    for (iter = 2; iter < poly->num_poly; iter++){
        
        pnew = (double) (2*iter-1) * x_norm * p[1] - (double)(iter-1) * p[0];
        pnew /= (double) iter;
        out += poly->coeff[iter] * pnew;
        p[0] = p[1];
        p[1] = pnew;
    }
    return out;
}

/********************************************************//**
*   Evaluate a polynomial expansion consisting of sequentially increasing 
*   order polynomials from the same family.
*
*   \param poly [in] - polynomial expansion
*   \param x [in] - location at which to evaluate
*
*   \return out - polynomial value
*************************************************************/
double orth_poly_expansion_eval(struct OrthPolyExpansion * poly, double x)
{
    double out = 0.0;
    if (poly->p->ptype == LEGENDRE){
        out = legendre_poly_expansion_eval(poly,x);
    }
    else{
        size_t iter = 0;
        double p [2];
        double pnew;
        
        double m = (poly->p->upper - poly->p->lower) / 
                        (poly->upper_bound- poly->lower_bound);
        double off = poly->p->upper - m *poly->upper_bound;
        double x_normalized =  m *x + off;

        p[0] = poly->p->const_term;
        out += p[0] * poly->coeff[iter];
        iter++;
        if (poly->num_poly > 1){
            p[1] = poly->p->lin_const + poly->p->lin_coeff * x_normalized;
            out += p[1] * poly->coeff[iter];
            iter++;
        }
        for (iter = 2; iter < poly->num_poly; iter++){
            pnew = eval_orth_poly_wp(poly->p, p[0], p[1], iter, x_normalized);
            out += poly->coeff[iter] * pnew;
            p[0] = p[1];
            p[1] = pnew;
        }
    }
    return out;
}

/********************************************************//**
*  Round an orthogonal polynomial expansion
*
*  \param p [inout] - orthogonal polynomial expansion
*
*  \note
*      (UNTESTED, use with care!!!! 
*************************************************************/
void orth_poly_expansion_round(struct OrthPolyExpansion ** p)
{   
    if (0 == 0){
        double thresh = ZEROTHRESH;
        //printf("thresh = %G\n",thresh);
        size_t jj = 0;
        //
        int allzero = 1;
	    for (jj = 0; jj < (*p)->num_poly;jj++){
	       if (fabs((*p)->coeff[jj]) < thresh){
               (*p)->coeff[jj] = 0.0;
           }
           else{
               allzero = 0;
           }
	    }
        if (allzero == 1){
            (*p)->num_poly = 1;
        }
        else {
            jj = 0;
            size_t end = (*p)->num_poly;
            if ((*p)->num_poly > 2){
                while (fabs((*p)->coeff[end-1]) < thresh){
                    end-=1;
                    if (end == 0){
                        break;
                    }
                }
                
                if (end > 0){
                    //printf("SHOULD NOT BE HERE\n");
                    size_t num_poly = end;
                    //
                    //double * new_coeff = calloc_double(num_poly);
                    //for (jj = 0; jj < num_poly; jj++){
                    //    new_coeff[jj] = (*p)->coeff[jj];
                   // }
                    //free((*p)->coeff); (*p)->coeff=NULL;
                    //(*p)->coeff = new_coeff;
                    (*p)->num_poly = num_poly;
                }
            }
        }
        //orth_poly_expansion_roundt(p,thresh);
    }
}

/********************************************************//**
*  Round an orthogonal polynomial expansion to a threshold
*
*  \param p [inout] - orthogonal polynomial expansion
*  \param thresh [in] - threshold (relative) to round to
*
*  \note
*      (UNTESTED, use with care!!!! 
*************************************************************/
void orth_poly_expansion_roundt(struct OrthPolyExpansion ** p, double thresh)
{   
    
    size_t jj = 0;
    double sum = 0.0;
	for (jj = 0; jj < (*p)->num_poly;jj++){
        sum += pow((*p)->coeff[jj],2);
	}
    size_t keep = (*p)->num_poly;
    if (sum <= thresh){
        keep = 1;
    }
    else{
        double sumrun = 0.0;
        for (jj = 0; jj < (*p)->num_poly; jj++){
            sumrun += pow((*p)->coeff[jj],2);
            if ( (sumrun / sum) > (1.0-thresh)){
                keep = jj+1;
                break;
            }
        }
    }
    //dprint((*p)->num_poly, (*p)->coeff);
    //printf("number keep = %zu\n",keep);
    //printf("tolerance = %G\n",thresh);
    double * new_coeff = calloc_double(keep);
    memmove(new_coeff,(*p)->coeff, keep * sizeof(double));
    free((*p)->coeff);
    (*p)->num_poly = keep;
    (*p)->nalloc = OPECALLOC;
    (*p)->coeff = new_coeff;
}



/********************************************************//**
*  Approximate a function with an orthogonal polynomial
*  series with a fixed number of basis
*
*  \param A [in] - function to approximate
*  \param args [in] - arguments to function
*  \param poly [inout] - orthogonal polynomial expansion
*
*  \note
*       Wont work for polynomial expansion with only the constant 
*       term.
*************************************************************/
void
orth_poly_expansion_approx(double (*A)(double,void *), void *args, 
            struct OrthPolyExpansion * poly)
{
    size_t ii, jj;
    double p[2];
    double pnew;


    double m = (poly->upper_bound - poly->lower_bound) / 
                (poly->p->upper - poly->p->lower);
    double off = poly->upper_bound - m * poly->p->upper;


    double * fvals = NULL;
    double * pt_un = NULL; // unormalized point
    double * pt = NULL;
    double * wt = NULL; 

    size_t nquad = poly->num_poly+1;

    switch (poly->p->ptype) {
        case CHEBYSHEV:
            pt = calloc_double(nquad);
            wt = calloc_double(nquad);
            cheb_gauss(poly->num_poly,pt,wt);
            break;
        case LEGENDRE:
            //nquad = poly->num_poly*2.0-1.0;//*2.0;
            pt = calloc_double(nquad);
            wt = calloc_double(nquad);
            gauss_legendre(poly->num_poly,pt,wt);
            //clenshaw_curtis(nquad,pt,wt);
            //for (ii = 0; ii < nquad; ii++){wt[ii] *= 0.5;}
            break;
        case STANDARD:
            fprintf(stderr, "Cannot call orth_poly_expansion_approx for STANDARD type\n");
            break;
        //default:
        //    fprintf(stderr, "Polynomial type does not exist: %d\n ", 
        //            poly->p->ptype);
    }
    
    fvals = calloc_double(nquad);
    pt_un = calloc_double(nquad);
    for (ii = 0; ii < nquad; ii++){
        pt_un[ii] =  m * pt[ii] + off;
        fvals[ii] = A(pt_un[ii],args)  * wt[ii];
    }
    

    if (poly->num_poly > 1){
        for (ii = 0; ii < nquad; ii++){ // loop over all points
            p[0] = poly->p->const_term;
            poly->coeff[0] += fvals[ii] * poly->p->const_term;
            
            p[1] = poly->p->lin_const + poly->p->lin_coeff * pt[ii];
            poly->coeff[1] += fvals[ii] * p[1] ;
            // loop over all coefficients
            for (jj = 2; jj < poly->num_poly; jj++){ 
                pnew = eval_orth_poly_wp(poly->p, p[0], p[1], jj, pt[ii]);
                poly->coeff[jj] += fvals[ii] * pnew;
                p[0] = p[1];
                p[1] = pnew;
            }
        }

        for (ii = 0; ii < poly->num_poly; ii++){
            poly->coeff[ii] /= poly->p->norm(ii);
        }

    }
    else{
        for (ii = 0; ii < nquad; ii++){
            poly->coeff[0] += fvals[ii] *poly->p->const_term;
        }
        poly->coeff[0] /= poly->p->norm(0);
    }
    free(fvals);
    free(wt);
    free(pt);
    free(pt_un);
    
}

/********************************************************//**
*   Create an approximation adaptively
*
*   \param A [in] - function to project
*   \param args [in] - arguments to function
*   \param ptype [in] - polynomial type
*   \param lower [in] - lower bound of input
*   \param upper [in] - upper bound of input
*   \param aoptsin [in] - approximation options
*   
*   \return poly
*
*   \note 
*       Follows general scheme that trefethan outlines about 
*       Chebfun in his book Approximation Theory and practice
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_approx_adapt(double (*A)(double,void *), void * args, 
                        enum poly_type ptype, double lower, double upper,
                        struct OpeAdaptOpts * aoptsin)
{
    int default_opts = 0;
    struct OpeAdaptOpts * aopts;
    if (aoptsin == NULL){
        if ( NULL == (aopts = malloc(sizeof(struct OpeAdaptOpts)))){
            fprintf(stderr, "failed to allocate memory for poly exp.\n");
            exit(1);
        }
        aopts->start_num = 8;
        aopts->coeffs_check = 2;
        aopts->tol = 1e-10;
        //aopts->tol = 1e-1;  
        default_opts = 1;
    }
    else{
        aopts = aoptsin;
    }
    size_t ii;
    size_t N = aopts->start_num;
    
    //printf("pre ptype=%d, N=%zu, lower=%G, upper=%G \n",ptype,N,lower,upper);
    if ((int)ptype > 10){
        printf("Warning: for some reason ptype is\n");
        printf("specified to be %d, reverting to 0\n",ptype);
        ptype = 0;
    }
    struct OrthPolyExpansion * poly = orth_poly_expansion_init(ptype,
                                        N, lower, upper);
    
    //printf("start ptype=%d, N=%zu, lower=%G, upper=%G \n",ptype,N,lower,upper);
    orth_poly_expansion_approx(A,args, poly);
    //printf("got first one\n");
    
    size_t coeffs_too_big = 0;
    for (ii = 0; ii < aopts->coeffs_check; ii++){
        if (fabs(poly->coeff[N-1-ii]) > aopts->tol){
            coeffs_too_big = 1;
            break;
        }
    }
    //printf("TOL SPECIFIED IS %G\n",aopts->tol);
    while (coeffs_too_big == 1){
        coeffs_too_big = 0;
	
        free(poly->coeff);
        N = N * 2 - 1; // for nested cc
        //N = N * 2 + 1; // 
        //N = N + 5;
        poly->num_poly = N;
        poly->coeff = calloc_double(N);
        //printf("Number of coefficients to check = %zu\n",aopts->coeffs_check);
        orth_poly_expansion_approx(A, args, poly);
	    double sum_coeff_squared = 0.0;
        for (ii = 0; ii < N; ii++){ 
            sum_coeff_squared += pow(poly->coeff[ii],2); 
        }
        sum_coeff_squared = fmax(sum_coeff_squared,ZEROTHRESH);
        //sum_coeff_squared = 1.0;
        for (ii = 0; ii < aopts->coeffs_check; ii++){
            //printf("aopts->tol=%3.15G last coefficients %3.15G\n",
            //        aopts->tol * sum_coeff_squared,
           	//	  poly->coeff[N-1-ii]);
            if (fabs(poly->coeff[N-1-ii]) > (aopts->tol * sum_coeff_squared) ){
                coeffs_too_big = 1;
                break;
            }
        }
        if (N > 100){
            //printf("Warning: num of poly is %zu: last coeff = %G \n",N,poly->coeff[N-1]);
            //printf("tolerance is %3.15G\n", aopts->tol * sum_coeff_squared);
            //printf("Considering using piecewise polynomials\n");
            /*
            printf("first 5 coeffs\n");

            size_t ll;
            for (ll = 0; ll<5;ll++){
                printf("%3.10G ",poly->coeff[ll]);
            }
            printf("\n");

            printf("Last 10 coeffs\n");
            for (ll = 0; ll<10;ll++){
                printf("%3.10G ",poly->coeff[N-10+ll]);
            }
            printf("\n");
            */
            coeffs_too_big = 0;
        }

    }
    
    orth_poly_expansion_round(&poly);

    // verify
    double pt = (upper - lower)*randu() + lower;
    double val_true = A(pt,args);
    double val_test = orth_poly_expansion_eval(poly,pt);
    double diff = val_true-val_test;
    double err = fabs(diff);
    if (fabs(val_true) > 1.0){
    //if (fabs(val_true) > ZEROTHRESH){
        err /= fabs(val_true);
    }
    if (err > 100.0*aopts->tol){
        //fprintf(stderr, "Approximating at point %G in (%3.15G,%3.15G)\n",pt,lower,upper);
        //fprintf(stderr, "leads to error %G, while tol is %G \n",err,aopts->tol);
        //fprintf(stderr, "actual value is %G \n",val_true);
        //fprintf(stderr, "predicted value is %3.15G \n",val_test);
        //fprintf(stderr, "%zu N coeffs, last coeffs are %3.15G,%3.15G \n",N,poly->coeff[N-2],poly->coeff[N-1]);
        //exit(1);
    }

    if (default_opts == 1){
        free(aopts);
    }
    return poly;
}

/********************************************************//**
*   Generate an orthonormal polynomial with pseudorandom coefficients
*   between [-1,1]
*
*   \param ptype [in] - polynomial type
*   \param maxorder [in] - maximum order of the polynomial
*   \param lower [in] - lower bound of input
*   \param upper [in] - upper bound of input
*
*   \return poly
*************************************************************/
struct OrthPolyExpansion * 
orth_poly_expansion_randu(enum poly_type ptype, size_t maxorder, double lower, double upper)
{
    struct OrthPolyExpansion * poly = orth_poly_expansion_init(ptype,
                                        maxorder+1, lower, upper);
    size_t ii;
    for (ii = 0; ii < poly->num_poly; ii++){
        poly->coeff[ii] = randu()*2.0-1.0;
    }
    return poly;
}

/********************************************************//**
*   Integrate a Chebyshev approximation
*
*   \param poly [in] - polynomial to integrate
*
*   \return out - integral of approximation
*************************************************************/
double
cheb_integrate2(struct OrthPolyExpansion * poly)
{
    size_t ii;
    double out = 0.0;
    double m = (poly->upper_bound - poly->lower_bound) / 
                    (poly->p->upper - poly->p->lower);
    for (ii = 0; ii < poly->num_poly; ii+=2){
        out += poly->coeff[ii] * 2.0 / (1.0 - (double) (ii*ii));
    }
    out = out * m;
    return out;
}

/********************************************************//**
*   Integrate a Legendre approximation
*
*   \param  poly [in] - polynomial to integrate
*
*   \return out - integral of approximation
*************************************************************/
double
legendre_integrate(struct OrthPolyExpansion * poly)
{
    double out = 0.0;
    double m = (poly->upper_bound - poly->lower_bound) / 
                    (poly->p->upper - poly->p->lower);

    out = poly->coeff[0] * 2.0;
    out = out * m;
    return out;
}

/********************************************************//**
*   Compute the product of two polynomial expansion
*
*   \param a [in] - first polynomial
*   \param b [in] - second polynomial
*
*   \return c - polynomial expansion
*
*   \note 
*        Computes c(x) = a(x)b(x) where c is same form as a
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_prod(struct OrthPolyExpansion * a,
                         struct OrthPolyExpansion * b)
{

    struct OrthPolyExpansion * c = NULL;
    double lb = a->lower_bound;
    double ub = a->upper_bound;

    enum poly_type p = a->p->ptype;
    if ( (p == LEGENDRE) && (a->num_poly < 25) && (b->num_poly < 25)){
        //printf("in special prod\n");
        //double lb = a->lower_bound;
        //double ub = b->upper_bound;
            
        size_t ii,jj;
        c = orth_poly_expansion_init(p, a->num_poly + b->num_poly+1, lb, ub);
        double * allprods = calloc_double(a->num_poly * b->num_poly);
        for (ii = 0; ii < a->num_poly; ii++){
            for (jj = 0; jj < b->num_poly; jj++){
                allprods[jj + ii * b->num_poly] = a->coeff[ii] * b->coeff[jj];
            }
        }
        
        //printf("A = \n");
        //print_orth_poly_expansion(a,1,NULL);

        //printf("B = \n");
        //print_orth_poly_expansion(b,1,NULL);

        //dprint2d_col(b->num_poly, a->num_poly, allprods);

        size_t kk;
        for (kk = 0; kk < c->num_poly; kk++){
            for (ii = 0; ii < a->num_poly; ii++){
                for (jj = 0; jj < b->num_poly; jj++){
                    c->coeff[kk] +=  lpolycoeffs[ii+jj*50+kk*2500] * 
                                        allprods[jj+ii*b->num_poly];
                }
            }
            //printf("c coeff[%zu]=%G\n",kk,c->coeff[kk]);
        }
        orth_poly_expansion_round(&c);
        free(allprods); allprods=NULL;
    }
    else{
        struct OrthPolyExpansion * comb[2];
        comb[0] = a;
        comb[1] = b;
        
        double norma = 0.0, normb = 0.0;
        size_t ii;
        for (ii = 0; ii < a->num_poly; ii++){
            norma += pow(a->coeff[ii],2);
        }
        for (ii = 0; ii < b->num_poly; ii++){
            normb += pow(b->coeff[ii],2);
        }
        
        if ( (norma < ZEROTHRESH) || (normb < ZEROTHRESH) ){
            //printf("in here \n");
            c = orth_poly_expansion_constant(0.0,a->p->ptype,lb,ub);
        }
        else{
            //printf(" total order of product = %zu\n",a->num_poly+b->num_poly);
            c = orth_poly_expansion_init(p, a->num_poly + b->num_poly+1, lb, ub);
            orth_poly_expansion_approx(&orth_poly_expansion_eval3,comb,c);
            orth_poly_expansion_round(&c);
        }
    }
    
    //*
    //printf("compute product\n");
    //struct OpeAdaptOpts ao;
    //ao.start_num = 3;
    //ao.coeffs_check = 2;
    //ao.tol = 1e-13;
    //c = orth_poly_expansion_approx_adapt(&orth_poly_expansion_eval3,comb, 
    //                    p, lb, ub, &ao);
    
    //orth_poly_expansion_round(&c);
    //printf("done\n");
    //*/
    return c;
}

/********************************************************//**
*   Compute the sum of the product between the functions in two arraysarrays
*
*   \param n [in] - number of functions
*   \param lda [in] - stride of first array
*   \param a [in] - array of orthonormal polynomial expansions
*   \param ldb [in] - stride of second array
*   \param b [in] - array of orthonormal polynomial expansions
*
*   \return c - polynomial expansion
*
*   \note 
*       If both arrays do not consist of only LEGENDRE polynomials
*       return NULL. All the functions need to have the same lower 
*       and upper bounds
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_sum_prod(size_t n, size_t lda, 
        struct OrthPolyExpansion ** a, size_t ldb,
        struct OrthPolyExpansion ** b)
{

    struct OrthPolyExpansion * c = NULL;
    double lb = a[0]->lower_bound;
    double ub = a[0]->upper_bound;

    size_t ii;
    size_t maxordera = 0;
    size_t maxorderb = 0;
    size_t maxorder = 0;
    //int legen = 1;
    for (ii = 0; ii < n; ii++){
        if ((a[ii*lda]->p->ptype != LEGENDRE) || (b[ii*ldb]->p->ptype != LEGENDRE)){
            //legen = 0;
            return c; // cant do it
        }
        size_t neworder = a[ii*lda]->num_poly + b[ii*ldb]->num_poly;
        if (neworder > maxorder){
            maxorder = neworder;
        }
        if (a[ii*lda]->num_poly > maxordera){
            maxordera = a[ii*lda]->num_poly;
        }
        if (b[ii*ldb]->num_poly > maxorderb){
            maxorderb = b[ii*ldb]->num_poly;
        }
    }
    if (maxorder > 50){
        fprintf(stderr, "Cant multiply functions of greater than 25 together. Increase size of legtenscoeffs\n");
        exit(1);
    }
    
    enum poly_type p = LEGENDRE;
    c = orth_poly_expansion_init(p, maxorder, lb, ub);
    size_t kk,jj,ll;
    double * allprods = calloc_double( maxorderb * maxordera);
    for (kk = 0; kk < n; kk++){
        for (ii = 0; ii < a[kk*lda]->num_poly; ii++){
            for (jj = 0; jj < b[kk*ldb]->num_poly; jj++){
                allprods[jj + ii * maxorderb] += 
                        a[kk*lda]->coeff[ii] * b[kk*ldb]->coeff[jj];
            }
        }
    }

    for (ll = 0; ll < c->num_poly; ll++){
        for (ii = 0; ii < maxordera; ii++){
            for (jj = 0; jj < maxorderb; jj++){
                c->coeff[ll] +=  lpolycoeffs[ii+jj*50+ll*2500] * 
                                    allprods[jj+ii*maxorderb];
            }
        }
    }
    free(allprods); allprods=NULL;
    orth_poly_expansion_round(&c);
    return c;
}

/********************************************************//**
*   Compute a linear combination of generic functions
*
*   \param n [in] - number of functions
*   \param ldx [in] - stride of first array
*   \param x [in] - functions
*   \param ldc [in] - stride of coefficients
*   \param c [in] - scaling coefficients
*
*   \return  out  = \f$sum_i=1^n coeff[ldc[i]] * gfa[ldgf[i]] \f$
*
*   \note 
*       If both arrays do not consist of only LEGENDRE polynomials
*       return NULL. All the functions need to have the same lower 
*       and upper bounds
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_lin_comb(size_t n, size_t ldx, 
        struct OrthPolyExpansion ** x, size_t ldc,
        double * c )
{

    struct OrthPolyExpansion * out = NULL;
    double lb = x[0]->lower_bound;
    double ub = x[0]->upper_bound;

    size_t ii;
    size_t maxorder = 0;
    //int legen = 1;
    for (ii = 0; ii < n; ii++){
        if (x[ii*ldx]->p->ptype != LEGENDRE){
            //legen = 0;
            return out; // cant do it
        }
        size_t neworder = x[ii*ldx]->num_poly;
        if (neworder > maxorder){
            maxorder = neworder;
        }
    }
    
    enum poly_type p = LEGENDRE;
    out = orth_poly_expansion_init(p, maxorder, lb, ub);
    size_t kk;
    for (kk = 0; kk < n; kk++){
        for (ii = 0; ii < x[kk*ldx]->num_poly; ii++){
            out->coeff[ii] +=  c[kk*ldc]*x[kk*ldx]->coeff[ii];
        }
    }
    orth_poly_expansion_round(&out);
    return out;
}



/********************************************************//**
*   Integrate an orthogonal polynomial expansion 
*
*   \param poly [in] - polynomial to integrate
*
*   \return out - Integral of approximation
*
*   \note 
*       Need to an 'else' or default behavior to switch case
*       int_{lb}^ub  f(x) dx
*************************************************************/
double
orth_poly_expansion_integrate(struct OrthPolyExpansion * poly)
{
    double out = 0.0;
    switch (poly->p->ptype){
        case LEGENDRE:
            out = legendre_integrate(poly);
            break;
        case CHEBYSHEV:
            out = cheb_integrate2(poly);
            break;
        case STANDARD:
            fprintf(stderr, "Cannot integrate STANDARD type\n");
            break;
        //default: 
        //    fprintf(stderr, "Polynomial type does not exist: %d\n ", 
        //            poly->p->ptype);
    }
    return out;
}

/********************************************************//**
*   Weighted inner product between two polynomial 
*   expansions of the same type
*
*   \param a [in] - first polynomial
*   \param b [in] - second polynomai
*
*   \return out - inner product
*
*   \note 
*          int_{lb}^ub  a(x)b(x) w(x) dx
*************************************************************/
double
orth_poly_expansion_inner_w(struct OrthPolyExpansion * a,
                            struct OrthPolyExpansion * b)
{
    assert(a->p->ptype == b->p->ptype);
    
    double out = 0.0;
    size_t N = a->num_poly < b->num_poly ? a->num_poly : b->num_poly;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        out += a->coeff[ii] * b->coeff[ii] * a->p->norm(ii); 
    }

    double m = (a->upper_bound - a->lower_bound) / 
                    (a->p->upper - a->p->lower);
    //printf("m=%3.15f\n",m);
    out *= m;
    return out;
}

/********************************************************//**
*   Inner product between two polynomial expansions of the same type
*
*   \param a [in] - first polynomial
*   \param b [in] - second polynomai
*
*   \return out - inner product
*
*   Notes: 
*          Computes int_{lb}^ub  a(x)b(x) dx by first
*          converting each polynomial to a Legendre polynomial
*************************************************************/
double
orth_poly_expansion_inner(struct OrthPolyExpansion * a,
                          struct OrthPolyExpansion * b)
{   
    struct OrthPolyExpansion * t1 = NULL;
    struct OrthPolyExpansion * t2 = NULL;
    
    int c1 = 0;
    int c2 = 0;
    if (a->p->ptype == CHEBYSHEV){
        t1 = orth_poly_expansion_init(LEGENDRE, a->num_poly,
                    a->lower_bound, a->upper_bound);
        orth_poly_expansion_approx(&orth_poly_expansion_eval2, a, t1);
        orth_poly_expansion_round(&t1);
        c1 = 1;
    }
    else if (a->p->ptype == LEGENDRE){
        t1 = a;
    }
    else{
        fprintf(stderr, "Don't know how to take inner product using polynomial type. \n");
    }

    if (b->p->ptype == CHEBYSHEV){
        t2 = orth_poly_expansion_init(LEGENDRE, b->num_poly,
                    b->lower_bound, b->upper_bound);
        orth_poly_expansion_approx(&orth_poly_expansion_eval2, b, t2);
        orth_poly_expansion_round(&t2);
        c2 = 1;
    }
    else if (b->p->ptype == LEGENDRE){
        t2 = b;
    }
    else{
        fprintf(stderr, "Don't know how to take inner product using polynomial type. \n");
    }
    
    /*
    printf("first poly=\n");
    print_orth_poly_expansion(t1,0,NULL);
    printf("second poly=\n");
    print_orth_poly_expansion(t2,0,NULL);
    */
    double out = orth_poly_expansion_inner_w(t1,t2) * 2.0;
    if (c1 == 1){
        orth_poly_expansion_free(t1);
    }
    if (c2 == 1){
        orth_poly_expansion_free(t2);
    }
    return out;
}

/********************************************************//**
*   Compute the norm of an orthogonal polynomial
*   expansion with respect to family weighting 
*   function
*
*   \param p [in] - polynomial to integrate
*
*   \return out - norm of function
*
*   \note
*        Computes sqrt(int f(x)^2 w(x) dx)
*************************************************************/
double orth_poly_expansion_norm_w(struct OrthPolyExpansion * p){
    size_t ii;
    double out = 0.0;
    for (ii = 0; ii < p->num_poly; ii++){
        out += pow(p->coeff[ii],2.0) * p->p->norm(ii);
    }

    double m = (p->upper_bound - p->lower_bound) / 
                    (p->p->upper - p->p->lower);
    out = out * m;
    return sqrt(out);
}

/********************************************************//**
*   Compute the norm of an orthogonal polynomial
*   expansion with respect to uniform weighting
*
*   \param p [in] - polynomial of which to obtain norm
*
*   \return out - norm of function
*
*   \note
*        Computes sqrt(int_a^b f(x)^2 dx)
*************************************************************/
double orth_poly_expansion_norm(struct OrthPolyExpansion * p){

    double out = 0.0;
    struct OrthPolyExpansion * temp;
    switch (p->p->ptype){
        case LEGENDRE:
            out = orth_poly_expansion_norm_w(p) * sqrt(2.0);
            break;
        case CHEBYSHEV:
            temp = orth_poly_expansion_init(LEGENDRE, p->num_poly,
                    p->lower_bound, p->upper_bound);
            orth_poly_expansion_approx(&orth_poly_expansion_eval2, p, temp);
            out = orth_poly_expansion_norm_w(temp) * sqrt(2.0);
            orth_poly_expansion_free(temp);
            break;
        case STANDARD:
            fprintf(stderr, "Cannot take norm of STANDARD type\n");
            break;
        //default:
        //    fprintf(stderr, "Polynomial type does not exist: %d\n ", 
        //            p->p->ptype);
    }
    return out;
}

/********************************************************//**
*   Multiply polynomial expansion by -1
*
*   \param p [inout] - polynomial multiply by -1
*************************************************************/
void 
orth_poly_expansion_flip_sign(struct OrthPolyExpansion * p)
{   
    size_t ii;
    for (ii = 0; ii < p->num_poly; ii++){
        p->coeff[ii] *= -1.0;
    }
}
/********************************************************//**
*   Multiply by scalar and overwrite expansion
*
*   \param a [in] - scaling factor for first polynomial
*   \param x [in] - polynomial to scale
*************************************************************/
void orth_poly_expansion_scale(double a, struct OrthPolyExpansion * x)
{
    
    size_t ii;
    for (ii = 0; ii < x->num_poly; ii++){
        x->coeff[ii] *= a;
    }
    orth_poly_expansion_round(&x);
}

/********************************************************//**
*   Multiply and add 3 expansions \f$ z \leftarrow ax + by + cz \f$
*
*   \param a [in] - scaling factor for first polynomial
*   \param x [in] - first polynomial
*   \param b [in] - scaling factor for second polynomial
*   \param y [in] - second polynomial
*   \param c [in] - scaling factor for third polynomial
*   \param z [in] - third polynomial
*
*************************************************************/
void
orth_poly_expansion_sum3_up(double a, struct OrthPolyExpansion * x,
                           double b, struct OrthPolyExpansion * y,
                           double c, struct OrthPolyExpansion * z)
{
    assert (x->p->ptype == y->p->ptype);
    assert (y->p->ptype == z->p->ptype);
    
    assert ( x != NULL );
    assert ( y != NULL );
    assert ( z != NULL );
    
    size_t ii;
    if ( (z->num_poly >= x->num_poly) && (z->num_poly >= y->num_poly) ){
        
        if (x->num_poly > y->num_poly){
            for (ii = 0; ii < y->num_poly; ii++){
                z->coeff[ii] = c*z->coeff[ii] + a*x->coeff[ii] + b*y->coeff[ii];
            }
            for (ii = y->num_poly; ii < x->num_poly; ii++){
                z->coeff[ii] = c*z->coeff[ii] + a*x->coeff[ii];
            }
            for (ii = x->num_poly; ii < z->num_poly; ii++){
                z->coeff[ii] = c*z->coeff[ii];
            }
        }
        else{
            for (ii = 0; ii < x->num_poly; ii++){
                z->coeff[ii] = c*z->coeff[ii] + a*x->coeff[ii] + b*y->coeff[ii];
            }
            for (ii = x->num_poly; ii < y->num_poly; ii++){
                z->coeff[ii] = c*z->coeff[ii] + b*y->coeff[ii];
            }
            for (ii = x->num_poly; ii < z->num_poly; ii++){
                z->coeff[ii] = c*z->coeff[ii];
            }
        }
    }
    else if ((z->num_poly >= x->num_poly) && ( z->num_poly < y->num_poly)) {
        double * temp = realloc(z->coeff, (y->num_poly)*sizeof(double));
        if (temp == NULL){
            fprintf(stderr,"cannot allocate new size fo z-coeff in sum3_up\n");
            exit(1);
        }
        else{
            z->coeff = temp;
        }
        for (ii = 0; ii < x->num_poly; ii++){
            z->coeff[ii] = c*z->coeff[ii]+a*x->coeff[ii]+b*y->coeff[ii];
        }
        for (ii = x->num_poly; ii < z->num_poly; ii++){
            z->coeff[ii] = c*z->coeff[ii] + b*y->coeff[ii];
        }
        for (ii = z->num_poly; ii < y->num_poly; ii++){
            z->coeff[ii] = b*y->coeff[ii];
        }
        z->num_poly = y->num_poly;
    }
    else if ( (z->num_poly < x->num_poly) && ( z->num_poly >= y->num_poly) ){
        double * temp = realloc(z->coeff, (x->num_poly)*sizeof(double));
        if (temp == NULL){
            fprintf(stderr,"cannot allocate new size fo z-coeff in sum3_up\n");
            exit(1);
        }
        else{
            z->coeff = temp;
        }
        for (ii = 0; ii < y->num_poly; ii++){
            z->coeff[ii] = c*z->coeff[ii]+a*x->coeff[ii]+b*y->coeff[ii];
        }
        for (ii = y->num_poly; ii < z->num_poly; ii++){
            z->coeff[ii] = c*z->coeff[ii] + a*x->coeff[ii];
        }
        for (ii = z->num_poly; ii < x->num_poly; ii++){
            z->coeff[ii] = a*x->coeff[ii];
        }
        z->num_poly = x->num_poly;
    }
    else if ( x->num_poly <= y->num_poly){
        double * temp = realloc(z->coeff, (y->num_poly)*sizeof(double));
        if (temp == NULL){
            fprintf(stderr,"cannot allocate new size fo z-coeff in sum3_up\n");
            exit(1);
        }
        for (ii = 0; ii < z->num_poly; ii++){
            z->coeff[ii] = c*z->coeff[ii]+a*x->coeff[ii]+b*y->coeff[ii];
        }
        for (ii = z->num_poly; ii < x->num_poly; ii++){
            z->coeff[ii] = a*x->coeff[ii] + b*y->coeff[ii];
        }
        for (ii = x->num_poly; ii < y->num_poly; ii++){
            z->coeff[ii] = b*y->coeff[ii];
        }
        z->num_poly = y->num_poly;
    }
    else if (y->num_poly <= x->num_poly) {
        double * temp = realloc(z->coeff, (x->num_poly)*sizeof(double));
        if (temp == NULL){
            fprintf(stderr,"cannot allocate new size fo z-coeff in sum3_up\n");
            exit(1);
        }
        for (ii = 0; ii < z->num_poly; ii++){
            z->coeff[ii] = c*z->coeff[ii]+a*x->coeff[ii]+b*y->coeff[ii];
        }
        for (ii = z->num_poly; ii < y->num_poly; ii++){
            z->coeff[ii] = a*x->coeff[ii] + b*y->coeff[ii];
        }
        for (ii = y->num_poly; ii < x->num_poly; ii++){
            z->coeff[ii] = a*x->coeff[ii];
        }
        z->num_poly = x->num_poly;
    }
    else{
        fprintf(stderr,"Haven't accounted for anything else?! %zu %zu %zu\n", 
                x->num_poly, y->num_poly, z->num_poly);
        exit(1);
    }
    orth_poly_expansion_round(&z);
}

/********************************************************//**
*   Multiply by scalar and add two orthgonal 
*   expansions of the same family together \f[ y \leftarrow ax + y /f]
*
*   \param a [in] - scaling factor for first polynomial
*   \param x [in] - first polynomial
*   \param y [in] - second polynomial
*
*   \return 1 if successfull 0 if error with allocating more space for y
*
*   \note 
*       Computes z=ax+by, where x and y are polynomial expansionx
*       Requires both polynomials to have the same upper 
*       and lower bounds
*   
*************************************************************/
int orth_poly_expansion_axpy(double a, struct OrthPolyExpansion * x,
                        struct OrthPolyExpansion * y)
{
        
    assert (y != NULL);
    assert (x != NULL);
    assert (x->p->ptype == y->p->ptype);
    assert ( fabs(x->lower_bound - y->lower_bound) < DBL_EPSILON );
    assert ( fabs(x->upper_bound - y->upper_bound) < DBL_EPSILON );
    
    if (x->num_poly < y->num_poly){
        // shouldnt need rounding here
        size_t ii;
        for (ii = 0; ii < x->num_poly; ii++){
            y->coeff[ii] += a * x->coeff[ii];
            if (fabs(y->coeff[ii]) < ZEROTHRESH){
                y->coeff[ii] = 0.0;
            }
        }
    }
    else{
        size_t ii;
        if (x->num_poly > y->nalloc){
            //printf("hereee\n");
            y->nalloc = x->num_poly+10;
            double * temp = realloc(y->coeff, (y->nalloc)*sizeof(double));
            if (temp == NULL){
                return 0;
            }
            else{
                y->coeff = temp;
                for (ii = y->num_poly; ii < y->nalloc; ii++){
                    y->coeff[ii] = 0.0;
                }
            }
            //printf("finished\n");
        }
        for (ii = y->num_poly; ii < x->num_poly; ii++){
            y->coeff[ii] = a * x->coeff[ii];
            if (fabs(y->coeff[ii]) < ZEROTHRESH){
                y->coeff[ii] = 0.0;
            }
        }
        for (ii = 0; ii < y->num_poly; ii++){
            y->coeff[ii] += a * x->coeff[ii];
            if (fabs(y->coeff[ii]) < ZEROTHRESH){
                y->coeff[ii] = 0.0;
            }
        }
        y->num_poly = x->num_poly;
        size_t nround = y->num_poly;
        for (ii = 0; ii < y->num_poly-1;ii++){
            if (fabs(y->coeff[y->num_poly-1-ii]) > ZEROTHRESH){
                break;
            }
            else{
                nround = nround-1;
            }
        }
        y->num_poly = nround;

    }
    return 1;
}


/********************************************************//**
*   Multiply by scalar and add two orthgonal 
*   expansions of the same family together
*
*   \param a [in] - scaling factor for first polynomial
*   \param x [in] - first polynomial
*   \param b [in] - scaling factor for second polynomial
*   \param y [in] - second polynomial
*
*   \return p - orthogonal polynomial expansion
*
*   \note 
*       Computes z=ax+by, where x and y are polynomial expansionx
*       Requires both polynomials to have the same upper 
*       and lower bounds
*   
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_daxpby(double a, struct OrthPolyExpansion * x,
                           double b, struct OrthPolyExpansion * y)
{
    /*
    printf("a=%G b=%G \n",a,b);
    printf("x=\n");
    print_orth_poly_expansion(x,0,NULL);
    printf("y=\n");
    print_orth_poly_expansion(y,0,NULL);
    */
    
    //double diffa = fabs(a-ZEROTHRESH);
    //double diffb = fabs(b-ZEROTHRESH);
    size_t ii;
    struct OrthPolyExpansion * p ;
    //if ( (x == NULL && y != NULL) || ((diffa <= ZEROTHRESH) && (y != NULL))){
    if ( (x == NULL && y != NULL)){
        //printf("b = %G\n",b);
        //if (diffb <= ZEROTHRESH){
        //    p = orth_poly_expansion_init(y->p->ptype,1,y->lower_bound, y->upper_bound);
       // }
       // else{    
            p = orth_poly_expansion_init(y->p->ptype,
                        y->num_poly, y->lower_bound, y->upper_bound);
            for (ii = 0; ii < y->num_poly; ii++){
                p->coeff[ii] = y->coeff[ii] * b;
            }
        //}
        orth_poly_expansion_round(&p);
        return p;
    }
    //if ( (y == NULL && x != NULL) || ((diffb <= ZEROTHRESH) && (x != NULL))){
    if ( (y == NULL && x != NULL)){
        //if (a <= ZEROTHRESH){
        //    p = orth_poly_expansion_init(x->p->ptype,1, x->lower_bound, x->upper_bound);
       // }
        //else{
            p = orth_poly_expansion_init(x->p->ptype,
                        x->num_poly, x->lower_bound, x->upper_bound);
            for (ii = 0; ii < x->num_poly; ii++){
                p->coeff[ii] = x->coeff[ii] * a;
            }
        //}
        orth_poly_expansion_round(&p);
        return p;
    }

    size_t N = x->num_poly > y->num_poly ? x->num_poly : y->num_poly;

    p = orth_poly_expansion_init(x->p->ptype,
                    N, x->lower_bound, x->upper_bound);
    
    size_t xN = x->num_poly;
    size_t yN = y->num_poly;

    //printf("diffa = %G, x==NULL %d\n",diffa,x==NULL);
    //printf("diffb = %G, y==NULL %d\n",diffb,y==NULL);
   // assert(diffa > ZEROTHRESH);
   // assert(diffb > ZEROTHRESH);
    if (xN > yN){
        for (ii = 0; ii < yN; ii++){
            p->coeff[ii] = x->coeff[ii]*a + y->coeff[ii]*b;           
            //if ( fabs(p->coeff[ii]) < ZEROTHRESH){
            //    p->coeff[ii] = 0.0;
           // }
        }
        for (ii = yN; ii < xN; ii++){
            p->coeff[ii] = x->coeff[ii]*a;
            //if ( fabs(p->coeff[ii]) < ZEROTHRESH){
            //    p->coeff[ii] = 0.0;
           // }
        }
    }
    else{
        for (ii = 0; ii < xN; ii++){
            p->coeff[ii] = x->coeff[ii]*a + y->coeff[ii]*b;           
            //if ( fabs(p->coeff[ii]) < ZEROTHRESH){
            //    p->coeff[ii] = 0.0;
           // }
        }
        for (ii = xN; ii < yN; ii++){
            p->coeff[ii] = y->coeff[ii]*b;
            //if ( fabs(p->coeff[ii]) < ZEROTHRESH){
            //    p->coeff[ii] = 0.0;
            //}
        }
    }

    orth_poly_expansion_round(&p);
    return p;
}

////////////////////////////////////////////////////////////////////////////
// Algorithms

/********************************************************//**
*   Obtain the real roots of a standard polynomial
*
*   \param p [in] - standard polynomial
*   \param nkeep [inout] - returns how many real roots tehre are
*
*   \return  real_roots : real roots of a standard polynomial
*
*   \note
*       Only roots within the bounds are returned
*************************************************************/
double *
standard_poly_real_roots(struct StandardPoly * p, size_t * nkeep)
{
    if (p->num_poly == 1) // constant function
    {   
        double * real_roots = NULL;
        *nkeep = 0;
        return real_roots;
    }
    else if (p->num_poly == 2){ // linear
        double root = -p->coeff[0] / p->coeff[1];
        
        if ((root > p->lower_bound) && (root < p->upper_bound)){
            *nkeep = 1;
        }
        else{
            *nkeep = 0;
        }
        double * real_roots = NULL;
        if (*nkeep == 1){
            real_roots = calloc_double(1);
            real_roots[0] = root;
        }
        return real_roots;
    }
    
    size_t nrows = p->num_poly-1;
    //printf("coeffs = \n");
    //dprint(p->num_poly, p->coeff);
    while (fabs(p->coeff[nrows]) < ZEROTHRESH ){
    //while (fabs(p->coeff[nrows]) < DBL_MIN){
        nrows--;
        if (nrows == 1){
            break;
        }
    }

    //printf("nrows left = %zu \n",  nrows);
    if (nrows == 1) // linear
    {
        double root = -p->coeff[0] / p->coeff[1];
        if ((root > p->lower_bound) && (root < p->upper_bound)){
            *nkeep = 1;
        }
        else{
            *nkeep = 0;
        }
        double * real_roots = NULL;
        if (*nkeep == 1){
            real_roots = calloc_double(1);
            real_roots[0] = root;
        }
        return real_roots;
    }
    else if (nrows == 0)
    {
        double * real_roots = NULL;
        *nkeep = 0;
        return real_roots;
    }

    // transpose of the companion matrix
    double * t_companion = calloc_double((p->num_poly-1)*(p->num_poly-1));
    size_t ii;
    

   // size_t m = nrows;
    t_companion[nrows-1] = -p->coeff[0]/p->coeff[nrows];
    for (ii = 1; ii < nrows; ii++){
        t_companion[ii * nrows + ii-1] = 1.0;
        t_companion[ii * nrows + nrows-1] = -p->coeff[ii]/p->coeff[nrows];
    }
    double * real = calloc_double(nrows);
    double * img = calloc_double(nrows);
    int info;
    int lwork = 8 * nrows;
    double * iwork = calloc_double(8 * nrows);
    //double * vl;
    //double * vr;
    int n = nrows;

    //printf("hello! n=%d \n",n);
    dgeev_("N","N", &n, t_companion, &n, real, img, NULL, &n,
            NULL, &n, iwork, &lwork, &info);
    
    //printf("info = %d", info);

    free (iwork);
    
    int * keep = calloc_int(nrows);
    *nkeep = 0;
    // the 1e-10 is kinda hacky
    for (ii = 0; ii < nrows; ii++){
        //printf("real[ii] - p->lower_bound = %G\n",real[ii]-p->lower_bound);
        //printf("real root = %3.15G, imag = %G \n",real[ii],img[ii]);
        //printf("lower thresh = %3.20G\n",p->lower_bound-1e-8);
        //printf("zero thresh = %3.20G\n",1e-8);
        //printf("upper thresh = %G\n",p->upper_bound+ZEROTHRESH);
        //printf("too low? %d \n", real[ii] < (p->lower_bound-1e-8));
        if ((fabs(img[ii]) < 1e-7) && 
            (real[ii] > (p->lower_bound-1e-8)) && 
            //(real[ii] >= (p->lower_bound-1e-7)) && 
            (real[ii] < (p->upper_bound+1e-8))) {
            //(real[ii] <= (p->upper_bound+1e-7))) {
        
            //*
            if (real[ii] < p->lower_bound){
                real[ii] = p->lower_bound;
            }
            if (real[ii] > p->upper_bound){
                real[ii] = p->upper_bound;
            }
            //*/

            keep[ii] = 1;
            *nkeep = *nkeep + 1;
            //printf("keep\n");
        }
        else{
            keep[ii] = 0;
        }
    }
    
    /*
    printf("real portions roots = ");
    dprint(nrows, real);
    printf("imag portions roots = ");
    for (ii = 0; ii < nrows; ii++) printf("%E ",img[ii]);
    printf("\n");
    //dprint(nrows, img);
    */

    double * real_roots = calloc_double(*nkeep);
    size_t counter = 0;
    for (ii = 0; ii < nrows; ii++){
        if (keep[ii] == 1){
            real_roots[counter] = real[ii];
            counter++;
        }

    }
    
    free(t_companion);
    free(real);
    free(img);
    free(keep);

    return real_roots;
}

/********************************************************//**
*   Obtain the real roots of a legendre polynomial expansion
*
*   \param p [in] - orthogonal polynomial expansion
*   \param nkeep [inout] - returns how many real roots tehre are
*
*   \return real_roots - real roots of an orthonormal polynomial expansion
*
*   \note
*       Only roots within the bounds are returned
*       Algorithm is based on eigenvalues of non-standard companion matrix from
*       Roots of Polynomials Expressed in terms of orthogonal polynomials
*       David Day and Louis Romero 2005
*************************************************************/
double * 
legendre_expansion_real_roots(struct OrthPolyExpansion * p, size_t * nkeep)
{

    double * real_roots = NULL; // output
    *nkeep = 0;

    double m = (p->upper_bound - p->lower_bound) / 
            (p->p->upper - p->p->lower);
    double off = p->upper_bound - m * p->p->upper;

    orth_poly_expansion_round(&p);
   // print_orth_poly_expansion(p,3,NULL);
    //printf("last 2 = %G\n",p->coeff[p->num_poly-1]);
    size_t N = p->num_poly-1;
    //printf("N = %zu\n",N);
    if (N == 0){
        return real_roots;
    }
    else if (N == 1){
        if (fabs(p->coeff[N]) <= ZEROTHRESH){
            return real_roots;
        }
        else{
            double root = -p->coeff[0] / p->coeff[1];
            if ( (root >= -1.0-ZEROTHRESH) && (root <= 1.0 - ZEROTHRESH)){
                if (root <-1.0){
                    root = -1.0;
                }
                else if (root > 1.0){
                    root = 1.0;
                }
                *nkeep = 1;
                real_roots = calloc_double(1);
                real_roots[0] = m*root+off;
            }
        }
    }
    else{
        double * nscompanion = calloc_double(N*N); // nonstandard companion
        size_t ii;
        double hnn1 = - (double) (N) / (2.0 * (double) (N) - 1.0);
        nscompanion[1] = 1.0;
        nscompanion[(N-1)*N] += hnn1 * p->coeff[0] / p->coeff[N];
        for (ii = 1; ii < N-1; ii++){
            double in = (double) ii;
            nscompanion[ii*N+ii-1] = in / ( 2.0 * in + 1.0);
            nscompanion[ii*N+ii+1] = (in + 1.0) / ( 2.0 * in + 1.0);
            nscompanion[(N-1)*N + ii] += hnn1 * p->coeff[ii] / p->coeff[N];
        }
        nscompanion[N*N-2] += (double) (N-1) / (2.0 * (double) (N-1) + 1.0);
        nscompanion[N*N-1] += hnn1 * p->coeff[N-1] / p->coeff[N];
        //printf("good up to here!\n");
        //dprint2d_col(N,N,nscompanion);

        int info;
        double * scale = calloc_double(N);
        //*
        //Balance
        size_t ILO, IHI;
        //printf("am I here? N=%zu \n",N);
        //dprint(N*N,nscompanion);
        dgebal_("S", &N, nscompanion, &N,&ILO,&IHI,scale,&info);
        //printf("yep\n");
        if (info < 0){
            fprintf(stderr, "Calling dgebl had error in %d-th input in the legendre_expansion_real_roots function\n",info);
            exit(1);
        }

        //printf("balanced!\n");
        //dprint2d_col(N,N,nscompanion);

        //IHI = M1;
        //printf("M1=%zu\n",M1);
        //printf("ilo=%zu\n",ILO);
        //printf("IHI=%zu\n",IHI);
        //*/

        double * real = calloc_double(N);
        double * img = calloc_double(N);
        //printf("allocated eigs N = %zu\n",N);
        size_t lwork = 8 * N;
        //printf("got lwork\n");
        double * iwork = calloc_double(8*N);
        //printf("go here");

        //dgeev_("N","N", &N, nscompanion, &N, real, img, NULL, &N,
        //        NULL, &N, iwork, &lwork, &info);
        dhseqr_("E","N",&N,&ILO,&IHI,nscompanion,&N,real,img,NULL,&N,iwork,&lwork,&info);
        //printf("done here");

        if (info < 0){
            fprintf(stderr, "Calling dhesqr had error in %d-th input in the legendre_expansion_real_roots function\n",info);
            exit(1);
        }
        else if(info > 0){
            //fprintf(stderr, "Eigenvalues are still uncovered in legendre_expansion_real_roots function\n");
           // printf("coeffs are \n");
           // dprint(p->num_poly, p->coeff);
           // printf("last 2 = %G\n",p->coeff[p->num_poly-1]);
           // exit(1);
        }

      //  printf("eigenvalues \n");
        size_t * keep = calloc_size_t(N);
        for (ii = 0; ii < N; ii++){
            //printf("(%3.15G, %3.15G)\n",real[ii],img[ii]);
            if ((fabs(img[ii]) < 1e-6) && (real[ii] > -1.0-1e-12) && (real[ii] < 1.0+1e-12)){
                if (real[ii] < -1.0){
                    real[ii] = -1.0;
                }
                else if (real[ii] > 1.0){
                    real[ii] = 1.0;
                }
                keep[ii] = 1;
                *nkeep = *nkeep + 1;
            }
        }
        
        
        if (*nkeep > 0){
            real_roots = calloc_double(*nkeep);
            size_t counter = 0;
            for (ii = 0; ii < N; ii++){
                if (keep[ii] == 1){
                    real_roots[counter] = real[ii]*m+off;
                    counter++;
                }
            }
        }
     

        free(keep); keep = NULL;
        free(iwork); iwork  = NULL;
        free(real); real = NULL;
        free(img); img = NULL;
        free(nscompanion); nscompanion = NULL;
        free(scale); scale = NULL;
    }
    return real_roots;
}


/********************************************************//**
*   Obtain the real roots of a orthogonal polynomial expansion
*
*   \param p [in] - orthogonal polynomial expansion
*   \param nkeep [inout] - returns how many real roots tehre are
*
*   \return real_roots - real roots of an orthonormal polynomial expansion
*
*   \note
*       Only roots within the bounds are returned
*************************************************************/
double *
orth_poly_expansion_real_roots(struct OrthPolyExpansion * p, size_t * nkeep)
{
    double * real_roots = NULL;
    if (p->p->ptype == LEGENDRE){
        real_roots = legendre_expansion_real_roots(p,nkeep);   
    }
    else{
        struct StandardPoly * sp = 
            orth_poly_expansion_to_standard_poly(p);
        real_roots = standard_poly_real_roots(sp,nkeep);
        standard_poly_free(sp);
    }
    return real_roots;
}

/********************************************************//**
*   Obtain the maximum of an orthogonal polynomial expansion
*
*   \param p [in] - orthogonal polynomial expansion
*   \param x [inout] - location of maximum value
*
*   \return  maxval - maximum value
*   
*   \note
*       if constant function then just returns the left most point
*************************************************************/
double orth_poly_expansion_max(struct OrthPolyExpansion * p, double * x)
{

    double maxval;
    double tempval;

    maxval = orth_poly_expansion_eval(p,p->lower_bound);
    *x = p->lower_bound;

    tempval = orth_poly_expansion_eval(p,p->upper_bound);
    if (tempval > maxval){
        maxval = tempval;
        *x = p->upper_bound;
    }
    
    if (p->num_poly > 2){
        size_t nroots;
        struct OrthPolyExpansion * deriv = orth_poly_expansion_deriv(p);
        double * roots = orth_poly_expansion_real_roots(deriv,&nroots);
        if (nroots > 0){
            size_t ii;
            for (ii = 0; ii < nroots; ii++){
                tempval = orth_poly_expansion_eval(p, roots[ii]);
                if (tempval > maxval){
                    *x = roots[ii];
                    maxval = tempval;
                }
            }
        }

        free(roots); roots = NULL;
        orth_poly_expansion_free(deriv); deriv = NULL;
    }
    return maxval;
}

/********************************************************//**
*   Obtain the minimum of an orthogonal polynomial expansion
*
*   \param p [in] - orthogonal polynomial expansion
*   \param x [inout] - location of minimum value
*
*   \return minval - minimum value
*************************************************************/
double orth_poly_expansion_min(struct OrthPolyExpansion * p, double * x)
{

    double minval;
    double tempval;

    minval = orth_poly_expansion_eval(p,p->lower_bound);
    *x = p->lower_bound;

    tempval = orth_poly_expansion_eval(p,p->upper_bound);
    if (tempval < minval){
        minval = tempval;
        *x = p->upper_bound;
    }
    
    if (p->num_poly > 2){
        size_t nroots;
        struct OrthPolyExpansion * deriv = orth_poly_expansion_deriv(p);
        double * roots = orth_poly_expansion_real_roots(deriv,&nroots);
        if (nroots > 0){
            size_t ii;
            for (ii = 0; ii < nroots; ii++){
                tempval = orth_poly_expansion_eval(p, roots[ii]);
                if (tempval < minval){
                    *x = roots[ii];
                    minval = tempval;
                }
            }
        }
        free(roots); roots = NULL;
        orth_poly_expansion_free(deriv); deriv = NULL;
    }
    return minval;
}

/********************************************************//**
*   Obtain the maximum in absolute value of an orthogonal polynomial expansion
*
*   \param p [in] - orthogonal polynomial expansion
*   \param x [inout] - location of maximum
*
*   \return maxval : maximum value (absolute value)
*
*   \note
*       if no roots then either lower or upper bound
*************************************************************/
double orth_poly_expansion_absmax(struct OrthPolyExpansion * p, double * x)
{

    //printf("in absmax\n");
   // print_orth_poly_expansion(p,3,NULL);
    //printf("%G\n", orth_poly_expansion_norm(p));
    
    double maxval;
    double norm = orth_poly_expansion_norm(p);
    
    if (norm < ZEROTHRESH) {
        *x = p->lower_bound;
        maxval = 0.0;
    }
    else{
        //printf("nroots=%zu\n", nroots);
        double tempval;

        maxval = fabs(orth_poly_expansion_eval(p,p->lower_bound));
        *x = p->lower_bound;

        tempval = fabs(orth_poly_expansion_eval(p,p->upper_bound));
        if (tempval > maxval){
            maxval = tempval;
            *x = p->upper_bound;
        }
        if (p->num_poly > 2){
            size_t nroots;
            struct OrthPolyExpansion * deriv = orth_poly_expansion_deriv(p);
            double * roots = orth_poly_expansion_real_roots(deriv,&nroots);
            if (nroots > 0){
                size_t ii;
                for (ii = 0; ii < nroots; ii++){
                    tempval = fabs(orth_poly_expansion_eval(p, roots[ii]));
                    if (tempval > maxval){
                        *x = roots[ii];
                        maxval = tempval;
                    }
                }
            }

            free(roots); roots = NULL;
            orth_poly_expansion_free(deriv); deriv = NULL;
        }
    }
    //printf("done\n");
    return maxval;
}


/////////////////////////////////////////////////////////
// Utilities
char * convert_ptype_to_char(enum poly_type ptype)
{   
    char * out = NULL;
    switch (ptype) {
        case LEGENDRE:
            out = "Legendre";
            break;
        case CHEBYSHEV:
            out = "Chebyshev";
            break;
        case STANDARD:
            out =  "Standard";
            break;
        //default:
        //    fprintf(stderr, "Polynomial type does not exist: %d\n ", ptype);
    }
    return out;

}
void print_orth_poly_expansion(struct OrthPolyExpansion * p, size_t prec, 
            void * args)
{

    if (args == NULL){
        printf("Orthogonal Polynomial Expansion:\n");
        printf("--------------------------------\n");
        printf("Polynomial basis is %s\n",convert_ptype_to_char(p->p->ptype));
        printf("Coefficients = ");
        size_t ii;
        for (ii = 0; ii < p->num_poly; ii++){
            if (prec == 0){
                printf("%3.1f ", p->coeff[ii]);
            }
            else if (prec == 1){
                printf("%3.3f ", p->coeff[ii]);
            }
            else if (prec == 2){
                printf("%3.15f ", p->coeff[ii]);
            }
            else{
                printf("%3.15E ", p->coeff[ii]);
            }
        }
        printf("\n");
    }
}

