#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "c3.h"


/************************************************************//**************** 
	Returns \[-x^i for i = 0,\ldots, n-1\]
    
    \param[in] N            - Number of data points
    \param[in] n            - number of basis functions
    \param[in] xinput_array - (N, ) locations at which to evaluate
    \param[out] out         - (n * N) (N columns), evaluations of the basis

    \returns 0 on success
*******************************************************************************/
int eval_qtilde_left(size_t N, size_t n, const double * xinput_array, double * out)
{
	for (size_t sample_ind = 0; sample_ind < N; sample_ind++){
		double x = xinput_array[sample_ind];
		double xin = 1.0;
		for (size_t jj = 0; jj < n; jj++){
			out[sample_ind * n + jj] = -xin;
			xin *= x;
		}
	}
	return 0;
}

/************************************************************//**************** 
	Returns \[ x^i for i = 0,\ldots, n-1\]
    
    \param[in] N            - Number of data points
    \param[in] n            - number of basis functions
    \param[in] xinput_array - (N, ) locations at which to evaluate
    \param[out] out         - (n * N) (N columns), evaluations of the basis

    \returns 0 on success
*******************************************************************************/
int eval_qtilde_right(size_t N, size_t n, const double * xinput_array, double * out)
{
	for (size_t sample_ind = 0; sample_ind < N; sample_ind++){
		double x = xinput_array[sample_ind];
		double xin = 1.0;
		for (size_t jj = 0; jj < n; jj++){
			out[sample_ind * n + jj] = xin;
			xin *= x;
		}
	}
	return 0;
}

/************************************************************//**************** 
	Returns \[ x^i + \tilde{q}_i(x) for i = 0,\ldots, n-1\]
    
    \param[in] N            - Number of data points
    \param[in] n            - number of basis functions
    \param[in] xinput_array - (N, ) locations at which to evaluate
    \param[in,out] qtilde   - (n * N) (N columns) evaluations of the basis

    \returns 0 on success

    \note Basis functions of the form
          \[ 
             q_j(x) = \tilde{q}_j(x) + \sum_{k=0}^{order} alpha_{kj} p_j(x) 
          \]
          where $p_j = x^j$ for $j = 0,\ldots,order$  and 
          $\tilde{q}_j(x) = p_j(x)$ for $x \geq 0$ and $\tilde{q}_j(x) = - p_j(x)$ for $x < 0$
*******************************************************************************/
int eval_q_update(size_t N, size_t n, const double * xinput_array, const double * alpha, double * qtilde)
{
	for (size_t sample_ind = 0; sample_ind < N; sample_ind++){
		double x = xinput_array[sample_ind];
		for (size_t jj = 0; jj < n; jj++) {
			double xout = 1.0;
			for (size_t ii = 0; ii < n; ii++) {
				qtilde[sample_ind * n + jj] += alpha[jj * n + ii]  * xout;
				xout = xout * x;
			}
		}
	}
	return 0;
}

/************************************************************//**************** 
	Evaluate piecewise-polynomial basis orthogonal to polynomials up to *order*
    
    \param[in] N            - Number of data points
    \param[in] order        - Polynomial order (order + 1 total basis functions)
    \param[in] xinput_array - (N, ) locations at which to evaluate
    \param[in] alpha        - (order+1 \times order+1) coefficients
    \param[in] out          - (order+1 * N) (N columns), evaluations of the basis

    \returns 0 on success

    \note \see @eval_q_update along with
*******************************************************************************/
int eval_q(size_t N, size_t order, const double * xinput_array, const double * alpha, double * out)
{
	size_t n = order+1;
	/* is it faster to do two loops, one with an if statement and one without? */
	for (size_t sample_ind = 0; sample_ind < N; sample_ind++){
		double x = xinput_array[sample_ind];
		if (x >= 0.0) {
			eval_qtilde_right(1, n, xinput_array + sample_ind, out + sample_ind * n);
		}
		else{
			eval_qtilde_left(1, n, xinput_array + sample_ind, out + sample_ind * n);
		}
	}
	
	eval_q_update(N, n, xinput_array, alpha, out);

	return 0;
}

/************************************************************//**************** 
	Evaluate piecewise-polynomial orthonormal basis
    
    \param[in]     N      - Number of data points
    \param[in]     order  - Polynomial order (order + 1 total basis functions)
    \param[in]     beta   - (order+1 \times order+1) coefficients
    \param[in,out] qvals  - (order * N) (N columns), 
                            On entry, evaluations of base (non-orthogonal) basis
                            On exit, evaluation of the basis

    \returns 0 on success

    \note Basis functions of the form, qvals come from eval_q
          \[ 
             r_j(x) = q_j(x) + \sum_{k=j+1}^{order} beta_{kj} r_j(x) 
          \]
          
 
*******************************************************************************/
int eval_r(size_t N, size_t order, const double * beta, double * qvals)
{
	int n = (int)order+1;

	for (size_t sample_ind = 0; sample_ind < N; sample_ind++){
		/* printf("On sample %zu\n", sample_ind); */
		for (int jj = order; jj >= 0; jj--) {
			for (size_t kk = jj+1; kk <= order; kk++) {
				qvals[sample_ind * n + jj] += beta[jj * n + kk] * qvals[sample_ind * n + kk];
			}
		}

		/* normalize */
		for (int jj = order; jj >= 0; jj--) {
			qvals[sample_ind * n + jj] /= (sqrt(beta[order * n + jj]) + 1e-15);
		}
	}

	return 0;
}

/************************************************************//**************** 
	Produce piecewise polynomials orthogonal to global polynomials up to given order .
    
    \param[in]  order - Polynomial order (order + 1 total basis functions)
    \param[out] X     - (order+1 \times order+1) Coefficients for evaluation
    \param[out] Q     - (order+1 \times order+1) inner product between all basis functions

    \returns 0 on success

    \note Comes from Alpert 1993, gram schmidt type
          basis functions are of the form 

          \[ 
             q_j(x) = \tilde{q}_j(x) + \sum_{k=0}^{order} X_{kj} p_j(x) 
          \]
          where $p_j = x^j$ for $j = 0,\ldots,order$  and 
          $\tilde{q}_j(x) = p_j(x)$ for $x \geq 0$ and $\tilde{q}_j(x) = - p_j(x)$ for $x < 0$
 
*******************************************************************************/
int orthogonalize_wrt_to_poly(size_t order,  double * X, double * Q)
{
	size_t n = order+1;
	size_t n2 = n * n;
	double *A = malloc(n2 * sizeof(double));
	double *B = malloc(n2 * sizeof(double));
	for (size_t ii = 0; ii <= order; ii++){
		for (size_t jj = 0; jj <= order; jj++) {
			A[ii * n + jj] = 1.0 / ((double) ii + jj + 1.0) - (pow(-1.0,ii+jj+1) / ((double) ii + jj + 1.0)) ;
			B[ii * n + jj] = -(1.0 / ((double) ii + jj + 1.0) + (pow(-1.0,ii+jj+1) / ((double) ii + jj + 1.0)));
		}
	}

	/* Verified */
	memmove(Q, A, n2 * sizeof(double));
	memmove(X, B, n2 * sizeof(double));

	int *ipiv = malloc(n * sizeof(int));
	double work_opt;
	int lwork = -1;
	int info = 0;
	dsysv_("L", (int*)&n, (int*)&n, A, (int*)&n, ipiv, X, (int*)&n, &work_opt, &lwork, &info);
	/* printf("optimal work = %zu\n", (size_t)work_opt); */
	/* printf("info = %d\n", info); */
	/* printf("Bshould = \n"); */
	/* dprint2d(n, n, B); */
	lwork = (int) work_opt;
	double * work = malloc(lwork * sizeof(double));
	dsysv_("L", (int*)&n, (int*)&n, A, (int*)&n, ipiv, X, (int*)&n, work, &lwork, &info);
	/* printf("info = %d\n", info); */
	assert( info == 0 );

	/* Q = -1.0X^TB + A*/
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, n, -1.0, X, n, B, n, 1.0, Q, n);


	free(A); A = NULL;
	free(B); B = NULL;
	free(ipiv); ipiv = NULL;
	free(work); work = NULL;
	return 0;
}


/************************************************************//**************** 
	Orthogonalize the basis of a piecewise polynomial (two pieces)
  
    \param[in]  order - Polynomial order (order + 1 total basis functions)
    \param[in]  Q     - Inner product between the basis
    \param[out] beta  - (order+1 x order + 1) matrix of coefficients
                        Lower-triangular part contains coefficients (excludes diagonal)
                        Last column contains the squared norms of the basis functions

    \returns 0 on success

    \note Comes from Alpert 1993, gram schmidt
*******************************************************************************/
int orthogonalize(size_t order, const double *Q, double * beta)
{

	size_t N = order + 1;
	beta[order * N + order] = Q[order * N + order];

	for (int jj = order - 1; jj >= 0; jj-- ) {
		beta[jj * N + order] = - Q[order * N + jj] / beta[order * N + order];
	}

	for (int ii = order - 1; ii >= 0; ii--) {
		/* Compute Un-normalized \beta_{ij} */
		for (int jj = ii-1; jj >= 0; jj--) {
			beta[jj * N + ii] = - Q[ii * N + jj];
			for (size_t kk = ii + 1; kk <= order; kk++) {
				beta[jj * N + ii] += (beta[ii * N + kk] * beta[jj * N + kk] * beta[order * N + kk]);
			}
		}

		/* Compute <r_i, r_i> */
		beta[order * N + ii] = Q[ii * N + ii];
		for (int kk = order ; kk >= ii+1; kk--) {			
			beta[order * N + ii] -= beta[order * N + kk] * pow(beta[ii * N + kk], 2);
		}
		/* assert (beta[order * N + ii] > 0); */

		/* Update beta_{ij} */
		for (int jj = ii-1; jj >= 0; jj--) {
			beta[jj * N + ii] /= beta[order * N + ii];
		}
	}

	printf("Norms: ");
	for (size_t ii = 0; ii <= order; ii++){
		printf("%3.2E ", beta[order * N + ii]);
	}
	printf("\n");
	
	return 0;
}

/** \struct MotherBasis 
 * \brief Contains coefficients for evaluating the mother basis multiresolution function
 * \var MotherBasis::order 
 * polynomial order to which the basis is orthogonal to 
 * \var MotherBasis::acoeff
 * coefficients for the evaluation of an intermediate basis orthogonal to global polynomials up to stated order
 * \var MotherBasis::bcoeff 
 * coefficients for the orthogonalization of the basis defined by acoeff 
 * \var MotherBasis::Q 
 * inner product between intermediate (non-orthogonal) basis functions
 */
struct MotherBasis
{
	size_t order;
	double * acoeff;
	double * bcoeff;
	double * Q;
};

/***********************************************************//**
    Allocate space for a MotherBsais

    \param[in] order - polynomial order

    \return mb - MotherBasis
***************************************************************/
struct MotherBasis * mother_basis_alloc(size_t order)
{
	struct MotherBasis * mb = malloc(sizeof(struct MotherBasis));
	mb->order = order;
	mb->acoeff = calloc((order+1) * (order+1), sizeof(double));
	mb->bcoeff = calloc((order+1) * (order+1), sizeof(double));
	mb->Q = calloc((order+1) * (order+1), sizeof(double));
	return mb;
}

/***********************************************************//**
    Free Memory of a MotherBasis

    \param[in,out] mb - struct to free
***************************************************************/
void mother_basis_free(struct MotherBasis * mb)
{
	if (mb != NULL) {
		free(mb->acoeff); mb->acoeff = NULL;
		free(mb->bcoeff); mb->bcoeff = NULL;
		free(mb->Q); mb->Q = NULL;
		free(mb); mb = NULL;
	}
}

/***********************************************************//**
    Compute coefficients defining the MotherBasis

    \param[in,out] mb - MotherBasis

	\returns 0 on success
***************************************************************/
int mother_basis_compute_coeffs(struct MotherBasis * mb)
{

	int orth_res = orthogonalize_wrt_to_poly(mb->order, mb->acoeff, mb->Q);
	assert(orth_res == 0);
	int beta_res = orthogonalize(mb->order, mb->Q, mb->bcoeff);
	assert(beta_res == 0);

	return 0;
}

/***********************************************************//**
    Create (alloc and compute coefficients) for a mother basis

    \param[in] order - polynomial order

	\returns mb - MotherBasis
***************************************************************/
struct MotherBasis * mother_basis_create(size_t order)
{
	struct MotherBasis * mb = mother_basis_alloc(order);
	mother_basis_compute_coeffs(mb);
	return mb;
}

/***********************************************************//**
    Evaluate the non-orthogonal portion of the basis 

	\see @eval_q
***************************************************************/
int mother_basis_eval_non_orth(const struct MotherBasis * mb, size_t N, const double * x, double * eval)
{
	assert (mb != NULL);
	int res = eval_q(N, mb->order, x, mb->acoeff, eval);
	return res;
}


/***********************************************************//**
    Evaluate the basis

	\see @eval_q and @eval_r
***************************************************************/
int mother_basis_eval_basis(const struct MotherBasis * mb, size_t N, const double * x, double * eval)
{
	assert (mb != NULL);
	int res = eval_q(N, mb->order, x, mb->acoeff, eval);
	assert (res == 0);
	res = eval_r(N, mb->order, mb->bcoeff, eval);
	assert (res == 0);
	return 0;
}

/***********************************************************//**
    Evaluate the basis with all x < 0

	\see @eval_qtilde_left  @eval_q_update and @eval_r
***************************************************************/
int mother_basis_eval_basis_left(const struct MotherBasis * mb, size_t N, const double * x, double * eval)
{
	assert (mb != NULL);
	size_t n = mb->order + 1;
	eval_qtilde_left(N, n, x, eval);
	eval_q_update(N, n, x, mb->acoeff, eval);
	eval_r(N, mb->order, mb->bcoeff, eval);
	return 0;
}

/***********************************************************//**
    Evaluate the basis with all x >= 0

	\see @eval_qtilde_right  @eval_q_update and @eval_r
***************************************************************/
int mother_basis_eval_basis_right(const struct MotherBasis * mb, size_t N, const double * x, double * eval)
{
	assert (mb != NULL);
	size_t n = mb->order + 1;
	eval_qtilde_right(N, n, x, eval);
	eval_q_update(N, n, x, mb->acoeff, eval);
	eval_r(N, mb->order, mb->bcoeff, eval);
	return 0;
}

void write_xy_to_file(size_t N, const double * x, const double * y, char *fname)
{
	FILE * fp = fopen(fname, "w");
	assert (fp != NULL);
	for (size_t ii = 0; ii < N; ii++){
		fprintf(fp, "%3.15f %3.15f\n", x[ii], y[ii]);
	}
	fclose(fp);
}

void write_to_file(struct MotherBasis *mb, size_t N, char *fname)
{
	double lb = -1.0;	
	double ub = 1.0;
	/* size_t N = 100000; */
	/* size_t N = 1000; */
	double * x = linspace(lb, ub, N);	
	double * evals = malloc(N * (mb->order+1) * sizeof(double));

	int res = mother_basis_eval_basis(mb, N, x, evals);
	assert (res == 0);
	
	FILE * fp = fopen(fname, "w");
	assert (fp != NULL);
	for (size_t ii = 0; ii < N; ii++){
		for (size_t jj = 0; jj <= mb->order; jj++){
			fprintf(fp, "%3.15f ", evals[ii * (mb->order+1) +jj]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	free(x); x = NULL;
	free(evals); evals = NULL;
}


void test_q_orthogonality(double lb, double ub, size_t N, size_t order, double * coeff, double *Q,
						  int verbose)
{
	
	double *x = linspace(lb, ub, N);
	double *evals = malloc(N * (order+1) * sizeof(double));
	eval_q(N, order, x, coeff, evals);

	double * orth_mat = calloc((order+1) * (order+1), sizeof(double));
	for (size_t ii = 0; ii < N-1; ii++){
		for (size_t jj = 0; jj <= order; jj++) {
			for (size_t kk = 0; kk <= order;  kk++) {
				orth_mat[jj * (order+1) + kk] +=
					0.5 * (evals[ii * (order+1) + jj] * evals[ii * (order+1) + kk] +
						   evals[(ii+1) * (order+1) + jj] * evals[(ii + 1) * (order+1) + kk]) *
					(x[ii+1] - x[ii]);
			}
		}
	}

	if (verbose != 0) {
		printf("Numerical Q\n");
		for (size_t ii = 0; ii <= order; ii++) {
			for (size_t jj = 0; jj <= order; jj++) {
				printf("%3.2E ", orth_mat[jj * (order+1) + ii]);
			}
			printf("\n");
		}
	}

	if (verbose != 0)
		printf("Analytical Q\n");	
	for (size_t ii = 0; ii <= order; ii++) {
		for (size_t jj = 0; jj <= order; jj++) {
			if (verbose != 0)
				printf("%3.2E ", Q[jj * (order+1) + ii]);
			double error = Q[jj * (order+1) + ii] - orth_mat[jj * (order+1) + ii];
			assert (fabs(error) < 1e-5);
		}
		if (verbose != 0)
			printf("\n");
	}	

	/* printf("\n\n\n"); */
	free(x); x = NULL;
	free(evals); evals = NULL;
	free(orth_mat); orth_mat = NULL;
}

void test_r_orthogonality(double lb, double ub, size_t N, size_t order, double * coeffA, double * coeffB, double *Q, int verbose)
{
	
	double *x = linspace(lb, ub, N);
	double *evals = malloc(N * (order+1) * sizeof(double));
	eval_q(N, order, x, coeffA, evals);
	eval_r(N, order, coeffB, evals);

	double * orth_mat = calloc((order+1) * (order+1), sizeof(double));
	for (size_t ii = 0; ii < N-1; ii++){
		for (size_t jj = 0; jj <= order; jj++) {
			for (size_t kk = 0; kk <= order;  kk++) {
				orth_mat[jj * (order+1) + kk] +=
					0.5 * (evals[ii * (order+1) + jj] * evals[ii * (order+1) + kk] +
						   evals[(ii+1) * (order+1) + jj] * evals[(ii + 1) * (order+1) + kk]) *
					(x[ii+1] - x[ii]);
			}
		}
	}

	if (verbose != 0) {
		printf("Numerical R\n");
		for (size_t ii = 0; ii <= order; ii++) {
			for (size_t jj = 0; jj <= order; jj++) {
				printf("%3.2E ", orth_mat[jj * (order+1) + ii]);
			}
			printf("\n");
		}
	}

	/* now check if beta conditions satisfied */
	if (verbose != 0) 
		printf("Coefficient errors\n");
	for (size_t ii = 0; ii <= order; ii++) {
		for (size_t jj = 0; jj < ii; jj++) {
			double val = - Q[ii * (order+1) + jj];
			for (size_t kk = ii+1; kk <= order; kk++) {
				val += coeffB[ii * (order+1) + kk] * coeffB[jj * (order+1) + kk] * coeffB[order * (order+1) + kk];
			}
			val /= coeffB[order * (order+1) + ii];
			double error = val - coeffB[ jj * (order + 1) + ii];
			if (verbose != 0)
				printf("error [%zu, %zu] = %3.2E\n", ii, jj, val - coeffB[ jj * (order + 1) + ii]);
			assert (fabs(error) <= 1e-12);
		}
	}

	/* printf("\n\n\n"); */
	free(x); x = NULL;
	free(evals); evals = NULL;
	free(orth_mat); orth_mat = NULL;
}

