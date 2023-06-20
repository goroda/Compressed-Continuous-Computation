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

static char * program_name;

void print_code_usage (FILE *, int) __attribute__ ((noreturn));
void print_code_usage (FILE * stream, int exit_code)
{
    fprintf(stream, "Wavelets \n\n");
    fprintf(stream, "Usage: %s options \n", program_name);
    fprintf(stream,
            " -h --help       Display this usage information.\n"
        );
    exit (exit_code);
}

int eval_q(size_t N, size_t order,
		   const double * xinput_array,
		   const double * alpha, double * out)
{
	size_t n = order+1;
	/* is it faster to do two loops, one with an if statement and one without? */
	for (size_t sample_ind = 0; sample_ind < N; sample_ind++){
		double x = xinput_array[sample_ind];
		/* printf("ii = %zu\n", sample_ind); */
		/* if (x >= 0.5) { */
		// loop over order
		double xin = 1.0;
		for (size_t jj = 0; jj < n; jj++) {

			/* \tilde{q} */
			/* if (x >= 0.5) { */
			if (x >= 0.0) {
				out[sample_ind * n + jj] = xin;
			}
			else{
				out[sample_ind * n + jj] = -xin;
			}

			double xout = 1.0;
			for (size_t ii = 0; ii < n; ii++) {
				out[sample_ind * n + jj] += alpha[jj * n + ii]  * xout;
				xout = xout * x;
			}
			xin = xin * x;
		}
		/* } */
		/* else { */
		/* 	double xin = 1.0; */
		/* 	for (size_t jj = 0; jj < n; jj++) { */

		/* 		/\* \tilde{q} *\/ */
		/* 		out[sample_ind * n + jj] = -xin; */

		/* 		double xout = 1.0; */
		/* 		for (size_t ii = 0; ii < n; ii++) { */
		/* 			out[sample_ind * n + jj] += alpha[jj * n + ii]  * xout; */
		/* 			xout = xout * x; */
		/* 		} */
		/* 		xin = xin * x; */
		/* 	} */
		/* } */
	}

	return 0;
}

int eval_r(size_t N, size_t order,
		   const double * beta,
		   double * qvals)
{
	int n = (int)order+1;
	/* is it faster to do two loops, one with an if statement and one without? */
	for (size_t sample_ind = 0; sample_ind < N; sample_ind++){
		for (int jj = order; jj >= 0; jj--) {
			/* qvals[sample_ind * n + jj] /= beta[order * n + jj];  */
			for (size_t kk = jj+1; kk <= order; kk++) {
				qvals[sample_ind * n + jj] += beta[jj * n + kk] * qvals[sample_ind * n + kk];
			}

		}
		/* /\* normalize *\/ */
		for (int jj = order; jj >= 0; jj--) {
			qvals[sample_ind * n + jj] /= sqrt(beta[order * n + jj]);
		}
	}

	return 0;
}

void test_q_orthogonality(double lb, double ub, size_t N, size_t order, double * coeff, double *Q)
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

	printf("Numerical Q\n");
	for (size_t ii = 0; ii <= order; ii++) {
		for (size_t jj = 0; jj <= order; jj++) {
			printf("%3.2E ", orth_mat[jj * (order+1) + ii]);
		}
		printf("\n");
	}

	printf("Analytical Q\n");	
	for (size_t ii = 0; ii <= order; ii++) {
		for (size_t jj = 0; jj <= order; jj++) {
			printf("%3.2E ", Q[jj * (order+1) + ii]);
		}
		printf("\n");
	}	

	printf("\n\n\n");
	free(x); x = NULL;
	free(evals); evals = NULL;
	free(orth_mat); orth_mat = NULL;
}

void test_r_orthogonality(double lb, double ub, size_t N, size_t order, double * coeffA, double * coeffB, double *Q)
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

	printf("Numerical R\n");
	for (size_t ii = 0; ii <= order; ii++) {
		for (size_t jj = 0; jj <= order; jj++) {
			printf("%3.2E ", orth_mat[jj * (order+1) + ii]);
		}
		printf("\n");
	}

	/* now check if beta conditions satisfied */
	printf("Coefficient errors\n");
	for (size_t ii = 0; ii <= order; ii++) {
		for (size_t jj = 0; jj < ii; jj++) {
			double val = - Q[ii * (order+1) + jj];
			for (size_t kk = ii+1; kk <= order; kk++) {
				val += coeffB[ii * (order+1) + kk] * coeffB[jj * (order+1) + kk] * coeffB[order * (order+1) + kk];
			}
			val /= coeffB[order * (order+1) + ii];
			printf("error [%zu, %zu] = %3.2E\n", ii, jj, val - coeffB[ jj * (order + 1) + ii]);
		}
	}

	printf("\n\n\n");
	free(x); x = NULL;
	free(evals); evals = NULL;
	free(orth_mat); orth_mat = NULL;
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

	/* test_q_orthogonality(-1.0, 1.0, 1000, order, X, Q); */

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

		/* Update beta_{ij} */
		for (int jj = ii-1; jj >= 0; jj--) {
			beta[jj * N + ii] /= beta[order * N + ii];
		}
	}
	return 0;
}


struct MotherBasis
{
	size_t order;
	double * acoeff;
	double * bcoeff;
};

struct MotherBasis * mother_basis_alloc(size_t order)
{
	struct MotherBasis * mb = malloc(sizeof(struct MotherBasis));
	mb->order = order;
	mb->acoeff = calloc((order+1) * (order+1), sizeof(double));
	mb->bcoeff = calloc((order+1) * (order+1), sizeof(double));

	return mb;
}

void mother_basis_free(struct MotherBasis * mb)
{
	if (mb != NULL) {
		free(mb->acoeff); mb->acoeff = NULL;
		free(mb->bcoeff); mb->bcoeff = NULL;
		free(mb); mb = NULL;
	}
}

int mother_basis_compute_coeffs(struct MotherBasis * mb)
{
	size_t n2 = (mb->order+1) * (mb->order+1);
	double * Q = calloc(n2, sizeof(double));

	int orth_res = orthogonalize_wrt_to_poly(mb->order, mb->acoeff, Q);
	assert(orth_res == 0);
	int beta_res = orthogonalize(mb->order, Q, mb->bcoeff);
	assert(beta_res == 0);

	free(Q); Q = NULL;
	return 0;
}

struct MotherBasis * mother_basis_create(size_t order)
{
	struct MotherBasis * mb = mother_basis_alloc(order);
	mother_basis_compute_coeffs(mb);
	return mb;
}

int mother_basis_eval_non_orth(const struct MotherBasis * mb, size_t N, const double * x, double * eval)
{
	assert (mb != NULL);
	int res = eval_q(N, mb->order, x, mb->acoeff, eval);
	return res;
}

int mother_basis_eval(const struct MotherBasis * mb, size_t N, const double * x, double * eval)
{
	assert (mb != NULL);
	int res = eval_q(N, mb->order, x, mb->acoeff, eval);
	assert (res == 0);
	res = eval_r(N, mb->order, mb->bcoeff, eval);
	assert (res == 0);
	return 0;
}

void run_pieces(size_t order)
{
	size_t n = order + 1;
	size_t n2 = n * n;

	double * coeff = calloc(n2, sizeof(double));
	double * beta = calloc(n2, sizeof(double));
	double * Q = calloc(n2, sizeof(double));
	
	int orth_res = orthogonalize_wrt_to_poly(order,  coeff, Q);
	assert(orth_res == 0);
	int beta_res = orthogonalize(order, Q, beta);
	assert(beta_res == 0);
	test_r_orthogonality(-1.0, 1.0, 1000000, order, coeff, beta,  Q);
	
	printf("beta = \n");
	for (size_t ii = 0; ii < n; ii++){
		for (size_t jj = 0; jj < n; jj++){
			printf("%3.2E ", beta[ii + jj * n]);
		}
		printf("\n");
	}
	
	double lb = -1.0;	
	double ub = 1.0;
	size_t N = 100000;
	/* size_t N = 1000; */
	double * x = linspace(lb, ub, N);	
	double * qvals = malloc(N * n * sizeof(double));
	int res = eval_q(N, order, x, coeff, qvals);
	assert (res == 0);

	FILE * fp = fopen("psi_non_orth.dat", "w");
	assert (fp != NULL);
	for (size_t ii = 0; ii < N; ii++){
		for (size_t jj = 0; jj < n; jj++){
			fprintf(fp, "%3.15f ", qvals[ii * n + jj]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	int res2 = eval_r(N, order, beta, qvals); /* qvals becomes rvals */
	assert (res2 == 0);
	fp = fopen("psi_orth.dat", "w");
	assert (fp != NULL);
	for (size_t ii = 0; ii < N; ii++){
		for (size_t jj = 0; jj < n; jj++){
			fprintf(fp, "%3.15f ", qvals[ii * n +jj]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	
	free(qvals); qvals = NULL;
	free(x); x = NULL;
	free(coeff); coeff = NULL;
	free(Q); Q = NULL;
	free(beta); beta = NULL;
}

int main(int argc, char * argv[])
{
    int next_option;
    const char * const short_options = "h";
    const struct option long_options[] = {
        { "help"      , 0, NULL, 'h' },
		{ NULL        ,  0, NULL, 0   }
    };

    program_name = argv[0];
    do {
        next_option = getopt_long (argc, argv, short_options, long_options, NULL);
        switch (next_option)
        {
            case 'h': 
                print_code_usage(stdout, 0);
              case '?': // The user specified an invalid option 
                print_code_usage (stderr, 1);
            case -1: // Done with options. 
                break;
            default: // Something unexpected
                abort();
        }

    } while (next_option != -1);


	size_t order = 5;
	/* run_pieces(order); */
	
	struct MotherBasis * mb = mother_basis_create(order);

	double lb = -1.0;	
	double ub = 1.0;
	size_t N = 100000;
	/* size_t N = 1000; */
	double * x = linspace(lb, ub, N);	
	double * evals = malloc(N * (order+1) * sizeof(double));

	int res = mother_basis_eval(mb, N, x, evals);
	assert (res == 0);
	
	FILE * fp = fopen("psi_orth.dat", "w");
	assert (fp != NULL);
	for (size_t ii = 0; ii < N; ii++){
		for (size_t jj = 0; jj <= order; jj++){
			fprintf(fp, "%3.15f ", evals[ii * (order+1) +jj]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	

	mother_basis_free(mb); mb = NULL;
	free(x); x = NULL;
	free(evals); evals = NULL;
	
}
