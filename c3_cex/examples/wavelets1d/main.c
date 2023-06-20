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
			qvals[sample_ind * n + jj] /= beta[order * n + jj];
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
	for (size_t ii = 0; ii <= order; ii++) {
		for (size_t jj = 0; jj < ii; jj++) {
			double val = - Q[ii * (order+1) + jj];
			for (size_t kk = ii+1; kk <= order; kk++) {
				val += coeffB[ii * (order+1) + kk] * coeffB[jj * (order+1) + kk] * pow(coeffB[order * (order+1) + kk], 2);
			}
			val /= pow(coeffB[order * (order+1) + ii], 2);
			printf("error [%zu, %zu] = %3.2E\n", ii, jj, val - coeffB[ jj * (order + 1) + ii]);
		}
	}
	
	/* printf("Analytical Q\n");	 */
	/* for (size_t ii = 0; ii <= order; ii++) { */
	/* 	for (size_t jj = 0; jj <= order; jj++) { */
	/* 		printf("%3.2E ", Q[jj * (order+1) + ii]); */
	/* 	} */
	/* 	printf("\n"); */
	/* }	 */

	printf("\n\n\n");
	free(x); x = NULL;
	free(evals); evals = NULL;
}


/* returns lower triangular beta */
double * orthogonalize(size_t N, double *A, const double *X, const double *B)
{

	size_t NN = N+1;
	
	/* Q = -1.0X^TB + A*/
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, NN, NN, NN,
				-1.0, X, NN, B, NN, 1.0, A, NN);

	double *Q = A;
	printf("Q = \n");
	for (size_t ii = 0; ii < NN; ii++) {
		for (size_t jj = 0; jj < NN; jj++) {
			printf("%3.5f ", A[jj * NN + ii]);				
		}
		printf("\n");
	}
	
	
	double *a = malloc(NN * sizeof(double));
	a[N] = Q[N * NN + N];

	printf("a[N] = %3.4f\n", a[N]);
	/* Transpose might be faster */
	double *beta = calloc(NN * NN, sizeof(double));  /* lower triangular, last column has normalizing factors */
	for (int jj = N-1; jj >= 0; jj-- ) {
		printf("jj = %d\n", jj);
		beta[jj * NN + N] = - Q[N * NN + jj] / a[N];		
	}

	printf("ii = N done\n");
	double * qk  = malloc(NN * sizeof(double));	
	for (int ii = N-1; ii >= 0; ii--) {
		printf("ii = %d\n", ii);
		/* Compute Un-normalized \beta_{ij} */
		for (int jj = ii-1; jj >= 0; jj--) {
			beta[jj * NN + ii] = - Q[ii * NN + jj];
			for (size_t kk = ii + 1; kk <= N; kk++) {
				beta[jj * NN + ii] += (beta[ii * NN + kk] * beta[jj * NN + kk] * a[kk]);
			}
			printf("beta[%d, %d] = %3.5f\n", ii, jj, beta[jj * NN + ii]);
		}

		/* Compute <r_i, r_i> */
		a[ii] = Q[ii * NN + ii];
		for (int kk = N ; kk >= ii+1; kk--) {			
			a[ii] -=  a[kk] * pow(beta[ii * NN + kk], 2);
		}

		/* Update beta_{ij} */
		for (int jj = ii-1; jj >= 0; jj--) {
			beta[jj * N + ii] /= a[ii];
		}
	}

	/* Store normalizing info */
	for (int ii = N-1; ii >= 0; ii--) {
		/* printf("ii = %d\n", ii); */
		beta[N * NN + ii] = sqrt(a[ii]);
	}
	beta[N * NN + N] = sqrt(Q[N * NN + N]);

	printf("<r_i, r_i> = \n");
	for (size_t ii = 0; ii < NN; ii++) {
		printf("%3.5f ", a[ii]);				
	}

	printf("\n");
	
	free(a); a = NULL;
	free(qk); qk = NULL;
	/* free(AX); AX = NULL; */
	return beta;
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
	size_t n = order + 1;
	size_t n2 = n * n;

	double *A = malloc(n2 * sizeof(double));
	double *B = malloc(n2 * sizeof(double));
	for (size_t ii = 0; ii <= order; ii++){
		for (size_t jj = 0; jj <= order; jj++) {
			/* A[ii * n + jj] = 1.0 / ((double) ii + jj + 1.0); */
			/* B[ii * n + jj] = -(1 - pow(0.5, ii + jj)) * A[ii * n + jj]; */
			
			A[ii * n + jj] = 1.0 / ((double) ii + jj + 1.0) - (pow(-1.0,ii+jj+1) / ((double) ii + jj + 1.0)) ;
			B[ii * n + jj] = -(1.0 / ((double) ii + jj + 1.0) + (pow(-1.0,ii+jj+1) / ((double) ii + jj + 1.0)));
		}
	}

	/* Verified */
	printf("A = \n");
	dprint2d_col(n, n, A);
	
	double *X = malloc(n2 * sizeof(double));
	memmove(X, B, n2 * sizeof(double));

	double *Acopy = malloc(n2 * sizeof(double));
	memmove(Acopy, A, n2 * sizeof(double));

	int *ipiv = malloc(n * sizeof(int));
	double work_opt;
	int lwork = -1;
	int info = 0;
	printf("B = \n");
	dprint2d_col(n, n, B);
	dsysv_("L", (int*)&n, (int*)&n, A, (int*)&n, ipiv, X, (int*)&n, &work_opt, &lwork, &info);
	printf("optimal work = %zu\n", (size_t)work_opt);
	printf("info = %d\n", info);
	printf("Bshould = \n");
	dprint2d(n, n, B);
	lwork = (int) work_opt;
	double * work = malloc(lwork * sizeof(double));
	dsysv_("L", (int*)&n, (int*)&n, A, (int*)&n, ipiv, X, (int*)&n, work, &lwork, &info);
	printf("info = %d\n", info);
	assert( info == 0 );
	
	printf("X = \n");
	dprint2d_col(n, n, X);


	double * Bcheck = malloc(n * n * sizeof(double));
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, Acopy, n, X, n, 0.0, Bcheck, n);

	printf("Bcheck = \n");
	dprint2d_col(n, n, Bcheck);
	/* exit(1); */

	double * beta = orthogonalize(order, Acopy, X, B);

	test_q_orthogonality(-1.0, 1.0, 1000, order, X, Acopy);
	test_r_orthogonality(-1.0, 1.0, 1000000, order, X, beta,  Acopy);
	
	/* Acopy becomes Q */
	/* check orthogonality */
	/* for (size_t ii = 0; ii < n; ii++){ */
	/* 	for (size_t jj = 0; jj < ii; jj++) { */
	/* 		printf("Checking orthogonality between r_{%zu} and r_{%zu}\n", ii, jj); */
	/* 		double inner = -beta[jj * n + ii] * pow(beta[(order)*n + ii], 2); */
	/* 		for (size_t kk = ii+1; kk < n; kk++) { */
	/* 			inner -= beta[ii * n + kk] * beta[jj * n + kk] * pow(beta[(order) * n + kk], 2); */
	/* 		} */
	/* 		inner /= beta[(order) * n + ii]; */
	/* 		printf("\t inner = %3.5E\n", inner); */
	/* 	} */
	/* } */
	
	printf("Final Q = %3.5E \n", sqrt(Acopy[order * n + order]));
	exit(1);

	
	printf("beta = \n");
	for (size_t ii = 0; ii < n; ii++){
		for (size_t jj = 0; jj < n; jj++){
			printf("%3.2E ", beta[ii + jj * n]);
		}
		printf("\n");
	}
	
	double lb = -1.0;	
	double ub = -1e-15;
	size_t N = 100000;
	/* size_t N = 1000; */
	double * x = linspace(lb, ub, N);	
	double * qvals = malloc(N * n * sizeof(double));
	int res = eval_q(N, order, x, X, qvals);
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
	free(x);
	free(qvals);

	lb = 1e-15;	
	ub = 1.0;
	x = linspace(lb, ub, N);	
	qvals = malloc(N * n * sizeof(double));
	res = eval_q(N, order, x, X, qvals);
	assert (res == 0);

	fp = fopen("psi_non_orth2.dat", "w");
	assert (fp != NULL);
	for (size_t ii = 0; ii < N; ii++){
		for (size_t jj = 0; jj < n; jj++){
			fprintf(fp, "%3.15f ", qvals[ii * n + jj]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	res2 = eval_r(N, order, beta, qvals); /* qvals becomes rvals */
	assert (res2 == 0);
	fp = fopen("psi_orth2.dat", "w");
	assert (fp != NULL);
	for (size_t ii = 0; ii < N; ii++){
		for (size_t jj = 0; jj < n; jj++){
			fprintf(fp, "%3.15f ", qvals[ii * n +jj]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);	

	
	free(qvals); qvals = NULL;
	free(ipiv); ipiv = NULL;
	free(x); x = NULL;
	free(A); A = NULL;
	free(Acopy); Acopy = NULL;
	free(B); B = NULL;
	free(X); X = NULL;
	free(work); work = NULL;
	free(beta); beta = NULL;
}
