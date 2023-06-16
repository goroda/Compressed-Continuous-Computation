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

int eval_qtilde(size_t N, size_t order,
				 const double * xinput_array,
				 const double * alpha, double * out)
{
	size_t n = order+1;
	/* is it faster to do two loops, one with an if statement and one without? */
	for (size_t sample_ind = 0; sample_ind < N; sample_ind++){
		double x = xinput_array[sample_ind];
		/* printf("ii = %zu\n", sample_ind); */
		if (x >= 0.5) {
			// loop over order
			double xin = 1.0;
			for (size_t jj = 0; jj < n; jj++) {

				/* \tilde{q} */
				out[sample_ind * n + jj] = xin;

				double xout = 1.0;
				for (size_t ii = 0; ii < n; ii++) {
					out[sample_ind * n + jj] += alpha[jj * n + ii]  * xout;
					xout = xout * x;
				}
				xin = xin * x;
			}
		}
		else {
			double xin = 1.0;
			for (size_t jj = 0; jj < n; jj++) {

				/* \tilde{q} */
				out[sample_ind * n + jj] = -xin;

				double xout = 1.0;
				for (size_t ii = 0; ii < n; ii++) {
					out[sample_ind * n + jj] += alpha[jj * n + ii]  * xout;
					xout = xout * x;
				}
				xin = xin * x;
			}
		}
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
			for (size_t kk = jj+1; kk <= order; kk++) {
				qvals[sample_ind * n + jj] += beta[jj * n + kk] * qvals[sample_ind * n + kk];
			}
		}
	}

	return 0;
}



/* returns lower triangular beta */
double * orthogonalize(size_t N, double *A, const double *X, const double *B)
{

	size_t NN = N+1;

	/* double * AX = calloc(NN * NN,  sizeof(double)); */
	/* cblas_dsymm(CblasColMajor, CblasLeft, CblasLower, NN, NN, 1.0, A, NN, X, NN, 0.0, AX, NN); */

	/* printf("AX = \n"); */
	/* for (size_t ii = 0; ii < NN; ii++) { */
	/* 	for (size_t jj = 0; jj < NN; jj++) { */
	/* 		printf("%3.5f ", AX[jj * NN + ii]);				 */
	/* 	} */
	/* 	printf("\n"); */
	/* } */
	
	
	/* Q = 3.0X^TB + A*/
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, NN, NN, NN,
				3.0, X, NN, B, NN, 1.0, A, NN);

	printf("Q = \n");
	for (size_t ii = 0; ii < NN; ii++) {
		for (size_t jj = 0; jj < NN; jj++) {
			printf("%3.5f ", A[jj * NN + ii]);				
		}
		printf("\n");
	}	
		
	/* /\* Compute Q = A + X^TAX + X^TB + B^TX*\/ */
	/* cblas_dsyr2k(CblasColMajor, CblasLower, CblasTrans, */
	/* 			 NN, NN, 1.0, X, NN, B, NN, 1.0, A, NN); */


	double *Q = A;
	printf("Q = \n");
	for (size_t ii = 0; ii < NN; ii++) {
		for (size_t jj = 0; jj < NN; jj++) {
			printf("%3.5f ", A[jj * NN + ii]);				
		}
		printf("\n");
	}
	
   
	
	printf("Lets start iteration\n");
	double *a = malloc(NN * sizeof(double));
	/* double a[6]; */
	printf("whats wrong?");
	a[N] = Q[N * NN + N];
	/* a[N] = 1.0; //A[N * NN + N]; */

	printf("a[N] = %3.4f\n", a[N]);
	/* Transpose might be faster */
	double *beta = calloc(NN * NN, sizeof(double));
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
			beta[jj * NN + ii] = - Q[ii * N + jj];
			for (size_t kk = ii + 1; kk <= N; kk++) {
				beta[jj * NN + ii] += beta[ii * NN + kk] * beta[jj * NN + kk] * a[kk];
			}
		}

		/* Compute <r_i, r_i> */
		a[ii] = Q[ii * NN + ii] + 2 * beta[ii * NN + N] * Q[N * NN + ii] +
			a[N] * pow(beta[ii * NN + N], 2);

		qk[N] = Q[N * NN + ii];
		for (int kk = N - 1; kk >= ii+1; kk--) {
			qk[kk] = Q[kk * NN + ii] + beta[kk * NN + N] * Q[N * NN + ii];
			for (int ll = N-1; ll >= kk + 1; ll--) {
				qk[kk] += beta[ll + kk * NN] * qk[ll];
			}
			a[ii] += 2 * beta[kk + ii * NN] * qk[kk] + a[kk] * pow(beta[ii * NN + kk], 2);
		}

		/* Update beta_{ij} */
		for (int jj = ii-1; jj >= 0; jj--) {
			beta[jj * N + ii] /= a[ii];
		}
	}

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


	size_t order = 4;
	size_t n = order + 1;
	size_t n2 = n * n;

	double *A = malloc(n2 * sizeof(double));
	double *B = malloc(n2 * sizeof(double));
	for (size_t ii = 0; ii <= order; ii++){
		for (size_t jj = 0; jj <= order; jj++) {
			A[ii * n + jj] = 1.0 / ((double) ii + jj + 1.0);
			B[ii * n + jj] = (1 - pow(0.5, ii + jj)) * A[ii * n + jj];
		}
	}
	double *X = malloc(n2 * sizeof(double));
	memmove(X, B, n2 * sizeof(double));

	double *Acopy = malloc(n2 * sizeof(double));
	memmove(Acopy, A, n2 * sizeof(double));

	int *ipiv = malloc(n * sizeof(int));
	double work_opt;
	int lwork = -1;
	int info = 0;
	printf("B = \n");
	dprint2d(n, n, B);
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
	dprint2d(n, n, X);

	double lb = 0.0;	
	double ub = 1.0;
	size_t N = 1000;
	double * x = linspace(lb, ub, N);	
	double * qvals = malloc(N * n * sizeof(double));
	int res = eval_qtilde(N, order, x, X, qvals);
	assert (res == 0);

	FILE * fp = fopen("psi_non_orth.dat", "w");
	assert (fp != NULL);
	for (size_t ii = 0; ii < N; ii++){
		for (size_t jj = 0; jj < n; jj++){
			fprintf(fp, "%3.5f ", qvals[ii * n +jj]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	/* TODO: Add test for expression for <q_j, q_i> as everything depends on it Q matches from python */
	/* but r is not orthogonal! */
	
	double * beta = orthogonalize(order, Acopy, X, B);
	
	int res2 = eval_r(N, order, beta, qvals); /* qvals becomes rvals */
	assert (res2 == 0);
	fp = fopen("psi_orth.dat", "w");
	assert (fp != NULL);
	for (size_t ii = 0; ii < N; ii++){
		for (size_t jj = 0; jj < n; jj++){
			fprintf(fp, "%3.5f ", qvals[ii * n +jj]);
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
