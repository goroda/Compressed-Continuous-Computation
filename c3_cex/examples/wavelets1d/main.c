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
		printf("ii = %zu\n", sample_ind);
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
			A[ii * n + jj] = 1.0 / ((double) ii + jj + 1.0);
			B[ii * n + jj] = (1 - pow(0.5, ii + jj)) * A[ii * n + jj];
		}
	}

	int *ipiv = malloc(n * sizeof(int));
	double work_opt;
	int lwork = -1;
	int info = 0;
	printf("B = \n");
	dprint2d(n, n, B);
	dsysv_("L", (int*)&n, (int*)&n, A, (int*)&n, ipiv, B, (int*)&n, &work_opt, &lwork, &info);
	printf("optimal work = %zu\n", (size_t)work_opt);
	printf("info = %d\n", info);
	printf("Bshould = \n");
	dprint2d(n, n, B);
	lwork = (int) work_opt;
	double * work = malloc(lwork * sizeof(double));
	dsysv_("L", (int*)&n, (int*)&n, A, (int*)&n, ipiv, B, (int*)&n, work, &lwork, &info);
	printf("info = %d\n", info);
	assert( info == 0 );
	
	printf("X = \n");
	dprint2d(n, n, B);

	double lb = 0.0;	
	double ub = 1.0;
	size_t N = 100;
	double * x = linspace(lb, ub, N);	
	double * out = malloc(N * n * sizeof(double));
	int res = eval_qtilde(N, order, x, B, out);
	assert (res == 0);

	FILE * fp = fopen("psi_non_orth.dat", "w");
	assert (fp != NULL);
	for (size_t ii = 0; ii < N; ii++){
		for (size_t jj = 0; jj < n; jj++){
			fprintf(fp, "%3.5f ", out[ii * n +jj]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	

	free(ipiv); ipiv = NULL;
	free(x); x = NULL;
	free(A); A = NULL;
	free(B); B = NULL;
	free(work); work = NULL;

}
