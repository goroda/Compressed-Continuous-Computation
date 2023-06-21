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

