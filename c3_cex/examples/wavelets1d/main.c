#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <getopt.h>

#include "c3.h"

void test_q_orthogonality(double, double, size_t, size_t, double *, double *, int);
void test_r_orthogonality(double, double, size_t, size_t, double *, double *, double *, int);
struct MotherBasis
{
	size_t order;
	double * acoeff;
	double * bcoeff;
	double * Q;
};
struct MotherBasis * mother_basis_create(size_t order);
int mother_basis_eval_basis(const struct MotherBasis * mb, size_t N, const double * x, double * eval);
void mother_basis_free(struct MotherBasis *);
int mother_basis_eval_basis_left(const struct MotherBasis * mb, size_t N, const double * x, double * eval);
int mother_basis_eval_basis_right(const struct MotherBasis * mb, size_t N, const double * x, double * eval);
	
void write_to_file(struct MotherBasis *mb, size_t N, char *fname);
void write_xy_to_file(size_t N, const double * x, const double * y, char *fname);

static char * program_name;

void print_code_usage (FILE *, int) __attribute__ ((noreturn));
void print_code_usage (FILE * stream, int exit_code)
{
    fprintf(stream, "Wavelets \n\n");
    fprintf(stream, "Usage: %s options \n", program_name);
    fprintf(stream,
            " -h --help       Display this usage information.\n"
			" -o --order      Polynomial order.\n"
        );
    exit (exit_code);
}

struct NodeData
{
	int id;
	int level;
	double * coeff;
};

struct NodeData * node_data_create(int id, int level, double * coeff)
{
	struct NodeData * nd = malloc(sizeof(struct NodeData));
	nd->id = id;
	nd->level = level;
	nd->coeff = coeff;

	return nd;
}

void node_data_free(struct NodeData * data)
{
	if (data != NULL) {
		free(data); data = NULL;
	}
}

struct TreeNode
{
	struct NodeData * data;
	struct TreeNode * left;
	struct TreeNode * right;
};

struct TreeNode * tree_node_create(struct NodeData * data)
{
	struct TreeNode * node = malloc(sizeof(struct TreeNode));
	node->left = NULL;
	node->right = NULL;
	node->data = data;

	return node;
}

void tree_node_free_recursive(struct TreeNode * tree)
{
	if (tree != NULL) {
		tree_node_free_recursive(tree->left); tree->left = NULL;
		tree_node_free_recursive(tree->right); tree->right = NULL;
		free(tree); tree = NULL;
	}
}

struct MRBOpts
{
	size_t order;
	size_t nquad;
};

struct MRBOpts * mrb_opts_alloc()
{
	struct MRBOpts * opts = malloc(sizeof(struct MRBOpts));
	opts->nquad = 0;

	return opts;
}

void mrb_opts_set_nquad(struct MRBOpts * opts, size_t nquad)
{
	opts->nquad = nquad;
}

void mrb_opts_set_order(struct MRBOpts * opts, size_t order)
{
	opts->order = order;
}


void mrb_opts_free(struct MRBOpts * opts)
{
	if (opts != NULL) {
		free(opts); opts = NULL;
	}
}
	
struct MultiResolutionBasis
{
	size_t order;
	struct OrthPolyExpansion * v0;
	double * w0_coeff;
	struct MotherBasis * mb;
	struct TreeNode * left;
	struct TreeNode * right;
};

struct MultiResolutionBasis * multi_resolution_basis_create(struct MRBOpts * opts)
{
	assert (opts != NULL);
	struct MultiResolutionBasis * mrb = malloc(sizeof(struct MultiResolutionBasis));
	mrb->order= opts->order;
	mrb->v0 = orth_poly_expansion_init(LEGENDRE, opts->order+1, -1, 1);
	mrb->w0_coeff = calloc(opts->order+1, sizeof(double));
	mrb->mb = mother_basis_create(opts->order);
	mrb->left = NULL;
	mrb->right = NULL;

	return mrb;
}

void multi_resolution_basis_free(struct MultiResolutionBasis * mrb)
{
	if (mrb != NULL) {
		orth_poly_expansion_free(mrb->v0);
		free(mrb->w0_coeff); mrb->w0_coeff = NULL;
		mother_basis_free(mrb->mb); mrb->mb = NULL;
		tree_node_free_recursive(mrb->left); mrb->left = NULL;
		tree_node_free_recursive(mrb->right); mrb->right = NULL;
		free(mrb); mrb = NULL;
	}
}

struct MultiResolutionBasis * fill_level0(struct MultiResolutionBasis * mrb, double * v0_coeff, double * w0_coeff)
{
	orth_poly_expansion_update_params(mrb->v0, mrb->order+1, v0_coeff);
	memmove(mrb->w0_coeff, w0_coeff, (mrb->order+1) * sizeof(double));
	return mrb;
}

int poly_basis_eval_add(size_t N, const double * x, size_t order, const double * coeff, double * out)
{

	for (size_t ii = 0; ii < N; ii++)
	{
		out[ii] += coeff[0]; /* note addition! */
		double p = x[ii];
		for (size_t jj = 1; jj <= order; jj++) {
			out[ii] += coeff[jj] * p;
			p *= x[ii];
		}
	}
	return 0;
}

int mother_basis_eval_add(const struct MotherBasis * mb, size_t N, const double * x,  const double * coeff, double * out)
{
	double * out_basis = calloc(N * (mb->order+1), sizeof(double));
	int res = mother_basis_eval_basis(mb, N, x, out_basis);
	if (res != 0)
		return res;
	for (size_t ii = 0; ii < N; ii++){
		for (size_t jj = 0; jj <= mb->order; jj++) {
			out[ii] += coeff[jj] * out_basis[ii * (mb->order+1) + jj];
		}
	}
	free(out_basis); out_basis = NULL;
	return 0;
}

void fit_level(double lb, double ub, size_t nquad, const double * quadpt,
			   struct Fwrap * f) {

	/* [-1, -1] -> [lb, ub] */
	/* pow(2, level) * (x - 1) */
	double b = 0.5 * (lb + ub);
	double m = 0.5 * (ub - lb);
	for (size_t ii = 0; ii < nquad; ii++) {
		pt_un1[ii] = m * quadpt[ii] + b;
	}

	/* NEED TO EVALUTE HIERARCHY */ 
	orth_poly_expansion_evalN(mrb->v0, nquad, pt_un1, 1, projv1, 1);

	/* Now the true function */
	int return_val = fwrap_eval(nquad, pt_un1, fvals1, f);
	assert(return_val == 0);

	for (size_t ii = 0; ii < nquad; ii++){
		fvals1[ii] = (fvals1[ii] - projv1[ii]) *  quadwt[ii];
	}
	/* printf("fvals1  = "); */
	/* for (size_t ii = 0; ii <= nquad; ii++){ */
	/* 	printf("%3.5f ", fvals1[ii]); */
	/* } */
	/* printf("\n"); */

	/* Evaluate the basis */
	size_t n = mrb->order+1;	
	double * basis = calloc(n * nquad, sizeof(double));
	return_val = mother_basis_eval_basis_left(mrb->mb, nquad, pt_un1, basis);
	assert(return_val == 0);

	/* Now project onto the basis */

}

int fit_w0(const struct MultiResolutionBasis * mrb,
		   struct Fwrap * f,
		   struct MRBOpts * opts)
{
	/* enum quad_rule qrule = C3_GAUSS_QUAD; */

	/* struct OpeOpts * opts = ope_opts_alloc(LEGENDRE); */
	/* ope_opts_free(opts); opts = NULL; */
	/* ope_opts */
	/* print_orth_poly_expansion(mrb->v0, 5, NULL, stdout); */
	orth_poly_expansion_approx_vec(mrb->v0, f, NULL);

	/* Zero it out */
	for (size_t ii = 0; ii <= mrb->order; ii++){
		mrb->w0_coeff[ii] = 0.0;
	}	
	
    size_t nquad = opts->nquad;
	
	double * quadpt = NULL;
    double * quadwt = NULL;
	int return_val = getLegPtsWts2(nquad,&quadpt,&quadwt);
	assert (return_val == 0);
		
	/* two sets of points */
	double pt_un1[200];
	double fvals1[200];
	double projv1[200];


	double pt_un2[200];
	double fvals2[200];
	double projv2[200];

	/* [-1, -1] -> [0, 1] */
	for (size_t ii = 0; ii < nquad; ii++){
		pt_un2[ii] = 0.5 * (quadpt[ii] + 1.0);
	}

	orth_poly_expansion_evalN(mrb->v0, nquad, pt_un2, 1, projv2, 1);
	return_val = fwrap_eval(nquad, pt_un2, fvals2, f);
	assert(return_val == 0);
	for (size_t ii = 0; ii < nquad; ii++) {
		fvals2[ii] = (fvals2[ii] - projv2[ii]) *  quadwt[ii];
	}
	
	printf("fvals2  = ");
	for (size_t ii = 0; ii <= nquad; ii++){
		printf("%3.5f ", fvals2[ii]);
	}
	printf("\n");
	
	return_val = mother_basis_eval_basis_right(mrb->mb, nquad, pt_un2, basis);
	assert(return_val == 0);
	for (size_t ii = 0; ii < nquad; ii++){
		for (size_t jj = 0; jj <= mrb->order; jj++) {
			mrb->w0_coeff[jj] += fvals2[ii] * basis[ii * n + jj];
		}
	}

	printf("Coefficients = ");
	for (size_t ii = 0; ii <= mrb->order; ii++){
		printf("%3.5f ", mrb->w0_coeff[ii]);
	}
	printf("\n");
	
	return 0;
}

double test_func (const double * x, void * args)
{
    assert (args == NULL);
    // 1 dimension
    
    double f = sin(5.0 * 3.14 * x[0]) * exp(-0.5 * (x[0] - 0.5) * (x[0] - 0.5) / 0.1);
	/* double f = 2.0 * pow(x[0], 4) - 3.0 * pow(x[0], 3) + 0.5 * pow(x[0], 2) + 2.0; */
	/* double f = 2.0; */
	/* printf("x = %3.2f, f = %3.2f\n", x[0], f); */

    return f;
}

	
int multi_resolution_basis_eval(const struct MultiResolutionBasis * mrb, size_t N, const double * x, double * out)
{
	orth_poly_expansion_evalN(mrb->v0, N, x, 1, out, 1);
	mother_basis_eval_add(mrb->mb, N, x, mrb->w0_coeff, out);
	/* TODO */
	/* recursively eval along tree */

	return 0;
}


int main(int argc, char * argv[])
{
    int next_option;
    const char * const short_options = "ho:";
    const struct option long_options[] = {
        { "help"      , no_argument      , NULL, 'h' },
		{ "order"     , required_argument, NULL, 'o' },
		{ NULL        ,  0, NULL, 0   }
    };

	size_t order = 5;
    program_name = argv[0];
    do {
        next_option = getopt_long (argc, argv, short_options, long_options, NULL);
        switch (next_option)
        {
            case 'h': 
                print_code_usage(stdout, 0);
		    case 'o': 
				order = strtoul(optarg, NULL, 10);
				break;
		    case '?': // The user specified an invalid option 
                print_code_usage (stderr, 1);
            case -1: // Done with options. 
                break;
            default: // Something unexpected
                abort();
        }

    } while (next_option != -1);

	printf("Using order: %zu\n", order);

	int mother_wavelet_test = 0;
	if (mother_wavelet_test == 1) {
		/* Basic functionality for Mother Basis */
		struct MotherBasis * mb = mother_basis_create(order);
		test_r_orthogonality(-1.0, 1.0, 1000000, order, mb->acoeff, mb->bcoeff,  mb->Q, 1);
		/* test_q_orthogonality(-1.0, 1.0, 1000000, order, mb->acoeff, mb->Q, 0); */
		/* write_to_file(mb, 100000, "psi_orth.dat"); */
		write_to_file(mb, 100, "psi_orth.dat");
		mother_basis_free(mb); mb = NULL;	
	}

	else{ 
		/* exit(1); */

		/* int seed = 4; */
		/* srand(seed); */
		/* int seed = 4; */
		srand(time(NULL));

		struct FunctionMonitor * fm = function_monitor_initnd(test_func, NULL, 1, 200);
		struct Fwrap * fw = fwrap_create(1, "general");
		fwrap_set_f(fw,function_monitor_eval,fm);

		struct MRBOpts * opts = mrb_opts_alloc();
		mrb_opts_set_order(opts, order);
		mrb_opts_set_nquad(opts, 2 * order);
		struct MultiResolutionBasis * mrb = multi_resolution_basis_create(opts);
		double * vcoeff = drandu(order+1);
		/* double * wcoeff = drandu(order+1); */
		double * wcoeff = dzeros(order+1);
		fill_level0(mrb, vcoeff, wcoeff);
		fit_w0(mrb, fw, opts);
	
		/* exit(1); */
		size_t N = 200;
		double * x = linspace(-1, 1, N);
		double * y = dzeros(N);
		int res = multi_resolution_basis_eval(mrb, N, x, y);
		assert (res == 0);
		write_xy_to_file(N, x, y, "mrb_evals.dat");

		for (size_t ii = 0; ii < N; ii++){
			fwrap_eval(1, x + ii, y + ii, fw);
		}
		write_xy_to_file(N, x, y, "testfunc.dat");

		free(x);
		free(y);

		multi_resolution_basis_free(mrb); mrb = NULL;
		free(vcoeff); vcoeff = NULL;
		free(wcoeff); wcoeff = NULL;
		fwrap_destroy(fw); fw = NULL;
		function_monitor_free(fm); fm = NULL;
	}

}
