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
void mother_basis_free(struct MotherBasis *);
void write_to_file(struct MotherBasis *mb, size_t N, char *fname);

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
	int level;
	double * coeff;
};
	
struct TreeNode
{
	struct NodeData * data;
	struct TreeNode * left;
	struct TreeNode * right;
};

struct MultiResolutionBasis
{
	double * v0_coeff;
	double * w0_coeff;
	struct MotherBasis * mb;
	struct TreeNode * left;
	struct TreeNode * right;
};

int multi_resolution_basis_eval(const struct MultiResolutionBasis * mr, size_t N, const double * x, double * out)
{
	/* eval v0 */
	/* eval w0 */
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
	/* Basic functionality for Mother Basis */
	struct MotherBasis * mb = mother_basis_create(order);
	test_r_orthogonality(-1.0, 1.0, 1000000, order, mb->acoeff, mb->bcoeff,  mb->Q, 0);
	test_q_orthogonality(-1.0, 1.0, 1000000, order, mb->acoeff, mb->Q, 0);
	write_to_file(mb, 100000, "psi_orth.dat");

	mother_basis_free(mb); mb = NULL;	
}
