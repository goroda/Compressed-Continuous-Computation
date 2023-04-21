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



	printf("hello!\n");

}
