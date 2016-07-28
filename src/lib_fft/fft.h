#ifndef FFT_H
#define FFT_H

#include <stdlib.h>
#include <complex.h>

int fft_slow(size_t, double complex *, size_t, double complex *, size_t);
int fft_base(size_t, double complex *, size_t, double complex *, size_t);

#endif
