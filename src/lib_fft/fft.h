#ifndef FFT_H
#define FFT_H

#include <stdlib.h>
#include <math.h>
#include <complex.h>

int fft_slow(size_t N, const double complex * xin, size_t sx, 
             double complex * xout, size_t sX);

int ifft_slow(size_t N, const double complex * xin, size_t sx, 
              double complex * xout, size_t sX);

int fft_base(size_t N, const double complex * x, size_t sx, 
             double complex * X, size_t sX);

int ifft_base(size_t N, const double complex * x, size_t sx, 
              double complex * X, size_t sX);

int fft(size_t N, const double complex * x, size_t sx, 
        double complex * X, size_t sX);

int ifft(size_t N, const double complex * x, size_t sx, 
         double complex * X, size_t sX);

#endif
