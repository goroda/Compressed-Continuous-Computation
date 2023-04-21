cmake_minimum_required (VERSION 3.20)


add_executable(wavelets1d main.c)
target_link_libraries(wavelets1d -g
  c3
  ${blas_lapack}
  -lm)
