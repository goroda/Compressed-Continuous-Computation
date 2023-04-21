cmake_minimum_required (VERSION 3.20)


add_executable(fregress fregress.c)
target_link_libraries(fregress -g
  c3
  ${blas_lapack}
  -lm)
