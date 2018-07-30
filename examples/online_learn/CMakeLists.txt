cmake_minimum_required (VERSION 2.6)


add_executable(online online.c)
target_link_libraries(online -g
  c3
  ${blas_lapack}
  -lm)
