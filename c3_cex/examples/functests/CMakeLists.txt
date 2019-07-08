cmake_minimum_required (VERSION 2.6)


add_executable(functest functest.c)
target_link_libraries(functest -g
  c3
  ${blas_lapack}
  -lm)



