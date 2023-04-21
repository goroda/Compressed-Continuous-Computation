cmake_minimum_required (VERSION 3.20)


add_executable(functest functest.c)
target_link_libraries(functest -g
  c3
  ${blas_lapack}
  -lm)



