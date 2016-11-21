cmake_minimum_required (VERSION 2.6)


add_executable(functest functest.c)
target_link_libraries(functest -g
  prob
  clinalg 
  funcs
  opt 
  quadrature 
  linalg 
  array 
  stringmanip
  ${blas_lapack}
  -lm)



