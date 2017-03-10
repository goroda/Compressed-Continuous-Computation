cmake_minimum_required (VERSION 2.6)


add_executable(cantilever cantilever.c)
target_link_libraries(cantilever -g
  approx_interface
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



