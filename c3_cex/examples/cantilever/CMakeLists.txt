cmake_minimum_required (VERSION 2.6)


add_executable(cantilever cantilever.c)
target_link_libraries(cantilever -g
  c3
  ${blas_lapack}
  -lm)



