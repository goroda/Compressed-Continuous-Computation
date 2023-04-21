cmake_minimum_required (VERSION 3.20)


add_executable(cantilever cantilever.c)
target_link_libraries(cantilever -g
  c3
  ${blas_lapack}
  -lm)



