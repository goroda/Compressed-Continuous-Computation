cmake_minimum_required (VERSION 3.20)


add_executable(elliptic1d elliptic.c)
target_link_libraries(elliptic1d 
  -g
  c3
    ${blas_lapack}
    -lm)



