cmake_minimum_required (VERSION 2.6)


add_executable(quadstudy quad.c)
target_link_libraries(quadstudy -g 
  c3
    ${blas_lapack}
    -lm)


