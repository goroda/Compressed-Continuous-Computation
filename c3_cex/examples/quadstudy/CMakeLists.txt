cmake_minimum_required (VERSION 3.20)


add_executable(quadstudy quad.c)
target_link_libraries(quadstudy -g 
  c3
    ${blas_lapack}
    -lm)


