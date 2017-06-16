cmake_minimum_required (VERSION 2.6)

add_executable(opescale opescale.c)
target_link_libraries(opescale -g 
    c3
    ${blas_lapack}
    -lm)


