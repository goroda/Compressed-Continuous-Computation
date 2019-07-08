cmake_minimum_required (VERSION 2.6)


add_executable(legprodtens tprodtens.c)
target_link_libraries(legprodtens -g 
    c3
    ${blas_lapack}
    -lm)


