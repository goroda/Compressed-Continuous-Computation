cmake_minimum_required (VERSION 3.20)


add_executable(legprodtens tprodtens.c)
target_link_libraries(legprodtens -g 
    c3
    ${blas_lapack}
    -lm)


