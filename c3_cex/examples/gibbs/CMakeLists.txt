cmake_minimum_required (VERSION 2.6)


add_executable(gibbs gibbs.c)
target_link_libraries(gibbs -g
    c3
    ${blas_lapack}
    -lm)


