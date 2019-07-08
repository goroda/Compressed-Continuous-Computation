cmake_minimum_required (VERSION 2.6)


add_executable(gauss gauss.c)
target_link_libraries(gauss -g
    c3
    ${blas_lapack}
    -lm)


