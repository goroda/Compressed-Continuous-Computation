cmake_minimum_required (VERSION 2.6)


add_executable(qtable qtable.c)
target_link_libraries(qtable -g 
    c3
    ${blas_lapack}
    -lm)


