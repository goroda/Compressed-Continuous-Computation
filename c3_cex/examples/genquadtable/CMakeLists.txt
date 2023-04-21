cmake_minimum_required (VERSION 3.20)


add_executable(qtable qtable.c)
target_link_libraries(qtable -g 
    c3
    ${blas_lapack}
    -lm)


