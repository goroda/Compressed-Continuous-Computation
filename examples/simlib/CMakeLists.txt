cmake_minimum_required (VERSION 2.6)


add_executable(simlib simlib.c)
target_link_libraries(simlib -g
    c3
    ${blas_lapack}
    -lm)



