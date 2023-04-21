cmake_minimum_required (VERSION 3.20)


add_executable(simlib simlib.c)
target_link_libraries(simlib -g
    c3
    ${blas_lapack}
    -lm)



