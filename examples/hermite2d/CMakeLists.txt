cmake_minimum_required (VERSION 2.6)


add_executable(hermite2d hermite2d.c)
target_link_libraries(hermite2d
    -g 
    c3
    ${blas_lapack} 
    -lm)

