cmake_minimum_required (VERSION 2.6)


add_executable(linelmconv linelmconv.c)
target_link_libraries(linelmconv
    -g 
    c3
    ${blas_lapack} 
    -lm)

