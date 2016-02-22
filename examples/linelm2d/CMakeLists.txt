cmake_minimum_required (VERSION 2.6)


add_executable(linelm2d linelm2d.c)
target_link_libraries(linelm2d
    -g 
    c3
    ${blas_lapack} 
    -lm)

