cmake_minimum_required (VERSION 2.6)


add_executable(linelm3d linelm3d.c)
target_link_libraries(linelm3d
    -g 
    c3
    ${blas_lapack} 
    -lm)

