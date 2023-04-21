cmake_minimum_required (VERSION 3.20)


add_executable(linelm3d linelm3d.c)
target_link_libraries(linelm3d
    -g 
    c3
    ${blas_lapack} 
    -lm)

