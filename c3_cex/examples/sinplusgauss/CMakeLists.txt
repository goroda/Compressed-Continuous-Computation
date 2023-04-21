cmake_minimum_required (VERSION 3.20)


add_executable(sinplusgauss sinrank2.c)
target_link_libraries(sinplusgauss -g
    c3
    ${blas_lapack}
    -lm)



