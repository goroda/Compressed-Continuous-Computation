cmake_minimum_required (VERSION 2.6)


add_executable(diffetc diffetc.c)
target_link_libraries(diffetc -g
    c3
    ${blas_lapack}
    -lm)



