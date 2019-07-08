cmake_minimum_required (VERSION 2.6)


add_executable(dmrgprodbench dmrgprod.c)
target_link_libraries(dmrgprodbench -g
    c3
    ${blas_lapack}
    -lm)



