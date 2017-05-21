cmake_minimum_required (VERSION 2.6)


add_executable(ft_paper_gauss_bump local_gaussian_bump.c)
target_link_libraries(ft_paper_gauss_bump -g
    c3                                          
    ${blas_lapack}
    -lm)

add_executable(sinrank2 sinrank2.c)
target_link_libraries(sinrank2 -g
     c3                          
    ${blas_lapack}
    -lm)

