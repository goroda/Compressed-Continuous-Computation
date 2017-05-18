cmake_minimum_required (VERSION 2.6)


add_executable(ft_paper_gauss_bump local_gaussian_bump.c)
target_link_libraries(ft_paper_gauss_bump -g
  approx_interface
    clinalg 
    funcs 
    opt
    quadrature 
    linalg 
    array 
    stringmanip
    ${blas_lapack}
    -lm)


