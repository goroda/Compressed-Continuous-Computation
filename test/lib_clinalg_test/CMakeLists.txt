cmake_minimum_required (VERSION 2.6)


file(GLOB SRC_FILES
  dmrg_test.c          
  ftarray_test.c          
  ft_test.c
  qmarray_test.c
  quasimatrix.c
  testfunctions.c
  main.c)
          
add_executable(clinalg_test ${SRC_FILES})
target_link_libraries(clinalg_test 
    clinalg 
    funcs
    opt 
    quadrature 
    linalg 
    array 
    stringmanip
    cutest 
    ${blas_lapack} 
    -lm)



