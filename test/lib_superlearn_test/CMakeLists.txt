cmake_minimum_required (VERSION 2.6)


file(GLOB SRC_FILES
  regress_test.c          
  main.c)
          
add_executable(superlearn_test ${SRC_FILES})
target_link_libraries(superlearn_test
    suplearn
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
