cmake_minimum_required (VERSION 3.20)


file(GLOB SRC_FILES
  regress_test.c          
  main.c)
          
add_executable(superlearn_test ${SRC_FILES})
target_link_libraries(superlearn_test
  c3
    cutest 
    ${blas_lapack} 
    -lm)
