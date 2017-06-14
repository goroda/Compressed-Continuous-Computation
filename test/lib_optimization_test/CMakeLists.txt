cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES 
  opt_test.c
  unconstrained_functions.c
  sgd_functions.c
  lbfgs_test.c
  bfgs_test.c
  batchgrad_test.c
  sgd_test.c
  AllMyTests.c)
add_executable(optimization_test ${SRC_FILES})
target_link_libraries(optimization_test 
    -g 
    c3
    cutest 
    ${blas_lapack} 
    -lm)


