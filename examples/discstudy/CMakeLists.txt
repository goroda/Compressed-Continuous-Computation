cmake_minimum_required (VERSION 2.6)

#SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fsanitize=address" )
add_executable(genznd genznd.c)
target_link_libraries(genznd -g
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



