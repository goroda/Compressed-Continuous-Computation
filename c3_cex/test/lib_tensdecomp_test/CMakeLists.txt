cmake_minimum_required (VERSION 2.6)


add_executable(tensdecomp_test tt_test.c)
add_executable(ttint_test tt_integrate_test.c)
if (APPLE)
    target_link_libraries(tensdecomp_test 
        tensdecomp 
        tensor 
        linalg 
        array 
        stringmanip
        cutest 
        ${macblas}/Accelerate)
    target_link_libraries(ttint_test
        tensdecomp 
        tensor 
        quadrature 
        linalg 
        array 
        stringmanip
        cutest 
        blas 
        lapack 
        -lm )
else ()
    target_link_libraries(tensdecomp_test 
        tensdecomp 
        tensor 
        linalg 
        array 
        stringmanip
        cutest 
        blas 
        lapack 
        -lm )
    target_link_libraries(ttint_test 
        tensdecomp 
        tensor 
        quadrature 
        linalg 
        array 
        stringmanip
        cutest 
        blas 
        lapack 
        -lm )
endif()


