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

add_executable(genz2d genz2d.c)
target_link_libraries(genz2d -g
    c3                             
    ${blas_lapack}
    -lm)

add_executable(sinsumstudy sinsum.c)
target_link_libraries(sinsumstudy -g
    c3                             
    ${blas_lapack}
    -lm)

add_executable(genznd genznd.c)
target_link_libraries(genznd -g
    c3                             
    ${blas_lapack}
    -lm)

add_executable(genznd_diff genznd_diff.c)
target_link_libraries(genznd_diff -g
    c3                             
    ${blas_lapack}
    -lm)

add_executable(singularity singularity.c)
target_link_libraries(singularity -g
    c3                             
    ${blas_lapack}
    -lm)  

