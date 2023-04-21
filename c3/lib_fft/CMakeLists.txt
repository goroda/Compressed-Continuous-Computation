cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES *.c)

if ( ${static_libs} )
    add_library(fft ${SRC_FILES})
    target_link_libraries(fft)
else ()
    add_library(fft SHARED ${SRC_FILES} )
    target_link_libraries(fft -fPIC linalg array)
endif() 

