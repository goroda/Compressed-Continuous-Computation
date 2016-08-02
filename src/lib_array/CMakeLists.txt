cmake_minimum_required (VERSION 2.6)

file(GLOB SRC_FILES *.c)
if (${static_libs} )
    add_library(array ${SRC_FILES})
else ()
    add_library(array SHARED ${SRC_FILES})
    target_link_libraries(array 
      -fPIC 
      stringmanip 
      -lm)
endif()

