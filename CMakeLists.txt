cmake_minimum_required (VERSION 2.6)
project (C3)

#create my librray
set(C3_VERSION_MAJOR 1 CACHE STRING "major version" FORCE)
set(C3_VERSION_MINOR 0 CACHE STRING "minor version" FORCE)
set(C3_VERSION "${C3_VERSION_MAJOR}.${C3_VERSION_MINOR}" CACHE STRING "version" FORCE)

option(C3_DEBUG_MODE "whether or not debug mode is activated" OFF)

# set(include_dest "include/c3-${C3_VERSION}")
# set(main_lib_dest "lib/c3-${C3_VERSION}")
# set(lib_dest "${main_lib_dest}/${CMAKE_BUILD_TYPE}")
set(include_dest "include/c3")
set(main_lib_dest "lib/c3")
set(lib_dest "${main_lib_dest}/${CMAKE_BUILD_TYPE}")

if (APPLE)
    # option(USE_CLANG "build application with clang" ON)
    # set(compiler_flags "-std=c99 -O3 -Wall -Weverything -Wno-shorten-64-to-32 -Wno-padded -Wno-missing-prototypes -Wno-format-nonliteral -Wno-disabled-macro-expansion -Wno-missing-noreturn")
    # set(CMAKE_C_COMPILER "clang")

    include_directories(/opt/local/include)
    find_library(macblas
        NAMES ACCELERATE
        PATHS ${CMAKE_OSX_SYSROOT}/System/Library
        PATH_SUFFIXES Frameworks
        NO_DEFAULT_PATH)
        
    set(blas_lapack ${macblas}/Accelerate)

    set(compiler_flags "-g -Wall -Wextra -O2 -std=c99 -flax-vector-conversions")
    set(CMAKE_C_COMPILER "/usr/local/bin/gcc-6")
else ()

    set(compiler_flags "-g -Wall -Wextra -O2 -std=c99")
    set(CMAKE_C_COMPILER "cc")

    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)

    include_directories(/usr/include)
    set(blas_lapack "-lblas -llapack")

    #find_package(SWIG REQUIRED)
    #include(${SWIG_USE_FILE})
endif()

set(CMAKE_CXX_COMPILER "gcc")

set(static_libs 1)
#set(static_libs 0)

include_directories(${BLAS_INCLUDE_DIR}, ${LAPACK_INCLUDE_DIR})

set(CMAKE_C_FLAGS "${compiler_flags}")

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories(src/lib_stringmanip)
include_directories(src/lib_array)
include_directories(src/lib_linalg)
include_directories(src/lib_fft)
include_directories(src/lib_clinalg)
include_directories(src/lib_quadrature)
include_directories(src/lib_funcs)
include_directories(src/lib_tensor)
include_directories(src/lib_tensdecomp)
include_directories(src/lib_funcdecomp)
include_directories(src/lib_optimization)
include_directories(src/lib_probability)
include_directories(include)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
add_subdirectory(benchmarks)

# include(cmake/c3-config.cmake)

install(FILES cmake/c3-config.cmake DESTINATION ${main_lib_dest})
install(EXPORT c3 DESTINATION ${lib_dest})


