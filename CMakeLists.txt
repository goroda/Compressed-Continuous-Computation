cmake_minimum_required (VERSION 2.6)
project (TTC)

if (APPLE)
    option(USE_CLANG "build application with clang" ON)
    include_directories(/opt/local/include)
    find_library(macblas
        NAMES ACCELERATE
        PATHS ${CMAKE_OSX_SYSROOT}/System/Library
        PATH_SUFFIXES Frameworks
        NO_DEFAULT_PATH)
        
    set(blas_lapack ${macblas}/Accelerate)
    set(compiler_flags "-std=c99 -O3 -Wall -Weverything -Wno-shorten-64-to-32 -Wno-padded -Wno-missing-prototypes -Wno-format-nonliteral -Wno-disabled-macro-expansion -Wno-missing-noreturn")
    set(CMAKE_C_COMPILER "clang")
else ()
    set(compiler_flags "-g -Wall -Wextra -O3 -std=c99")
    set(CMAKE_C_COMPILER "gcc")

    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)

    include_directories(/usr/include)
    set(blas_lapack "-lblas -llapack")

    #find_package(SWIG REQUIRED)
    #include(${SWIG_USE_FILE})
endif()

set(CMAKE_CXX_COMPILER "gcc")

#set(static_libs 1)
set(static_libs 0)

include_directories(${BLAS_INCLUDE_DIR}, ${LAPACK_INCLUDE_DIR})

set(CMAKE_C_FLAGS "${compiler_flags}")

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories(src/lib_stringmanip)
include_directories(src/lib_array)
include_directories(src/lib_linalg)
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


