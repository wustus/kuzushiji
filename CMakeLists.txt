
cmake_minimum_required(VERSION 3.10)

project(kuzushiji)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_C_COMPILER "/opt/homebrew/Cellar/llvm/19.1.6/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/llvm/19.1.6/bin/clang++")

if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY omp)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_INCLUDE_DIR "/opt/homebrew/Cellar/llvm/19.1.6/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/Cellar/llvm/19.1.6/lib/libomp.dylib")
    include_directories(${OpenMP_CXX_INCLUDE_DIR})
    link_directories("/opt/homebrew/Cellar/llvm/19.1.6/lib")
endif()

include_directories(/usr/local/include)

list(APPEND SOURCES
    src/main.cpp
    src/reader.h
    src/reader.cpp
    src/network.h
    src/network.cpp
)

add_executable(kuzushiji ${SOURCES})
set(CMAKE_BUILD_TYPE Release)
set_target_properties(kuzushiji PROPERTIES RUNTIME_OUTPUT_DIRECTORY bin)
find_package(OpenMP REQUIRED)
target_link_libraries(kuzushiji PRIVATE OpenMP::OpenMP_CXX)
