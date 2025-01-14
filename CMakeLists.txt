
cmake_minimum_required(VERSION 3.10)

project(kuzushiji)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

include_directories(/usr/local/include)

list(APPEND SOURCES
    src/main.cpp
    src/reader.h
    src/reader.cpp
    src/network.h
    src/network.cpp
)

find_package(MLX CONFIG REQUIRED)
add_executable(kuzushiji ${SOURCES})
set(CMAKE_BUILD_TYPE Release)
set_target_properties(kuzushiji PROPERTIES RUNTIME_OUTPUT_DIRECTORY bin)
target_link_libraries(kuzushiji PRIVATE mlx)
