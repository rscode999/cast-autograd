cmake_minimum_required(VERSION 3.16)
project(CAST_Autograd LANGUAGES CXX)

# Enforce C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define source files
set(SOURCES
    main.cpp
    include/tensor_graph/tensor_operator.cpp
)

# Build the executable
add_executable(cast_autograd ${SOURCES})

# Include local headers & third-party libraries
target_include_directories(cast_autograd PRIVATE
    include
    lib/xtensor/include
    lib/xtl/include
    lib/xtensor-blas/include
)
