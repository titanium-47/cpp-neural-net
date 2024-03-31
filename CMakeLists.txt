cmake_minimum_required(VERSION 3.0.0)
project(NeuralNet VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_executable(NeuralNet src/main.cpp)

target_include_directories(NeuralNet
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
