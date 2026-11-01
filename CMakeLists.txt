cmake_minimum_required(VERSION 3.18)
project(FinalProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(nnbridge cpp/bridge.cpp)
target_include_directories(nnbridge PRIVATE cpp)
set_target_properties(nnbridge PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_executable(servidor_central cpp/servidor_central.cpp)
target_include_directories(servidor_central PRIVATE cpp)
set_target_properties(servidor_central PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(WIN32)
    target_link_libraries(nnbridge PRIVATE ws2_32)
    target_link_libraries(servidor_central PRIVATE ws2_32)
endif()