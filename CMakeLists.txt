cmake_minimum_required(VERSION 3.1)
project(ffm)

option(USEOMP "Enable OpenMP." 1)

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/include")

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/src")

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/tests")
