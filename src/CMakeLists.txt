# find_package(PkgConfig REQUIRED)
# pkg_check_modules(openblas REQUIRED "openblas")
# include_directories(${openblas_INCLUDE_DIRS})


# add_custom_command(copycl
#   OUTPUT "{CMAKE_CURRENT_BINARY_DIR}/update_block.cl"
#   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_LIST_DIR}/update_block.cl ${CMAKE_CURRENT_BINARY_DIR}/update_block.cl)

add_library(ffm ffm.cc timer.cc)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  target_link_libraries(ffm ${OpenMP_CXX_LIBRARIES})
endif()

find_package(OpenCL REQUIRED)
# SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -lEasyCL" )
# SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lEasyCL" )

target_link_libraries (ffm OpenCL::OpenCL)

file(COPY ${CMAKE_CURRENT_LIST_DIR}/update_block.cl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# install(FILES
#   ${CMAKE_CURRENT_LIST_DIR}/update_block.cl
#   DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/src")

add_executable(ffm-train ffm-train.cc)
target_link_libraries (ffm-train ffm)

add_executable(ffm-predict ffm-predict.cc)
target_link_libraries (ffm-predict ffm)
