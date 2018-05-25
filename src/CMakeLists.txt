add_library(ffm ffm.cc timer.cc)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  target_link_libraries(ffm ${OpenMP_CXX_LIBRARIES})
endif()

if (USESSE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSESSE -O3 -march=native")
endif()

if(USECL)
  find_package(OpenCL REQUIRED)
  target_link_libraries (ffm OpenCL::OpenCL)
endif()

file(COPY ${CMAKE_CURRENT_LIST_DIR}/update_block.cl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(ffm-train ffm-train.cc)
target_link_libraries (ffm-train ffm)

add_executable(ffm-predict ffm-predict.cc)
target_link_libraries (ffm-predict ffm)
