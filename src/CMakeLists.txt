find_package(PkgConfig REQUIRED)
pkg_check_modules(openblas REQUIRED "openblas")
include_directories(${openblas_INCLUDE_DIRS})

add_library(ffm ffm.cc timer.cc)
target_link_libraries(ffm ${openblas_LIBRARIES})

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  target_link_libraries(ffm ${OpenMP_CXX_LIBRARIES})
endif()

add_executable(ffm-train ffm-train.cc)
add_executable(ffm-predict ffm-predict.cc)

target_link_libraries (ffm-train ffm)
target_link_libraries (ffm-predict ffm)
