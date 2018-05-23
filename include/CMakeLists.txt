configure_file(ffm.h.in ${CMAKE_CURRENT_BINARY_DIR}/ffm.h)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/ffm.h
  DESTINATION "${CMAKE_INSTALL_PREFIX}/include")
install(FILES
  ${CMAKE_CURRENT_LIST_DIR/timer.h}
  DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libffm")
