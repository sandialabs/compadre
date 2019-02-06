foreach (exe cedr_test cedr_test_mpi_device_ptr)
  add_executable (${exe} ${exe}.cpp)
  set_target_properties (${exe} PROPERTIES
    COMPILE_FLAGS ${COMPOSE_COMPILE_FLAGS}
    LINK_FLAGS ${COMPOSE_LINK_FLAGS})
  target_include_directories (${exe} PRIVATE ${COMPOSE_INCLUDES})
  target_link_libraries (${exe} ${PROJECT_NAME} ${COMPOSE_LIBRARIES})
endforeach ()

add_test (NAME cedr-test-unit
  COMMAND $<TARGET_FILE:cedr_test> -t)
add_test (NAME cedr-test-mpi-device-ptr
  COMMAND $<TARGET_FILE:cedr_test_mpi_device_ptr>)
add_test (NAME cedr-test-unit-mpi
  COMMAND ${COMPOSE_TEST_MPIRUN} ${COMPOSE_TEST_MPIFLAGS} -np ${COMPOSE_TEST_NRANK}
  $<TARGET_FILE:cedr_test> -t --proc-random -nc 111 -nt 11)
add_test (NAME cedr-test-t1d
  COMMAND $<TARGET_FILE:cedr_test> -t -t1d -nc 111)
