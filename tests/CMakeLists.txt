# Host-only tests: no GPU, no model file. `ctest -LE gpu` runs them (R11).
#
# arch_static_test includes radiance's own test harness (tests/rad_test.h: a binary that checked
# nothing exits 77, which ctest reports as skipped rather than passed) and the arch plugin source.
add_executable(arch_static_test arch_static_test.cpp)
target_link_libraries(arch_static_test PRIVATE radiance::abi)
target_include_directories(arch_static_test PRIVATE
  ${CMAKE_SOURCE_DIR}/arch ${RADIANCE_SRC}/arch ${RADIANCE_SRC}/tests)
add_test(NAME arch_static COMMAND arch_static_test)
set_tests_properties(arch_static PROPERTIES SKIP_RETURN_CODE 77 LABELS "host")

# The core names no model (tests/core_purity.cmake): a grep, so it runs in milliseconds and needs no build.
add_test(NAME core_headers_name_no_arch
         COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/core_purity.cmake)
set_tests_properties(core_headers_name_no_arch PROPERTIES LABELS "host")
