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

# The core under a second, toy adapter (adapter_core_test.cpp). Deliberately WITHOUT ${RADIANCE_SRC}/arch
# on the include path: radiance's installed arch headers (radiance::abi) and the harness only, so a core
# header that reached for the in-tree architecture would not compile here.
add_executable(adapter_core_test adapter_core_test.cpp)
target_link_libraries(adapter_core_test PRIVATE radiance::abi)
target_include_directories(adapter_core_test PRIVATE
  ${CMAKE_SOURCE_DIR}/arch ${CMAKE_CURRENT_SOURCE_DIR} ${RADIANCE_SRC}/tests)
add_test(NAME adapter_core COMMAND adapter_core_test)
set_tests_properties(adapter_core PROPERTIES SKIP_RETURN_CODE 77 LABELS "host")

# The core names no model (tests/core_purity.cmake): a grep, so it runs in milliseconds and needs no build.
add_test(NAME core_headers_name_no_arch
         COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/core_purity.cmake)
set_tests_properties(core_headers_name_no_arch PROPERTIES LABELS "host")
