# ridgefill -- the kernel library of the RidgeFill plugin: ridgefill_mask, ridgefill_select, ridgefill_drop_rows,
# ridgefill_rho_update, ridgefill_state_correct and ridgefill_state_read, a host row (the oracle) and a device row each.
#
# Built from the plugin's top-level CMakeLists.txt (add_subdirectory), or on its own against an
# installed radiance -- `cmake -S kernels -B build -DCMAKE_PREFIX_PATH=<radiance prefix>` -- the
# same rad_add_plugin either way, as radiance's own libraries do.
#
# HOST-ONLY BUILDS. The device rows exist when the radiance found has a device build
# (RAD_HIP_FOUND) and the caller did not pass -DRAD_WITH_HIP=OFF. Without them the plugin carries
# its host rows only: the row table leaves the device rows out (RIDGEFILL_HAVE_HIP), because a row whose
# launch symbol is not in the .so would fail the engine's dlopen, not just that op.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(ridgefill LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  enable_testing()
  set(RIDGEFILL_STANDALONE ON)
endif()

set(RIDGEFILL_HIP OFF)
if(RAD_HIP_FOUND AND NOT (DEFINED RAD_WITH_HIP AND NOT RAD_WITH_HIP))
  set(RIDGEFILL_HIP ON)
endif()
if(NOT RIDGEFILL_HIP)
  # rad_add_plugin reads this from the caller's scope; a host-only build must not hand it the
  # device sources (it would refuse them for want of the HIP language).
  set(RAD_HIP_FOUND OFF)
endif()
if(RIDGEFILL_HIP AND RIDGEFILL_STANDALONE)
  # The GPU targets come from the radiance found (or -DRAD_GPU_TARGETS), never from this file.
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${RAD_GPU_TARGETS}")
  endif()
  enable_language(HIP)
endif()

set(RIDGEFILL_HIP_SOURCES "")
if(RIDGEFILL_HIP)
  set(RIDGEFILL_HIP_SOURCES mask.hip rho.hip state_correct.hip select.hip hazard.hip)
endif()

rad_add_plugin(ridgefill kernels SOURCES rows.cpp host_ref.cpp forward.cpp HIP_SOURCES ${RIDGEFILL_HIP_SOURCES})
target_link_libraries(ridgefill PRIVATE ${CMAKE_DL_LIBS})   # forward.cpp finds libr4d / libref (dlopen NOLOAD)

# -ffp-contract=off on the HOST rows too (rad_add_plugin sets it for device code only): the oracle
# has to round `a*b + c` twice exactly as the device rows do, or "host == device, max diff 0"
# depends on whether the host compiler contracted into an FMA.
target_compile_options(ridgefill PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
if(RIDGEFILL_HIP)
  string(REPLACE ";" " " RIDGEFILL_TARGET_LIST "${RAD_GPU_TARGETS} host")
  target_compile_definitions(ridgefill PRIVATE RIDGEFILL_HAVE_HIP=1 "RIDGEFILL_BUILD_TARGET=\"${RIDGEFILL_TARGET_LIST}\"")
endif()

# ------------------------------------------------------------------ tests/kernel_test
#
# One binary, two ctest entries: `host` runs everything that needs no card (the host rows against
# hand-computed cases and the NumPy fixture), `gpu` runs the device rows against the host rows and
# is labelled `gpu`, so `ctest -LE gpu` is the no-card suite. A case that cannot run prints SKIP
# with its reason; a binary that checked nothing exits 77, which ctest reports as skipped.
set(RIDGEFILL_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tests)
add_executable(kernel_test ${RIDGEFILL_TEST_DIR}/kernel_test.cpp)
target_link_libraries(kernel_test PRIVATE radiance::abi ${CMAKE_DL_LIBS})
target_compile_options(kernel_test PRIVATE -ffp-contract=off)
if(RIDGEFILL_HIP)
  target_compile_definitions(kernel_test PRIVATE RIDGEFILL_TEST_HIP=1)
  target_link_libraries(kernel_test PRIVATE ${RAD_HIP_RUNTIME})
endif()
add_dependencies(kernel_test ridgefill)

set(RIDGEFILL_TEST_ENV
  "RIDGEFILL_RHO_FIXTURE=${RIDGEFILL_TEST_DIR}/fixtures/rho_numpy.txt"
  "RIDGEFILL_ROWSEL_FIXTURE=${RIDGEFILL_TEST_DIR}/fixtures/rowsel_quick9.json")
# ridgefill_gemm_nt_bias forwards to the installed engine's own gemm rows: the runs load those libraries
# first, as the engine does; `ridgefill_kernels_alone` loads none and checks the op then has no row.
set(RIDGEFILL_GEMM_SOURCES "")
foreach(lib libr4d libref)
  if(DEFINED RADIANCE_KERNELS_DIR AND EXISTS "${RADIANCE_KERNELS_DIR}/${lib}.so")
    list(APPEND RIDGEFILL_GEMM_SOURCES "${RADIANCE_KERNELS_DIR}/${lib}.so")
  endif()
endforeach()
add_test(NAME ridgefill_kernels_host COMMAND kernel_test $<TARGET_FILE:ridgefill> host ${RIDGEFILL_GEMM_SOURCES})
add_test(NAME ridgefill_kernels_alone COMMAND kernel_test $<TARGET_FILE:ridgefill> host)
add_test(NAME ridgefill_kernels_gpu COMMAND kernel_test $<TARGET_FILE:ridgefill> gpu ${RIDGEFILL_GEMM_SOURCES})
set_tests_properties(ridgefill_kernels_host ridgefill_kernels_alone ridgefill_kernels_gpu PROPERTIES
  SKIP_RETURN_CODE 77 ENVIRONMENT "${RIDGEFILL_TEST_ENV}")
set_tests_properties(ridgefill_kernels_gpu PROPERTIES LABELS gpu)
