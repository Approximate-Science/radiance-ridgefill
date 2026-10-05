# kva -- the kernel library of the KVA / RidgeFill plugin: kva_mask, kva_select, kva_drop_rows,
# kva_rho_update, kva_state_correct and kva_state_read, a host row (the oracle) and a device row each.
#
# Built from the plugin's top-level CMakeLists.txt (add_subdirectory), or on its own against an
# installed radiance -- `cmake -S kernels -B build -DCMAKE_PREFIX_PATH=<radiance prefix>` -- the
# same rad_add_plugin either way, as radiance's own libraries do.
#
# HOST-ONLY BUILDS. The device rows exist when the radiance found has a device build
# (RAD_HIP_FOUND) and the caller did not pass -DRAD_WITH_HIP=OFF. Without them the plugin carries
# its host rows only: the row table leaves the device rows out (KVA_HAVE_HIP), because a row whose
# launch symbol is not in the .so would fail the engine's dlopen, not just that op.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(kva LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
  enable_testing()
  set(KVA_STANDALONE ON)
endif()

set(KVA_HIP OFF)
if(RAD_HIP_FOUND AND NOT (DEFINED RAD_WITH_HIP AND NOT RAD_WITH_HIP))
  set(KVA_HIP ON)
endif()
if(NOT KVA_HIP)
  # rad_add_plugin reads this from the caller's scope; a host-only build must not hand it the
  # device sources (it would refuse them for want of the HIP language).
  set(RAD_HIP_FOUND OFF)
endif()
if(KVA_HIP AND KVA_STANDALONE)
  # The GPU targets come from the radiance found (or -DRAD_GPU_TARGETS), never from this file.
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${RAD_GPU_TARGETS}")
  endif()
  enable_language(HIP)
endif()

set(KVA_HIP_SOURCES "")
if(KVA_HIP)
  set(KVA_HIP_SOURCES mask.hip rho.hip state_correct.hip select.hip hazard.hip)
endif()

rad_add_plugin(kva kernels SOURCES rows.cpp host_ref.cpp forward.cpp HIP_SOURCES ${KVA_HIP_SOURCES})
target_link_libraries(kva PRIVATE ${CMAKE_DL_LIBS})   # forward.cpp finds libr4d / libref (dlopen NOLOAD)

# -ffp-contract=off on the HOST rows too (rad_add_plugin sets it for device code only): the oracle
# has to round `a*b + c` twice exactly as the device rows do, or "host == device, max diff 0"
# depends on whether the host compiler contracted into an FMA.
target_compile_options(kva PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
if(KVA_HIP)
  string(REPLACE ";" " " KVA_TARGET_LIST "${RAD_GPU_TARGETS} host")
  target_compile_definitions(kva PRIVATE KVA_HAVE_HIP=1 "KVA_BUILD_TARGET=\"${KVA_TARGET_LIST}\"")
endif()

# ------------------------------------------------------------------ tests/kernel_test
#
# One binary, two ctest entries: `host` runs everything that needs no card (the host rows against
# hand-computed cases and the NumPy fixture), `gpu` runs the device rows against the host rows and
# is labelled `gpu`, so `ctest -LE gpu` is the no-card suite. A case that cannot run prints SKIP
# with its reason; a binary that checked nothing exits 77, which ctest reports as skipped.
set(KVA_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tests)
add_executable(kernel_test ${KVA_TEST_DIR}/kernel_test.cpp)
target_link_libraries(kernel_test PRIVATE radiance::abi ${CMAKE_DL_LIBS})
target_compile_options(kernel_test PRIVATE -ffp-contract=off)
if(KVA_HIP)
  target_compile_definitions(kernel_test PRIVATE KVA_TEST_HIP=1)
  target_link_libraries(kernel_test PRIVATE ${RAD_HIP_RUNTIME})
endif()
add_dependencies(kernel_test kva)

set(KVA_TEST_ENV
  "KVA_RHO_FIXTURE=${KVA_TEST_DIR}/fixtures/rho_numpy.txt"
  "KVA_ROWSEL_FIXTURE=${KVA_TEST_DIR}/fixtures/rowsel_quick9.json")
# kva_gemm_nt_bias forwards to the installed engine's own gemm rows: the runs load those libraries
# first, as the engine does; `kva_kernels_alone` loads none and checks the op then has no row.
set(KVA_GEMM_SOURCES "")
foreach(lib libr4d libref)
  if(DEFINED RADIANCE_KERNELS_DIR AND EXISTS "${RADIANCE_KERNELS_DIR}/${lib}.so")
    list(APPEND KVA_GEMM_SOURCES "${RADIANCE_KERNELS_DIR}/${lib}.so")
  endif()
endforeach()
add_test(NAME kva_kernels_host COMMAND kernel_test $<TARGET_FILE:kva> host ${KVA_GEMM_SOURCES})
add_test(NAME kva_kernels_alone COMMAND kernel_test $<TARGET_FILE:kva> host)
add_test(NAME kva_kernels_gpu COMMAND kernel_test $<TARGET_FILE:kva> gpu ${KVA_GEMM_SOURCES})
set_tests_properties(kva_kernels_host kva_kernels_alone kva_kernels_gpu PROPERTIES
  SKIP_RETURN_CODE 77 ENVIRONMENT "${KVA_TEST_ENV}")
set_tests_properties(kva_kernels_gpu PROPERTIES LABELS gpu)
