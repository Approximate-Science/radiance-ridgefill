# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
# radiance-ridgefill -- RidgeFill for radiance's Qwen4-Exp (Qwen3.8-Flash-Next), as plugins.
#
#   arch/     qwen4exp_fp8.so, plugin name qwen4exp_ridgefill: the in-tree Qwen4-Exp plugin #included and
#             shadowed by FILE NAME on $RADIANCE_HOME (PLAN D1, D2)
#   kernels/  ridgefill.so, the three new ops (another lane's directory; added when it exists)
#   tests/    host-only tests (`ctest -LE gpu`)
#
# Build: README.md. The two inputs are an installed radiance (CMAKE_PREFIX_PATH) and a source
# checkout of the SAME radiance (RADIANCE_SRC), because the arch plugin compiles the in-tree
# qwen4exp_fp8.cpp, which radiance does not install.
cmake_minimum_required(VERSION 3.21)
project(radiance_ridgefill LANGUAGES CXX)

# Where `cmake --install` puts the plugins: a $RADIANCE_HOME inside this repo (git-ignored), put in
# front of the installation's home at run time. Set BEFORE find_package, whose config would
# otherwise default it to the installed radiance's own home -- and an install there would replace
# the in-tree qwen4exp_fp8.so instead of shadowing it.
set(RAD_PLUGIN_INSTALL_HOME "${CMAKE_CURRENT_SOURCE_DIR}/home" CACHE PATH
    "the \$RADIANCE_HOME the RidgeFill plugins install under")

set(RAD_WITH_HIP "AUTO" CACHE STRING
    "ON | OFF | AUTO -- compile kernels/'s device rows (AUTO: when the installed radiance has HIP)")
set_property(CACHE RAD_WITH_HIP PROPERTY STRINGS ON OFF AUTO)

find_package(radiance REQUIRED)
rad_plugin_defaults()

# ---------------------------------------------------------------- RADIANCE_SRC must BE the install
# The arch plugin #includes arch/qwen4exp_fp8/qwen4exp_fp8.cpp from RADIANCE_SRC and compiles it
# against the INSTALLED headers. A checkout at another tag gives a plugin that is neither the
# installed in-tree plugin nor anything tested -- `off` mode would stop being stock (R7) with
# nothing to say so. So configure refuses it (R10), by two checks:
#
#  1. THE RELEASE. find_package's version is the ABI number (15.0.0 at v1.0.8 AND at v1.0.13, radiance
#     CMakeLists.txt:426-429), so it cannot tell two releases apart. The release lives in the
#     engine binary (RAD_VERSION, core/CMakeLists.txt:25), so the installed `radiance` must carry
#     the string RADIANCE_SRC's project(VERSION) names. Skipped -- and said -- when the prefix has
#     no engine binary (a headers-only install).
#  2. THE HEADERS. Every installed abi/ and arch/ header must be byte-identical to RADIANCE_SRC's,
#     because those are what the included .cpp actually compiles against. This also refuses a
#     checkout at the right tag with local edits.
set(RADIANCE_SRC "" CACHE PATH
    "radiance source checkout at the installed version (the arch plugin #includes from it)")
if(NOT RADIANCE_SRC OR NOT EXISTS "${RADIANCE_SRC}/arch/qwen4exp_fp8/qwen4exp_fp8.cpp")
  message(FATAL_ERROR
    "RADIANCE_SRC='${RADIANCE_SRC}' is not a radiance source checkout (no "
    "arch/qwen4exp_fp8/qwen4exp_fp8.cpp). Pass -DRADIANCE_SRC=<checkout at the installed version>.")
endif()
file(STRINGS "${RADIANCE_SRC}/CMakeLists.txt" _ridgefill_proj REGEX "^project\\(radiance VERSION ")
string(REGEX MATCH "VERSION ([0-9]+\\.[0-9]+\\.[0-9]+)" _ridgefill_m "${_ridgefill_proj}")
set(RADIANCE_SRC_VERSION "${CMAKE_MATCH_1}")
if(NOT RADIANCE_SRC_VERSION)
  message(FATAL_ERROR "could not read project(radiance VERSION ...) from ${RADIANCE_SRC}/CMakeLists.txt")
endif()

get_filename_component(_ridgefill_prefix "${RADIANCE_INCLUDE_DIR}/../.." ABSOLUTE)
set(_ridgefill_engine "${_ridgefill_prefix}/bin/radiance")
if(EXISTS "${_ridgefill_engine}")
  file(STRINGS "${_ridgefill_engine}" _ridgefill_versions REGEX "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  if(NOT RADIANCE_SRC_VERSION IN_LIST _ridgefill_versions)
    message(FATAL_ERROR
      "RADIANCE_SRC is radiance ${RADIANCE_SRC_VERSION} and the installed radiance is not: "
      "${_ridgefill_engine} carries version string(s) '${_ridgefill_versions}'. Check RADIANCE_SRC out at the "
      "installed release, or point CMAKE_PREFIX_PATH at an install of ${RADIANCE_SRC_VERSION}.")
  endif()
  message(STATUS "radiance ${RADIANCE_SRC_VERSION}: RADIANCE_SRC and ${_ridgefill_engine} agree")
else()
  message(STATUS "radiance release check SKIPPED: ${_ridgefill_engine} does not exist (headers only); "
                 "the header check below still runs")
endif()

function(ridgefill_require_same installed source)
  set(have "")
  if(EXISTS "${installed}")
    file(SHA256 "${installed}" have)
  endif()
  file(SHA256 "${source}" want)
  if(NOT have STREQUAL want)
    message(FATAL_ERROR
      "RADIANCE_SRC (radiance ${RADIANCE_SRC_VERSION}) and the installed radiance (ABI "
      "${RADIANCE_ABI_VERSION}) differ: ${source} is not ${installed}. The arch plugin compiles "
      "RADIANCE_SRC's qwen4exp_fp8.cpp against the installed headers, so they must be the same "
      "release, unmodified.")
  endif()
endfunction()
file(GLOB _ridgefill_abi RELATIVE "${RADIANCE_SRC}/abi" "${RADIANCE_SRC}/abi/*.h")
file(GLOB _ridgefill_arch RELATIVE "${RADIANCE_SRC}/arch/common" "${RADIANCE_SRC}/arch/common/*.h")
foreach(_h IN LISTS _ridgefill_abi)
  ridgefill_require_same("${RADIANCE_INCLUDE_DIR}/${_h}" "${RADIANCE_SRC}/abi/${_h}")
endforeach()
foreach(_h IN LISTS _ridgefill_arch)
  ridgefill_require_same("${RADIANCE_INCLUDE_DIR}/arch/${_h}" "${RADIANCE_SRC}/arch/common/${_h}")
endforeach()

# ---------------------------------------------------------------- device code
# -DRAD_WITH_HIP=OFF is the host-only build (R11): no device sources even when the installed
# radiance has HIP. rad_add_plugin reads RAD_HIP_FOUND from this scope.
if(RAD_WITH_HIP STREQUAL "OFF")
  set(RAD_HIP_FOUND OFF)
elseif(RAD_WITH_HIP STREQUAL "ON" AND NOT RAD_HIP_FOUND)
  message(FATAL_ERROR "RAD_WITH_HIP=ON and the installed radiance was built without HIP")
endif()
if(RAD_HIP_FOUND)
  # The targets are the radiance install's (or -DRAD_GPU_TARGETS), never this file's; without this
  # CMake asks the build machine for a card, and a build container has none.
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${RAD_GPU_TARGETS}")
  endif()
  enable_language(HIP)
endif()
message(STATUS "radiance-ridgefill: device rows ${RAD_HIP_FOUND}, GPU targets '${RAD_GPU_TARGETS}', "
               "installs into ${RAD_PLUGIN_INSTALL_HOME}")

enable_testing()
add_subdirectory(arch)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/CMakeLists.txt")
  add_subdirectory(kernels)
endif()
add_subdirectory(tests)
