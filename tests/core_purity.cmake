# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
# core_purity.cmake -- the build-time gate core_headers_name_no_arch (notes/adapter-split-spec.md §4.5).
#
# Every arch/ridgefill_*.h is CORE: it must not name the in-tree architecture, its block types or their
# configs, so a second model's adapter compiles the core unchanged. A grep, not a compile: the compile
# half of the proof is adapter_core_test, which builds the core with no in-tree source on its path.
# Run as `cmake -DSRC=<repo> -P core_purity.cmake`.
set(FORBIDDEN "qwen4exp|GdnFP8|MoeFP8|AttnGatedFP8|QsaIndexer|HyperConn|gcfg|hccfg|moecfg|ple_layer")

file(GLOB headers RELATIVE "${SRC}/arch" "${SRC}/arch/ridgefill_*.h")
set(bad "")
foreach(h IN LISTS headers)
  file(STRINGS "${SRC}/arch/${h}" hits REGEX "${FORBIDDEN}")
  if(hits)
    list(GET hits 0 first)
    string(APPEND bad "  ${h} names a model: ${first}\n")
  endif()
endforeach()
if(bad)
  message(FATAL_ERROR "core_headers_name_no_arch:\n${bad}")
endif()
list(LENGTH headers n)
message(STATUS "core_headers_name_no_arch: ${n} core headers, none names a model")
