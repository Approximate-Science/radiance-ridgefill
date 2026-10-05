# core_purity.cmake -- the build-time gate core_headers_name_no_arch (notes/adapter-split-spec.md §4.5).
#
# Every arch/kva_*.h is CORE: it must not name the in-tree architecture or its block types, so a second
# model's adapter can compile the core unchanged. Run as `cmake -DSRC=<repo> -P core_purity.cmake`.
#
# While the split is in progress, KVA_PENDING lists the core headers not yet converted. The gate fails
# both ways: a header outside the list that names a model (a regression), and a header in the list
# that no longer does (it is done -- take it off, so the list only shrinks). At the split's end the list
# is empty.
set(KVA_PENDING
  kva_config.h kva_declare.h kva_declare_masked.h kva_dump.h kva_final.h kva_guard.h kva_hazard.h
  kva_projector.h)
set(FORBIDDEN "qwen4exp|GdnFP8|MoeFP8|AttnGatedFP8|QsaIndexer|HyperConn|gcfg|hccfg|moecfg|ple_layer")

file(GLOB headers RELATIVE "${SRC}/arch" "${SRC}/arch/kva_*.h")
set(bad "")
foreach(h IN LISTS headers)
  file(STRINGS "${SRC}/arch/${h}" hits REGEX "${FORBIDDEN}")
  list(FIND KVA_PENDING "${h}" pending)
  if(hits AND pending EQUAL -1)
    list(GET hits 0 first)
    string(APPEND bad "  ${h} names a model: ${first}\n")
  elseif(NOT hits AND NOT pending EQUAL -1)
    string(APPEND bad "  ${h} is pure now: take it off KVA_PENDING in tests/core_purity.cmake\n")
  endif()
endforeach()
if(bad)
  message(FATAL_ERROR "core_headers_name_no_arch:\n${bad}")
endif()
list(LENGTH headers n)
list(LENGTH KVA_PENDING p)
message(STATUS "core_headers_name_no_arch: ${n} core headers, ${p} still pending")
