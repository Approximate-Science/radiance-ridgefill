# Copyright 2026 Dylan Johnston and tcclaviger
# SPDX-License-Identifier: Apache-2.0
# THE TARGET IS NAMED qwen4exp_fp8 ON PURPOSE (PLAN D1). The engine refuses two plugins claiming
# (qwen4exp, ""), and rad-convert --reuse needs that same arch id, so this plugin claims it too and
# replaces the in-tree one by FILE STEM: architectures/qwen4exp_fp8.so in a $RADIANCE_HOME entry
# ahead of the installation's shadows the installed file of the same name (radiance
# core/plugin/loader.cpp:529-564). Its plugin NAME is qwen4exp_ridgefill, which is what rad-info prints.
#
# ${RADIANCE_SRC}/arch is on the include path for exactly one file, qwen4exp_fp8/qwen4exp_fp8.cpp;
# the headers it includes (<arch/rad_arch.h> ...) resolve to the installed ones, which the top-level
# CMakeLists.txt has checked are RADIANCE_SRC's byte for byte.
rad_add_plugin(qwen4exp_fp8 architectures
               SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/qwen4exp_ridgefill.cpp
               INCLUDES ${RADIANCE_SRC}/arch)

# The release the plugin is built against, for the open-time engine guard (ridgefill_guard.h, R83): the
# top-level CMakeLists.txt has already required RADIANCE_SRC and the installed engine to agree on it.
target_compile_definitions(qwen4exp_fp8 PRIVATE RIDGEFILL_RADIANCE_VERSION="${RADIANCE_SRC_VERSION}")
target_link_libraries(qwen4exp_fp8 PRIVATE ${CMAKE_DL_LIBS})
