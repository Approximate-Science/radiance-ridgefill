/* Copyright 2026 Dylan Johnston and tcclaviger
 * SPDX-License-Identifier: Apache-2.0 */
/* folder_fixture.h -- a projector folder the host tests build in memory (ridgefill_folder.h's Folder, handed to
 * load_folder through g_folder_for_test), and the tiny container its manifest's tokenizer hash points at.
 * Include after the core (ridgefill_folder.h) and rad_fake.h.
 */
#ifndef RIDGEFILL_TEST_FOLDER_FIXTURE_H
#define RIDGEFILL_TEST_FOLDER_FIXTURE_H

#include <cstring>
#include <string>
#include <vector>

namespace {

/* ==================================================================== the projector folder */

/* A container of a few KiB (abi/rad_format.h): three tokens (the third with empty text), one merge
 * and one 16-byte entry. tools/ridgefill_projector.py's test builds the same bytes and expects the same
 * two hashes (kTinyVocab, kTinyAnchor), which is what ties the two canonical forms together. */
const char* kTinyVocab  = "3988fb447f719ad3fc2c75e5a0fa3daeb2b6a5e10e964744619d1dfbc6e95ff6";
const char* kTinyAnchor = "be45cb2605bf36bebde684841a28f0fd43c69850a3dce5fedba69928ee3a8991";

std::vector<unsigned char> tiny_container_bytes() {
    std::vector<unsigned char> f(4096, 0);
    auto put = [&](size_t at, const void* p, size_t n) { std::memcpy(f.data() + at, p, n); };
    RadFileHeader h{};
    h.magic = RAD_MAGIC; h.version = RAD_FORMAT_VER; h.file_bytes = 4096;
    const char blob[] = "\0tok_a\0tok_b\0anchor.weight";   /* offsets 0, 1, 7, 13 */
    h.str_off = 256; h.str_bytes = sizeof blob;
    h.vocab_off = 512; h.vocab_bytes = 168;
    h.dir_off = 1024; h.dir_count = 1;
    h.plane_off = 1280; h.plane_count = 1;
    h.data_off = 2048; h.data_bytes = 2048;
    put(0, &h, sizeof h);
    put(256, blob, sizeof blob);
    RadVocabHeader v{};
    v.kind = RAD_TOK_BPE; v.n_tokens = 3; v.n_merges = 1;
    v.tok_text_off = 640; v.tok_type_off = 664; v.merge_off = 672;
    put(512, &v, sizeof v);
    const uint64_t text[3] = { 1, 7, 0 };
    const uint8_t type[3] = { 1, 1, 3 };
    const uint32_t merge[2] = { 0, 1 };
    put(640, text, sizeof text); put(664, type, sizeof type); put(672, merge, sizeof merge);
    RadFileEntry e{};
    e.name = 13; e.rank = 1; e.shape[0] = 16; e.layer = -1; e.expert = -1; e.n_planes = 1;
    e.offset = 2048; e.bytes = 16;
    put(1024, &e, sizeof e);
    const RadFilePlane pl{ 2048, 16 };
    put(1280, &pl, sizeof pl);
    for (int i = 0; i < 16; ++i) f[2048 + (size_t)i] = (unsigned char)i;
    return f;
}

/* The tiny container on disk, once per process; its path. */
const std::string& tiny_container() {
    static std::string path;
    if (path.empty()) {
        char t[] = "/tmp/ridgefill_tiny_XXXXXX";
        const int fd = mkstemp(t);
        const std::vector<unsigned char> f = tiny_container_bytes();
        if (fd >= 0 && write(fd, f.data(), f.size()) == (ssize_t)f.size()) path = t;
        if (fd >= 0) close(fd);
    }
    return path;
}

/* Bytes every test tensor points at (zeros, one projector map's worth): the uploads copy from them. */
const unsigned char* tensor_bytes() {
    static std::vector<unsigned char> z((size_t)2560 * 10240 * 2, 0);
    return z.data();
}

ridgefill::Folder g_test_folder;

/* The plugin forgets every folder and copy; the next declare looks again (every case starts so). */
void reset_projector() {
    ridgefill::g_folder_for_test = nullptr;
    ridgefill::g_i8_rows_for_test = nullptr;
    ridgefill::g_loaded = ridgefill::Loaded{};
    ridgefill::free_uploads();
    g_mem.copies.clear();
}

void add_tensor(const std::string& name, uint32_t dtype, std::vector<int64_t> shape) {
    ridgefill::FolderTensor t;
    t.data = tensor_bytes();
    t.dtype = dtype;
    t.shape = shape;
    int64_t n = 1;
    for (int64_t e : shape) n *= e;
    t.bytes = rad_dtype_bytes(dtype, n);
    g_test_folder.tensors[name] = t;
}

}  /* namespace */

#endif /* RIDGEFILL_TEST_FOLDER_FIXTURE_H */
