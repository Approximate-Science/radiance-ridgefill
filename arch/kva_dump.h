/* kva_dump.h -- RADIANCE_KVA_DUMP=<dir>: debug copies of what a filled chunk computed, to the host.
 *
 * DEBUG ONLY, AND IT COSTS: every record SYNCHRONISES the stream mid-step and copies device memory to
 * pageable host memory. The directory is read at declare; with it unset none of this runs, and no
 * measured run sets it. Written by rank 0 only (the boundary stream and the selected rows are the
 * same on every rank). A recorded pass replayed later does not call step(), so a dump run should
 * use prompts whose chunks do not repeat a pass key (every prefill chunk of a fresh prompt is issued
 * live: radiance core/runtime/ctx.cpp:1305-1316).
 *
 *   <dir>/boundary.p<P>.npy  f32 [n_tok, hc*n_embd]: the residual stream entering layer S of the
 *                            approximate chunk whose first row is at absolute position P (R17)
 *   <dir>/boundary.jsonl     one line per such file: {"chunk_start", "n_tok", "file", "token_ids"}
 *   <dir>/rows.jsonl         quality mode, one line per approximate chunk: {"chunk_start", "n_tok",
 *                            "rows_idx", "token_ids"} -- the format tools/rows_compare.py reads (R39)
 */
#ifndef QWEN4EXP_KVA_DUMP_H
#define QWEN4EXP_KVA_DUMP_H

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace qwen4exp_kva {

using namespace rad::arch;

/* Device to host, after everything issued so far has run. */
inline bool dump_read(RadCtx* c, void* host, const void* dev, int64_t bytes) {
    RadStream s = rad_stream(c);
    return dev && rad_stream_sync(s) >= 0 && rad_memcpy_async(host, dev, bytes, s) >= 0 &&
           rad_stream_sync(s) >= 0;
}

inline std::string dump_ids_json(const std::vector<int32_t>& v) {
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + std::to_string(v[i]);
    return out + "]";
}

/* The chunk's absolute first position and its token ids, read off the batch's device arrays. */
inline bool dump_chunk(RadCtx* c, const RadBatch* b, int32_t* start, std::vector<int32_t>* ids) {
    ids->assign((size_t)b->n_tok, 0);
    return dump_read(c, start, b->positions, 4) &&
           dump_read(c, ids->data(), b->token_ids, b->n_tok * 4);
}

inline void dump_line(const std::string& path, const std::string& line) {
    if (FILE* f = std::fopen(path.c_str(), "a")) {
        std::fprintf(f, "%s\n", line.c_str());
        std::fclose(f);
    } else {
        std::fprintf(stderr, "radiance: qwen4exp_kva: RADIANCE_KVA_DUMP: cannot append to %s\n",
                     path.c_str());
    }
}

/* A bf16 [rows, cols] plane as a little-endian f32 .npy, which numpy loads with no help. */
inline bool dump_npy_f32(const std::string& path, const std::vector<uint16_t>& bf16, int64_t rows,
                         int64_t cols) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(rows) +
                      ", " + std::to_string(cols) + "), }";
    while ((10 + hdr.size() + 1) % 64) hdr += ' ';
    hdr += '\n';
    const uint16_t hl = (uint16_t)hdr.size();
    std::fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    std::fwrite(&hl, 2, 1, f);
    std::fwrite(hdr.data(), 1, hdr.size(), f);
    std::vector<float> row((size_t)cols);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < cols; ++j) {
            const uint32_t bits = (uint32_t)bf16[(size_t)(r * cols + j)] << 16;
            std::memcpy(&row[(size_t)j], &bits, 4);
        }
        std::fwrite(row.data(), 4, (size_t)cols, f);
    }
    return std::fclose(f) == 0;
}

/* R17: the stream entering layer S (b_h after layer S-1's ffn write), every approximate chunk. */
inline void dump_boundary(RadCtx* c, const std::string& dir, rad_buf b_h, int64_t wide,
                          const RadBatch* b) {
    int32_t start = -1;
    std::vector<int32_t> ids;
    std::vector<uint16_t> h((size_t)(b->n_tok * wide));
    if (!dump_chunk(c, b, &start, &ids) ||
        !dump_read(c, h.data(), rad_buf_ptr(c, b_h), (int64_t)h.size() * 2)) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: RADIANCE_KVA_DUMP: device read failed\n");
        return;
    }
    const std::string file = "boundary.p" + std::to_string(start) + ".npy";
    if (!dump_npy_f32(dir + "/" + file, h, b->n_tok, wide))
        std::fprintf(stderr, "radiance: qwen4exp_kva: RADIANCE_KVA_DUMP: cannot write %s/%s\n",
                     dir.c_str(), file.c_str());
    dump_line(dir + "/boundary.jsonl",
              "{\"chunk_start\": " + std::to_string(start) + ", \"n_tok\": " +
              std::to_string(b->n_tok) + ", \"file\": \"" + file + "\", \"token_ids\": " +
              dump_ids_json(ids) + "}");
}

/* R39: the rows kva_rowsel kept in this chunk (chunk-relative, ascending, -1 padded). */
inline void dump_rows(RadCtx* c, const std::string& dir, rad_buf rows_idx, int64_t cap,
                      const RadBatch* b) {
    int32_t start = -1;
    std::vector<int32_t> ids, rows((size_t)cap);
    if (!dump_chunk(c, b, &start, &ids) ||
        !dump_read(c, rows.data(), rad_buf_ptr(c, rows_idx), cap * 4)) {
        std::fprintf(stderr, "radiance: qwen4exp_kva: RADIANCE_KVA_DUMP: device read failed\n");
        return;
    }
    dump_line(dir + "/rows.jsonl",
              "{\"chunk_start\": " + std::to_string(start) + ", \"n_tok\": " +
              std::to_string(b->n_tok) + ", \"rows_idx\": " + dump_ids_json(rows) +
              ", \"token_ids\": " + dump_ids_json(ids) + "}");
}

}  /* namespace qwen4exp_kva */

#endif /* QWEN4EXP_KVA_DUMP_H */
