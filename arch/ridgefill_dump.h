/* ridgefill_dump.h -- the debug copies to the host: RADIANCE_RIDGEFILL_DUMP (what a filled chunk computed) and
 * the Stage 6 captures RADIANCE_RIDGEFILL_CAPTURE / RADIANCE_RIDGEFILL_CAPTURE_STATE (the refit's data).
 * notes/arch.md "Capture" is the file layout the fitting side reads; this file writes exactly it.
 *
 * DEBUG ONLY, AND IT COSTS: every record SYNCHRONISES the stream mid-step and copies device memory to
 * pageable host memory. The directories are read at declare; with them unset none of this runs, and
 * no measured run sets them. The issue sequence is never changed except by the state capture's own
 * ridgefill_state_read copies (no ABI call returns a KV-pool pointer).
 *
 * ONLY A PASS ISSUED LIVE IS RECORDED. A pass the engine replays from its recording does not call
 * step() (radiance core/runtime/ctx.cpp:1258-1278): a prefill pass is recorded the second time its
 * key is seen and played after that, unless the expert stager arms it live. Every record therefore
 * carries the chunk's identity, and a reader checks the count; a run that comes up short is rerun
 * with --profile-ops, which turns recording off (ctx.cpp:120).
 *
 *   <dir>/boundary.p<P>.npy  f32 [n_tok, hc*n_embd]: the residual stream entering layer S of the
 *                            approximate chunk whose first row is at absolute position P (R17)
 *   <dir>/boundary.jsonl     one line per such file: {"chunk_start", "n_tok", "file", "token_ids"}
 *   <dir>/mask.jsonl         one line per masked approximate chunk: {"chunk_start", "n_tok",
 *                            "n_ahead", "b", "s_lb", "n_seq", "n_seq_decode", "bounds", "mask"}
 *                            (R45; R58' reads the steps with two prefills)
 *   <dir>/rows.jsonl         the same chunks: {"chunk_start", "n_tok", "rows_idx", "token_ids"},
 *                            rows_idx = the window's exact rows -- what tools/rows_compare.py reads (R39)
 *
 * RADIANCE_RIDGEFILL_DUMP_LOGITS=<dir> (any mode, off included; tools/logit_compare.py reads it): every
 * live trunk pass with output rows, after the step, each rank's logits rows --
 *   <dir>/logits.<key>.r<R>.npy  f32 [n_out, width]: rank R's `logits` buffer rows
 *   <dir>/logits.jsonl           one line per file: {"file", "rank", "width", "rows": [[out row index,
 *                                step row, sequence, position, token], ...]}
 * Replayed passes call no step(): run the server with --profile-ops (it turns recording off).
 *
 * RADIANCE_RIDGEFILL_CAPTURE_STATE on a MIXED step (R61; scripts/state_compare.py reads it):
 *   <dir>/mixed.<key>.r<R>.npy  f32 [n_seq, layers, heads, V, K]: every sequence's late delta-net
 *                               states after the step, rank R's heads; <key> = the step's first
 *                               position and the FNV-1a of all its token ids
 *   <dir>/mixed.jsonl           one line per file: {"file", "key", "n_seq", "n_seq_decode", "cu",
 *                               "starts" (each sequence's first position), "layers", "approximate"}
 */
#ifndef RIDGEFILL_DUMP_H
#define RIDGEFILL_DUMP_H

#include "ridgefill_log.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <initializer_list>
#include <vector>

namespace ridgefill {

using namespace rad::arch;

/* Device to host, after everything issued so far has run. */
/* A device read on the step's stream: sync, copy, sync. It stalls the step twice, which is why every dump
 * and capture sits behind a switch no measured run sets. */
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

/* One JSON line appended to `path` (the dumps' index files), or the reason on stderr. */
inline void dump_line(const std::string& path, const std::string& line) {
    if (FILE* f = std::fopen(path.c_str(), "a")) {
        std::fprintf(f, "%s\n", line.c_str());
        std::fclose(f);
    } else {
        std::fprintf(stderr, "radiance: %s: RADIANCE_RIDGEFILL_DUMP: cannot append to %s\n", g_log_name,
                     path.c_str());
    }
}

/* One .npy: `descr` is numpy's ("<u2" for bf16 bits, "<i4", "<f4"), the data row-major. */
inline bool dump_npy(const std::string& path, const char* descr, std::initializer_list<int64_t> shape,
                     const void* data, int64_t bytes) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::string dims;
    for (int64_t d : shape) dims += std::to_string(d) + ", ";
    std::string hdr = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': (" +
                      dims + "), }";
    while ((10 + hdr.size() + 1) % 64) hdr += ' ';
    hdr += '\n';
    const uint16_t hl = (uint16_t)hdr.size();
    std::fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    std::fwrite(&hl, 2, 1, f);
    std::fwrite(hdr.data(), 1, hdr.size(), f);
    std::fwrite(data, 1, (size_t)bytes, f);
    return std::fclose(f) == 0;
}

/* A bf16 [rows, cols] plane as an f32 .npy (the R17 boundary, read with no help). */
inline bool dump_npy_f32(const std::string& path, const std::vector<uint16_t>& bf16, int64_t rows,
                         int64_t cols) {
    std::vector<float> f((size_t)(rows * cols));
    for (size_t i = 0; i < f.size(); ++i) {
        const uint32_t bits = (uint32_t)bf16[i] << 16;
        std::memcpy(&f[i], &bits, 4);
    }
    return dump_npy(path, "<f4", {rows, cols}, f.data(), (int64_t)f.size() * 4);
}

/* R17: the stream entering layer S (b_h after layer S-1's ffn write), every approximate chunk. */
inline void dump_boundary(RadCtx* c, const std::string& dir, rad_buf b_h, int64_t wide,
                          const RadBatch* b) {
    int32_t start = -1;
    std::vector<int32_t> ids;
    std::vector<uint16_t> h((size_t)(b->n_tok * wide));
    if (!dump_chunk(c, b, &start, &ids) ||
        !dump_read(c, h.data(), rad_buf_ptr(c, b_h), (int64_t)h.size() * 2)) {
        std::fprintf(stderr, "radiance: %s: RADIANCE_RIDGEFILL_DUMP: device read failed\n", g_log_name);
        return;
    }
    const std::string file = "boundary.p" + std::to_string(start) + ".npy";
    if (!dump_npy_f32(dir + "/" + file, h, b->n_tok, wide))
        std::fprintf(stderr, "radiance: %s: RADIANCE_RIDGEFILL_DUMP: cannot write %s/%s\n", g_log_name,
                     dir.c_str(), file.c_str());
    dump_line(dir + "/boundary.jsonl",
              "{\"chunk_start\": " + std::to_string(start) + ", \"n_tok\": " +
              std::to_string(b->n_tok) + ", \"file\": \"" + file + "\", \"token_ids\": " +
              dump_ids_json(ids) + "}");
}

/* R45 / R39: the device mask of a masked approximate chunk. mask.jsonl carries the whole mask (one
 * character a row, '1' = approximated) with the host's b and s_lb and the device's bounds, which is
 * what R45 compares with the rule's transcription; rows.jsonl carries the window's exact rows in the
 * format tools/rows_compare.py reads (chunk-relative, ascending). */
inline void dump_mask(RadCtx* c, const std::string& dir, rad_buf mask, rad_buf bounds,
                      const RadBatch* b, int64_t bulk_end, int64_t s_lb) {
    int32_t start = -1, bnd[4] = {};
    std::vector<int32_t> ids, m((size_t)b->n_tok), exact;
    if (!dump_chunk(c, b, &start, &ids) || !dump_read(c, m.data(), rad_buf_ptr(c, mask), b->n_tok * 4) ||
        !dump_read(c, bnd, rad_buf_ptr(c, bounds), 16)) {
        std::fprintf(stderr, "radiance: %s: RADIANCE_RIDGEFILL_DUMP: device read failed\n", g_log_name);
        return;
    }
    std::string bits((size_t)b->n_tok, '0');
    for (int64_t i = 0; i < b->n_tok; ++i) {
        bits[(size_t)i] = m[(size_t)i] ? '1' : '0';
        if (i >= bnd[0] && i < bnd[1] && !m[(size_t)i]) exact.push_back((int32_t)i);
    }
    const std::string head = "{\"chunk_start\": " + std::to_string(start) + ", \"n_tok\": " +
                             std::to_string(b->n_tok);
    dump_line(dir + "/mask.jsonl",
              head + ", \"n_ahead\": " + std::to_string(b->n_ahead) + ", \"b\": " +
              std::to_string(bulk_end) + ", \"s_lb\": " + std::to_string(s_lb) + ", \"n_seq\": " +
              std::to_string(b->n_seq) + ", \"n_seq_decode\": " + std::to_string(b->n_seq_decode) + ", \"bounds\": [" +
              std::to_string(bnd[0]) + ", " + std::to_string(bnd[1]) + ", " + std::to_string(bnd[2]) +
              ", " + std::to_string(bnd[3]) + "], \"mask\": \"" + bits + "\"}");
    dump_line(dir + "/rows.jsonl", head + ", \"rows_idx\": " + dump_ids_json(exact) +
                                       ", \"token_ids\": " + dump_ids_json(ids) + "}");
}

/* ================================================================== Stage 6 captures
 * (notes/arch.md "Capture"). Keyed by the chunk's first position and FNV-1a 64 of its token ids,
 * so two prompts' chunks at one position never collide and no host counter is kept. */
constexpr int64_t kCaptureStride = 8;   /* tcc's capture stride (KVA-FACTS §8 step 1) */

inline std::string chunk_key(int32_t start, const std::vector<int32_t>& ids) {
    uint64_t h = 1469598103934665603ull;
    for (int32_t v : ids)
        for (int b = 0; b < 4; ++b) { h ^= (uint8_t)((uint32_t)v >> (8 * b)); h *= 1099511628211ull; }
    char buf[64];
    std::snprintf(buf, sizeof buf, "p%d.h%016llx", start, (unsigned long long)h);
    return buf;
}

/* One exact chunk's capture (rank 0): its ids and positions, the rows a stride-8 capture keeps, and the
 * late layers whose block input it read -- what tools/refit's fit consumes. */
struct Capture {
    std::string            dir, prefix;
    int32_t                start = -1;
    std::vector<int32_t>   ids, pos, rows;
    std::vector<int>       layers;
};

/* The chunk's ids and positions, and which rows a stride-8 capture keeps (position % 8 == 0). */
inline bool capture_begin(RadCtx* c, const RadBatch* b, const std::string& dir, Capture* cap) {
    cap->dir = dir;
    cap->ids.assign((size_t)b->n_tok, 0);
    cap->pos.assign((size_t)b->n_tok, 0);
    if (!dump_read(c, cap->ids.data(), b->token_ids, b->n_tok * 4) ||
        !dump_read(c, cap->pos.data(), b->positions, b->n_tok * 4))
        return false;
    cap->start = cap->pos.empty() ? -1 : cap->pos[0];
    cap->prefix = "chunk." + chunk_key(cap->start, cap->ids);
    for (int64_t i = 0; i < b->n_tok; ++i)
        if (cap->pos[(size_t)i] % kCaptureStride == 0) cap->rows.push_back((int32_t)i);
    return true;
}

/* The captured rows of a bf16 [n_tok, width] buffer, as `<prefix>.<name>.npy` ('<u2'). */
inline bool capture_rows(RadCtx* c, const Capture& cap, const std::string& name, rad_buf buf,
                         int64_t n_tok, int64_t width) {
    std::vector<uint16_t> all((size_t)(n_tok * width)), kept;
    if (!dump_read(c, all.data(), rad_buf_ptr(c, buf), (int64_t)all.size() * 2)) return false;
    kept.reserve(cap.rows.size() * (size_t)width);
    for (int32_t r : cap.rows)
        kept.insert(kept.end(), all.begin() + (long)r * width, all.begin() + (long)(r + 1) * width);
    return dump_npy(cap.dir + "/" + cap.prefix + "." + name + ".npy", "<u2",
                    {(int64_t)cap.rows.size(), width}, kept.data(), (int64_t)kept.size() * 2);
}

inline std::string ints_json(const std::vector<int>& v) {
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + std::to_string(v[i]);
    return out + "]";
}

/* The chunk's index arrays and its capture.jsonl line: the geometry the fit checks the files against. */
inline void capture_end(const Capture& cap, int64_t split, int64_t hidden, int64_t hc) {
    const std::string base = cap.dir + "/" + cap.prefix;
    const int64_t n = (int64_t)cap.ids.size(), r = (int64_t)cap.rows.size();
    const bool ok = dump_npy(base + ".rows.npy", "<i4", {r}, cap.rows.data(), r * 4) &&
                    dump_npy(base + ".ids.npy", "<i4", {n}, cap.ids.data(), n * 4) &&
                    dump_npy(base + ".pos.npy", "<i4", {n}, cap.pos.data(), n * 4);
    if (!ok) std::fprintf(stderr, "radiance: %s: RADIANCE_RIDGEFILL_CAPTURE: cannot write %s.*\n", g_log_name,
                          base.c_str());
    dump_line(cap.dir + "/capture.jsonl",
              "{\"prefix\": \"" + cap.prefix + "\", \"chunk_start\": " + std::to_string(cap.start) +
              ", \"n_tok\": " + std::to_string(n) + ", \"stride\": " + std::to_string(kCaptureStride) +
              ", \"rows\": " + std::to_string(r) + ", \"split\": " + std::to_string(split) +
              ", \"layers\": " + ints_json(cap.layers) + ", \"hidden\": " + std::to_string(hidden) +
              ", \"hc\": " + std::to_string(hc) + ", \"dtype\": \"bf16\"}");
}

/* One rank's late delta-net states for one chunk, gathered layer by layer. */
struct StateDump {
    std::vector<int>   layers;
    std::vector<float> data;    /* [layers.size(), heads, V, K] in `layers` order */
};

/* One (chunk, rank) state file plus its index line; `approximate` says whether the chunk was RidgeFill's. */
inline void state_end(RadCtx* c, const std::string& dir, const RadBatch* b, const StateDump& sd,
                      int64_t heads, int64_t v, int64_t k, int rank, int world, bool approximate,
                      const char* mode) {
    int32_t start = -1;
    std::vector<int32_t> ids;
    if (!dump_chunk(c, b, &start, &ids)) {
        std::fprintf(stderr, "radiance: %s: RADIANCE_RIDGEFILL_CAPTURE_STATE: device read failed\n", g_log_name);
        return;
    }
    /* Ascending layer order whatever order the reads came in. */
    std::vector<size_t> order(sd.layers.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t z) { return sd.layers[a] < sd.layers[z]; });
    const size_t per = (size_t)(heads * v * k);
    std::vector<float> out;
    std::vector<int> layers;
    for (size_t i : order) {
        layers.push_back(sd.layers[i]);
        out.insert(out.end(), sd.data.begin() + (long)(i * per), sd.data.begin() + (long)((i + 1) * per));
    }
    const std::string file = "state." + chunk_key(start, ids) + ".r" + std::to_string(rank) + ".npy";
    if (!dump_npy(dir + "/" + file, "<f4", {(int64_t)layers.size(), heads, v, k}, out.data(),
                  (int64_t)out.size() * 4))
        std::fprintf(stderr, "radiance: %s: RADIANCE_RIDGEFILL_CAPTURE_STATE: cannot write %s/%s\n", g_log_name,
                     dir.c_str(), file.c_str());
    dump_line(dir + "/state.jsonl",
              "{\"file\": \"" + file + "\", \"chunk_start\": " + std::to_string(start) +
              ", \"last_position\": " + std::to_string(start + b->n_tok - 1) + ", \"n_tok\": " +
              std::to_string(b->n_tok) + ", \"rank\": " + std::to_string(rank) + ", \"world\": " +
              std::to_string(world) + ", \"heads\": [" + std::to_string(rank * heads) + ", " +
              std::to_string((rank + 1) * heads) + "], \"layers\": " + ints_json(layers) +
              ", \"approximate\": " + (approximate ? "true" : "false") + ", \"mode\": \"" + mode +
              "\", \"applied_before_copy\": false}");
}

/* One trunk pass's logits rows, with what each row is: its sequence (from cu_seqlens), its
 * position and its token. Debug only (synchronises). */
inline void dump_logits(RadCtx* c, const std::string& dir, const RadBatch* b, rad_buf logits, int64_t width,
                        int rank) {
    int32_t start = -1;
    std::vector<int32_t> ids, cu((size_t)b->n_seq + 1), pos((size_t)b->n_tok), out((size_t)b->n_out);
    std::vector<float> rows((size_t)(b->n_out * width));
    if (!dump_chunk(c, b, &start, &ids) || !dump_read(c, cu.data(), b->cu_seqlens, (int64_t)cu.size() * 4) ||
        !dump_read(c, pos.data(), b->positions, b->n_tok * 4) ||
        !dump_read(c, out.data(), b->out_ids, b->n_out * 4) ||
        !dump_read(c, rows.data(), rad_buf_ptr(c, logits), (int64_t)rows.size() * 4)) {
        std::fprintf(stderr, "radiance: %s: RADIANCE_RIDGEFILL_DUMP_LOGITS: device read failed\n", g_log_name);
        return;
    }
    const std::string file = "logits." + chunk_key(start, ids) + ".r" + std::to_string(rank) + ".npy";
    if (!dump_npy(dir + "/" + file, "<f4", {b->n_out, width}, rows.data(), (int64_t)rows.size() * 4))
        std::fprintf(stderr, "radiance: %s: RADIANCE_RIDGEFILL_DUMP_LOGITS: cannot write %s\n", g_log_name, file.c_str());
    std::string list = "[";
    for (int64_t j = 0; j < b->n_out; ++j) {
        const int32_t row = out[(size_t)j];
        int64_t seq = 0;
        while (seq + 1 < b->n_seq && cu[(size_t)seq + 1] <= row) ++seq;
        list += (j ? ", [" : "[") + std::to_string(j) + ", " + std::to_string(row) + ", " + std::to_string(seq) +
                ", " + std::to_string(pos[(size_t)row]) + ", " + std::to_string(ids[(size_t)row]) + "]";
    }
    dump_line(dir + "/logits.jsonl", "{\"file\": \"" + file + "\", \"rank\": " + std::to_string(rank) +
                                         ", \"width\": " + std::to_string(width) + ", \"rows\": " + list + "]}");
}

/* One mixed step's states (R61): the batch's sequence bounds and first positions read back, the
 * states written as one .npy, one line in mixed.jsonl. `dims` = {heads, V, K} of this rank. */
inline void mixed_state_end(RadCtx* c, const std::string& dir, const RadBatch* b, const std::vector<float>& data,
                            const std::vector<int>& layers, std::initializer_list<int64_t> dims, int rank,
                            bool approximate, const char* mode) {
    int32_t start = -1;
    std::vector<int32_t> ids, cu((size_t)b->n_seq + 1), pos((size_t)b->n_tok);
    if (!dump_chunk(c, b, &start, &ids) || !dump_read(c, cu.data(), b->cu_seqlens, (int64_t)cu.size() * 4) ||
        !dump_read(c, pos.data(), b->positions, b->n_tok * 4)) {
        std::fprintf(stderr, "radiance: %s: RADIANCE_RIDGEFILL_CAPTURE_STATE: device read failed\n", g_log_name);
        return;
    }
    std::vector<int> starts, cus(cu.begin(), cu.end());
    for (int64_t i = 0; i < b->n_seq; ++i) starts.push_back(pos[(size_t)cu[(size_t)i]]);
    const auto d = dims.begin();
    const std::string key = chunk_key(start, ids), file = "mixed." + key + ".r" + std::to_string(rank) + ".npy";
    if (!dump_npy(dir + "/" + file, "<f4", {b->n_seq, (int64_t)layers.size(), d[0], d[1], d[2]}, data.data(),
                  (int64_t)data.size() * 4))
        std::fprintf(stderr, "radiance: %s: RADIANCE_RIDGEFILL_CAPTURE_STATE: cannot write %s/%s\n", g_log_name,
                     dir.c_str(), file.c_str());
    dump_line(dir + "/mixed.jsonl",
              "{\"file\": \"" + file + "\", \"key\": \"" + key + "\", \"rank\": " + std::to_string(rank) +
              ", \"n_seq\": " + std::to_string(b->n_seq) + ", \"n_seq_decode\": " + std::to_string(b->n_seq_decode) +
              ", \"cu\": " + ints_json(cus) + ", \"starts\": " + ints_json(starts) + ", \"layers\": " +
              ints_json(layers) + ", \"approximate\": " + (approximate ? "true" : "false") + ", \"mode\": \"" +
              mode + "\"}");
}

}  /* namespace ridgefill */

#endif /* RIDGEFILL_DUMP_H */
